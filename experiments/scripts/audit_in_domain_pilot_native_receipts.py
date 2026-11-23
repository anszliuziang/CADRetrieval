"""Independent read-only audit of pilot score, timing and optimizer receipts."""

from hashlib import file_digest
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def native_audit(folder, levels):
    background = read(folder / "background.json")
    check(background["planned_pairs"] == 192 and len(background["pairs"]) == 192,
          f"{folder}: wrong background denominator")
    priors = {}
    for j in range(4):
        entries = [row for row in background["pairs"] if row["candidate_i"] == j]
        check(len(entries) == 48 and len({row["query_text_sha256"] for row in entries}) == 48,
              f"{folder}: bad background axis for {j}")
        if all(row["status"] == "scored" for row in entries):
            s0 = np.asarray([row["s0"] for row in entries], dtype=np.float64)
            prior = float(np.logaddexp.reduce(s0) - math.log(48))
            check(math.isclose(prior, background["candidate_logmeanexp"][j], abs_tol=2e-9, rel_tol=0),
                  f"{folder}: independent S48 prior differs at {j}")
            priors[j] = prior
    methods = {}
    for level in levels:
        result = read(folder / f"{level}_readout.json")
        axis = result["uid_axis"]
        positions = {(i, j) for i in range(4) for j in range(4)}
        valid = {}
        for method in ("single", "batched"):
            rows = result[method]["pairs"]
            check(len(rows) == 16 and {(r["query_i"], r["candidate_i"]) for r in rows} == positions,
                  f"{folder}/{level}/{method}: score axis incomplete")
            losses = []
            for row in rows:
                check(row["candidate_uid"] == axis[row["candidate_i"]]
                      and row["query_uid"] == result["query_axis"][row["query_i"]]["uid"],
                      f"{folder}/{level}/{method}: q/CAD identity differs")
                if row["status"] == "scored":
                    delta = abs(row["raw_nll"] + row["s0"] / row["effective_units"])
                    check(math.isclose(delta, row["nll_abs_delta"], abs_tol=1e-12, rel_tol=0),
                          f"{folder}/{level}: NLL arithmetic differs")
                    losses.append(delta)
            valid[method] = {"scored": len(losses), "max_nll_delta": max(losses, default=None)}
        calibrated = 0
        for row in result["pairs"]:
            if row["status"] == "scored" and row["candidate_i"] in priors:
                check(math.isclose(row["s0"] - priors[row["candidate_i"]], row["s48"], abs_tol=2e-9, rel_tol=0),
                      f"{folder}/{level}: independent S48 differs")
                calibrated += 1
        proof = result["causal_prefix_probe"]
        distribution = None
        if "raw_tensors" in proof:
            tensors = torch.load(folder / proof["raw_tensors"], map_location="cpu", weights_only=True)
            shift = (tensors["log_probabilities_fp32"][0] - tensors["log_probabilities_fp32"][1]).abs()
            delta = float(shift.max())
            check(delta == proof["max_log_probability_abs_delta"]
                  and int((shift > proof["whole_log_distribution_atol"]).sum()) == proof["n_over_tolerance"],
                  f"{folder}/{level}: distribution proof differs")
            distribution = {"max_logprob_delta": delta, "shape": list(shift.shape), "passed": proof["passed"]}
        methods[level] = {
            "coverage": valid, "s48_verified": calibrated, "whole_next_step": distribution,
            "pair_checks_passed": result["calibration_audit"]["pairs_passed"],
            "candidate_target_invariance": all(p["passed"] for p in result["candidate_target_invariance"]),
            "passed": result["passed"],
        }
    return {"background_scored": background["scored_pairs"], "verified_prior_columns": len(priors), "levels": methods}


def timing_audit(folder):
    result = read(folder / "hiercad_timing.json")
    snapshots = {}
    for cell in result["cells"]:
        condition, band = cell["condition"], cell["band"]
        raw = read(folder / f"timing_{condition}_{band}.json")
        windows = raw["windows"]
        expected = raw["steady_pairs_per_window"]
        valid = [w for w in windows if w["phase"] == "steady"
                 and w["measurement"]["scored_pairs"] == expected
                 and w["measurement"]["planned_pairs"] == expected
                 and w["measurement"]["wall_seconds"] > 0]
        rates = [expected / w["measurement"]["wall_seconds"] for w in valid]
        if rates:
            check(cell["throughput"]["pairs_per_second_low"] == min(rates)
                  and cell["throughput"]["pairs_per_second_high"] == max(rates),
                  f"{folder}: timing aggregate differs for {condition}/{band}")
        else:
            check(cell["throughput"] is None, f"{folder}: unsupported throughput")
        coverage = raw["target_length_coverage"]
        snapshots.setdefault(condition, {})[band] = {
            "rates": [min(rates), max(rates)] if rates else None,
            "complete_steady_windows": len(valid), "attempted_steady_windows": sum(w["phase"] == "steady" for w in windows),
            "pair_count_per_steady_window": expected,
            "max_peak_allocated_bytes": max((w["peak_allocated_bytes"] or 0 for w in windows), default=0),
            "context": [cell["gallery_pairs_context_fits"], cell["gallery_pairs_expected"]],
            "query_preparation_failures": len(cell["query_preparation_failures"]),
            "max_measured_input_tokens": max((x["max_input_tokens"] for x in coverage["selected"]), default=None),
        }
    return {"windows": snapshots, "projection": result["projection"]}


def training_audit(folder):
    path = folder / "hiercad_training.json"
    if not path.is_file():
        alternative = folder / "hiercad_training_failure.json"
        return {"status": "unavailable", "detail": read(alternative) if alternative.is_file() else None}
    result = read(path)
    facts = {}
    for arm in result["stages"]:
        rows = []
        for step in arm["steps"]:
            if step["status"] == "updated":
                check(step["optimizer_step_observed"] == 1 and step["parameter_sha256_before"] != step["parameter_sha256_after"],
                      f"{folder}: reported update has no actual parameter-change witness")
            rows.append({k: v for k, v in step.items()
                         if k in ("status", "condition", "train_step_seconds", "forward_seconds", "backward_seconds",
                                  "clip_and_optimizer_seconds", "peak_allocated_bytes", "loss_components", "error",
                                  "error_type", "optimizer_step_observed")})
        facts[arm["variant"]] = rows
    return {"status": "measured", "arms": facts, "settings": result["training_settings"],
            "scope": result["scope"], "record_sha256": sha(path)}


def main():
    root = Path(sys.argv[1])
    stage = root / "hiercad_with_occ"
    facts = {}
    for source, levels in (("t2cad-v1.0", ("L0", "L1", "L2", "L3")), ("ct-deepcad", ("CT",))):
        folder = stage / f"pilot_{source}" / source
        actual = {key: int((stage / f"{key}_{source}.exit").read_text()) for key in ("smoke", "pilot")}
        model_config = read(folder.parent / "run_config.json")
        smoke = read(stage / f"smoke_{source}" / source / "long_context_smoke.json")
        prep = read(folder / "selection.json")
        results = {"actual_exits": actual}
        results.update(
            smoke={"passed": smoke["passed"], "input_tokens": [x["total_input_tokens"] for x in smoke["pairs"]],
                   "peak_allocated_bytes": smoke["peak_allocated_bytes"],
                   "max_batch_delta_per_unit": max(x["score_delta_per_unit"] for x in smoke["pairs"])},
            selection={k: sum(bool(x.get(v, {}).get(w, False)) for x in prep["candidate_checks"])
                       for k, v, w in (("local_kernel_replay", "kernel_replay", "passed"),
                                       ("official_response_byte_match", "released_validation_reference", "response_bytes_equal"))},
            stage_status=read(folder.parent / "summary.json"),
            execution_sha=model_config["execution_sha"], device=model_config["physical_type"],
            controller_exit=int((stage / "controller.exit").read_text()),
            device_mapping=(stage / f"{source}.visible_device").read_text().strip(),
            readout=native_audit(folder, levels),
            timing=timing_audit(folder),
            training=training_audit(folder),
        )
        facts[source] = results
    original = root / "models" / "text2cad_native"
    facts["Text2CAD_native"] = {
        name: native_audit(original / name, levels)
        for name, levels in (("t2cad-v1.0", ("L0", "L1", "L2", "L3")), ("ct-deepcad", ("CT",)))
    }
    print(json.dumps(facts, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
