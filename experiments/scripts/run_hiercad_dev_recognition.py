#!/usr/bin/env python3
"""Score the existing 40 paired development fixtures with the released HierCAD readout."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from experiments.scripts.run_910a_text2cad_dev_recognition import (
    HISTORIES,
    VARIANTS,
    fixture_design_key,
    load_fixtures,
    member_name,
    metrics,
    write_json,
    write_row,
)
from experiments.src.idea01 import in_domain_pilot_native as native
from experiments.src.idea01.in_domain_pilot import minimal_member
from experiments.src.idea01.in_domain_pilot_geometry import fingerprint

PRIOR_LOAD = "hiercad_with_occ/pilot_ct-deepcad/model_load.json"
PRIOR_PANEL = "hiercad_with_occ/pilot_ct-deepcad/ct-deepcad/CT_readout.json"
REFERENCE_SHA = {
    PRIOR_LOAD: "313772df7e8060ab2505692c686942031909f49b372d087bf7845202055a8759",
    PRIOR_PANEL: "8657caad0d238cceb7395c7c851a6c3a4176c416779497cbdf8550d9aab026f4",
}
ENTITIES = ("A", "B")


def pinned_json(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned reference differs: {path}")
    return json.loads(data)


def verify_file(path, expected):
    with path.open("rb") as stream:
        actual = file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Pinned asset differs: {path}")


def verify_assets(args):
    report = pinned_json(args.reference / PRIOR_LOAD, REFERENCE_SHA[PRIOR_LOAD])
    for local, previous in ((args.base, report["base"]), (args.adapter, report["adapter"])):
        for name, item in previous["files"].items():
            file = local / name
            if file.stat().st_size != item["bytes"]:
                raise ValueError(f"Pinned asset size differs: {file}")
            verify_file(file, item["sha256"])
    verify_file(args.minimal_zip, report["minimal_archive_sha256"])
    return report


def replay_panel(model, tokenizer, device, reference, minimal_zip):
    prior = pinned_json(reference / PRIOR_PANEL, REFERENCE_SHA[PRIOR_PANEL])
    expected = prior["single"]["pairs"]
    axes = {(row["query_i"], row["candidate_i"]) for row in expected}
    if len(expected) != 16 or axes != set(product(range(4), repeat=2)):
        raise ValueError("Reference HierCAD singleton score axes changed")
    archive = native.MinimalArchive(minimal_zip)
    backend = native.HierCadReadout(model, tokenizer, archive, device)
    rows = []
    try:
        for row in expected:
            q = prior["query_axis"][row["query_i"]]
            uid = prior["uid_axis"][row["candidate_i"]]
            if q["uid"] != row["query_uid"] or uid != row["candidate_uid"]:
                raise ValueError("Reference HierCAD UID axes changed")
            info = backend.prepare({"minimal_member": minimal_member(uid)}, q["text"])
            cells, _ = backend.score([info])
            actual = cells[0]
            same = all(actual[key] == row[key] for key in (
                "input_sha256", "target_sha256", "target_token_sha256", "effective_units", "query_tokens",
            ))
            err = abs(actual["s0"] - row["s0"]) / row["effective_units"]
            rows.append({"query_i": row["query_i"], "candidate_i": row["candidate_i"],
                         "reference_s0": row["s0"], "actual": actual, "abs_mean_delta": err,
                         "identity_exact": same, "passed": same and err <= native.H_SCORE_MEAN_ATOL
                         and actual["logit_dtype"] == "torch.bfloat16"})
    finally:
        archive.close()
    return {"passed": all(row["passed"] for row in rows), "pairs": rows,
            "max_abs_mean_delta": max(row["abs_mean_delta"] for row in rows)}


class FixedTargets:
    def __init__(self, histories):
        self.targets = {}
        for row in histories:
            minimal = row["normalized_minimal"]
            if fingerprint(minimal) != row["normalized_minimal_sha256"]:
                raise ValueError("HierCAD input programme hash differs")
            target = native.hiercad_target(minimal)
            prior = row["hiercad_target"]
            if (target.response != prior["response"] or native.digest(target.response) != prior["response_sha256"]
                    or fingerprint(target.replay_program) != prior["replay_program_sha256"]):
                raise ValueError("Author renderer output differs from replayed geometry")
            self.targets[member_name(row["pair_id"], row["history"])] = target, row["normalized_minimal_sha256"]

    def load(self, member):
        return self.targets[member]


def encode_cells(backend, pair, indexed, order, variant):
    pair_id = pair["pair_id"]
    result = []
    for query, entity in product(ENTITIES, repeat=2):
        history = f"{entity}{order}"
        saved = indexed[pair_id, history]["inputs"]["hiercad"]
        prior = saved["caption_inputs"][query][variant]
        text = pair["captions"][query][variant]
        prepared = backend.prepare({"minimal_member": member_name(pair_id, history)}, text)
        ids = prepared["encoded"]
        target = {"target_ids": ids.target_ids,
                  "target_mask": [label != -100 for label in ids.labels[ids.target_start:]]}
        if (prior["status"] != "ok" or prior["truncated"] or not saved["query_invariant_target"]
                or fingerprint(ids.input_ids) != prior["full_input_sha256"]
                or fingerprint(ids.input_ids[:ids.target_start]) != prior["input_sha256"]
                or fingerprint(target) != saved["input_sha256"] or len(ids.target_ids) != saved["target_tokens"]):
            raise ValueError("Unverified target labels in HierCAD paired fixture")
        result.append(prepared)
    return result


def check_scores(scores):
    if len(scores) != 4 or any(score["logit_dtype"] != "torch.bfloat16" or not math.isfinite(score["s0"])
                              for score in scores):
        raise ValueError("Invalid BF16 HierCAD score")
    return [score["s0"] for score in scores]


def run_paired(backend, pairs, indexed, output):
    aggregates = defaultdict(Counter)
    layouts = []
    with (output / "views.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for i, pair in enumerate(pairs, start=1):
            family, attr, index = fixture_design_key(pair)
            truth = pair["replay_caption_truth"]["hiercad"]
            if (pair["geometry"]["hiercad"]["status"] != "verified"
                    or any(part["status"] != "pass" for part in pair["models"]["hiercad"].values())
                    or any(truth[h] != {e: h[0] == e for e in ENTITIES} for h in HISTORIES)):
                raise ValueError("Pair lacks native HierCAD geometry or caption truth checks")
            group = aggregates[(family, attr)]
            outcomes = []
            for order, variant in product(range(2), VARIANTS):
                prepared = encode_cells(backend, pair, indexed, order, variant)
                scores = [backend.score([one])[0][0] for one in prepared]
                single = check_scores(scores)
                values = metrics(single)
                outcomes.append(values)
                key = {"pair_id": pair["pair_id"], "order": order, "variant": variant}
                write_row(stream, {**key, "scores": scores, "metrics": values})
                if index == 0 and order == 0 and variant in ("near_first", "far_first_rephrased"):
                    batched, details = backend.score(prepared)
                    batch_values = metrics(check_scores(batched))
                    layouts.append({**key, "batch_context": details["context"],
                                    "batched_scores": batched, "batched_metrics": batch_values,
                                    "abs_mean_deltas": [abs(a["s0"] - b["s0"]) / a["effective_units"]
                                                        for a, b in zip(scores, batched, strict=True)],
                                    "both_cads_changed": values["both_cads"] != batch_values["both_cads"],
                                    "both_captions_changed": values["both_captions"] != batch_values["both_captions"]})
            joint = all(row["both_cads"] for row in outcomes)
            swap = all(row["swapped_both_cads"] for row in outcomes)
            group.update(pairs=1, views=len(outcomes), scores=4 * len(outcomes),
                         joint=joint, joint_swap=swap, N_twice=int(joint) + int(swap),
                         both_cads=sum(row["both_cads"] for row in outcomes),
                         both_captions=sum(row["both_captions"] for row in outcomes),
                         D_positive=sum(row["D"] > 0 for row in outcomes), ties=sum(row["ties"] for row in outcomes))
            if i % 10 == 0:
                print(f"HierCAD completed {i}/40 pairs", flush=True)
    if len(layouts) != 8 or sum(row["scores"] for row in aggregates.values()) != 1280:
        raise ValueError("Incomplete singleton or layout score coverage")
    write_json(output / "layouts.json", layouts)
    return [{"family": f, "attribute": a, **counts} for (f, a), counts in aggregates.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "minimal-zip", "hiercad-repo", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Use the clean deployed implementation")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    verified = verify_assets(args)
    pairs, histories, indexed = load_fixtures(args.reference)
    revision = native.use_hiercad_repo(args.hiercad_repo)
    tokenizer = native.load_hiercad_tokenizer(args.base)
    store = FixedTargets(histories)
    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model = native.hiercad_scoring_model(args.base, args.adapter, device, torch.bfloat16)
    backend = native.HierCadReadout(model, tokenizer, store, device)
    write_json(args.out / "runtime.json", {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                 text=True).strip(), "chip": torch.npu.get_device_name(device), "dtype": "bfloat16",
                 "attention": model.config._attn_implementation, "hiercad_source": revision,
                 "verified_model_files": verified, "setup_wall_seconds": time.perf_counter() - started})
    reference = replay_panel(model, tokenizer, device, args.reference, args.minimal_zip)
    write_json(args.out / "replay.json", reference)
    print(f"HierCAD singleton replay passed={reference['passed']}", flush=True)
    if not reference["passed"]:
        return 3
    score_start = time.perf_counter()
    aggregates = run_paired(backend, pairs, indexed, args.out)
    torch.npu.synchronize()
    write_json(args.out / "summary.json", {"groups": aggregates, "missing_scores": 0,
               "new_layout_scores": 32, "reserved_peak_bytes": torch.npu.max_memory_reserved(),
               "allocated_peak_bytes": torch.npu.max_memory_allocated(),
               "paired_wall_seconds": time.perf_counter() - score_start,
               "total_wall_seconds": time.perf_counter() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
