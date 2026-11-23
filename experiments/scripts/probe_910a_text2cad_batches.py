#!/usr/bin/env python3
"""Check representative full and tail Text2CAD batches on a qualified 910A."""

import argparse
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import torch_npu

from experiments.scripts.run_910a_text2cad_dev_recognition import PINS, REPLAY, read_pinned, write_json
from experiments.src.idea01.in_domain_pilot import cad_member
from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore
from experiments.src.idea01.text2cad_port.loader import load_frozen_text2cad


def check_batch(backend, prepared, expected):
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    start = time.perf_counter()
    result, _ = backend.score(prepared)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    scores = []
    for actual, reference in zip(result, expected, strict=True):
        same = all(actual[key] == reference[key] for key in (
            "query_tokens", "effective_units", "input_sha256", "target_sha256",
        ))
        delta = abs(actual["s0"] - reference["s0"])
        tolerance = max(0.001, 0.000001 * abs(reference["s0"]))
        scores.append({"query_i": reference["query_i"], "candidate_i": reference["candidate_i"],
                       "actual": actual, "reference_s0": reference["s0"],
                       "delta": delta, "tolerance": tolerance, "input_and_mask_match": same,
                       "passed": same and actual["logit_dtype"] == "torch.float32"
                       and math.isfinite(delta) and delta <= tolerance})
    return {"batch": len(result), "passed": all(row["passed"] for row in scores),
            "wall_seconds": elapsed, "peak_reserved_bytes": torch.npu.max_memory_reserved(), "scores": scores,
            "max_abs_delta": max(row["delta"] for row in scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Use the clean deployed revision")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    pins = json.loads(PINS.read_bytes())
    assets = args.assets
    revision = pins["dataset_revision"]
    prior = json.loads(read_pinned(assets / "reference", REPLAY))
    expected = prior["single"]["pairs"]
    if len(expected) != 16 or {(row["query_i"], row["candidate_i"]) for row in expected} != set(
        product(range(4), repeat=2)
    ):
        raise ValueError("Expected the existing 16 unique historical singleton results")
    members = [cad_member(uid) for uid in prior["uid_axis"]]
    store = ZipCandidateStore(assets / "datasets/Text2CAD-v1.0" / revision / "cad_seq.zip", members)
    if store.preload()["n_loaded"] != 4:
        raise ValueError("Historical candidates are not all valid")
    model, info = load_frozen_text2cad(
        assets / "models/Text2CAD-v1.0" / revision / "Text2CAD_1.0.pth",
        assets / "models/bert-large-uncased" / pins["bert_revision"],
        device, expected_keys_manifest=Path("experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"),
    )
    write_json(args.out / "runtime.json", {"load": info, "hardware": torch.npu.get_device_name(device),
               "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "precision": "preserve", "reference": REPLAY})
    backend = Text2CadReadout(model, store, dict(zip(members, range(4), strict=True)), device)
    prepared = [backend.prepare({"cad_member": members[row["candidate_i"]]},
                                prior["query_axis"][row["query_i"]]["text"]) for row in expected]
    checks = []
    for batch in (16, 14):
        for repeat in (0, 1):
            result = check_batch(backend, prepared[:batch], expected[:batch])
            write_json(args.out / f"batch_{batch}_repeat_{repeat}.json", result)
            checks.append({key: value for key, value in result.items() if key != "scores"})
            print(f"batch {batch} repeat {repeat} passed={result['passed']} delta={result['max_abs_delta']:.8g}",
                  flush=True)
            if not result["passed"]:
                write_json(args.out / "summary.json", {"passed": False, "checks": checks})
                return 3
    write_json(args.out / "summary.json", {"passed": True, "checks": checks})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
