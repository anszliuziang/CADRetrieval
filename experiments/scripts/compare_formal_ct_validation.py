#!/usr/bin/env python3
"""Pair the completed CT tower's validation ranks with the authenticated 4,096-UID pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.scripts.audit_in_domain_pilot_scores import fingerprint, read, require, rows, verify_ranking

PILOT_SHA256 = "4d9ec79bcafdd94a378d59effa1361c2de1d008379ae99a5cf8780a9ffece911"
CUTOFFS = (1, 5, 10, 20)
BOOTSTRAP_SEED = 20261123
REPLICATES = 5000


def verify_pilot(root):
    manifest = root / "SHA256SUMS"
    require(fingerprint(manifest) == PILOT_SHA256, "Unknown archived pilot")
    pins = dict((name, digest) for digest, name in (line.split("  ", 1) for line in manifest.read_text().splitlines()))
    require(set(pins) | {manifest.name} == {file.name for file in root.iterdir()}, "Pilot evidence coverage changed")
    require(all(fingerprint(root / name) == sha for name, sha in pins.items()), "Pilot evidence SHA mismatch")


def compare(args):
    verify_pilot(args.pilot)
    formal_proof = read(args.formal_audit)
    require(formal_proof["all_passed"] is True and formal_proof["training"]["uids"] == 148046
            and formal_proof["runner_summary_sha256"] == fingerprint(args.formal / "run.json"),
            "The formal checkpoint does not have a verified full-training receipt")
    previous = read(args.pilot / "run.json")
    current = read(args.formal / "run.json")
    require(all(record["source"] == "ct-deepcad" and record["method"] == "dual-tower"
                and record["status"] == "completed" for record in (previous, current)), "Wrong retrieval runs")
    require(previous["training"]["actual_training_uids"] == 4096 and current["training"]["actual_training_uids"] == 148046,
            "The training exposure or dataset changed")
    for metadata_name in ("prediction_queries.jsonl", "prediction_gallery.jsonl", "prediction_scores.npy", "ranks.jsonl"):
        require(fingerprint(args.formal / metadata_name) == current["files"][metadata_name],
                f"Formal validation evidence changed at {metadata_name}")
    queries = rows(args.pilot / "prediction_queries.jsonl")
    gallery = rows(args.pilot / "prediction_gallery.jsonl")
    require(len(queries) == 512 and len(gallery) == 2048 and all(row["level"] == "CT" for row in queries),
            "Expected the original CT validation panel")
    metrics = {label: verify_ranking(path, queries, gallery, False)["metrics"]["CT"]
               for label, path in (("pilot_4096", args.pilot), ("formal_148046", args.formal))}
    before, after = (np.asarray([row["rank"] for row in rows(path / "ranks.jsonl")], dtype=np.int64)
                     for path in (args.pilot, args.formal))

    # Resample exact raw-text groups once per draw, giving each sampled tagged query one vote.
    group_axis, groups = np.unique([record["text"] for record in queries], return_inverse=True)
    group_sizes = np.bincount(groups)
    random = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = random.integers(0, len(group_axis), size=(REPLICATES, len(group_axis)))
    denominator = group_sizes[sampled].sum(axis=1)
    paired = {}
    for k in CUTOFFS:
        delta = (after <= k).astype(np.int64) - (before <= k).astype(np.int64)
        group_delta = np.bincount(groups, weights=delta)
        draws = 100.0 * group_delta[sampled].sum(axis=1) / denominator
        paired[f"R@{k}"] = {"difference_pp": float(100.0 * delta.mean()),
                            "bootstrap_95_percent": np.quantile(draws, [0.025, 0.975]).tolist()}
    args.out.mkdir()
    with (args.out / "paired_ranks.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("uid", "exact_text_group", "pilot_4096_rank", "formal_148046_rank", "rank_gain"))
        for query, group, old, new in zip(queries, groups, before, after, strict=True):
            writer.writerow((query["uid"], int(group), int(old), int(new), int(old - new)))
    return {"scope": "same frozen CT validation panel, descriptive cross-run comparison",
            "queries": len(queries), "gallery": len(gallery), "exact_text_groups": len(group_axis),
            "metrics": metrics, "paired_differences": paired, "replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
            "rank_improved": int((after < before).sum()), "rank_equal": int((after == before).sum()),
            "rank_worsened": int((after > before).sum()),
            "pilot_manifest_sha256": PILOT_SHA256, "formal_audit_sha256": fingerprint(args.formal_audit),
            "formal_runner_summary_sha256": formal_proof["runner_summary_sha256"],
            "paired_ranks_sha256": fingerprint(args.out / "paired_ranks.csv")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "formal", "formal-audit", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    with (args.out / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
