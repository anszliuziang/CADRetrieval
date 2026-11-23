#!/usr/bin/env python3
"""Independently verify the saved validation ranks and one-epoch exposure ledgers."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

import numpy as np


LEVELS = {"t2cad-v1.0": ("L0", "L1", "L2", "L3"), "ct-deepcad": ("CT",)}
CUTOFFS = (1, 5, 10, 20)


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify_training(folder, registered, levels, method):
    pairs = [(row["texts"][level], row["uid"]) for row in registered for level in levels]
    known_edges = set(pairs)
    exposures = rows(folder / "training_exposure.jsonl")
    expected_exposures = [
        {"uid": row["uid"], "pair_exposures": len(levels), "level_exposures": dict.fromkeys(levels, 1)}
        for row in registered
    ]
    require(exposures == expected_exposures, f"{folder.name}: exposure ledger differs from registrations")
    visited = []
    counts = Counter()
    steps = rows(folder / "training_steps.jsonl")
    for index, step in enumerate(steps, 1):
        indices = step["train_pair_indices"]
        require(index == step["update"] and len(indices) == step["global_pairs"] and isfinite(step["loss"]),
                f"{folder.name}: invalid update {index}")
        require(all(type(i) is int and 0 <= i < len(pairs) for i in indices), f"{folder.name}: invalid pair index")
        visited.extend(indices)
        batch = [pairs[i] for i in indices]
        off_diag = [(i, j) for i, (text, _) in enumerate(batch) for j, (_, uid) in enumerate(batch)
                    if i != j and (text, uid) in known_edges]
        same_uid = sum(batch[i][1] == batch[j][1] for i, j in off_diag)
        same_text = sum(batch[i][0] == batch[j][0] for i, j in off_diag)
        independent = {
            "off_diagonal_cells": len(batch) * (len(batch) - 1),
            "known_off_diagonal_positives": len(off_diag),
            "known_off_diagonal_same_uid": same_uid,
            "known_off_diagonal_same_text": same_text,
            "known_off_diagonal_other_registration": sum(
                batch[i][1] != batch[j][1] and batch[i][0] != batch[j][0] for i, j in off_diag),
            "text_anchors_with_known_off_diagonal_positive": len({i for i, _ in off_diag}),
            "cad_anchors_with_known_off_diagonal_positive": len({j for _, j in off_diag}),
        }
        require(all(step[key] == expected for key, expected in independent.items()),
                f"{folder.name}: false-negative tally differs at update {index}")
        counts.update(independent)
    if method == "spectrabrep":
        for direction in ("text_to_cad", "cad_to_text"):
            counts[f"known_negative_treatment_{direction}"] = counts["known_off_diagonal_positives"]
    require(sorted(visited) == list(range(len(pairs))), f"{folder.name}: training pairs are not an exact permutation")
    summary = read(folder / "training_summary.json")
    require(summary["actual_training_uids"] == len(registered) == 4096
            and summary["actual_training_pairs"] == len(pairs)
            and summary["actual_updates"] == len(steps)
            and summary["known_positive_audit"] == dict(counts),
            f"{folder.name}: inconsistent training summary")
    return {"uids": len(registered), "pairs": len(pairs), "updates": len(steps), "positive_counts": dict(counts)}


def verify_ranking(folder, queries, gallery, ridge):
    require(rows(folder / "prediction_queries.jsonl") == queries, f"{folder.name}: query axis differs")
    require(rows(folder / "prediction_gallery.jsonl") == gallery, f"{folder.name}: candidate axis differs")
    matrix = np.load(folder / "prediction_scores.npy", mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (len(queries), len(gallery)), f"{folder.name}: matrix dimension mismatch")
    rank_path = folder / ("query_ranks.jsonl" if ridge else "ranks.jsonl")
    saved = rows(rank_path)
    require(len(saved) == len(queries), f"{folder.name}: rank coverage mismatch")
    columns = {row["uid"]: j for j, row in enumerate(gallery)}
    indices = np.array([row["cad_index"] for row in gallery], dtype=np.int64)
    by_level = {}
    for i, (query, reported) in enumerate(zip(queries, saved, strict=True)):
        score = matrix[i]
        valid = np.flatnonzero(np.isfinite(score))
        order = valid[np.lexsort((indices[valid], -score[valid]))]
        match = np.flatnonzero(order == columns[query["uid"]])
        rank = int(match[0]) + 1 if len(match) else len(gallery) + 1
        by_level.setdefault(query["level"], []).append(rank)
        expected = {"uid": query["uid"], "level": query["level"],
                    "text_sha256": sha256(query["text"].encode("utf-8")).hexdigest(),
                    "rank": rank, "valid_candidates": len(valid), "failed": not len(match)}
        require(reported == expected, f"{folder.name}: rank or identity differs at row {i}")
    results = {}
    for level, ranks in by_level.items():
        hits = {str(k): sum(r <= k for r in ranks) for k in CUTOFFS}
        results[level] = {"queries": len(ranks), "candidates": len(gallery), "hits": hits,
                          **{f"R@{k}": 100.0 * hits[str(k)] / len(ranks) for k in CUTOFFS},
                          "MedR": float(median(ranks)), "failures": sum(r > len(gallery) for r in ranks)}
    reported = read(folder / "summary.json")["validation_metrics"] if ridge else read(folder / "metrics.json")
    require(reported == results, f"{folder.name}: recomputed validation metrics differ")
    return {"rank_rows_verified": len(saved), "metrics": results, "rank_sha256": fingerprint(rank_path),
            "matrix_sha256": fingerprint(folder / "prediction_scores.npy")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.run / "manifests"
    receipt = read(manifest / "manifest_receipt.json")
    facts = {}
    for source, levels in LEVELS.items():
        suffix = "t" if source == "t2cad-v1.0" else "ct"
        scoped = manifest / source
        files = ("pilot_train.jsonl", "validation_targets.jsonl", "validation_gallery.jsonl")
        for name in files:
            require(fingerprint(scoped / name) == receipt["file_sha256"][f"{source}/{name}"],
                    f"{source}: frozen file hash differs for {name}")
        train, targets, gallery = (rows(scoped / name) for name in files)
        axes = [{row["uid"] for row in axis} for axis in (train, targets, gallery)]
        require([len(axis) for axis in (train, targets, gallery)] == [len(axis) for axis in axes]
                == [4096, 512, 2048] and not axes[0] & axes[2] and axes[1] <= axes[2],
                f"{source}: invalid validation identities")
        queries = [{"uid": row["uid"], "level": level, "text": row["texts"][level]}
                   for level in levels for row in targets]
        facts[source] = {}
        for method in ("ridge_full", "ridge_pilot", "dual", "spectra"):
            ridge = method.startswith("ridge")
            relative = f"ridge_{source}_{method[6:]}" if ridge else f"models/{method}_{suffix}"
            folder = args.run / relative
            exit_code = int((args.run / f"{relative}.exit").read_text())
            require(exit_code == 0, f"{relative}: execution exited {exit_code}")
            verified = verify_ranking(folder, queries, gallery, ridge)
            if not ridge:
                verified["training"] = verify_training(folder, train, levels,
                                                       "spectrabrep" if method == "spectra" else "dual")
            facts[source][method] = verified
            print(source, method, verified["rank_rows_verified"], flush=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"all_passed": True, "scope": "Validation only", "sources": facts,
                   "manifest_receipt_sha256": fingerprint(manifest / "manifest_receipt.json")},
                  stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
