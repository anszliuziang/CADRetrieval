#!/usr/bin/env python3
"""Freeze the two DeepCAD tables and replay their full-train ridge on validation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from math import comb
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_ridge import raw_pairs

RECEIPT_SHA = "3c297920d0905b3dd5f14aa1b5ac163245a17b958cc3c2e1169ee65d5a8d055d"
RIDGE_SUMMARY_SHA = {
    "t2cad-v1.0": "fbd31e29eca995690ddae0fc3c6b10765f17abe84cdf122ff7f7d4664f6000eb",
    "ct-deepcad": "70c95f76b3564228c476d2772bbe238c40afdfc2738eb463a88fcdb881640313",
}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def record_json(path):
    return json.loads(path.read_bytes())


def check_file(path, expected):
    actual = pilot.sha256_file(path)
    require(actual == expected, f"Not the registered file: {path.name}, got {actual}")
    return actual


def verify_prior(root):
    archive = root / "manifests"
    check_file(archive / "manifest_receipt.json", RECEIPT_SHA)
    receipt = record_json(archive / "manifest_receipt.json")
    observed = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    require(observed == set(receipt["file_sha256"]) | {"manifest_receipt.json"}, "Frozen manifest paths differ")
    for relative, expected in receipt["file_sha256"].items():
        check_file(archive / relative, expected)
    return archive, receipt


def rand_controls(n, groups):
    """Exact random positive and same-caption hits for the registered gallery."""
    q = sum(groups.values())
    exact = {f"R@{k}": 100 * k / n for k in pilot.CUTOFFS}
    den = comb(n, 20)
    h20 = sum((Fraction(den - comb(n - m, 20), den) for m in groups.values()), Fraction())
    return {**exact, "MedR": (n + 1) / 2, "queries": q, "candidates": n,
            "distinct_texts": len(groups), "singleton_texts": sum(m == 1 for m in groups.values()),
            "singleton_R@20": exact["R@20"], "H@20": float(100 * h20 / len(groups)),
            "uid_R@1_ceiling": 100 * len(groups) / q,
            "uid_R@20_ceiling": 100 * sum(min(m, 20) for m in groups.values()) / q}


def candidate_axes(gallery, gallery_dir):
    require([r["cad_index"] for r in gallery] == list(range(len(gallery))), "Gallery indices differ")
    fields = ("cad_index", "cad_member", "minimal_member", "quantized_sha256", "member_sha256", "n_fields")
    pilot.write_jsonl(gallery_dir / "candidate_inputs.jsonl", (
        {f: r[f] for f in fields} for r in gallery))
    pilot.write_jsonl(gallery_dir / "candidate_uid_map.jsonl", (
        {"cad_index": r["cad_index"], "uid": r["uid"]} for r in gallery))


def query_axes(rows, levels, out):
    """Expose a text-only score axis. Registered targets are a distinct evaluation file."""
    rank_records = {}
    for level in levels:
        grouped = defaultdict(list)
        for r in rows:
            grouped[r["texts"][level]].append(r)
        texts = sorted(grouped, key=lambda t: (pilot.sha256_bytes(t.encode("utf-8")), t))
        text_ids = {t: i for i, t in enumerate(texts)}
        pilot.write_jsonl(out / f"{level}_score_queries.jsonl", (
            {"text_index": text_ids[t], "text": t, "text_sha256": pilot.sha256_bytes(t.encode("utf-8"))}
            for t in texts))
        pilot.write_jsonl(out / f"{level}_qrels.jsonl", (
            {"query_index": k, "uid": r["uid"], "text_index": text_ids[r["texts"][level]],
             "registered_quantized_sha256": r["quantized_sha256"]} for k, r in enumerate(rows)))
        rank_records[level] = {t: len(grouped[t]) for t in texts}
    return rank_records


def independent_uid_ranks(scores, queries, gallery):
    index = {r["uid"]: i for i, r in enumerate(gallery)}
    columns = np.arange(len(gallery))
    ranks = []
    for q, row in zip(queries, scores, strict=True):
        sorted_positions = np.lexsort((columns, -row))
        ranks.append(1 + int(np.flatnonzero(sorted_positions == index[q["uid"]])[0]))
    return ranks


def replay_ridge(root, prior, source, training, val_targets, val_gallery, out):
    import joblib
    import sklearn
    from sklearn.preprocessing import normalize

    folder = root / f"ridge_{source}_full"
    check_file(folder / "summary.json", RIDGE_SUMMARY_SHA[source])
    origin = record_json(folder / "summary.json")
    prior_scores = np.load(folder / "prediction_scores.npy", allow_pickle=False)
    model = joblib.load(folder / "fitted_ridge.joblib")
    levels = pilot.SOURCE_LEVELS[source]
    q = raw_pairs(val_targets, levels)
    require(origin["train_uids"] == len(training)
            and origin["train_pairs"] == len(training) * len(levels)
            and origin["training_subset"] == "full" and origin["selected_alpha"] == model["alpha"],
            f"Training coverage or selected weight identity differs: {source}")
    require(origin["selected_alpha"] == max(
        (float(alpha) for alpha in origin["regularizations"]),
        key=lambda alpha: (origin["regularizations"][str(alpha)]["macro_R@20"], alpha)),
        "Archived alpha violates validation R@20 then larger-alpha rule")
    require(q == pilot.read_jsonl(folder / "prediction_queries.jsonl")
            and val_gallery == pilot.read_jsonl(folder / "prediction_gallery.jsonl"), "Ridge validation axes differ")
    text, scaler = model["tfidf"], model["feature_scaler"]
    members = pilot.read_jsonl(prior / "feature_axis.jsonl")
    entries = {r["uid"]: i for i, r in enumerate(members)}
    raw = np.load(prior / "program_count_features.npy", mmap_mode="r", allow_pickle=False)
    require(raw.shape == (len(members), 29) and len(entries) == len(members), "Bad ridge feature axis")
    require(scaler.n_samples_seen_ == len(training), "Ridge feature scaler has wrong training count")
    cap = normalize(scaler.transform(raw[[entries[r["uid"]] for r in val_gallery]]))
    query = normalize(text.transform(r["text"] for r in q) @ model["coefficients"].T + model["bias"])
    scores = query @ cap.T
    require(scores.shape == prior_scores.shape and np.isfinite(scores).all(), "Wrong replay score coverage")
    max_delta = float(np.max(np.abs(scores - prior_scores)))
    require(max_delta <= 1e-12, f"Ridge score replay diverged by {max_delta}")
    results, metrics = pilot.evaluate_rankings(scores, q, val_gallery)
    independent = independent_uid_ranks(scores, q, val_gallery)
    require(independent == [r["rank"] for r in results], "Independent exact ranking differs")
    old = pilot.read_jsonl(folder / "query_ranks.jsonl")
    require(results == old and metrics == origin["validation_metrics"], "Validation ranks differ")
    pilot.write_jsonl(out / "ridge_validation_ranks.jsonl", iter(results))
    pilot.write_json(out / "ridge_validation.json", {"source": source, "training_uids": len(training),
                     "training_pairs": origin["train_pairs"], "alpha": model["alpha"],
                     "feature_dim": 29, "evaluated_full_text": True, "sklearn": sklearn.__version__,
                     "weight_sha256": pilot.sha256_file(folder / "fitted_ridge.joblib"),
                     "reference_score_sha256": pilot.sha256_file(folder / "prediction_scores.npy"),
                     "scores_replayed": int(scores.size), "max_abs_delta": max_delta,
                     "original_ranks_and_independent_ranks_exact": len(independent), "metrics": metrics})
    return {"exact_ranks": len(results), "scores_replayed": int(scores.size), "max_abs_delta": max_delta,
            "selected_alpha": model["alpha"], "metrics": metrics}


def run_source(root, prior, receipt, source, out, test_gallery, official_splits):
    src = prior / source
    source_out = out / source
    source_out.mkdir()
    files = {name: pilot.read_jsonl(src / path) for name, path in (
        ("train", "train.jsonl"), ("validation", "validation.jsonl"), ("test", "test_queries.jsonl"))}
    levels = pilot.SOURCE_LEVELS[source]
    for split, rows in files.items():
        order = [r["uid"] for r in rows]
        expected = pilot.COHORTS[source][split]
        require((len(order), pilot.sha256_lines(order)) == expected
                and len(set(order)) == len(order) and set(order) <= official_splits[split],
                f"{source} {split} UID freeze differs")
        require(all(tuple(r["texts"]) == levels and all(r["texts"].values()) for r in rows), "Missing frozen text")
    val = {r["uid"]: r for r in files["validation"]}
    texts_val = {uid: row["texts"] for uid, row in val.items()}
    val_uids, gallery_uids = pilot.select_panel(texts_val, source)
    val_targets = pilot.read_jsonl(src / "validation_targets.jsonl")
    val_gallery = pilot.read_jsonl(src / "validation_gallery.jsonl")
    require(len(val_targets) == 512 and len(val_gallery) == 2048
            and [r["uid"] for r in val_targets] == val_uids
            and [r["uid"] for r in val_gallery] == gallery_uids
            and val_targets == [val[uid] for uid in val_uids], "Validation target or gallery selection differs")
    bg = pilot.select_background(texts_val, source)
    require(len(bg) == 48 and len({r["text"] for r in bg}) == 48
            and bg == pilot.read_jsonl(src / "validation_background.jsonl"), "Background selection differs")
    pins = receipt["sources"][source]
    for entry, actual in (
        ("validation_target_set_sha256", pilot.sha256_lines(sorted(val_uids))),
        ("validation_gallery_sha256", pilot.sha256_lines(gallery_uids)),
        ("validation_background_sha256", pilot.sha256_lines(r["text"] for r in bg)),
    ):
        require(pins[entry] == actual, f"Validation/background hash differs at {entry}")
    for name in ("validation", "test"):
        (source_out / name).mkdir()
    candidate_axes(val_gallery, source_out / "validation")
    query_axes(val_targets, levels, source_out / "validation")
    pilot.write_jsonl(source_out / "validation_background_score_queries.jsonl", (
        {"text_index": i, "text": r["text"], "text_sha256": pilot.sha256_bytes(r["text"].encode("utf-8"))}
        for i, r in enumerate(bg)))
    membership = {r["uid"]: r["quantized_sha256"] for r in test_gallery}
    require(all(membership[r["uid"]] == r["quantized_sha256"] for r in files["test"]), "Mismatched registered test program")
    group_counts = query_axes(files["test"], levels, source_out / "test")
    random = {level: rand_controls(len(test_gallery), group_counts[level]) for level in levels}
    if source == "ct-deepcad":
        require((random["CT"]["distinct_texts"], random["CT"]["singleton_texts"]) == (2032, 1432),
                "CT group axis differs")
    old_text = {txt for r in files["train"] for txt in r["texts"].values()}
    old_program = {r["quantized_sha256"] for r in files["train"]}
    exposure = {level: {"identical_training_caption": sum(r["texts"][level] in old_text for r in files["test"]),
                        "identical_training_quantized_program": sum(r["quantized_sha256"] in old_program
                                                                   for r in files["test"])} for level in levels}
    ridge = replay_ridge(root, prior, source, files["train"], val_targets, val_gallery, source_out)
    out_summary = {"sources": {k: len(v) for k, v in files.items()}, "validation_gallery": len(val_gallery),
                   "background_queries": len(bg), "test_gallery": len(test_gallery),
                   "analytic_uniform_random": random, "exact_train_exposure": exposure, "validation_ridge": ridge}
    pilot.write_json(source_out / "summary.json", out_summary)
    return out_summary, {r["uid"] for r in files["test"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-pilot", required=True, type=Path)
    p.add_argument("--official-split", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    started = time.monotonic()
    prior, receipt = verify_prior(a.prior_pilot)
    check_file(a.official_split, pilot.INPUT_HASHES["split"])
    splits = pilot.official_splits(a.official_split)
    gallery = pilot.read_jsonl(prior / "test_gallery.jsonl")
    ids = [r["uid"] for r in gallery]
    require(len(ids) == 8046 and set(ids) == splits["test"]
            and pilot.sha256_lines(ids) == receipt["test_cad_axis_sha256"], "Official test gallery differs")
    a.out.mkdir()
    candidate_axes(gallery, a.out)
    prepared = {}
    for name in pilot.SOURCE_LEVELS:
        prepared[name] = run_source(a.prior_pilot, prior, receipt, name, a.out, gallery, splits)
        print(json.dumps({"source": name, "complete": True,
                          "validation_ranks": prepared[name][0]["validation_ridge"]["exact_ranks"]}), flush=True)
    overlap = len(prepared["t2cad-v1.0"][1] & prepared["ct-deepcad"][1])
    require(overlap == 7134, "T/CT common test UID count differs")
    metadata = {"scope": "formal-split inventory, validation score replay, exact random controls",
                "trained_method_test_scoring_executed": False, "input_manifest_receipt_sha256": RECEIPT_SHA,
                "test_gallery_order_sha256": receipt["test_cad_axis_sha256"], "shared_test_uids": overlap,
                "sources": {k: v[0] for k, v in prepared.items()}, "elapsed_s": time.monotonic() - started}
    pilot.write_json(a.out / "summary.json", metadata)
    checksums = {n.relative_to(a.out).as_posix(): pilot.sha256_file(n)
                 for n in sorted(a.out.rglob("*")) if n.is_file()}
    pilot.write_json(a.out / "files_sha256.json", checksums)


if __name__ == "__main__":
    main()
