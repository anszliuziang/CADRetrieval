#!/usr/bin/env python
"""P1 analysis driver: recompute every summary from the RAW per-pair records.

Reads pairs.jsonl (raw per-pair score records from run_p1_matrix.py), the fixed
selection manifest (p1_selection.json) and the pinned validation texts; writes
to --out-dir:
  metrics.json            all diagnostics with bootstrap CIs and baseline timing
  per_query_table.jsonl   per-query ranks/win rates for the three score arms
  near_duplicates.jsonl   score-independent program token Jaccard screen

Exits non-zero if the raw records do not exactly cover the fixed pair plan
with status ok + finite scores (no success-only subsets, plan sec.4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.coverage import comment_line_ratio, has_export_boilerplate  # noqa: E402
from idea01.data_audit import read_jsonl  # noqa: E402
from idea01.p1_analysis import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_RESAMPLES,
    RANDOM_BASELINE_SEED,
    SCORE_ARMS,
    AnalysisError,
    baseline_empty,
    baseline_length,
    baseline_random,
    baseline_tfidf,
    empty_score_map,
    near_duplicate_screen,
    ranking_analysis,
    same_candidate_margins,
    score_lookup,
    stratified_diagnostics,
    validate_pair_records,
)

EXIT_OK = 0
EXIT_ANOMALY = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="raw pairs.jsonl from run_p1_matrix.py")
    ap.add_argument("--selection", required=True, help="p1_selection.json (fixed selection manifest)")
    ap.add_argument("--data-dir", required=True, help="pinned dataset dir (validation texts)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    records = [json.loads(ln) for ln in Path(args.pairs).read_text(encoding="utf-8").splitlines() if ln.strip()]

    expected_ids = [p["pair_id"] for p in selection["pairs"]]
    try:
        validate_pair_records(records, expected_ids)
    except AnalysisError as e:
        print(f"[p1-analysis] raw record validation failed: {e}", flush=True)
        return EXIT_ANOMALY

    rows = read_jsonl(Path(args.data_dir) / "data" / "data_val.jsonl")
    query_texts: dict[str, str] = {}
    programs: dict[str, str] = {}
    for qp in selection["queries"]:
        query_texts[qp["uid"]] = rows[qp["row_index"]]["input"]
        programs[qp["uid"]] = rows[qp["row_index"]]["output"]
        for c in qp["candidates"]:
            programs[c["uid"]] = rows[c["row_index"]]["output"]

    table = score_lookup(records)
    empty_sums = empty_score_map(records)
    metrics: dict = {
        "task_id": selection.get("task_id"),
        "seed": args.seed,
        "n_resamples": args.n_resamples,
        "n_pairs": len(records),
        "labels": [
            "all ranking numbers are exact-UID pairing sanity on official paired data, NOT semantic retrieval R@K",
            "mean arm is a length diagnostic, not a full conditional probability",
            "corrected arm = S(q,x)-S(empty,x) is a known candidate-prior correction, not a new method",
        ],
    }

    t0 = time.time()
    margins = same_candidate_margins(table, selection)
    metrics["primary_same_candidate_margin"] = margins
    print(
        f"[p1-analysis] margin: mean={margins['mean_margin']['point']:.3f} "
        f"CI=[{margins['mean_margin']['ci_lo']:.3f},{margins['mean_margin']['ci_hi']:.3f}] "
        f"n_pos={margins['n_positive']}/{margins['n']}",
        flush=True,
    )

    metrics["ranking_by_arm"] = {}
    per_query_rows: dict[str, dict] = {}
    for arm in SCORE_ARMS:
        ra = ranking_analysis(table, selection, empty_sums, arm, args.n_resamples, args.seed)
        metrics["ranking_by_arm"][arm] = ra
        for p in ra["per_query"]:
            row = per_query_rows.setdefault(p["query_uid"], {"query_uid": p["query_uid"], "query_index": p["query_index"]})
            row[f"rank_{arm}"] = p["rank"]
            row[f"top1_{arm}"] = p["top1_strict"]
            row[f"winrate_{arm}"] = p["win_rate"]
        print(
            f"[p1-analysis] arm={arm}: mean_rank={ra['mean_rank']:.2f} top1={ra['top1_strict_rate']['point']:.3f} "
            f"win={ra['mean_win_rate']['point']:.3f}",
            flush=True,
        )

    metrics["baselines"] = {}
    for name, fn in (
        ("random_fixed_seed", lambda: baseline_random(selection, RANDOM_BASELINE_SEED)),
        ("response_length_ascending", lambda: baseline_length(selection)),
        ("empty_query_score", lambda: baseline_empty(selection, empty_sums)),
        ("char3_tfidf_cosine", lambda: baseline_tfidf(selection, query_texts, programs)),
    ):
        tb = time.time()
        res = fn()
        res["elapsed_s"] = round(time.time() - tb, 3)
        metrics["baselines"][name] = res
        print(
            f"[p1-analysis] baseline={name}: mean_rank={res['mean_rank']:.2f} "
            f"top1={res['top1_strict_rate']['point']:.3f} win={res['mean_win_rate']['point']:.3f} "
            f"({res['elapsed_s']}s)",
            flush=True,
        )

    nd = near_duplicate_screen(selection, programs)
    metrics["near_duplicate_screen"] = {k: v for k, v in nd.items() if k != "pairs"}
    with open(out_dir / "near_duplicates.jsonl", "w", encoding="utf-8") as f:
        for r in nd["pairs"]:
            f.write(json.dumps(r) + "\n")
    if nd["data_admission_error"]:
        print("[p1-analysis] DATA ADMISSION ERROR: exact-identical programs across uids; stop and fix (plan sec.3)", flush=True)

    metrics["stratified_diagnostics"] = stratified_diagnostics(
        table, selection, margins, programs, comment_line_ratio, has_export_boilerplate
    )
    metrics["elapsed_s"] = round(time.time() - t0, 2)

    with open(out_dir / "per_query_table.jsonl", "w", encoding="utf-8") as f:
        for uid in sorted(per_query_rows, key=lambda u: per_query_rows[u]["query_index"]):
            f.write(json.dumps(per_query_rows[uid]) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[p1-analysis] wrote {out_dir / 'metrics.json'}", flush=True)
    return EXIT_ANOMALY if nd["data_admission_error"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
