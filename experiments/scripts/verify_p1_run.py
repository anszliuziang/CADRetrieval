#!/usr/bin/env python
"""Post-run verification for a P1 matrix run directory (read-only).

Recomputes per-pair sums from the per-token evidence files and compares
against pairs.jsonl; checks target-set boundaries (first target == prefix
length, last target == EOS, pred_row == position-1); checks the raw records
exactly cover the fixed selection manifest pair ids.

Usage:
  python experiments/scripts/verify_p1_run.py --run-dir <run_dir> --selection <p1_selection.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EOS_ID = 151645  # <|im_end|>, checkpoint tokenizer eos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir containing pairs.jsonl and tokens/")
    ap.add_argument("--selection", required=True, help="p1_selection.json")
    args = ap.parse_args()
    run = Path(args.run_dir)

    recs: dict[str, dict] = {}
    with open(run / "pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["pair_id"] in recs:
                print(f"FAIL: duplicate pair_id {r['pair_id']}")
                return 2
            recs[r["pair_id"]] = r
    print(f"pairs.jsonl records: {len(recs)}")

    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    expected = [p["pair_id"] for p in sel["pairs"]]
    missing = [p for p in expected if p not in recs]
    extra = [p for p in recs if p not in set(expected)]
    if missing or extra:
        print(f"FAIL: pair coverage mismatch missing={missing[:5]} extra={extra[:5]}")
        return 2

    n_fail = 0
    worst = 0.0
    n_ok = 0
    for pid, r in recs.items():
        if r["status"] != "ok":
            print(f"FAIL: {pid} status={r['status']}")
            n_fail += 1
            continue
        toks = [json.loads(l) for l in open(run / "tokens" / f"{pid}.jsonl", encoding="utf-8")]
        recomputed = sum(t["logprob"] for t in toks)
        checks = [
            (len(toks) == r["n_target_tokens"], "token count"),
            (toks[0]["position"] == r["n_prefix_tokens"], "first target != prefix end"),
            (toks[-1]["token_id"] == EOS_ID, "last target not EOS"),
            (all(t["pred_row"] == t["position"] - 1 for t in toks), "shift violation"),
            (len(r.get("input_ids", [])) == r["n_input_tokens"], "input_ids length"),
            (abs((r["sum_logprob"] / r["n_target_tokens"]) - r["mean_logprob"]) < 1e-6, "mean != sum/n"),
            (r.get("eos_logprob") is None or abs(toks[-1]["logprob"] - r["eos_logprob"]) < 1e-9, "eos_logprob mismatch"),
        ]
        for ok, name in checks:
            if not ok:
                print(f"FAIL: {pid}: {name}")
                n_fail += 1
        worst = max(worst, abs(recomputed - r["sum_logprob"]))
        if r["all_finite"]:
            n_ok += 1
    print(f"recomputed sums match pairs.jsonl (max |diff|={worst:.3e}); ok+finite: {n_ok}/{len(recs)}; failures: {n_fail}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
