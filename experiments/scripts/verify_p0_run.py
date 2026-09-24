#!/usr/bin/env python
"""Post-run verification + descriptive stats for a P0 smoke run directory.

Read-only. Recomputes per-pair sums from the per-token evidence files and
compares against pairs.jsonl; checks target-set boundaries (first target ==
prefix length, last target == EOS); reports condition margins. This is a
verification aid, not a retrieval metric: no R@K is computed (out of scope).

Usage:
  python experiments/scripts/verify_p0_run.py --run-dir experiments/runs/idea01_p0_20260924_r01
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

EOS_ID = 151645  # <|im_end|>, checkpoint tokenizer eos (recorded in environment.json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)

    recs = {}
    with open(run / "smoke" / "pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["pair_id"]] = r
    print(f"pairs.jsonl records: {len(recs)}")

    worst = 0.0
    eos_by_cond: dict[str, list[float]] = {}
    n_ok = 0
    for pid, r in recs.items():
        toks = [json.loads(l) for l in open(run / "smoke" / "tokens" / f"{pid}.jsonl", encoding="utf-8")]
        recomputed = sum(t["logprob"] for t in toks)
        assert len(toks) == r["n_target_tokens"], f"{pid}: token count mismatch"
        assert toks[0]["position"] == r["n_prefix_tokens"], f"{pid}: first target != prefix end"
        assert toks[-1]["token_id"] == EOS_ID, f"{pid}: last target is not EOS"
        assert all(t["pred_row"] == t["position"] - 1 for t in toks), f"{pid}: shift violation"
        worst = max(worst, abs(recomputed - r["sum_logprob"]))
        eos_by_cond.setdefault(r["condition"], []).append(toks[-1]["logprob"])
        if r["status"] == "ok" and r["all_finite"]:
            n_ok += 1
    print(f"recomputed sums match pairs.jsonl (max |diff|={worst:.3e}); ok+finite: {n_ok}/{len(recs)}")

    for cond, v in sorted(eos_by_cond.items()):
        print(f"EOS logprob [{cond}]: mean={statistics.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}")

    margins = []
    for c in range(20):
        d = {r["condition"]: r["sum_logprob"] for pid, r in recs.items() if pid.startswith(f"c{c:02d}_")}
        if len(d) == 3:
            margins.append((d["correct"] - d["mismatched"], d["correct"] - d["empty"]))
    mm = [a for a, _ in margins]
    me = [b for _, b in margins]
    n_pos = sum(1 for a, b in margins if a > 0 and b > 0)
    print(f"candidates with correct>mismatched and correct>empty: {n_pos}/{len(margins)}")
    print(f"margin correct-mismatched: mean={statistics.mean(mm):.3f} min={min(mm):.3f} max={max(mm):.3f}")
    print(f"margin correct-empty:      mean={statistics.mean(me):.3f} min={min(me):.3f} max={max(me):.3f}")

    sel = json.loads((run / "coverage" / "selection.json").read_text(encoding="utf-8"))
    print(f"selection: rule={sel['rule']!r} seed={sel['seed']} n={len(sel['candidates'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
