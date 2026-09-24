#!/usr/bin/env python
"""Token coverage audit + deterministic P0 selection (CPU-only, tokenizer only).

Uses the checkpoint tokenizer with use_fast=False, add_special_tokens=False.
Writes to --out-dir:
  coverage_<split>.jsonl    per-row measurements
  coverage_summary.json     per-split aggregate stats
  selection.json            P0 candidate list + pair plan (validation only)

Requires the UID manifest produced by run_data_audit.py (--audit-dir).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.coverage import measure_row, summarize  # noqa: E402
from idea01.prompts import build_prefix  # noqa: E402
from idea01.selection import build_pair_plan, select_candidates  # noqa: E402

SPLIT_FILES = {
    "train": "data/data_train.jsonl",
    "val": "data/data_val.jsonl",
    "test": "data/data_test.jsonl",
}


def load_uid_manifest(audit_dir: Path, split: str) -> list[str | None]:
    path = audit_dir / f"uid_manifest_{split}.jsonl"
    uids: list[str | None] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            uids.append(json.loads(line)["uid"])
    return uids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="checkpoint dir (tokenizer files)")
    ap.add_argument("--audit-dir", required=True, help="dir with uid_manifest_<split>.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-candidates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--selection-split", default="val")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # deferred: heavy import

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    eos_id = tok.eos_token_id
    print(f"[coverage] tokenizer loaded in {time.time()-t0:.1f}s, class={type(tok).__name__}, eos_id={eos_id}", flush=True)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    summary: dict[str, dict] = {}
    measured: dict[str, list] = {}
    for split in args.splits.split(","):
        path = data_dir / SPLIT_FILES[split]
        rows_cov = []
        t1 = time.time()
        with open(path, "r", encoding="utf-8") as f, open(
            out_dir / f"coverage_{split}.jsonl", "w", encoding="utf-8"
        ) as out:
            for i, line in enumerate(f):
                r = json.loads(line)
                cov = measure_row(i, r["input"], r["output"], encode, eos_id, args.max_len, build_prefix)
                rows_cov.append(cov)
                out.write(json.dumps(dataclasses.asdict(cov)) + "\n")
                if (i + 1) % 1000 == 0:
                    print(f"[coverage] {split}: {i+1} rows, {time.time()-t1:.1f}s", flush=True)
        summary[split] = summarize(rows_cov, args.max_len)
        measured[split] = rows_cov
        print(f"[coverage] {split} done: {json.dumps(summary[split])}", flush=True)

    with open(out_dir / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[coverage] wrote {out_dir / 'coverage_summary.json'}", flush=True)

    # ---- P0 selection on the selection split ----
    ss = args.selection_split
    uids = load_uid_manifest(Path(args.audit_dir), ss)
    rows_cov = measured[ss]
    if len(uids) != len(rows_cov):
        raise RuntimeError(f"uid manifest ({len(uids)}) != rows ({len(rows_cov)}) for split {ss}")
    cands = select_candidates(
        row_uids=uids,
        fits=[c.fits_max_len for c in rows_cov],
        boundary_oks=[c.boundary_ok for c in rows_cov],
        n_full_tokens=[c.n_full_tokens for c in rows_cov],
        n_candidates=args.n_candidates,
        seed=args.seed,
    )
    pairs = build_pair_plan(cands)
    selection = {
        "rule": "eligible=(unique uid, full<=max_len, boundary_ok); sort uid asc; first n",
        "selection_split": ss,
        "max_len": args.max_len,
        "n_candidates": len(cands),
        "seed": args.seed,
        "candidates": [dataclasses.asdict(c) for c in cands],
        "pairs": pairs,
    }
    with open(out_dir / "selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"[coverage] selection: {len(cands)} candidates, {len(pairs)} pairs -> {out_dir / 'selection.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
