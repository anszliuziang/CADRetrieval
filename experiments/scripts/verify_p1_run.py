#!/usr/bin/env python
"""Post-run verification for a P1 matrix run directory (read-only).

Hardened per Codex review 2026-09-25 (R2): every discrepancy is a counted
failure and yields a non-zero exit. Recomputes per-pair sums from the
per-token evidence files and compares against pairs.jsonl WITH an explicit
tolerance; checks target-set contiguity (positions == prefix_len .. prefix_len
+n_target-1), shift (pred_row == position-1), token_id == input_ids[position],
last id == EOS, mean == sum/n, eos_logprob == last token logprob, and exact
coverage of the fixed manifest pair ids (duplicate/missing/extra all fail).

Usage:
  python experiments/scripts/verify_p1_run.py --run-dir <run_dir> --selection <p1_selection.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.p1_verify import verify_run  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 2


def load_tokens(run: Path, pair_id: str) -> list[dict] | None:
    path = run / "tokens" / f"{pair_id}.jsonl"
    if not path.exists():
        return None
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="dir containing pairs.jsonl and tokens/")
    ap.add_argument("--selection", required=True, help="p1_selection.json (fixed manifest)")
    args = ap.parse_args()
    run = Path(args.run_dir)

    records = [
        json.loads(ln)
        for ln in (run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    expected = [p["pair_id"] for p in sel["pairs"]]
    tokens_by_pair = {pid: load_tokens(run, pid) for pid in {r["pair_id"] for r in records} | set(expected)}

    report = verify_run(records, tokens_by_pair, expected)
    print(
        f"records={report['n_records']} expected={report['n_expected']} "
        f"failures={report['n_failures']} (eos_id={report['eos_id']}, rel_tol={report['rel_tol']})"
    )
    for f in report["failures"][:20]:
        print(f"FAIL: {f}")
    if report["ok"]:
        print("verify_p1_run: OK — per-token sums, boundaries, shift, token ids and coverage all consistent")
        return EXIT_OK
    print(f"verify_p1_run: {report['n_failures']} failure(s); refusing to treat raw scores as valid")
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
