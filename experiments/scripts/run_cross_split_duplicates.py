#!/usr/bin/env python
"""Cross-split exact-duplicate supplement audit (Codex review 2026-09-25 item 3).

CPU-only. The original `audit/data_audit.json` is left untouched; this writes an
independent supplement JSON with pairwise exact-input / exact-output
intersections (sha256 digest sets; any non-zero hit would be re-verified
byte-exact and reported with examples). Also re-reports UID intersections from
the original uid manifests for a single-place summary.

Usage:
  python experiments/scripts/run_cross_split_duplicates.py \
    --data-dir <pinned data dir> --audit-dir <run>/audit --out <run>/review_supplement/cross_split_duplicates.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.data_audit import load_text_digest_sets  # noqa: E402

SPLIT_FILES = {
    "train": "data/data_train.jsonl",
    "val": "data/data_val.jsonl",
    "test": "data/data_test.jsonl",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--audit-dir", required=True, help="original audit dir with uid_manifest_<split>.jsonl (read-only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    digest_sets: dict[str, tuple[set[str], set[str]]] = {}
    uid_sets: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}
    for split, rel in SPLIT_FILES.items():
        t0 = time.time()
        digest_sets[split] = load_text_digest_sets(data_dir / rel)
        manifest = Path(args.audit_dir) / f"uid_manifest_{split}.jsonl"
        uids = set()
        n = 0
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                u = json.loads(line)["uid"]
                n += 1
                if u is not None:
                    uids.add(u)
        uid_sets[split] = uids
        row_counts[split] = n
        print(f"[xsplit] {split}: {n} rows, {len(digest_sets[split][0])} unique inputs, "
              f"{len(digest_sets[split][1])} unique outputs ({time.time()-t0:.1f}s)", flush=True)

    pairs = {}
    splits = sorted(SPLIT_FILES)
    for i, sa in enumerate(splits):
        for sb in splits[i + 1 :]:
            ia = digest_sets[sa][0] & digest_sets[sb][0]
            oa = digest_sets[sa][1] & digest_sets[sb][1]
            pairs[f"{sa}__{sb}"] = {
                "uid_intersection": len(uid_sets[sa] & uid_sets[sb]),
                "input_intersection": len(ia),
                "output_intersection": len(oa),
            }
            if ia or oa:
                pairs[f"{sa}__{sb}"]["WARNING"] = (
                    "non-zero digest intersection; must be re-verified byte-exact before use"
                )

    result = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Codex review 2026-09-25 item 3: pairwise exact-input / exact-output cross-split intersections (original data_audit.json left unchanged)",
        "method": "sha256 hexdigest of utf-8 text per row field; digest-set intersection counts; zero count means no byte-identical shared text up to collision resistance",
        "data_dir": str(data_dir),
        "row_counts": row_counts,
        "unique_counts": {
            s: {"unique_inputs": len(digest_sets[s][0]), "unique_outputs": len(digest_sets[s][1])}
            for s in splits
        },
        "pairs": pairs,
        "note": "exact duplicate check only; near-duplicates / geometric equivalence / design families are NOT measured here",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[xsplit] wrote {out_path}", flush=True)
    for k, v in pairs.items():
        print(f"[xsplit] {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
