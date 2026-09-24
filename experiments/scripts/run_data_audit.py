#!/usr/bin/env python
"""Run the JSONL <-> CSV <-> ZIP source/UID/leakage audit.

CPU-only. Writes to --out-dir:
  data_audit.json          per-split counts + cross-split overlaps
  uid_manifest_<split>.jsonl  row_index, uid (or null), map_status per row

Example (remote):
  python experiments/scripts/run_data_audit.py \
    --data-dir /home/ma-user/work/CADRetrieval_runtime/datasets/IEEE-T-ASE/4091ce30ad85b774a9231484f9b40372dc0451d3 \
    --out-dir  <run_dir>/audit
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.data_audit import (  # noqa: E402
    audit_split,
    audit_to_dict,
    build_intermediate_index,
    cross_split_overlap,
    read_jsonl,
)

SPLIT_FILES = {
    "train": "data/data_train.jsonl",
    "val": "data/data_val.jsonl",
    "test": "data/data_test.jsonl",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "text2cad_v1.1.csv"
    zip_path = data_dir / "CadQuery.zip"

    print(f"[audit] indexing CSV {csv_path} ...", flush=True)
    index, n_csv = build_intermediate_index(csv_path)
    n_ambiguous_keys = sum(1 for v in index.values() if len(v) > 1)
    print(f"[audit] csv rows={n_csv} unique intermediate keys={len(index)} ambiguous keys={n_ambiguous_keys}", flush=True)

    audits = {}
    with zipfile.ZipFile(zip_path) as zf:
        for split in args.splits.split(","):
            path = data_dir / SPLIT_FILES[split]
            rows = read_jsonl(path)
            print(f"[audit] {split}: {len(rows)} rows; auditing against CSV+ZIP ...", flush=True)
            a = audit_split(split, rows, index, zf)
            audits[split] = a
            with open(out_dir / f"uid_manifest_{split}.jsonl", "w", encoding="utf-8") as f:
                for i, uid in enumerate(a.row_uids):
                    status = "unique" if uid is not None else "unmapped"
                    f.write(json.dumps({"row_index": i, "uid": uid, "map_status": status}) + "\n")
            print(f"[audit] {split}: {json.dumps(audit_to_dict(a))}", flush=True)

    result = {
        "csv_path": str(csv_path),
        "zip_path": str(zip_path),
        "n_csv_rows": n_csv,
        "n_csv_unique_intermediate_keys": len(index),
        "n_csv_ambiguous_intermediate_keys": n_ambiguous_keys,
        "splits": {s: audit_to_dict(a) for s, a in audits.items()},
        "cross_split": cross_split_overlap(audits),
        "notes": [
            "join key: JSONL input == CSV intermediate (verbatim)",
            "UID identifies the source Text2CAD sample, NOT a design family; family-disjoint cannot be claimed",
        ],
    }
    with open(out_dir / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[audit] wrote {out_dir / 'data_audit.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
