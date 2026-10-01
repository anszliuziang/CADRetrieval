"""Build the fixed Text2CAD v1.0 pair space from the frozen M0 manifests (B2).

CPU-only, zero model forward. Reads M0 ``query.jsonl`` + ``gallery.jsonl``
(NEVER qrels.jsonl), validates every field, builds the canonical unique-text
index (sorted by (text_sha256, text)) and the CAD store index (verified
ascending cad_id), cross-checks the pinned cad_seq.zip, optionally runs the
zero-forward per-candidate structural scan (weights_only loads + audited
validate_cad_vec), and writes the deterministic artifacts to --out-dir:

  texts.jsonl / cad_store.jsonl       worker-facing (no query->CAD link;
                                        CAD zip member path contains UID)
  evaluation_map.json                 evaluation-only linkage (never the runner)
  pair_manifest.json                  small binding manifest with artifact SHAs
  cad_length_index.jsonl              only with --scan-cad-lengths

Example (local audit box):
    python experiments/scripts/build_text2cad_pair_manifest.py \
        --query-jsonl .cache/m0_real_run_v3/query.jsonl \
        --gallery-jsonl .cache/m0_real_run_v3/gallery.jsonl \
        --cad-seq-zip .cache/text2cad_official/cad_seq.zip \
        --out-dir ../CADRetrievalIdea1_runtime/pair_manifest_v2 \
        --repo-root . --scan-cad-lengths
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_pair_index import (  # noqa: E402
    PairIndexError,
    build_pair_manifest,
    scan_cad_lengths,
    verify_zip_members,
    write_pair_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-jsonl", required=True, type=Path)
    parser.add_argument("--gallery-jsonl", required=True, type=Path)
    parser.add_argument("--cad-seq-zip", required=True, type=Path,
                        help="pinned official cad_seq.zip (SHA-verified)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="output directory (must be OUTSIDE the Git checkout)")
    parser.add_argument("--repo-root", type=Path, default=Path("."),
                        help="Git checkout root; outputs inside it are refused")
    parser.add_argument("--scan-cad-lengths", action="store_true",
                        help="zero-forward per-candidate structural scan (torch, CPU)")
    args = parser.parse_args()
    try:
        built = build_pair_manifest(args.query_jsonl, args.gallery_jsonl)
        members = [e.member for e in built["cad_entries"]]
        zip_report = verify_zip_members(args.cad_seq_zip, members)
        length_records = None
        if args.scan_cad_lengths:
            length_records = scan_cad_lengths(args.cad_seq_zip, members)
        result = write_pair_manifest(
            args.out_dir,
            built,
            repo_root=args.repo_root,
            zip_report=zip_report,
            cad_length_records=length_records,
        )
        print(json.dumps(result["summary"], sort_keys=True))
        return 0
    except PairIndexError as exc:
        print(json.dumps({"status": "failed", "reason": exc.reason, "detail": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
