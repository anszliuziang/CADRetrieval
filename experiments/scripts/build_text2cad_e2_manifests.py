#!/usr/bin/env python
"""Build the three E2 background manifests + the committed expected-config.

torch-free, zero forward. Sources (all SHA-pinned):
  --original-manifest-dir  the frozen pair_manifest_v4 dir (texts/cad_store/
                           cad_length_index/pair_manifest.json)
  --out-root               runtime dir for the derived manifests (checkout
                           outside on servers); refuses to overwrite
  --config-out             repo path of the committed expected-config
                           (experiments/configs/text2cad_e2_background_expected_20261112.json)

Derived manifests (schema text2cad-pair-manifest-v1, same as v4):
  b0/     texts = [""]                    cad_store/lengths = byte-copies of v4 (8,046)
  b1/     texts = ["a CAD model."]        same as b0
  replay/ texts = 8 pinned original texts cad_store/lengths = FIRST 1,024 rows of v4
          (a prefix, so replay cad_index equals the G4 cad_index)

The expected-config pins every artifact sha256, the original cad_store sha
(the b0/b1 candidate-store equality reference), the replay text mapping, the
allowed authorization ref, and the cad_seq.zip sha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_LENGTH_FILE,
    CAD_SEQ_ZIP_SHA256,
    CAD_STORE_FILE,
    PAIR_MANIFEST_FILE,
    SCHEMA_VERSION,
    TEXTS_FILE,
    load_texts,
    read_jsonl_strict,
    sha256_file,
    sha256_text,
)

B0_TEXT = ""
B1_TEXT = "a CAD model."
REPLAY_ORIGINAL_TEXT_INDICES = [0, 1, 2, 3, 1000, 2000, 3000, 3643]
REPLAY_N_CADS = 1024
AUTHORIZATION_REF = "origin/glm-night-20261112"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--config-out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.original_manifest_dir)
    out_root = Path(args.out_root)
    if out_root.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing {out_root} (use --force only after review)")
    if args.force:
        import shutil
        shutil.rmtree(out_root)

    original_manifest = json.loads((src / PAIR_MANIFEST_FILE).read_text(encoding="utf-8"))
    if original_manifest.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("original manifest schema mismatch")
    original_texts = load_texts(src)
    if len(original_texts) != 3644:
        raise SystemExit(f"original texts {len(original_texts)} != 3644")
    store_rows = read_jsonl_strict(src / CAD_STORE_FILE, frozenset({"cad_index", "member"}), CAD_STORE_FILE)
    if len(store_rows) != 8046:
        raise SystemExit(f"original cad store {len(store_rows)} != 8046")
    length_rows = read_jsonl_strict(src / CAD_LENGTH_FILE, frozenset({"cad_index", "n_real_tokens", "n_valid_x", "n_valid_y"}), CAD_LENGTH_FILE)
    if len(length_rows) != 8046:
        raise SystemExit(f"original cad length index {len(length_rows)} != 8046")

    out_root.mkdir(parents=True, exist_ok=False)
    fingerprints: dict[str, dict] = {}

    def emit(which: str, text_rows: list[dict], store_bytes: bytes, length_bytes: bytes,
             n_unique: int, n_gallery: int) -> None:
        d = out_root / which
        d.mkdir(parents=True, exist_ok=False)
        (d / TEXTS_FILE).write_bytes(b"".join(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in text_rows))
        (d / CAD_STORE_FILE).write_bytes(store_bytes)
        (d / CAD_LENGTH_FILE).write_bytes(length_bytes)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "e2_background": which,
            "derived_from_original": {
                "pair_manifest_sha256": sha256_file(src / PAIR_MANIFEST_FILE),
                "cad_store_sha256": sha256_file(src / CAD_STORE_FILE),
            },
            "artifacts": {
                CAD_LENGTH_FILE: {"file": CAD_LENGTH_FILE, "sha256": sha256_file(d / CAD_LENGTH_FILE)},
                CAD_STORE_FILE: {"file": CAD_STORE_FILE, "sha256": sha256_file(d / CAD_STORE_FILE)},
                TEXTS_FILE: {"file": TEXTS_FILE, "sha256": sha256_file(d / TEXTS_FILE)},
            },
            "cad_seq_zip": original_manifest["cad_seq_zip"],
            "counts": {
                "n_unique_text": n_unique,
                "n_gallery": n_gallery,
                "n_pairs": n_unique * n_gallery,
                "n_query": 0,
            },
            "expectations_checked": [],
            "inputs": {"note": "E2 background derivation; see e2 expected-config for pins"},
        }
        (d / PAIR_MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        fingerprints[which] = {
            "texts_sha256": sha256_file(d / TEXTS_FILE),
            "cad_store_sha256": sha256_file(d / CAD_STORE_FILE),
            "cad_length_index_sha256": sha256_file(d / CAD_LENGTH_FILE),
            "pair_manifest_sha256": sha256_file(d / PAIR_MANIFEST_FILE),
        }
        print(f"[e2-build] {which}: {n_unique}x{n_gallery} -> {d}")

    full_store = (src / CAD_STORE_FILE).read_bytes()
    full_length = (src / CAD_LENGTH_FILE).read_bytes()

    def text_row(idx: int, text: str) -> dict:
        return {"text_index": idx, "text": text, "text_sha256": sha256_text(text), "n_chars": len(text)}

    emit("b0", [text_row(0, B0_TEXT)], full_store, full_length, 1, 8046)
    emit("b1", [text_row(0, B1_TEXT)], full_store, full_length, 1, 8046)

    replay_rows = []
    replay_meta = []
    for new_idx, orig_idx in enumerate(REPLAY_ORIGINAL_TEXT_INDICES):
        t = original_texts[orig_idx]
        replay_rows.append(text_row(new_idx, t))
        replay_meta.append({"original_text_index": orig_idx, "text": t, "text_sha256": sha256_text(t)})
    prefix_store = b"".join(
        json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in store_rows[:REPLAY_N_CADS]
    )
    prefix_length = b"".join(
        json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in length_rows[:REPLAY_N_CADS]
    )
    emit("replay", replay_rows, prefix_store, prefix_length, len(REPLAY_ORIGINAL_TEXT_INDICES), REPLAY_N_CADS)

    config = {
        "schema": "text2cad-e2-background-expected-v1",
        "authorization_ref": AUTHORIZATION_REF,
        "plan": "experiments/plans/IDEA01_E2_EMPTY_TEXT_BACKGROUND_2026-11-12.md",
        "backgrounds": fingerprints,
        "original_cad_store_sha256": sha256_file(src / CAD_STORE_FILE),
        "replay_texts": replay_meta,
        "replay_n_cads": REPLAY_N_CADS,
        "cad_seq_zip_sha256": CAD_SEQ_ZIP_SHA256,
        "budget_note": "caps live in ACTIVE_TASK e2_* fields at launch; totals >= 24,285 pairs / <= 900 card-seconds / deadline e2_deadline_utc",
    }
    Path(args.config_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config_out).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[e2-build] wrote committed expected-config {args.config_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
