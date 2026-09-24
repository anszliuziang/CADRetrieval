#!/usr/bin/env python
"""Write a NON-self-referencing sha256 manifest for a run dir (Codex review
2026-09-25 item 4).

The manifest lists every regular file under --root except the manifest file
itself (any previously existing manifests are included as ordinary content).
Format is `sha256sum -c` compatible: "<hex>  ./<relative/path>" with posix
separators and LF line endings, so it verifies identically on Linux and with
Git-Bash coreutils.

Usage:
  python experiments/scripts/make_manifest.py --root <run-dir> [--out MANIFEST_SHA256_v2.txt]
  (cd <run-dir> && sha256sum -c MANIFEST_SHA256_v2.txt)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest_lines(root: Path, out_name: str) -> list[str]:
    root = root.resolve()
    out_abs = (root / out_name).resolve()
    lines = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == out_abs:
            continue  # never self-reference (review item 4)
        rel = p.relative_to(root).as_posix()
        lines.append(f"{sha256_file(p)}  ./{rel}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="MANIFEST_SHA256_v2.txt", help="manifest file name inside --root")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"[manifest] not a directory: {root}", file=sys.stderr)
        return 1
    lines = build_manifest_lines(root, args.out)
    out_path = root / args.out
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[manifest] wrote {out_path} with {len(lines)} entries (self excluded)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
