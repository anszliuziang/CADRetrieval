"""Audit the frozen in-domain pilot's release representations and cross-split duplicates."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.src.idea01.in_domain_pilot_representations import run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifests", "official-zip", "minimal-zip", "ct-zip", "out"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--text2cad-root", type=Path, help="Pinned official Text2CAD source checkout with CadSeqProc")
    parser.add_argument("--deepcad-root", type=Path, help="Official DeepCAD source checkout with OCC available")
    parser.add_argument("--replay-count", type=int, default=16, help="Number of train/validation UIDs for bounded B-Rep replay")
    args = parser.parse_args()
    if args.replay_count < 0:
        parser.error("--replay-count must be nonnegative")
    report = run_audit(args.manifests, args.official_zip, args.minimal_zip, args.ct_zip, args.out,
                       text2cad_root=args.text2cad_root, deepcad_root=args.deepcad_root, replay_count=args.replay_count)
    print(json.dumps({"output": str(args.out), "coverage_unique_uids": report["coverage_unique_uids"],
                      "converter_provenance": report["converter_provenance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
