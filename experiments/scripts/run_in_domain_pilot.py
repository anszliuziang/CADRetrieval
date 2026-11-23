"""Materialize the frozen in-domain pilot or fit its CPU unordered baseline."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.src.idea01.in_domain_pilot import Assets, SOURCE_LEVELS, build_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True, dest="mode")
    prepare = modes.add_parser("prepare")
    for argument in ("annotations", "split", "official-zip", "minimal-zip", "ct-zip", "old-test-gallery", "out"):
        prepare.add_argument(f"--{argument}", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=12)
    ridge = modes.add_parser("ridge")
    ridge.add_argument("--manifests", type=Path, required=True)
    ridge.add_argument("--source", choices=SOURCE_LEVELS, required=True)
    ridge.add_argument("--subset", choices=("full", "pilot"), required=True)
    ridge.add_argument("--out", type=Path, required=True)
    ridge.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.mode == "prepare":
        assets = Assets(args.annotations, args.split, args.official_zip, args.minimal_zip,
                        args.ct_zip, args.old_test_gallery)
        report = build_manifests(assets, args.out, args.workers)
        shown = {"official": report["official_split_counts"], "sources": report["sources"]}
    else:
        from experiments.src.idea01.in_domain_pilot_ridge import fit_baseline

        report = fit_baseline(args.manifests, args.source, args.subset, args.out, args.workers)
        shown = {key: report[key] for key in ("selected_alpha", "train_pairs", "validation_metrics")}
    print(json.dumps(shown, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
