"""Evaluate frozen E1 score arrays locally without importing model code."""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_e1_eval import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="frozen evaluation manifests")
    parser.add_argument("--scores", required=True, type=Path, help="directory of E1 score archives")
    parser.add_argument("--output", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--source-only", action="store_true", help="evaluate B encoding without opening C files")
    args = parser.parse_args(argv)
    result = evaluate(args.input, args.scores, args.output, source_only=args.source_only)
    print(f"E1 {result['phase']}, source encoding {result['encoding_check']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
