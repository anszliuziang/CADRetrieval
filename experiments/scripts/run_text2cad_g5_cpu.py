"""Single authorized G5 CPU evaluation; outer operator captures actual process exit."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from text2cad_g5_cpu import run_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--runtime-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--approved-sha', required=True)
    p.add_argument('--authorization-ref', required=True)
    a = p.parse_args()
    try:
        run_evaluation(Path(__file__).parents[2], a.data_root, a.runtime_root, a.output,
                       a.approved_sha, a.authorization_ref)
    except Exception as exc:
        print(json.dumps(dict(status='failed', error_type=type(exc).__name__, error=str(exc))), flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
