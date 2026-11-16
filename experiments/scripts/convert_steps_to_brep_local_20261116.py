"""Local batch converter: mmcad_steps_20261115/*.step -> brep npz files via the
official exanos/MMCAD step_to_brep.py, using the local cadclip env (pythonocc
7.7.2). Process pool over the covered STEPs; resumable (existing npz skipped);
failures recorded. Output feeds the share4 MM-CAD run's brep_cache.

  D:/miniconda/envs/cadclip/python.exe experiments/scripts/convert_steps_to_brep_local_20261116.py \
      --steps-dir D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/mmcad_steps_20261115 \
      --out-dir D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/mmcad_brep_npz_20261116 \
      --workers 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MMCAD = REPO / ".cache/MMCAD-official"


def convert_one(args):
    step, npz = args
    if npz.exists() and npz.stat().st_size > 0:
        return step.stem, "cached"
    r = subprocess.run([sys.executable, str(MMCAD / "scripts" / "step_to_brep.py"),
                        str(step), "-o", str(npz)], capture_output=True, text=True)
    if r.returncode == 0 and npz.exists() and npz.stat().st_size > 0:
        return step.stem, "ok"
    return step.stem, f"fail: {r.stderr.strip()[-160:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    steps = sorted(Path(a.steps_dir).glob("*.step"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, out / (s.stem + ".npz")) for s in steps]
    print(f"converting {len(jobs)} STEPs with {a.workers} workers", flush=True)
    t0 = time.time()
    fails = {}
    n_ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        # map() returns an ordered result iterator (as_completed needs futures)
        for k, (stem, status) in enumerate(ex.map(convert_one, jobs)):
            if status == "fail" or status.startswith("fail"):
                fails[stem] = status
            else:
                n_ok += 1
            if (k + 1) % 400 == 0 or k + 1 == len(jobs):
                print(f"{k + 1}/{len(jobs)} ok={n_ok} fail={len(fails)} "
                      f"seconds={round(time.time() - t0, 1)}", flush=True)
    import json
    (out / "conversion_summary.json").write_text(
        json.dumps(dict(expected=len(jobs), ok=n_ok, failures=fails,
                        seconds=round(time.time() - t0, 1)), indent=1) + "\n",
        encoding="utf-8", newline="\n")
    print(f"DONE ok={n_ok}/{len(jobs)} seconds={round(time.time() - t0, 1)}", flush=True)
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
