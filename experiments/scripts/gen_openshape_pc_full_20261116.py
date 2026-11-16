"""Full-library point-cloud generation for OpenShape (covered gallery):
minimal_json -> CadSeqProc OCC B-Rep -> mesh -> 10,000-point surface sample
-> bbox-center normalize [-1,1] + neutral RGB 0.5. Resumable (existing npz
skipped). Runs in the local cadclip env (pythonocc).

  D:/miniconda/envs/cadclip/python.exe experiments/scripts/gen_openshape_pc_full_20261116.py \
      --out D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/openshape_pc_full_20261116
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".cache" / "Text2CAD-official"))
from CadSeqProc.cad_sequence import CADSequence  # noqa: E402

GALLERY = REPO / ".cache" / "m0_real_run_v3" / "gallery.jsonl"
MINIMAL_ZIP = REPO / ".cache" / "text2cad_official" / "text2cad_v1.0" / "misc" / "minimal_json" / "minimal_json_0000_0099.zip"
COVERED_FROM = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_clip_g5_20261115/coverage.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gal = [json.loads(l) for l in open(GALLERY, encoding="utf-8")]
    if COVERED_FROM.exists():
        missing = set(json.loads(COVERED_FROM.read_text(encoding="utf-8")).get("missing_cad_ids", []))
        gal = [r for r in gal if r["cad_id"] not in missing]
    z = zipfile.ZipFile(MINIMAL_ZIP)
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)
    log(dict(phase="plan", n=len(gal)))

    ok = 0
    fails = {}
    t0 = time.time()
    for r in gal:
        cid, member = r["cad_id"], r["minimal_json_member"]
        dst = out / (cid.replace("/", "__") + "_pc.npz")
        if dst.exists() and dst.stat().st_size > 0:
            ok += 1
            continue
        try:
            seq = CADSequence.from_minimal_json(json.loads(z.read(member)))
            seq.create_cad_model()
            seq.create_mesh()
            seq.sample_points(n_points=10000)
            pts = np.asarray(seq.points, dtype=np.float64)
            lo, hi = pts.min(0), pts.max(0)
            pts = (pts - (lo + hi) / 2) / ((hi - lo).max() / 2 + 1e-12)
            np.savez_compressed(dst, points=pts.astype(np.float32),
                                rgb=np.full((10000, 3), 0.5, dtype=np.float32))
            ok += 1
        except Exception as e:  # noqa: BLE001
            fails[cid] = f"{type(e).__name__}: {str(e)[:100]}"
        if (ok + len(fails)) % 400 == 0 or ok + len(fails) == len(gal):
            log(dict(phase="progress", done=ok + len(fails), total=len(gal),
                     ok=ok, failures=len(fails), seconds=round(time.time() - t0, 1)))
    summary = dict(schema="openshape-pc-full-v1", expected=len(gal), ok=ok,
                   failures=fails, seconds=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
    log(dict(phase="complete", ok=ok, total=len(gal), failures=len(fails)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
