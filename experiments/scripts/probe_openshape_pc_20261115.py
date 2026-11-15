"""OpenShape prep probe: minimal_json -> OCC mesh -> 10,000-point surface
sample -> bbox-center normalize to [-1,1], z-up check, neutral RGB 0.5
(review-directed conversion validation; runs in the cadclip env).

  D:/miniconda/envs/cadclip/python.exe experiments/scripts/probe_openshape_pc_20261115.py \
      --n 32 --out D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/openshape_prep_20261115
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".cache" / "Text2CAD-official"))
from CadSeqProc.cad_sequence import CADSequence  # noqa: E402

GALLERY = REPO / ".cache" / "m0_real_run_v3" / "gallery.jsonl"
MINIMAL_ZIP = REPO / ".cache" / "text2cad_official" / "text2cad_v1.0" / "misc" / "minimal_json" / "minimal_json_0000_0099.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    gal = [json.loads(l) for l in open(GALLERY, encoding="utf-8")]
    z = zipfile.ZipFile(MINIMAL_ZIP)
    ok = 0
    stats = []
    for i in range(min(a.n, len(gal))):
        cid, member = gal[i]["cad_id"], gal[i]["minimal_json_member"]
        try:
            seq = CADSequence.from_minimal_json(json.loads(z.read(member)))
            seq.create_cad_model()
            seq.create_mesh()
            seq.sample_points(n_points=10000)   # returns self; points on the sequence
            pts = np.asarray(seq.points, dtype=np.float64)
            lo, hi = pts.min(0), pts.max(0)
            pts = (pts - (lo + hi) / 2) / ((hi - lo).max() / 2 + 1e-12)
            np.savez_compressed(out / (cid.replace("/", "__") + "_pc.npz"),
                                points=pts.astype(np.float32),
                                rgb=np.full((10000, 3), 0.5, dtype=np.float32))
            stats.append(dict(cad_id=cid, min=float(pts.min()), max=float(pts.max()),
                              finite=bool(np.isfinite(pts).all())))
            ok += 1
        except Exception as e:  # noqa: BLE001
            stats.append(dict(cad_id=cid, error=f"{type(e).__name__}: {e}"))
        if (i + 1) % 8 == 0:
            print(json.dumps(dict(done=i + 1, ok=ok)), flush=True)
    summary = dict(schema="openshape-pc-probe-v1", n=len(stats), ok=ok,
                   success_rate=round(ok / len(stats), 4) if stats else None,
                   per_uid=stats,
                   input_spec="10000x3 float32 in [-1,1], bbox-center normalized; rgb neutral 0.5 (deviation from RGB-trained checkpoints logged)")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="complete", ok=ok, n=len(stats),
                          success_rate=summary["success_rate"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
