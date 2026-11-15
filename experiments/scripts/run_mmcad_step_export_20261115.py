"""Export STEP B-Reps for the CLIP-covered gallery (MM-CAD round 1,
reviewer-directed: the formal CLIP run's steps/ dir was empty; the 32 probe
STEPs were not a gallery export).

minimal_json -> CadSeqProc OCC B-Rep -> save_stp, for every UID in the CLIP
run's coverage.json (default 8,032). Skip-existing; one retry per UID;
failures recorded with traces. Same geometry path as the frozen CLIP spec.
Zero NPU; cadclip env.

  D:/miniconda/envs/cadclip/python.exe experiments/scripts/run_mmcad_step_export_20261115.py \
      --clip-run D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_clip_g5_20261115 \
      --out-dir D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/mmcad_steps_20261115
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".cache" / "Text2CAD-official"))
from CadSeqProc.cad_sequence import CADSequence  # noqa: E402

GALLERY = REPO / ".cache" / "m0_real_run_v3" / "gallery.jsonl"
MINIMAL_ZIP = REPO / ".cache" / "text2cad_official" / "text2cad_v1.0" / "misc" / "minimal_json" / "minimal_json_0000_0099.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-run", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    coverage = json.loads((Path(a.clip_run) / "coverage.json").read_text(encoding="utf-8"))
    covered = _covered_ids(coverage)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gal = {r["cad_id"]: r for r in (json.loads(l) for l in open(GALLERY, encoding="utf-8"))}
    z = zipfile.ZipFile(MINIMAL_ZIP)
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)
    log(dict(phase="plan", n_uids=len(covered)))

    failures = {}
    t0 = time.time()
    done = 0
    for cid in covered:
        dst = out / (cid.replace("/", "__") + ".step")
        if dst.exists() and dst.stat().st_size > 0:
            done += 1
            continue
        ok = False
        for attempt in (1, 2):
            try:
                row = gal[cid]
                seq = CADSequence.from_minimal_json(json.loads(z.read(row["minimal_json_member"])))
                seq.create_cad_model()
                seq.save_stp(filename=dst.stem, output_folder=str(out), type="step")
                if not (dst.exists() and dst.stat().st_size > 0):
                    # save_stp may append its own suffix; normalise the name
                    alt = out / (dst.stem + "_final.step")
                    if alt.exists():
                        alt.rename(dst)
                ok = dst.exists() and dst.stat().st_size > 0
                break
            except Exception as e:  # noqa: BLE001
                failures.setdefault(cid, []).append(f"attempt{attempt}: {type(e).__name__}: {e}")
                (out / f"step_failure_{cid.replace('/', '__')}.txt").write_text(
                    traceback.format_exc(), encoding="utf-8")
        done += 1
        if done % 400 == 0 or done == len(covered):
            log(dict(phase="progress", done=done, total=len(covered),
                     seconds=round(time.time() - t0, 1), failures=len(failures)))
    log(dict(phase="done", exported=done - len(failures), failures=len(failures),
             seconds=round(time.time() - t0, 1)))
    (out / "export_summary.json").write_text(json.dumps(dict(
        schema="mmcad-step-export-v1", expected=len(covered),
        exported=done - len(failures), failures={k: v for k, v in failures.items()},
        seconds=round(time.time() - t0, 1)), indent=1) + "\n",
        encoding="utf-8", newline="\n")
    return 0 if not failures else 2


def _covered_ids(coverage: dict):
    """Covered = expected minus missing (coverage.json stores missing only)."""
    missing = set(coverage.get("missing_cad_ids", []))
    gallery = [json.loads(l)["cad_id"] for l in open(GALLERY, encoding="utf-8")]
    return [c for c in gallery if c not in missing]


if __name__ == "__main__":
    raise SystemExit(main())
