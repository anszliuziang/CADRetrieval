"""CLIP prep probe: real minimal_json -> OCC geometry -> fixed-view render ->
CLIP image encoding on a length-stratified UID sample (plan
IDEA01_G5_EXTERNAL_BASELINES_2026-11-15 section 3; reviewer-sanctioned batch
to measure success rate and timing BEFORE any full-library run).

Run in the isolated `cadclip` env (pythonocc-core 7.7.2 from conda-forge —
CadSeqProc binds OCC.Core.*, not OCP — plus trimesh/matplotlib/transformers/
CPU torch). Zero NPU. Outputs per-UID per-stage timings and failure reasons,
the rendered PNGs, and a summary JSON with success rates.

  D:/miniconda/envs/cadclip/python.exe experiments/scripts/probe_clip_render_20261115.py \
      --out-dir D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/clip_probe_20261115 \
      [--per-quartile 8] [--limit N] [--skip-clip] [--workers 8]
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

REPO_DEFAULT = Path(__file__).resolve().parents[2]
CADSEQPROC = REPO_DEFAULT / ".cache/Text2CAD-official/CadSeqProc"
MINIMAL_ZIP = REPO_DEFAULT / ".cache/text2cad_official/text2cad_v1.0/misc/minimal_json/minimal_json_0000_0099.zip"
LENGTHS = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/cad_length_index.jsonl")
GALLERY = REPO_DEFAULT / ".cache/m0_real_run_v3/gallery.jsonl"

# probe view spec (frozen for this batch; production spec decided at the
# method round): single canonical isometric view, white background
VIEW = dict(azim=45, elev=30)


def pick_uids(per_quartile: int):
    """First N cad_ids per gallery-length quartile under the G5 boundary rule
    (lower bin includes the exact cutpoint: searchsorted(cuts, L, 'left'))."""
    lengths = [json.loads(l) for l in open(LENGTHS, encoding="utf-8")]
    gallery = [json.loads(l) for l in open(GALLERY, encoding="utf-8")]
    assert len(lengths) == len(gallery) == 8046
    L = np.array([r["n_valid_x"] + r["n_valid_y"] for r in lengths], dtype=np.int64)
    cuts = np.quantile(L, [.25, .5, .75])
    bins = np.searchsorted(cuts, L, side="left")
    picks = []
    for b in range(4):
        idx = np.flatnonzero(bins == b)[:per_quartile]
        for i in idx:
            picks.append(dict(cad_index=int(i), cad_id=gallery[i]["cad_id"],
                              n_valid_total=int(L[i]), quartile=int(b + 1),
                              minimal_json_member=gallery[i]["minimal_json_member"]))
    return picks, cuts.tolist()


def render_mesh_png(mesh, out_png: Path) -> None:
    """matplotlib-based offscreen render (no GL): fixed canonical isometric
    view of the normalized mesh, white background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(v[f], facecolor="#9db8d9", edgecolor="none")
    ax.add_collection3d(pc)
    center = v.mean(axis=0)
    radius = float(np.max(np.linalg.norm(v - center, axis=1))) or 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=VIEW["elev"], azim=VIEW["azim"])
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-quartile", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--text", default="a small cad model",
                    help="sample L0-style text for the text-tower timing demo")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "renders").mkdir()
    (out / "steps").mkdir()

    sys.path.insert(0, str(CADSEQPROC.parent))
    from CadSeqProc.cad_sequence import CADSequence  # noqa: E402

    picks, cuts = pick_uids(a.per_quartile)
    if a.limit:
        picks = picks[: a.limit]
    z = zipfile.ZipFile(MINIMAL_ZIP)
    print(json.dumps(dict(phase="plan", n_uids=len(picks), quartile_cuts=cuts,
                          view=VIEW)), flush=True)

    clip_model = clip_proc = None
    clip_load_s = None
    if not a.skip_clip:
        from transformers import CLIPImageProcessor, CLIPModel
        t0 = time.time()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_load_s = round(time.time() - t0, 1)
        print(json.dumps(dict(phase="clip_loaded", seconds=clip_load_s)), flush=True)

    results = []
    t_all = time.time()
    for p in picks:
        rec = dict(**p, stages={}, ok=False, error=None)
        try:
            t0 = time.time()
            data = json.loads(z.read(p["minimal_json_member"]))
            rec["stages"]["json_parse_s"] = round(time.time() - t0, 3)

            t0 = time.time()
            seq = CADSequence.from_minimal_json(data)
            # NOTE: seq.normalize() needs properties.bounding_box which the
            # dataset's minimal_json does NOT carry (integration gap logged);
            # the renderer below normalizes the mesh geometrically instead.
            rec["stages"]["parse_normalize_s"] = round(time.time() - t0, 3)
            rec["normalize_skipped"] = True

            t0 = time.time()
            seq.create_cad_model()
            rec["stages"]["occ_brep_s"] = round(time.time() - t0, 3)

            t0 = time.time()
            step_path = out / "steps" / (p["cad_id"].replace("/", "__") + ".step")
            seq.save_stp(filename=step_path.stem, output_folder=str(step_path.parent),
                         type="step")
            rec["stages"]["step_export_s"] = round(time.time() - t0, 3)
            rec["step_bytes"] = step_path.stat().st_size if step_path.exists() else 0

            t0 = time.time()
            seq.create_mesh()
            rec["stages"]["mesh_s"] = round(time.time() - t0, 3)

            t0 = time.time()
            png = out / "renders" / (p["cad_id"].replace("/", "__") + ".png")
            render_mesh_png(seq.mesh, png)
            rec["stages"]["render_s"] = round(time.time() - t0, 3)
            rec["render_png"] = str(png)

            if clip_model is not None:
                from PIL import Image
                t0 = time.time()
                img = Image.open(png).convert("RGB")
                inputs = clip_proc(images=[img], return_tensors="pt")
                import torch
                with torch.no_grad():
                    feat_raw = clip_model.get_image_features(**inputs)
                # this transformers version returns the vision ModelOutput with
                # pooler_output REPLACED by the projected image embeds; older
                # versions returned the raw tensor
                feat = feat_raw if torch.is_tensor(feat_raw) else feat_raw.pooler_output
                rec["stages"]["clip_image_s"] = round(time.time() - t0, 3)
                rec["clip_dim"] = int(feat.shape[-1])
                rec["clip_vec_first_component"] = float(feat[0][0])

            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc(limit=4)
        results.append(rec)
        print(json.dumps(dict(phase="uid", cad_id=p["cad_id"], ok=rec["ok"],
                              stages=rec["stages"], error=rec["error"])), flush=True)

    if clip_model is not None and a.text:
        from transformers import CLIPTokenizerFast
        import torch
        tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        t0 = time.time()
        with torch.no_grad():
            tfeat = clip_model.get_text_features(**tok([a.text], return_tensors="pt",
                                                       padding=True))
        text_encode_s = round(time.time() - t0, 3)
    else:
        text_encode_s = None

    n_ok = sum(r["ok"] for r in results)
    stage_keys = ("json_parse_s", "parse_normalize_s", "occ_brep_s", "step_export_s",
                  "mesh_s", "render_s", "clip_image_s")
    stage_stats = {}
    for k in stage_keys:
        vals = [r["stages"][k] for r in results if r["ok"] and k in r["stages"]]
        stage_stats[k] = dict(n=len(vals),
                              mean_s=round(float(np.mean(vals)), 3) if vals else None,
                              max_s=round(float(np.max(vals)), 3) if vals else None)
    fails = [dict(cad_id=r["cad_id"], error=r["error"]) for r in results if not r["ok"]]
    summary = dict(schema="clip-prep-probe-v1",
                   n_uids=len(results), n_ok=n_ok,
                   success_rate=round(n_ok / len(results), 4) if results else None,
                   quartile_cuts=cuts, view=VIEW,
                   clip_load_s=clip_load_s, text_encode_s=text_encode_s,
                   stage_seconds=stage_stats,
                   failures=fails,
                   total_wall_s=round(time.time() - t_all, 1),
                   full_library_estimate_note="stage means x 8046 for planning only")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    (out / "per_uid.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="complete", output=str(out), summary=summary)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
