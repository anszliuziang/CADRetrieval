"""MM-CAD encode validation (reviewer-directed: verify B-Rep encoding with the
official reference part, then measure CPU batch timing on gallery STEPs).

Pipeline per UID: STEP (already exported for the covered gallery) ->
scripts/step_to_brep.py -> RAW npz -> inference.prepare_brep() with the
released brep_norm_stats_v4.npz -> inference.embed_brep().

Stage A: official brep_reference round-trip - embed the reference part from
our own conversion path and compare against the official released embedding
(cosine must be ~1).
Stage B: batch timing - convert+prepare+embed N gallery STEPs; also text-tower
timing over a sample of the frozen unique texts.

  HF_TOKEN from env (gated dataset exanos/MMCAD); run in cadclip env.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MMCAD = REPO / ".cache" / "MMCAD-official"
STEPS = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/mmcad_steps_20261115")
TEXTS = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/texts.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-batch", type=int, default=16, help="gallery STEPs for timing")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)

    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    sys.path.insert(0, str(MMCAD))
    from huggingface_hub import hf_hub_download, snapshot_download

    ck = hf_hub_download("exanos/MMCAD", "checkpoints/baseline_trimodal_v4.pth",
                         repo_type="dataset")
    stats_p = hf_hub_download("exanos/MMCAD", "checkpoints/brep_norm_stats_v4.npz",
                              repo_type="dataset")
    ref_dir = Path(snapshot_download("exanos/MMCAD", repo_type="dataset",
                                     allow_patterns=["checkpoints/brep_reference/*"]))
    ref_dir = ref_dir / "checkpoints" / "brep_reference"
    ref_files = sorted(p.name for p in ref_dir.rglob("*") if p.is_file())
    log(dict(phase="reference_files", files=ref_files))

    import torch
    import inference as mm  # noqa: E402  (official MMCAD inference module)

    t0 = time.time()
    model = mm.load_joint_model(ck, device="cpu")
    load_s = round(time.time() - t0, 1)
    stats_np = np.load(stats_p)
    ff_mean, ff_std = stats_np["ff_mean"], stats_np["ff_std"]
    log(dict(phase="model_loaded", seconds=load_s,
             stats_keys=list(stats_np.files)))

    # ---- stage A: official reference round-trip ---------------------------
    # brep_reference/*.npz carries raw_* extractor fields, in_* model-ready
    # fields, and the official embedding(768) — run OUR prepare+embed path on
    # the raw fields and compare cosines against both the official final
    # embedding and (sanity) the in_* direct path.
    ref_npzs = sorted(p for p in ref_dir.rglob("*.npz"))
    log(dict(phase="reference_found", files=[p.name for p in ref_npzs]))
    stage_a = dict(passed=True, per_file=[])
    for p in ref_npzs:
        zr = np.load(p)
        raw = {k[len("raw_"):]: zr[k] for k in zr.files if k.startswith("raw_")}
        rec = dict(file=p.name)
        if set(("face_features", "face_centroids", "edge_to_faces", "face_mask")) <= set(raw):
            prepared = mm.prepare_brep(raw, ff_mean, ff_std)
            with torch.no_grad():
                z = mm.embed_brep(model, prepared, dim=768, device="cpu")
            zo = z.numpy().reshape(-1).astype(np.float64)
            rec["our_embed_dim"] = int(zo.shape[0])
            rec["our_embed_norm"] = round(float(np.linalg.norm(zo)), 6)
            for tag, key in (("official_embedding", "embedding"), ("official_embedding_raw", "embedding_raw")):
                if key in zr.files:
                    e = zr[key].reshape(-1).astype(np.float64)
                    rec[f"cosine_{tag}"] = round(float(zo @ e / (np.linalg.norm(zo) * np.linalg.norm(e) + 1e-12)), 6)
        # sanity: in_* fields are the model-ready inputs — embed directly
        if all(k in zr.files for k in ("in_face_features", "in_face_centroids",
                                       "in_edge_to_faces", "in_face_mask", "in_face_normals")):
            direct = {
                "face_features": torch.as_tensor(zr["in_face_features"]).unsqueeze(0),
                "face_centroids": torch.as_tensor(zr["in_face_centroids"]).unsqueeze(0),
                "edge_to_faces": torch.as_tensor(zr["in_edge_to_faces"]).unsqueeze(0),
                "face_mask": torch.as_tensor(zr["in_face_mask"]).unsqueeze(0),
                "face_normals": torch.as_tensor(zr["in_face_normals"]).unsqueeze(0),
            }
            with torch.no_grad():
                zd = mm.embed_brep(model, direct, dim=768, device="cpu")
            zdr = zd.numpy().reshape(-1).astype(np.float64)
            if "embedding" in zr.files:
                e = zr["embedding"].reshape(-1).astype(np.float64)
                rec["cosine_direct_in_fields_vs_official"] = round(
                    float(zdr @ e / (np.linalg.norm(zdr) * np.linalg.norm(e) + 1e-12)), 6)
        rec["passed"] = bool(rec.get("cosine_official_embedding", 0) > 0.999)
        stage_a["per_file"].append(rec)
        stage_a["passed"] = stage_a["passed"] and rec["passed"]
    log(dict(phase="stage_a_reference_roundtrip", **stage_a))

    # ---- stage B: gallery batch timing ------------------------------------
    steps = sorted(STEPS.glob("*.step"))[: a.n_batch]
    sys.path.insert(0, str(MMCAD / "scripts"))
    t_conv = t_prep = t_emb = 0.0
    n_ok = 0
    dims = []
    for s in steps:
        npz = out / (s.stem + "_brep.npz")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(MMCAD / "scripts" / "step_to_brep.py"),
                            str(s), "-o", str(npz)], capture_output=True, text=True)
        t_conv += time.time() - t0
        if r.returncode != 0 or not npz.exists():
            log(dict(phase="brep_fail", step=s.name, stderr=r.stderr[-200:]))
            continue
        t0 = time.time()
        prepared = mm.prepare_brep(dict(np.load(npz)), ff_mean, ff_std)
        t_prep += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            z = mm.embed_brep(model, prepared, dim=768, device="cpu")
        t_emb += time.time() - t0
        if bool(np.isfinite(z.numpy()).all()):
            n_ok += 1
            dims.append(int(z.shape[-1]))
    text_rows = [json.loads(l)["text"] for l in open(TEXTS, encoding="utf-8")][:64]
    t0 = time.time()
    with torch.no_grad():
        for t in text_rows[:16]:
            mm.embed_text(model, t, dim=768, device="cpu")
    text_s = round(time.time() - t0, 1)
    summary = dict(schema="mmcad-encode-probe-v1",
                   stage_a=stage_a,
                   batch=dict(n=len(steps), ok=n_ok,
                              convert_s_per_uid=round(t_conv / max(len(steps), 1), 3),
                              prepare_s_per_uid=round(t_prep / max(len(steps), 1), 3),
                              embed_s_per_uid=round(t_emb / max(len(steps), 1), 3),
                              dims=sorted(set(dims))),
                   text_s_per_16=round(text_s, 2),
                   model_load_s=load_s, zero_npu=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    log(dict(phase="complete", summary=summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
