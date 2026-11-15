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
    ref_raw = None
    expected = None
    for p in sorted(ref_dir.rglob("*")):
        if p.suffix == ".npz" and "raw" in p.name.lower():
            ref_raw = p
        if p.suffix in (".json", ".npy") and "embed" in p.name.lower():
            expected = p
    log(dict(phase="reference_found", raw=str(ref_raw), expected=str(expected)))
    stage_a = dict(passed=False)
    if ref_raw is not None:
        raw = dict(np.load(ref_raw))
        prepared = mm.prepare_brep(raw, ff_mean, ff_std)
        with torch.no_grad():
            z = mm.embed_brep(model, prepared, dim=768, device="cpu")
        stage_a["our_embed_dim"] = int(z.shape[-1])
        stage_a["our_embed_norm"] = round(float(np.linalg.norm(z.numpy())), 6)
        if expected is not None and expected.suffix == ".json":
            exp = json.loads(expected.read_text(encoding="utf-8"))
            for key in ("embedding", "vector", "embed", "z"):
                if isinstance(exp, dict) and key in exp:
                    exp = np.asarray(exp[key], dtype=np.float64)
                    break
            if isinstance(exp, np.ndarray):
                e = exp.reshape(-1)
                zo = z.numpy().reshape(-1).astype(np.float64)
                cos = float(zo @ e / (np.linalg.norm(zo) * np.linalg.norm(e) + 1e-12))
                stage_a["official_dim"] = int(e.shape[0])
                stage_a["cosine_vs_official"] = round(cos, 6)
                stage_a["passed"] = bool(cos > 0.999)
        elif expected is not None and expected.suffix == ".npy":
            e = np.load(expected).reshape(-1).astype(np.float64)
            zo = z.numpy().reshape(-1).astype(np.float64)
            cos = float(zo @ e / (np.linalg.norm(zo) * np.linalg.norm(e) + 1e-12))
            stage_a["official_dim"] = int(e.shape[0])
            stage_a["cosine_vs_official"] = round(cos, 6)
            stage_a["passed"] = bool(cos > 0.999)
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
