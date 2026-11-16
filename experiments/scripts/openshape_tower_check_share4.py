"""OpenShape dual-tower validation on yunnao-share4 (CPU).

Stage PC  : PointBERT G14 (openshape-demo-support load_pc_encoder) over the
            transferred gallery point clouds (4 npz).
Stage TEXT: OpenCLIP ViT-bigG-14 (laion2b) loaded from the hf-mirror snapshot,
            16 frozen unique texts, CPU single-text latency.
Factual record: 4x16 cosine grid (both towers share the OpenShape joint
space). No pass/fail beyond finiteness.

  cadretreival-glm-torch271 python openshape_tower_check_share4.py --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pc-dir", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--n-texts", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)

    import torch  # noqa: E402

    summary = dict(schema="openshape-tower-check-v1", host="yunnao-share4",
                   device="cpu", torch=torch.__version__, zero_npu_forward=True)

    # ---- stage PC: PointBERT G14 ------------------------------------------
    demo_support = None
    for p in Path("/home/ma-user/work/CADRetrieval_runtime/hf_home/hub").glob(
            "models--OpenShape--openshape-demo-support/snapshots/*/openshape"):
        demo_support = str(p)
    # PROBE-ONLY dgl shim: pip dgl has no torch-2.7.1 graphbolt build, so the
    # PointBERT FPS op is served by a documented deterministic torch FPS at
    # probe_shims/dgl (production sampling spec frozen at the method round).
    sys.path.insert(0, "/home/ma-user/work/CADRetrieval_runtime/probe_shims")
    sys.path.insert(0, str(Path(demo_support).parent))
    import openshape  # noqa: E402

    t0 = time.time()
    pc_model = openshape.load_pc_encoder("openshape-pointbert-vitg14-rgb")
    pc_model.eval()
    summary["pc_load_s"] = round(time.time() - t0, 1)

    npzs = sorted(Path(a.pc_dir).glob("*_pc.npz"))[:4]
    pc_embeds = []
    t0 = time.time()
    for p in npzs:
        z = np.load(p)
        pc6 = np.concatenate([z["points"], z["rgb"]], axis=1)  # (10000, 6)
        x = torch.tensor(pc6[:, [0, 2, 1, 3, 4, 5]].T[None], dtype=torch.float32)
        with torch.no_grad():
            e = pc_model(x)
        pc_embeds.append(e.numpy().astype(np.float64))
    pc_s = round(time.time() - t0, 2)
    E = np.vstack(pc_embeds)
    summary["pc"] = dict(n=len(npzs), seconds=pc_s,
                         embed_dim=int(E.shape[1]),
                         all_finite=bool(np.isfinite(E).all()),
                         norms=[round(float(np.linalg.norm(E[i])), 4) for i in range(len(E))])
    log(dict(phase="pc_done", **{k: summary["pc"][k] for k in ("n", "seconds", "embed_dim", "all_finite")}))

    # ---- stage TEXT: OpenCLIP ViT-bigG-14 ---------------------------------
    import open_clip  # noqa: E402

    ckpt = None
    for p in Path("/home/ma-user/work/CADRetrieval_runtime/hf_home/hub").glob(
            "models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/*/open_clip_pytorch_model.bin"):
        ckpt = str(p)
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained=ckpt, cache_dir="/home/ma-user/work/CADRetrieval_runtime/hf_home")
    model.eval()
    summary["text_load_s"] = round(time.time() - t0, 1)
    tok = open_clip.get_tokenizer("ViT-bigG-14")

    texts = [json.loads(l)["text"] for l in open(a.texts, encoding="utf-8")][: a.n_texts]
    lat = []
    T = []
    for t in texts:
        t0 = time.time()
        with torch.no_grad():
            e = model.encode_text(tok([t]).to("cpu")).numpy().astype(np.float64)
        lat.append(round(time.time() - t0, 3))
        T.append(e / (np.linalg.norm(e) + 1e-12))
    T = np.vstack(T)
    summary["text"] = dict(n=len(texts), latency_s_first=lat[0],
                           latency_s_mean_rest=round(float(np.mean(lat[1:])), 3),
                           embed_dim=int(T.shape[1]),
                           all_finite=bool(np.isfinite(T).all()),
                           all_nonzero=bool((np.linalg.norm(T, axis=1) > 0).all()))
    log(dict(phase="text_done", **{k: summary["text"][k] for k in
                                   ("n", "latency_s_first", "latency_s_mean_rest", "embed_dim", "all_finite")}))

    # ---- factual cross-tower grid -----------------------------------------
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    grid = (En @ T.T).tolist()
    summary["cosine_grid_pc_x_text"] = [[round(v, 4) for v in row] for row in grid]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    log(dict(phase="complete", out=str(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
