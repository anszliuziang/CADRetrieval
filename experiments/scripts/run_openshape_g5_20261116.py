"""OpenShape G5 full-library evaluation on yunnao-share4 (approved sequence
step 3).

Stage 0  pins: frozen protocol inputs, SUM artifact sidecar sha, PointBERT
         model.pt + bigG checkpoint present
Stage 1  first-batch gate: 4 gallery point clouds -> PointBERT tower (1280-d
         finite) + 2 frozen texts -> bigG tower (1280-d finite non-zero)
Stage 2  full covered gallery PC tower embed (FPS shim documented; resumable
         via per-uid embed cache)
Stage 3  bigG text tower over the 3,644 frozen unique texts
Stage 4  binary score artifact -> unified entry vs certified SUM (sliced
         axis) with paired bootstrap seed 20261112 x5000

Zero NPU; cadretrieval-glm-torch271 env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

BATCH = 32
PC_BATCH = 8


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--pc-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sum-npy", required=True)
    ap.add_argument("--sum-sidecar", required=True)
    ap.add_argument("--n-texts", type=int, default=16)
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)
    occ_python = sys.executable
    t_start = time.time()

    sys.path.insert(0, str(repo / "experiments" / "src"))
    import text2cad_g5_cpu as g5  # noqa: E402
    from text2cad_g5_eval_entry import ScoreSet, evaluate, load_frozen_protocol  # noqa: E402

    protocol = load_frozen_protocol(repo, Path(a.data_root), Path(a.runtime_root),
                                    require_certificate=True)
    sum_side = json.loads(Path(a.sum_sidecar).read_text(encoding="utf-8"))
    g5.require(sum_side["matrix_sha256"] == sha256_file(Path(a.sum_npy)),
               "SUM artifact matrix sha differs from sidecar pin")
    sum_method = sum_side.get("method") or sum_side.get("meta", {}).get("method")
    g5.require(sum_method == "text2cad_sum_replay", "SUM artifact provenance differs")

    import torch  # noqa: E402
    torch.set_num_threads(16)

    # probe FPS shim + demo-support inference lib
    sys.path.insert(0, "/home/ma-user/work/CADRetrieval_runtime/probe_shims")
    demo_support = None
    for p in Path("/home/ma-user/work/CADRetrieval_runtime/hf_home/hub").glob(
            "models--OpenShape--openshape-demo-support/snapshots/*/openshape"):
        demo_support = str(p)
    g5.require(demo_support is not None, "demo-support library missing")
    sys.path.insert(0, str(Path(demo_support).parent))
    import openshape  # noqa: E402

    import open_clip  # noqa: E402
    from PIL import Image  # noqa: E402
    from ppat_rgb import PointPatchTransformer, Projected  # noqa: F401  (import check)

    ckpt = None
    for p in Path("/home/ma-user/work/CADRetrieval_runtime/hf_home/hub").glob(
            "models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/*/open_clip_pytorch_model.bin"):
        ckpt = str(p)
    g5.require(ckpt is not None, "bigG open_clip checkpoint missing")
    pb = None
    for p in Path("/home/ma-user/work/CADRetrieval_runtime/hf_home/hub").glob(
            "models--OpenShape--openshape-pointbert-vitg14-rgb/snapshots/*/model.pt"):
        pb = str(p)
    g5.require(pb is not None, "PointBERT model.pt missing")
    spec_sha = sha256_file(Path(pb))
    log(dict(phase="pins_ok", bigg=str(ckpt)[-40:], pointbert_sha=spec_sha[:12]))

    # covered axis: from the CLIP coverage contract transferred with the PCs
    cov_path = Path(a.pc_dir) / "coverage.json"
    if cov_path.exists():
        covered_ids = sorted(set(json.loads(cov_path.read_text(encoding="utf-8"))["covered_cad_ids"]))
    else:
        pc_dir = Path(a.pc_dir)
        covered_ids = sorted(p.name.replace("_pc.npz", "").replace("__", "/")
                             for p in pc_dir.glob("*_pc.npz"))
    g5.require(len(covered_ids) > 0, "no covered point clouds found")
    log(dict(phase="covered_axis", n=len(covered_ids)))

    # ---- models -------------------------------------------------------------
    t0 = time.time()
    pc_model = openshape.load_pc_encoder("openshape-pointbert-vitg14-rgb")
    pc_model.eval()
    pc_load_s = round(time.time() - t0, 1)
    t0 = time.time()
    t_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained=ckpt,
        cache_dir="/home/ma-user/work/CADRetrieval_runtime/hf_home")
    t_model.eval()
    tok = open_clip.get_tokenizer("ViT-bigG-14")
    text_load_s = round(time.time() - t0, 1)
    log(dict(phase="models_loaded", pc_s=pc_load_s, text_s=text_load_s))

    def embed_pc(npz_path: Path) -> np.ndarray:
        z = np.load(npz_path)
        pc6 = np.concatenate([z["points"], z["rgb"]], axis=1)
        x = torch.tensor(pc6[:, [0, 2, 1, 3, 4, 5]].T[None], dtype=torch.float32)
        with torch.no_grad():
            e = pc_model(x)
        return e.numpy().astype(np.float64).reshape(-1)

    # ---- stage 1: first-batch gate ------------------------------------------
    npzs = sorted(Path(a.pc_dir).glob("*_pc.npz"))[:4]
    gate_embeds = [embed_pc(p) for p in npzs]
    gate_pc = bool(all(np.isfinite(e).all() and np.linalg.norm(e) > 0 for e in gate_embeds))
    texts = [json.loads(l)["text"] for l in
             open(Path(a.runtime_root) / "pair_manifest_v4_worker" / "texts.jsonl", encoding="utf-8")]
    g5.require(len(texts) == 3644, "frozen unique text count differs")
    with torch.no_grad():
        te = t_model.encode_text(tok(texts[:2]).to("cpu")).numpy().astype(np.float64)
    gate_text = bool(np.isfinite(te).all() and (np.linalg.norm(te, axis=1) > 0).all())
    gate = dict(pc_finite=gate_pc, text_finite=gate_text, pc_dim=int(gate_embeds[0].shape[0]),
                text_dim=int(te.shape[1]), passed=gate_pc and gate_text)
    log(dict(phase="first_batch_gate", **gate))
    if not gate["passed"]:
        log(dict(phase="GATE_FAILED"))
        return 1

    # ---- stage 2: full covered gallery PC embed (resumable) ------------------
    cache = out / "pc_embeds"
    cache.mkdir(exist_ok=True)
    axis_ids = []
    t0 = time.time()
    n_new = 0
    for k, p in enumerate(sorted(Path(a.pc_dir).glob("*_pc.npz"))):
        cid = p.name.replace("_pc.npz", "").replace("__", "/")
        axis_ids.append(cid)
        epath = cache / (p.stem + "_e1280.npy")
        if not epath.exists():
            e = embed_pc(p)
            np.save(epath, e.astype(np.float32))
            n_new += 1
        if (k + 1) % 200 == 0 or k + 1 == len(axis_ids):
            log(dict(phase="pc_progress", done=k + 1, total=len(axis_ids),
                     new=n_new, seconds=round(time.time() - t0, 1)))
    pc_embeds = np.vstack([np.load(cache / (c.replace("/", "__") + "_pc_e1280.npy")).astype(np.float64)
                           for c in axis_ids])
    g5.require(bool(np.isfinite(pc_embeds).all() and
                    (np.linalg.norm(pc_embeds, axis=1) > 0).all()), "PC embeddings bad")
    pc_s = round(time.time() - t0, 1)
    log(dict(phase="pc_embed_done", n=len(axis_ids), seconds=pc_s))

    # ---- stage 3: text tower --------------------------------------------------
    T = np.empty((3644, 1280), dtype=np.float64)
    t0 = time.time()
    for i in range(0, 3644, BATCH):
        with torch.no_grad():
            e = t_model.encode_text(tok(texts[i:i + BATCH]).to("cpu")).numpy().astype(np.float64)
        T[i:i + len(e)] = e
        if (i // BATCH) % 10 == 0:
            log(dict(phase="text_progress", done=i + len(e), seconds=round(time.time() - t0, 1)))
    text_s = round(time.time() - t0, 1)
    g5.require(bool(np.isfinite(T).all() and (np.linalg.norm(T, axis=1) > 0).all()),
               "text embeddings non-finite/zero")
    log(dict(phase="text_done", seconds=text_s))

    # ---- artifact + evaluation ------------------------------------------------
    cad_ids_full = None
    try:
        cand = [r["cad_id"] for r in
                (json.loads(l) for l in open(Path(a.runtime_root) / "pair_manifest_v4_worker" / "cad_store.jsonl",
                                             encoding="utf-8"))]
        cad_ids_full = cand if all(cand) else None
    except KeyError:
        cad_ids_full = None
    if cad_ids_full is None:
        mp = json.loads((Path(a.runtime_root) / "pair_manifest_v4" / "evaluation_map.json").read_text(encoding="utf-8"))
        cad_ids_full = mp["cad_index_to_cad_id"]
    g5.require(len(cad_ids_full) == 8046, "cad axis resolution failed")
    idx_of = {c: i for i, c in enumerate(cad_ids_full)}
    covered_idx = sorted(idx_of[c] for c in axis_ids)

    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    En = pc_embeds / (np.linalg.norm(pc_embeds, axis=1, keepdims=True) + 1e-12)
    sim = Tn @ En.T  # (3644, n_covered) cosine, direction higher
    npy = out / "openshape_scores_subset_axis_f64.npy"
    side = out / "openshape_scores_sidecar.json"
    ScoreSet.write_binary_artifact(
        sim, side, npy, direction="higher",
        text_indices=list(range(3644)), cad_indices=covered_idx, dtype="float64",
        meta=dict(method="openshape_pointbert_vitg14_rgb",
                  text_tower="openclip vit bigG 14 laion2b",
                  pointbert_sha256=spec_sha[:16],
                  fps="probe shim (deterministic standard FPS); production spec frozen at method round",
                  coverage=dict(covered=len(covered_idx)), zero_npu=True))
    log(dict(phase="artifact_written", matrix_sha256=sha256_file(npy)[:16]))

    clip_set = ScoreSet.from_binary_artifact(npy, side)
    clip_set.meta["name"] = "openshape_pointbert_vitg14_rgb"
    sum_raw = np.fromfile(Path(a.sum_npy), dtype="<f8").reshape(3644, 8046)
    sub = np.ascontiguousarray(sum_raw[:, covered_idx])
    sum_npy = out / "sum_scores_subset_axis_f64.npy"
    sum_side_p = out / "sum_scores_subset_sidecar.json"
    ScoreSet.write_binary_artifact(sub, sum_side_p, sum_npy, direction="higher",
                                   text_indices=list(range(3644)),
                                   cad_indices=sorted(covered_idx), dtype="float64",
                                   meta=dict(method="text2cad_sum_certified",
                                             sliced_from=str(Path(a.sum_npy)),
                                             source_sidecar_sha256=sha256_file(Path(a.sum_sidecar))))
    sum_set = ScoreSet.from_binary_artifact(sum_npy, sum_side_p)
    sum_set.meta["name"] = "text2cad_sum_certified"
    eval_out = out / "entry_evaluation"
    results = evaluate(clip_set, protocol=protocol, output=eval_out,
                       method_meta=dict(name="openshape_pointbert_vitg14_rgb",
                                        text_tower="openclip vit bigG 14 laion2b",
                                        spec_sha256=spec_sha[:16],
                                        fps="probe shim documented"),
                       cad_subset=sorted(covered_idx), paired_sum_set=sum_set,
                       bootstrap=dict(seed=20261112, replicates=5000))

    hist = {}
    for line in open(repo / "experiments/runs/idea01_text2cad_g5_cpu_20261003" / "query_ranks.jsonl",
                     encoding="utf-8"):
        row = json.loads(line)
        hist[row["query_id"]] = row["ranks"]["sum"]
    ours = {}
    for line in (eval_out / "query_ranks.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        ours[row["query_id"]] = row["ranks"]["text2cad_sum_certified"]
    n_match = sum(1 for q in ours if hist.get(q) == ours[q])
    g5_assert = dict(mode="sub_library", sum_arm_queries=len(ours), match_certified=n_match,
                     note="SUM ranks recomputed on the covered axis (sub-library)")
    (out / "sum_arm_certified_check.json").write_text(
        json.dumps(g5_assert, indent=1) + "\n", encoding="utf-8", newline="\n")

    lines = []
    for p in sorted(out.rglob("*")):
        if (p.is_file() and p.name != "SHA256SUMS"
                and "pc_embeds" not in p.parts and p.suffix != ".png"):
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(out).as_posix())
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = dict(status="complete", out=str(out),
                   gate=gate, coverage=dict(covered=len(covered_idx)),
                   timing=dict(pc_load_s=pc_load_s, text_load_s=text_load_s,
                               pc_embed_s=pc_s, text_encode_s=text_s),
                   metrics=results["metrics"], hit_counts=results["hit_counts"],
                   paired_r20=results["statistics"]["paired_ci95"]["R@20"],
                   paired_medr=results["statistics"]["paired_ci95"]["MedR"],
                   sum_arm_check=g5_assert,
                   elapsed_s=round(time.time() - t_start, 1), zero_npu=True)
    (out / "final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n")
    log(dict(phase="complete", metrics=results["metrics"],
             paired_r20=summary["paired_r20"], elapsed_s=summary["elapsed_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
