"""MM-CAD G5 v2 re-run (local, cadclip env) — review round: fixes the two
findings that invalidated the first formal run.

Fix 1 (weights): the checkpoint's text-tower keys use the segment
`auto_model` while the runtime pipeline names it `model` — strict=False
silently degraded to a hybrid (base EmbeddingGemma + trained B-Rep tower,
314/314 keys unmatched). v2 remaps `.auto_model.` -> `.model.` on
text_encoder keys and loads with strict=True (0 missing / 0 unexpected
verified before any embedding).

Fix 2 (topology): the official step_to_brep.py adjacency extraction fails on
this pythonocc build (topexp.MapShapesAndAncestors attribute error swallowed
per-edge -> ALL edge_to_faces -1). v2 uses step_to_brep_fixed_20261118.py:
correct OCC symbol + IsSame key resolution so shared edges carry the full
2-face adjacency on both traversal instances, with per-edge errors recorded.

Zero NPU; everything local. New independent run dir.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TEXT_PROMPT = "title: none | text: {q}"  # official inference.py template
BATCH = 32


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def convert_one(job):
    step, npz = job
    if npz.exists() and npz.stat().st_size > 0:
        z = np.load(npz)
        if int(z["adjacency_error_count"]) == 0 and int(((z["edge_to_faces"][:, 0] >= 0) & (z["edge_to_faces"][:, 1] >= 0)).sum()) > 0:
            return step.stem, "cached"
        return step.stem, "stale"
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "step_to_brep_fixed_20261118.py"),
                        str(step), "-o", str(npz)], capture_output=True, text=True)
    if r.returncode == 0 and npz.exists() and npz.stat().st_size > 0:
        return step.stem, "ok"
    return step.stem, f"fail: {r.stderr.strip()[-160:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--mmcad-repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--embeddinggemma-dir", required=True)
    ap.add_argument("--steps-dir", required=True)
    ap.add_argument("--brep-cache", required=True, help="fixed-extractor npz cache (new dir)")
    ap.add_argument("--sum-npy", required=True)
    ap.add_argument("--sum-sidecar", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)
    t_start = time.time()

    sys.path.insert(0, str(repo / "experiments" / "src"))
    import text2cad_g5_cpu as g5  # noqa: E402
    from text2cad_g5_eval_entry import ScoreSet, evaluate, load_frozen_protocol  # noqa: E402

    protocol = load_frozen_protocol(repo, Path(a.data_root), Path(a.runtime_root),
                                    require_certificate=True)
    texts = [json.loads(l)["text"] for l in
             open(Path(a.runtime_root) / "pair_manifest_v4_worker" / "texts.jsonl", encoding="utf-8")]
    g5.require(len(texts) == 3644, "frozen unique text count differs")

    # ---- stage 1: fixed conversion (resumable, adjacency-validating) --------
    steps = sorted(Path(a.steps_dir).glob("*.step"))
    brep_cache = Path(a.brep_cache)
    brep_cache.mkdir(parents=True, exist_ok=True)
    jobs = [(s, brep_cache / (s.stem + ".npz")) for s in steps]
    log(dict(phase="convert_plan", n=len(jobs), workers=a.workers))
    t0 = time.time()
    conv_fail = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (stem, status) in enumerate(ex.map(convert_one, jobs)):
            if status.startswith("fail") or status == "stale":
                conv_fail[stem] = status
            if (k + 1) % 400 == 0 or k + 1 == len(jobs):
                log(dict(phase="convert_progress", done=k + 1, total=len(jobs),
                         failures=len(conv_fail), seconds=round(time.time() - t0, 1)))
    convert_s = round(time.time() - t0, 1)
    ok_steps = [s for s, j in zip(steps, jobs) if j[1].exists() and j[1].stat().st_size > 0]
    log(dict(phase="convert_done", ok=len(ok_steps), failures=len(conv_fail), seconds=convert_s))

    # ---- stage 2: strict-load joint model ------------------------------------
    sys.path.insert(0, str(Path(a.mmcad_repo)))
    import inference as mm  # noqa: E402
    import torch  # noqa: E402
    torch.set_num_threads(a.threads)

    mm.Config.TEXT_MODEL = str(Path(a.embeddinggemma_dir))
    mm.EmbeddingGemmaEncoder.__init__.__defaults__ = (str(Path(a.embeddinggemma_dir)),)
    t0 = time.time()
    model = mm.Baseline(mm.EmbeddingGemmaEncoder(), mm.Config)
    ck = torch.load(Path(a.ckpt), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    fixed_sd = {k.replace(".auto_model.", ".model.") if k.startswith("text_encoder._pipeline.") else k: v
                for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(fixed_sd, strict=True)
    # load_joint_model does eval().to(device) — the inlined path must too:
    # SentenceTransformer auto-moved the text tower to npu:0 (torch_npu
    # present), and cpu inputs then crash the text forward
    model.eval().to("cpu")
    load_s = round(time.time() - t0, 1)
    log(dict(phase="strict_load", missing=len(missing), unexpected=len(unexpected),
             epoch=ck.get("epoch"), seconds=load_s))
    g5.require(len(missing) == 0 and len(unexpected) == 0,
               "strict load failed — the hybrid degradation must not repeat")
    stats_np = np.load(Path(a.stats))
    ff_mean, ff_std = stats_np["ff_mean"], stats_np["ff_std"]

    def embed_prepared(prepared):
        with torch.no_grad():
            z = mm.embed_brep(model, prepared, dim=768, device="cpu")
        return z.numpy().astype(np.float64).reshape(-1)

    # ---- stage 3: full covered gallery embed ---------------------------------
    emb = np.empty((len(steps), 768), dtype=np.float64)
    emb_fail = {}
    embeds_cache = out / "embeds_cache"
    embeds_cache.mkdir(exist_ok=True)
    t0 = time.time()
    n_ok = 0
    for i, s in enumerate(steps):
        npz = brep_cache / (s.stem + ".npz")
        epath = embeds_cache / (s.stem + "_e768.npy")
        try:
            if epath.exists() and epath.stat().st_size > 0:
                emb[i] = np.load(epath)
                n_ok += 1
                continue
            zr = np.load(npz)
            g5.require(int(zr["adjacency_error_count"]) == 0, "cached npz has adjacency errors")
            z = embed_prepared(mm.prepare_brep(dict(zr), ff_mean, ff_std))
            g5.require(bool(np.isfinite(z).all() and np.linalg.norm(z) > 0), "bad embedding")
            np.save(epath, z.astype(np.float32))
            emb[i] = z
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            emb_fail[s.stem] = f"{type(e).__name__}: {str(e)[:120]}"
        if (i + 1) % 400 == 0 or i + 1 == len(steps):
            log(dict(phase="embed_progress", done=i + 1, total=len(steps),
                     ok=n_ok, failures=len(emb_fail), seconds=round(time.time() - t0, 1)))
    embed_s = round(time.time() - t0, 1)
    log(dict(phase="embed_done", ok=n_ok, failures=len(emb_fail), seconds=embed_s))

    ok_steps = [s for s in steps if s.stem not in emb_fail]
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
    idx_of = {c: i for i, c in enumerate(cad_ids_full)}
    # cad ids come from the npz names themselves (built from the gallery
    # cad_id during conversion) — cad_store.jsonl on this host has no cad_id
    cache_names = {p.name for p in brep_cache.glob("*.npz")}
    covered_cad_ids = sorted(n.replace(".npz", "").replace("__", "/") for n in cache_names)
    covered_idx = sorted(idx_of[c] for c in covered_cad_ids)
    g5.require(len(covered_idx) > 0, "covered axis empty")

    # ---- stage 4: text tower ---------------------------------------------------
    T = np.empty((3644, 768), dtype=np.float64)
    t0 = time.time()
    for i in range(0, 3644, BATCH):
        chunk = [TEXT_PROMPT.format(q=t) for t in texts[i:i + BATCH]]
        tokd = model.text_encoder.tokenizer(chunk, padding=True, truncation=True,
                                            max_length=mm.Config.TEXT_MAX_LENGTH,
                                            return_tensors="pt").to("cpu")
        with torch.no_grad():
            z = mm.truncate(model.text_encoder(tokd["input_ids"], tokd["attention_mask"]), 768)
        T[i:i + len(z)] = z.numpy().astype(np.float64)
        if (i // BATCH) % 20 == 0:
            log(dict(phase="text_progress", done=i + len(z), seconds=round(time.time() - t0, 1)))
    text_s = round(time.time() - t0, 1)
    g5.require(bool(np.isfinite(T).all() and (np.linalg.norm(T, axis=1) > 0).all()),
               "text embeddings non-finite/zero")
    log(dict(phase="text_done", seconds=text_s))

    # ---- stage 5: artifact + evaluation ----------------------------------------
    En = emb[[steps.index(s) for s in ok_steps]]
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    Enn = En / (np.linalg.norm(En, axis=1, keepdims=True) + 1e-12)
    sim = Tn @ Enn.T
    npy = out / "mmcad_v2_scores_subset_axis_f64.npy"
    side = out / "mmcad_v2_scores_sidecar.json"
    ScoreSet.write_binary_artifact(
        sim, side, npy, direction="higher",
        text_indices=list(range(3644)), cad_indices=covered_idx, dtype="float64",
        meta=dict(method="mmcad_baseline_trimodal_v4_strict",
                  ckpt=str(ck), ckpt_sha256=sha256_file(Path(a.ckpt))[:16],
                  strict_load=True, text_key_remap="auto_model->model",
                  fixed_topology=True, dim=768, text_prompt=TEXT_PROMPT,
                  coverage=dict(covered=len(covered_idx), convert_fail=len(conv_fail),
                                embed_fail=len(emb_fail)),
                  zero_npu=True))
    log(dict(phase="artifact_written", matrix_sha256=sha256_file(npy)[:16]))

    clip_set = ScoreSet.from_binary_artifact(npy, side)
    clip_set.meta["name"] = "mmcad_baseline_trimodal_v4_strict"
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
                       method_meta=dict(name="mmcad_baseline_trimodal_v4_strict",
                                        ckpt=str(ck), strict_load=True, dim=768,
                                        text_prompt=TEXT_PROMPT,
                                        spec="official path + fixed topology extraction"),
                       cad_subset=sorted(covered_idx), paired_sum_set=sum_set,
                       bootstrap=dict(seed=20261112, replicates=5000))

    lines = []
    for p in sorted(out.rglob("*")):
        if (p.is_file() and p.name != "SHA256SUMS"
                and "brep_cache" not in p.parts):
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(out).as_posix())
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = dict(status="complete", out=str(out),
                   strict_load=dict(missing=0, unexpected=0),
                   timing=dict(convert_s=convert_s, embed_s=embed_s, text_s=text_s,
                               model_load_s=load_s),
                   coverage=dict(covered=len(covered_idx), convert_fail=len(conv_fail),
                                 embed_fail=len(emb_fail),
                                 convert_fail_ids=sorted(conv_fail),
                                 embed_fail_ids=sorted(emb_fail)),
                   metrics=results["metrics"], hit_counts=results["hit_counts"],
                   paired_r20=results["statistics"]["paired_ci95"]["R@20"],
                   paired_medr=results["statistics"]["paired_ci95"]["MedR"],
                   elapsed_s=round(time.time() - t_start, 1), zero_npu=True)
    (out / "final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n")
    log(dict(phase="complete", metrics=results["metrics"],
             paired_r20=summary["paired_r20"], elapsed_s=summary["elapsed_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
