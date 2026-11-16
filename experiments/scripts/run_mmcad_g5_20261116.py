"""MM-CAD G5 full-library evaluation on yunnao-share4 (user-approved sequence:
CLIP -> MM-CAD -> OpenShape, 2026-11-16).

Stage 0  pins: frozen G5 protocol inputs (entry load_frozen_protocol), SUM
         artifact sidecar sha, checkpoint/stats/reference files exist
Stage 1  first-batch gate: official brep_reference round-trip through OUR
         conversion path (cosine vs released embedding must be >0.999) plus
         16 gallery STEPs embedded finitely
Stage 2  full covered gallery: STEP -> step_to_brep npz (cached, resumable)
         -> prepare_brep(stats) -> embed_brep dim=768
Stage 3  text tower over the 3,644 frozen unique texts (official
         "title: none | text: {q}" prompt template)
Stage 4  binary score artifact (text 3644 x covered cad axis, direction
         higher) -> unified evaluation entry vs certified SUM (sliced to the
         same axis) with shared-draw paired bootstrap seed 20261112 x5000

Zero NPU; cadretrieval-glm-torch271 env on share4. New independent run dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TEXT_PROMPT = "title: none | text: {q}"  # official inference.py template
BATCH = 32


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--mmcad-repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--steps-dir", required=True)
    ap.add_argument("--sum-npy", required=True)
    ap.add_argument("--sum-sidecar", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-gate", type=int, default=16)
    ap.add_argument("--embeddinggemma-dir", default=None,
                    help="local EmbeddingGemma-300m snapshot dir; overrides Config.TEXT_MODEL "
                         "so the SentenceTransformer loads offline from this path")
    ap.add_argument("--occ-python", default=None,
                    help="python executable that has pythonocc (step_to_brep subprocess); "
                         "defaults to this script's interpreter")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out_dir)
    resuming = out.exists()
    out.mkdir(parents=True, exist_ok=True)
    if resuming:
        log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)
        log(dict(phase="resume", dir=str(out),
                 note="existing run dir: cached brep npz are reused; a stale entry_evaluation "
                      "must be removed manually"))
    log = lambda o: print(json.dumps(o, ensure_ascii=False, default=str), flush=True)
    occ_python = a.occ_python or sys.executable
    t_start = time.time()

    # ---- stage 0: pins ------------------------------------------------------
    sys.path.insert(0, str(repo / "experiments" / "src"))
    import text2cad_g5_cpu as g5  # noqa: E402
    from text2cad_g5_eval_entry import ScoreSet, evaluate, load_frozen_protocol  # noqa: E402

    protocol = load_frozen_protocol(repo, Path(a.data_root), Path(a.runtime_root),
                                    require_certificate=True)
    linked = protocol["linked"]
    sum_side = json.loads(Path(a.sum_sidecar).read_text(encoding="utf-8"))
    g5.require(sum_side["matrix_sha256"] == sha256_file(Path(a.sum_npy)),
               "SUM artifact matrix sha differs from sidecar pin")
    sum_method = sum_side.get("method") or sum_side.get("meta", {}).get("method")
    g5.require(sum_method == "text2cad_sum_replay", f"SUM artifact provenance differs: {sum_method!r}")
    ck = Path(a.ckpt)
    stats_p = Path(a.stats)
    g5.require(ck.exists() and ck.stat().st_size > 1e9, "checkpoint missing or too small")
    g5.require(stats_p.exists(), "stats missing")
    refs = sorted(Path("/home/ma-user/work/CADRetrieval_runtime/mmcad_weights/reference").glob("*.npz"))
    g5.require(len(refs) == 3, f"expected 3 official reference parts, found {len(refs)}")
    log(dict(phase="pins_ok", ckpt_gb=round(ck.stat().st_size / 1e9, 2),
             stats=sha256_file(stats_p)[:12]))

    # ---- load model ---------------------------------------------------------
    sys.path.insert(0, str(Path(a.mmcad_repo)))
    import inference as mm  # noqa: E402
    import torch  # noqa: E402

    eg_dir = None
    if a.embeddinggemma_dir:
        eg_dir = Path(a.embeddinggemma_dir)
        g5.require((eg_dir / "config.json").exists() and (eg_dir / "modules.json").exists(),
                   "EmbeddingGemma local snapshot incomplete (config.json/modules.json)")
        mm.Config.TEXT_MODEL = str(eg_dir)
        # the encoder's default argument was bound at class-definition time —
        # patching Config alone cannot reach it; replace the bound default so
        # SentenceTransformer loads the local snapshot instead of the hub id
        mm.EmbeddingGemmaEncoder.__init__.__defaults__ = (str(eg_dir),)

    t0 = time.time()
    model = mm.load_joint_model(str(ck), device="cpu")
    ff_mean = stats_np = np.load(stats_p)["ff_mean"]
    ff_std = np.load(stats_p)["ff_std"]
    model_load_s = round(time.time() - t0, 1)
    log(dict(phase="model_loaded", seconds=model_load_s))

    def embed_prepared(prepared):
        with torch.no_grad():
            z = mm.embed_brep(model, prepared, dim=768, device="cpu")
        return z.numpy().astype(np.float64).reshape(-1)

    # ---- stage 1: first-batch gate -----------------------------------------
    gate = dict(reference=[], gallery_ok=0, gallery_fail=[], passed=True)
    for p in refs:
        zr = np.load(p)
        raw = {k[len("raw_"):]: zr[k] for k in zr.files if k.startswith("raw_")}
        z = embed_prepared(mm.prepare_brep(raw, ff_mean, ff_std))
        e = zr["embedding"].reshape(-1).astype(np.float64)
        cos = float(z @ e / (np.linalg.norm(z) * np.linalg.norm(e) + 1e-12))
        gate["reference"].append(dict(file=p.name, cosine=round(cos, 6)))
        gate["passed"] = gate["passed"] and cos > 0.999
    steps_all = sorted(Path(a.steps_dir).glob("*.step"))
    for s in steps_all[: a.n_gate]:
        npz = out / "brep_cache" / (s.stem + ".npz")
        npz.parent.mkdir(exist_ok=True)
        if not npz.exists():
            r = subprocess.run([occ_python, str(Path(a.mmcad_repo) / "scripts" / "step_to_brep.py"),
                                str(s), "-o", str(npz)], capture_output=True, text=True)
            if r.returncode != 0 or not npz.exists():
                gate["gallery_fail"].append(s.name)
                gate["passed"] = False
                continue
        z = embed_prepared(mm.prepare_brep(dict(np.load(npz)), ff_mean, ff_std))
        if np.isfinite(z).all() and np.linalg.norm(z) > 0:
            gate["gallery_ok"] += 1
        else:
            gate["gallery_fail"].append(s.name)
            gate["passed"] = False
    log(dict(phase="first_batch_gate", **gate))
    if not gate["passed"]:
        log(dict(phase="GATE_FAILED"))
        return 1

    # ---- stage 2: full covered gallery --------------------------------------
    covered = [s for s in steps_all if s.name not in set(gate["gallery_fail"])]
    emb = np.empty((len(steps_all), 768), dtype=np.float64)
    fail2 = {}
    cache = out / "brep_cache"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    n_ok = 0
    for i, s in enumerate(steps_all):
        npz = cache / (s.stem + ".npz")
        try:
            if not npz.exists():
                r = subprocess.run([occ_python, str(Path(a.mmcad_repo) / "scripts" / "step_to_brep.py"),
                                    str(s), "-o", str(npz)], capture_output=True, text=True)
                if r.returncode != 0 or not npz.exists():
                    raise RuntimeError(r.stderr[-160:])
            z = embed_prepared(mm.prepare_brep(dict(np.load(npz)), ff_mean, ff_std))
            if not (np.isfinite(z).all() and np.linalg.norm(z) > 0):
                raise RuntimeError("non-finite/zero embedding")
            emb[i] = z
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            fail2[s.stem] = f"{type(e).__name__}: {str(e)[:120]}"
            emb[i] = 0.0
        if (i + 1) % 200 == 0 or i == len(steps_all) - 1:
            log(dict(phase="gallery_progress", done=i + 1, total=len(steps_all),
                     ok=n_ok, failures=len(fail2), seconds=round(time.time() - t0, 1)))
    ok_idx = [i for i, s in enumerate(steps_all) if s.stem not in fail2]
    log(dict(phase="gallery_done", ok=n_ok, failures=len(fail2),
             seconds=round(time.time() - t0, 1)))

    cad_ids_full = None
    try:
        cand = [r["cad_id"] for r in
                (json.loads(l) for l in open(Path(a.runtime_root) / "pair_manifest_v4_worker" / "cad_store.jsonl",
                                             encoding="utf-8"))]
        if all(c for c in cand):
            cad_ids_full = cand
    except KeyError:
        cad_ids_full = None
    if cad_ids_full is None:
        mp = json.loads((Path(a.runtime_root) / "pair_manifest_v4" / "evaluation_map.json").read_text(encoding="utf-8"))
        cad_ids_full = mp["cad_index_to_cad_id"]
    g5.require(len(cad_ids_full) == 8046 and all(cad_ids_full), "cad axis resolution failed")
    stem_of = {c.replace("/", "__"): i for i, c in enumerate(cad_ids_full)}
    covered_idx = sorted(stem_of[Path(s).stem] for s in steps_all if Path(s).stem not in fail2)

    # ---- stage 3: text tower -------------------------------------------------
    text_rows = [json.loads(l) for l in open(Path(a.runtime_root) / "pair_manifest_v4_worker" / "texts.jsonl",
                                             encoding="utf-8")]
    texts = [r["text"] for r in text_rows]
    assert len(texts) == 3644
    T = np.empty((3644, 768), dtype=np.float64)
    t0 = time.time()
    for i in range(0, 3644, BATCH):
        chunk = [TEXT_PROMPT.format(q=t) for t in texts[i:i + BATCH]]
        tokd = model.text_encoder.tokenizer(chunk, padding=True, truncation=True,
                                            max_length=mm.Config.TEXT_MAX_LENGTH,
                                            return_tensors="pt").to("cpu")
        with torch.no_grad():
            z = model.text_encoder(tokd["input_ids"], tokd["attention_mask"])
        z = mm.truncate(z, 768).numpy().astype(np.float64)
        T[i:i + len(chunk)] = z
        if (i // BATCH) % 20 == 0:
            log(dict(phase="text_progress", done=i + len(chunk), seconds=round(time.time() - t0, 1)))
    text_s = round(time.time() - t0, 1)
    g5.require(bool(np.isfinite(T).all() and (np.linalg.norm(T, axis=1) > 0).all()),
               "text embeddings non-finite/zero")

    # ---- artifact + evaluation ----------------------------------------------
    En = emb[ok_idx]
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    Enn = En / (np.linalg.norm(En, axis=1, keepdims=True) + 1e-12)
    sim = Tn @ Enn.T  # (3644, n_covered) cosine, direction higher
    npy = out / "mmcad_scores_subset_axis_f64.npy"
    side = out / "mmcad_scores_sidecar.json"
    ScoreSet.write_binary_artifact(
        sim, side, npy, direction="higher",
        text_indices=list(range(3644)), cad_indices=covered_idx, dtype="float64",
        meta=dict(method="mmcad_baseline_trimodal_v4",
                  ckpt=str(ck), ckpt_sha256=sha256_file(ck)[:16],
                  dim=768, text_prompt=TEXT_PROMPT,
                  coverage=dict(covered=len(covered_idx), failures=len(fail2)),
                  zero_npu=True))
    log(dict(phase="artifact_written", matrix_sha256=sha256_file(npy)[:16]))
    timing = dict(model_load_s=model_load_s, gallery_s=round(time.time() - t0, 1),
                  text_encode_s=text_s, gallery_failures=fail2,
                  covered=len(covered_idx), failed=len(fail2))
    (out / "encode_timing.json").write_text(json.dumps(timing, indent=1) + "\n",
                                            encoding="utf-8", newline="\n")
    if a.skip_eval:
        log(dict(phase="eval_skipped"))
        return 0

    clip_set = ScoreSet.from_binary_artifact(npy, side)
    clip_set.meta["name"] = "mmcad_baseline_trimodal_v4"
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
                       method_meta=dict(name="mmcad_baseline_trimodal_v4",
                                        ckpt=str(ck), dim=768, text_prompt=TEXT_PROMPT,
                                        spec="official exanos/MMCAD inference path"),
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
    g5_assert = dict(mode="sub_library", sum_arm_queries=len(ours),
                     match_certified=n_match,
                     note="SUM ranks recomputed on the covered axis (sub-library)")
    (out / "sum_arm_certified_check.json").write_text(
        json.dumps(g5_assert, indent=1) + "\n", encoding="utf-8", newline="\n")

    lines = []
    for p in sorted(out.rglob("*")):
        if (p.is_file() and p.name != "SHA256SUMS"
                and "brep_cache" not in p.parts):
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(out).as_posix())
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = dict(status="complete", out=str(out),
                   gate=gate, timing=timing,
                   coverage=dict(covered=len(covered_idx), failed=len(fail2),
                                 failed_ids=sorted(fail2)),
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
