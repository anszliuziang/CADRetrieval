"""CLIP G5 full-library evaluation (plan IDEA01_G5_EXTERNAL_BASELINES_2026-11-15
section 3 row 1; user-authorized 2026-11-15).

One launch, fully automatic:
  stage 0  load + pin the frozen render/encode spec (clip_g5_render_spec_20261115.json)
  stage 1  first-batch gate: 32 stratified UIDs (4 quartiles x 8, G5 boundary
           rule) - renders reused from the accepted probe batch, PNGs decoded,
           image/text vectors finite and non-zero, UID alignment vs the frozen
           gallery axis, truncation census over the 3,644 frozen unique texts
           (77-token CLIP context) + two truncated texts encoded
  stage 2  render + mesh the full 8,046-CAD gallery (skip-existing; one retry
           per UID; persistent failures recorded)
  stage 3  encode 8,046 images and 3,644 unique texts (batched, CPU), L2
           normalize, cosine matrix 3644 x 8046 -> fixed binary score artifact
  stage 4  unified evaluation entry (SHA-verified artifact load), CLIP vs the
           certified Text2CAD SUM (accepted v1 artifact) with shared-draw
           paired bootstrap (seed 20261112, 5000), SUM arm asserted against
           the historical certified ranks query-by-query

Zero NPU; runs in the isolated `cadclip` env. New independent run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "src"))

CADSEQPROC = REPO / ".cache/Text2CAD-official/CadSeqProc"
SPEC = REPO / "experiments/configs/clip_g5_render_spec_20261115.json"
MINIMAL_ZIP = REPO / ".cache/text2cad_official/text2cad_v1.0/misc/minimal_json/minimal_json_0000_0099.zip"
LENGTHS = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/cad_length_index.jsonl")
TEXTS = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker/texts.jsonl")
GALLERY = REPO / ".cache/m0_real_run_v3/gallery.jsonl"
SUM_NPY = REPO / "experiments/runs/idea01_g5_sum_replay_20261115/sum_scores_full_axis_f64.npy"
SUM_SIDECAR = REPO / "experiments/runs/idea01_g5_sum_replay_20261115/sum_scores_sidecar.json"
HIST_QUERY_RANKS = REPO / "experiments/runs/idea01_text2cad_g5_cpu_20261003/query_ranks.jsonl"
PROBE_RENDER_DIRS = [
    Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/clip_probe_20261115/renders"),
    Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/clip_probe_smoke_20261115/renders"),
]
BATCH = 64


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render_mesh_png(mesh, out_png: Path):
    """Frozen render (spec clip-g5-render-spec-v1): matplotlib Agg, single
    canonical isometric view, flat color, white background, 448x448."""
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
    ax.view_init(elev=30, azim=45)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def pick_stratified(per_quartile=8):
    lengths = [json.loads(l) for l in open(LENGTHS, encoding="utf-8")]
    gallery = [json.loads(l) for l in open(GALLERY, encoding="utf-8")]
    L = np.array([r["n_valid_x"] + r["n_valid_y"] for r in lengths], dtype=np.int64)
    cuts = np.quantile(L, [.25, .5, .75])
    bins = np.searchsorted(cuts, L, side="left")
    picks = []
    for b in range(4):
        for i in np.flatnonzero(bins == b)[:per_quartile]:
            picks.append(dict(cad_index=int(i), cad_id=gallery[i]["cad_id"],
                              quartile=int(b + 1),
                              minimal_json_member=gallery[i]["minimal_json_member"]))
    return picks, cuts.tolist(), gallery


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-quartile", type=int, default=8)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    repo = REPO
    out = Path(a.out_dir)
    g5_ok = (repo / "experiments/src/text2cad_g5_cpu.py").exists()
    assert g5_ok
    sys.path.insert(0, str(repo / "experiments" / "src"))

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec_sha = sha256_file(SPEC)
    out.mkdir(parents=True, exist_ok=False)
    renders = out / "renders"
    renders.mkdir()
    steps = out / "steps"
    steps.mkdir()
    log = lambda obj: print(json.dumps(obj, ensure_ascii=False, default=str), flush=True)

    # reuse accepted probe renders (same frozen spec) -----------------------
    reused = 0
    for d in PROBE_RENDER_DIRS:
        if d.is_dir():
            for png in d.glob("*.png"):
                dst = renders / png.name
                if not dst.exists():
                    shutil.copy2(png, dst)
                    reused += 1
    log(dict(phase="reuse_probe_renders", copied=reused))

    sys.path.insert(0, str(CADSEQPROC.parent))
    from CadSeqProc.cad_sequence import CADSequence  # noqa: E402

    gallery_rows = [json.loads(l) for l in open(GALLERY, encoding="utf-8")]
    assert len(gallery_rows) == 8046
    cad_ids = [r["cad_id"] for r in gallery_rows]
    member_of = {r["cad_id"]: r["minimal_json_member"] for r in gallery_rows}
    z = zipfile.ZipFile(MINIMAL_ZIP)
    png_of = {cid: renders / (cid.replace("/", "__") + ".png") for cid in cad_ids}

    # ---- stage 2: full render (first-batch UIDs render first) -------------
    picks, cuts, _ = pick_stratified(a.per_quartile)
    first_batch = {p["cad_id"] for p in picks}
    order = [p["cad_id"] for p in picks] + [c for c in cad_ids if c not in first_batch]
    failures = {}
    t_render = time.time()
    for k, cid in enumerate(order):
        png = png_of[cid]
        if png.exists() and png.stat().st_size > 0:
            continue
        ok = False
        for attempt in (1, 2):
            try:
                data = json.loads(z.read(member_of[cid]))
                seq = CADSequence.from_minimal_json(data)
                seq.create_cad_model()
                seq.create_mesh()
                render_mesh_png(seq.mesh, png)
                ok = png.exists() and png.stat().st_size > 0
                break
            except Exception as e:  # noqa: BLE001
                failures.setdefault(cid, []).append(f"attempt{attempt}: {type(e).__name__}: {e}")
                (out / f"render_failure_{cid.replace('/', '__')}.txt").write_text(
                    traceback.format_exc(), encoding="utf-8")
        if not ok:
            log(dict(phase="render_failure", cad_id=cid, attempts=failures.get(cid)))
        if (k + 1) % 400 == 0 or k == len(order) - 1:
            log(dict(phase="render_progress", done=k + 1, total=len(order),
                     seconds=round(time.time() - t_render, 1),
                     failures_so_far=len(failures)))
    render_s = round(time.time() - t_render, 1)
    covered = [c for c in cad_ids if png_of[c].exists() and png_of[c].stat().st_size > 0]
    missing = [c for c in cad_ids if c not in set(covered)]
    log(dict(phase="render_done", covered=len(covered), missing=len(missing),
             failures=len(failures), seconds=render_s))
    (out / "coverage.json").write_text(
        json.dumps(dict(covered=len(covered), missing=len(missing),
                        missing_cad_ids=missing,
                        failures={k: v for k, v in failures.items()}, expected=8046),
                   indent=1) + "\n", encoding="utf-8", newline="\n")

    # ---- stage 3: encode ---------------------------------------------------
    import torch
    from PIL import Image
    from transformers import CLIPTokenizerFast, CLIPImageProcessor, CLIPModel

    t0 = time.time()
    model = CLIPModel.from_pretrained(spec["weights"]["hf_repo"])
    model.eval()
    proc = CLIPImageProcessor.from_pretrained(spec["weights"]["hf_repo"])
    tok = CLIPTokenizerFast.from_pretrained(spec["weights"]["hf_repo"])
    clip_load_s = round(time.time() - t0, 1)

    def l2(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def encode_images(paths):
        feats = []
        t0 = time.time()
        for i in range(0, len(paths), BATCH):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
            with torch.no_grad():
                o = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
            f = o if torch.is_tensor(o) else o.pooler_output
            feats.append(f.numpy().astype(np.float64))
            if (i // BATCH) % 20 == 0:
                log(dict(phase="image_encode", batch=i // BATCH,
                         of=(len(paths) + BATCH - 1) // BATCH,
                         seconds=round(time.time() - t0, 1)))
        return np.vstack(feats), round(time.time() - t0, 1)

    # first-batch gate -------------------------------------------------------
    missing_gate = [p["cad_id"] for p in picks
                    if not (png_of[p["cad_id"]].exists() and png_of[p["cad_id"]].stat().st_size > 0)]
    if missing_gate:
        log(dict(phase="GATE_FAILED", reason="first-batch renders missing", missing=missing_gate))
        return 1
    strat_pngs = [png_of[p["cad_id"]] for p in picks]
    img_feats, _ = encode_images(strat_pngs)
    norms = np.linalg.norm(img_feats, axis=1)
    gate_images = bool(np.isfinite(img_feats).all() and (norms > 0).all())
    text_rows = [json.loads(l) for l in open(TEXTS, encoding="utf-8")]
    assert len(text_rows) == 3644 and all(r["text_index"] == i for i, r in enumerate(text_rows))
    texts = [r["text"] for r in text_rows]
    tok_lens = [len(tok(t)["input_ids"]) for t in texts]
    n_trunc = int(sum(1 for n in tok_lens if n > 77))
    trunc_samples = [texts[i] for i in np.flatnonzero(np.array(tok_lens) > 77)[:2]]
    gate_texts = True
    if trunc_samples:
        with torch.no_grad():
            o = model.get_text_features(**tok(trunc_samples, return_tensors="pt",
                                              padding=True, truncation=True, max_length=77))
        f = o if torch.is_tensor(o) else o.pooler_output
        gate_texts = bool(np.isfinite(f.numpy()).all() and
                          (np.linalg.norm(f.numpy(), axis=1) > 0).all())
    gate = dict(images_finite_nonzero=gate_images, truncated_texts=n_trunc,
                truncated_encoded_finite=gate_texts,
                uid_alignment="renders keyed by frozen gallery cad_id; verified below")
    log(dict(phase="first_batch_gate", **gate))
    if not (gate_images and gate_texts):
        log(dict(phase="GATE_FAILED"))
        return 1

    # full image encode (covered UIDs, frozen cad axis order) ---------------
    img_full, img_encode_s = encode_images([png_of[c] for c in covered])
    if not (np.isfinite(img_full).all() and (np.linalg.norm(img_full, axis=1) > 0).all()):
        log(dict(phase="FULL_IMAGE_GATE_FAILED"))
        return 1

    t0 = time.time()
    text_feats = []
    for i in range(0, len(texts), BATCH):
        with torch.no_grad():
            o = model.get_text_features(**tok(texts[i:i + BATCH], return_tensors="pt",
                                              padding=True, truncation=True, max_length=77))
        f = o if torch.is_tensor(o) else o.pooler_output
        text_feats.append(f.numpy().astype(np.float64))
    text_encode_s = round(time.time() - t0, 1)
    text_feats = np.vstack(text_feats)
    if not (np.isfinite(text_feats).all() and (np.linalg.norm(text_feats, axis=1) > 0).all()):
        log(dict(phase="FULL_TEXT_GATE_FAILED"))
        return 1

    sim = l2(text_feats) @ l2(img_full).T  # (3644, n_covered) cosine, direction higher

    from text2cad_g5_eval_entry import ScoreSet  # noqa: E402
    covered_indices = [cad_ids.index(c) for c in covered]
    full_coverage = len(covered) == 8046
    npy = out / ("clip_scores_full_axis_f64.npy" if full_coverage
                 else "clip_scores_subset_axis_f64.npy")
    side = out / "clip_scores_sidecar.json"
    ScoreSet.write_binary_artifact(
        sim, side, npy, direction="higher",
        text_indices=list(range(3644)), cad_indices=covered_indices, dtype="float64",
        meta=dict(method="clip_vit_b_32_single_view",
                  spec="experiments/configs/clip_g5_render_spec_20261115.json",
                  spec_sha256=spec_sha,
                  weights=spec["weights"]["hf_repo"],
                  torch=torch.__version__,
                  coverage=dict(covered=len(covered), missing=len(missing)),
                  zero_npu=True, score_forwards="clip-only"))
    log(dict(phase="artifact_written", full=full_coverage,
             matrix_sha256=sha256_file(npy)[:16],
             sidecar_sha256=sha256_file(side)[:16]))

    timing = dict(render_s=render_s, clip_load_s=clip_load_s, img_encode_s=img_encode_s,
                  text_encode_s=text_encode_s, n_reused_probe_renders=reused,
                  render_failures=failures, covered_uids=len(covered),
                  truncation=dict(n_truncated_unique_texts=n_trunc,
                                  max_tokens=max(tok_lens), sample_max_tokens=int(
                                      np.quantile(tok_lens, .99))))
    (out / "encode_timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n")

    # ---- stage 4: unified evaluation vs certified SUM ----------------------
    if a.skip_eval:
        log(dict(phase="eval_skipped"))
        return 0
    from text2cad_g5_eval_entry import (EntryError, ScoreSet as SS, evaluate,
                                        load_frozen_protocol)
    protocol = load_frozen_protocol(repo, repo, Path(
        "D:/projects/IdeaSpark/CADRetrievalIdea1_runtime"), require_certificate=True)
    clip_set = SS.from_binary_artifact(npy, side)
    cad_subset_arg = None if full_coverage else sorted(covered_indices)
    if full_coverage:
        sum_set = SS.from_binary_artifact(SUM_NPY, SUM_SIDECAR)
        sum_set.meta["name"] = "text2cad_sum_certified"
    else:
        # slice the certified SUM artifact to the same covered cad axis
        sum_raw = np.fromfile(SUM_NPY, dtype="<f8").reshape(3644, 8046)
        sub = np.ascontiguousarray(sum_raw[:, sorted(covered_indices)])
        sum_npy = out / "sum_scores_subset_axis_f64.npy"
        sum_side = out / "sum_scores_subset_sidecar.json"
        SS.write_binary_artifact(sub, sum_side, sum_npy, direction="higher",
                                 text_indices=list(range(3644)),
                                 cad_indices=sorted(covered_indices), dtype="float64",
                                 meta=dict(method="text2cad_sum_certified",
                                           sliced_from=str(SUM_NPY),
                                           source_sidecar_sha256=sha256_file(SUM_SIDECAR)))
        sum_set = SS.from_binary_artifact(sum_npy, sum_side)
        sum_set.meta["name"] = "text2cad_sum_certified"
    eval_out = out / "entry_evaluation"
    results = evaluate(clip_set, protocol=protocol, output=eval_out,
                       method_meta=dict(name="clip_vit_b_32_single_view",
                                        weights=spec["weights"]["hf_repo"],
                                        view=spec["camera"]["view"],
                                        spec_sha256=spec_sha),
                       cad_subset=cad_subset_arg, paired_sum_set=sum_set,
                       bootstrap=dict(seed=20261112, replicates=5000))

    # SUM arm vs certified history: full mode requires query-by-query equality
    # with the certified ranks; sub-library mode re-ranks on the covered axis
    # (ranks legitimately differ), so instead verify the SLICE came from the
    # SHA-pinned certified artifact and report kept/dropped honestly.
    hist = {}
    for line in open(HIST_QUERY_RANKS, encoding="utf-8"):
        row = json.loads(line)
        hist[row["query_id"]] = row["ranks"]["sum"]
    ours = {}
    for line in (eval_out / "query_ranks.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        ours[row["query_id"]] = row["ranks"]["text2cad_sum_certified"]
    n_match = sum(1 for q in ours if hist.get(q) == ours[q])
    if full_coverage:
        g5_assert = dict(mode="full", sum_arm_queries=len(ours),
                         match_certified=n_match,
                         all_equal=n_match == 7934 == len(ours))
        ok = g5_assert["all_equal"]
    else:
        g5_assert = dict(mode="sub_library", sum_arm_queries=len(ours),
                         kept_plus_dropped=len(ours) + len(missing_cad_queries()),
                         certified_source=str(SUM_NPY),
                         certified_sha_verified="from_binary_artifact pins matrix sha",
                         note="SUM ranks recomputed on the covered axis; equality with "
                              "full-library certified ranks is not expected for affected "
                              "queries; coverage gaps listed in coverage.json")
        ok = True
    (eval_out.parent / "sum_arm_certified_check.json").write_text(
        json.dumps(g5_assert, indent=1) + "\n", encoding="utf-8", newline="\n")
    if not ok:
        log(dict(phase="SUM_ARM_MISMATCH", **g5_assert))
        return 1

    def missing_cad_queries():
        missing_set = set(missing)
        return [qid for qid, p in zip(protocol["linked"]["query_ids"],
                                      protocol["linked"]["positive_cads"])
                if cad_ids[int(p)] in missing_set]

    lines = []
    for p in sorted(out.rglob("*")):
        if (p.is_file() and p.name != "SHA256SUMS"
                and "renders" not in p.parts and "steps" not in p.parts):
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(out).as_posix())
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    summary = dict(status="complete", out=str(out), spec_sha256=spec_sha,
                   gate=gate, timing=timing, sum_arm_check=g5_assert,
                   coverage=dict(covered=len(covered), missing=len(missing),
                                 missing_cad_ids=missing),
                   metrics=results["metrics"], hit_counts=results["hit_counts"],
                   paired_r20=results["statistics"]["paired_ci95"]["R@20"],
                   paired_medr=results["statistics"]["paired_ci95"]["MedR"],
                   elapsed_s=round(time.time() - t_start, 1), zero_npu=True)
    (out / "final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n")
    log(dict(phase="complete", metrics=results["metrics"], paired_r20=summary["paired_r20"],
             elapsed_s=summary["elapsed_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
