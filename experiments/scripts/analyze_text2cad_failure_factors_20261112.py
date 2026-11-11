#!/usr/bin/env python
"""Exploratory failure-factor diagnostics on the frozen G4 Text2CAD matrix; no model forwards.

GLM 2026-11-12. Authorized by GLM under the 2026-11-11 takeover governance (local CPU,
zero NPU, zero scored forwards, reuse of the certified G4 matrix). This script and its
variant definitions below were written BEFORE looking at any new variant result; all
outputs are exploratory diagnostics and NOT confirmatory evidence. Method selection on
this (already-analyzed) test matrix must never be presented as validated.

Frozen readout variants (defined prior to execution):
  v0_sum                raw two_head_factorized_logprob_sum score_total (G5 primary)
  v1_mean               mean_total (G5 preregistered length diagnostic)
  v2_additive_residual  two-way additive fit M ~= grand + row_t + col_c (alternating
                        means, 25 iterations); rank by residual within each text row
  v3_halfsplit_marginal split the 3644 unique texts into halves A/B (numpy default_rng
                        20261112 permutation); candidate baseline b_X(c) = mean score of
                        candidate c over half-X texts; queries of half-A texts are scored
                        M[t,c] - b_B(c) and vice versa (out-of-half baselines); pooled
                        query-equal-weight metrics
  v5_length_residual    per-text-row OLS fit score ~ a_t + b_t * len_c (len = pinned
                        n_valid_x+n_valid_y); rank by residual

Binding checks before any computation: frozen input SHA pins (G5 frozen config),
G4 chunk SHA256SUMS coverage, per-chunk header binding (git sha 56ac0f8e..., checkpoint
235b137e..., pair manifest 62086beb..., score name, float32, 8046/8046 scored), and
restoration of all 7934 formal G5 positive ranks for v0_sum.

Outputs (out-dir, must be outside the Git checkout): results.json, positive ranks
.npy per variant, SHA256SUMS. Console summary. Exit non-zero on any binding failure.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "experiments" / "configs" / "text2cad_g5_cpu_frozen_20261003.json"
RAW = REPO / "experiments" / "runs" / "idea01_text2cad_g4_full_20261002"
G5 = REPO / "experiments" / "runs" / "idea01_text2cad_g5_cpu_20261003"
SEED = 20261112
N_TEXT, N_CAD = 3644, 8046
EXPECTED_GIT = "56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719"
EXPECTED_CKPT = "235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598"
EXPECTED_MANIFEST = "62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557"
BINDING_REFS = {  # previously certified values the new computation must reproduce
    "ss_share_after_row_removal_sum": 0.996824,
    "col_effect_len_pearson_sum": -0.757089,
    "strict_win_query_mean": 0.87959,
}
VARIANTS = ("v0_sum", "v1_mean", "v2_additive_residual", "v3_halfsplit_marginal", "v5_length_residual")


def die(msg):
    print(f"[faildiag] FATAL: {msg}", flush=True)
    sys.exit(2)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


RUNTIME = Path(r"D:\projects\IdeaSpark\CADRetrievalIdea1_runtime")


def pinned_path(spec):
    root = {"data": REPO, "runtime": RUNTIME}[spec["root"]]
    return root / spec["path"]


def check_pins(cfg):
    for name, spec in cfg["inputs"].items():
        got = digest(pinned_path(spec))
        if got != spec["sha256"]:
            die(f"input pin mismatch {name}: {got}")
    print("[faildiag] all 8 frozen input pins OK", flush=True)


def manifest_entries(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        a, b = parts
        if len(a) == 64 and a == a.lower() and all(c in "0123456789abcdef" for c in a):
            h, rel = a, b
        else:
            h, rel = b, a
        rel = rel.lstrip("./")
        out[rel] = h
    return out


def link_inputs():
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    mapping = json.loads((pinned_path(cfg["inputs"]["mapping"])).read_text(encoding="utf-8"))
    queries = jsonl(pinned_path(cfg["inputs"]["queries"]))
    qrels = {r["query_id"]: r for r in jsonl(pinned_path(cfg["inputs"]["qrels"]))}
    texts = jsonl(pinned_path(cfg["inputs"]["texts"]))
    n_tx = len({r["text_index"] for r in texts})
    if n_tx != N_TEXT:
        die(f"text count {n_tx} != {N_TEXT}")
    cad_ids = mapping["cad_index_to_cad_id"]
    if len(cad_ids) != N_CAD or len(set(cad_ids)) != N_CAD:
        die("cad mapping size")
    cad_index = {cid: i for i, cid in enumerate(cad_ids)}
    q2t = mapping["query_to_text"]
    groups, positives, qids = [], [], []
    targets = [set() for _ in range(N_TEXT)]
    for row in queries:
        qid = row["query_id"]
        ti = q2t[qid]
        if not (type(ti) is int and 0 <= ti < N_TEXT):
            die("bad query mapping")
        rel = qrels[qid]
        if rel["relevance"] != 1 or rel["cad_id"] not in cad_index:
            die("bad positive")
        ci = cad_index[rel["cad_id"]]
        if ci in targets[ti]:
            die("duplicate positive in text group")
        targets[ti].add(ci)
        groups.append(ti)
        positives.append(ci)
        qids.append(qid)
    sizes = np.bincount(groups, minlength=N_TEXT)
    if not (sizes > 0).all():
        die("unqueried text group")
    lengths = np.zeros(N_CAD, dtype=np.int32)
    seen = set()
    for r in jsonl(pinned_path(cfg["inputs"]["lengths"])):
        i = r["cad_index"]
        if i in seen:
            die("dup length row")
        seen.add(i)
        lengths[i] = r["n_valid_x"] + r["n_valid_y"]
    if seen != set(range(N_CAD)) or not (lengths > 0).all():
        die("length coverage")
    print(f"[faildiag] inputs linked: {len(qids)} queries, {N_TEXT} texts, {N_CAD} cads", flush=True)
    return (np.asarray(groups, np.int32), np.asarray(positives, np.int32), qids,
            sizes, lengths, cad_ids)


def load_matrix():
    entries = manifest_entries(RAW / "SHA256SUMS")
    scores = np.full((N_TEXT, N_CAD), np.nan, dtype=np.float32)
    means = np.full((N_TEXT, N_CAD), np.nan, dtype=np.float32)
    lens = np.zeros((N_TEXT, N_CAD), dtype=np.int16)
    n_chunks = 0
    t0 = time.time()
    for card in ("card0", "card1"):
        for f in sorted((RAW / card).glob("chunk_*.jsonl")):
            rel = f"{card}/{f.name}"
            want = entries.get(rel)
            if want is None:
                die(f"chunk not in manifest: {rel}")
            if digest(f) != want:
                die(f"chunk hash mismatch: {rel}")
            with open(f, "r", encoding="utf-8") as fh:
                header = json.loads(fh.readline())
                if header.get("record_type") != "chunk_header":
                    die(f"bad header {rel}")
                if (header["git_sha"] != EXPECTED_GIT or header["checkpoint_sha256"] != EXPECTED_CKPT
                        or header["pair_manifest_sha256"] != EXPECTED_MANIFEST
                        or header["score_name"] != "two_head_factorized_logprob_sum"
                        or header["torch_dtype"] != "float32" or header["n_scored"] != N_CAD
                        or header["n_rejected"] != 0):
                    die(f"chunk header binding differs: {rel}")
                tr = header["text_range"]
                if tr[1] - tr[0] != 1:
                    die("multi-text chunk unexpected")
                t = tr[0]
                rows = [json.loads(ln) for ln in fh]
            if len(rows) != N_CAD:
                die(f"chunk row count {rel}")
            st = np.fromiter((r["score_total"] for r in rows), dtype=np.float64, count=N_CAD)
            mt = np.fromiter((r["mean_total"] for r in rows), dtype=np.float64, count=N_CAD)
            ln = np.fromiter((r["n_valid_total"] for r in rows), dtype=np.int32, count=N_CAD)
            scores[t] = st.astype(np.float32)
            means[t] = mt.astype(np.float32)
            lens[t] = ln.astype(np.int16)
            n_chunks += 1
            if n_chunks % 500 == 0:
                print(f"[faildiag] loaded {n_chunks} chunks ({time.time()-t0:.0f}s)", flush=True)
    if n_chunks != N_TEXT:
        die(f"loaded {n_chunks} chunks, expected {N_TEXT}")
    if not np.isfinite(scores).all():
        die("non-finite scores in matrix")
    print(f"[faildiag] matrix loaded in {time.time()-t0:.0f}s", flush=True)
    return scores, means, lens.astype(np.int32)


def rank_rows(S):
    order = np.argsort(-S, axis=1, kind="stable")
    ranks = np.empty(S.shape, dtype=np.int32)
    np.put_along_axis(ranks, order, np.arange(1, S.shape[1] + 1, dtype=np.int32), axis=1)
    return ranks


def metrics(ranks, groups, positives, lengths):
    r = ranks[groups, positives].astype(np.float64)
    out = {}
    for k in (1, 5, 10, 20):
        out[f"R@{k}"] = float(np.mean(r <= k) * 100.0)
    out["MedR"] = float(np.median(r))
    out["MeanR"] = float(np.mean(r))
    bounds = np.searchsorted(np.array([27, 40, 70]), lengths[positives], side="left")
    names = ["len_q1_le27", "len_q2_28_40", "len_q3_41_70", "len_q4_gt70"]
    for qi, name in enumerate(names):
        m = bounds == qi
        out[f"R@20_{name}"] = float(np.mean(r[m] <= 20) * 100.0) if m.any() else None
    uniq_sizes = np.bincount(groups, minlength=N_TEXT)
    is_unique = uniq_sizes[groups] == 1
    out["R@20_unique"] = float(np.mean(r[is_unique] <= 20) * 100.0)
    out["R@20_repeated"] = float(np.mean(r[~is_unique] <= 20) * 100.0)
    out["n"] = int(len(r))
    return out, r


def additive_residual(S, iters=25):
    """Balanced two-way additive fit by alternating means (complete matrix => converges
    to the least-squares decomposition M = grand + row_t + col_c + resid)."""
    grand = float(S.mean())
    A = S.astype(np.float64)
    row = A.mean(axis=1) - grand
    col = np.zeros(A.shape[1], dtype=np.float64)
    for _ in range(iters):
        col = (A - grand - row[:, None]).mean(axis=0)
        row = (A - grand - col[None, :]).mean(axis=1)
    resid = A - grand - row[:, None] - col[None, :]
    return resid.astype(np.float32), row, col, grand


def length_residual(S, lengths):
    A = S.astype(np.float64)
    x = lengths.astype(np.float64)
    n = A.shape[1]
    sx = x.sum()
    sxx = float((x * x).sum())
    sy = A.sum(axis=1)
    sxy = A @ x
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    resid = A - (a[:, None] + b[:, None] * x[None, :])
    return resid.astype(np.float32), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[faildiag] python={platform.python_version()} numpy={np.__version__}", flush=True)

    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    check_pins(cfg)
    groups, positives, qids, sizes, lengths, cad_ids = link_inputs()
    scores, means, lens = load_matrix()
    if not np.array_equal(lens, np.broadcast_to(lengths[None, :], lens.shape)):
        die("chunk n_valid_total differs from pinned lengths")

    # bind to formal G5 v0 ranks before anything new
    old = {r["query_id"]: r for r in jsonl(G5 / "query_ranks.jsonl")}
    if len(old) != 7934:
        die("formal rank coverage")
    sum_ranks = rank_rows(scores)
    v0_rank_of_pos = sum_ranks[groups, positives]
    bad = [q for i, q in enumerate(qids)
           if old[q]["ranks"]["sum"] != int(v0_rank_of_pos[i])]
    if bad:
        die(f"v0 ranks differ from formal G5 for {len(bad)} queries, e.g. {bad[:3]}")
    print("[faildiag] v0_sum ranks identical to formal G5 query_ranks (7934/7934)", flush=True)

    # common-effect diagnostics (bind to 10-04 certified values)
    A = scores.astype(np.float64)
    resid2, row_eff, col_eff, grand = additive_residual(scores)
    ss_col = float((col_eff ** 2).sum() * N_TEXT)
    ss_resid = float((resid2 ** 2).sum())
    ss_share = ss_col / (ss_col + ss_resid)
    col_len_r = float(np.corrcoef(col_eff, lengths.astype(np.float64))[0, 1])
    # strict win fraction per query: own text's score for the positive cad, vs every
    # text row's score for the same cad; denominator = all 3644 text rows (own row is
    # never a strict win of itself).
    own = A[groups, positives]
    greater = (A[:, positives] < own[None, :]).sum(axis=0)  # per query
    strict_win_fraction = greater / N_TEXT
    strict_win_query_mean = float(strict_win_fraction.mean())
    print(f"[faildiag] bindings: ss_share={ss_share:.6f} (ref {BINDING_REFS['ss_share_after_row_removal_sum']}), "
          f"col_len_r={col_len_r:.6f} (ref {BINDING_REFS['col_effect_len_pearson_sum']}), "
          f"strict_win_mean={strict_win_query_mean:.6f} (ref {BINDING_REFS['strict_win_query_mean']})", flush=True)
    if (abs(ss_share - BINDING_REFS["ss_share_after_row_removal_sum"]) > 5e-5
            or abs(col_len_r - BINDING_REFS["col_effect_len_pearson_sum"]) > 5e-3
            or abs(strict_win_query_mean - BINDING_REFS["strict_win_query_mean"]) > 5e-4):
        die("common-effect bindings differ from certified 10-04 diagnostics")

    # frozen variants
    results = {"exploratory": True, "seed": SEED,
               "binding": {"git_sha": EXPECTED_GIT, "checkpoint_sha256": EXPECTED_CKPT,
                           "pair_manifest_sha256": EXPECTED_MANIFEST,
                           "formal_v0_ranks_restored": True, "n_queries": len(qids)},
               "common_effects": {"ss_share_after_row_removal": ss_share,
                                   "col_effect_len_pearson": col_len_r,
                                   "strict_win_query_mean": strict_win_query_mean},
               "variants": {}}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_TEXT)
    halfA = np.zeros(N_TEXT, dtype=bool)
    halfA[perm[: N_TEXT // 2]] = True
    bA = A[halfA].mean(axis=0)
    bB = A[~halfA].mean(axis=0)
    S_v3 = scores - np.where(halfA[:, None], bB[None, :], bA[None, :]).astype(np.float32)
    S_v5, b5 = length_residual(scores, lengths)

    variant_scores = {
        "v0_sum": scores,
        "v1_mean": means,
        "v2_additive_residual": resid2,
        "v3_halfsplit_marginal": S_v3,
        "v5_length_residual": S_v5,
    }
    v0_r = v0_rank_of_pos.astype(np.float64)
    for name in VARIANTS:
        t0 = time.time()
        S = variant_scores[name]
        ranks = rank_rows(S)
        m, r = metrics(ranks, groups, positives, lengths)
        keep = {
            "metrics": {k: v for k, v in m.items()},
            "elapsed_rank_s": round(time.time() - t0, 1),
        }
        if name != "v0_sum":
            d = r - v0_r
            keep["migration_vs_v0"] = {
                "median_rank_delta": float(np.median(d)),
                "frac_improved": float(np.mean(d < 0)),
                "frac_worsened": float(np.mean(d > 0)),
                "top20_v0_keep_top20": float(np.mean(r[v0_r <= 20] <= 20)),
                "n_pos_new_top20": int((r <= 20).sum()),
            }
        results["variants"][name] = keep
        np.save(out / f"positive_ranks_{name}.npy", r.astype(np.int32))
        print(f"[faildiag] {name}: R@20={m['R@20']:.4f}% MedR={m['MedR']:.1f} "
              f"R@20_lenQ1={m['R@20_len_q1_le27']:.3f} R@20_lenQ4={m['R@20_len_q4_gt70']:.3f} "
              f"({keep['elapsed_rank_s']}s)", flush=True)

    results["col_effect_top"] = {
        "highest_baseline_cads": [
            {"cad_index": int(i), "cad_id": cad_ids[i], "col_effect": float(col_eff[i]),
             "length": int(lengths[i])}
            for i in np.argsort(-col_eff)[:10]
        ],
        "lowest_baseline_cads": [
            {"cad_index": int(i), "cad_id": cad_ids[i], "col_effect": float(col_eff[i]),
             "length": int(lengths[i])}
            for i in np.argsort(col_eff)[:10]
        ],
    }
    with open(out / "results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(f"[faildiag] wrote {out/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
