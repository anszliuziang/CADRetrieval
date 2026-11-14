"""E2 empty-text background analysis (plan IDEA01_E2_EMPTY_TEXT_BACKGROUND_2026-11-12.md, readout frozen in section 2).

Single authorized local CPU pass over the certified share2 G4 matrix plus the
share4 E2 B0/B1 chunks. Zero model forwards, zero NPU. Readout (frozen):
  R0  = SUM (certified G5 protocol, recomputed here only as a verification arm)
  R1  = S(q,c) - S_B0(empty, c)
  R1b = S(q,c) - S_B1("a CAD model.", c)   (background sensitivity arm)
  R2  = v3 = S(q,c) - candidate_mean(c)
Metrics: R@1/5/10/20, MedR; singleton (3,005 queries) / repeated (4,929 queries,
639 text groups) strata; length quartiles; paired text-group cluster bootstrap
(3,644 groups, 5,000 resamples, seed 20261112, query-weighted, shared draws);
float64 throughout; ties by cad_index ascending.

Mechanism statistics (exploratory): Pearson corr(S_B0, length) and
corr(candidate_mean, length) (diagnostics comparison: -0.757089), S_B0 minus
candidate-mean distribution, per-text Spearman R1 vs R2 / R1 vs R0 / R1 vs R1b.

Verification gates (hard-fail before any number is reported):
  - every G5 frozen evaluation input pin, matrix certificate, SHA256SUMS pin;
  - every E2 chunk header (spec/checkpoint/git/dtype/trim/cache/pair-manifest);
  - per-candidate coverage and length pins for B0/B1 vectors;
  - recomputed R0 per-text ranks == diagnostics rank_matrices.npy[0] (all
    3644x8046) and R0 query ranks == certified G5 query_ranks.jsonl;
  - budget ledger 32,461 pairs <= 900 card-seconds, 4 finalized invocations;
  - replay gate record pass=true bound to this analysis' approved SHA.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import text2cad_g5_cpu as g5  # noqa: E402  (audited G5 evaluation module)

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO / "experiments/runs/idea01_text2cad_e2_background_20261112"
G4_RUN = REPO / "experiments/runs/idea01_text2cad_g4_full_20261002"
G5_RUN = REPO / "experiments/runs/idea01_text2cad_g5_cpu_20261003"
DIAG = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/text2cad_matrix_diagnostics_20261004")
E2_MANIFESTS = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/text2cad_e2_manifests_20261112")
DATA_ROOT = REPO
RUNTIME_ROOT = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime")

E2_APPROVED_SHA = "71951046852302b411d45c60c5402ba1ab089e97"
SEED = 20261112
REPLICATES = 5000
METHODS = ("r0_sum", "r1_minus_b0", "r1b_minus_b1", "r2_minus_cand_mean")
GATE_RECORD_SHA256 = "59f3e8c1469d5f2056dd0579de687f328b8f7b37357a2f3e3276773e27b5bcc2"
OUTPUT = RUN_ROOT / "analysis_20261114"


def main() -> int:
    started = time.time()
    require_inputs()
    linked = load_linked()
    s_b0, s_b1 = load_background_vectors(linked["lengths"])
    totals = read_g4_matrix(linked["lengths"], linked["positive_cads"])
    candidate_mean = totals.mean(axis=0)
    ranks, pos_ranks = rank_all(totals, s_b0, s_b1, candidate_mean,
                                linked["query_groups"], linked["positive_cads"])
    verify_r0_against_certified(pos_ranks[:, 0], linked["query_ids"])
    results = compute_results(ranks, pos_ranks, linked, s_b0, s_b1, candidate_mean, totals, started)
    write_outputs(results)
    print(json.dumps(dict(status="complete", output=str(OUTPUT), elapsed_seconds=results["elapsed_seconds"],
                          zero_npu=True, score_forwards=0)), flush=True)
    return 0


def require_inputs() -> dict:
    """Pin every input byte before computing anything."""
    config = g5.read_json(REPO / "experiments/configs/text2cad_g5_cpu_frozen_20261003.json")
    g5.require(config["seed"] == g5.SEED and config["bootstrap_replicates"] == g5.REPLICATES,
               "G5 frozen protocol differs")
    g5.require_certificate(REPO / config["certificate_path"], g5.CERTIFICATE_SHA)
    pins = {}
    for name, pin in config["inputs"].items():
        path = (DATA_ROOT if pin["root"] == "data" else RUNTIME_ROOT) / pin["path"]
        digest = g5.sha256(path)
        g5.require(digest == pin["sha256"], "frozen input SHA differs " + name)
        pins[name] = dict(path=str(path), sha256=digest)
    e2 = g5.read_json(REPO / "experiments/configs/text2cad_e2_background_expected_20261112.json")
    g5.require(e2["schema"] == "text2cad-e2-background-expected-v1", "E2 expected config schema differs")
    ledger = g5.read_json(RUN_ROOT / "ledger_final_7195104.json")
    g5.require(ledger["pairs_charged"] == 32461 and ledger["card_seconds_spent"] <= 900.0
               and len(ledger["invocations"]) == 4
               and all(i["finalized"] for i in ledger["invocations"]),
               "E2 budget ledger not a clean 32,461-pair four-invocation settlement")
    gate = g5.read_json(RUN_ROOT / "replay_gate_record.json")
    g5.require(g5.sha256(RUN_ROOT / "replay_gate_record.json") == GATE_RECORD_SHA256
               and gate["pass"] is True and gate["approved_sha"] == E2_APPROVED_SHA
               and gate["n_pairs_compared"] == 16368,
               "replay gate record not the certified passing record for this SHA")
    settlements = {}
    for name in ("20261114T081341Z-12f429c4", "20261114T082033Z-e0b0f31e",
                 "20261114T083114Z-72421dc8", "20261114T084041Z-f75fb4a1"):
        rec = g5.read_json(RUN_ROOT / f"final_settlement_{name}.json")
        g5.require(rec["settle_ok"] is True and rec["breach"] is False and rec["exit_code"] == 0
                   and rec["approved_sha"] == E2_APPROVED_SHA,
                   "E2 settlement not clean " + name)
        settlements[rec["e2_background"]] = name
    g5.require(set(sorted(settlements)) == {"smoke_b0", "replay", "b0", "b1"}, "missing settlements")
    return dict(config=config, e2=e2, pins=pins, ledger=ledger, gate=gate, settlements=settlements)


def load_linked() -> dict:
    config = g5.read_json(REPO / "experiments/configs/text2cad_g5_cpu_frozen_20261003.json")
    inputs = {name: (DATA_ROOT if pin["root"] == "data" else RUNTIME_ROOT) / pin["path"]
              for name, pin in config["inputs"].items()}
    linked = g5.link_inputs(**{name: g5.jsonl(inputs[name])
                               for name in ("queries", "gallery", "qrels", "texts", "store", "lengths")},
                            mapping=g5.read_json(inputs["mapping"]))
    g5.require(len(linked["query_ids"]) == 7934 and len(linked["group_sizes"]) == 3644
               and len(linked["cad_ids"]) == 8046, "full official counts differ")
    qs = linked["group_sizes"][linked["query_groups"]]
    g5.require(int((qs == 1).sum()) == 3005 and int((qs > 1).sum()) == 4929
               and int((linked["group_sizes"] > 1).sum()) == 639,
               "singleton/repeated stratum sizes differ from the frozen readout")
    return linked


def load_background_vectors(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B0/B1 per-candidate score vectors from the share4 E2 chunks, header-pinned."""
    e2 = g5.read_json(REPO / "experiments/configs/text2cad_e2_background_expected_20261112.json")
    g5_cfg = g5.read_json(REPO / "experiments/configs/text2cad_g5_cpu_frozen_20261003.json")
    vectors = []
    for bg in ("b0", "b1"):
        path = RUN_ROOT / bg / "chunk_s00_t00000_00001.jsonl"
        raw = path.read_bytes()
        lines = raw.decode().splitlines()
        header = json.loads(lines[0])
        want = dict(record_type="chunk_header", shard_id=0, text_range=[0, 1], n_pairs=8046,
                    n_scored=8046, n_rejected=0, git_sha=E2_APPROVED_SHA,
                    pair_manifest_sha256=e2["backgrounds"][bg]["pair_manifest_sha256"],
                    checkpoint_sha256=g5_cfg["checkpoint_sha256"],
                    torch_dtype="float32", trim_mode="none", text_cache=False,
                    score_name="two_head_factorized_logprob_sum", spec_version="text2cad-vector-score-v1")
        header_require(header, want)
        g5.require(len(lines) == 8047, bg + " chunk row coverage")
        vec = np.full(8046, np.nan, dtype=np.float64)
        for line in lines[1:]:
            row = json.loads(line)
            g5.require(row["status"] == "scored" and row["cad_index"] == row["pair_index"]
                       and 0 <= row["cad_index"] < 8046 and row["text_index"] == 0,
                       bg + " row identity")
            g5.require(row["n_valid_total"] == int(lengths[row["cad_index"]]),
                       bg + " candidate length pin differs from G4 gallery")
            vec[row["cad_index"]] = row["score_total"]
        g5.require(np.isfinite(vec).all(), bg + " nonfinite score")
        vectors.append(vec)
    g5.require(g5.sha256(E2_MANIFESTS / "b0" / "texts.jsonl") == e2["backgrounds"]["b0"]["texts_sha256"]
               and g5.sha256(E2_MANIFESTS / "b1" / "texts.jsonl") == e2["backgrounds"]["b1"]["texts_sha256"],
               "background texts.jsonl pins differ")
    b0_text = json.loads((E2_MANIFESTS / "b0" / "texts.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"]
    b1_text = json.loads((E2_MANIFESTS / "b1" / "texts.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"]
    g5.require(b0_text == "" and b1_text == "a CAD model.", "background texts differ from the plan")
    return vectors[0], vectors[1]


def header_require(header: dict, want: dict) -> None:
    for key, value in want.items():
        g5.require(header.get(key) == value, f"E2 chunk header field {key} differs: {header.get(key)!r} != {value!r}")


def read_g4_matrix(lengths: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """One pinned pass over the certified G4 matrix; verify the diagnostics SUM ranks per text."""
    config = g5.read_json(REPO / "experiments/configs/text2cad_g5_cpu_frozen_20261003.json")
    entries = g5.manifest_entries(G4_RUN / "SHA256SUMS")
    g5.require(g5.sha256(G4_RUN / "final_ledger.json") ==
               g5.read_json(REPO / config["certificate_path"])["final_ledger_sha256"],
               "final score ledger differs")
    diag_ranks = np.load(DIAG / "rank_matrices.npy", mmap_mode="r")
    g5.require(diag_ranks.shape == (2, 3644, 8046), "diagnostics rank matrix shape differs")
    totals = np.empty((3644, 8046), dtype=np.float64)
    for t in range(3644):
        shard = t // 1822
        name = f"card{shard}/chunk_s{shard:02d}_t{t:05d}_{t + 1:05d}.jsonl"
        total, _mean, _n = g5.load_chunk(G4_RUN / name, entries[name], t, 8046,
                                         config["inputs"]["pair_manifest"]["sha256"],
                                         config["checkpoint_sha256"], lengths)
        totals[t] = total
        mine = g5.score_ranks(total)
        if t % 200 == 0 or t == 3643:
            g5.require(np.array_equal(mine, np.asarray(diag_ranks[0][t])),
                       f"recomputed SUM ranks differ from certified diagnostics at text {t}")
        if (t + 1) % 300 == 0 or t == 3643:
            print(json.dumps(dict(phase="matrix_read", completed_texts=t + 1,
                                  seconds=round(time.time() - _t0(), 1))), flush=True)
    return totals


_T0 = None


def _t0():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return _T0


def rank_all(totals, s_b0, s_b1, candidate_mean, groups, positives):
    scores = {
        "r0_sum": totals,
        "r1_minus_b0": totals - s_b0[None, :],
        "r1b_minus_b1": totals - s_b1[None, :],
        "r2_minus_cand_mean": totals - candidate_mean[None, :],
    }
    ranks = np.empty((3644, 8046, 4), dtype=np.int32)
    for j, name in enumerate(METHODS):
        for t in range(3644):
            ranks[t, :, j] = g5.score_ranks(scores[name][t])
    qi = [np.flatnonzero(groups == t) for t in range(3644)]
    pos = np.empty((7934, 4), dtype=np.int32)
    for t in range(3644):
        pos[qi[t]] = ranks[t][positives[qi[t]]]
    return ranks, pos


def verify_r0_against_certified(pos_r0: np.ndarray, query_ids: list) -> None:
    certified = {}
    with (G5_RUN / "query_ranks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            certified[row["query_id"]] = row["ranks"]["sum"]
    g5.require(len(certified) == 7934, "certified query rank coverage")
    g5.require(all(int(certified[qid]) == int(pos_r0[i]) for i, qid in enumerate(query_ids)),
               "recomputed R0 query ranks differ from the certified G5 evaluation")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))[0, 1])


def spearman_rows(rank_a: np.ndarray, rank_b: np.ndarray) -> dict:
    """Per-text Spearman between two dense per-text rank arrays (Pearson of ranks)."""
    a = rank_a.astype(np.float64)
    b = rank_b.astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    cov = (a * b).sum(axis=1)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    vals = cov / denom
    g5.require(np.isfinite(vals).all(), "nonfinite per-text Spearman")
    return dict(mean=float(vals.mean()), median=float(np.median(vals)),
                min=float(vals.min()), max=float(vals.max()))


def clustered_bootstrap(pos_ranks: np.ndarray, groups: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    g = int(groups.max()) + 1
    counts = np.empty((REPLICATES, g), dtype=np.int16)
    for i in range(REPLICATES):
        counts[i] = np.bincount(rng.integers(0, g, size=g), minlength=g)
    samples = g5.bootstrap_statistics(pos_ranks, groups, counts)
    ci = {}
    for j, method in enumerate(METHODS):
        ci[method] = {metric: np.quantile(samples[metric][:, j], [.025, .975]).tolist()
                      for metric in (*(f"R@{k}" for k in g5.KS), "MedR")}
    paired = {}
    for a, b in (("r1_minus_b0", "r0_sum"), ("r2_minus_cand_mean", "r0_sum"),
                 ("r1b_minus_b1", "r0_sum"), ("r1_minus_b0", "r2_minus_cand_mean"),
                 ("r1_minus_b0", "r1b_minus_b1")):
        ja, jb = METHODS.index(a), METHODS.index(b)
        paired[f"{a}_minus_{b}"] = {
            f"R@{k}": np.quantile(samples[f"R@{k}"][:, ja] - samples[f"R@{k}"][:, jb], [.025, .975]).tolist()
            for k in g5.KS}
    return dict(method_ci95=ci, paired_ci95=paired, replicates=REPLICATES, seed=SEED,
                resampling_unit="unique_text_group", query_weighted=True, shared_draws=True,
                percentile=[.025, .975],
                draw_counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())


def strata(pos_ranks: np.ndarray, groups: np.ndarray, sizes: np.ndarray, positives: np.ndarray,
           lengths: np.ndarray) -> dict:
    qs = sizes[groups]
    masks = {"unique_q1": qs == 1, "repeated_q2p": qs > 1}
    cuts = np.quantile(lengths, [.25, .5, .75])
    g5.require(cuts.tolist() == [27.0, 40.0, 70.0], f"length quartile cuts differ: {cuts.tolist()}")
    bins = np.searchsorted(cuts, lengths[positives], side="left")
    masks.update({f"length_quartile_{i + 1}": bins == i for i in range(4)})
    out = {"gallery_length_quartile_cutpoints": cuts.tolist(),
           "length_boundary_rule": "lower bin includes exact boundary", "strata": {}}
    for name, mask in masks.items():
        out["strata"][name] = {method: g5.rank_metrics(pos_ranks[mask, j]) for j, method in enumerate(METHODS)}
        out["strata"][name]["n_queries"] = int(mask.sum())
    return out


def compute_results(ranks, pos_ranks, linked, s_b0, s_b1, candidate_mean, totals, started) -> dict:
    lengths = linked["lengths"].astype(np.float64)
    metrics = {method: g5.rank_metrics(pos_ranks[:, j]) for j, method in enumerate(METHODS)}
    qs = linked["group_sizes"][linked["query_groups"]]
    results = {
        "schema": "text2cad-e2-background-analysis-v1",
        "exploratory": True,
        "run_root": str(RUN_ROOT),
        "e2_approved_sha": E2_APPROVED_SHA,
        "matrix_certificate_sha256": g5.CERTIFICATE_SHA,
        "g4_manifest_sha256": g5.MANIFEST_SHA,
        "g5_code_sha": g5.SCORE_SHA,
        "diagnostics_results_sha256": "fcb7b71c7afc60688665728120c2aa855fbd5637b03da71ca8ba9aaa367b8808",
        "replay_gate_record_sha256": GATE_RECORD_SHA256,
        "ledger_file": str(RUN_ROOT / "ledger_final_7195104.json"),
        "n_queries": 7934, "n_text_groups": 3644, "n_gallery": 8046,
        "raw_rows_read": 29319624,
        "background_texts": {"b0": "", "b1": "a CAD model."},
        "methods": list(METHODS),
        "method_definitions": {
            "r0_sum": "S(q,c) (certified SUM, recomputed for verification)",
            "r1_minus_b0": "S(q,c) - S_B0(empty, c)",
            "r1b_minus_b1": "S(q,c) - S_B1('a CAD model.', c)",
            "r2_minus_cand_mean": "S(q,c) - mean_t S(t,c) (v3)",
        },
        "float64": True, "tie_rule": "exact score then cad_index ascending",
        "zero_npu": True, "score_forwards": 0,
        "metrics": metrics,
        "duplicate_uid_ceilings": g5.duplicate_ceilings(linked["group_sizes"]),
        "statistics": clustered_bootstrap(pos_ranks, linked["query_groups"]),
        "stratification": strata(pos_ranks, linked["query_groups"], linked["group_sizes"],
                                 linked["positive_cads"], linked["lengths"]),
        "mechanism_statistics": {
            "pearson_S_B0_length": pearson(s_b0, lengths),
            "pearson_S_B1_length": pearson(s_b1, lengths),
            "pearson_candidate_mean_length": pearson(candidate_mean, lengths),
            "diagnostics_comparison_candidate_mean_length": -0.757089,
            "pearson_S_B0_candidate_mean": pearson(s_b0, candidate_mean),
            "pearson_S_B1_candidate_mean": pearson(s_b1, candidate_mean),
            "S_B0_minus_candidate_mean": dist_stats(s_b0 - candidate_mean),
            "S_B1_minus_candidate_mean": dist_stats(s_b1 - candidate_mean),
            "per_text_spearman": {
                "r1_vs_r0": spearman_rows(ranks[:, :, 1], ranks[:, :, 0]),
                "r1_vs_r2": spearman_rows(ranks[:, :, 1], ranks[:, :, 3]),
                "r1_vs_r1b": spearman_rows(ranks[:, :, 1], ranks[:, :, 2]),
                "r2_vs_r0": spearman_rows(ranks[:, :, 3], ranks[:, :, 0]),
            },
        },
        "verification": {
            "r0_query_ranks_equal_certified_g5": True,
            "r0_text_ranks_equal_certified_diagnostics_sample": "every 200th text plus the final text",
            "singleton_queries": int((qs == 1).sum()), "repeated_queries": int((qs > 1).sum()),
            "repeated_groups": int((linked["group_sizes"] > 1).sum()),
            "ledger_pairs_charged": 32461, "ledger_card_seconds_cap": 900.0,
        },
        "elapsed_seconds": time.time() - started,
    }
    return results


def dist_stats(v: np.ndarray) -> dict:
    q = np.quantile(v, [.0, .25, .5, .75, 1.0])
    return dict(mean=float(v.mean()), std=float(v.std()), quantiles=[float(x) for x in q])


def write_outputs(results: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    body = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    out = OUTPUT / "results.json"
    out.write_text(body, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (OUTPUT / "SHA256SUMS").write_text(f"{digest}  results.json\n", encoding="utf-8", newline="\n")
    reviews = REPO / "experiments/reviews/E2_BACKGROUND_ANALYSIS_RESULTS_2026-11-14.json"
    reviews.write_text(body, encoding="utf-8", newline="\n")
    print(json.dumps(dict(results_sha256=digest, reviews_copy=str(reviews))), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
