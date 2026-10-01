"""Text2CAD G1/G2 representative benchmark + optimization equivalence gates (M1c B2).

Task contract: experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md
(G1/G2) and experiments/plans/IDEA01_TWO_910C_AUTONOMOUS_ROADMAP_2026-09-30.md
(two-card route + seven-day gate). THIS MODULE ONLY IMPLEMENTS THE PROGRAMS
AND GATES -- the current real-model forward budget is 0, so nothing here has
been timed on the real model; every forward-consuming mode goes through the
same fail-closed authorization + shared budget ledger as the matrix scorer
(``text2cad_frozen_runner``).

Modes (CLI ``main``):

  * ``select``   -- pure CPU, no model: deterministically generate the G1/G2
    pair selection (1 spot pair, 10 fixed batch-consistency pairs crossing
    short/medium/long CAD x short/medium/long/longest text, and 30
    length-strata x 20 representative throughput pairs) from the pair
    manifest's texts.jsonl + cad_length_index.jsonl. This is how the
    committed selection config was produced; it never reads qrels.
  * ``eta``      -- pure CPU, no model: combine the TWO cards' measured
    throughput results (overlapping measurement windows REQUIRED -- a
    single-card result is never scaled by an ideal 2x), weight by the real
    length distribution of all 29,319,624 pairs, add measured cold-start and
    per-run overhead and the >=1.30 safety factor, and compare against the
    seven-day gate (two-card effective >= ~63.0 pairs/s).
  * ``spot``     -- G1: ONE fixed pair, batch=1, full-length reference
    forward; records per-valid-token FP32 log-prob detail, logits stats,
    termination/padding facts. Real forward -> authorized + budget-charged.
  * ``batch-consistency`` -- G1: the fixed ten pairs scored at batch=1 and at
    each requested batch>1 on the full-length reference path; per-pair
    valid-position logits AND final scores compared within the
    pre-registered tolerances below; any drift aborts non-zero.
  * ``trim-check`` / ``cache-check`` -- the Stage-B2 optimization gates: same
    real pairs through the reference path vs ``trim_mode=batch_max`` /
    ``text_cache=True``; logits on valid positions and scores compared.
    Tolerances are the pre-registered constants below; a violation keeps the
    optimization OFF (defaults) -- tolerances are never widened after seeing
    a failure.
  * ``throughput`` -- G2: batch tiers 1/2/4/8 (16/32 only with
    ``--allow-large-batch``; OOM at a tier is recorded and stops escalation),
    SAME fixed stratified pair sets per tier, >=2 warmup passes, repeated
    reps; per-cell timing separates pure forward+score from end-to-end
    (assembly + forward + score + record write) time, pairs/s,
    valid-items/s, peak device memory, OOM/failure counts. Every forwarded
    pair (warmups and repeats included) is charged to the shared approval
    ledger BEFORE it forwards.

Candidate-side prefix caching (decoder layers 0-1 have no cross-attention) is
NOT implemented in this stage: it needs its own same-pair equivalence gate
design and review first (B1 review condition 2); only the tolerance/gate
protocol is established here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from .cadmium_runner import RunnerError  # audited shared error type
from .text2cad_frozen_runner import (
    PRODUCTION_REPO_ROOT,
    BertTextCache,
    RunnerConfig,
    SharedBudgetLedger,
    ZipCandidateStore,
    _CachedEmbedder,
    _load_expected_pair_hashes,
    assemble_batch,
    atomic_publish,
    authorize_text2cad_run,
    canonical_budget_path,
    check_revision_paths,
    ensure_outside_checkout,
    environment_record,
    load_gated_model_and_store,
    new_invocation_id,
    score_pair_token_diagnostic,
)
from .text2cad_pair_index import (
    CAD_LENGTH_FILE,
    EXPECTED_N_PAIRS,
    PAIR_MANIFEST_FILE,
    TEXTS_FILE,
    PairIndexError,
    PairSpace,
    load_cad_length_index,
    load_cad_store,
    load_pair_manifest,
    load_texts,
    read_jsonl_strict,
    sha256_file,
)
from .text2cad_port.loader import PINNED_CHECKPOINT_SHA256
from .text2cad_port.model import teacher_forced_logits
from .text2cad_vector_scorer import SCORE_NAME, SCORE_SPEC_VERSION, two_head_factorized_score

# ---------------------------------------------------------------------------
# Pre-registered constants (G1/G2 contract)
# ---------------------------------------------------------------------------

#: Seven-day gate over the exact full matrix, with the safety factor applied.
SEVEN_DAY_SECONDS = 168 * 3600  # 604,800
ETA_SAFETY_FACTOR = 1.30  # contract: at least 1.30
FULL_MATRIX_PAIRS = EXPECTED_N_PAIRS  # 29,319,624
#: Exact zero-overhead lower bound; the actual ETA adds measured overhead.
TWO_CARD_MIN_PAIRS_PER_S = FULL_MATRIX_PAIRS * ETA_SAFETY_FACTOR / SEVEN_DAY_SECONDS

#: Stratification for representative timing: texts by (n_chars, text_index)
#: rank into terciles, CADs by (n_real_tokens, cad_index) rank into deciles.
N_TEXT_TERCILES = 3
N_CAD_DECILES = 10
N_STRATA = N_TEXT_TERCILES * N_CAD_DECILES
THROUGHPUT_TEXTS_PER_CELL = 4
THROUGHPUT_CADS_PER_CELL = 5
THROUGHPUT_PAIRS_PER_CELL = THROUGHPUT_TEXTS_PER_CELL * THROUGHPUT_CADS_PER_CELL  # 20

#: Fixed ten-pair batch-consistency table: (text length quantile, CAD length
#: quantile) -- crosses short/medium/long CAD with short/medium/long text and
#: includes the LONGEST text against a P90 CAD.
TEN_PAIR_QUANTILES = [
    (10, 10), (10, 50), (10, 90),
    (50, 10), (50, 50), (50, 90),
    (90, 10), (90, 50), (90, 90),
    (100, 90),
]

#: PRE-REGISTERED G1 tolerances (provisionally fixed now, to be confirmed
#: against real-device measurements at G1; a gate failure at these values
#: STOPS the run and the optimization/batching stays off -- tolerances are
#: never widened after seeing test).
G1_LOGITS_RTOL = 1e-4
G1_LOGITS_ATOL = 1e-5
G1_SCORE_RTOL = 1e-6
G1_SCORE_ATOL = 1e-4

DEFAULT_BATCH_TIERS = [1, 2, 4, 8]
LARGE_BATCH_TIERS = [16, 32]
DEFAULT_WARMUP_PASSES = 2
DEFAULT_REPS = 5
DEFAULT_CONSISTENCY_BATCHES = [2, 5, 10]

SELECTION_SCHEMA_VERSION = "text2cad-g1g2-selection-v1"
BENCH_RESULT_SCHEMA_VERSION = "text2cad-bench-result-v1"

#: Committed (reviewed) selection config the production CLI pins by path.
SELECTION_CONFIG_REL = "experiments/configs/text2cad_g1g2_pair_selection_20260930.json"
EXPECTED_PAIR_HASHES_REL = "experiments/configs/text2cad_pair_manifest_expected_20260930.json"
EXPECTED_KEYS_MANIFEST_REL = "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"

#: Use the same ledger identity as the matrix shard on each physical device.
#: This permits the two different cards to measure concurrently and prevents
#: a benchmark and a matrix shard from using the same card at once.
BENCH_LEDGER_SHARD_BY_DEVICE = {"npu:0": 0, "npu:1": 1}

OOM_PATTERN = "out of memory"


# ---------------------------------------------------------------------------
# Selection generation (deterministic; pure CPU; never reads qrels)
# ---------------------------------------------------------------------------


def _rank_order(indices: Sequence[int], key_of: Callable[[int], int]) -> list[int]:
    return sorted(indices, key=lambda i: (key_of(i), i))


def _pick_at_quantile(ordered: Sequence[int], quantile: float) -> int:
    if not 0 <= quantile <= 100:
        raise RunnerError(f"quantile must be in [0,100], got {quantile}")
    return ordered[round((len(ordered) - 1) * quantile / 100)]


def _stratum_of(rank: int, n: int, n_strata: int) -> int:
    return min(n_strata - 1, int(n_strata * rank / n))


def _evenly_spaced(ordered: Sequence[int], k: int) -> list[int]:
    """k evenly spaced picks from an ordered list (endpoints included)."""
    if len(ordered) <= k:
        return list(ordered)
    if k == 1:
        return [ordered[0]]
    picks = sorted({round(i * (len(ordered) - 1) / (k - 1)) for i in range(k)})
    return [ordered[i] for i in picks]


def generate_selection(
    text_records: Sequence[dict[str, Any]],
    cad_length_records: Sequence[dict[str, Any]],
    *,
    texts_sha256: str,
    cad_length_sha256: str,
    pair_manifest_sha256: str,
) -> dict[str, Any]:
    """Deterministic G1/G2 pair selection from pair-manifest artifacts.

    Stratification: texts ranked by (n_chars, text_index) into terciles
    (short/medium/long), CADs ranked by (n_real_tokens, cad_index) into
    deciles. Throughput cells take 4 evenly spaced texts x 5 evenly spaced
    CADs per cell (20 pairs, sorted by pair_index). The spot pair is the
    median-rank text x median-rank CAD; the ten batch-consistency pairs
    follow TEN_PAIR_QUANTILES. Everything is materialized with verifying
    facts so the benchmark CLI can re-check the selection against the live
    manifest before any forward.
    """
    n_texts = len(text_records)
    n_cads = len(cad_length_records)
    pair_space = PairSpace(n_texts=n_texts, n_cads=n_cads)

    n_chars = {rec["text_index"]: rec["n_chars"] for rec in text_records}
    n_real = {rec["cad_index"]: rec["n_real_tokens"] for rec in cad_length_records}
    text_rank_order = _rank_order(range(n_texts), n_chars.get)
    cad_rank_order = _rank_order(range(n_cads), n_real.get)

    def pair_entry(text_index: int, cad_index: int) -> dict[str, int]:
        return {
            "pair_index": pair_space.pair_index_of(text_index, cad_index),
            "text_index": text_index,
            "cad_index": cad_index,
            "n_chars": n_chars[text_index],
            "n_real_tokens": n_real[cad_index],
        }

    spot = pair_entry(_pick_at_quantile(text_rank_order, 50), _pick_at_quantile(cad_rank_order, 50))
    ten_pair = [
        pair_entry(_pick_at_quantile(text_rank_order, tq), _pick_at_quantile(cad_rank_order, cq))
        for tq, cq in TEN_PAIR_QUANTILES
    ]

    texts_by_tercile: list[list[int]] = [[] for _ in range(N_TEXT_TERCILES)]
    for rank, text_index in enumerate(text_rank_order):
        texts_by_tercile[_stratum_of(rank, n_texts, N_TEXT_TERCILES)].append(text_index)
    cads_by_decile: list[list[int]] = [[] for _ in range(N_CAD_DECILES)]
    for rank, cad_index in enumerate(cad_rank_order):
        cads_by_decile[_stratum_of(rank, n_cads, N_CAD_DECILES)].append(cad_index)

    strata: dict[str, list[dict[str, int]]] = {}
    shortfalls: list[str] = []
    for tercile in range(N_TEXT_TERCILES):
        for decile in range(N_CAD_DECILES):
            cell_texts = _evenly_spaced(sorted(texts_by_tercile[tercile]), THROUGHPUT_TEXTS_PER_CELL)
            cell_cads = _evenly_spaced(sorted(cads_by_decile[decile]), THROUGHPUT_CADS_PER_CELL)
            pairs = sorted(
                (pair_entry(t, c) for t in cell_texts for c in cell_cads),
                key=lambda e: e["pair_index"],
            )
            key = f"text_t{tercile}_cad_d{decile}"
            strata[key] = pairs
            if len(pairs) < THROUGHPUT_PAIRS_PER_CELL:
                shortfalls.append(f"{key}: {len(pairs)} < {THROUGHPUT_PAIRS_PER_CELL}")

    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "provenance": {
            "texts_jsonl_sha256": texts_sha256,
            "cad_length_index_sha256": cad_length_sha256,
            "pair_manifest_sha256": pair_manifest_sha256,
            "generated_by": "text2cad_speed_bench.generate_selection (CPU, zero forward)",
        },
        "rules": {
            "text_strata": f"rank by (n_chars, text_index) into {N_TEXT_TERCILES} terciles",
            "cad_strata": f"rank by (n_real_tokens, cad_index) into {N_CAD_DECILES} deciles",
            "throughput_cell": (
                f"{THROUGHPUT_TEXTS_PER_CELL} texts x {THROUGHPUT_CADS_PER_CELL} cads, "
                "evenly spaced, cross product"
            ),
            "spot": "median-rank text x median-rank CAD",
            "ten_pair_quantiles": TEN_PAIR_QUANTILES,
        },
        "counts": {
            "n_texts": n_texts,
            "n_cads": n_cads,
            "n_pairs": pair_space.n_pairs,
            "n_throughput_pairs": sum(len(v) for v in strata.values()),
            "strata_shortfalls": shortfalls,
        },
        "spot": spot,
        "ten_pair": ten_pair,
        "throughput_strata": strata,
    }


def verify_selection_against_manifest(
    selection: dict[str, Any],
    manifest_dir,
    pair_space: PairSpace,
) -> None:
    """Re-check a committed selection against the LIVE pair manifest.

    Fails closed on: schema drift, provenance hash drift (the selection must
    have been generated from exactly these artifacts), pair indices out of
    range, or any recorded fact (n_chars / n_real_tokens) disagreeing with
    the live artifacts.
    """
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise RunnerError(
            f"selection schema {selection.get('schema_version')!r} != {SELECTION_SCHEMA_VERSION!r}"
        )
    provenance = selection.get("provenance") or {}
    manifest_dir = Path(manifest_dir)
    live = {
        "texts_jsonl_sha256": sha256_file(manifest_dir / TEXTS_FILE),
        "cad_length_index_sha256": sha256_file(manifest_dir / CAD_LENGTH_FILE),
        "pair_manifest_sha256": sha256_file(manifest_dir / PAIR_MANIFEST_FILE),
    }
    for key, value in live.items():
        if provenance.get(key) != value:
            raise RunnerError(
                f"selection provenance {key} {provenance.get(key)!r} != live {value!r}; "
                "the selection was not generated from these artifacts"
            )
    text_records = read_jsonl_strict(
        manifest_dir / TEXTS_FILE,
        frozenset({"text_index", "text", "text_sha256", "n_chars"}),
        TEXTS_FILE,
    )
    length_records = load_cad_length_index(manifest_dir)
    n_chars = {rec["text_index"]: rec["n_chars"] for rec in text_records}
    n_real = {rec["cad_index"]: rec["n_real_tokens"] for rec in length_records}

    def check_entry(entry: dict[str, Any], where: str) -> None:
        pair_index = entry["pair_index"]
        text_index, cad_index = pair_space.coords_of(pair_index)  # range-checked
        if entry["text_index"] != text_index or entry["cad_index"] != cad_index:
            raise RunnerError(f"{where}: pair_index/coords mismatch")
        if entry["n_chars"] != n_chars[text_index]:
            raise RunnerError(f"{where}: n_chars drift")
        if entry["n_real_tokens"] != n_real[cad_index]:
            raise RunnerError(f"{where}: n_real_tokens drift")

    check_entry(selection["spot"], "spot")
    for i, entry in enumerate(selection["ten_pair"]):
        check_entry(entry, f"ten_pair[{i}]")
    for key, entries in sorted(selection["throughput_strata"].items()):
        for i, entry in enumerate(entries):
            check_entry(entry, f"{key}[{i}]")


def full_matrix_cell_counts(
    text_records: Sequence[dict[str, Any]],
    cad_length_records: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Exact pair counts of the FULL 3,644x8,046 matrix per stratification cell."""
    n_texts = len(text_records)
    n_cads = len(cad_length_records)
    text_rank_order = _rank_order(
        range(n_texts), {rec["text_index"]: rec["n_chars"] for rec in text_records}.get
    )
    cad_rank_order = _rank_order(
        range(n_cads), {rec["cad_index"]: rec["n_real_tokens"] for rec in cad_length_records}.get
    )
    tercile_sizes = [0] * N_TEXT_TERCILES
    for rank in range(n_texts):
        tercile_sizes[_stratum_of(rank, n_texts, N_TEXT_TERCILES)] += 1
    decile_sizes = [0] * N_CAD_DECILES
    for rank in range(n_cads):
        decile_sizes[_stratum_of(rank, n_cads, N_CAD_DECILES)] += 1
    counts: dict[str, int] = {}
    for t in range(N_TEXT_TERCILES):
        for d in range(N_CAD_DECILES):
            counts[f"text_t{t}_cad_d{d}"] = tercile_sizes[t] * decile_sizes[d]
    return counts


# ---------------------------------------------------------------------------
# ETA (pure calculation; two-card honesty; fail-closed)
# ---------------------------------------------------------------------------


def combine_two_card_results(result_a: dict[str, Any], result_b: dict[str, Any]) -> dict[str, float]:
    """Per-cell two-card end-to-end rate from two CONCURRENT measurements.

    Both results must carry measurement windows that overlap in wall-clock
    time (the cards measured simultaneously); a single measurement is never
    scaled by an ideal 2x. Per cell the aggregate rate is the sum of the two
    cards' end-to-end rates. Any cell missing on either card refuses the
    combination (fail-closed).
    """
    for label, result in (("A", result_a), ("B", result_b)):
        if result.get("schema_version") != BENCH_RESULT_SCHEMA_VERSION:
            raise RunnerError(f"card {label}: not a {BENCH_RESULT_SCHEMA_VERSION} result")
        if not result.get("measurement_window"):
            raise RunnerError(f"card {label}: missing measurement_window")
    wa = result_a["measurement_window"]
    wb = result_b["measurement_window"]
    if not (
        float(wa["start_epoch"]) <= float(wb["end_epoch"])
        and float(wb["start_epoch"]) <= float(wa["end_epoch"])
    ):
        raise RunnerError(
            "card result measurement windows do not overlap; refusing to "
            "combine non-concurrent measurements as a two-card rate"
        )
    overlap = min(float(wa["end_epoch"]), float(wb["end_epoch"])) - max(float(wa["start_epoch"]), float(wb["start_epoch"]))
    shortest = min(float(wa["end_epoch"]) - float(wa["start_epoch"]), float(wb["end_epoch"]) - float(wb["start_epoch"]))
    if shortest <= 0 or overlap < 0.8 * shortest:
        raise RunnerError("card result windows must overlap for at least 80% of the shorter run")
    rates: dict[str, float] = {}
    cells_a = result_a.get("cell_e2e_pairs_per_s") or {}
    cells_b = result_b.get("cell_e2e_pairs_per_s") or {}
    for cell in sorted(set(cells_a) | set(cells_b)):
        ra, rb = cells_a.get(cell), cells_b.get(cell)
        if ra is None or rb is None:
            raise RunnerError(f"cell {cell} missing on one card; refusing partial combination")
        rates[cell] = float(ra) + float(rb)
    return rates


def parse_batch_tiers(raw: str, allow_large_batch: bool) -> list[int]:
    """Only the reviewed tiers can reach a forward; large tiers need HBM approval."""
    try:
        tiers = [int(t) for t in raw.split(",") if t]
    except ValueError as exc:
        raise RunnerError(f"invalid batch tiers {raw!r}") from exc
    allowed = set(DEFAULT_BATCH_TIERS) | (set(LARGE_BATCH_TIERS) if allow_large_batch else set())
    if not tiers or len(tiers) != len(set(tiers)) or not set(tiers) <= allowed:
        raise RunnerError(f"batch tiers {tiers} are not approved; allowed: {sorted(allowed)}")
    if allow_large_batch:
        tiers = sorted(set(tiers) | set(LARGE_BATCH_TIERS))
    return sorted(tiers)


def validate_eta_card_results(
    results: Sequence[dict[str, Any]], paths: Sequence[str], tier: int,
    *, expected_manifest_sha256: str, expected_selection_sha256: str,
    expected_cells: set[str],
) -> None:
    """Reject duplicated or mismatched card runs before summing their rates."""
    if len(results) != 2 or len(paths) != 2 or Path(paths[0]).resolve() == Path(paths[1]).resolve():
        raise RunnerError("eta requires two distinct card result files")
    if any(not isinstance(r.get("invocation_id"), str) or not r["invocation_id"] for r in results) or results[0].get("invocation_id") == results[1].get("invocation_id"):
        raise RunnerError("eta requires two distinct throughput invocations")
    devices = {((r.get("environment") or {}).get("device")) for r in results}
    if devices != {"npu:0", "npu:1"}:
        raise RunnerError("eta requires one successful throughput result from each NPU")
    keys = ("git_sha", "pair_manifest_sha256", "checkpoint_sha256", "torch_dtype", "score_name", "spec_version")
    for key in keys:
        values = [(r.get("environment") or {}).get(key) for r in results]
        if not values[0] or values[0] != values[1]:
            raise RunnerError(f"eta card results disagree on {key}")
    for result in results:
        env = result.get("environment") or {}
        if result.get("mode") != "throughput" or result.get("schema_version") != BENCH_RESULT_SCHEMA_VERSION:
            raise RunnerError("eta requires successful throughput result schema")
        if env["pair_manifest_sha256"] != expected_manifest_sha256 or result.get("selection_sha256") != expected_selection_sha256:
            raise RunnerError("eta result does not match the reviewed pair manifest and selection")
        if env["checkpoint_sha256"] != PINNED_CHECKPOINT_SHA256 or env["torch_dtype"] != "float32" or env.get("trim_mode") != "none" or env.get("text_cache") is not False:
            raise RunnerError("eta result does not use the approved frozen reference configuration")
        if env["score_name"] != SCORE_NAME or env["spec_version"] != SCORE_SPEC_VERSION:
            raise RunnerError("eta result uses an unreviewed score definition")
        if len(env["git_sha"]) != 40 or not set(env["git_sha"]) <= set("0123456789abcdef"):
            raise RunnerError("eta result must carry a complete reviewed Git SHA")
        auth = result.get("authorization") or {}
        if auth.get("approved_sha") != env["git_sha"] or (result.get("budget") or {}).get("approved_sha") != env["git_sha"]:
            raise RunnerError("eta result code SHA differs from its authorization or budget")
        tier_record = next((t for t in result.get("tiers", []) if t.get("batch_size") == tier), None)
        if not tier_record or tier_record.get("status") != "ok":
            raise RunnerError(f"eta tier {tier} did not complete successfully on both cards")
        rates = result.get("cell_e2e_pairs_per_s_by_tier") or {}
        if set(rates) != expected_cells:
            raise RunnerError("eta result cell coverage differs from the full selection")
        for cell in expected_cells:
            rate = rates[cell].get(str(tier))
            if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
                raise RunnerError(f"eta cell {cell} has no finite positive tier {tier} rate")


def estimate_full_matrix_seconds(
    cell_pair_counts: dict[str, int],
    cell_e2e_rates: dict[str, float],
    *,
    overhead_seconds: float,
    safety_factor: float = ETA_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Length-distribution-weighted full-matrix ETA (fail-closed).

    ``eta = (sum_cells n_pairs_cell / rate_cell + overhead) * safety_factor``.
    Every cell of the full matrix must have a positive measured end-to-end
    rate; a missing or non-positive rate refuses the estimate (no
    extrapolation from short-sample kernels). ``safety_factor`` must be >=
    1.30 (contract). ``overhead_seconds`` must carry cold start, IO, failure
    retry and output writing as measured/estimated by the caller (recorded
    verbatim in the report).
    """
    if safety_factor < ETA_SAFETY_FACTOR:
        raise RunnerError(f"safety_factor {safety_factor} < required {ETA_SAFETY_FACTOR}")
    if overhead_seconds < 0:
        raise RunnerError(f"overhead_seconds must be >= 0, got {overhead_seconds}")
    if sum(cell_pair_counts.values()) != FULL_MATRIX_PAIRS:
        raise RunnerError(
            f"cell_pair_counts sum {sum(cell_pair_counts.values())} != "
            f"{FULL_MATRIX_PAIRS}; the ETA covers the exact full matrix only"
        )
    per_cell: list[dict[str, Any]] = []
    compute_seconds = 0.0
    for cell, n_pairs in sorted(cell_pair_counts.items()):
        rate = cell_e2e_rates.get(cell)
        if rate is None or not isinstance(rate, (int, float)) or rate <= 0:
            raise RunnerError(
                f"cell {cell} has no positive measured end-to-end rate; "
                "refusing to extrapolate"
            )
        seconds = n_pairs / float(rate)
        compute_seconds += seconds
        per_cell.append(
            {"cell": cell, "n_pairs": n_pairs, "rate_pairs_per_s": float(rate), "seconds": seconds}
        )
    eta_seconds = (compute_seconds + overhead_seconds) * safety_factor
    effective_rate = FULL_MATRIX_PAIRS / eta_seconds
    return {
        "eta_seconds": eta_seconds,
        "eta_hours": eta_seconds / 3600.0,
        "compute_seconds_presafety": compute_seconds,
        "overhead_seconds": overhead_seconds,
        "safety_factor": safety_factor,
        "measured_aggregate_pairs_per_s_preoverhead_presafety": (
            FULL_MATRIX_PAIRS / compute_seconds if compute_seconds > 0 else None
        ),
        "effective_two_card_pairs_per_s": effective_rate,
        "required_two_card_pairs_per_s": TWO_CARD_MIN_PAIRS_PER_S,
        "required_meaning": (
            "the plan's seven-day gate expressed as a MEASURED two-card end-to-end "
            "rate: pass iff measured >= 29,319,624*1.30/604,800 ~= 63.0 pairs/s, "
            "equivalently eta_seconds <= 604,800 with the safety factor applied"
        ),
        "seven_day_seconds": SEVEN_DAY_SECONDS,
        "passes_seven_day_gate": eta_seconds <= SEVEN_DAY_SECONDS,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Detailed scoring for the G1 gates (keeps CPU FP32 logits; tiny pair counts)
# ---------------------------------------------------------------------------


def score_pairs_detailed(
    model,
    store: ZipCandidateStore,
    texts: Sequence[str],
    pair_list: Sequence[tuple[int, int]],
    device: torch.device,
    *,
    batch_size: int,
    trim_mode: str = "none",
    text_cache: bool = False,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Score ``(text_index, cad_index)`` pairs and KEEP the CPU FP32 logits.

    Returns ``(logits_per_pair [(T,2,267)], score_dicts)`` with T=271 on the
    reference path (or batch-max length under the trim gate). Used only by
    the G1 gates where pair counts are tiny; the matrix path discards logits
    after scoring. When ``text_cache`` is set, the model's embedder is
    temporarily wrapped in the unique-text cache and restored afterwards.
    """
    original_embedder = model.base_text_embedder
    if text_cache:
        model.base_text_embedder = _CachedEmbedder(BertTextCache(original_embedder))
    try:
        all_logits: list[torch.Tensor] = []
        all_scores: list[dict[str, Any]] = []
        for start in range(0, len(pair_list), batch_size):
            chunk = pair_list[start : start + batch_size]
            batch_texts = [texts[t] for t, _ in chunk]
            batch_cads = [c for _, c in chunk]
            vec_dict, mask_cad_dict, cad_seq_len = assemble_batch(
                store, batch_cads, device, trim_mode=trim_mode
            )
            with torch.inference_mode():
                pred, _target, _shifted = teacher_forced_logits(
                    model, batch_texts, vec_dict, mask_cad_dict, cad_seq_len=cad_seq_len
                )
            logits_cpu = pred.to(torch.float32).cpu()
            scores = two_head_factorized_score(
                logits_cpu, vec_dict["cad_vec"].cpu(), mask_cad_dict["key_padding_mask"].cpu()
            )
            for row in range(logits_cpu.shape[0]):
                all_logits.append(logits_cpu[row])
            all_scores.extend(scores)
        return all_logits, all_scores
    finally:
        if text_cache:
            model.base_text_embedder = original_embedder


def compare_gate_results(
    reference: tuple[list[torch.Tensor], list[dict[str, Any]]],
    alternate: tuple[list[torch.Tensor], list[dict[str, Any]]],
    store: ZipCandidateStore,
    pair_list: Sequence[tuple[int, int]],
    *,
    logits_rtol: float = G1_LOGITS_RTOL,
    logits_atol: float = G1_LOGITS_ATOL,
    score_rtol: float = G1_SCORE_RTOL,
    score_atol: float = G1_SCORE_ATOL,
) -> dict[str, Any]:
    """Compare two runs over the SAME pairs: valid-position logits + scores.

    Logits are compared elementwise only at prediction rows whose target is a
    real token (``t <= n_real_tokens - 2``), per pair -- padding-position
    logits are legitimately affected by batching/trimming and are never
    scored. Score fields compared: score_x / score_y / score_total (within
    tolerance), n_valid_* / valid_seq_len / n_real_tokens (exact). A
    violation sets ``passed=False`` and the caller MUST refuse; the gates
    fail closed and the tolerances are never widened after seeing a failure.
    """
    ref_logits, ref_scores = reference
    alt_logits, alt_scores = alternate
    if len(ref_logits) != len(pair_list) or len(alt_logits) != len(pair_list):
        raise RunnerError("gate comparison: result count != pair count")
    per_pair: list[dict[str, Any]] = []
    passed = True
    for i, (text_index, cad_index) in enumerate(pair_list):
        n_real = int(store.get(cad_index)["n_real_tokens"])
        n_rows = n_real - 1  # compared prediction rows (real-token targets)
        ref = ref_logits[i][:n_rows].to(torch.float64)
        alt = alt_logits[i][:n_rows].to(torch.float64)
        if ref.shape != alt.shape:
            passed = False
            per_pair.append(
                {"pair": i, "passed": False,
                 "reason": f"shape {tuple(ref.shape)} vs {tuple(alt.shape)}"}
            )
            continue
        diff = (ref - alt).abs()
        tol = logits_atol + logits_rtol * ref.abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        n_viol = int((diff > tol).sum())
        rs, alts = ref_scores[i], alt_scores[i]
        count_keys_ok = all(
            rs[key] == alts[key]
            for key in ("n_valid_x", "n_valid_y", "n_valid_total", "valid_seq_len", "n_real_tokens")
        )
        score_ok = count_keys_ok and all(
            abs(rs[key] - alts[key]) <= score_atol + score_rtol * abs(rs[key])
            for key in ("score_x", "score_y", "score_total")
        )
        pair_ok = n_viol == 0 and score_ok
        passed = passed and pair_ok
        per_pair.append(
            {
                "pair": i,
                "text_index": text_index,
                "cad_index": cad_index,
                "n_rows_compared": n_real - 1,
                "max_abs_logit_diff": max_abs,
                "n_logit_violations": n_viol,
                "score_ref": rs["score_total"],
                "score_alt": alts["score_total"],
                "counts_exact": count_keys_ok,
                "score_ok": score_ok,
                "passed": pair_ok,
            }
        )
    return {
        "passed": passed,
        "tolerances": {
            "logits_rtol": logits_rtol,
            "logits_atol": logits_atol,
            "score_rtol": score_rtol,
            "score_atol": score_atol,
        },
        "per_pair": per_pair,
    }


def logits_stats(logits: torch.Tensor) -> dict[str, Any]:
    """Finite/absmax/extremes summary of one pair's logits (spot check)."""
    data = logits.to(torch.float64)
    return {
        "shape": list(logits.shape),
        "all_finite": bool(torch.isfinite(logits).all()),
        "abs_max": float(data.abs().max()),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def peak_device_memory_bytes(device: str) -> Optional[int]:
    if device.startswith("npu") and hasattr(torch, "npu"):
        try:
            return int(torch.npu.max_memory_allocated())
        except Exception:  # pragma: no cover - device specific
            return None
    return None


def reset_peak_device_memory(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception:  # pragma: no cover - device specific
            pass


def plan_throughput_forwards(
    cells: dict[str, Sequence[tuple[int, int]]],
    batch_tiers: Sequence[int],
    warmup_passes: int,
    reps: int,
    warmup_batches_per_pass: int = 2,
) -> int:
    """Exact number of forwards a throughput run will execute (budget check).

    Per tier: ``warmup_passes`` passes over the first ``warmup_batches_per_pass``
    batches of the FIRST non-empty cell, then ``reps`` full passes over every
    cell's pairs. Warmups and repeats are real forwards and count in full.
    """
    first_cell_pairs = 0
    for key in sorted(cells):
        if cells[key]:
            first_cell_pairs = len(cells[key])
            break
    total_pairs = sum(len(v) for v in cells.values())
    forwards = 0
    for tier in batch_tiers:
        n_warmup_batches = min(warmup_batches_per_pass, max(1, (first_cell_pairs + tier - 1) // tier))
        warmup_pairs = min(first_cell_pairs, n_warmup_batches * tier)
        forwards += warmup_passes * warmup_pairs + reps * total_pairs
    return forwards


def run_throughput_tiers(
    score_batch_fn: Callable[[list[tuple[int, int]]], tuple[list[dict[str, Any]], dict[str, float]]],
    cells: dict[str, list[tuple[int, int]]],
    *,
    batch_tiers: Sequence[int],
    warmup_passes: int = DEFAULT_WARMUP_PASSES,
    reps: int = DEFAULT_REPS,
    device: str = "cpu",
    charge_fn: Optional[Callable[[int], None]] = None,
    out_dir=None,
) -> dict[str, Any]:
    """Measure per-tier, per-cell throughput over the SAME fixed pair sets.

    ``score_batch_fn(batch) -> (scores, {"preprocess_s", "forward_score_s"})``
    times its own phases (assembly vs forward+transfer+score). The driver
    adds record serialization+write time (one ``bench_records_t{T}_r{R}.jsonl``
    per tier+rep, atomic-publish, never overwrite) to form the honest
    end-to-end rep wall. ``charge_fn(n)`` runs BEFORE every forwarded batch
    (warmups included); it is None ONLY in synthetic tests -- the CLI always
    passes the ledger charge. OOM at a tier records ``status="oom"`` and
    stops escalation to larger tiers; any other exception aborts the run.
    """
    if warmup_passes < 2:
        raise RunnerError(f"warmup_passes must be >= 2 (contract), got {warmup_passes}")
    if reps < 1:
        raise RunnerError(f"reps must be >= 1, got {reps}")
    tiers = sorted(int(t) for t in batch_tiers)
    if len(set(tiers)) != len(tiers) or not tiers or tiers[0] < 1:
        raise RunnerError(f"batch tiers must be distinct positive ints, got {tiers}")

    cell_e2e_rates: dict[str, dict[str, Optional[float]]] = {}
    tier_results: list[dict[str, Any]] = []
    escalation_stop: Optional[str] = None
    for tier in tiers:
        if escalation_stop is not None:
            tier_results.append(
                {"batch_size": tier, "status": f"skipped_after_{escalation_stop}"}
            )
            continue
        tier_record: dict[str, Any] = {"batch_size": tier, "status": "ok", "cells": {}}
        reset_peak_device_memory(device)
        try:
            # Warmup: two passes over the first two batches of the first
            # non-empty cell (charged, untimed).
            first_cell = next((k for k in sorted(cells) if cells[k]), None)
            if first_cell is not None:
                warmup_pairs = cells[first_cell][: 2 * tier]
                for _ in range(warmup_passes):
                    for start in range(0, len(warmup_pairs), tier):
                        batch = warmup_pairs[start : start + tier]
                        if charge_fn is not None:
                            charge_fn(len(batch))
                        score_batch_fn(batch)
            for cell in sorted(cells):
                pairs = cells[cell]
                if not pairs:
                    continue
                batches = [pairs[i : i + tier] for i in range(0, len(pairs), tier)]
                cell_reps: list[dict[str, Any]] = []
                for rep in range(reps):
                    rep_start = time.perf_counter()
                    preprocess_s = 0.0
                    forward_score_s = 0.0
                    valid_items = 0
                    rep_records: list[dict[str, Any]] = []
                    for batch in batches:
                        if charge_fn is not None:
                            charge_fn(len(batch))
                        scores, timing = score_batch_fn(batch)
                        preprocess_s += float(timing["preprocess_s"])
                        forward_score_s += float(timing["forward_score_s"])
                        valid_items += sum(int(s["n_valid_total"]) for s in scores)
                        rep_records.extend(scores)
                    write_start = time.perf_counter()
                    if out_dir is not None:
                        atomic_publish(
                            Path(out_dir) / f"bench_records_t{tier}_r{rep}_{cell}.jsonl",
                            [json.dumps(s, ensure_ascii=False) for s in rep_records],
                        )
                    else:
                        json.dumps(rep_records)  # same serialization cost in tests
                    write_s = time.perf_counter() - write_start
                    e2e_s = time.perf_counter() - rep_start
                    cell_reps.append(
                        {
                            "rep": rep,
                            "pairs": len(pairs),
                            "preprocess_s": preprocess_s,
                            "forward_score_s": forward_score_s,
                            "write_s": write_s,
                            "e2e_s": e2e_s,
                            "pairs_per_s_forward_score": (
                                len(pairs) / forward_score_s if forward_score_s > 0 else None
                            ),
                            "pairs_per_s_e2e": len(pairs) / e2e_s if e2e_s > 0 else None,
                            "valid_items_per_s_e2e": valid_items / e2e_s if e2e_s > 0 else None,
                        }
                    )
                e2e_rates = sorted(r["pairs_per_s_e2e"] for r in cell_reps if r["pairs_per_s_e2e"])
                median_e2e = e2e_rates[len(e2e_rates) // 2] if e2e_rates else None
                tier_record["cells"][cell] = {
                    "reps": cell_reps,
                    "median_pairs_per_s_e2e": median_e2e,
                }
                cell_e2e_rates.setdefault(cell, {})[str(tier)] = median_e2e
            tier_record["peak_device_memory_bytes"] = peak_device_memory_bytes(device)
        except RuntimeError as exc:
            if OOM_PATTERN in str(exc).lower():
                tier_record["status"] = "oom"
                tier_record["error"] = str(exc)[:200]
                tier_record["peak_device_memory_bytes"] = peak_device_memory_bytes(device)
                tier_results.append(tier_record)
                escalation_stop = "oom"
                continue
            raise
        tier_results.append(tier_record)
    return {
        "tiers": tier_results,
        "cell_e2e_pairs_per_s_by_tier": cell_e2e_rates,
        "warmup_passes": warmup_passes,
        "reps": reps,
        "escalation_stop": escalation_stop,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FORWARD_MODES = ("spot", "batch-consistency", "trim-check", "cache-check", "throughput")


def _pairs_of(entries: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(int(e["text_index"]), int(e["cad_index"])) for e in entries]


def _gated_setup(args) -> dict[str, Any]:
    """Shared fail-closed gate sequence for every forward-consuming mode.

    Same order as the matrix scorer CLI: reviewed fingerprints -> manifest ->
    selection pin+verify -> authorization (fetched origin/master) ->
    production path + committed manifests pinned by path -> revision paths ->
    shared budget ledger -> strict model load + candidate preload. Returns
    everything the modes need; NOTHING has forwarded when this returns.
    """
    repo_root = Path(args.repo_root).resolve()
    canonical_hashes = (repo_root / EXPECTED_PAIR_HASHES_REL).resolve()
    hashes_path = Path(args.expected_pair_hashes or canonical_hashes).resolve()
    if hashes_path != canonical_hashes:
        raise RunnerError(
            f"forward mode requires the committed reviewed pair fingerprints "
            f"{canonical_hashes}"
        )
    if args.cards_used != 1:
        raise RunnerError("one-device benchmark process must charge exactly one card")
    if args.device not in BENCH_LEDGER_SHARD_BY_DEVICE:
        raise RunnerError("benchmark device must be npu:0 or npu:1")
    if args.torch_dtype != "float32":
        raise RunnerError("G1/G2 forward is restricted to float32 until dtype evidence is approved")
    fingerprints = _load_expected_pair_hashes(hashes_path)
    manifest = load_pair_manifest(args.pair_manifest_dir, expected_sha256=fingerprints)
    texts = load_texts(args.pair_manifest_dir)
    members = load_cad_store(args.pair_manifest_dir)
    pair_space = PairSpace(
        n_texts=manifest["counts"]["n_unique_text"], n_cads=manifest["counts"]["n_gallery"]
    )
    selection_path = Path(args.selection).resolve()
    if selection_path != (repo_root / SELECTION_CONFIG_REL).resolve():
        raise RunnerError(
            f"--selection must be the committed reviewed file {repo_root / SELECTION_CONFIG_REL}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    verify_selection_against_manifest(selection, args.pair_manifest_dir, pair_space)

    out_dir = Path(args.out_dir).resolve()
    ensure_outside_checkout(out_dir, repo_root, "out_dir")

    authorization = authorize_text2cad_run(repo_root, cards_used=args.cards_used)
    if repo_root != PRODUCTION_REPO_ROOT:
        raise RunnerError(f"production checkout must be {PRODUCTION_REPO_ROOT}")
    keys_manifest = Path(args.expected_keys_manifest).resolve()
    if keys_manifest != (repo_root / EXPECTED_KEYS_MANIFEST_REL).resolve():
        raise RunnerError(
            f"expected-keys manifest must be the committed reviewed file "
            f"{repo_root / EXPECTED_KEYS_MANIFEST_REL}"
        )
    cfg = RunnerConfig(
        checkpoint_path=args.checkpoint,
        bert_dir=args.bert_dir,
        cad_seq_zip=args.cad_seq_zip,
        pair_manifest_dir=args.pair_manifest_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        batch_size=args.batch_size,
        expected_keys_manifest=str(keys_manifest),
        trim_mode="none",
        text_cache=False,
    )
    check_revision_paths(cfg)
    ledger = SharedBudgetLedger(
        canonical_budget_path(authorization.approved_sha),
        approved_sha=authorization.approved_sha,
        max_pairs=authorization.max_pairs,
        max_card_seconds=authorization.card_hours * 3600.0,
    )
    ledger.check_card_clock()
    invocation_id = new_invocation_id()
    stale = ledger.begin_invocation(invocation_id, BENCH_LEDGER_SHARD_BY_DEVICE[args.device], args.cards_used)
    cold_start = time.perf_counter()
    try:
        model, load_report, store, preload_report = load_gated_model_and_store(cfg, members)
    except Exception:
        ledger.finalize_invocation(invocation_id, exit_code=2, note="model_or_candidate_load_failed")
        raise
    cold_start_s = time.perf_counter() - cold_start
    return {
        "cfg": cfg,
        "repo_root": repo_root,
        "manifest": manifest,
        "fingerprints": fingerprints,
        "texts": texts,
        "pair_space": pair_space,
        "selection": selection,
        "out_dir": out_dir,
        "authorization": authorization,
        "ledger": ledger,
        "invocation_id": invocation_id,
        "model": model,
        "store": store,
        "load_report": load_report,
        "preload_report": preload_report,
        "cold_start_s": cold_start_s,
        "stale_invocations_auto_finalized": stale,
    }


def _write_result(setup: dict[str, Any], mode: str, started_epoch: float, payload: dict[str, Any]) -> Path:
    """Finalize the ledger invocation and atomically publish the result."""
    ledger: SharedBudgetLedger = setup["ledger"]
    invocation_id = setup["invocation_id"]
    ledger.finalize_invocation(invocation_id, exit_code=0, note="ok")
    result = {
        "schema_version": BENCH_RESULT_SCHEMA_VERSION,
        "mode": mode,
        "invocation_id": invocation_id,
        "measurement_window": {"start_epoch": started_epoch, "end_epoch": time.time()},
        "cold_start_s": setup["cold_start_s"],
        "selection_sha256": sha256_file(setup["repo_root"] / SELECTION_CONFIG_REL),
        "environment": environment_record(setup["cfg"], {
            "git_sha": setup["authorization"].approved_sha,
            "pair_manifest_sha256": setup["fingerprints"].get(PAIR_MANIFEST_FILE),
        }),
        "authorization": setup["authorization"].to_dict(),
        "budget": ledger.to_dict(),
        "stale_invocations_auto_finalized": setup["stale_invocations_auto_finalized"],
        **payload,
    }
    out_path = setup["out_dir"] / f"bench_{mode}_{invocation_id}.json"
    atomic_publish(out_path, [json.dumps(result, indent=2, sort_keys=True)])
    return out_path


def _fail_invocation(setup: Optional[dict[str, Any]], note: str) -> None:
    if setup and setup.get("ledger") is not None and setup.get("invocation_id"):
        try:
            setup["ledger"].finalize_invocation(setup["invocation_id"], exit_code=2, note=note)
        except RunnerError:
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Text2CAD G1/G2 benchmark + optimization equivalence gates. Pure-CPU "
            "modes: select, eta. Forward-consuming modes (spot, "
            "batch-consistency, trim-check, cache-check, throughput) share the "
            "matrix scorer's fail-closed gates: reviewed pair-manifest "
            "fingerprints, freshly fetched origin/master pilot_authorized with "
            "approved_sha == detached clean HEAD, committed selection/keys "
            "manifests pinned by path, and the shared two-card budget ledger "
            "charged before every forward."
        )
    )
    parser.add_argument("--mode", required=True,
                        choices=["select", "eta", *FORWARD_MODES])
    parser.add_argument("--checkpoint")
    parser.add_argument("--bert-dir")
    parser.add_argument("--cad-seq-zip")
    parser.add_argument("--pair-manifest-dir", required=True)
    parser.add_argument("--expected-keys-manifest")
    parser.add_argument("--expected-pair-hashes", default=None)
    parser.add_argument("--selection", default=None,
                        help="committed selection config (production: the pinned in-repo path)")
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cards-used", type=int, default=1, choices=[1])
    parser.add_argument("--batch-tiers", default=",".join(str(t) for t in DEFAULT_BATCH_TIERS))
    parser.add_argument("--allow-large-batch", action="store_true",
                        help="permit the 16/32 tiers (HBM-gated; OOM stops escalation)")
    parser.add_argument("--consistency-batches",
                        default=",".join(str(b) for b in DEFAULT_CONSISTENCY_BATCHES))
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--warmup-passes", type=int, default=DEFAULT_WARMUP_PASSES)
    parser.add_argument("--production-batch-size", type=int, default=None,
                        help="eta: which tier's per-cell rates to use")
    parser.add_argument("--card-result", action="append", default=[],
                        help="eta: a per-card throughput result JSON (pass exactly two)")
    parser.add_argument("--overhead-seconds", type=float, default=None,
                        help="eta: cold start + IO + retry + write overhead; default uses the "
                             "results' measured cold starts")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default=None, help="select/eta: output JSON path")
    args = parser.parse_args(argv)

    try:
        manifest_dir = Path(args.pair_manifest_dir)
        if args.mode == "select":
            # Pure CPU selection generation (how the committed config was made).
            text_records = read_jsonl_strict(
                manifest_dir / TEXTS_FILE,
                frozenset({"text_index", "text", "text_sha256", "n_chars"}),
                TEXTS_FILE,
            )
            length_records = load_cad_length_index(manifest_dir)
            selection = generate_selection(
                text_records,
                length_records,
                texts_sha256=sha256_file(manifest_dir / TEXTS_FILE),
                cad_length_sha256=sha256_file(manifest_dir / CAD_LENGTH_FILE),
                pair_manifest_sha256=sha256_file(manifest_dir / PAIR_MANIFEST_FILE),
            )
            payload = json.dumps(selection, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
            if args.out:
                out_path = Path(args.out)
                if out_path.exists():
                    raise RunnerError(f"refusing to overwrite {out_path}")
                out_path.write_text(payload, encoding="utf-8", newline="\n")
            else:
                print(payload)
            return 0

        if args.mode == "eta":
            if len(args.card_result) != 2:
                raise RunnerError("eta requires exactly two --card-result files (one per card; "
                                  "a single card is never scaled by an ideal 2x)")
            results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.card_result]
            tier = args.production_batch_size
            if tier is None:
                raise RunnerError("eta requires --production-batch-size (which tier's rates to use)")
            repo_root = Path(args.repo_root).resolve()
            reviewed = _load_expected_pair_hashes(repo_root / EXPECTED_PAIR_HASHES_REL)
            load_pair_manifest(manifest_dir, expected_sha256=reviewed)
            selection_path = repo_root / SELECTION_CONFIG_REL
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            validate_eta_card_results(
                results, args.card_result, tier,
                expected_manifest_sha256=reviewed[PAIR_MANIFEST_FILE],
                expected_selection_sha256=sha256_file(selection_path),
                expected_cells=set(selection["throughput_strata"]),
            )
            per_card_rates = []
            for result in results:
                by_tier = (result.get("cell_e2e_pairs_per_s_by_tier") or {})
                cell_rates = {}
                for cell, tiers in by_tier.items():
                    rate = tiers.get(str(tier))
                    if rate is None:
                        raise RunnerError(f"cell {cell} lacks tier {tier} in a card result")
                    cell_rates[cell] = rate
                per_card_rates.append({
                    "schema_version": result.get("schema_version"),
                    "measurement_window": result.get("measurement_window"),
                    "cell_e2e_pairs_per_s": cell_rates,
                })
            combined = combine_two_card_results(per_card_rates[0], per_card_rates[1])
            text_records = read_jsonl_strict(
                manifest_dir / TEXTS_FILE,
                frozenset({"text_index", "text", "text_sha256", "n_chars"}),
                TEXTS_FILE,
            )
            cell_counts = full_matrix_cell_counts(text_records, load_cad_length_index(manifest_dir))
            overhead = args.overhead_seconds
            if overhead is None:
                overhead = sum(float(r.get("cold_start_s") or 0.0) for r in results)
            estimate = estimate_full_matrix_seconds(
                cell_counts, combined, overhead_seconds=overhead
            )
            estimate["inputs"] = {
                "card_results": [str(p) for p in args.card_result],
                "production_batch_size": tier,
                "overhead_seconds": overhead,
            }
            payload = json.dumps(estimate, indent=1, sort_keys=True) + "\n"
            if args.out:
                out_path = Path(args.out)
                if out_path.exists():
                    raise RunnerError(f"refusing to overwrite {out_path}")
                out_path.write_text(payload, encoding="utf-8", newline="\n")
            else:
                print(payload)
            return 0 if estimate["passes_seven_day_gate"] else 3

        # ---- forward-consuming modes (authorized + budget-charged) ----
        if not (args.checkpoint and args.bert_dir and args.cad_seq_zip):
            raise RunnerError("forward modes require --checkpoint/--bert-dir/--cad-seq-zip")
        if not args.expected_keys_manifest:
            raise RunnerError("forward modes require --expected-keys-manifest (never omitted)")
        if not args.out_dir:
            raise RunnerError("forward modes require --out-dir (outside the checkout)")
        if not args.selection:
            raise RunnerError("forward modes require --selection (committed reviewed config)")
        if args.mode == "throughput":
            tiers = parse_batch_tiers(args.batch_tiers, args.allow_large_batch)
        started_epoch = time.time()
        setup: Optional[dict[str, Any]] = None
        try:
            setup = _gated_setup(args)
            ledger: SharedBudgetLedger = setup["ledger"]
            invocation_id = setup["invocation_id"]
            model = setup["model"]
            store = setup["store"]
            texts = setup["texts"]
            selection = setup["selection"]
            device = torch.device(args.device)
            out_dir = setup["out_dir"]

            def charge(n: int) -> None:
                ledger.charge_pairs(invocation_id, n)
                ledger.heartbeat(invocation_id)

            if args.mode == "spot":
                entry = selection["spot"]
                pair = (int(entry["text_index"]), int(entry["cad_index"]))
                if ledger.remaining_pairs() < 1:
                    raise RunnerError("approved pair budget exhausted")
                charge(1)
                logits, scores = score_pairs_detailed(
                    model, store, texts, [pair], device, batch_size=1
                )
                n_real = int(store.get(pair[1])["n_real_tokens"])
                payload = {
                    "spot": entry,
                    "score": scores[0],
                    "logits_stats": logits_stats(logits[0]),
                    "token_detail": score_pair_token_diagnostic(
                        logits[0], store.get(pair[1])["cad_vec"]
                    ),
                    "termination": {
                        "terminal_start_prediction_row": n_real - 2,
                        "terminal_x_label": 1,
                        "note": "terminal row detail is token_detail.x[-1]: its x label is "
                                "START(1) and it is scored; its y factor is masked. Padding "
                                "rows contribute nothing (compare n_valid_* with the 271 "
                                "prediction rows in logits_stats.shape).",
                    },
                }
                out_path = _write_result(setup, "spot", started_epoch, payload)
                print(json.dumps({"status": "ok", "result": str(out_path)}))
                return 0

            if args.mode == "batch-consistency":
                pairs = _pairs_of(selection["ten_pair"])
                batches = [int(b) for b in args.consistency_batches.split(",") if b]
                planned = len(pairs) * (1 + len(batches))
                if planned > ledger.remaining_pairs():
                    raise RunnerError(
                        f"planned {planned} gate forwards exceed remaining budget "
                        f"{ledger.remaining_pairs()}"
                    )
                charge(len(pairs))
                reference = score_pairs_detailed(model, store, texts, pairs, device, batch_size=1)
                comparisons = []
                for b in batches:
                    charge(len(pairs))
                    alt = score_pairs_detailed(model, store, texts, pairs, device, batch_size=b)
                    comparisons.append(
                        {"batch_size": b,
                         "report": compare_gate_results(reference, alt, store, pairs)}
                    )
                passed = all(c["report"]["passed"] for c in comparisons)
                out_path = _write_result(
                    setup, "batch-consistency", started_epoch,
                    {"pairs": selection["ten_pair"], "comparisons": comparisons, "passed": passed},
                )
                print(json.dumps({"status": "ok" if passed else "gate_failed",
                                  "result": str(out_path)}))
                return 0 if passed else 3

            if args.mode in ("trim-check", "cache-check"):
                pairs = _pairs_of(selection["ten_pair"])
                planned = 2 * len(pairs)
                if planned > ledger.remaining_pairs():
                    raise RunnerError(
                        f"planned {planned} gate forwards exceed remaining budget "
                        f"{ledger.remaining_pairs()}"
                    )
                charge(len(pairs))
                reference = score_pairs_detailed(model, store, texts, pairs, device, batch_size=1)
                charge(len(pairs))
                alternate = score_pairs_detailed(
                    model, store, texts, pairs, device,
                    batch_size=len(pairs),
                    trim_mode="batch_max" if args.mode == "trim-check" else "none",
                    text_cache=args.mode == "cache-check",
                )
                report = compare_gate_results(reference, alternate, store, pairs)
                out_path = _write_result(
                    setup, args.mode, started_epoch,
                    {"pairs": selection["ten_pair"], "gate": args.mode, "report": report},
                )
                print(json.dumps({"status": "ok" if report["passed"] else "gate_failed",
                                  "result": str(out_path)}))
                return 0 if report["passed"] else 3

            if args.mode == "throughput":
                cells = {
                    key: _pairs_of(entries)
                    for key, entries in sorted(selection["throughput_strata"].items())
                }
                planned = plan_throughput_forwards(cells, tiers, args.warmup_passes, args.reps)
                if planned > ledger.remaining_pairs():
                    raise RunnerError(
                        f"planned {planned} throughput forwards exceed remaining budget "
                        f"{ledger.remaining_pairs()}"
                    )
                model_forward_device = device

                def score_batch_fn(batch: list[tuple[int, int]]):
                    t0 = time.perf_counter()
                    vec_dict, mask_cad_dict, cad_seq_len = assemble_batch(
                        store, [c for _, c in batch], model_forward_device, trim_mode="none"
                    )
                    t1 = time.perf_counter()
                    with torch.inference_mode():
                        pred, _t, _s = teacher_forced_logits(
                            model, [texts[t] for t, _ in batch], vec_dict, mask_cad_dict,
                            cad_seq_len=cad_seq_len,
                        )
                    logits_cpu = pred.to(torch.float32).cpu()
                    scores = two_head_factorized_score(
                        logits_cpu, vec_dict["cad_vec"].cpu(),
                        mask_cad_dict["key_padding_mask"].cpu(),
                    )
                    t2 = time.perf_counter()
                    return scores, {"preprocess_s": t1 - t0, "forward_score_s": t2 - t1}

                measurement = run_throughput_tiers(
                    score_batch_fn,
                    cells,
                    batch_tiers=tiers,
                    warmup_passes=args.warmup_passes,
                    reps=args.reps,
                    device=args.device,
                    charge_fn=charge,
                    out_dir=out_dir,
                )
                out_path = _write_result(setup, "throughput", started_epoch, measurement)
                print(json.dumps({"status": "ok", "result": str(out_path),
                                  "escalation_stop": measurement["escalation_stop"]}))
                return 0

            raise RunnerError(f"unknown mode {args.mode!r}")
        except Exception:
            _fail_invocation(setup, "aborted")
            raise
    except (RunnerError, ValueError, PairIndexError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
