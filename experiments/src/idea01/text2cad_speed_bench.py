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
  * ``throughput`` -- G2 production protocol (repaired 2026-10-01): the
    committed G2 selection's REAL adjacent-CAD production batches (full
    tiers plus honest tails, never undersized stand-ins), batch tiers 1/2/4/8
    (16/32 only when explicitly requested AND ``--allow-large-batch``; OOM at
    a tier is recorded and stops escalation), >=2 warmup passes, >=3 timed
    reps per cell, per-(tier, cell, rep) measured windows with barrier waits
    recorded separately (``--sync-dir``/``--sync-group`` synchronize the two
    card processes at every sync point; peer failure stops both sides).
    Every forwarded pair (warmups and repeats included) is charged to the
    shared approval ledger BEFORE it forwards.
  * ``audit-text-lengths`` -- pure CPU, tokenizer ONLY (never the BERT
    model): real BERT token lengths (special tokens included, no truncation)
    for every unique text, bound to the texts/artifact SHAs, output outside
    the checkout.
  * ``select-g2`` -- pure CPU: build the G2 selection config (BERT-token
    text strata, production batch windows and exact shard weights, legacy
    spot/ten_pair preserved, new 64-pair profile grid) from the audited
    token lengths; LF output, never overwrites, binds all input SHAs.
  * ``profile-consistency`` -- extended G1: the fixed 64-pair profile grid
    (8 token-length x 8 CAD-length quantiles) through the batch=1 reference
    and ten fixed batch16/32 + trim32/cache32 + trim+cache variants (704
    scored pairs), pre-registered tolerances, per-variant valid-row logits
    persisted for independent recomputation; any failure stops with raw
    evidence and a non-zero exit. Passing does NOT approve production
    optimizations.
  * ``eta-g2``    -- pure CPU: the sharded full-matrix ETA -- each card's
    exact shard cell weights priced at that cell's slowest-rep per-batch
    E2E, per-card cold starts, the SLOWER card's completion as wall clock
    (never pooled ideal-2x), per-rep 80% window-overlap evidence required,
    overhead never below the measured cold starts, >=1.30 safety factor,
    exact 29,319,624-pair coverage and the 604,800 s seven-day gate.

Candidate-side prefix caching (decoder layers 0-1 have no cross-attention) is
NOT implemented in this stage: it needs its own same-pair equivalence gate
design and review first (B1 review condition 2); only the tolerance/gate
protocol is established here.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from .cadmium_runner import RunnerError  # audited shared error type
from .text2cad_bench_protocol import (  # G2 repair: pure-CPU protocol module
    G2_RESULT_SCHEMA_VERSION,
    G2_SELECTION_SCHEMA_VERSION,
    SHARD_BY_DEVICE,
    SyncBarrier,
    build_g2_selection,
    build_text_token_audit,
    build_tier_plans,
    estimate_sharded_eta,
    plan_production_forwards,
    tier_plans_sha256,
    token_lengths_from_audit,
    verify_g2_selection,
    dual_shard_plan_binding,
)
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
from .text2cad_port.loader import PINNED_BERT_FILE_SHA256, PINNED_CHECKPOINT_SHA256, verify_bert_asset_shas
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

#: Committed (reviewed) G2 selection config the G2-repair forward/ETA entries
#: pin by path. NOT committed yet: it is generated by ``select-g2`` from the
#: audited real BERT token lengths and then reviewed/committed by Codex --
#: until it exists, every mode that pins it fails closed (which is the
#: intended state for this implementation-only stage).
G2_SELECTION_CONFIG_REL = "experiments/configs/text2cad_g2_pair_selection_20261001.json"

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
    """Only the reviewed tiers can reach a forward; large tiers need HBM approval.

    G2 repair: ``allow_large_batch`` only PERMITS the explicitly requested
    16/32 tiers -- it never auto-appends tiers the operator did not ask for
    (the old behaviour silently ran 16/32 on every large-batch-approved run).
    """
    try:
        tiers = [int(t) for t in raw.split(",") if t]
    except ValueError as exc:
        raise RunnerError(f"invalid batch tiers {raw!r}") from exc
    allowed = set(DEFAULT_BATCH_TIERS) | (set(LARGE_BATCH_TIERS) if allow_large_batch else set())
    if not tiers or len(tiers) != len(set(tiers)) or not set(tiers) <= allowed:
        raise RunnerError(f"batch tiers {tiers} are not approved; allowed: {sorted(allowed)}")
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


FAILED_LOGITS_SCHEMA = "text2cad-failed-logits-v1"


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
    cache_stats_out: Optional[dict[str, Any]] = None,
    evidence_dir=None,
    evidence_label: str = "",
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Score ``(text_index, cad_index)`` pairs and KEEP the CPU FP32 logits.

    Returns ``(logits_per_pair [(T,2,267)], score_dicts)`` with T=271 on the
    reference path (or batch-max length under the trim gate). Used only by
    the G1 gates where pair counts are tiny; the matrix path discards logits
    after scoring. When ``text_cache`` is set, the model's embedder is
    temporarily wrapped in the unique-text cache and restored afterwards;
    ``cache_stats_out`` (if given) receives the cache's hit/miss counters.
    If a batch's forward/score raises (e.g. non-finite logits) and
    ``evidence_dir`` is set, the failing batch's RAW logits, pair ids and
    batch ordinal are persisted there BEFORE the error propagates (R2 fix 8)
    -- a failure never leaves only a bare exception.
    """
    original_embedder = model.base_text_embedder
    cache: Optional[BertTextCache] = None
    if text_cache:
        cache = BertTextCache(original_embedder)
        model.base_text_embedder = _CachedEmbedder(cache)
    try:
        all_logits: list[torch.Tensor] = []
        all_scores: list[dict[str, Any]] = []
        for start in range(0, len(pair_list), batch_size):
            chunk = pair_list[start : start + batch_size]
            batch_texts = [texts[t] for t, _ in chunk]
            batch_cads = [c for _, c in chunk]
            logits_cpu: Optional[torch.Tensor] = None
            batch_ordinal = start // batch_size
            try:
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
            except Exception as exc:
                if evidence_dir is not None:
                    payload = {
                        "schema_version": FAILED_LOGITS_SCHEMA,
                        "variant": evidence_label,
                        "batch_ordinal": batch_ordinal,
                        "pairs": [[int(t), int(c)] for t, c in chunk],
                        "logits": logits_cpu,  # None if the forward itself died
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    label = evidence_label or "run"
                    _atomic_publish_bytes(
                        Path(evidence_dir) / f"failed_logits_{label}_b{batch_ordinal}.pt",
                        buffer.getvalue(),
                    )
                raise
            for row in range(logits_cpu.shape[0]):
                all_logits.append(logits_cpu[row])
            all_scores.extend(scores)
        if cache_stats_out is not None and cache is not None:
            cache_stats_out.update(cache.stats())
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
    """Peak HBM of the EXPLICIT device; an NPU read failure raises (never None).

    ``None`` is returned only on non-NPU devices (CPU fake drivers/tests) --
    on a real NPU a failed read is a hard error, so a missing value can never
    masquerade as valid HBM evidence.
    """
    if device.startswith("npu") and hasattr(torch, "npu"):
        return int(torch.npu.max_memory_allocated(device))
    return None


def reset_peak_device_memory(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)


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
# G2 repair: profile-consistency (704-pair extended G1 equivalence entry)
# ---------------------------------------------------------------------------

#: Fixed variant table (name, batch_size, trim_mode, text_cache). The
#: reference is always batch=1 full-length uncached; the CLI has NO switches
#: for the candidate pairs or the tolerances.
PROFILE_REFERENCE_VARIANT = ("reference_b1", 1, "none", False)
PROFILE_VARIANTS = (
    ("batch16", 16, "none", False),
    ("batch32", 32, "none", False),
    ("trim32", 32, "batch_max", False),
    ("cache32", 32, "none", True),
    ("trim_cache_b1", 1, "batch_max", True),
    ("trim_cache_b2", 2, "batch_max", True),
    ("trim_cache_b4", 4, "batch_max", True),
    ("trim_cache_b8", 8, "batch_max", True),
    ("trim_cache_b16", 16, "batch_max", True),
    ("trim_cache_b32", 32, "batch_max", True),
)
PROFILE_LOGITS_SCHEMA = "text2cad-profile-logits-v1"


def plan_profile_consistency_forwards(n_pairs: int) -> int:
    """Exact scored pairs: reference + every variant over the same fixed pairs."""
    return int(n_pairs) * (1 + len(PROFILE_VARIANTS))


def _atomic_publish_bytes(path: Path, data: bytes) -> None:
    """Binary counterpart of atomic_publish (hard-link, fail-if-exists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RunnerError(f"output already exists, refusing overwrite: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RunnerError(f"output appeared during run, refusing overwrite: {path}") from exc
        except OSError as exc:
            raise RunnerError(f"cannot atomically publish {path}: {exc}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _save_profile_logits(
    out_dir, variant: str, pair_list: Sequence[tuple[int, int]],
    logits: Sequence[torch.Tensor], store: ZipCandidateStore,
) -> str:
    """Persist per-pair VALID-ROW logits (weights_only-loadable) for Codex.

    Only prediction rows whose target is a real token are kept -- the same
    rows the gate compares -- so Codex can recompute every number from raw
    evidence instead of trusting booleans.
    """
    rows = []
    for tensor, (_t, cad_index) in zip(logits, pair_list):
        n_real = int(store.get(cad_index)["n_real_tokens"])
        rows.append(tensor[: n_real - 1].to(torch.float32).cpu().clone())
    payload = {
        "schema_version": PROFILE_LOGITS_SCHEMA,
        "variant": variant,
        "pairs": [[int(t), int(c)] for t, c in pair_list],
        "logits_valid_rows": rows,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    name = f"profile_logits_{variant}.pt"
    _atomic_publish_bytes(Path(out_dir) / name, buffer.getvalue())
    return name


def _profile_variant_report(
    name: str,
    batch_size: int,
    trim_mode: str,
    text_cache: bool,
    reference: tuple[list[torch.Tensor], list[dict[str, Any]]],
    alternate: tuple[list[torch.Tensor], list[dict[str, Any]]],
    store: ZipCandidateStore,
    pair_list: Sequence[tuple[int, int]],
    cache_stats: Optional[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    """Gate comparison + normalized diffs + finiteness for one variant."""
    gate = compare_gate_results(reference, alternate, store, pair_list)
    ref_logits, ref_scores = reference
    alt_logits, alt_scores = alternate
    per_pair: list[dict[str, Any]] = []
    max_abs_score = 0.0
    max_norm_score = 0.0
    max_abs_logit = 0.0
    max_norm_logit = 0.0
    all_finite = True
    for i, (text_index, cad_index) in enumerate(pair_list):
        n_real = int(store.get(cad_index)["n_real_tokens"])
        n_rows = n_real - 1
        ref = ref_logits[i][:n_rows].to(torch.float64)
        alt = alt_logits[i][:n_rows].to(torch.float64)
        if ref.shape != alt.shape or not (torch.isfinite(ref).all() and torch.isfinite(alt).all()):
            all_finite = False
        if ref.shape == alt.shape:
            diff = (ref - alt).abs()
            tol = G1_LOGITS_ATOL + G1_LOGITS_RTOL * ref.abs()
            if diff.numel():
                max_abs_logit = max(max_abs_logit, float(diff.max()))
                max_norm_logit = max(max_norm_logit, float((diff / tol).max()))
        rs, alts = ref_scores[i], alt_scores[i]
        for key in ("score_x", "score_y", "score_total", "mean_x", "mean_y", "mean_total"):
            if not (math.isfinite(rs[key]) and math.isfinite(alts[key])):
                all_finite = False
        # R2: per-head AND total diffs; the summary maxima aggregate over all
        # three components so a one-head drift can never cancel out
        head_diffs = {}
        head_norms = {}
        for key_name in ("score_x", "score_y", "score_total"):
            d = abs(rs[key_name] - alts[key_name])
            head_diffs[key_name] = d
            head_norms[key_name] = d / (G1_SCORE_ATOL + G1_SCORE_RTOL * abs(rs[key_name]))
        max_abs_score = max(max_abs_score, *head_diffs.values())
        max_norm_score = max(max_norm_score, *head_norms.values())
        gate_pair = gate["per_pair"][i] if i < len(gate["per_pair"]) else {}
        per_pair.append(
            {
                "pair": i,
                "text_index": text_index,
                "cad_index": cad_index,
                "score_x_ref": rs["score_x"],
                "score_y_ref": rs["score_y"],
                "score_total_ref": rs["score_total"],
                "score_x_alt": alts["score_x"],
                "score_y_alt": alts["score_y"],
                "score_total_alt": alts["score_total"],
                "abs_score_diff_x": head_diffs["score_x"],
                "abs_score_diff_y": head_diffs["score_y"],
                "abs_score_diff_total": head_diffs["score_total"],
                "normalized_score_diff_x": head_norms["score_x"],
                "normalized_score_diff_y": head_norms["score_y"],
                "normalized_score_diff_total": head_norms["score_total"],
                "n_valid_x_ref": rs["n_valid_x"],
                "n_valid_y_ref": rs["n_valid_y"],
                "n_valid_total_ref": rs["n_valid_total"],
                "n_valid_x_alt": alts["n_valid_x"],
                "n_valid_y_alt": alts["n_valid_y"],
                "n_valid_total_alt": alts["n_valid_total"],
                "valid_seq_len_ref": rs["valid_seq_len"],
                "n_real_tokens_ref": rs["n_real_tokens"],
                "max_abs_logit_diff": gate_pair.get("max_abs_logit_diff"),
                "n_logit_violations": gate_pair.get("n_logit_violations"),
                "passed": gate_pair.get("passed"),
            }
        )
    n_pairs = len(pair_list)
    real_sizes = [
        min(batch_size, n_pairs - start) for start in range(0, n_pairs, batch_size)
    ]
    return {
        "variant": name,
        "batch_size_requested": batch_size,
        "real_batch_sizes": real_sizes,
        "trim_mode": trim_mode,
        "text_cache": text_cache,
        "cache_stats": cache_stats or None,
        "elapsed_s": elapsed_s,
        "all_finite": all_finite,
        "max_abs_score_diff": max_abs_score,
        "max_normalized_score_diff": max_norm_score,
        "max_abs_logit_diff": max_abs_logit,
        "max_normalized_logit_diff": max_norm_logit,
        "per_pair": per_pair,
        "gate": gate,
        "passed": bool(gate["passed"]) and all_finite,
    }


def run_profile_consistency(
    model,
    store: ZipCandidateStore,
    texts: Sequence[str],
    pair_list: Sequence[tuple[int, int]],
    device: torch.device,
    *,
    charge_fn: Optional[Callable[[int], None]] = None,
    out_dir=None,
) -> dict[str, Any]:
    """The 704-pair extended G1 equivalence entry (64 fixed pairs x 11 runs).

    Reference batch=1 full-length uncached, then the ten fixed variants
    (batch16/32, trim32, cache32, trim+cache at 1/2/4/8/16/32), each charged
    BEFORE its forwards. Every variant's valid-row logits are persisted for
    independent recomputation; the first failed comparison stops the run with
    the raw evidence on disk. Passing this gate does NOT approve any
    production optimization -- throughput and matrix stay reference-only
    until Codex explicitly extends authorization.
    """
    if out_dir is None:
        raise RunnerError("profile-consistency requires an out_dir for the logits evidence")
    n_pairs = len(pair_list)

    def charge(n: int) -> None:
        if charge_fn is not None:
            charge_fn(n)

    started = time.perf_counter()
    charge(n_pairs)
    name, b, trim, cache = PROFILE_REFERENCE_VARIANT
    reference = score_pairs_detailed(
        model, store, texts, pair_list, device, batch_size=b, trim_mode=trim,
        text_cache=cache, evidence_dir=out_dir, evidence_label=name,
    )
    reference_file = _save_profile_logits(out_dir, name, pair_list, reference[0], store)
    variants: list[dict[str, Any]] = []
    passed = True
    for name, b, trim, cache in PROFILE_VARIANTS:
        stats: dict[str, Any] = {}
        variant_start = time.perf_counter()
        charge(n_pairs)
        alternate = score_pairs_detailed(
            model, store, texts, pair_list, device,
            batch_size=b, trim_mode=trim, text_cache=cache, cache_stats_out=stats,
            evidence_dir=out_dir, evidence_label=name,
        )
        logits_file = _save_profile_logits(out_dir, name, pair_list, alternate[0], store)
        report = _profile_variant_report(
            name, b, trim, cache, reference, alternate, store, pair_list,
            stats if cache else None, time.perf_counter() - variant_start,
        )
        report["logits_file"] = logits_file
        variants.append(report)
        if not report["passed"]:
            passed = False
            break
    scored = n_pairs * (1 + len(variants))
    return {
        "passed": passed,
        "n_pairs": n_pairs,
        "total_scored_pairs": scored,
        "reference": {
            "variant": PROFILE_REFERENCE_VARIANT[0],
            "batch_size": 1,
            "trim_mode": "none",
            "text_cache": False,
            "logits_file": reference_file,
        },
        "variants": variants,
        "elapsed_s": time.perf_counter() - started,
        "note": (
            "equivalence on the fixed 64-pair profile grid within the "
            "pre-registered G1 tolerances; this gate does NOT approve "
            "production optimizations (throughput/matrix stay reference-only)"
        ),
    }


# ---------------------------------------------------------------------------
# G2 repair: real production-batch throughput driver
# ---------------------------------------------------------------------------

#: Contract floors for the production protocol (G2 repair).
G2_MIN_REPS = 3
G2_DEFAULT_REPS = 3


def run_production_batch_tiers(
    score_batch_fn: Callable[[list[tuple[int, int]]], tuple[list[dict[str, Any]], dict[str, float]]],
    tier_plans: Sequence[dict[str, Any]],
    *,
    warmup_passes: int = DEFAULT_WARMUP_PASSES,
    reps: int = G2_DEFAULT_REPS,
    device: str = "cpu",
    charge_fn: Optional[Callable[[int], None]] = None,
    out_dir=None,
    sync: Optional[SyncBarrier] = None,
    batch_facts_fn: Optional[Callable[[list[tuple[int, int]]], dict[str, Any]]] = None,
    local_plan_sha256: Optional[str] = None,
    expected_peer_local_plan_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Measure per-(tier, cell, rep) REAL production batches.

    The driver consumes ``build_tier_plans`` output VERBATIM: pre-built
    adjacent-CAD batches, one cell after another in the canonical sync order.
    A full-layer batch shorter than its tier (the diagnosed nominal-32 /
    real-20 symptom) refuses the run, as does a tail batch of the wrong size.
    Per rep it records: real batch sizes, pair order, max CAD length, actual
    text token/padding widths (via ``batch_facts_fn``), epoch start/end of the
    measured window (barrier waits excluded -- ``sync_wait_s`` is recorded
    separately and counts as card time via the invocation clock), phase
    timings, write time, E2E and per-batch E2E. >=2 warmups and >=3 timed
    reps are contract floors. OOM signals the peer, records ``status="oom"``
    and stops escalation; any other error signals the peer and aborts. When
    ``local_plan_sha256`` is set it is announced at every sync point and the
    peer's announced local plan hash must equal
    ``expected_peer_local_plan_sha256`` -- a card measuring the wrong shard's
    profile is refused at the barrier.
    """
    if warmup_passes < 2:
        raise RunnerError(f"warmup_passes must be >= 2 (contract), got {warmup_passes}")
    if reps < G2_MIN_REPS:
        raise RunnerError(f"reps must be >= {G2_MIN_REPS} (contract), got {reps}")
    for plan in tier_plans:
        tier = int(plan["batch_size"])
        for cell in plan["cells"]:
            if not cell["batches"]:
                raise RunnerError(f"cell {cell['cell']} has no sample batches")
            for batch in cell["batches"]:
                expected = cell["batch_size"] if cell["is_tail"] else tier
                if len(batch) != expected:
                    raise RunnerError(
                        f"cell {cell['cell']}: batch size {len(batch)} != required "
                        f"{expected}; refusing undersized/padded production batches"
                    )

    def signal(point: str, kind: str, exc: BaseException) -> None:
        if sync is not None:
            try:
                sync.signal_failure(point, kind, f"{type(exc).__name__}: {exc}")
            except RunnerError:
                pass

    tier_results: list[dict[str, Any]] = []
    escalation_stop: Optional[str] = None
    for plan in tier_plans:
        tier = int(plan["batch_size"])
        if escalation_stop is not None:
            tier_results.append({"batch_size": tier, "status": f"skipped_after_{escalation_stop}"})
            continue
        tier_record: dict[str, Any] = {"batch_size": tier, "status": "ok", "cells": {}}
        reset_peak_device_memory(device)
        try:
            for cell in plan["cells"]:
                batches = cell["batches"]
                # Warmup (charged, untimed): EVERY cell gets its own
                # warmup_passes over its first two sample batches (R2 fix 7)
                # -- the same enumeration the budget plan counts.
                for _ in range(warmup_passes):
                    for batch in batches[:2]:
                        if charge_fn is not None:
                            charge_fn(len(batch))
                        score_batch_fn(batch)
                cell_reps: list[dict[str, Any]] = []
                for rep in range(reps):
                    point = f"t{tier}_{cell['point']}_r{rep}"
                    sync_wait_s = 0.0
                    if sync is not None:
                        wait_start = time.perf_counter()
                        peer = sync.arrive(
                            point,
                            {
                                "n_batches": len(batches),
                                "n_pairs": sum(len(b) for b in batches),
                                "batch_sizes": [len(b) for b in batches],
                                "local_plan_sha256": local_plan_sha256,
                            },
                        )
                        sync_wait_s = time.perf_counter() - wait_start
                        if (
                            expected_peer_local_plan_sha256 is not None
                            and peer.get("local_plan_sha256") != expected_peer_local_plan_sha256
                        ):
                            raise RunnerError(
                                f"peer local plan SHA {peer.get('local_plan_sha256')!r} != "
                                f"expected {expected_peer_local_plan_sha256!r} at {point}: "
                                "the peer card is not measuring the expected other-shard "
                                "profile"
                            )
                    start_epoch = time.time()
                    rep_start = time.perf_counter()
                    preprocess_s = 0.0
                    forward_score_s = 0.0
                    valid_items = 0
                    rep_records: list[dict[str, Any]] = []
                    batch_records: list[dict[str, Any]] = []
                    for ordinal, batch in enumerate(batches):
                        if charge_fn is not None:
                            charge_fn(len(batch))
                        scores, timing = score_batch_fn(batch)
                        preprocess_s += float(timing["preprocess_s"])
                        forward_score_s += float(timing["forward_score_s"])
                        valid_items += sum(int(s["n_valid_total"]) for s in scores)
                        rep_records.extend(scores)
                        facts = batch_facts_fn(batch) if batch_facts_fn else {}
                        batch_records.append(
                            {
                                "batch_ordinal": ordinal,
                                "size": len(batch),
                                "pairs": [[int(t), int(c)] for t, c in batch],
                                "max_cad_len": facts.get("max_cad_len"),
                                "text_tokens": facts.get("text_tokens"),
                                "text_pad_width": facts.get("text_pad_width"),
                            }
                        )
                    write_start = time.perf_counter()
                    if out_dir is not None:
                        atomic_publish(
                            Path(out_dir) / f"bench_records_t{tier}_{cell['cell']}_r{rep}.jsonl",
                            [json.dumps(s, ensure_ascii=False) for s in rep_records],
                        )
                    else:
                        json.dumps(rep_records)  # same serialization cost in tests
                    write_s = time.perf_counter() - write_start
                    e2e_s = time.perf_counter() - rep_start
                    cell_reps.append(
                        {
                            "rep": rep,
                            "point": cell["point"],
                            "window": {"start_epoch": start_epoch, "end_epoch": time.time()},
                            "sync_wait_s": sync_wait_s,
                            "n_batches": len(batches),
                            "n_pairs": sum(len(b) for b in batches),
                            "preprocess_s": preprocess_s,
                            "forward_score_s": forward_score_s,
                            "write_s": write_s,
                            "e2e_s": e2e_s,
                            "per_batch_e2e_s": e2e_s / len(batches),
                            "valid_items_per_s_e2e": valid_items / e2e_s if e2e_s > 0 else None,
                            "batches": batch_records,
                        }
                    )
                per_batch = [r["per_batch_e2e_s"] for r in cell_reps]
                tier_record["cells"][cell["cell"]] = {
                    "point": cell["point"],
                    "shard_id": cell["shard_id"],
                    "text_tier": cell["text_tier"],
                    "layer": cell["layer"],
                    "batch_size": cell["batch_size"],
                    "is_tail": cell["is_tail"],
                    "reps": cell_reps,
                    "per_batch_e2e_s_reps": per_batch,
                    # conservative: the SLOWEST full rep prices this cell
                    "per_batch_e2e_s_max": max(per_batch),
                }
            tier_record["peak_device_memory_bytes"] = peak_device_memory_bytes(device)
            tier_record["hbm_device"] = device
        except RuntimeError as exc:
            if OOM_PATTERN in str(exc).lower():
                signal(f"t{tier}_run", "oom", exc)
                tier_record["status"] = "oom"
                tier_record["error"] = str(exc)[:200]
                tier_record["peak_device_memory_bytes"] = peak_device_memory_bytes(device)
                tier_record["hbm_device"] = device
                tier_results.append(tier_record)
                escalation_stop = "oom"
                continue
            signal(f"t{tier}_run", "error", exc)
            raise
        except Exception as exc:  # noqa: BLE001 -- peer must be told before aborting
            signal(f"t{tier}_run", "error", exc)
            raise
        tier_results.append(tier_record)
    return {
        "tiers": tier_results,
        "warmup_passes": warmup_passes,
        "reps": reps,
        "escalation_stop": escalation_stop,
        "plan_sha256": tier_plans_sha256(tier_plans),
    }


# ---------------------------------------------------------------------------
# G2 repair: pure-CPU token audit + selection generation entries
# ---------------------------------------------------------------------------


def _load_g2_manifest_records(manifest_dir) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load + internally verify the pair-manifest artifacts the G2 selection needs."""
    manifest_dir = Path(manifest_dir)
    load_pair_manifest(manifest_dir)  # schema + artifact SHA self-consistency
    text_records = read_jsonl_strict(
        manifest_dir / TEXTS_FILE,
        frozenset({"text_index", "text", "text_sha256", "n_chars"}),
        TEXTS_FILE,
    )
    length_records = load_cad_length_index(manifest_dir)
    return text_records, length_records


def _write_config_lf(out_path, payload: str, label: str) -> Path:
    out = Path(out_path)
    if out.exists():
        raise RunnerError(f"refusing to overwrite {out} ({label})")
    out.write_text(payload, encoding="utf-8", newline="\n")
    return out


def run_audit_text_lengths(args) -> Path:
    """``audit-text-lengths``: real BERT token lengths for every unique text.

    Pure CPU; loads ONLY the pinned local BertTokenizer (never the BERT
    model). The output path is checked (outside checkout, no overwrite)
    BEFORE any tokenizer import so refusal paths never touch transformers.
    """
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        raise RunnerError("audit-text-lengths requires --out (outside the checkout)")
    if out_path.exists():
        raise RunnerError(f"refusing to overwrite {out_path} (token audit)")
    repo_root = Path(args.repo_root).resolve()
    ensure_outside_checkout(out_path, repo_root, "token audit output")

    manifest_dir = Path(args.pair_manifest_dir)
    text_records, _lengths = _load_g2_manifest_records(manifest_dir)
    bert_report = verify_bert_asset_shas(args.bert_dir)
    bert_shas = {name: info["sha256"] for name, info in bert_report.items()}

    from transformers import BertTokenizer  # lazy: tokenizer ONLY, never BertModel

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_dir), local_files_only=True)

    def token_count_of(text: str) -> int:
        # attention_mask sum == true token count incl. [CLS]/[SEP]; no padding,
        # no truncation (same measurement as the frozen runner's dry-run audit)
        return int(sum(tokenizer(text)["attention_mask"]))

    audit = build_text_token_audit(
        text_records,
        token_count_of,
        texts_jsonl_sha256=sha256_file(manifest_dir / TEXTS_FILE),
        bert_file_sha256=bert_shas,
    )
    payload = json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    return _write_config_lf(out_path, payload, "token audit")


def run_select_g2(args) -> Path:
    """``select-g2``: generate the G2 selection config from the audited lengths.

    Pure CPU. Binds the live pair-manifest artifact SHAs, the token-audit file
    SHA and the pinned BERT file SHAs into the config provenance; LF output;
    never overwrites. The produced file is a REVIEW CANDIDATE -- forward modes
    pin the committed path (G2_SELECTION_CONFIG_REL) and re-verify everything.
    """
    if not args.token_audit:
        raise RunnerError("select-g2 requires --token-audit (derived audit JSON)")
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        raise RunnerError("select-g2 requires --out")
    if out_path.exists():
        raise RunnerError(f"refusing to overwrite {out_path} (g2 selection)")

    manifest_dir = Path(args.pair_manifest_dir)
    text_records, length_records = _load_g2_manifest_records(manifest_dir)
    audit = json.loads(Path(args.token_audit).read_text(encoding="utf-8"))
    token_lengths = token_lengths_from_audit(
        audit,
        expected_texts_sha256=sha256_file(manifest_dir / TEXTS_FILE),
        expected_bert_file_sha256=dict(PINNED_BERT_FILE_SHA256),
        n_texts=len(text_records),
    )
    selection = build_g2_selection(text_records, length_records, token_lengths)
    selection["provenance"] = {
        "texts_jsonl_sha256": sha256_file(manifest_dir / TEXTS_FILE),
        "cad_length_index_sha256": sha256_file(manifest_dir / CAD_LENGTH_FILE),
        "pair_manifest_sha256": sha256_file(manifest_dir / PAIR_MANIFEST_FILE),
        "token_audit_sha256": sha256_file(args.token_audit),
        "bert_file_sha256": dict(sorted(PINNED_BERT_FILE_SHA256.items())),
        "generated_by": "text2cad_speed_bench.run_select_g2 (CPU, zero forward)",
    }
    payload = json.dumps(selection, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    return _write_config_lf(out_path, payload, "g2 selection")


def run_eta_g2(args) -> tuple[dict[str, Any], Path | None]:
    """``eta-g2``: sharded slower-card full-matrix ETA from two card results.

    Pure CPU. Binds the committed G2 selection (pinned by path, re-verified
    against the live manifest) and the reviewed pair fingerprints; each card
    result must be a v2 throughput result bound to THIS selection, the
    reviewed manifest, the pinned checkpoint and the frozen reference
    configuration. The estimator itself (``estimate_sharded_eta``) adds the
    exact-weights / slower-card / overlap / overhead rules. Returns the
    estimate and the written path (None when printed to stdout).
    """
    if len(args.card_result) != 2:
        raise RunnerError(
            "eta-g2 requires exactly two --card-result files (one per card; a "
            "single card is never scaled by an ideal 2x)"
        )
    tier = args.production_batch_size
    if tier is None:
        raise RunnerError("eta-g2 requires --production-batch-size (which tier to price)")
    paths = [Path(p).resolve() for p in args.card_result]
    if paths[0] == paths[1]:
        raise RunnerError("eta-g2 requires two distinct card result files")
    results = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

    repo_root = Path(args.repo_root).resolve()
    reviewed = _load_expected_pair_hashes(repo_root / EXPECTED_PAIR_HASHES_REL)
    manifest_dir = Path(args.pair_manifest_dir)
    manifest = load_pair_manifest(manifest_dir, expected_sha256=reviewed)
    pair_space = PairSpace(
        n_texts=manifest["counts"]["n_unique_text"], n_cads=manifest["counts"]["n_gallery"]
    )
    selection_path = repo_root / G2_SELECTION_CONFIG_REL
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    verify_g2_selection_against_manifest(selection, manifest_dir, pair_space)
    # R2 fix 6: the actual token-audit file must back the selection
    verify_g2_token_audit(selection, args.token_audit, manifest_dir)
    selection_sha = sha256_file(selection_path)

    results_by_device: dict[str, dict[str, Any]] = {}
    for path, result in zip(paths, results):
        env = result.get("environment") or {}
        device = env.get("device")
        if device not in SHARD_BY_DEVICE or device in results_by_device:
            raise RunnerError(
                f"eta-g2 card results must be one per NPU device, got {device!r} ({path})"
            )
        if result.get("selection_sha256") != selection_sha:
            raise RunnerError(
                f"{device}: result is not bound to the committed G2 selection"
            )
        if env.get("pair_manifest_sha256") != reviewed[PAIR_MANIFEST_FILE]:
            raise RunnerError(f"{device}: result does not match the reviewed pair manifest")
        if env.get("checkpoint_sha256") != PINNED_CHECKPOINT_SHA256:
            raise RunnerError(f"{device}: result does not use the pinned checkpoint")
        if env.get("score_name") != SCORE_NAME or env.get("spec_version") != SCORE_SPEC_VERSION:
            raise RunnerError(f"{device}: result uses an unreviewed score definition")
        git_sha = env.get("git_sha")
        auth_sha = (result.get("authorization") or {}).get("approved_sha")
        budget_sha = (result.get("budget") or {}).get("approved_sha")
        if not git_sha or auth_sha != git_sha or budget_sha != git_sha:
            raise RunnerError(
                f"{device}: result code SHA differs from its authorization or budget"
            )
        results_by_device[device] = result

    estimate = estimate_sharded_eta(
        selection, results_by_device, int(tier), overhead_seconds=args.overhead_seconds
    )
    estimate["inputs"] = {
        "card_results": [str(p) for p in paths],
        "production_batch_size": int(tier),
        "overhead_seconds": args.overhead_seconds,
        "g2_selection_sha256": selection_sha,
    }
    written: Path | None = None
    payload = json.dumps(estimate, indent=1, sort_keys=True) + "\n"
    if args.out:
        written = _write_config_lf(Path(args.out), payload, "eta-g2 estimate")
    else:
        print(payload)
    return estimate, written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FORWARD_MODES = ("spot", "batch-consistency", "trim-check", "cache-check", "throughput", "profile-consistency")

#: New-protocol results (G2 production throughput, profile-consistency) carry
#: this schema; the legacy v1 ETA refuses them and the G2 ETA refuses v1 --
#: the two schemas are never interchangeable for G2 ETA purposes.
BENCH_RESULT_SCHEMA_VERSION_G2 = G2_RESULT_SCHEMA_VERSION


#: Exact provenance key set of a committed G2 selection (R2 fix 5 -- an
#: injected evaluation_map/qrels key anywhere in the file refuses it).
G2_PROVENANCE_KEYS = frozenset({
    "texts_jsonl_sha256",
    "cad_length_index_sha256",
    "pair_manifest_sha256",
    "token_audit_sha256",
    "bert_file_sha256",
    "generated_by",
})


def verify_g2_selection_against_manifest(
    selection: dict[str, Any],
    manifest_dir,
    pair_space: PairSpace,
) -> None:
    """G2 counterpart of the legacy selection verifier (fail-closed).

    Checks provenance against the LIVE artifacts (texts / CAD length index /
    pair manifest / pinned BERT files / token-audit binding), then re-derives
    every weight, sample and profile entry from the live records via
    ``verify_g2_selection``.
    """
    if selection.get("schema_version") != G2_SELECTION_SCHEMA_VERSION:
        raise RunnerError(
            f"selection schema {selection.get('schema_version')!r} != "
            f"{G2_SELECTION_SCHEMA_VERSION!r} (the G2 entries never accept the "
            "legacy selection schema)"
        )
    manifest_dir = Path(manifest_dir)
    provenance = selection.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(G2_PROVENANCE_KEYS):
        raise RunnerError(
            f"g2 selection provenance keys must be exactly {sorted(G2_PROVENANCE_KEYS)}, "
            f"got {sorted(provenance) if isinstance(provenance, dict) else provenance!r}"
        )
    live = {
        "texts_jsonl_sha256": sha256_file(manifest_dir / TEXTS_FILE),
        "cad_length_index_sha256": sha256_file(manifest_dir / CAD_LENGTH_FILE),
        "pair_manifest_sha256": sha256_file(manifest_dir / PAIR_MANIFEST_FILE),
    }
    for key, value in live.items():
        if provenance.get(key) != value:
            raise RunnerError(
                f"g2 selection provenance {key} {provenance.get(key)!r} != live {value!r}"
            )
    if provenance.get("bert_file_sha256") != dict(sorted(PINNED_BERT_FILE_SHA256.items())):
        raise RunnerError("g2 selection BERT asset binding differs from the pinned files")
    audit_sha = provenance.get("token_audit_sha256")
    if not isinstance(audit_sha, str) or len(audit_sha) != 64:
        raise RunnerError("g2 selection lacks its token-audit SHA binding")
    text_records = read_jsonl_strict(
        manifest_dir / TEXTS_FILE,
        frozenset({"text_index", "text", "text_sha256", "n_chars"}),
        TEXTS_FILE,
    )
    length_records = load_cad_length_index(manifest_dir)
    embedded = selection.get("text_token_lengths")
    if not isinstance(embedded, list) or len(embedded) != len(text_records):
        raise RunnerError("g2 selection token-length table does not cover the texts")
    verify_g2_selection(
        selection, text_records, length_records,
        {i: int(v) for i, v in enumerate(embedded)},
    )


def verify_g2_token_audit(
    selection: dict[str, Any],
    token_audit_path,
    manifest_dir,
) -> None:
    """Bind the ACTUAL token-audit file to the selection at runtime (R2 fix 6).

    The committed selection pins the audit by SHA; this check (run by every
    G2 entry that touches forwards or the ETA) requires the real file to
    exist, to hash to the pinned value, to carry the audited schema and the
    texts/BERT provenance, and for its per-text token lengths to equal the
    selection's embedded table -- a bare 64-hex string or a self-consistent
    embedded table alone is not accepted.
    """
    if not token_audit_path:
        raise RunnerError(
            "G2 entries require --token-audit: the reviewed derived token-length "
            "asset the selection was generated from"
        )
    path = Path(token_audit_path)
    if not path.is_file():
        raise RunnerError(f"token audit file {path} does not exist")
    provenance = selection.get("provenance") or {}
    expected_sha = provenance.get("token_audit_sha256")
    if sha256_file(path) != expected_sha:
        raise RunnerError(
            f"token audit file SHA does not match the selection binding {expected_sha!r}"
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    manifest_dir = Path(manifest_dir)
    embedded_table = selection.get("text_token_lengths") or []
    token_lengths = token_lengths_from_audit(
        audit,
        expected_texts_sha256=sha256_file(manifest_dir / TEXTS_FILE),
        expected_bert_file_sha256=dict(PINNED_BERT_FILE_SHA256),
        n_texts=len(embedded_table),
    )
    embedded = {i: int(v) for i, v in enumerate(embedded_table)}
    if token_lengths != embedded:
        raise RunnerError(
            "token audit contents differ from the selection's embedded table"
        )


def _pairs_of(entries: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(int(e["text_index"]), int(e["cad_index"])) for e in entries]


def _gated_setup(args, *, g2: bool = False) -> dict[str, Any]:
    """Shared fail-closed gate sequence for every forward-consuming mode.

    Same order as the matrix scorer CLI: reviewed fingerprints -> manifest ->
    selection pin+verify -> authorization (fetched origin/master) ->
    production path + committed manifests pinned by path -> revision paths ->
    shared budget ledger -> strict model load + candidate preload. Returns
    everything the modes need; NOTHING has forwarded when this returns.
    ``g2=True`` pins the committed G2 selection (new schema, BERT-token
    strata, production batches); ``g2=False`` pins the legacy G1 selection.
    """
    selection_rel = G2_SELECTION_CONFIG_REL if g2 else SELECTION_CONFIG_REL
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
    if selection_path != (repo_root / selection_rel).resolve():
        raise RunnerError(
            f"--selection must be the committed reviewed file {repo_root / selection_rel}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if g2:
        verify_g2_selection_against_manifest(selection, args.pair_manifest_dir, pair_space)
        # R2 fix 6: the actual token-audit file must back the selection
        verify_g2_token_audit(selection, args.token_audit, args.pair_manifest_dir)
    else:
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
        "selection_rel": selection_rel,
        "g2": g2,
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


def _write_result(
    setup: dict[str, Any],
    mode: str,
    started_epoch: float,
    payload: dict[str, Any],
    *,
    schema_version: str = BENCH_RESULT_SCHEMA_VERSION,
    exit_code: int = 0,
    note: str = "ok",
) -> Path:
    """Finalize the ledger invocation with the TRUE outcome and publish the result.

    A gate that failed (CLI exit 3) must record that failure in the ledger --
    the invocation never closes as ``exit_code=0`` when the run did not pass
    (R2 fix 8).
    """
    ledger: SharedBudgetLedger = setup["ledger"]
    invocation_id = setup["invocation_id"]
    ledger.finalize_invocation(invocation_id, exit_code=exit_code, note=note)
    result = {
        "schema_version": schema_version,
        "mode": mode,
        "invocation_id": invocation_id,
        "measurement_window": {"start_epoch": started_epoch, "end_epoch": time.time()},
        "cold_start_s": setup["cold_start_s"],
        "selection_sha256": sha256_file(setup["repo_root"] / setup["selection_rel"]),
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
                        choices=["select", "eta", "eta-g2", "audit-text-lengths", "select-g2", *FORWARD_MODES])
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
    parser.add_argument("--reps", type=int, default=G2_DEFAULT_REPS)
    parser.add_argument("--warmup-passes", type=int, default=DEFAULT_WARMUP_PASSES)
    parser.add_argument("--sync-dir", default=None,
                        help="throughput: two-process sync marker dir (outside checkout)")
    parser.add_argument("--sync-group", default=None,
                        help="throughput: shared sync group name of the two card processes")
    parser.add_argument("--production-batch-size", type=int, default=None,
                        help="eta: which tier's per-cell rates to use")
    parser.add_argument("--card-result", action="append", default=[],
                        help="eta: a per-card throughput result JSON (pass exactly two)")
    parser.add_argument("--overhead-seconds", type=float, default=None,
                        help="eta: cold start + IO + retry + write overhead; default uses the "
                             "results' measured cold starts")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default=None, help="select/eta/select-g2/audit: output JSON path")
    parser.add_argument("--token-audit", default=None,
                        help="select-g2: committed text token length audit JSON")
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

        if args.mode == "audit-text-lengths":
            if not args.bert_dir:
                raise RunnerError("audit-text-lengths requires --bert-dir (pinned local files)")
            out_path = run_audit_text_lengths(args)
            print(json.dumps({"status": "ok", "result": str(out_path)}))
            return 0

        if args.mode == "select-g2":
            out_path = run_select_g2(args)
            print(json.dumps({"status": "ok", "result": str(out_path)}))
            return 0

        if args.mode == "eta-g2":
            estimate, written = run_eta_g2(args)
            print(json.dumps({
                "status": "ok",
                "result": str(written) if written else None,
                "eta_seconds": estimate["eta_seconds"],
                "passes_seven_day_gate": estimate["passes_seven_day_gate"],
            }))
            return 0 if estimate["passes_seven_day_gate"] else 3

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
            # R2: the real two-card measurement cannot bypass the sync -- both
            # args are mandatory (checked BEFORE any asset/manifest load).
            if not (args.sync_dir and args.sync_group):
                raise RunnerError(
                    "throughput requires BOTH --sync-dir and --sync-group: the "
                    "two card processes must synchronize at every sync point; "
                    "an unsynchronized run is not valid two-card evidence"
                )
        started_epoch = time.time()
        setup: Optional[dict[str, Any]] = None
        try:
            setup = _gated_setup(args, g2=args.mode in ("throughput", "profile-consistency"))
            ledger: SharedBudgetLedger = setup["ledger"]
            invocation_id = setup["invocation_id"]
            model = setup["model"]
            store = setup["store"]
            texts = setup["texts"]
            selection = setup["selection"]
            repo_root = setup["repo_root"]
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
                    exit_code=0 if passed else 3, note="ok" if passed else "gate_failed",
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
                    exit_code=0 if report["passed"] else 3,
                    note="ok" if report["passed"] else "gate_failed",
                )
                print(json.dumps({"status": "ok" if report["passed"] else "gate_failed",
                                  "result": str(out_path)}))
                return 0 if report["passed"] else 3

            if args.mode == "profile-consistency":
                # G2 extended G1: the fixed 64-pair profile grid (8 token x 8
                # CAD quantiles) through the reference path and the ten fixed
                # batch/trim/cache variants -- 704 scored pairs, fixed
                # tolerances, raw valid-row logits persisted for Codex.
                pairs = [
                    (int(e["text_index"]), int(e["cad_index"]))
                    for e in selection["profile64"]
                ]
                planned = plan_profile_consistency_forwards(len(pairs))
                if planned > ledger.remaining_pairs():
                    raise RunnerError(
                        f"planned {planned} profile-consistency forwards exceed "
                        f"remaining budget {ledger.remaining_pairs()}"
                    )
                measurement = run_profile_consistency(
                    model, store, texts, pairs, device,
                    charge_fn=charge, out_dir=out_dir,
                )
                passed = measurement["passed"]
                out_path = _write_result(
                    setup, "profile-consistency", started_epoch,
                    {"profile64": selection["profile64"], **measurement},
                    schema_version=BENCH_RESULT_SCHEMA_VERSION_G2,
                    exit_code=0 if passed else 3, note="ok" if passed else "gate_failed",
                )
                print(json.dumps({"status": "ok" if passed else "gate_failed",
                                  "result": str(out_path)}))
                return 0 if passed else 3

            if args.mode == "throughput":
                # G2 production protocol: the committed G2 selection (real
                # adjacent-CAD batches, BERT-token strata), pre-built tier
                # plans the driver consumes verbatim, per-rep sync points
                # between the two cards, and a budget plan from the SAME
                # enumeration the driver executes. The barrier binding uses
                # the COMMON dual-shard plan hash (fixed shard order); each
                # card's local plan hash is announced and cross-checked.
                shard_id = BENCH_LEDGER_SHARD_BY_DEVICE[args.device]
                plan_binding = dual_shard_plan_binding(selection, tiers)
                plans = plan_binding["plans_by_shard"][shard_id]
                local_plan_sha = plan_binding["local_plan_sha256_by_shard"][shard_id]
                peer_local_plan_sha = plan_binding["local_plan_sha256_by_shard"][1 - shard_id]
                common_plan_sha = plan_binding["common_plan_sha256"]
                planned = plan_production_forwards(plans, args.warmup_passes, args.reps)
                if planned > ledger.remaining_pairs():
                    raise RunnerError(
                        f"planned {planned} throughput forwards exceed remaining budget "
                        f"{ledger.remaining_pairs()}"
                    )
                sync_dir = Path(args.sync_dir).resolve()
                ensure_outside_checkout(sync_dir, repo_root, "sync dir")
                sync_dir.mkdir(parents=True, exist_ok=True)
                sync = SyncBarrier(
                    sync_dir,
                    args.sync_group,
                    args.device,
                    {
                        "git_sha": setup["authorization"].approved_sha,
                        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
                        "pair_manifest_sha256": setup["fingerprints"].get(PAIR_MANIFEST_FILE),
                        "selection_sha256": sha256_file(repo_root / setup["selection_rel"]),
                        "common_plan_sha256": common_plan_sha,
                    },
                )
                model_forward_device = device
                token_lens = [int(v) for v in selection["text_token_lengths"]]

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

                def batch_facts_fn(batch: list[tuple[int, int]]) -> dict[str, Any]:
                    lens = [token_lens[t] for t, _ in batch]
                    return {
                        "max_cad_len": max(int(store.get(c)["n_real_tokens"]) for _, c in batch),
                        "text_tokens": lens,
                        "text_pad_width": max(lens),
                    }

                measurement = run_production_batch_tiers(
                    score_batch_fn,
                    plans,
                    warmup_passes=args.warmup_passes,
                    reps=args.reps,
                    device=args.device,
                    charge_fn=charge,
                    out_dir=out_dir,
                    sync=sync,
                    batch_facts_fn=batch_facts_fn,
                    local_plan_sha256=local_plan_sha,
                    expected_peer_local_plan_sha256=peer_local_plan_sha,
                )
                out_path = _write_result(
                    setup, "throughput", started_epoch,
                    {**measurement,
                     "batch_tiers": list(tiers),
                     "sync": {
                        "enabled": True,
                        "group": args.sync_group,
                        "timeout_s": sync.timeout_s,
                        "common_plan_sha256": common_plan_sha,
                        "local_plan_sha256": local_plan_sha,
                        "shard_id": shard_id,
                    }},
                    schema_version=BENCH_RESULT_SCHEMA_VERSION_G2,
                )
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
