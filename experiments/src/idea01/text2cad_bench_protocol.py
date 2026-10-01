"""Text2CAD G2 production-batch bench protocol (M1c G2 repair, 2026-10-01).

Task contract: experiments/plans/IDEA01_M1C_G2_PROTOCOL_REPAIR_2026-10-01.md
and the Codex G1 batch review
(experiments/reviews/IDEA01_M1C_G1_BATCH_CODEX_REVIEW_2026-10-01.md).

This module is pure CPU and performs NO model forward. It repairs the
diagnosed G2 protocol gaps:

  * REAL production batches: with the contract-fixed ``texts_per_chunk=1``,
    each text's CADs are scored in cad_index order, so a production batch is
    an ADJACENT cad_index slice. ``production_cad_windows`` enumerates the
    exact windows (full windows of the tier size plus one honest tail);
    nothing smaller than the tier is ever passed off as a full batch, and
    the tail is never padded with duplicate CADs.
  * EXACT shard weights: text shards are the very same contiguous ranges the
    production matrix scorer uses (``PairSpace.shard_text_range(2, k)``);
    per (tier, shard, text tercile, CAD layer) the selection records the
    exact batch count / pair count / batch size / batch max CAD length of the
    full 3,644 x 8,046 matrix implicitly (no 29,319,624-row table).
  * REAL text lengths: texts are stratified by audited BERT tokenizer
    lengths (special tokens included, no truncation), not raw char counts.
  * SLOWER-CARD ETA: ``estimate_sharded_eta`` charges every cell's exact
    production batch count at that cell's measured per-batch end-to-end
    cost (conservative: max over reps) on the card that owns the shard, adds
    the measured cold start per card, takes the MAX of the two cards'
    completion times (never a pooled ideal-2x rate), then applies extra
    overhead and the >=1.30 safety factor against the 604,800 s gate.
  * CONCURRENCY EVIDENCE: ``verify_rep_window_overlaps`` requires, for every
    (tier, cell, rep) used by the ETA, that the two cards' measured windows
    overlap for at least 80% of the shorter window; a global window is not
    accepted as concurrency evidence.

The legacy spot/ten_pair entries are reproduced byte-identically (same
n_chars/n_real_tokens quantile algorithm as ``generate_selection``) so the
old G1 gates keep their meaning; the NEW schema is NOT interchangeable with
the old one for G2 ETA purposes.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .cadmium_runner import RunnerError
from .text2cad_pair_index import PairSpace, sha256_text

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

G2_SELECTION_SCHEMA_VERSION = "text2cad-g2-selection-v1"
TOKEN_AUDIT_SCHEMA_VERSION = "text2cad-text-token-lengths-v1"

#: Production batch tiers (fixed by the contract; CLI never changes them).
G2_BATCH_TIERS = (1, 2, 4, 8, 16, 32)
G2_TEXT_TERCILES = 3
G2_CAD_DECILES = 10
G2_TEXTS_PER_CHUNK = 1
G2_N_SHARDS = 2

#: Extended G1 profile grid: 8 text-token-length quantiles x 8 CAD-length
#: quantiles; q=100 pins the REAL longest text and longest CAD.
PROFILE_QUANTILES = (0, 10, 25, 50, 75, 90, 99, 100)

#: Per non-empty cell the measurement samples the two endpoint texts
#: (token-length extremes within shard x tercile) and the two endpoint
#: windows (batch-max-CAD-length extremes within the layer).
G2_SAMPLE_TEXTS_PER_CELL = 2
G2_SAMPLE_WINDOWS_PER_CELL = 2

#: Legacy ten-pair table (kept identical to the old selection config).
_TEN_PAIR_QUANTILES = [
    (10, 10), (10, 50), (10, 90),
    (50, 10), (50, 50), (50, 90),
    (90, 10), (90, 50), (90, 90),
    (100, 90),
]

FULL_MATRIX_PAIRS = 29_319_624
SEVEN_DAY_SECONDS = 168 * 3600
ETA_SAFETY_FACTOR = 1.30
MIN_WINDOW_OVERLAP_FRACTION = 0.80
SYNC_TIMEOUT_SECONDS = 180.0

#: Result schema produced by the G2-repair forward modes; the legacy v1 ETA
#: refuses it and this ETA refuses v1 (the two are never interchangeable).
G2_RESULT_SCHEMA_VERSION = "text2cad-bench-result-v2"
SHARD_BY_DEVICE = {"npu:0": 0, "npu:1": 1}


# ---------------------------------------------------------------------------
# Small pure helpers (same semantics as the legacy selection algorithm)
# ---------------------------------------------------------------------------


def _rank_order(indices, key_of) -> list[int]:
    return sorted(indices, key=lambda i: (key_of(i), i))


def _pick_at_quantile(ordered: Sequence[int], quantile: float) -> int:
    if not 0 <= quantile <= 100:
        raise RunnerError(f"quantile must be in [0,100], got {quantile}")
    return ordered[round((len(ordered) - 1) * quantile / 100)]


def _stratum_of(rank: int, n: int, n_strata: int) -> int:
    return min(n_strata - 1, int(n_strata * rank / n))


# ---------------------------------------------------------------------------
# Production batch windows
# ---------------------------------------------------------------------------


def production_cad_windows(n_cads: int, batch_size: int) -> list[tuple[int, int]]:
    """Exact production CAD windows: adjacent slices, full tiers then one tail.

    With the contract-fixed ``texts_per_chunk=1`` every text's CADs are scored
    in cad_index order, so the production batches for a tier are exactly
    ``[(0,T), (T,2T), ...]`` plus a final tail window holding the remaining
    ``n_cads % T`` CADs (never dropped, never padded with duplicates).
    """
    if n_cads < 1 or batch_size < 1:
        raise RunnerError(f"production windows need positive dims, got {n_cads}/{batch_size}")
    windows = [
        (start, start + batch_size)
        for start in range(0, n_cads - batch_size + 1, batch_size)
    ]
    tail_start = (n_cads // batch_size) * batch_size
    if tail_start < n_cads:
        windows.append((tail_start, n_cads))
    return windows


def _validate_inputs(
    text_records: Sequence[dict[str, Any]],
    cad_length_records: Sequence[dict[str, Any]],
    text_token_lengths: dict[int, int],
) -> None:
    for i, rec in enumerate(text_records):
        if rec["text_index"] != i:
            raise RunnerError(f"text_records must be in text_index order, drift at {i}")
    for i, rec in enumerate(cad_length_records):
        if rec["cad_index"] != i or not isinstance(rec["n_real_tokens"], int) or rec["n_real_tokens"] < 1:
            raise RunnerError(f"cad_length_records drift at cad_index {i}")
    expected = set(range(len(text_records)))
    if set(text_token_lengths) != expected:
        raise RunnerError(
            f"text token length audit covers {len(text_token_lengths)} indices, "
            f"expected exactly {len(expected)}"
        )
    for index, n_tokens in text_token_lengths.items():
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1:
            raise RunnerError(
                f"text token length for text_index {index} must be a positive int, "
                f"got {n_tokens!r}"
            )


# ---------------------------------------------------------------------------
# G2 selection construction
# ---------------------------------------------------------------------------


def build_g2_selection(
    text_records: Sequence[dict[str, Any]],
    cad_length_records: Sequence[dict[str, Any]],
    text_token_lengths: dict[int, int],
    *,
    batch_tiers: Sequence[int] = G2_BATCH_TIERS,
    texts_per_chunk: int = G2_TEXTS_PER_CHUNK,
    n_shards: int = G2_N_SHARDS,
) -> dict[str, Any]:
    """Deterministic G2 production-batch selection (pure CPU, never reads qrels).

    Stratification: texts ranked by (n_bert_tokens, text_index) into terciles;
    production CAD windows (adjacent slices) ranked by (batch max real CAD
    length, window ordinal) into deciles, tail windows in their own layer.
    Text shards are exactly ``PairSpace.shard_text_range(2, k)`` -- the same
    boundary the production matrix scorer uses, so the ETA weights are the
    real per-card loads. ``spot``/``ten_pair`` are byte-identical to the
    legacy selection's algorithm; ``profile64`` is the new 8x8 token/CAD
    quantile grid for the extended G1 equivalence gate.
    """
    if sorted(set(int(t) for t in batch_tiers)) != sorted(G2_BATCH_TIERS):
        raise RunnerError(
            f"batch tiers must be exactly the production tiers {G2_BATCH_TIERS}, "
            f"got {sorted(batch_tiers)}"
        )
    if texts_per_chunk != G2_TEXTS_PER_CHUNK:
        raise RunnerError(
            f"the contract fixes texts_per_chunk={G2_TEXTS_PER_CHUNK}, got {texts_per_chunk}"
        )
    if n_shards != G2_N_SHARDS:
        raise RunnerError(f"the contract fixes n_shards={G2_N_SHARDS}, got {n_shards}")
    _validate_inputs(text_records, cad_length_records, text_token_lengths)

    n_texts = len(text_records)
    n_cads = len(cad_length_records)
    pair_space = PairSpace(n_texts=n_texts, n_cads=n_cads)
    n_chars = {rec["text_index"]: rec["n_chars"] for rec in text_records}
    n_real = {rec["cad_index"]: rec["n_real_tokens"] for rec in cad_length_records}
    n_tokens = {int(k): int(v) for k, v in text_token_lengths.items()}

    # -- legacy spot / ten_pair (identical algorithm: n_chars / n_real ranks)
    char_rank_order = _rank_order(range(n_texts), n_chars.get)
    cad_rank_order = _rank_order(range(n_cads), n_real.get)

    def legacy_entry(text_index: int, cad_index: int) -> dict[str, int]:
        return {
            "pair_index": pair_space.pair_index_of(text_index, cad_index),
            "text_index": text_index,
            "cad_index": cad_index,
            "n_chars": n_chars[text_index],
            "n_real_tokens": n_real[cad_index],
        }

    spot = legacy_entry(_pick_at_quantile(char_rank_order, 50), _pick_at_quantile(cad_rank_order, 50))
    ten_pair = [
        legacy_entry(_pick_at_quantile(char_rank_order, tq), _pick_at_quantile(cad_rank_order, cq))
        for tq, cq in _TEN_PAIR_QUANTILES
    ]

    # -- token-length text terciles
    token_rank_order = _rank_order(range(n_texts), n_tokens.get)
    text_tier_of: dict[int, int] = {}
    for rank, text_index in enumerate(token_rank_order):
        text_tier_of[text_index] = _stratum_of(rank, n_texts, G2_TEXT_TERCILES)

    # -- profile64: token-length quantiles x CAD-length quantiles
    profile64: list[dict[str, int]] = []
    for tq in PROFILE_QUANTILES:
        for cq in PROFILE_QUANTILES:
            text_index = _pick_at_quantile(token_rank_order, tq)
            cad_index = _pick_at_quantile(cad_rank_order, cq)
            profile64.append(
                {
                    "pair_index": pair_space.pair_index_of(text_index, cad_index),
                    "text_index": text_index,
                    "cad_index": cad_index,
                    "text_quantile": tq,
                    "cad_quantile": cq,
                    "n_bert_tokens": n_tokens[text_index],
                    "n_real_tokens": n_real[cad_index],
                }
            )

    # -- text shards (identical boundary rule to the production matrix scorer)
    text_shards = []
    shard_texts: list[list[int]] = []
    for shard_id in range(n_shards):
        t0, t1 = pair_space.shard_text_range(n_shards, shard_id)
        text_shards.append({"shard_id": shard_id, "text_range": [t0, t1]})
        shard_texts.append(list(range(t0, t1)))

    # -- per-tier production windows, layers, cells
    tiers: dict[str, Any] = {}
    per_tier_counts: dict[str, Any] = {}
    sample_shortfalls: list[str] = []
    for tier in G2_BATCH_TIERS:
        windows = production_cad_windows(n_cads, tier)
        n_full = n_cads // tier
        tail_size = n_cads % tier
        full_windows = windows[:n_full]
        tail_window = windows[n_full:]  # [] or [(start, n_cads)]
        window_max = [
            max(n_real[c] for c in range(start, end)) for start, end in windows
        ]
        # full windows ranked by (batch max real length, ordinal) into deciles
        full_order = sorted(range(n_full), key=lambda w: (window_max[w], w))
        layer_of_full: dict[int, int] = {}
        for rank, w in enumerate(full_order):
            layer_of_full[w] = _stratum_of(rank, n_full, G2_CAD_DECILES)
        layers: list[list[int]] = [[] for _ in range(G2_CAD_DECILES)]
        for w, layer in layer_of_full.items():
            layers[layer].append(w)

        full_layers = []
        for d in range(G2_CAD_DECILES):
            ws = layers[d]
            full_layers.append(
                {
                    "layer": f"d{d}",
                    "n_windows": len(ws),
                    "batch_size": tier,
                    "batch_max_len_min": min((window_max[w] for w in ws), default=None),
                    "batch_max_len_max": max((window_max[w] for w in ws), default=None),
                }
            )
        tail_summary = None
        if tail_window:
            tail_summary = {
                "layer": "tail",
                "n_windows": 1,
                "batch_size": tail_size,
                "batch_max_len_min": window_max[n_full],
                "batch_max_len_max": window_max[n_full],
            }

        def cell_for(shard_id: int, text_tier: int, layer_name: str,
                     layer_windows: list[int], batch_size: int) -> Optional[tuple[str, dict[str, Any]]]:
            texts_in = [t for t in shard_texts[shard_id] if text_tier_of[t] == text_tier]
            if not texts_in or not layer_windows:
                return None
            cell_key = f"shard{shard_id}_tt{text_tier}_{layer_name}"
            # endpoint texts: token-length extremes within this shard x tier
            text_extremes = sorted(
                {min(texts_in, key=lambda t: (n_tokens[t], t)),
                 max(texts_in, key=lambda t: (n_tokens[t], t))}
            )
            # endpoint windows: batch-max-length extremes within this layer
            win_extremes = sorted(
                {min(layer_windows, key=lambda w: (window_max[w], w)),
                 max(layer_windows, key=lambda w: (window_max[w], w))}
            )
            shortfall: list[str] = []
            if len(text_extremes) < G2_SAMPLE_TEXTS_PER_CELL:
                shortfall.append(f"texts: {len(text_extremes)} < {G2_SAMPLE_TEXTS_PER_CELL}")
            if len(win_extremes) < G2_SAMPLE_WINDOWS_PER_CELL:
                shortfall.append(f"windows: {len(win_extremes)} < {G2_SAMPLE_WINDOWS_PER_CELL}")
            for note in shortfall:
                sample_shortfalls.append(f"{cell_key} (tier {tier}): {note}")
            n_batches = len(texts_in) * len(layer_windows)
            return cell_key, {
                "shard_id": shard_id,
                "text_tier": text_tier,
                "layer": layer_name,
                "n_texts": len(texts_in),
                "n_windows": len(layer_windows),
                "batch_size": batch_size,
                "n_batches": n_batches,
                "n_pairs": n_batches * batch_size,
                "batch_max_len_min": min(window_max[w] for w in layer_windows),
                "batch_max_len_max": max(window_max[w] for w in layer_windows),
                "samples": {
                    "texts": text_extremes,
                    "windows": [
                        {
                            "cad_start": windows[w][0],
                            "cad_end": windows[w][1],
                            "batch_max_len": window_max[w],
                        }
                        for w in win_extremes
                    ],
                    "shortfall": shortfall,
                },
            }

        cells: dict[str, Any] = {}
        for shard_id in range(n_shards):
            for text_tier in range(G2_TEXT_TERCILES):
                for d in range(G2_CAD_DECILES):
                    built = cell_for(shard_id, text_tier, f"d{d}", layers[d], tier)
                    if built is not None:
                        cells[built[0]] = built[1]
                if tail_window:
                    built = cell_for(shard_id, text_tier, "tail", [n_full], tail_size)
                    if built is not None:
                        cells[built[0]] = built[1]

        tier_key = str(tier)
        tiers[tier_key] = {
            "batch_size": tier,
            "windows": {
                "n_full": n_full,
                "tail_size": tail_size,
                "full_layers": full_layers,
                "tail": tail_summary,
            },
            "cells": dict(sorted(cells.items())),
        }
        per_tier_counts[tier_key] = {
            "n_pairs": sum(c["n_pairs"] for c in cells.values()),
            "n_batches": sum(c["n_batches"] for c in cells.values()),
            "n_sample_batches": sum(
                len(c["samples"]["texts"]) * len(c["samples"]["windows"])
                for c in cells.values()
            ),
        }

    return {
        "schema_version": G2_SELECTION_SCHEMA_VERSION,
        "rules": {
            "text_strata": (
                f"rank by (n_bert_tokens, text_index) into {G2_TEXT_TERCILES} terciles"
            ),
            "text_shards": "PairSpace.shard_text_range(2, k) -- identical to the production matrix scorer",
            "cad_batches": (
                "adjacent cad_index slices (texts_per_chunk=1): full windows of the "
                "tier size ranked by (batch max n_real_tokens, ordinal) into "
                f"{G2_CAD_DECILES} deciles; the tail window is its own layer"
            ),
            "cell_samples": (
                f"{G2_SAMPLE_TEXTS_PER_CELL} endpoint texts x "
                f"{G2_SAMPLE_WINDOWS_PER_CELL} endpoint windows per non-empty cell; "
                "deduped with recorded shortfall"
            ),
            "spot": "median-rank text x median-rank CAD (legacy n_chars/n_real ranks)",
            "ten_pair_quantiles": [list(q) for q in _TEN_PAIR_QUANTILES],
            "profile64_quantiles": list(PROFILE_QUANTILES),
            "batch_tiers": list(G2_BATCH_TIERS),
            "texts_per_chunk": G2_TEXTS_PER_CHUNK,
            "n_shards": G2_N_SHARDS,
        },
        "counts": {
            "n_texts": n_texts,
            "n_cads": n_cads,
            "n_pairs": pair_space.n_pairs,
            "per_tier": per_tier_counts,
            "sample_shortfalls": sample_shortfalls,
        },
        "text_token_lengths": [n_tokens[i] for i in range(n_texts)],
        "text_shards": text_shards,
        "spot": spot,
        "ten_pair": ten_pair,
        "profile64": profile64,
        "tiers": tiers,
    }


def sample_batches_for_cell(cell: dict[str, Any]) -> list[list[tuple[int, int]]]:
    """Materialize a cell's measured sample batches as explicit pair lists.

    Each returned batch is the cross of one sample text with one sample
    window: a REAL adjacent cad_index slice of exactly the cell's batch size
    (the tail cell's smaller honest size). The throughput driver consumes
    these batches verbatim -- it never re-flattens them into another
    grouping.
    """
    batches: list[list[tuple[int, int]]] = []
    for text_index in cell["samples"]["texts"]:
        for window in cell["samples"]["windows"]:
            batches.append(
                [(text_index, cad_index) for cad_index in range(window["cad_start"], window["cad_end"])]
            )
    return batches


def _diff_path(expected: Any, actual: Any, path: str) -> Optional[str]:
    """First structural difference between two JSON-like trees (path or None).

    STRICT key sets (R2 fix 5): a key present in ``actual`` but not in
    ``expected`` is drift exactly like a missing or changed one -- an injected
    ``qrels``/``evaluation_map``/caption field can never ride into the worker
    inside a selection file.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return path or "<root>"
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in expected:
                return sub  # injected extra key
            if key not in actual:
                return sub
            found = _diff_path(expected[key], actual[key], sub)
            if found is not None:
                return found
        return None
    if isinstance(expected, (list, tuple)):
        # JSON round-trips turn tuples into lists; both spellings are equal here.
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return path
        for i, (e, a) in enumerate(zip(expected, actual)):
            found = _diff_path(e, a, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    if expected != actual:
        return path
    return None


def verify_g2_selection(
    selection: dict[str, Any],
    text_records: Sequence[dict[str, Any]],
    cad_length_records: Sequence[dict[str, Any]],
    text_token_lengths: dict[int, int],
) -> None:
    """Re-derive every selection fact from live inputs; fail closed on drift.

    The whole derived content (weights, samples, profile grid, token lengths,
    spot/ten_pair) is recomputed from the LIVE records and compared exactly;
    provenance/extra top-level keys are ignored here (the CLI verifies those
    against the reviewed artifacts separately).
    """
    if selection.get("schema_version") != G2_SELECTION_SCHEMA_VERSION:
        raise RunnerError(
            f"g2 selection schema {selection.get('schema_version')!r} != "
            f"{G2_SELECTION_SCHEMA_VERSION!r}"
        )
    expected = build_g2_selection(
        text_records, cad_length_records, text_token_lengths
    )
    # strict top-level allowlist: exactly the derived sections plus the
    # CLI-attached provenance; any injected input (qrels, evaluation_map,
    # captions, ...) refuses the whole file
    extra_top = sorted(set(selection) - set(expected) - {"provenance"})
    if extra_top:
        raise RunnerError(
            f"g2 selection carries non-allowlisted top-level keys {extra_top}"
        )
    for key in expected:
        if key not in selection:
            raise RunnerError(f"g2 selection is missing section {key!r}")
        drift = _diff_path(expected[key], selection[key], key)
        if drift is not None:
            raise RunnerError(f"g2 selection drift at {drift}")


# ---------------------------------------------------------------------------
# Text token length audit (pure CPU; tokenizer ONLY -- never the BERT model)
# ---------------------------------------------------------------------------


def build_text_token_audit(
    text_records: Sequence[dict[str, Any]],
    token_count_of: Callable[[str], int],
    *,
    texts_jsonl_sha256: str,
    bert_file_sha256: dict[str, str],
    max_seq_len: int = 512,
) -> dict[str, Any]:
    """Audit every unique text's real BERT token length (special tokens included).

    ``token_count_of(text)`` must return the tokenizer's token count for one
    text WITHOUT truncation (the CLI wires the pinned local BertTokenizer's
    ``attention_mask`` sum -- the exact same measurement the dry-run audit
    uses). Any text over ``max_seq_len`` is a hard failure: the protocol
    never truncates silently.
    """
    lengths: list[dict[str, int]] = []
    over: list[tuple[int, int]] = []
    for rec in text_records:
        n_tokens = int(token_count_of(rec["text"]))
        if n_tokens < 1:
            raise RunnerError(f"tokenizer returned {n_tokens} tokens for text_index {rec['text_index']}")
        if n_tokens > max_seq_len:
            over.append((rec["text_index"], n_tokens))
        lengths.append({"text_index": rec["text_index"], "n_bert_tokens": n_tokens})
    if over:
        raise RunnerError(
            f"{len(over)} text(s) exceed {max_seq_len} BERT tokens, e.g. {over[:3]}; "
            "refusing (no silent truncation)"
        )
    return {
        "schema_version": TOKEN_AUDIT_SCHEMA_VERSION,
        "provenance": {
            "texts_jsonl_sha256": texts_jsonl_sha256,
            "bert_file_sha256": dict(sorted(bert_file_sha256.items())),
            "method": "pinned local BertTokenizer, special tokens included, no truncation",
            "model_loaded": False,
        },
        "max_seq_len": max_seq_len,
        "lengths": lengths,
    }


def token_lengths_from_audit(
    audit: dict[str, Any],
    *,
    expected_texts_sha256: str,
    expected_bert_file_sha256: dict[str, str],
    n_texts: int,
) -> dict[int, int]:
    """Validate a committed token audit and return {text_index: n_bert_tokens}.

    Fail closed on: schema drift, input-text SHA drift (the audit must have
    been produced from exactly these texts), BERT asset drift, or incomplete /
    duplicated text coverage.
    """
    if audit.get("schema_version") != TOKEN_AUDIT_SCHEMA_VERSION:
        raise RunnerError(
            f"token audit schema {audit.get('schema_version')!r} != {TOKEN_AUDIT_SCHEMA_VERSION!r}"
        )
    provenance = audit.get("provenance") or {}
    if provenance.get("texts_jsonl_sha256") != expected_texts_sha256:
        raise RunnerError("token audit was not produced from these texts.jsonl bytes")
    if provenance.get("bert_file_sha256") != dict(sorted(expected_bert_file_sha256.items())):
        raise RunnerError("token audit BERT asset fingerprints differ from the pinned files")
    lengths = audit.get("lengths")
    if not isinstance(lengths, list) or len(lengths) != n_texts:
        raise RunnerError(
            f"token audit covers {len(lengths) if isinstance(lengths, list) else 0} texts, "
            f"expected {n_texts}"
        )
    out: dict[int, int] = {}
    for entry in lengths:
        index = entry["text_index"]
        n_tokens = entry["n_bert_tokens"]
        if index in out or not isinstance(index, int) or not 0 <= index < n_texts:
            raise RunnerError(f"token audit has a bad/duplicate text_index {index!r}")
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1:
            raise RunnerError(f"token audit has a bad length for text_index {index}")
        out[index] = n_tokens
    if len(out) != n_texts:
        raise RunnerError("token audit text coverage is not complete")
    return out


# ---------------------------------------------------------------------------
# Tier plans: pre-built real batches the throughput driver consumes verbatim
# ---------------------------------------------------------------------------

_LAYER_ORDER = {f"d{d}": d for d in range(G2_CAD_DECILES)}
_LAYER_ORDER["tail"] = G2_CAD_DECILES


def build_tier_plans(
    selection: dict[str, Any], tiers: Sequence[int], shard_id: int
) -> list[dict[str, Any]]:
    """Per requested tier: the card's cell plans with PRE-BUILT real batches.

    Cells are iterated in the canonical (text_tier, layer) order so the two
    cards hit the same sync points (``tt{t}_{layer}``) at the same time; the
    shard id is part of the cell key but never of the sync point. The driver
    consumes these batches verbatim -- it never re-flattens them into another
    grouping, so the budget plan and the executed forwards share one
    enumeration.
    """
    if selection.get("schema_version") != G2_SELECTION_SCHEMA_VERSION:
        raise RunnerError(f"tier plans require a {G2_SELECTION_SCHEMA_VERSION} selection")
    if shard_id not in (0, 1):
        raise RunnerError(f"shard id must be 0 or 1, got {shard_id}")
    plans: list[dict[str, Any]] = []
    for tier in tiers:
        tier_key = str(int(tier))
        tier_entry = (selection.get("tiers") or {}).get(tier_key)
        if tier_entry is None:
            raise RunnerError(f"selection has no tier {tier_key}")
        cells = tier_entry["cells"]
        cell_plans: list[dict[str, Any]] = []
        ordered = sorted(
            (c for c in cells.values() if c["shard_id"] == shard_id),
            key=lambda c: (c["text_tier"], _LAYER_ORDER[c["layer"]]),
        )
        for cell in ordered:
            batches = sample_batches_for_cell(cell)
            cell_plans.append(
                {
                    "cell": f"shard{shard_id}_tt{cell['text_tier']}_{cell['layer']}",
                    "point": f"tt{cell['text_tier']}_{cell['layer']}",
                    "shard_id": shard_id,
                    "text_tier": cell["text_tier"],
                    "layer": cell["layer"],
                    "batch_size": cell["batch_size"],
                    "is_tail": cell["layer"] == "tail",
                    "batches": batches,
                }
            )
        plans.append({"batch_size": int(tier), "cells": cell_plans})
    return plans


def plan_production_forwards(
    tier_plans: Sequence[dict[str, Any]], warmup_passes: int, reps: int
) -> int:
    """Exact forwards (warmups + timed reps) over the SAME enumeration the
    driver executes -- the budget check and the charges cannot diverge.

    R2 fix 7: warmup is per CELL (``warmup_passes`` passes over the cell's
    first two sample batches), not just over the first cell of the tier.
    """
    total = 0
    for plan in tier_plans:
        warmup_pairs = sum(
            len(b) for cell in plan["cells"] for b in cell["batches"][:2]
        )
        rep_pairs = sum(len(b) for cell in plan["cells"] for b in cell["batches"])
        total += warmup_passes * warmup_pairs + reps * rep_pairs
    return total


def tier_plans_sha256(tier_plans: Sequence[dict[str, Any]]) -> str:
    """Binding hash of the ACTUAL measurement profile (cells + batch sizes)."""
    return sha256_text(json.dumps(_plan_summary(tier_plans), sort_keys=True))


def combined_tier_plans_sha256(
    plans_shard0: Sequence[dict[str, Any]], plans_shard1: Sequence[dict[str, Any]]
) -> str:
    """COMMON two-card binding over BOTH shards' plans, in FIXED shard order.

    The two cards measure different texts/cells, so their local plan hashes
    differ by construction; the barrier binding that must be identical on
    both sides is this combined hash over ``[shard0 plans, shard1 plans]``.
    """
    summary = [
        {"shard_id": 0, "plan": json.loads(json.dumps(_plan_summary(plans_shard0)))},
        {"shard_id": 1, "plan": json.loads(json.dumps(_plan_summary(plans_shard1)))},
    ]
    return sha256_text(json.dumps(summary, sort_keys=True))


def _plan_summary(tier_plans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "batch_size": plan["batch_size"],
            "cells": [
                {
                    "cell": cell["cell"],
                    "point": cell["point"],
                    "batch_sizes": [len(b) for b in cell["batches"]],
                    "pairs": [[list(p) for p in b] for b in cell["batches"]],
                }
                for cell in plan["cells"]
            ],
        }
        for plan in tier_plans
    ]


def dual_shard_plan_binding(
    selection: dict[str, Any], tiers: Sequence[int]
) -> dict[str, Any]:
    """Build both shards' tier plans and the common/local binding hashes.

    This is the SINGLE construction used by the throughput CLI (and mirrored
    by tests): the common hash binds the barrier, each card's local hash is
    recorded in its result and cross-checked against the peer's announced
    value at every sync point, and the ETA rebuilds both from the committed
    selection to verify the measurement profile.
    """
    plans0 = build_tier_plans(selection, tiers, 0)
    plans1 = build_tier_plans(selection, tiers, 1)
    return {
        "plans_by_shard": {0: plans0, 1: plans1},
        "common_plan_sha256": combined_tier_plans_sha256(plans0, plans1),
        "local_plan_sha256_by_shard": {
            0: tier_plans_sha256(plans0),
            1: tier_plans_sha256(plans1),
        },
    }


# ---------------------------------------------------------------------------
# Two-process file sync barrier (per tier/cell/rep; fail-closed; evidence kept)
# ---------------------------------------------------------------------------

SYNC_MARKER_SCHEMA = "text2cad-g2-sync-v1"
_POINT_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _device_token(device: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", device)


class SyncBarrier:
    """File-based barrier synchronizing two same-host card processes.

    Each process arrives at every (tier, cell, rep) sync point with a marker
    file carrying the run binding (code/model/manifest/selection SHAs and the
    actual plan hash); the arrival blocks until the peer's marker for the
    same point appears, and the peer's binding must equal ours -- an old file,
    a foreign schema, a different binding, or a marker from the SAME device
    all refuse. Waiting is bounded by ``timeout_s`` (contract: 180 s) and is
    charged to card time by the caller, never mixed into the pure measured
    window. A peer error/OOM failure marker stops the other side at its next
    arrival; markers are evidence and are never deleted or overwritten (no
    automatic reruns).
    """

    def __init__(
        self,
        sync_dir,
        group: str,
        device: str,
        binding: dict[str, Any],
        *,
        timeout_s: float = SYNC_TIMEOUT_SECONDS,
        poll_interval_s: float = 0.2,
        now_fn=time.time,
        sleep_fn=time.sleep,
    ) -> None:
        if not group or not _POINT_RE.match(group):
            raise RunnerError(f"bad sync group name {group!r}")
        self.sync_dir = Path(sync_dir)
        if not self.sync_dir.is_dir():
            raise RunnerError(f"sync dir {self.sync_dir} does not exist")
        self.group = group
        self.device = device
        self.device_token = _device_token(device)
        self.binding = dict(binding)
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn

    def _marker_path(self, point: str, device_token: str, failed: bool = False) -> Path:
        suffix = ".FAILED.json" if failed else ".json"
        return self.sync_dir / f"{self.group}__{point}__{device_token}{suffix}"

    def _write_marker(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RunnerError(f"sync marker {path} exists; refusing a rerun at the same point")
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(payload, indent=1, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise RunnerError(
                    f"sync marker {path} appeared concurrently (same device claiming "
                    "the point twice?); refusing"
                ) from exc
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def signal_failure(self, point: str, kind: str, detail: str) -> None:
        """Leave a peer-visible failure marker (evidence; never auto-cleared)."""
        if not _POINT_RE.match(point):
            point = "run"
        payload = {
            "schema_version": SYNC_MARKER_SCHEMA,
            "group": self.group,
            "point": point,
            "device": self.device,
            "binding": self.binding,
            "failure": {"kind": kind, "detail": detail[:500], "epoch": self.now_fn()},
        }
        path = self._marker_path(point, self.device_token, failed=True)
        if not path.exists():
            self._write_marker(path, payload)

    def _peer_failure(self) -> Optional[dict[str, Any]]:
        for path in sorted(self.sync_dir.glob(f"{self.group}__*__*.FAILED.json")):
            if path.name.endswith(f"__{self.device_token}.FAILED.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if data.get("schema_version") != SYNC_MARKER_SCHEMA or data.get("group") != self.group:
                continue
            return {"path": path.name, "failure": data.get("failure")}
        return None

    def arrive(self, point: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not _POINT_RE.match(point):
            raise RunnerError(f"bad sync point name {point!r}")
        own = self._marker_path(point, self.device_token)
        self._write_marker(
            own,
            {
                "schema_version": SYNC_MARKER_SCHEMA,
                "group": self.group,
                "point": point,
                "device": self.device,
                "binding": self.binding,
                "payload": payload,
                "epoch": self.now_fn(),
            },
        )
        deadline = self.now_fn() + self.timeout_s
        prefix = f"{self.group}__{point}__"
        while True:
            peer_failure = self._peer_failure()
            if peer_failure is not None:
                raise RunnerError(
                    f"sync peer failed ({peer_failure['path']}: "
                    f"{peer_failure['failure']}); stopping this card too, "
                    "evidence preserved"
                )
            candidates = [
                p for p in self.sync_dir.glob(f"{prefix}*.json")
                if not p.name.endswith(".FAILED.json") and p != own
            ]
            if candidates:
                peer_path = candidates[0]
                try:
                    peer = json.loads(peer_path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    raise RunnerError(f"unreadable sync marker {peer_path}: {exc}") from exc
                if peer.get("schema_version") != SYNC_MARKER_SCHEMA:
                    raise RunnerError(
                        f"sync marker {peer_path.name} has schema "
                        f"{peer.get('schema_version')!r}; refusing a stale/foreign file"
                    )
                if peer.get("group") != self.group:
                    raise RunnerError(f"sync marker {peer_path.name} belongs to another group")
                if peer.get("device") == self.device:
                    raise RunnerError(
                        f"sync peer marker claims the same device {self.device}; "
                        "refusing a single card posing as two"
                    )
                if peer.get("binding") != self.binding:
                    raise RunnerError(
                        "sync peer binding differs (code/model/manifest/selection/plan); "
                        "refusing to synchronize mismatched runs"
                    )
                return dict(peer.get("payload") or {})
            if self.now_fn() > deadline:
                raise RunnerError(
                    f"sync barrier at {point} timeout after {self.timeout_s}s waiting "
                    "for the peer card; evidence preserved, no automatic rerun"
                )
            self.sleep_fn(self.poll_interval_s)


# ---------------------------------------------------------------------------
# Per-rep concurrency evidence (windows; the global window is not accepted)
# ---------------------------------------------------------------------------


def _window_ok(window: Any) -> bool:
    if not isinstance(window, dict):
        return False
    try:
        start = float(window["start_epoch"])
        end = float(window["end_epoch"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(start) and math.isfinite(end) and start <= end


def window_overlap_fraction(window_a: dict[str, float], window_b: dict[str, float]) -> float:
    """Overlap of two windows as a fraction of the SHORTER window (0..1)."""
    if not (_window_ok(window_a) and _window_ok(window_b)):
        raise RunnerError(f"invalid measurement window(s): {window_a} / {window_b}")
    a0, a1 = float(window_a["start_epoch"]), float(window_a["end_epoch"])
    b0, b1 = float(window_b["start_epoch"]), float(window_b["end_epoch"])
    shorter = min(a1 - a0, b1 - b0)
    if shorter <= 0:
        raise RunnerError("zero-length measurement window cannot evidence concurrency")
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap / shorter


def verify_concurrent_rep_windows(
    reps_a: Sequence[dict[str, Any]],
    reps_b: Sequence[dict[str, Any]],
    *,
    min_fraction: float = MIN_WINDOW_OVERLAP_FRACTION,
) -> None:
    """Per-(point, rep) concurrency evidence between the two cards.

    Every sync point measured on one card must appear on the other with a
    window that is finite, not reversed, and overlapping for at least
    ``min_fraction`` of the SHORTER window. Missing points, staggered or
    disjoint windows all refuse -- a global overlapping window is NOT
    accepted as concurrency evidence (the diagnosed G2 gap).
    """
    if not reps_a or not reps_b:
        raise RunnerError("missing rep windows on one card; refusing the combination")
    by_point_b = {r["point"]: r for r in reps_b}
    for rep_a in reps_a:
        point = rep_a["point"]
        rep_b = by_point_b.get(point)
        if rep_b is None:
            raise RunnerError(f"sync point {point} missing on the peer card")
        fraction = window_overlap_fraction(rep_a["window"], rep_b["window"])
        if fraction < min_fraction:
            raise RunnerError(
                f"sync point {point}: window overlap {fraction:.3f} < {min_fraction}; "
                "the two cards did not actually measure concurrently"
            )


# ---------------------------------------------------------------------------
# Sharded ETA: fixed production shards, the SLOWER card is the wall clock
# ---------------------------------------------------------------------------


def estimate_sharded_eta(
    selection: dict[str, Any],
    results_by_device: dict[str, dict[str, Any]],
    tier: int,
    *,
    overhead_seconds: Optional[float] = None,
    safety_factor: float = ETA_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Full-matrix ETA from the two cards' sharded measurements (fail-closed).

    Per card: ``sum over its shard's cells (exact production batch count x
    the cell's conservative per-batch E2E -- the MAX over the >=3 reps, each
    rep priced as full-rep E2E / real batch count) + that card's measured
    cold start``. The wall clock is the MAX of the two cards' completion
    times -- never a pooled ideal-2x rate. ``overhead_seconds=None`` keeps
    the measured cold starts as the only overhead; an explicit value must be
    finite and >= the sum of both cold starts (the extra is additional
    wall-clock overhead AFTER both cards; 0 cannot bypass cold start). The
    total is multiplied by ``safety_factor`` (>= 1.30) and compared against
    the 604,800 s seven-day gate. Every (cell, rep) window pair must prove
    >=80% overlap of the shorter window. The cell weights must sum to the
    exact 29,319,624-pair matrix -- subsets and rerank are refused outright.
    """
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(safety_factor)
        or safety_factor < ETA_SAFETY_FACTOR
    ):
        raise RunnerError(
            f"safety_factor must be finite and >= {ETA_SAFETY_FACTOR}, got {safety_factor!r}"
        )
    if selection.get("schema_version") != G2_SELECTION_SCHEMA_VERSION:
        raise RunnerError(
            f"estimate_sharded_eta requires a {G2_SELECTION_SCHEMA_VERSION} selection"
        )
    if set(results_by_device) != set(SHARD_BY_DEVICE):
        raise RunnerError(
            "exactly one throughput result per device (npu:0 and npu:1) is required"
        )
    tier_key = str(int(tier))
    tier_entry = (selection.get("tiers") or {}).get(tier_key)
    if tier_entry is None:
        raise RunnerError(f"selection has no tier {tier_key}")
    n_pairs_total = int(selection["counts"]["n_pairs"])
    if n_pairs_total != FULL_MATRIX_PAIRS:
        raise RunnerError(
            f"selection covers {n_pairs_total} pairs, not the exact full matrix "
            f"{FULL_MATRIX_PAIRS:,}"
        )
    if sum(c["n_pairs"] for c in tier_entry["cells"].values()) != FULL_MATRIX_PAIRS:
        raise RunnerError(
            f"tier {tier_key} cell weights do not sum to the exact full matrix "
            f"{FULL_MATRIX_PAIRS:,}"
        )

    invocations: set[str] = set()
    envs: dict[str, dict[str, Any]] = {}
    sync_groups: set[str] = set()
    common_plan_shas: set[str] = set()
    local_plan_shas: dict[str, Any] = {}
    batch_tiers_by_card: dict[str, list[int]] = {}
    for device, result in results_by_device.items():
        if result.get("schema_version") != G2_RESULT_SCHEMA_VERSION:
            raise RunnerError(
                f"{device}: schema {result.get('schema_version')!r} != "
                f"{G2_RESULT_SCHEMA_VERSION!r} (legacy results are never accepted "
                "by the G2 ETA)"
            )
        if result.get("mode") != "throughput":
            raise RunnerError(f"{device}: not a throughput result")
        invocation = result.get("invocation_id")
        if not isinstance(invocation, str) or not invocation:
            raise RunnerError(f"{device}: missing invocation_id")
        invocations.add(invocation)
        env = result.get("environment") or {}
        envs[device] = env
        # card identity: the result filed under a device must claim that device
        if env.get("device") != device:
            raise RunnerError(
                f"{device}: environment claims device {env.get('device')!r}; card "
                "identity mismatch"
            )
        # per-card reference configuration (BOTH cards are checked, always)
        if (
            env.get("torch_dtype") != "float32"
            or env.get("trim_mode") != "none"
            or env.get("text_cache") is not False
        ):
            raise RunnerError(
                f"{device}: not the frozen reference configuration (requires "
                "float32/full-length/uncached)"
            )
        # mandatory two-card sync evidence (R2): real concurrent measurement
        sync = result.get("sync") or {}
        if sync.get("enabled") is not True or not sync.get("group"):
            raise RunnerError(
                f"{device}: result lacks mandatory sync evidence (sync.enabled "
                "must be true with a shared group)"
            )
        sync_groups.add(str(sync["group"]))
        common_plan_shas.add(sync.get("common_plan_sha256"))
        local_plan_shas[device] = sync.get("local_plan_sha256")
        # requested tier set: explicit, duplicate-free, containing the priced tier
        batch_tiers = result.get("batch_tiers")
        if (
            not isinstance(batch_tiers, list)
            or not batch_tiers
            or len(set(batch_tiers)) != len(batch_tiers)
            or not all(isinstance(t, int) and not isinstance(t, bool) for t in batch_tiers)
        ):
            raise RunnerError(f"{device}: bad batch_tiers record {batch_tiers!r}")
        if int(tier) not in batch_tiers:
            raise RunnerError(f"{device}: priced tier {tier_key} was not measured")
        tier_entries = result.get("tiers") or []
        tier_sizes = [t.get("batch_size") for t in tier_entries]
        if len(set(tier_sizes)) != len(tier_sizes) or set(tier_sizes) != set(batch_tiers):
            raise RunnerError(
                f"{device}: duplicate/missing/extra tier records vs batch_tiers"
            )
        batch_tiers_by_card[device] = sorted(int(t) for t in batch_tiers)
    if len(invocations) != 2:
        raise RunnerError("the two card results must come from distinct invocations")
    if len(sync_groups) != 1:
        raise RunnerError(
            "the two card results must come from the SAME sync group "
            f"(got {sorted(sync_groups)})"
        )
    if batch_tiers_by_card["npu:0"] != batch_tiers_by_card["npu:1"]:
        raise RunnerError("the two cards did not measure the same tier set")
    env0, env1 = envs["npu:0"], envs["npu:1"]
    for key in ("checkpoint_sha256", "pair_manifest_sha256", "score_name", "spec_version"):
        if not env0.get(key) or env0.get(key) != env1.get(key):
            raise RunnerError(f"card results disagree on {key}")
    git0, git1 = env0.get("git_sha"), env1.get("git_sha")
    if (
        not isinstance(git0, str)
        or git0 != git1
        or len(git0) != 40
        or not set(git0) <= set("0123456789abcdef")
    ):
        raise RunnerError("card results must carry the same complete reviewed git_sha")
    selection_shas = {r.get("selection_sha256") for r in results_by_device.values()}
    if len(selection_shas) != 1 or not isinstance(next(iter(selection_shas)), str) or len(next(iter(selection_shas))) != 64:
        raise RunnerError("card results disagree on / lack the selection SHA binding")

    # Rebuild the reviewed measurement profile from the committed selection
    # and verify BOTH cards' plan bindings against it (R2 fix 3/4).
    rebuilt = dual_shard_plan_binding(selection, batch_tiers_by_card["npu:0"])
    if common_plan_shas != {rebuilt["common_plan_sha256"]}:
        raise RunnerError(
            "the two results' common plan binding does not match the profile "
            "rebuilt from the committed selection (or they disagree)"
        )
    for device, shard in SHARD_BY_DEVICE.items():
        if local_plan_shas[device] != rebuilt["local_plan_sha256_by_shard"][shard]:
            raise RunnerError(
                f"{device}: local plan SHA does not match the rebuilt {shard} shard "
                "profile"
            )

    per_card: dict[str, Any] = {}
    rep_windows_by_card: dict[str, list[dict[str, Any]]] = {}
    for device, shard in SHARD_BY_DEVICE.items():
        result = results_by_device[device]
        tier_record = next(
            (t for t in result.get("tiers", []) if t.get("batch_size") == int(tier)), None
        )
        if tier_record is None or tier_record.get("status") != "ok":
            raise RunnerError(f"{device}: tier {tier_key} did not complete successfully")
        cold = result.get("cold_start_s")
        if isinstance(cold, bool) or not isinstance(cold, (int, float)) or not math.isfinite(cold) or cold < 0:
            raise RunnerError(f"{device}: bad cold_start_s {cold!r}")
        # peak HBM must be a real reading taken on THIS card
        peak = tier_record.get("peak_device_memory_bytes")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
            raise RunnerError(
                f"{device}: tier {tier_key} has no valid peak HBM reading "
                f"({peak!r}); None/zero is not valid evidence"
            )
        if tier_record.get("hbm_device") != device:
            raise RunnerError(
                f"{device}: HBM reading was not taken on this card "
                f"({tier_record.get('hbm_device')!r})"
            )
        expected_cells = {
            key: cell
            for key, cell in tier_entry["cells"].items()
            if cell["shard_id"] == shard
        }
        measured_cells = tier_record.get("cells") or {}
        if set(measured_cells) != set(expected_cells):
            raise RunnerError(
                f"{device}: measured cell set differs from the selection "
                f"(missing={sorted(set(expected_cells) - set(measured_cells))[:3]}, "
                f"extra={sorted(set(measured_cells) - set(expected_cells))[:3]})"
            )
        compute_seconds = 0.0
        cell_details: list[dict[str, Any]] = []
        rep_windows: list[dict[str, Any]] = []
        for cell_key in sorted(expected_cells):
            cell = expected_cells[cell_key]
            measured = measured_cells[cell_key]
            expected_point = f"tt{cell['text_tier']}_{cell['layer']}"
            if measured.get("point") != expected_point:
                raise RunnerError(
                    f"{device}: cell {cell_key} was measured under a different profile"
                )
            expected_batches = sample_batches_for_cell(cell)
            n_sample = len(expected_batches)
            reps = measured.get("reps") or []
            rep_indices = [r.get("rep") for r in reps]
            if (
                len(rep_indices) < 3
                or sorted(rep_indices) != list(range(len(rep_indices)))
            ):
                raise RunnerError(
                    f"{device}: cell {cell_key} reps must number at least 3 unique "
                    f"contiguous indices from 0, got {sorted(rep_indices)}"
                )
            per_batch_costs: list[float] = []
            for rep in reps:
                if (
                    rep.get("n_batches") != n_sample
                    or rep.get("n_pairs") != n_sample * cell["batch_size"]
                ):
                    raise RunnerError(
                        f"{device}: cell {cell_key} rep profile does not match the "
                        "selection's actual batches"
                    )
                # every batch of every rep carries the REAL pairs in order
                recorded = rep.get("batches")
                if not isinstance(recorded, list) or len(recorded) != n_sample:
                    raise RunnerError(
                        f"{device}: cell {cell_key} rep {rep.get('rep')} lacks "
                        "per-batch records"
                    )
                for k, (rec_batch, exp_batch) in enumerate(zip(recorded, expected_batches)):
                    expected_pairs = [[int(t), int(c)] for t, c in exp_batch]
                    if (
                        rec_batch.get("pairs") != expected_pairs
                        or rec_batch.get("size") != len(exp_batch)
                        or rec_batch.get("batch_ordinal") != k
                    ):
                        raise RunnerError(
                            f"{device}: cell {cell_key} rep {rep.get('rep')} batch {k} "
                            "pair order/size differs from the reviewed plan"
                        )
                e2e = rep.get("e2e_s")
                if isinstance(e2e, bool) or not isinstance(e2e, (int, float)) or not math.isfinite(e2e) or e2e <= 0:
                    raise RunnerError(
                        f"{device}: cell {cell_key} rep e2e is not finite/positive: {e2e!r}"
                    )
                per_batch_costs.append(float(e2e) / n_sample)
                rep_windows.append(
                    {"point": f"{expected_point}_r{rep.get('rep')}", "window": rep.get("window")}
                )
            # conservative: the SLOWEST full rep prices this cell
            per_batch_max = max(per_batch_costs)
            seconds = cell["n_batches"] * per_batch_max
            compute_seconds += seconds
            cell_details.append(
                {
                    "cell": cell_key,
                    "n_batches": cell["n_batches"],
                    "batch_size": cell["batch_size"],
                    "per_batch_e2e_s_max": per_batch_max,
                    "seconds": seconds,
                    "n_pairs": cell["n_pairs"],
                }
            )
        per_card[device] = {
            "shard_id": shard,
            "n_cells": len(cell_details),
            "n_batches": sum(c["n_batches"] for c in cell_details),
            "n_pairs": sum(c["n_pairs"] for c in cell_details),
            "compute_seconds": compute_seconds,
            "cold_start_s": float(cold),
            "completion_s": float(cold) + compute_seconds,
            "cells": cell_details,
        }
        rep_windows_by_card[device] = rep_windows

    # Concurrency evidence: the two cards must present the IDENTICAL
    # (cell, rep) point set -- duplicates inside one card, missing points or
    # extra points on either side all refuse before any window math.
    keys0 = [r["point"] for r in rep_windows_by_card["npu:0"]]
    keys1 = [r["point"] for r in rep_windows_by_card["npu:1"]]
    if len(set(keys0)) != len(keys0) or len(set(keys1)) != len(keys1):
        raise RunnerError("duplicate sync points inside a card's measurement")
    if set(keys0) != set(keys1):
        raise RunnerError(
            "the two cards' sync point sets differ (missing or extra cell/rep)"
        )
    verify_concurrent_rep_windows(
        rep_windows_by_card["npu:0"], rep_windows_by_card["npu:1"]
    )

    cold_sum = per_card["npu:0"]["cold_start_s"] + per_card["npu:1"]["cold_start_s"]
    if overhead_seconds is None:
        extra_overhead = 0.0
    else:
        if (
            isinstance(overhead_seconds, bool)
            or not isinstance(overhead_seconds, (int, float))
            or not math.isfinite(overhead_seconds)
            or overhead_seconds < cold_sum
        ):
            raise RunnerError(
                f"overhead_seconds must be finite and >= the two cards' cold starts "
                f"({cold_sum:.3f} s); got {overhead_seconds!r} -- measured cold start "
                "cannot be bypassed"
            )
        extra_overhead = float(overhead_seconds) - cold_sum
    wall = max(card["completion_s"] for card in per_card.values()) + extra_overhead
    eta_seconds = wall * float(safety_factor)
    slower_card = max(per_card, key=lambda d: per_card[d]["completion_s"])
    return {
        "tier": int(tier),
        "full_matrix_pairs": FULL_MATRIX_PAIRS,
        "per_card": per_card,
        "slower_card": slower_card,
        "cold_start_seconds_sum": cold_sum,
        "extra_overhead_seconds": extra_overhead,
        "overhead_seconds_input": overhead_seconds,
        "wall_seconds_presafety": wall,
        "safety_factor": float(safety_factor),
        "eta_seconds": eta_seconds,
        "eta_hours": eta_seconds / 3600.0,
        "effective_two_card_pairs_per_s": FULL_MATRIX_PAIRS / eta_seconds,
        "required_two_card_pairs_per_s": FULL_MATRIX_PAIRS * ETA_SAFETY_FACTOR / SEVEN_DAY_SECONDS,
        "seven_day_seconds": SEVEN_DAY_SECONDS,
        "passes_seven_day_gate": eta_seconds <= SEVEN_DAY_SECONDS,
        "notes": (
            "per-card compute = sum over the card's shard cells (exact production "
            "batch count x slowest-rep per-batch E2E); write/ledger time is inside "
            "the measured E2E and is NOT added again; the wall clock is the slower "
            "card's completion (never a pooled ideal-2x); extra I/O or failure "
            "margin must enter via overhead_seconds and is reported verbatim"
        ),
    }
