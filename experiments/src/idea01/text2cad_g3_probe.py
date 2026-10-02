"""Text2CAD G3 full-gallery probe: fixed blind 100-text selection + CPU ETA (2026-10-02).

Task contract: experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md
and the Codex G2 batch16 results review
(experiments/reviews/IDEA01_M1C_G2_B16_CODEX_RESULTS_REVIEW_2026-10-02.md).

This module is pure CPU and performs NO model forward anywhere -- it is
torch-free at import (the pinned asset SHA constants are duplicated from the
audited loader with a unit-test cross-check, the same pattern as
``text2cad_pair_index.CAD_SEQ_ZIP_SHA256``). It implements:

  * the pre-registered BLIND probe selection: inside each production text
    shard (``PairSpace.shard_text_range(2, k)`` = [0,1822) and [1822,3644))
    the texts are ranked by (audited BERT token length, text_index) and the
    50 evenly spaced ranks ``round((1822-1)*j/49)``, j=0..49 are taken --
    endpoints included, so each shard's shortest AND longest text is always
    probed. The selection reads ONLY the four frozen worker artifacts, the
    audited token-length JSON and the frozen G2 config; it never loads a
    tokenizer, never reads any relevance/UID linkage or any prior model
    score. ``select-g3`` refuses to overwrite its output.
  * strict verification: schema, exact provenance key set, live worker /
    token-audit / G2-config fingerprints, official dimensions (3,644 texts x
    8,046 gallery -- a different gallery size refuses), and a full
    re-derivation of every probe/fact from the live records (strict
    structural diff; injected keys anywhere refuse).
  * the production-runner scope helpers: per-shard probe text sets, the
    per-query timing sidecar (appended line-by-line AFTER the chunk's atomic
    publish completes, so a timing line always implies the chunk exists) and
    its strict reader (a torn final line fails closed).
  * ``eta-g3``: the two-card full-matrix ETA from the two REAL probe runs --
    per card per global text layer (``min(2, 3*rank//3644)`` over the global
    token-length rank, identical to G2) the SLOWEST complete-query E2E
    (score + serialization + atomic publish) times that layer's FULL query
    count in the card's shard, plus the card's measured cold start; the wall
    clock is the MAX of the two cards (never a pooled ideal-2x), times the
    >=1.30 safety factor, strictly compared against the 604,800 s seven-day
    gate. Both probe summaries must prove: 50 queries x 8,046 pairs each, no
    missing layer, same host, the device<->shard correspondence, a finalized
    successful shared-ledger invocation, identical code/model/score/probe
    fingerprints, and an >=80% overlap of the two cards' actual scoring
    windows (first query start to last query publish end).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .cadmium_runner import RunnerError
from .text2cad_bench_protocol import (
    ETA_SAFETY_FACTOR,
    MIN_WINDOW_OVERLAP_FRACTION,
    SEVEN_DAY_SECONDS,
    SHARD_BY_DEVICE,
    _diff_path,
    token_lengths_from_audit,
    window_overlap_fraction,
)
from .text2cad_pair_index import (
    CAD_LENGTH_FILE,
    CAD_STORE_FILE,
    EXPECTED_N_GALLERY,
    EXPECTED_N_UNIQUE_TEXT,
    PAIR_MANIFEST_FILE,
    TEXTS_FILE,
    PairSpace,
    chunk_file_name,
    load_cad_store,
    load_pair_manifest,
    read_jsonl_strict,
    sha256_file,
)

#: Score identity duplicated from text2cad_vector_scorer so this module stays
#: torch-free at import (a unit test cross-checks both values).
SCORE_NAME = "two_head_factorized_logprob_sum"
SCORE_SPEC_VERSION = "text2cad-vector-score-v1"

#: Run-summary schema produced by the frozen runner (duplicated so this
#: module stays torch-free; a unit test cross-checks the value).
RUN_SUMMARY_SCHEMA_VERSION = "text2cad-run-summary-v1"

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

G3_SELECTION_SCHEMA_VERSION = "text2cad-g3-probe-selection-v1"
G3_TIMING_SCHEMA_VERSION = "text2cad-g3-timing-v1"

#: Committed (reviewed) files pinned by path; the production runner accepts
#: ONLY the in-repo probe selection at this exact path.
G3_PROBE_SELECTION_REL = "experiments/configs/text2cad_g3_probe_selection_20261002.json"
#: Frozen G2 config whose SHA the G3 provenance binds (duplicated constant --
#: a unit test cross-checks it against text2cad_speed_bench.G2_SELECTION_CONFIG_REL).
G2_SELECTION_CONFIG_REL = "experiments/configs/text2cad_g2_pair_selection_20261001.json"
#: Reviewed pair-manifest fingerprints (same committed file the runner pins).
EXPECTED_PAIR_HASHES_REL = "experiments/configs/text2cad_pair_manifest_expected_20260930.json"

#: This round is fixed to the reviewed batch16 / tail-14 production shape.
G3_BATCH_SIZE = 16
G3_TEXTS_PER_CHUNK = 1
G3_N_SHARDS = 2
G3_PICKS_PER_SHARD = 50
G3_TEXT_LAYERS = 3

#: Chunk-header scope field binding outputs/resume to the probe selection.
PROBE_SCOPE_FIELD = "probe_selection_sha256"

#: Exact provenance key set of a committed G3 probe selection (an injected
#: key anywhere in the file refuses it).
G3_PROVENANCE_KEYS = frozenset({
    "texts_jsonl_sha256",
    "cad_store_jsonl_sha256",
    "cad_length_index_sha256",
    "pair_manifest_sha256",
    "token_audit_sha256",
    "g2_selection_sha256",
    "generated_by",
})

#: Pinned asset fingerprints duplicated from text2cad_port.loader so this
#: module stays torch-free at import; unit tests cross-check both values.
PINNED_CHECKPOINT_SHA256 = "235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598"
PINNED_BERT_FILE_SHA256 = {
    "config.json": "f9a38c9ffa369f288c3206d0bf352dfed11d7bdee93aa5b2de6d7fdc7bff7871",
    "model.safetensors": "28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c4",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


# ---------------------------------------------------------------------------
# Selection construction (deterministic; blind to relevance; pure CPU)
# ---------------------------------------------------------------------------


def text_layer_of(rank: int, n_texts: int) -> int:
    """Global token-length-rank layer, identical to the G2 tercile rule."""
    if not 0 <= rank < n_texts:
        raise RunnerError(f"layer rank {rank} outside [0, {n_texts})")
    return min(G3_TEXT_LAYERS - 1, G3_TEXT_LAYERS * rank // n_texts)


def global_text_layer_map(text_token_lengths: dict[int, int]) -> dict[int, int]:
    """text_index -> layer under the GLOBAL (n_bert_tokens, text_index) rank."""
    n_texts = len(text_token_lengths)
    rank_order = sorted(text_token_lengths, key=lambda i: (text_token_lengths[i], i))
    return {text_index: text_layer_of(rank, n_texts) for rank, text_index in enumerate(rank_order)}


def shard_pick_positions(shard_size: int, picks: int) -> list[int]:
    """The pre-registered evenly spaced rank picks, endpoints included.

    Position j is ``round((shard_size-1)*j/(picks-1))``; the sorted dedup must
    keep exactly ``picks`` positions or the contract cannot be satisfied.
    """
    if shard_size < 1 or picks < 2 or picks > shard_size:
        raise RunnerError(
            f"need 1 <= picks < shard_size with picks >= 2, got picks={picks} "
            f"shard_size={shard_size}"
        )
    positions = sorted({round(j * (shard_size - 1) / (picks - 1)) for j in range(picks)})
    if len(positions) != picks:
        raise RunnerError(
            f"deduped pick positions shrank to {len(positions)} < {picks}; the "
            "contract requires exactly the requested count per shard"
        )
    if positions[0] != 0 or positions[-1] != shard_size - 1:
        raise RunnerError("pick positions must include both shard endpoints")
    return positions


def _validate_token_lengths(text_token_lengths: dict[int, int], n_texts: int) -> None:
    if set(text_token_lengths) != set(range(n_texts)):
        raise RunnerError(
            f"token lengths cover {len(text_token_lengths)} indices, expected exactly {n_texts}"
        )
    for index, n_tokens in text_token_lengths.items():
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1:
            raise RunnerError(f"bad token length for text_index {index}: {n_tokens!r}")


def build_g3_probe_selection(
    text_records: Sequence[dict[str, Any]],
    n_cads: int,
    text_token_lengths: dict[int, int],
    *,
    n_shards: int = G3_N_SHARDS,
    picks_per_shard: int = G3_PICKS_PER_SHARD,
) -> dict[str, Any]:
    """Deterministic blind probe selection (pure CPU; no relevance input).

    ``text_records`` are the worker texts.jsonl rows in text_index order (only
    ``text_index`` is consulted -- never the text content); ``n_cads`` is the
    live gallery size; ``text_token_lengths`` is the audited per-text BERT
    token count. Returns the derived selection sections (no provenance -- the
    CLI attaches it). Every fact is recomputed by the verifier.
    """
    if n_shards != G3_N_SHARDS:
        raise RunnerError(f"the contract fixes n_shards={G3_N_SHARDS}, got {n_shards}")
    if picks_per_shard != G3_PICKS_PER_SHARD:
        raise RunnerError(
            f"the contract fixes {G3_PICKS_PER_SHARD} picks per shard, got {picks_per_shard}"
        )
    for i, rec in enumerate(text_records):
        if rec["text_index"] != i:
            raise RunnerError(f"text_records must be in text_index order, drift at {i}")
    if n_cads < 1:
        raise RunnerError(f"gallery size must be positive, got {n_cads}")
    n_texts = len(text_records)
    _validate_token_lengths(text_token_lengths, n_texts)
    lengths = {int(k): int(v) for k, v in text_token_lengths.items()}
    pair_space = PairSpace(n_texts=n_texts, n_cads=n_cads)
    layer_of = global_text_layer_map(lengths)

    probes: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for shard_id in range(n_shards):
        t0, t1 = pair_space.shard_text_range(n_shards, shard_id)
        ordered = sorted(range(t0, t1), key=lambda i: (lengths[i], i))
        positions = shard_pick_positions(t1 - t0, picks_per_shard)
        picked = [ordered[p] for p in positions]
        # endpoints of the ordered shard == shortest and longest texts
        if lengths[picked[0]] != lengths[ordered[0]] or lengths[picked[-1]] != lengths[ordered[-1]]:
            raise RunnerError(f"shard {shard_id}: picks lost the shard length endpoints")
        layer_text_counts = {layer: 0 for layer in range(G3_TEXT_LAYERS)}
        for text_index in range(t0, t1):
            layer_text_counts[layer_of[text_index]] += 1
        layer_probe_counts = {layer: 0 for layer in range(G3_TEXT_LAYERS)}
        for j, text_index in enumerate(picked):
            layer = layer_of[text_index]
            layer_probe_counts[layer] += 1
            probes.append(
                {
                    "text_index": text_index,
                    "shard_id": shard_id,
                    "shard_pick": j,
                    "n_bert_tokens": lengths[text_index],
                    "text_layer": layer,
                }
            )
        missing = [layer for layer, count in layer_probe_counts.items() if count == 0]
        if missing:
            raise RunnerError(
                f"shard {shard_id}: global text layer(s) {missing} have no probe; "
                "the contract requires at least one probe per shard per layer"
            )
        shards.append(
            {
                "shard_id": shard_id,
                "text_range": [t0, t1],
                "n_texts": t1 - t0,
                "n_probe": len(picked),
                "layer_text_counts": {str(k): v for k, v in sorted(layer_text_counts.items())},
                "layer_probe_counts": {str(k): v for k, v in sorted(layer_probe_counts.items())},
            }
        )

    return {
        "schema_version": G3_SELECTION_SCHEMA_VERSION,
        "rules": {
            "text_shards": "PairSpace.shard_text_range(2, k) -- identical to the production matrix scorer",
            "pick": (
                f"inside each shard rank texts by (n_bert_tokens, text_index); take "
                f"positions round((shard_size-1)*j/{picks_per_shard - 1}), "
                f"j=0..{picks_per_shard - 1} (endpoints included; sorted dedup must "
                f"keep exactly {picks_per_shard})"
            ),
            "text_layers": (
                f"global (n_bert_tokens, text_index) rank into {G3_TEXT_LAYERS} "
                "layers: layer = min(2, 3*rank//n_texts) -- identical to G2"
            ),
            "gallery": (
                "each selected query still scores the COMPLETE official gallery; "
                "the worker gallery and global text/pair indices are never "
                "reordered, filtered or rebuilt"
            ),
            "production_shape": (
                f"batch {G3_BATCH_SIZE} plus the honest tail, FP32, full-length, "
                "no trim, no text cache, texts_per_chunk=1"
            ),
            "blindness": (
                "inputs are the four frozen worker artifacts, the audited token "
                "lengths and the frozen G2 config only; no tokenizer, no "
                "relevance/UID linkage, no prior model scores"
            ),
        },
        "counts": {
            "n_texts": n_texts,
            "n_cads": n_cads,
            "n_pairs": pair_space.n_pairs,
            "n_shards": n_shards,
            "picks_per_shard": picks_per_shard,
            "n_probe_texts": len(probes),
            "gallery_per_query": n_cads,
            "probe_pairs_per_shard": picks_per_shard * n_cads,
            "probe_pairs_total": len(probes) * n_cads,
        },
        "text_token_lengths": [lengths[i] for i in range(n_texts)],
        "text_shards": [
            {"shard_id": shard["shard_id"], "text_range": shard["text_range"]}
            for shard in shards
        ],
        "shards": shards,
        "probes": probes,
    }


def verify_official_dimensions(n_texts: int, n_cads: int) -> None:
    """The probe is defined ONLY for the official frozen pair space."""
    if n_texts != EXPECTED_N_UNIQUE_TEXT or n_cads != EXPECTED_N_GALLERY:
        raise RunnerError(
            f"probe requires the official {EXPECTED_N_UNIQUE_TEXT} texts x "
            f"{EXPECTED_N_GALLERY} gallery, got {n_texts} x {n_cads}"
        )


def verify_g3_probe_selection(
    selection: dict[str, Any],
    text_records: Sequence[dict[str, Any]],
    n_cads: int,
    text_token_lengths: dict[int, int],
) -> None:
    """Re-derive every selection fact from live inputs; fail closed on drift."""
    if selection.get("schema_version") != G3_SELECTION_SCHEMA_VERSION:
        raise RunnerError(
            f"probe selection schema {selection.get('schema_version')!r} != "
            f"{G3_SELECTION_SCHEMA_VERSION!r}"
        )
    verify_official_dimensions(len(text_records), n_cads)
    expected = build_g3_probe_selection(text_records, n_cads, text_token_lengths)
    extra_top = sorted(set(selection) - set(expected) - {"provenance"})
    if extra_top:
        raise RunnerError(
            f"probe selection carries non-allowlisted top-level keys {extra_top}"
        )
    for key in expected:
        if key not in selection:
            raise RunnerError(f"probe selection is missing section {key!r}")
        drift = _diff_path(expected[key], selection[key], key)
        if drift is not None:
            raise RunnerError(f"probe selection drift at {drift}")


# ---------------------------------------------------------------------------
# Manifest / asset binding (live fingerprints; strict token-audit binding)
# ---------------------------------------------------------------------------


def load_g3_manifest_records(manifest_dir) -> tuple[list[dict[str, Any]], int]:
    """Load the worker artifacts the probe needs; returns (text_records, n_cads).

    The pair manifest's schema and internal artifact SHAs are verified; the
    CAD store is schema-checked (the gallery identity is bound by SHA in the
    provenance even though the selection only needs its size).
    """
    manifest_dir = Path(manifest_dir)
    load_pair_manifest(manifest_dir)
    text_records = read_jsonl_strict(
        manifest_dir / TEXTS_FILE,
        frozenset({"text_index", "text", "text_sha256", "n_chars"}),
        TEXTS_FILE,
    )
    members = load_cad_store(manifest_dir)
    return text_records, len(members)


def verify_g3_probe_selection_against_manifest(
    selection: dict[str, Any],
    manifest_dir,
) -> None:
    """Full binding: schema, exact provenance, live fingerprints, re-derivation."""
    if selection.get("schema_version") != G3_SELECTION_SCHEMA_VERSION:
        raise RunnerError(
            f"probe selection schema {selection.get('schema_version')!r} != "
            f"{G3_SELECTION_SCHEMA_VERSION!r}"
        )
    manifest_dir = Path(manifest_dir)
    provenance = selection.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(G3_PROVENANCE_KEYS):
        raise RunnerError(
            f"probe selection provenance keys must be exactly {sorted(G3_PROVENANCE_KEYS)}, "
            f"got {sorted(provenance) if isinstance(provenance, dict) else provenance!r}"
        )
    live = {}
    for key, name in (
        ("texts_jsonl_sha256", TEXTS_FILE),
        ("cad_store_jsonl_sha256", CAD_STORE_FILE),
        ("cad_length_index_sha256", CAD_LENGTH_FILE),
        ("pair_manifest_sha256", PAIR_MANIFEST_FILE),
    ):
        path = manifest_dir / name
        if not path.is_file():
            raise RunnerError(f"worker artifact {path} is missing")
        live[key] = sha256_file(path)
    for key, value in live.items():
        if provenance.get(key) != value:
            raise RunnerError(
                f"probe selection provenance {key} {provenance.get(key)!r} != live {value!r}; "
                "the selection was not generated from these artifacts"
            )
    for key in ("token_audit_sha256", "g2_selection_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= set("0123456789abcdef"):
            raise RunnerError(f"probe selection has no valid {key} binding")
    text_records, n_cads = load_g3_manifest_records(manifest_dir)
    embedded = selection.get("text_token_lengths")
    if not isinstance(embedded, list) or len(embedded) != len(text_records):
        raise RunnerError("probe selection token-length table does not cover the texts")
    verify_g3_probe_selection(
        selection, text_records, n_cads, {i: int(v) for i, v in enumerate(embedded)}
    )


def verify_g3_token_audit(selection: dict[str, Any], token_audit_path, manifest_dir) -> None:
    """Bind the ACTUAL token-audit file to the selection at runtime.

    The committed selection pins the audit by SHA; this requires the real
    file to exist, hash to the pinned value, carry the audited schema and the
    texts/BERT provenance, and equal the selection's embedded table.
    """
    if not token_audit_path:
        raise RunnerError(
            "G3 entries require --token-audit: the reviewed derived token-length "
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
        raise RunnerError("token audit contents differ from the selection's embedded table")


def verify_g3_g2_config_binding(selection: dict[str, Any], repo_root) -> str:
    """The frozen G2 config must hash to the provenance binding; returns its SHA."""
    path = Path(repo_root) / G2_SELECTION_CONFIG_REL
    if not path.is_file():
        raise RunnerError(f"frozen G2 config {path} does not exist")
    actual = sha256_file(path)
    expected = (selection.get("provenance") or {}).get("g2_selection_sha256")
    if actual != expected:
        raise RunnerError(
            f"frozen G2 config SHA {actual} != the selection binding {expected!r}"
        )
    return actual


# ---------------------------------------------------------------------------
# Runner-facing probe helpers (scope, timing sidecar)
# ---------------------------------------------------------------------------


def probe_texts_for_shard(selection: dict[str, Any], shard_id: int) -> list[int]:
    """Sorted global text indices the shard must probe (from the verified selection)."""
    return sorted(
        int(e["text_index"]) for e in selection["probes"] if int(e["shard_id"]) == shard_id
    )


def probe_facts_for_shard(selection: dict[str, Any], shard_id: int) -> dict[int, dict[str, int]]:
    """text_index -> {text_layer, n_bert_tokens, shard_pick} for the shard's probes."""
    return {
        int(e["text_index"]): {
            "text_layer": int(e["text_layer"]),
            "n_bert_tokens": int(e["n_bert_tokens"]),
            "shard_pick": int(e["shard_pick"]),
        }
        for e in selection["probes"]
        if int(e["shard_id"]) == shard_id
    }


def timing_sidecar_name(shard_id: int) -> str:
    return f"timing_s{shard_id:02d}.jsonl"


def append_timing_record(path, record: dict[str, Any]) -> None:
    """Append ONE complete timing line and fsync (per-query sidecar).

    Appended AFTER the query's chunk was atomically published, so a complete
    line always implies the chunk file exists. A crash can only tear the LAST
    line; the strict reader refuses the file then (fail closed).
    """
    path = Path(path)
    if path.exists() and not path.is_file():
        raise RunnerError(f"timing sidecar {path} is not a regular file")
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_timing_records(path) -> list[dict[str, Any]]:
    """Strict sidecar reader: every line must be a complete JSON object."""
    path = Path(path)
    if not path.is_file():
        raise RunnerError(f"timing sidecar {path} does not exist")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(
                    f"timing sidecar {path} line {lineno} is not complete JSON "
                    f"(torn write?): {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise RunnerError(f"timing sidecar {path} line {lineno} is not an object")
            records.append(record)
    return records


def verified_sidecar_text_set(
    records: Sequence[dict[str, Any]],
    *,
    shard_id: int,
    scope: dict[str, Any],
    path_label: str,
) -> set[int]:
    """Minimal runner-side sidecar check for resume: scope + shard + uniqueness.

    Returns the set of timed query text indices. Every record must carry the
    current scope (probe SHA, code/asset/score bindings, device) and this
    shard; a foreign, drifted or duplicated line refuses the resume. The ETA
    applies the STRICT full verification on top of this.
    """
    texts: set[int] = set()
    for i, record in enumerate(records):
        where = f"{path_label} record {i}"
        if record.get("record_type") != "query_timing" or record.get("schema_version") != G3_TIMING_SCHEMA_VERSION:
            raise RunnerError(f"{where}: not a {G3_TIMING_SCHEMA_VERSION} query_timing record")
        for field, expected in scope.items():
            if record.get(field) != expected:
                raise RunnerError(
                    f"{where}: scope {field} {record.get(field)!r} != {expected!r}; "
                    "the sidecar belongs to a different run scope"
                )
        if record.get("shard_id") != shard_id:
            raise RunnerError(f"{where}: shard_id {record.get('shard_id')!r} != {shard_id}")
        text_index = record.get("text_index")
        if isinstance(text_index, bool) or not isinstance(text_index, int):
            raise RunnerError(f"{where}: bad text_index {text_index!r}")
        if text_index in texts:
            raise RunnerError(f"{where}: text_index {text_index} timed twice")
        texts.add(text_index)
    return texts


def build_query_timing_record(
    *,
    invocation_id: str,
    shard_id: int,
    device: str,
    text_index: int,
    probe_facts: dict[str, int],
    start_epoch: float,
    end_epoch: float,
    n_pairs: int,
    n_batches: int,
    n_scored: int,
    n_rejected: int,
    n_valid_total: int,
    chunk_file: str,
    scope: dict[str, Any],
) -> dict[str, Any]:
    """One per-query timing record (E2E = score + serialize + atomic publish)."""
    return {
        "record_type": "query_timing",
        "schema_version": G3_TIMING_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "shard_id": int(shard_id),
        "device": device,
        "text_index": int(text_index),
        "text_range": [int(text_index), int(text_index) + G3_TEXTS_PER_CHUNK],
        "text_layer": int(probe_facts["text_layer"]),
        "n_bert_tokens": int(probe_facts["n_bert_tokens"]),
        "shard_pick": int(probe_facts["shard_pick"]),
        "window": {"start_epoch": float(start_epoch), "end_epoch": float(end_epoch)},
        "e2e_s": float(end_epoch) - float(start_epoch),
        "n_pairs": int(n_pairs),
        "n_batches": int(n_batches),
        "n_scored": int(n_scored),
        "n_rejected": int(n_rejected),
        "n_valid_total": int(n_valid_total),
        "chunk_file": chunk_file,
        PROBE_SCOPE_FIELD: scope[PROBE_SCOPE_FIELD],
        "git_sha": scope["git_sha"],
        "pair_manifest_sha256": scope["pair_manifest_sha256"],
        "checkpoint_sha256": scope["checkpoint_sha256"],
        "score_name": scope["score_name"],
        "spec_version": scope["spec_version"],
    }


# ---------------------------------------------------------------------------
# ETA input verification (summaries + timing sidecars of the two real runs)
# ---------------------------------------------------------------------------

#: Environment fields the two cards must agree on (bound into every summary).
_ETA_ENV_SHARED_FIELDS = (
    "git_sha",
    "pair_manifest_sha256",
    "checkpoint_sha256",
    "score_name",
    "spec_version",
    PROBE_SCOPE_FIELD,
)


def _require_finite_positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RunnerError(f"{label} is not a finite number: {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        raise RunnerError(f"{label} must be {'>= 0' if allow_zero else '> 0'}, got {value!r}")
    return float(value)


def verify_probe_summary(summary: dict[str, Any], *, device: str, path_label: str) -> dict[str, Any]:
    """One card's run summary must prove a successful, correctly scoped probe."""
    shard = SHARD_BY_DEVICE[device]
    if summary.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION:
        raise RunnerError(f"{path_label}: not a {RUN_SUMMARY_SCHEMA_VERSION} run summary")
    if summary.get("status") != "ok" or summary.get("failure") is not None:
        raise RunnerError(
            f"{path_label}: status {summary.get('status')!r} / failure "
            f"{summary.get('failure')!r}; the ETA prices only successful runs"
        )
    if summary.get("device") != device:
        raise RunnerError(
            f"{path_label}: summary device {summary.get('device')!r} != {device!r}"
        )
    if summary.get("shard_id") != shard or summary.get("n_shards") != G3_N_SHARDS:
        raise RunnerError(
            f"{path_label}: shard {summary.get('shard_id')!r}/{summary.get('n_shards')!r} "
            f"!= the {device}<->shard{shard} correspondence"
        )
    if summary.get("batch_size") != G3_BATCH_SIZE:
        raise RunnerError(f"{path_label}: batch_size {summary.get('batch_size')!r} != {G3_BATCH_SIZE}")
    if summary.get("texts_per_chunk") != G3_TEXTS_PER_CHUNK:
        raise RunnerError(f"{path_label}: texts_per_chunk must be {G3_TEXTS_PER_CHUNK}")
    if (
        summary.get("torch_dtype") != "float32"
        or summary.get("trim_mode") != "none"
        or summary.get("text_cache") is not False
    ):
        raise RunnerError(f"{path_label}: not the frozen reference configuration")
    if summary.get("score_name") != SCORE_NAME or summary.get("spec_version") != SCORE_SPEC_VERSION:
        raise RunnerError(f"{path_label}: unreviewed score definition")
    if summary.get("checkpoint_sha256") != PINNED_CHECKPOINT_SHA256:
        raise RunnerError(f"{path_label}: not the pinned checkpoint")
    probe_sha = summary.get(PROBE_SCOPE_FIELD)
    if not isinstance(probe_sha, str) or len(probe_sha) != 64:
        raise RunnerError(f"{path_label}: summary carries no probe selection scope")
    git_sha = summary.get("git_sha")
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or not set(git_sha) <= set("0123456789abcdef")
    ):
        raise RunnerError(f"{path_label}: summary must carry a complete reviewed git_sha")
    if (summary.get("authorization") or {}).get("approved_sha") != git_sha:
        raise RunnerError(f"{path_label}: authorization SHA differs from the code SHA")
    cold = summary.get("cold_start_s")
    _require_finite_positive(cold, f"{path_label} cold_start_s", allow_zero=True)
    peak = summary.get("peak_device_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise RunnerError(
            f"{path_label}: no valid peak HBM reading ({peak!r}); None/zero is not evidence"
        )
    if summary.get("hbm_device") != device:
        raise RunnerError(
            f"{path_label}: HBM reading was not taken on {device} ({summary.get('hbm_device')!r})"
        )
    timing_file = summary.get("timing_file")
    if not isinstance(timing_file, str) or Path(timing_file).name != timing_file:
        raise RunnerError(f"{path_label}: bad timing_file reference {timing_file!r}")
    if timing_file != timing_sidecar_name(shard):
        raise RunnerError(
            f"{path_label}: timing_file {timing_file!r} != this shard's sidecar "
            f"{timing_sidecar_name(shard)!r}"
        )
    return {
        "shard_id": shard,
        "git_sha": git_sha,
        "probe_sha": probe_sha,
        "cold_start_s": float(cold),
        "timing_file": timing_file,
    }


def verify_probe_budget_evidence(summary: dict[str, Any], *, device: str, path_label: str) -> dict[str, str]:
    """The summary's own ledger invocation: finished, successful, one card, same host."""
    budget = summary.get("budget") or {}
    invocation_id = summary.get("invocation_id")
    if not isinstance(invocation_id, str) or not invocation_id:
        raise RunnerError(f"{path_label}: summary has no invocation_id")
    if budget.get("approved_sha") != (summary.get("authorization") or {}).get("approved_sha"):
        raise RunnerError(f"{path_label}: budget SHA differs from the authorization SHA")
    budget_file = budget.get("budget_file")
    if not isinstance(budget_file, str) or not budget_file:
        raise RunnerError(f"{path_label}: budget record lacks budget_file")
    records = [
        inv for inv in (budget.get("invocations") or [])
        if isinstance(inv, dict) and inv.get("invocation_id") == invocation_id
    ]
    if len(records) != 1:
        raise RunnerError(
            f"{path_label}: expected exactly one ledger record for {invocation_id!r}, "
            f"got {len(records)}"
        )
    inv = records[0]
    host = inv.get("host")
    if not isinstance(host, str) or not host:
        raise RunnerError(f"{path_label}: ledger invocation has no valid host")
    if inv.get("shard_id") != SHARD_BY_DEVICE[device]:
        raise RunnerError(f"{path_label}: ledger invocation shard does not match {device}")
    if inv.get("cards_used") != 1:
        raise RunnerError(f"{path_label}: ledger invocation cards_used != 1")
    if inv.get("finalized") is not True:
        raise RunnerError(f"{path_label}: ledger invocation is not finalized")
    exit_code = inv.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        raise RunnerError(
            f"{path_label}: ledger invocation exit_code {exit_code!r} is not a successful integer 0"
        )
    return {"host": host, "budget_file": budget_file, "invocation_id": invocation_id}


def verify_probe_timing_records(
    records: Sequence[dict[str, Any]],
    *,
    shard_id: int,
    device: str,
    selection: dict[str, Any],
    expected_scope: dict[str, Any],
    path_label: str,
) -> list[dict[str, Any]]:
    """The sidecar must cover EXACTLY the shard's 50 probes, each complete once."""
    n_cads = int(selection["counts"]["n_cads"])
    expected_batches = (n_cads + G3_BATCH_SIZE - 1) // G3_BATCH_SIZE
    probe_facts = probe_facts_for_shard(selection, shard_id)
    seen: dict[int, dict[str, Any]] = {}
    for i, record in enumerate(records):
        where = f"{path_label} record {i}"
        if record.get("record_type") != "query_timing" or record.get("schema_version") != G3_TIMING_SCHEMA_VERSION:
            raise RunnerError(f"{where}: not a {G3_TIMING_SCHEMA_VERSION} query_timing record")
        for field, expected in expected_scope.items():
            if record.get(field) != expected:
                raise RunnerError(
                    f"{where}: scope {field} {record.get(field)!r} != {expected!r}"
                )
        if record.get("shard_id") != shard_id or record.get("device") != device:
            raise RunnerError(f"{where}: shard/device mismatch for {device}/shard{shard_id}")
        text_index = record.get("text_index")
        if not isinstance(text_index, int) or isinstance(text_index, bool) or text_index not in probe_facts:
            raise RunnerError(f"{where}: text_index {text_index!r} is not one of this shard's probes")
        if text_index in seen:
            raise RunnerError(f"{where}: text_index {text_index} timed twice (duplicate query)")
        facts = probe_facts[text_index]
        if record.get("text_layer") != facts["text_layer"] or record.get("n_bert_tokens") != facts["n_bert_tokens"]:
            raise RunnerError(f"{where}: layer/length facts differ from the verified selection")
        if record.get("text_range") != [text_index, text_index + G3_TEXTS_PER_CHUNK]:
            raise RunnerError(f"{where}: text_range is not the single-query chunk")
        window = record.get("window")
        try:
            start = float(window["start_epoch"])
            end = float(window["end_epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError(f"{where}: invalid window {window!r}") from exc
        if not (math.isfinite(start) and math.isfinite(end)) or start > end:
            raise RunnerError(f"{where}: window is not finite/ordered")
        e2e = record.get("e2e_s")
        _require_finite_positive(e2e, f"{where} e2e_s")
        if not math.isclose(float(e2e), end - start, rel_tol=1e-6, abs_tol=1e-6):
            raise RunnerError(f"{where}: e2e_s does not equal end-start of its window")
        if record.get("n_pairs") != n_cads:
            raise RunnerError(
                f"{where}: n_pairs {record.get('n_pairs')!r} != the complete gallery {n_cads}"
            )
        if record.get("n_batches") != expected_batches:
            raise RunnerError(
                f"{where}: n_batches {record.get('n_batches')!r} != the batch{G3_BATCH_SIZE} "
                f"production count {expected_batches}"
            )
        if record.get("n_scored") != n_cads or record.get("n_rejected") != 0:
            raise RunnerError(
                f"{where}: query is not complete (scored={record.get('n_scored')!r} "
                f"rejected={record.get('n_rejected')!r}); the ETA prices only "
                "complete-gallery queries"
            )
        n_valid = record.get("n_valid_total")
        if isinstance(n_valid, bool) or not isinstance(n_valid, int) or n_valid <= 0:
            raise RunnerError(f"{where}: n_valid_total {n_valid!r} is not a positive int")
        if record.get("chunk_file") != chunk_file_name(shard_id, text_index, text_index + G3_TEXTS_PER_CHUNK):
            raise RunnerError(f"{where}: chunk_file does not name this query's chunk")
        seen[text_index] = record
    missing = sorted(set(probe_facts) - set(seen))
    if missing:
        raise RunnerError(
            f"{path_label}: {len(missing)} probe query(ies) have no timing record, "
            f"e.g. text_index {missing[:3]}; a probe chunk published without its "
            "timing line is unverifiable -- stop, do not fabricate times"
        )
    return [seen[t] for t in sorted(seen)]


# ---------------------------------------------------------------------------
# The two-card G3 ETA (slower card; per-layer slowest query; fail-closed)
# ---------------------------------------------------------------------------


def estimate_g3_probe_eta(
    selection: dict[str, Any],
    evidence_by_device: dict[str, dict[str, Any]],
    *,
    expected_probe_selection_sha256: str,
    overhead_seconds: Optional[float] = None,
    safety_factor: float = ETA_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Full-matrix ETA from the two REAL probe runs (fail-closed).

    Per card: ``sum over the global text layers (the card's slowest complete
    probe-query E2E in that layer x the layer's FULL text count in the card's
    shard) + the card's measured cold start``. The wall clock is the MAX of
    the two cards -- never a pooled ideal-2x rate. ``overhead_seconds=None``
    keeps the measured cold starts as the only overhead; an explicit value
    must be finite and >= the sum of both cold starts. The total is
    multiplied by ``safety_factor`` (>= 1.30) and compared against the
    604,800 s gate. The coefficient is never widened after seeing results.
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
    if selection.get("schema_version") != G3_SELECTION_SCHEMA_VERSION:
        raise RunnerError(f"eta-g3 requires a {G3_SELECTION_SCHEMA_VERSION} selection")
    verify_official_dimensions(int(selection["counts"]["n_texts"]), int(selection["counts"]["n_cads"]))
    if set(evidence_by_device) != set(SHARD_BY_DEVICE):
        raise RunnerError("exactly one probe evidence bundle per device (npu:0 and npu:1)")

    n_cads = int(selection["counts"]["n_cads"])
    full_pairs = int(selection["counts"]["n_pairs"])
    per_card: dict[str, Any] = {}
    windows: dict[str, dict[str, float]] = {}
    hosts: set[str] = set()
    ledgers: set[str] = set()
    invocation_ids: set[str] = set()
    env_shared: dict[str, dict[str, Any]] = {}
    for device, shard in sorted(SHARD_BY_DEVICE.items(), key=lambda kv: kv[1]):
        evidence = evidence_by_device[device]
        summary = evidence.get("summary") or {}
        records = evidence.get("timing_records")
        if not isinstance(records, list) or not records:
            raise RunnerError(f"{device}: no timing records")
        info = verify_probe_summary(summary, device=device, path_label=f"{device} summary")
        budget_info = verify_probe_budget_evidence(summary, device=device, path_label=f"{device} summary")
        if info["probe_sha"] != expected_probe_selection_sha256:
            raise RunnerError(
                f"{device}: run is bound to probe selection {info['probe_sha']!r}, "
                f"not the committed {expected_probe_selection_sha256!r}"
            )
        env_shared[device] = {field: summary.get(field) for field in _ETA_ENV_SHARED_FIELDS}
        scope = {
            PROBE_SCOPE_FIELD: expected_probe_selection_sha256,
            "git_sha": info["git_sha"],
            "pair_manifest_sha256": summary.get("pair_manifest_sha256"),
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "score_name": SCORE_NAME,
            "spec_version": SCORE_SPEC_VERSION,
        }
        if not scope["pair_manifest_sha256"]:
            raise RunnerError(f"{device}: summary lacks the pair-manifest binding")
        ordered = verify_probe_timing_records(
            records,
            shard_id=shard,
            device=device,
            selection=selection,
            expected_scope=scope,
            path_label=f"{device} timing",
        )
        hosts.add(budget_info["host"])
        ledgers.add(budget_info["budget_file"])
        invocation_ids.add(budget_info["invocation_id"])
        # chunk accounting cross-check: this invocation plus the previously
        # completed ones must account for the shard's full probe set
        completed_now = summary.get("chunks_completed_now")
        completed_before = summary.get("chunks_previously_completed")
        if (
            not isinstance(completed_now, int) or isinstance(completed_now, bool)
            or not isinstance(completed_before, int) or isinstance(completed_before, bool)
            or completed_now + completed_before != len(ordered)
        ):
            raise RunnerError(
                f"{device}: chunk accounting {completed_now!r}+{completed_before!r} "
                f"!= the {len(ordered)} timed queries"
            )
        shard_entry = next(s for s in selection["shards"] if int(s["shard_id"]) == shard)
        layer_full = {int(k): int(v) for k, v in shard_entry["layer_text_counts"].items()}
        layer_probe = {int(k): int(v) for k, v in shard_entry["layer_probe_counts"].items()}
        layers: dict[int, dict[str, Any]] = {}
        compute_seconds = 0.0
        for layer in range(G3_TEXT_LAYERS):
            layer_records = [r for r in ordered if r["text_layer"] == layer]
            if not layer_records:
                raise RunnerError(
                    f"{device}: no probe query in global text layer {layer}; "
                    "a missing layer cannot be priced and cannot be substituted"
                )
            slowest = max(layer_records, key=lambda r: (float(r["e2e_s"]), int(r["text_index"])))
            n_full = layer_full.get(layer, 0)
            if n_full <= 0:
                raise RunnerError(f"{device}: selection records no full texts in layer {layer}")
            seconds = float(slowest["e2e_s"]) * n_full
            compute_seconds += seconds
            layers[layer] = {
                "n_full_texts": n_full,
                "n_probe": len(layer_records),
                "n_probe_expected": layer_probe.get(layer),
                "slowest_text_index": int(slowest["text_index"]),
                "slowest_e2e_s": float(slowest["e2e_s"]),
                "seconds": seconds,
            }
        cold = float(summary["cold_start_s"])
        per_card[device] = {
            "shard_id": shard,
            "n_probe_queries": len(ordered),
            "n_pairs_scored": len(ordered) * n_cads,
            "cold_start_s": cold,
            "compute_seconds": compute_seconds,
            "completion_s": cold + compute_seconds,
            "layers": {str(k): v for k, v in sorted(layers.items())},
        }
        windows[device] = {
            "start_epoch": min(float(r["window"]["start_epoch"]) for r in ordered),
            "end_epoch": max(float(r["window"]["end_epoch"]) for r in ordered),
        }

    if len(invocation_ids) != 2:
        raise RunnerError("the two probe runs must be distinct invocations")
    if len(hosts) != 1:
        raise RunnerError(f"probe results come from different hosts {sorted(hosts)}")
    if len(ledgers) != 1:
        raise RunnerError(f"probe results were charged to different ledgers {sorted(ledgers)}")
    for field in _ETA_ENV_SHARED_FIELDS:
        values = [env_shared[d][field] for d in SHARD_BY_DEVICE]
        if not values[0] or values[0] != values[1]:
            raise RunnerError(f"the two probe runs disagree on {field}")

    overlap = window_overlap_fraction(windows["npu:0"], windows["npu:1"])
    if overlap < MIN_WINDOW_OVERLAP_FRACTION:
        raise RunnerError(
            f"actual scoring windows overlap {overlap:.6f} < {MIN_WINDOW_OVERLAP_FRACTION} "
            "of the shorter window; the two cards did not probe concurrently"
        )

    cold_sum = sum(card["cold_start_s"] for card in per_card.values())
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
        "schema_version": "text2cad-g3-eta-v1",
        "full_matrix_pairs": full_pairs,
        "probe_pairs_total": int(selection["counts"]["probe_pairs_total"]),
        "per_card": per_card,
        "slower_card": slower_card,
        "scoring_windows": windows,
        "window_overlap_fraction": overlap,
        "cold_start_seconds_sum": cold_sum,
        "extra_overhead_seconds": extra_overhead,
        "overhead_seconds_input": overhead_seconds,
        "wall_seconds_presafety": wall,
        "safety_factor": float(safety_factor),
        "eta_seconds": eta_seconds,
        "eta_hours": eta_seconds / 3600.0,
        "seven_day_seconds": SEVEN_DAY_SECONDS,
        "passes_seven_day_gate": eta_seconds <= SEVEN_DAY_SECONDS,
        "notes": (
            "per-card compute = sum over global text layers (slowest complete "
            "probe-query E2E in the layer x the layer's FULL text count in the "
            "card's shard); query E2E includes score + serialization + atomic "
            "chunk publish; the wall clock is the slower card's completion "
            "(never a pooled ideal-2x); extra I/O or failure margin enters via "
            "overhead_seconds and is reported verbatim"
        ),
    }


# ---------------------------------------------------------------------------
# CLI: select-g3 (generate the review-candidate config) / eta-g3 (CPU ETA)
# ---------------------------------------------------------------------------


def _load_expected_pair_hashes(path) -> dict[str, str]:
    """Reviewed artifact fingerprints (committed config -> {file: sha256})."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RunnerError(f"cannot read expected pair-manifest hashes {path}: {exc}") from exc
    artifacts = data.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RunnerError(f"{path}: missing artifact_sha256 object")
    out = {str(k): str(v) for k, v in artifacts.items()}
    manifest_sha = data.get("pair_manifest_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise RunnerError(f"{path}: missing/invalid pair_manifest_sha256")
    out[PAIR_MANIFEST_FILE] = manifest_sha
    return out


def _write_config_lf(out_path, payload: str, label: str) -> Path:
    out = Path(out_path)
    if out.exists():
        raise RunnerError(f"refusing to overwrite {out} ({label})")
    out.write_text(payload, encoding="utf-8", newline="\n")
    return out


def run_select_g3(args) -> Path:
    """``select-g3``: generate the fixed blind probe selection config.

    Pure CPU. Reads the four frozen worker artifacts, the audited token
    lengths and the frozen G2 config (hashed, never parsed for content
    decisions); binds every input SHA into the provenance; LF output; never
    overwrites. The produced file is a REVIEW CANDIDATE -- the production
    runner pins the committed path and re-verifies everything.
    """
    if not args.token_audit:
        raise RunnerError("select-g3 requires --token-audit (derived audit JSON)")
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        raise RunnerError("select-g3 requires --out")
    if out_path.exists():
        raise RunnerError(f"refusing to overwrite {out_path} (g3 probe selection)")
    repo_root = Path(args.repo_root).resolve()
    g2_path = repo_root / G2_SELECTION_CONFIG_REL
    if not g2_path.is_file():
        raise RunnerError(f"frozen G2 config {g2_path} does not exist")

    manifest_dir = Path(args.pair_manifest_dir)
    text_records, n_cads = load_g3_manifest_records(manifest_dir)
    verify_official_dimensions(len(text_records), n_cads)
    audit = json.loads(Path(args.token_audit).read_text(encoding="utf-8"))
    token_lengths = token_lengths_from_audit(
        audit,
        expected_texts_sha256=sha256_file(manifest_dir / TEXTS_FILE),
        expected_bert_file_sha256=dict(PINNED_BERT_FILE_SHA256),
        n_texts=len(text_records),
    )
    selection = build_g3_probe_selection(text_records, n_cads, token_lengths)
    selection["provenance"] = {
        "texts_jsonl_sha256": sha256_file(manifest_dir / TEXTS_FILE),
        "cad_store_jsonl_sha256": sha256_file(manifest_dir / CAD_STORE_FILE),
        "cad_length_index_sha256": sha256_file(manifest_dir / CAD_LENGTH_FILE),
        "pair_manifest_sha256": sha256_file(manifest_dir / PAIR_MANIFEST_FILE),
        "token_audit_sha256": sha256_file(args.token_audit),
        "g2_selection_sha256": sha256_file(g2_path),
        "generated_by": "text2cad_g3_probe.run_select_g3 (CPU, zero forward)",
    }
    payload = json.dumps(selection, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    return _write_config_lf(out_path, payload, "g3 probe selection")


def run_eta_g3(args) -> tuple[dict[str, Any], Optional[Path]]:
    """``eta-g3``: two-card full-matrix ETA from the two REAL probe runs.

    Pure CPU. Re-verifies the committed probe selection against the live
    manifest + token audit + frozen G2 config, binds both summaries and both
    timing sidecars to the committed selection SHA and the reviewed
    fingerprints, then applies the slower-card / per-layer-slowest-query /
    cold-start / overlap / safety-factor rules.
    """
    if len(args.card_summary) != 2:
        raise RunnerError(
            "eta-g3 requires exactly two --card-summary files (one per card; a "
            "single card is never scaled by an ideal 2x)"
        )
    paths = [Path(p).resolve() for p in args.card_summary]
    if paths[0] == paths[1]:
        raise RunnerError("eta-g3 requires two distinct card summary files")
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

    repo_root = Path(args.repo_root).resolve()
    reviewed = _load_expected_pair_hashes(repo_root / EXPECTED_PAIR_HASHES_REL)
    manifest_dir = Path(args.pair_manifest_dir)
    manifest = load_pair_manifest(manifest_dir, expected_sha256=reviewed)
    verify_official_dimensions(
        int(manifest["counts"]["n_unique_text"]), int(manifest["counts"]["n_gallery"])
    )
    selection_path = repo_root / G3_PROBE_SELECTION_REL
    if not selection_path.is_file():
        raise RunnerError(f"committed probe selection {selection_path} does not exist")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    verify_g3_probe_selection_against_manifest(selection, manifest_dir)
    verify_g3_token_audit(selection, args.token_audit, manifest_dir)
    g2_sha = verify_g3_g2_config_binding(selection, repo_root)
    selection_sha = sha256_file(selection_path)

    evidence_by_device: dict[str, dict[str, Any]] = {}
    for path, summary in zip(paths, summaries):
        device = summary.get("device")
        if device not in SHARD_BY_DEVICE or device in evidence_by_device:
            raise RunnerError(
                f"eta-g3 summaries must be one per NPU device, got {device!r} ({path})"
            )
        timing_name = summary.get("timing_file")
        if not isinstance(timing_name, str) or Path(timing_name).name != timing_name:
            raise RunnerError(f"{device}: bad timing_file reference {timing_name!r}")
        timing_path = path.parent / timing_name
        records = read_timing_records(timing_path)
        for record in records:
            chunk = record.get("chunk_file")
            if isinstance(chunk, str) and Path(chunk).name == chunk:
                if not (path.parent / chunk).is_file():
                    raise RunnerError(
                        f"{device}: timing claims chunk {chunk} but {path.parent / chunk} "
                        "does not exist; a timing line must imply its published chunk"
                    )
        evidence_by_device[device] = {"summary": summary, "timing_records": records}

    estimate = estimate_g3_probe_eta(
        selection,
        evidence_by_device,
        expected_probe_selection_sha256=selection_sha,
        overhead_seconds=args.overhead_seconds,
    )
    estimate["inputs"] = {
        "card_summaries": [str(p) for p in paths],
        "overhead_seconds": args.overhead_seconds,
        "g3_probe_selection_sha256": selection_sha,
        "g2_selection_sha256": g2_sha,
        "pair_manifest_sha256": reviewed[PAIR_MANIFEST_FILE],
    }
    payload = json.dumps(estimate, indent=1, sort_keys=True) + "\n"
    written: Optional[Path] = None
    if args.out:
        written = _write_config_lf(Path(args.out), payload, "eta-g3 estimate")
    else:
        print(payload)
    return estimate, written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Text2CAD G3 full-gallery probe utilities (pure CPU, zero forward). "
            "select-g3 generates the fixed blind 100-text probe selection from "
            "the frozen worker artifacts + audited token lengths + frozen G2 "
            "config; eta-g3 computes the two-card full-matrix ETA from the two "
            "REAL probe run summaries + timing sidecars (slower card, per-layer "
            "slowest complete query, measured cold starts, >=80% window "
            "overlap, >=1.30 safety factor, 604,800 s gate)."
        )
    )
    parser.add_argument("--mode", required=True, choices=["select-g3", "eta-g3"])
    parser.add_argument("--pair-manifest-dir", required=True)
    parser.add_argument("--token-audit", default=None,
                        help="reviewed derived token-length audit JSON (required by both modes)")
    parser.add_argument("--card-summary", action="append", default=[],
                        help="eta-g3: a per-card probe run summary JSON (pass exactly two)")
    parser.add_argument("--overhead-seconds", type=float, default=None,
                        help="eta-g3: total overhead; must be >= the two measured cold starts")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default=None, help="output JSON path (never overwritten)")
    args = parser.parse_args(argv)

    try:
        if args.mode == "select-g3":
            out_path = run_select_g3(args)
            print(json.dumps({"status": "ok", "result": str(out_path)}))
            return 0
        estimate, written = run_eta_g3(args)
        print(json.dumps({
            "status": "ok",
            "result": str(written) if written else None,
            "eta_seconds": estimate["eta_seconds"],
            "passes_seven_day_gate": estimate["passes_seven_day_gate"],
        }))
        return 0 if estimate["passes_seven_day_gate"] else 3
    except (RunnerError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
