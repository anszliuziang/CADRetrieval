#!/usr/bin/env python
"""Frozen Text2CAD retrieval-accuracy night: prepare / replay / score.

Plan: experiments/plans/IDEA01_TEXT2CAD_RETRIEVAL_ACCURACY_2026-11-18.md
(the "small-batch then matrix" contract; run order table assigns this script
the prepare/replay/score stages and eval_text2cad_accuracy_night.py the
fit/evaluate stages; experiments/scripts/launch_text2cad_accuracy_stage.py is
the single stage controller that invokes `score` per NPU shard and then the
evaluator, so this script's argv is FIXED by that controller).

Three subcommands, all sharing one strict-refusal style (any gate failure
exits non-zero with the reason; nothing is silently skipped):

  prepare  CPU only, zero forward, no torch import. Builds
      <out>/{validation,test}/inputs.json + eval.json (the artifact schema
      of the plan, section "输入约定") plus <out>/prepare_report.json.
      - validation gallery: every official v1.0 validation UID that has a
        verifiable cad_seq.zip member, UID-ascending, ZIP member uniqueness
        checked; L0 text = abstract.strip(); queries without CAD or without
        text are dropped and counted; text groups verbatim-identical to any
        G5 test text are excluded; groups are ordered by
        SHA256("cad-rank-dev-20261118\n" + text) and split 48 background /
        256 fit / 256 select;
      - test gallery: the certified G5 cad_store (8,046) verbatim; texts and
        original queries recovered from the G5 query/qrels artifacts, deduped
        by strip(), ordered by SHA256("cad-rank-test-20261118\n" + text),
        first 1,024 groups kept (a smaller preregistered prefix is applied at
        score/evaluate time via --text-stop, never here);
      - the 48 validation background texts are reused verbatim as the test
        axis's background texts; validation-vs-test UID intersection must be
        empty; NFKC/casefold/punctuation collision counting is diagnostic.

  replay   NPU forward on the certified E2 REPLAY subset only: two non-empty
      replay texts x (one original full batch of 16 + the original real tail
      batch of 14) = 60 pairs, members and shapes preserved from the pinned
      replay manifest; every consumed record is SHA-pinned against the
      certified E2 replay_gate_record.json; the native SUM column must match
      the certified reference within max(1e-3, 1e-6*|ref|) with exact valid
      counts, and the six field-group sums must add back to SUM within 1e-8.

  score    One NPU shard of one axis: strict frozen load (load_frozen_text2cad
      + ZipCandidateStore over inputs.json members), FP32/full272/b16 honest
      tail, one forward per pair, text cache OFF, trim OFF. Per text publishes
      chunks/t{index:05d}.npz with scores float64 (n_cads, 7): column 0 is the
      audited historical SUM (two_head_factorized_score) and columns 1..6 are
      the six mutually exclusive field-group sums in the fixed column order
      sketch_x, sketch_y, extrude_numeric, extrude_boolean, delimiter,
      terminal. Sharding is by text index modulo --shards. counts.npy
      (int64 (n_cads, 6) candidate group counts) is published exclusively by
      shard 0; every shard additionally writes its own counts receipt and
      verifies the published bytes equal its own computation.

Isolation (CLAUDE.md sec.5.1): inputs.json carries only texts and candidate
member strings -- no UID, no query_id, no caption metadata, no relevance.
UID/query linkage lives only in eval.json, which this script's score path
never opens (the controller passes no --eval to score).

All asset SHA-256 pins below are copied from the plan's "输入约定" table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.m0_manifest import (  # noqa: E402
    index_cad_seq_zip,
    load_split_json,
    stream_csv_uid_l0,
)

SCHEMA_INPUTS = "text2cad-accuracy-inputs-v1"
SCHEMA_EVAL = "text2cad-accuracy-eval-v1"
SCHEMA_PREPARE_REPORT = "text2cad-accuracy-prepare-report-v1"
SCHEMA_REPLAY_RESULT = "text2cad-accuracy-replay-result-v1"
SCHEMA_SHARD_SUMMARY = "text2cad-accuracy-shard-summary-v1"
SCHEMA_COUNTS_RECEIPT = "text2cad-accuracy-counts-receipt-v1"

DEV_SALT = "cad-rank-dev-20261118\n"
TEST_SALT = "cad-rank-test-20261118\n"
N_BACKGROUND = 48
N_FIT = 256
N_SELECT = 256
N_TEST_GROUPS = 1024
#: preregistered test prefixes (plan: 1,024/768/512/256/128; the controller
#: enforces the same set)
TEST_PREFIXES = (1024, 768, 512, 256, 128)
ROLE_BACKGROUND = "background"
ROLE_FIT = "fit"
ROLE_SELECT = "select"
ROLE_TEST = "test"

#: fixed field-group column order (plan section "共享前向与正确性门" item 2)
GROUP_NAMES = ("sketch_x", "sketch_y", "extrude_numeric", "extrude_boolean",
               "delimiter", "terminal")
N_GROUPS = len(GROUP_NAMES)
#: |sum(groups) - historical SUM| <= 1e-8 and group counts sum exactly
GROUP_TOTAL_TOL = 1e-8
#: replay tolerance (plan item 4), identical to the E2 gate constants
REPLAY_TOLERANCE_ABS = 1e-3
REPLAY_TOLERANCE_REL = 1e-6
#: the certified E2 replay tail batch is the original G4 tail (dense 2032..2045
#: maps to original cad_index 8032..8045; the replay store is 127*16 + 14)
REPLAY_TAIL_ORIGINAL_FIRST = 8032
REPLAY_TAIL_ORIGINAL_LAST = 8045
BATCH_SIZE = 16  # G4 protocol (honest tail)

# ---------------------------------------------------------------------------
# Pinned assets (plan table "输入约定"; verified on every prepare/replay/score)
# ---------------------------------------------------------------------------

PINNED_CSV_SHA256 = "32427ca7169f8b93411b64a025c387cdb0b55d9b9ce0e9cf9755ac8a3528f4a3"
PINNED_SPLIT_JSON_SHA256 = "4ea6db9027b4cebee8f37903b6601b73b810adacd808a4874dbe847c7c508e7b"
PINNED_CAD_SEQ_ZIP_SHA256 = "9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa"
PINNED_G5_QUERY_SHA256 = "a0d8504f5a5555e5312dd38a31321a6ea254087e6390e786559cfc0b6d307fcd"
PINNED_G5_GALLERY_SHA256 = "09782bd6d316e7b0863c70f1df9cd6504e7dc41ce3bb9a433821b1c22ab85e30"
PINNED_G5_QRELS_SHA256 = "f0677928ae2412ca4bcd8952c95d89f0568c5bb214c7bb53e82e0059555c87d6"
PINNED_G5_CAD_STORE_SHA256 = "2f441110f013bb614b0bd2fdb7a3b49569a231bbf3f8eaa2c03ac1d725cb1380"
PINNED_E2_GATE_RECORD_SHA256 = "59f3e8c1469d5f2056dd0579de687f328b8f7b37357a2f3e3276773e27b5bcc2"

#: frozen expectations, seen by Codex's local preflight (plan section 方案验收)
EXPECTED_VALIDATION = {
    "n_gallery": 8925,
    "n_queries": 8829,
    "n_groups": 4023,
    "n_groups_after_exclusion": 3143,
}
EXPECTED_TEST = {"n_gallery": 8046, "n_queries": 7934, "n_unique_texts": 3644}

QUERY_RECORD_KEYS = frozenset({"level", "n_chars", "query_id", "text", "text_sha256"})
GALLERY_RECORD_KEYS = frozenset(
    {"cad_id", "cad_seq_member", "cadmium_native_present", "minimal_json_member",
     "n_representations"}
)
QRELS_RECORD_KEYS = frozenset({"query_id", "cad_id", "relevance"})
CAD_STORE_RECORD_KEYS = frozenset({"cad_index", "member"})


class NightError(RuntimeError):
    """Hard refusal of a prepare/replay/score stage; exit non-zero, no forward."""


# ---------------------------------------------------------------------------
# Hashing / canonical JSON
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def salted_rank_key(salt: str, text: str) -> str:
    """Fixed selection order: SHA256(salt + text) hex, lexicographic."""
    return hashlib.sha256((salt + text).encode("utf-8")).hexdigest()


def normalized_for_diagnostic(text: str) -> str:
    """NFKC + casefold + punctuation-to-space + whitespace collapse.

    Diagnostic collision counting ONLY (plan: the second count is never a
    selection or exclusion rule).
    """
    nfkc = unicodedata.normalize("NFKC", text).casefold()
    chars = [" " if unicodedata.category(ch).startswith("P") else ch for ch in nfkc]
    return " ".join("".join(chars).split())


def canonical_json_bytes(obj) -> bytes:
    """UTF-8 / LF / canonical JSON (sorted keys, no insignificant whitespace)."""
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def write_canonical_json(path, obj) -> str:
    """Atomic canonical-JSON publish; returns the file's sha256."""
    path = Path(path)
    data = canonical_json_bytes(obj)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def read_strict_jsonl(path, allowed_keys: frozenset, label: str) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or set(obj) != set(allowed_keys):
                raise NightError(
                    f"{label} line {lineno}: keys must be exactly {sorted(allowed_keys)}"
                )
            records.append(obj)
    return records


def require_pin(path, expected_sha: str, label: str) -> str:
    got = sha256_file(path)
    if got != expected_sha:
        raise NightError(f"{label} sha256 {got} != pinned {expected_sha}")
    return got


# ---------------------------------------------------------------------------
# prepare: axis construction
# ---------------------------------------------------------------------------


def sort_groups_by_salt(groups: dict[str, list], salt: str) -> list[tuple[str, list]]:
    """Fixed order: SHA256(salt + text) lexicographic; an (impossible) hash
    collision is a refusal, never a silent tie-break."""
    keyed: dict[str, tuple[str, list]] = {}
    for text, uids in groups.items():
        key = salted_rank_key(salt, text)
        if key in keyed:
            raise NightError(f"salt hash collision between two distinct text groups ({key})")
        keyed[key] = (text, uids)
    return [keyed[key] for key in sorted(keyed)]


def build_validation_axis(csv_path: Path, split_json_path: Path, cad_zip_path: Path,
                          g5_dir: Path) -> dict:
    """Validation gallery + background/fit/select text groups (pure CPU)."""
    csv_sha = require_pin(csv_path, PINNED_CSV_SHA256, "text2cad_v1.0.csv")
    split_sha = require_pin(split_json_path, PINNED_SPLIT_JSON_SHA256, "train_test_val.json")
    zip_sha = require_pin(cad_zip_path, PINNED_CAD_SEQ_ZIP_SHA256, "cad_seq.zip")
    query_sha = require_pin(Path(g5_dir) / "query.jsonl", PINNED_G5_QUERY_SHA256, "G5 query.jsonl")

    splits = load_split_json(split_json_path)
    validation_uids = list(splits["validation"])
    if len(set(validation_uids)) != len(validation_uids):
        raise NightError("train_test_val.json validation split contains duplicate UIDs")

    # ZIP member index with uniqueness check (plan: 按 UID 升序核对 ZIP 成员唯一)
    member_of_uid = index_cad_seq_zip(cad_zip_path)
    if len(set(member_of_uid.values())) != len(member_of_uid):
        raise NightError("cad_seq.zip contains two members resolving to the same UID")

    gallery_uids = sorted(u for u in validation_uids if u in member_of_uid)
    if len(gallery_uids) != EXPECTED_VALIDATION["n_gallery"]:
        raise NightError(
            f"validation gallery has {len(gallery_uids)} UIDs with verifiable CAD, "
            f"expected frozen {EXPECTED_VALIDATION['n_gallery']}"
        )
    members = [member_of_uid[u] for u in gallery_uids]

    csv_index = stream_csv_uid_l0(csv_path)
    # queries: registered UID present in the gallery AND non-empty L0 text
    query_uids = [u for u in gallery_uids if csv_index.get(u, {}).get("abstract")]
    if len(query_uids) != EXPECTED_VALIDATION["n_queries"]:
        raise NightError(
            f"validation non-empty L0 queries = {len(query_uids)}, "
            f"expected frozen {EXPECTED_VALIDATION['n_queries']}"
        )

    groups: dict[str, list[str]] = {}
    for uid in query_uids:
        groups.setdefault(csv_index[uid]["abstract"], []).append(uid)
    for text in groups:
        groups[text].sort()
    if len(groups) != EXPECTED_VALIDATION["n_groups"]:
        raise NightError(
            f"validation unique text groups = {len(groups)}, "
            f"expected frozen {EXPECTED_VALIDATION['n_groups']}"
        )

    # G5 test text set (verbatim, already strip()ed by the G5 protocol)
    g5_queries = read_strict_jsonl(Path(g5_dir) / "query.jsonl", QUERY_RECORD_KEYS, "query.jsonl")
    g5_texts = sorted({row["text"].strip() for row in g5_queries})
    if len(g5_texts) != EXPECTED_TEST["n_unique_texts"]:
        raise NightError(
            f"G5 unique stripped texts = {len(g5_texts)}, expected frozen "
            f"{EXPECTED_TEST['n_unique_texts']}"
        )
    g5_text_set = set(g5_texts)
    g5_norm_set = {normalized_for_diagnostic(t) for t in g5_texts}

    verbatim_hits = sorted(t for t in groups if t in g5_text_set)
    remaining = {t: uids for t, uids in groups.items() if t not in g5_text_set}
    if len(remaining) != EXPECTED_VALIDATION["n_groups_after_exclusion"]:
        raise NightError(
            f"validation groups after G5-verbatim exclusion = {len(remaining)}, "
            f"expected frozen {EXPECTED_VALIDATION['n_groups_after_exclusion']}"
        )
    normalized_hits = sum(1 for t in remaining if normalized_for_diagnostic(t) in g5_norm_set)

    ordered = sort_groups_by_salt(remaining, DEV_SALT)
    if len(ordered) < N_BACKGROUND + N_FIT + N_SELECT:
        raise NightError(
            f"only {len(ordered)} eligible validation groups; the frozen 48/256/256 "
            "selection does not fit"
        )
    roles = ([ROLE_BACKGROUND] * N_BACKGROUND + [ROLE_FIT] * N_FIT + [ROLE_SELECT] * N_SELECT)
    texts = [text for text, _ in ordered[: len(roles)]]

    cad_index_of_uid = {u: i for i, u in enumerate(gallery_uids)}
    queries = []
    for text_index, (text, uids) in enumerate(ordered[N_BACKGROUND: len(roles)]):
        for uid in uids:
            queries.append({
                "query_id": None,  # filled below, in fixed (text_index, cad_index) order
                "text_index": text_index + N_BACKGROUND,
                "cad_index": cad_index_of_uid[uid],
            })
    queries.sort(key=lambda q: (q["text_index"], q["cad_index"]))
    for rank, q in enumerate(queries):
        q["query_id"] = f"V{rank:06d}"

    return {
        "axis": "validation",
        "texts": texts,
        "text_roles": roles,
        "members": members,
        "cad_uids": gallery_uids,
        "queries": queries,
        "report": {
            "asset_sha256": {
                "text2cad_v1.0.csv": csv_sha,
                "train_test_val.json": split_sha,
                "cad_seq.zip": zip_sha,
                "g5_query.jsonl": query_sha,
            },
            "n_validation_split_uids": len(validation_uids),
            "n_gallery": len(gallery_uids),
            "n_queries": len(query_uids),
            "n_unique_text_groups": len(groups),
            "n_groups_excluded_g5_verbatim": len(verbatim_hits),
            "n_groups_after_exclusion": len(remaining),
            "n_groups_diagnostic_nfkc_collision": int(normalized_hits),
            "n_background": N_BACKGROUND,
            "n_fit": N_FIT,
            "n_select": N_SELECT,
            "excluded_g5_verbatim_texts": verbatim_hits,
            "selection_rule": "SHA256('cad-rank-dev-20261118\\n' + text) lexicographic; "
                              "background = first 48, fit = next 256, select = next 256",
        },
    }


def build_test_axis(g5_dir: Path, background_texts: list[str]) -> dict:
    """Test gallery (certified G5 cad_store) + fixed test text groups (pure CPU)."""
    pins = {
        "query.jsonl": (QUERY_RECORD_KEYS, PINNED_G5_QUERY_SHA256),
        "gallery.jsonl": (GALLERY_RECORD_KEYS, PINNED_G5_GALLERY_SHA256),
        "qrels.jsonl": (QRELS_RECORD_KEYS, PINNED_G5_QRELS_SHA256),
        "cad_store.jsonl": (CAD_STORE_RECORD_KEYS, PINNED_G5_CAD_STORE_SHA256),
    }
    shas = {}
    for name, (keys, pin) in pins.items():
        shas[name] = require_pin(Path(g5_dir) / name, pin, f"G5 {name}")
    queries_raw = read_strict_jsonl(Path(g5_dir) / "query.jsonl", QUERY_RECORD_KEYS, "query.jsonl")
    gallery = read_strict_jsonl(Path(g5_dir) / "gallery.jsonl", GALLERY_RECORD_KEYS, "gallery.jsonl")
    qrels = read_strict_jsonl(Path(g5_dir) / "qrels.jsonl", QRELS_RECORD_KEYS, "qrels.jsonl")
    cad_store = read_strict_jsonl(Path(g5_dir) / "cad_store.jsonl", CAD_STORE_RECORD_KEYS, "cad_store.jsonl")

    if len(gallery) != EXPECTED_TEST["n_gallery"] or len(cad_store) != EXPECTED_TEST["n_gallery"]:
        raise NightError(
            f"G5 gallery/cad_store have {len(gallery)}/{len(cad_store)} rows, "
            f"expected frozen {EXPECTED_TEST['n_gallery']}"
        )
    if [row["cad_index"] for row in cad_store] != list(range(len(cad_store))):
        raise NightError("G5 cad_store cad_index is not dense 0..n-1")
    cad_ids = [row["cad_id"] for row in gallery]
    if cad_ids != sorted(cad_ids) or len(set(cad_ids)) != len(cad_ids):
        raise NightError("G5 gallery cad_id order is not unique-ascending (protocol identity)")
    for i, row in enumerate(cad_store):
        if row["member"] != gallery[i]["cad_seq_member"]:
            raise NightError(
                f"G5 cad_store[{i}] member differs from gallery[{i}].cad_seq_member "
                "(certified linkage drift)"
            )
    if len({row["query_id"] for row in queries_raw}) != len(queries_raw):
        raise NightError("G5 query.jsonl has duplicate query_id")
    if len(queries_raw) != EXPECTED_TEST["n_queries"]:
        raise NightError(
            f"G5 queries = {len(queries_raw)}, expected frozen {EXPECTED_TEST['n_queries']}"
        )
    for row in queries_raw:
        text = row["text"]
        if (row["level"] != "L0" or not isinstance(text, str) or not text or text != text.strip()
                or row["text_sha256"] != sha256_text(text) or row["n_chars"] != len(text)):
            raise NightError(f"G5 query {row['query_id']}: identity fields drifted")

    positive_of = {}
    for row in qrels:
        qid = row["query_id"]
        if qid in positive_of:
            raise NightError(f"G5 qrels has two rows for {qid}")
        if row["relevance"] != 1:
            raise NightError(f"G5 qrels {qid}: relevance must be 1")
        positive_of[qid] = row["cad_id"]
    if set(positive_of) != {row["query_id"] for row in queries_raw}:
        raise NightError("G5 qrels coverage differs from query.jsonl")
    cad_index_of = {cad_id: i for i, cad_id in enumerate(cad_ids)}
    unknown = sorted(set(positive_of.values()) - set(cad_index_of))
    if unknown:
        raise NightError(f"G5 qrels positives outside the gallery: {unknown[:5]}")

    groups: dict[str, list[dict]] = {}
    for row in queries_raw:
        groups.setdefault(row["text"].strip(), []).append(row)
    ordered = sort_groups_by_salt(groups, TEST_SALT)
    if len(ordered) < N_TEST_GROUPS:
        raise NightError(
            f"only {len(ordered)} G5 unique text groups; frozen {N_TEST_GROUPS} do not fit"
        )
    chosen = ordered[:N_TEST_GROUPS]
    roles = [ROLE_BACKGROUND] * N_BACKGROUND + [ROLE_TEST] * len(chosen)
    texts = list(background_texts) + [text for text, _ in chosen]
    if len(background_texts) != N_BACKGROUND:
        raise NightError(f"background texts {len(background_texts)} != frozen {N_BACKGROUND}")

    queries = []
    for offset, (text, rows) in enumerate(chosen):
        text_index = N_BACKGROUND + offset
        for row in sorted(rows, key=lambda r: r["query_id"]):
            queries.append({
                "query_id": row["query_id"],
                "text_index": text_index,
                "cad_index": cad_index_of[positive_of[row["query_id"]]],
            })
    queries.sort(key=lambda q: (q["text_index"], q["cad_index"]))

    return {
        "axis": "test",
        "texts": texts,
        "text_roles": roles,
        "members": [row["member"] for row in cad_store],
        "cad_uids": cad_ids,
        "queries": queries,
        "report": {
            "asset_sha256": {f"g5_{name.replace('.jsonl', '')}": sha for name, sha in shas.items()},
            "n_gallery": len(cad_ids),
            "n_queries": len(queries_raw),
            "n_unique_text_groups": len(groups),
            "n_test_groups": len(chosen),
            "n_background": N_BACKGROUND,
            "background_source": "the 48 validation background texts, verbatim",
            "queries_in_selected_groups": len(queries),
            "selection_rule": "SHA256('cad-rank-test-20261118\\n' + text) lexicographic; "
                              f"first {N_TEST_GROUPS} groups; a smaller preregistered "
                              "prefix is applied at score/evaluate time via --text-stop",
        },
    }


def write_axis(prepared_root: Path, payload: dict) -> dict:
    """Publish <root>/<axis>/inputs.json + eval.json per the plan's schema."""
    axis = payload["axis"]
    axis_dir = Path(prepared_root) / axis
    axis_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = axis_dir / "inputs.json"
    eval_path = axis_dir / "eval.json"
    for path in (inputs_path, eval_path):
        if path.exists():
            raise NightError(f"refusing to overwrite existing artifact: {path}")
    inputs_sha = write_canonical_json(inputs_path, {
        "schema": SCHEMA_INPUTS,
        "axis": axis,
        "texts": payload["texts"],
        "members": payload["members"],
    })
    eval_sha = write_canonical_json(eval_path, {
        "schema": SCHEMA_EVAL,
        "axis": axis,
        "inputs_sha256": inputs_sha,
        "cad_uids": payload["cad_uids"],
        "text_roles": payload["text_roles"],
        "queries": payload["queries"],
    })
    if len(payload["texts"]) != len(payload["text_roles"]):
        raise NightError("texts/text_roles length drift")
    return {"axis": axis, "inputs_sha256": inputs_sha, "eval_sha256": eval_sha,
            "n_texts": len(payload["texts"]), "n_members": len(payload["members"]),
            "n_queries": len(payload["queries"])}


def run_prepare(args) -> int:
    started = time.time()
    out = Path(args.out)
    if out.exists():
        raise NightError(f"prepared output root already exists: {out}")
    validation = build_validation_axis(
        Path(args.text2cad_csv), Path(args.split_json), Path(args.cad_zip), Path(args.g5_dir))
    test = build_test_axis(Path(args.g5_dir), validation["texts"][:N_BACKGROUND])

    val_uids = set(validation["cad_uids"])
    test_uids = set(test["cad_uids"])
    overlap = sorted(val_uids & test_uids)
    if overlap:
        raise NightError(
            f"validation and test galleries share {len(overlap)} UIDs (e.g. {overlap[:5]}); "
            "the plan requires an empty intersection"
        )

    axes = [write_axis(out, validation), write_axis(out, test)]
    report = {
        "schema": SCHEMA_PREPARE_REPORT,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": None,  # filled just before the final write below
        "assets": {
            "text2cad_csv": str(args.text2cad_csv),
            "split_json": str(args.split_json),
            "cad_zip": str(args.cad_zip),
            "g5_dir": str(args.g5_dir),
        },
        "axes": axes,
        "uid_intersection_nonempty": False,
        "validation": validation["report"],
        "test": test["report"],
    }
    report["elapsed_seconds"] = round(time.time() - started, 3)
    write_canonical_json(out / "prepare_report.json", report)
    print(json.dumps({"status": "prepare_complete", "out": str(out),
                      "axes": [{k: v for k, v in a.items()} for a in axes],
                      "uid_intersection": 0}, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------
# score: field-group arithmetic (torch-bearing; imported lazily by cmd_score /
# cmd_replay only, so `prepare` and the unit tests stay torch-free)
# ---------------------------------------------------------------------------


def candidate_group_tables(cad_vec):
    """Per-candidate field-group assignment from the official target rows.

    ``cad_vec`` is the official int (272, 2) tensor; rows 1..N-1 are targets.
    Returns (member_flat_indices int64 (n,), member_group int8 (n,), counts
    int64 (6,)) where a flat index p encodes (target row p // 2, head p % 2)
    over the (N-1) x 2 factor grid. Any non-zero target field outside the six
    groups is a hard refusal (plan: 非法或未归组的非零字段拒绝).
    """
    import torch

    target = cad_vec[1:].to(torch.long)
    x, y = target[:, 0], target[:, 1]
    valid_x, valid_y = x != 0, y != 0
    coordinate = (x >= 11) & (y >= 11)
    numeric = (x >= 11) & (y == 0)
    boolean = (x >= 7) & (x <= 10) & (y == 0)
    delimiter = (x >= 2) & (x <= 6) & (y == 0)
    terminal = (x == 1) & (y == 0)
    gx = torch.full(x.shape, -1, dtype=torch.int8)
    gx[coordinate] = 0
    gx[numeric] = 2
    gx[boolean] = 3
    gx[delimiter] = 4
    gx[terminal] = 5
    gy = torch.full(y.shape, -1, dtype=torch.int8)
    gy[coordinate] = 1
    ungrouped = (valid_x & (gx < 0)) | (valid_y & (gy < 0))
    if bool(ungrouped.any()):
        rows = ungrouped.nonzero(as_tuple=True)[0][:5].tolist()
        raise NightError(
            f"candidate has non-zero target field(s) outside the six fixed groups at "
            f"target rows {rows} (x,y first offenders: "
            f"{[(int(x[r]), int(y[r])) for r in ungrouped.nonzero(as_tuple=True)[0][:5]]})"
        )
    n_valid_total = int(valid_x.sum()) + int(valid_y.sum())
    x_rows = (gx >= 0).nonzero(as_tuple=True)[0]
    y_rows = (gy >= 0).nonzero(as_tuple=True)[0]
    flat = torch.cat([x_rows * 2, y_rows * 2 + 1])
    group = torch.cat([gx[x_rows], gy[y_rows]])
    order = torch.argsort(flat)
    flat, group = flat[order], group[order]
    counts = torch.bincount(group, minlength=N_GROUPS).to(torch.int64)
    if int(counts.sum()) != n_valid_total:
        raise NightError("group counts do not sum to n_valid_total (internal arithmetic drift)")
    if counts[-1] != 1 or not bool(terminal[(valid_x | valid_y).nonzero(as_tuple=True)[0][-1]]):
        raise NightError("expected exactly one final target token [1, 0]")
    return flat.to(torch.int64), group.to(torch.int8), counts


def build_gallery_group_layout(store, n_cads: int) -> dict:
    """Padded (n_cads, M) member/group tables + (n_cads, 6) counts for a store."""
    import numpy as np
    import torch

    pos_rows, grp_rows, count_rows = [], [], []
    for cad_index in range(n_cads):
        flat, group, counts = candidate_group_tables(store.get(cad_index)["cad_vec"])
        pos_rows.append(flat)
        grp_rows.append(group)
        count_rows.append(counts)
    width = int(max((len(p) for p in pos_rows), default=1))
    pos = torch.zeros((n_cads, width), dtype=torch.int64)
    grp = torch.full((n_cads, width), -1, dtype=torch.int8)
    for cad_index, (p, g) in enumerate(zip(pos_rows, grp_rows)):
        pos[cad_index, : len(p)] = p
        grp[cad_index, : len(g)] = g
    counts = torch.stack(count_rows)
    return {
        "pos": pos,
        "grp": grp,
        "counts": counts,
        "counts_numpy": counts.numpy().astype(np.int64, copy=False),
        "width": width,
    }


def counts_sha256(counts_row_major_int64) -> str:
    """Plan: counts SHA is over int64 little-endian C-order data bytes."""
    import numpy as np

    arr = np.ascontiguousarray(counts_row_major_int64, dtype=np.dtype("<i8"))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def grouped_batch_scores(model, store, text: str, cad_indices, device, layout: dict):
    """One teacher-forced forward per pair; returns per-pair (sum, groups).

    Column 0 of the returned sums is the audited historical SUM computed by
    ``two_head_factorized_score`` (same FP32 log_softmax kernel, float64
    accumulation); columns 1..6 are the six fixed field-group sums reduced in
    float64 from the SAME logits. Per plan item 2 the group sums must add back
    to SUM within 1e-8 and the group counts must equal the candidate's pinned
    counts row exactly; any violation is a hard refusal.
    """
    import torch  # noqa: F401  (guarded import for clarity)

    from idea01.text2cad_frozen_runner import assemble_batch
    from idea01.text2cad_port.model import teacher_forced_logits
    from idea01.text2cad_vector_scorer import two_head_factorized_score

    vec_dict, mask_cad_dict, cad_seq_len = assemble_batch(
        store, list(cad_indices), device, trim_mode="none")
    with torch.inference_mode():
        pred, _target, _shifted = teacher_forced_logits(
            model, [text] * len(cad_indices), vec_dict, mask_cad_dict,
            cad_seq_len=cad_seq_len)
    logits_cpu = pred.to(torch.float32).cpu()
    cad_vec_cpu = vec_dict["cad_vec"].cpu()
    kpm_cpu = mask_cad_dict["key_padding_mask"].cpu()

    historical = two_head_factorized_score(logits_cpu, cad_vec_cpu, kpm_cpu)
    assert isinstance(historical, list)
    total = [float(r["score_total"]) for r in historical]

    # (B, N-1, 2) float64 per-factor log-probabilities from the SAME logits
    target = cad_vec_cpu[:, 1:].to(torch.long)
    log_probs = torch.nn.functional.log_softmax(logits_cpu, dim=-1)
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).to(torch.float64)
    if not bool(torch.isfinite(gathered).all()):
        raise NightError("non-finite label log-probabilities")
    flat = gathered.reshape(gathered.shape[0], -1)              # (B, (N-1)*2)

    idx = torch.as_tensor(list(cad_indices), dtype=torch.int64)
    pos = layout["pos"][idx]                                    # (B, M)
    grp = layout["grp"][idx]                                    # (B, M)
    values = flat.gather(1, pos) * (grp >= 0)                   # zero on padding lanes
    sums = torch.empty((len(cad_indices), N_GROUPS), dtype=torch.float64)
    for g in range(N_GROUPS):
        sums[:, g] = (values * (grp == g)).sum(dim=1)
    group_counts = layout["counts"][idx]

    sums_numpy = sums.numpy()
    for b, cad_index in enumerate(cad_indices):
        group_total = float(sums_numpy[b].sum())
        delta = abs(group_total - total[b])
        if delta > GROUP_TOTAL_TOL:
            raise NightError(
                f"cad_index {cad_index}: group sums add to {group_total!r} but the "
                f"historical SUM is {total[b]!r} (|delta|={delta:.3e} > {GROUP_TOTAL_TOL})"
            )
        if int(group_counts[b].sum()) != historical[b]["n_valid_total"]:
            raise NightError(f"cad_index {cad_index}: group counts differ from native score")
    return total, sums_numpy


def publish_chunk(chunks_dir: Path, text_index: int, scores, text: str,
                  inputs_sha: str, counts_sha: str) -> str:
    """Atomic process-local publish of chunks/t{index:05d}.npz (fail-if-exists)."""
    import numpy as np

    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_dir / f"t{text_index:05d}.npz"
    if path.exists():
        raise NightError(f"chunk already exists, refusing overwrite: {path}")
    fd, tmp_name = tempfile_mkstemp(chunks_dir, path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(
                stream,
                scores=scores,
                text_index=np.int64(text_index),
                text_sha256=np.str_(sha256_text(text)),
                inputs_sha256=np.str_(inputs_sha),
                counts_sha256=np.str_(counts_sha),
            )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise NightError(f"chunk appeared during publish: {path}") from exc
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return sha256_file(path)


def tempfile_mkstemp(directory: Path, name: str):
    import tempfile

    return tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=str(directory))




# ---------------------------------------------------------------------------
# score: shard orchestration
# ---------------------------------------------------------------------------


def validate_text_stop(axis: str, text_stop: int, n_texts: int) -> int:
    if text_stop is None or text_stop <= 0:
        raise NightError(f"--text-stop must be positive, got {text_stop}")
    if text_stop > n_texts:
        raise NightError(f"--text-stop {text_stop} exceeds the inputs.json axis ({n_texts} texts)")
    if axis == "validation":
        want = N_BACKGROUND + N_FIT + N_SELECT
        if text_stop != want:
            raise NightError(
                f"validation axis requires the full preregistered text axis "
                f"--text-stop {want}, got {text_stop}"
            )
    else:
        if text_stop <= N_BACKGROUND or (text_stop - N_BACKGROUND) not in TEST_PREFIXES:
            raise NightError(
                f"test axis --text-stop - {N_BACKGROUND} must be one of {TEST_PREFIXES}, "
                f"got {text_stop - N_BACKGROUND}"
            )
    return text_stop


def git(repo_root: Path, *argv: str) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), *argv], capture_output=True)
    if result.returncode != 0:
        raise NightError(
            f"git {' '.join(argv)} failed ({result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout.decode("utf-8").strip()


def require_clean_detached(repo_root: Path) -> dict:
    """Same checkout gate the stage controller applies (tracked-tree dirtiness
    and a named branch both refuse; untracked scratch files do not)."""
    dirty = git(repo_root, "status", "--porcelain")
    if dirty:
        raise NightError(f"git worktree not clean:\n{dirty}")
    result = subprocess.run(["git", "-C", str(repo_root), "symbolic-ref", "-q", "HEAD"],
                            capture_output=True)
    if result.returncode == 0:
        branch = result.stdout.decode("utf-8").strip()
        raise NightError(f"HEAD is not detached (on {branch!r}); plan requires a detached SHA")
    return {"head": git(repo_root, "rev-parse", "HEAD")}


def ensure_outside_checkout(path: Path, repo_root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    root = Path(repo_root).resolve()
    if resolved == root or root in resolved.parents:
        raise NightError(f"{label} must be outside the Git checkout: {resolved}")
    return resolved


def load_axis_inputs(inputs_path: Path, expected_sha: str) -> dict:
    got = sha256_file(inputs_path)
    if got != expected_sha:
        raise NightError(f"inputs.json sha256 {got} != expected {expected_sha}")
    payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_INPUTS:
        raise NightError(f"inputs.json schema {payload.get('schema')!r} != {SCHEMA_INPUTS!r}")
    if payload.get("axis") not in ("validation", "test"):
        raise NightError(f"inputs.json axis {payload.get('axis')!r} unknown")
    texts = payload.get("texts")
    members = payload.get("members")
    if not isinstance(texts, list) or not isinstance(members, list):
        raise NightError("inputs.json texts/members must be string lists")
    if not all(isinstance(t, str) and t for t in texts):
        raise NightError("inputs.json texts must be non-empty strings")
    if not all(isinstance(m, str) and m for m in members):
        raise NightError("inputs.json members must be non-empty strings")
    if len(set(members)) != len(members):
        raise NightError("inputs.json members contain duplicates")
    return payload


def publish_counts(out_dir: Path, shard: int, counts, counts_sha: str) -> None:
    import numpy as np

    receipt = {
        "schema": SCHEMA_COUNTS_RECEIPT,
        "shard": shard,
        "n_cads": int(counts.shape[0]),
        "n_groups": N_GROUPS,
        "counts_sha256": counts_sha,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_canonical_json(out_dir / f"counts_receipt_shard{shard}.json", receipt)
    if shard == 0:
        path = out_dir / "counts.npy"
        if path.exists():
            raise NightError(f"counts.npy already exists, refusing overwrite: {path}")
        fd, tmp_name = tempfile_mkstemp(out_dir, "counts.npy")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, np.ascontiguousarray(counts, dtype=np.dtype("<i8")))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def wait_and_verify_counts(out_dir: Path, counts_sha: str, shard: int,
                           timeout_seconds: float) -> dict:
    """A non-zero shard blocks (bounded) for shard 0's counts.npy and verifies
    the published bytes equal its own computation; a timeout is a hard refusal
    (shard 0 publishes counts immediately after preload, before any forward)."""
    path = out_dir / "counts.npy"
    deadline = time.time() + timeout_seconds
    while not path.exists():
        if time.time() >= deadline:
            raise NightError(
                f"counts.npy did not appear within {timeout_seconds:.0f}s; shard 0 may have failed"
            )
        time.sleep(1.0)
    import numpy as np

    published = np.load(path, allow_pickle=False)
    if published.dtype != np.dtype("<i8") or published.ndim != 2 or published.shape[1] != N_GROUPS:
        raise NightError(
            f"counts.npy must be int64 (n_cads, {N_GROUPS}), got {published.dtype} {published.shape}"
        )
    published_sha = counts_sha256(published)
    if published_sha != counts_sha:
        raise NightError(
            f"published counts.npy sha {published_sha} != this shard's own computation {counts_sha}"
        )
    return {"counts_sha256": published_sha, "n_cads": int(published.shape[0])}


def score_shard_texts(model, store, layout, texts, out_dir, shard: int, shards: int,
                      text_stop: int, inputs_sha: str, counts_sha: str) -> dict:
    """Score this shard's texts (index modulo shards); publish one chunk per text."""
    import numpy as np

    n_cads = layout["counts"].shape[0]
    device = next(model.parameters()).device
    rows_written = 0
    started = time.time()
    per_text_seconds = []
    for text_index in range(text_stop):
        if text_index % shards != shard:
            continue
        t0 = time.time()
        scores = np.empty((n_cads, 1 + N_GROUPS), dtype=np.float64)
        for start in range(0, n_cads, BATCH_SIZE):
            batch = range(start, min(start + BATCH_SIZE, n_cads))
            total, group_sums = grouped_batch_scores(
                model, store, texts[text_index], batch, device, layout)
            scores[start: start + len(total), 0] = total
            scores[start: start + len(total), 1:] = group_sums
        drift = float(np.abs(scores[:, 1:].sum(axis=1) - scores[:, 0]).max())
        if drift > GROUP_TOTAL_TOL:
            raise NightError(
                f"text {text_index}: |sum(groups) - SUM| = {drift:.3e} > {GROUP_TOTAL_TOL}"
            )
        chunk_sha = publish_chunk(out_dir / "chunks", text_index, scores, texts[text_index],
                                  inputs_sha, counts_sha)
        rows_written += 1
        per_text_seconds.append(round(time.time() - t0, 3))
        print(json.dumps({
            "phase": "text_scored", "shard": shard, "text_index": text_index,
            "chunk_sha256": chunk_sha, "seconds": per_text_seconds[-1],
            "elapsed_seconds": round(time.time() - started, 1),
        }), flush=True)
    return {
        "n_texts_scored": rows_written,
        "per_text_seconds": per_text_seconds,
        "scoring_seconds": round(time.time() - started, 3),
    }


def initialize_npu(device):
    import torch
    import torch_npu
    import transformers

    torch.set_num_threads(1)
    torch.npu.set_device(device)
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "transformers": str(transformers.__version__),
        "device": device,
    }


def run_score(args) -> int:
    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    inputs_payload = load_axis_inputs(Path(args.inputs), args.expected_inputs_sha)
    axis = inputs_payload["axis"]
    texts = inputs_payload["texts"]
    members = inputs_payload["members"]
    text_stop = validate_text_stop(axis, args.text_stop, len(texts))
    shard, shards = args.shard, args.shards
    if not (0 <= shard < shards):
        raise NightError(f"--shard must be in [0, --shards), got {shard}/{shards}")
    if not re.fullmatch(r"npu:\d+", args.device):
        raise NightError(f"--device must be npu:<index>, got {args.device!r}")
    out_dir = ensure_outside_checkout(Path(args.out), repo_root, "--out")
    out_dir.mkdir(parents=True, exist_ok=True)
    git_state = require_clean_detached(repo_root)

    # torch-bearing imports stay on the score path only
    from idea01.text2cad_frozen_runner import (
        PINNED_CHECKPOINT_SHA256,
        RunnerConfig,
        check_revision_paths,
        load_gated_model_and_store,
        new_invocation_id,
    )

    keys_manifest = Path(args.keys_manifest)
    if not keys_manifest.is_file():
        raise NightError(f"--keys-manifest is not a file: {keys_manifest}")
    cfg = RunnerConfig(
        checkpoint_path=str(args.checkpoint),
        bert_dir=str(args.bert_dir),
        cad_seq_zip=str(args.cad_zip),
        pair_manifest_dir="(inputs.json axis; no pair manifest)",
        device=args.device,
        torch_dtype="float32",
        batch_size=BATCH_SIZE,
        expected_keys_manifest=str(keys_manifest),
        trim_mode="none",
        text_cache=False,
        shard_id=shard,
        n_shards=shards,
        texts_per_chunk=1,
    )
    check_revision_paths(cfg)
    environment = initialize_npu(args.device)
    model, load_report, store, preload_report = load_gated_model_and_store(cfg, members=members)

    layout = build_gallery_group_layout(store, len(members))
    counts_sha = counts_sha256(layout["counts_numpy"])
    publish_counts(out_dir, shard, layout["counts_numpy"], counts_sha)
    counts_gate = (
        {"published_by": "shard 0 (this shard)"}
        if shard == 0
        else wait_and_verify_counts(out_dir, counts_sha, shard, args.counts_wait_seconds)
    )

    scoring = score_shard_texts(model, store, layout, texts, out_dir, shard, shards,
                                text_stop, args.expected_inputs_sha, counts_sha)
    summary = {
        "schema": SCHEMA_SHARD_SUMMARY,
        "invocation_id": new_invocation_id(),
        "axis": axis,
        "shard": shard,
        "shards": shards,
        "device": args.device,
        "git": git_state,
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "inputs_sha256": args.expected_inputs_sha,
        "counts_sha256": counts_sha,
        "counts_gate": counts_gate,
        "text_stop": text_stop,
        "planned_pairs": sum(
            len(members) for t in range(text_stop) if t % shards == shard),
        "n_cads": len(members),
        "batch_size": BATCH_SIZE,
        "torch_dtype": "float32",
        "trim_mode": "none",
        "text_cache": False,
        "environment": environment,
        "strict_load": load_report["strict_load"],
        "model_load_report_sha256": hashlib.sha256(
            json.dumps(load_report, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "preload_report": preload_report,
        "scoring": scoring,
        "elapsed_seconds": round(time.time() - started, 3),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_canonical_json(out_dir / f"shard_summary_s{shard}.json", summary)
    print(json.dumps({"status": "shard_complete", "shard": shard,
                      "texts_scored": scoring["n_texts_scored"],
                      "planned_pairs": summary["planned_pairs"]}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# replay: 60 certified-pair NPU gate
# ---------------------------------------------------------------------------


def load_e2_expected_config(repo_root: Path) -> dict:
    from idea01.text2cad_e2_runner import E2Error, load_e2_config
    from idea01.text2cad_pair_index import (
        CAD_LENGTH_FILE,
        CAD_STORE_FILE,
        PAIR_MANIFEST_FILE,
        TEXTS_FILE,
    )

    try:
        cfg = load_e2_config(repo_root)
    except E2Error as exc:
        raise NightError(f"committed E2 expected-config refused: {exc}") from exc
    cfg["_replay_files"] = {
        TEXTS_FILE: cfg["backgrounds"]["replay"]["texts_sha256"],
        CAD_STORE_FILE: cfg["backgrounds"]["replay"]["cad_store_sha256"],
        CAD_LENGTH_FILE: cfg["backgrounds"]["replay"]["cad_length_index_sha256"],
        PAIR_MANIFEST_FILE: cfg["backgrounds"]["replay"]["pair_manifest_sha256"],
    }
    return cfg


def verify_replay_gate_record(path: Path) -> dict:
    got = sha256_file(path)
    if got != PINNED_E2_GATE_RECORD_SHA256:
        raise NightError(f"E2 replay_gate_record.json sha256 {got} != pinned "
                         f"{PINNED_E2_GATE_RECORD_SHA256}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "text2cad-e2-replay-gate-v1":
        raise NightError(f"E2 gate record schema {record.get('schema')!r} unexpected")
    if record.get("pass") is not True:
        raise NightError("E2 gate record says pass != true; the certified replay never passed")
    approved = record.get("approved_sha")
    if not isinstance(approved, str) or not re.fullmatch(r"[0-9a-f]{40}", approved):
        raise NightError(f"E2 gate record approved_sha malformed: {approved!r}")
    dense = record.get("dense_to_original_cad_index")
    if not isinstance(dense, list) or not all(isinstance(v, int) for v in dense):
        raise NightError("E2 gate record dense_to_original_cad_index missing/malformed")
    pins = record.get("replay_chunk_sha256")
    if not isinstance(pins, dict) or not pins:
        raise NightError("E2 gate record replay_chunk_sha256 missing")
    return record


def verify_replay_manifest(manifest_dir: Path, cfg: dict) -> tuple[list[str], list[str]]:
    from idea01.text2cad_pair_index import load_cad_store, load_pair_manifest, load_texts

    manifest = load_pair_manifest(manifest_dir, expected_sha256=cfg["_replay_files"])
    counts = manifest["counts"]
    want_t = int(cfg["backgrounds"]["replay"]["n_unique_text"])
    want_c = int(cfg["backgrounds"]["replay"]["n_gallery"])
    if counts.get("n_unique_text") != want_t or counts.get("n_gallery") != want_c:
        raise NightError(
            f"replay manifest counts {counts} != pinned {want_t}x{want_c}"
        )
    texts = load_texts(manifest_dir)
    members = load_cad_store(manifest_dir)
    if len(texts) != want_t or len(members) != want_c:
        raise NightError(
            f"replay manifest holds {len(texts)} texts / {len(members)} members, "
            f"expected {want_t}/{want_c}"
        )
    return texts, members


def parse_replay_reference(path: Path, n_cads: int) -> dict[int, dict]:
    """Parse one certified E2 replay chunk into {dense_cad_index: record}."""
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0].get("record_type") != "chunk_header":
        raise NightError(f"{path.name}: first record is not a chunk_header")
    body = rows[1:]
    reference: dict[int, dict] = {}
    for row in body:
        dense = int(row["cad_index"])
        if dense in reference:
            raise NightError(f"{path.name}: duplicate candidate {dense}")
        if row.get("status") != "scored":
            raise NightError(f"{path.name}: candidate {dense} status {row.get('status')!r}")
        if not 0 <= dense < n_cads:
            raise NightError(f"{path.name}: dense cad_index {dense} outside the replay store")
        reference[dense] = {
            "score_total": float(row["score_total"]),
            "n_valid_total": int(row["n_valid_total"]),
            "n_valid_x": int(row["n_valid_x"]),
            "n_valid_y": int(row["n_valid_y"]),
        }
    if len(reference) != n_cads:
        raise NightError(f"{path.name}: {len(reference)} reference rows, expected {n_cads}")
    return reference


def compare_replay_pairs(records: list[dict]) -> dict:
    """Pure comparison gate over already-collected per-pair records (unit-tested)."""
    violations = []
    max_delta = 0.0
    for rec in records:
        tol = max(REPLAY_TOLERANCE_ABS, REPLAY_TOLERANCE_REL * abs(rec["reference_sum"]))
        delta = abs(rec["sum"] - rec["reference_sum"])
        max_delta = max(max_delta, delta)
        reasons = []
        if not all(math.isfinite(rec[name]) for name in ("sum", "reference_sum", "group_total")):
            reasons.append("nonfinite")
        if delta > tol:
            reasons.append("sum_tolerance")
        if rec["n_valid_total"] != rec["reference_n_valid_total"]:
            reasons.append("n_valid_total")
        if abs(rec["group_total"] - rec["sum"]) > GROUP_TOTAL_TOL:
            reasons.append("group_total")
        if len(rec["group_counts"]) != N_GROUPS or sum(rec["group_counts"]) != rec["n_valid_total"]:
            reasons.append("group_counts")
        if reasons:
            violations.append({"text_index": rec["text_index"],
                               "dense_cad_index": rec["dense_cad_index"],
                               "cad_index": rec["cad_index"],
                               "delta": delta, "tolerance": tol,
                               "reasons": reasons})
    return {"n_pairs": len(records), "n_violations": len(violations),
            "max_abs_delta": max_delta, "violations": violations[:50]}


def run_replay(args) -> int:
    import torch

    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    if not re.fullmatch(r"npu:\d+", args.device):
        raise NightError(f"--device must be npu:<index>, got {args.device!r}")
    out_dir = ensure_outside_checkout(Path(args.out), repo_root, "--out")
    if out_dir.exists():
        raise NightError(f"replay output directory already exists: {out_dir}")
    git_state = require_clean_detached(repo_root)

    cfg = load_e2_expected_config(repo_root)
    gate_path = Path(args.e2_run_root) / "replay_gate_record.json"
    if not gate_path.is_file():
        raise NightError(f"E2 replay gate record not found: {gate_path}")
    gate = verify_replay_gate_record(gate_path)
    texts, members = verify_replay_manifest(Path(args.replay_manifest_dir), cfg)
    n_cads = len(members)
    dense_to_original = [int(v) for v in gate["dense_to_original_cad_index"]]
    if len(dense_to_original) != n_cads:
        raise NightError(
            f"gate record mapping has {len(dense_to_original)} entries, replay store has {n_cads}"
        )

    selected = [int(v) for v in str(args.replay_texts).split(",")]
    if selected != [0, 1]:
        raise NightError("only the two preregistered replay texts 0 and 1 are enabled")
    for k in selected:
        if not 0 <= k < len(texts):
            raise NightError(f"--replay-texts index {k} outside the replay manifest")
        if not texts[k].strip():
            raise NightError(f"--replay-texts index {k} is empty; the plan requires non-empty")

    tail_first = n_cads - (REPLAY_TAIL_ORIGINAL_LAST - REPLAY_TAIL_ORIGINAL_FIRST + 1)
    tail_dense = list(range(tail_first, n_cads))
    if [dense_to_original[d] for d in tail_dense] != list(
            range(REPLAY_TAIL_ORIGINAL_FIRST, REPLAY_TAIL_ORIGINAL_LAST + 1)):
        raise NightError(
            "the replay store's dense tail does not map to the original G4 tail batch "
            f"{REPLAY_TAIL_ORIGINAL_FIRST}..{REPLAY_TAIL_ORIGINAL_LAST}"
        )
    full_dense = list(range(BATCH_SIZE))
    if dense_to_original[:BATCH_SIZE] != full_dense:
        raise NightError("replay prefix differs from the original first G4 batch")

    chunk_pins = {}
    references = {}
    for k in selected:
        chunk_name = f"chunk_s00_t{k:05d}_{k + 1:05d}.jsonl"
        chunk_path = Path(args.e2_run_root) / "replay" / chunk_name
        chunk_sha = sha256_file(chunk_path)
        if chunk_sha != gate["replay_chunk_sha256"].get(chunk_name):
            raise NightError(f"{chunk_name} failed the certified E2 chunk SHA check")
        chunk_pins[chunk_name] = chunk_sha
        references[k] = parse_replay_reference(chunk_path, n_cads)

    from idea01.text2cad_frozen_runner import (
        PINNED_CHECKPOINT_SHA256,
        RunnerConfig,
        check_revision_paths,
        load_gated_model_and_store,
        new_invocation_id,
    )

    keys_manifest = Path(args.keys_manifest)
    if not keys_manifest.is_file():
        raise NightError(f"--keys-manifest is not a file: {keys_manifest}")
    cfg_run = RunnerConfig(
        checkpoint_path=str(args.checkpoint),
        bert_dir=str(args.bert_dir),
        cad_seq_zip=str(args.cad_zip),
        pair_manifest_dir=str(args.replay_manifest_dir),
        device=args.device,
        torch_dtype="float32",
        batch_size=BATCH_SIZE,
        expected_keys_manifest=str(keys_manifest),
        trim_mode="none",
        text_cache=False,
        shard_id=0,
        n_shards=1,
        texts_per_chunk=1,
    )
    check_revision_paths(cfg_run)
    environment = initialize_npu(args.device)
    model, load_report, store, preload_report = load_gated_model_and_store(cfg_run, members=members)
    layout = build_gallery_group_layout(store, n_cads)
    counts_sha = counts_sha256(layout["counts_numpy"])

    device = torch.device(args.device)
    records = []
    for k in selected:
        reference = references[k]
        for dense_block in (full_dense, tail_dense):
            total, group_sums = grouped_batch_scores(
                model, store, texts[k], dense_block, device, layout)
            for b, dense in enumerate(dense_block):
                ref = reference[dense]
                records.append({
                    "text_index": k,
                    "dense_cad_index": dense,
                    "cad_index": dense_to_original[dense],
                    "reference_sum": ref["score_total"],
                    "sum": float(total[b]),
                    "reference_n_valid_total": ref["n_valid_total"],
                    "n_valid_total": int(layout["counts_numpy"][dense].sum()),
                    "group_total": float(group_sums[b].sum()),
                    "group_counts": [int(v) for v in layout["counts_numpy"][dense]],
                    "group_sums": [float(v) for v in group_sums[b]],
                })

    comparison = compare_replay_pairs(records)
    passed = comparison["n_pairs"] == len(selected) * (BATCH_SIZE + len(tail_dense)) \
        and comparison["n_violations"] == 0
    result = {
        "schema": SCHEMA_REPLAY_RESULT,
        "passed": bool(passed),
        "e2_gate_record_sha256": PINNED_E2_GATE_RECORD_SHA256,
        "e2_gate_approved_sha": gate["approved_sha"],
        "e2_run_root": str(args.e2_run_root),
        "replay_manifest_dir": str(args.replay_manifest_dir),
        "git": git_state,
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "cad_seq_zip_sha256": sha256_file(args.cad_zip),
        "replay_texts_selected": selected,
        "replay_text_nonempty": [bool(texts[k].strip()) for k in selected],
        "pairs_planned": len(selected) * (BATCH_SIZE + len(tail_dense)),
        "counts_sha256": counts_sha,
        "replay_chunk_sha256": chunk_pins,
        "comparison": comparison,
        "model_load_report_sha256": hashlib.sha256(
            json.dumps(load_report, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "preload_report": preload_report,
        "environment": environment,
        "strict_load": load_report["strict_load"],
        "records": records,
        "device": args.device,
        "elapsed_seconds": round(time.time() - started, 3),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True)
    result_path = out_dir / "replay_result.json"
    result_sha = write_canonical_json(result_path, result)
    lines = [f"{result_sha}  replay_result.json"]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "replay_complete", "passed": passed,
                      "n_pairs": comparison["n_pairs"],
                      "max_abs_delta": comparison["max_abs_delta"],
                      "out": str(out_dir)}), flush=True)
    if not passed:
        raise NightError(
            f"replay gate FAILED: {comparison['n_violations']} violation(s), "
            f"max |delta| = {comparison['max_abs_delta']:.3e}"
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p_prepare = sub.add_parser("prepare", help="build validation/test inputs+eval artifacts (CPU only)")
    p_prepare.add_argument("--text2cad-csv", required=True)
    p_prepare.add_argument("--split-json", required=True)
    p_prepare.add_argument("--cad-zip", required=True)
    p_prepare.add_argument("--g5-dir", required=True,
                           help="directory with the certified G5 query/gallery/qrels/cad_store.jsonl")
    p_prepare.add_argument("--out", required=True)
    p_prepare.set_defaults(func=lambda a: run_prepare(a))

    p_replay = sub.add_parser("replay", help="60-pair certified E2 replay gate on one NPU")
    p_replay.add_argument("--e2-run-root", required=True,
                          help="E2 run root holding replay_gate_record.json and replay/chunk_*.jsonl")
    p_replay.add_argument("--replay-manifest-dir", required=True)
    p_replay.add_argument("--checkpoint", required=True)
    p_replay.add_argument("--bert-dir", required=True)
    p_replay.add_argument("--keys-manifest", required=True)
    p_replay.add_argument("--cad-zip", required=True)
    p_replay.add_argument("--device", default="npu:0")
    p_replay.add_argument("--out", required=True)
    p_replay.add_argument("--replay-texts", default="0,1",
                          help="comma-separated replay text indices (default 0,1; 2x(16+14)=60 pairs)")
    p_replay.set_defaults(func=lambda a: run_replay(a))

    p_score = sub.add_parser("score", help="one NPU shard of one axis (argv fixed by the stage controller)")
    p_score.add_argument("--inputs", required=True)
    p_score.add_argument("--expected-inputs-sha", required=True)
    p_score.add_argument("--checkpoint", required=True)
    p_score.add_argument("--bert-dir", required=True)
    p_score.add_argument("--keys-manifest", required=True)
    p_score.add_argument("--cad-zip", required=True)
    p_score.add_argument("--device", default="npu:0")
    p_score.add_argument("--out", required=True)
    p_score.add_argument("--shard", type=int, default=0)
    p_score.add_argument("--shards", type=int, default=1)
    p_score.add_argument("--text-stop", type=int, required=True)
    p_score.add_argument("--counts-wait-seconds", type=float, default=900.0)
    p_score.set_defaults(func=lambda a: run_score(a))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NightError as exc:
        print(f"ACCURACY-NIGHT FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001 - every refusal exits non-zero with the reason
        print(f"ACCURACY-NIGHT UNEXPECTED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
