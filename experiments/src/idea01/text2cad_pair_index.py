"""Text2CAD v1.0 outer pair/index builder (M1c stage B2) -- UID-aware outer layer.

Task contract: experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md
(G0.5) and experiments/reviews/IDEA01_M1C_STAGE_B1_CODEX_REVIEW_2026-09-30.md
(B2 condition 1). This module builds the FIXED pair space for the full
retrieval matrix from the frozen M0 manifests:

  * unique text index: the 3,644 unique verbatim ``strip()``-normalized L0
    query texts of the 7,934 non-empty test queries, in a canonical content
    order -- sorted by (text_sha256, text) so the index is independent of the
    (pseudorandom) query order and reproducible from the text set alone;
  * CAD store index: the 8,046 official test candidates in gallery order
    (verified to be ascending ``cad_id``), each mapped to its
    ``cad_seq/<pref>/<tail>/seq/<tail>.pth`` member inside the pinned
    ``cad_seq.zip``;
  * implicit exact Cartesian product: ``pair_index = text_index * N_CADS +
    cad_index`` (text-major), so ``pair_index`` ranges over
    ``[0, 3644*8046) = [0, 29_319_624)`` without ever materializing the pair
    list. Sharding splits the TEXT range contiguously (shard s of S gets text
    indices ``[floor(s*3644/S), floor((s+1)*3644/S)`` via round), hence a
    disjoint contiguous pair range per shard; chunking groups
    ``texts_per_chunk`` consecutive texts per output file so a resume can
    verify completed chunks (exact pair sequence + binding fields) and skip
    them -- every pair is scored exactly once.

Isolation contract (CLAUDE.md sec.5.1, subplan G0.5):
  * this OUTER module reads ``query.jsonl`` / ``gallery.jsonl`` and may keep
    ``cad_id`` / ``query_id`` linkage in ``evaluation_map.json`` -- it NEVER
    reads ``qrels.jsonl``, the raw CSV, audit files or anything carrying
    relevance information;
  * the WORKER-facing artifacts are ``texts.jsonl`` (text_index + text +
    integrity fields) and ``cad_store.jsonl`` (cad_index + zip member path);
    they contain no UID, no query_id, no caption metadata and no relevance
    signal. The zip member path embeds the official storage locator (which
    happens to contain the CAD UID string); it is a storage key only, carries
    no query<->CAD relation, and the worker treats it as opaque;
  * ``evaluation_map.json`` (query_id -> text_index, cad_index -> cad_id) is
    written to a SEPARATE file marked evaluation-only; the scoring runner has
    no parameter that opens it, and mapping scores back to UIDs belongs to the
    independent evaluation stage that also owns qrels.

Expectations from the frozen M0 audit (7934 queries / 3644 unique texts /
8046 gallery / 29,319,624 pairs) are TESTED, never trusted: any mismatch is a
hard failure of the build.

This module is torch-free at import (the optional per-candidate length scan
imports torch lazily) and performs no model forward anywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

# ---------------------------------------------------------------------------
# Fixed expectations (frozen M0 audit; TESTED on every build, never trusted)
# ---------------------------------------------------------------------------

EXPECTED_N_QUERY = 7934
EXPECTED_N_UNIQUE_TEXT = 3644
EXPECTED_N_GALLERY = 8046
EXPECTED_N_PAIRS = EXPECTED_N_UNIQUE_TEXT * EXPECTED_N_GALLERY  # 29,319,624

SCHEMA_VERSION = "text2cad-pair-manifest-v1"
PAIR_FORMULA = "pair_index = text_index * n_cads + cad_index (text-major)"

#: Exact key sets of the frozen M0 records (m0_manifest.py, schema m0-v1).
QUERY_RECORD_KEYS = frozenset({"level", "n_chars", "query_id", "text", "text_sha256"})
GALLERY_RECORD_KEYS = frozenset(
    {"cad_id", "cad_seq_member", "cadmium_native_present", "minimal_json_member", "n_representations"}
)

#: Worker-facing artifact schemas (isolation: no UID, no query_id).
TEXTS_RECORD_KEYS = frozenset({"text_index", "text", "text_sha256", "n_chars"})
CAD_STORE_RECORD_KEYS = frozenset({"cad_index", "member"})
CAD_LENGTH_RECORD_KEYS = frozenset({"cad_index", "n_real_tokens", "n_valid_x", "n_valid_y"})

_QUERY_ID_RE = re.compile(r"^Q\d{5}$")
_CAD_ID_RE = re.compile(r"^\d{4}/\d{8}$")
_MEMBER_RE = re.compile(r"^cad_seq/(\d{4})/(\d{8})/seq/\2\.pth$")

#: SHA256 of the pinned official cad_seq.zip (configs/text2cad_original_asset_sha256_20260930.json;
#: duplicated here so the builder stays torch-free at import -- a unit test
#: cross-checks the value against the committed config).
CAD_SEQ_ZIP_SHA256 = "9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa"

#: Artifact file names produced by the builder.
TEXTS_FILE = "texts.jsonl"
CAD_STORE_FILE = "cad_store.jsonl"
CAD_LENGTH_FILE = "cad_length_index.jsonl"
PAIR_MANIFEST_FILE = "pair_manifest.json"
EVALUATION_MAP_FILE = "evaluation_map.json"

# ---------------------------------------------------------------------------
# Rejection reason codes (stable strings; asserted in tests and ledgers)
# ---------------------------------------------------------------------------

REASON_BAD_JSON = "bad_json"
REASON_DUPLICATE_KEY = "duplicate_key"
REASON_SCHEMA_MISMATCH = "schema_mismatch"
REASON_FIELD_MISMATCH = "field_mismatch"
REASON_DUPLICATE_ID = "duplicate_id"
REASON_ORDER_MISMATCH = "order_mismatch"
REASON_COUNT_MISMATCH = "count_mismatch"
REASON_TEXT_DRIFT = "text_drift"
REASON_BACKGROUND_TEXT_MISMATCH = "background_text_mismatch"
REASON_MEMBER_MISMATCH = "member_mismatch"
REASON_ZIP_MEMBER_MISSING = "zip_member_missing"
REASON_ASSET_SHA_MISMATCH = "asset_sha256_mismatch"
REASON_INSIDE_CHECKOUT = "output_inside_checkout"
REASON_OUTPUT_EXISTS = "output_exists"
REASON_BAD_SHARD = "bad_shard"
REASON_BAD_CHUNK = "bad_chunk"
REASON_BAD_PAIR_INDEX = "bad_pair_index"


class PairIndexError(ValueError):
    """Raised when a manifest input or pair-space operation violates the contract.

    ``reason`` is one of the ``REASON_*`` constants so callers can ledger
    rejections without parsing messages.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# Hashing / JSONL helpers
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise PairIndexError(REASON_DUPLICATE_KEY, f"duplicate key {key!r} in JSONL line")
        obj[key] = value
    return obj


def read_jsonl_strict(path, allowed_keys: frozenset, label: str) -> list[dict[str, Any]]:
    """Read a JSONL file requiring EXACTLY ``allowed_keys`` per line.

    Any deviation (bad JSON, duplicate keys, missing/extra keys) is a hard
    failure -- the pair builder never guesses at schema drift.
    """
    records: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line, object_pairs_hook=_no_duplicate_keys)
                except json.JSONDecodeError as exc:
                    raise PairIndexError(
                        REASON_BAD_JSON, f"{label} line {lineno}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise PairIndexError(
                        REASON_BAD_JSON,
                        f"{label} line {lineno}: expected object, got {type(obj).__name__}",
                    )
                keys = set(obj)
                if keys != set(allowed_keys):
                    raise PairIndexError(
                        REASON_SCHEMA_MISMATCH,
                        f"{label} line {lineno}: keys must be exactly "
                        f"{sorted(allowed_keys)}; missing={sorted(set(allowed_keys) - keys)} "
                        f"extra={sorted(keys - set(allowed_keys))}",
                    )
                records.append(obj)
    except OSError as exc:
        raise PairIndexError(REASON_BAD_JSON, f"cannot read {label} {path}: {exc}") from exc
    return records


def write_jsonl(path, records: Iterable[dict[str, Any]]) -> None:
    """Deterministic JSONL writer (sorted keys, \\n newlines, UTF-8)."""
    buf: list[str] = []
    for record in records:
        buf.append(_canonical_json(record))
    Path(path).write_text("\n".join(buf) + ("\n" if buf else ""), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Frozen M0 input validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextEntry:
    """One unique query text in canonical order (worker-facing)."""

    text_index: int
    text: str
    text_sha256: str
    n_chars: int


@dataclass(frozen=True)
class CadEntry:
    """One gallery candidate in gallery order (worker-facing storage key)."""

    cad_index: int
    member: str


def load_query_records(query_path) -> list[dict[str, Any]]:
    """Load and fully validate the frozen M0 ``query.jsonl``.

    Per-record checks: exact schema, ``level == "L0"``, ``query_id`` matches
    ``Q%05d`` and is unique, ``n_chars == len(text)``,
    ``text_sha256 == sha256(text)``, text is a non-empty str already in its
    ``strip()`` form (the protocol's only normalization; a text that differs
    from its strip() would change the unique-text identity and is refused).
    """
    records = read_jsonl_strict(query_path, QUERY_RECORD_KEYS, "query.jsonl")
    seen_ids: set[str] = set()
    for i, rec in enumerate(records):
        qid = rec["query_id"]
        if not isinstance(qid, str) or not _QUERY_ID_RE.match(qid):
            raise PairIndexError(
                REASON_FIELD_MISMATCH, f"query.jsonl line {i + 1}: bad query_id {qid!r}"
            )
        if qid in seen_ids:
            raise PairIndexError(REASON_DUPLICATE_ID, f"duplicate query_id {qid!r}")
        seen_ids.add(qid)
        if rec["level"] != "L0":
            raise PairIndexError(
                REASON_FIELD_MISMATCH, f"{qid}: level must be 'L0', got {rec['level']!r}"
            )
        text = rec["text"]
        if not isinstance(text, str) or not text:
            raise PairIndexError(REASON_TEXT_DRIFT, f"{qid}: text must be a non-empty str")
        if text != text.strip():
            raise PairIndexError(
                REASON_TEXT_DRIFT,
                f"{qid}: text differs from its strip() form; the frozen protocol "
                "stores strip()-normalized text and this builder never re-normalizes",
            )
        if rec["n_chars"] != len(text):
            raise PairIndexError(
                REASON_FIELD_MISMATCH,
                f"{qid}: n_chars {rec['n_chars']!r} != len(text) {len(text)}",
            )
        if rec["text_sha256"] != sha256_text(text):
            raise PairIndexError(
                REASON_FIELD_MISMATCH, f"{qid}: text_sha256 does not match text bytes"
            )
    return records


def load_gallery_records(gallery_path) -> list[dict[str, Any]]:
    """Load and fully validate the frozen M0 ``gallery.jsonl``.

    Per-record checks: exact schema, ``cad_id`` matches ``NNNN/NNNNNNNN`` and
    is unique, records are in ascending ``cad_id`` order (the frozen gallery
    order), and ``cad_seq_member`` is present and equals the canonical member
    path derived from ``cad_id`` (this arm scores official cad_seq vectors;
    a missing member is unscoreable and fails the build loudly).
    """
    records = read_jsonl_strict(gallery_path, GALLERY_RECORD_KEYS, "gallery.jsonl")
    seen_ids: set[str] = set()
    previous: Optional[str] = None
    for i, rec in enumerate(records):
        cad_id = rec["cad_id"]
        if not isinstance(cad_id, str) or not _CAD_ID_RE.match(cad_id):
            raise PairIndexError(
                REASON_FIELD_MISMATCH, f"gallery.jsonl line {i + 1}: bad cad_id {cad_id!r}"
            )
        if cad_id in seen_ids:
            raise PairIndexError(REASON_DUPLICATE_ID, f"duplicate cad_id {cad_id!r}")
        seen_ids.add(cad_id)
        if previous is not None and cad_id <= previous:
            raise PairIndexError(
                REASON_ORDER_MISMATCH,
                f"gallery.jsonl is not in ascending cad_id order at line {i + 1} "
                f"({previous!r} -> {cad_id!r})",
            )
        previous = cad_id
        member = rec["cad_seq_member"]
        expected_member = f"cad_seq/{cad_id}/seq/{cad_id.split('/')[-1]}.pth"
        if not isinstance(member, str) or not _MEMBER_RE.match(member):
            raise PairIndexError(
                REASON_MEMBER_MISMATCH, f"{cad_id}: bad cad_seq_member {member!r}"
            )
        if member != expected_member:
            raise PairIndexError(
                REASON_MEMBER_MISMATCH,
                f"{cad_id}: cad_seq_member {member!r} != canonical {expected_member!r}",
            )
    return records


def build_text_index(query_records: Sequence[dict[str, Any]]) -> tuple[list[TextEntry], dict[str, int]]:
    """Canonical unique-text index from validated query records.

    Deduplication is by EXACT string equality (texts are already strip()
    normalized by the frozen protocol; this function never re-normalizes).
    The canonical order is ``sorted by (text_sha256, text)`` -- content
    addressed, independent of the (pseudorandom) query order, and
    reproducible from the text set alone.

    Returns ``(entries, query_to_text_index)``; the latter is evaluation-only
    linkage (query_id -> text_index) and is written to a separate file.
    """
    unique: dict[str, str] = {}  # text -> sha (identity check)
    for rec in query_records:
        sha = unique.setdefault(rec["text"], rec["text_sha256"])
        if sha != rec["text_sha256"]:  # defensive; load_query_records already checks
            raise PairIndexError(REASON_TEXT_DRIFT, "sha256 collision or drift on text")
    ordered = sorted(unique, key=lambda t: (unique[t], t))
    entries = [
        TextEntry(
            text_index=i,
            text=text,
            text_sha256=unique[text],
            n_chars=len(text),
        )
        for i, text in enumerate(ordered)
    ]
    text_to_index = {entry.text: entry.text_index for entry in entries}
    query_to_text = {rec["query_id"]: text_to_index[rec["text"]] for rec in query_records}
    return entries, query_to_text


def build_cad_store(gallery_records: Sequence[dict[str, Any]]) -> list[CadEntry]:
    """CAD store index in verified ascending-cad_id gallery order."""
    return [
        CadEntry(cad_index=i, member=rec["cad_seq_member"])
        for i, rec in enumerate(gallery_records)
    ]


def verify_zip_members(zip_path, members: Sequence[str], expected_sha256: str = CAD_SEQ_ZIP_SHA256) -> dict:
    """Cross-check the CAD store against the pinned ``cad_seq.zip``.

    Verifies the zip file SHA256, that every store member exists exactly once,
    and records how many extra ``.pth`` members the archive carries (the
    train/validation splits share the zip -- extras are expected, not an
    error). Reads the central directory only; no tensor bytes are loaded.
    """
    zip_path = Path(zip_path)
    actual = sha256_file(zip_path)
    if actual != expected_sha256:
        raise PairIndexError(
            REASON_ASSET_SHA_MISMATCH,
            f"cad_seq.zip sha256 {actual} != pinned {expected_sha256}",
        )
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    name_counts: dict[str, int] = {}
    for name in names:
        name_counts[name] = name_counts.get(name, 0) + 1
    missing = [m for m in members if name_counts.get(m, 0) != 1]
    if missing:
        raise PairIndexError(
            REASON_ZIP_MEMBER_MISSING,
            f"{len(missing)} store member(s) not present exactly once in the zip, "
            f"e.g. {missing[:3]}",
        )
    member_set = set(members)
    n_extra_pth = sum(1 for n in names if n.endswith(".pth") and n not in member_set)
    return {
        "zip_sha256": actual,
        "n_zip_entries": len(names),
        "n_store_members": len(members),
        "n_extra_pth_members": n_extra_pth,
    }


def scan_cad_lengths(zip_path, members: Sequence[str]) -> list[dict[str, int]]:
    """Zero-forward structural scan: per-candidate length/valid-item facts.

    Loads each member with ``torch.load(..., weights_only=True)`` (CPU, no
    model forward), runs the audited stage-A ``validate_cad_vec`` and records
    ``(cad_index, n_real_tokens, n_valid_x, n_valid_y)``. Any candidate that
    fails validation aborts the build with the stable scorer reason -- the
    2026-09-30 full-gallery audit passed 8,046/8,046, so a failure here means
    asset drift. Torch is imported lazily so the builder module stays
    torch-free at import.
    """
    import io  # local: only needed for the scan

    import torch  # lazy: keeps the builder torch-free at import

    from .text2cad_vector_scorer import validate_cad_vec

    out: list[dict[str, int]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for cad_index, member in enumerate(members):
            try:
                sample = torch.load(
                    io.BytesIO(archive.read(member)), map_location="cpu", weights_only=True
                )
                diag = validate_cad_vec(
                    sample["vec"]["cad_vec"],
                    sample["mask_cad_dict"]["key_padding_mask"],
                    sample["vec"]["flag_vec"],
                    sample["vec"]["index_vec"],
                )
            except Exception as exc:
                raise PairIndexError(
                    REASON_MEMBER_MISMATCH,
                    f"candidate scan failed at cad_index {cad_index} ({member}): "
                    f"{getattr(exc, 'reason', type(exc).__name__)}: {exc}",
                ) from exc
            out.append(
                {
                    "cad_index": cad_index,
                    "n_real_tokens": diag.n_real_tokens,
                    "n_valid_x": diag.n_valid_x,
                    "n_valid_y": diag.n_valid_y,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Pair space (implicit Cartesian product) + sharding/chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpace:
    """The implicit exact pair space: text-major Cartesian product.

    ``pair_index = text_index * n_cads + cad_index``; text-major order makes
    consecutive pairs share the same query text, which keeps the (gated,
    off-by-default) BERT text cache effective and makes chunk = a whole text
    row of the matrix.
    """

    n_texts: int
    n_cads: int

    def __post_init__(self) -> None:
        if self.n_texts < 1 or self.n_cads < 1:
            raise PairIndexError(
                REASON_COUNT_MISMATCH, f"PairSpace needs positive dims, got {self}"
            )

    @property
    def n_pairs(self) -> int:
        return self.n_texts * self.n_cads

    def coords_of(self, pair_index: int) -> tuple[int, int]:
        """pair_index -> (text_index, cad_index); range-checked."""
        if not 0 <= pair_index < self.n_pairs:
            raise PairIndexError(
                REASON_BAD_PAIR_INDEX,
                f"pair_index {pair_index} outside [0, {self.n_pairs})",
            )
        return divmod(pair_index, self.n_cads)

    def pair_index_of(self, text_index: int, cad_index: int) -> int:
        if not 0 <= text_index < self.n_texts or not 0 <= cad_index < self.n_cads:
            raise PairIndexError(
                REASON_BAD_PAIR_INDEX,
                f"(text_index {text_index}, cad_index {cad_index}) outside "
                f"{self.n_texts}x{self.n_cads}",
            )
        return text_index * self.n_cads + cad_index

    def shard_text_range(self, n_shards: int, shard_id: int) -> tuple[int, int]:
        """Contiguous, disjoint, complete text ranges across shards.

        Boundary k of S shards is ``round(k * n_texts / n_shards)`` (banker-
        free: Python round on .5 goes to even, but the formula is what the
        committed manifest documents and tests pin; every text is covered by
        exactly one shard for any 1 <= n_shards <= n_texts).
        """
        if not 1 <= n_shards <= self.n_texts or not 0 <= shard_id < n_shards:
            raise PairIndexError(
                REASON_BAD_SHARD,
                f"need 1 <= n_shards <= {self.n_texts} and 0 <= shard_id < n_shards, "
                f"got n_shards={n_shards}, shard_id={shard_id}",
            )
        t0 = round(shard_id * self.n_texts / n_shards)
        t1 = round((shard_id + 1) * self.n_texts / n_shards)
        return t0, t1

    def shard_pair_range(self, n_shards: int, shard_id: int) -> tuple[int, int]:
        t0, t1 = self.shard_text_range(n_shards, shard_id)
        return t0 * self.n_cads, t1 * self.n_cads

    def iter_chunks(
        self, n_shards: int, shard_id: int, texts_per_chunk: int = 1
    ) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(chunk_ordinal, t0, t1, pair_start)`` over the shard.

        Chunks group ``texts_per_chunk`` consecutive texts; pair range of a
        chunk is ``[t0*n_cads, t1*n_cads)``. ``texts_per_chunk`` must be >= 1.
        """
        if texts_per_chunk < 1:
            raise PairIndexError(
                REASON_BAD_CHUNK, f"texts_per_chunk must be >= 1, got {texts_per_chunk}"
            )
        t_lo, t_hi = self.shard_text_range(n_shards, shard_id)
        ordinal = 0
        t = t_lo
        while t < t_hi:
            t1 = min(t + texts_per_chunk, t_hi)
            yield ordinal, t, t1, t * self.n_cads
            ordinal += 1
            t = t1

    def chunk_pair_indices(self, t0: int, t1: int) -> range:
        """The exact pair_index sequence of chunk [t0, t1): ascending, dense."""
        if not 0 <= t0 < t1 <= self.n_texts:
            raise PairIndexError(REASON_BAD_CHUNK, f"bad text range [{t0}, {t1})")
        return range(t0 * self.n_cads, t1 * self.n_cads)


def chunk_file_name(shard_id: int, t0: int, t1: int) -> str:
    return f"chunk_s{shard_id:02d}_t{t0:05d}_{t1:05d}.jsonl"


#: Binding fields every chunk header must carry; resume re-verifies them.
CHUNK_BINDING_FIELDS = (
    "schema_version",
    "pair_manifest_sha256",
    "git_sha",
    "checkpoint_sha256",
    "device",
    "torch_dtype",
    "trim_mode",
    "text_cache",
    "score_name",
    "spec_version",
)


def verify_chunk_records(
    records: Sequence[dict[str, Any]],
    pair_space: PairSpace,
    shard_id: int,
    t0: int,
    t1: int,
    expected_binding: dict[str, Any],
    extra_binding_fields: Sequence[str] = (),
) -> list[str]:
    """Verify a completed chunk file for resume; returns problems ([] = OK).

    A chunk is acceptable ONLY as: one ``chunk_header`` record whose binding
    fields equal ``expected_binding`` and whose shard/text-range matches the
    file name, followed by EXACTLY the dense ascending pair_index sequence of
    the chunk, each with a finite complete score. A rejected or incomplete
    row cannot stand in for one of the required exact matrix scores.

    ``extra_binding_fields`` are ADDITIONAL header keys that must equal
    ``expected_binding.get(field)`` -- even when that expected value is None
    (a scoped run, e.g. a probe selection, refuses chunks from the unscoped
    full matrix and vice versa). The default keeps the historical behavior.
    """
    problems: list[str] = []
    expected_pairs = pair_space.chunk_pair_indices(t0, t1)
    if not records:
        return ["empty chunk file"]
    header = records[0]
    if header.get("record_type") != "chunk_header":
        return ["first record is not a chunk_header"]
    if header.get("shard_id") != shard_id or header.get("text_range") != [t0, t1]:
        problems.append(
            f"header shard/range mismatch: {header.get('shard_id')}/"
            f"{header.get('text_range')} vs s{shard_id}/[{t0},{t1})"
        )
    if header.get("n_pairs") != len(expected_pairs) or header.get("n_scored") != len(expected_pairs) or header.get("n_rejected") != 0:
        problems.append("header pair/scored/rejected counts do not describe a complete exact chunk")
    if header.get("pair_index_first") != expected_pairs.start or header.get("pair_index_last") != expected_pairs.stop - 1:
        problems.append("header pair_index endpoints do not match the exact chunk")
    for field in CHUNK_BINDING_FIELDS:
        if field in expected_binding and header.get(field) != expected_binding[field]:
            problems.append(
                f"header binding {field}: {header.get(field)!r} != {expected_binding[field]!r}"
            )
    for field in extra_binding_fields:
        if header.get(field) != expected_binding.get(field):
            problems.append(
                f"header scope {field}: {header.get(field)!r} != {expected_binding.get(field)!r}"
            )
    body = records[1:]
    if len(body) != len(expected_pairs):
        problems.append(f"chunk holds {len(body)} pair records, expected {len(expected_pairs)}")
        return problems
    for offset, (record, pair_index) in enumerate(zip(body, expected_pairs)):
        if record.get("record_type") != "pair":
            problems.append(f"record {offset}: record_type != 'pair'")
            break
        if record.get("pair_index") != pair_index:
            problems.append(
                f"record {offset}: pair_index {record.get('pair_index')} != expected {pair_index}"
            )
            break
        text_index, cad_index = pair_space.coords_of(pair_index)
        if record.get("text_index") != text_index or record.get("cad_index") != cad_index:
            problems.append(f"record {offset}: text/cad index mismatch")
            break
        if record.get("status") != "scored":
            problems.append(
                f"record {offset}: status {record.get('status')!r} is not scored; re-score chunk"
            )
            break
        for field in ("score_x", "score_y", "score_total", "mean_x", "mean_y", "mean_total"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"record {offset}: {field} is absent or nonfinite")
                break
        if problems:
            break
        for field in ("n_valid_x", "n_valid_y", "n_valid_total", "valid_seq_len", "n_real_tokens"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                problems.append(f"record {offset}: {field} is absent or invalid")
                break
        if problems:
            break
        if record["n_valid_x"] + record["n_valid_y"] != record["n_valid_total"]:
            problems.append(f"record {offset}: valid-factor counts disagree")
            break
        if not math.isclose(record["score_x"] + record["score_y"], record["score_total"], rel_tol=1e-6, abs_tol=1e-3):
            problems.append(f"record {offset}: factor scores do not sum to total")
            break
    return problems


# ---------------------------------------------------------------------------
# Builder assembly
# ---------------------------------------------------------------------------


def _expect(checks: list[dict[str, Any]], name: str, got: Any, expected: Any) -> None:
    checks.append({"name": name, "expected": expected, "got": got, "match": got == expected})


def build_pair_manifest(
    query_path,
    gallery_path,
    *,
    expected_n_query: int = EXPECTED_N_QUERY,
    expected_n_unique_text: int = EXPECTED_N_UNIQUE_TEXT,
    expected_n_gallery: int = EXPECTED_N_GALLERY,
) -> dict[str, Any]:
    """Build the complete pair-space description from the frozen M0 manifests.

    Returns a dict with the in-memory artifacts under ``artifacts`` and the
    manifest under ``manifest`` (hashes of the artifact *content* are filled
    by :func:`write_pair_manifest`, which is the only writer). Raises
    PairIndexError on any contract violation; expectation mismatches are
    hard failures, not warnings.
    """
    query_records = load_query_records(query_path)
    gallery_records = load_gallery_records(gallery_path)
    text_entries, query_to_text = build_text_index(query_records)
    cad_entries = build_cad_store(gallery_records)

    checks: list[dict[str, Any]] = []
    _expect(checks, "n_query", len(query_records), expected_n_query)
    _expect(checks, "n_unique_text", len(text_entries), expected_n_unique_text)
    _expect(checks, "n_gallery", len(cad_entries), expected_n_gallery)
    _expect(
        checks,
        "n_pairs",
        len(text_entries) * len(cad_entries),
        expected_n_unique_text * expected_n_gallery,
    )
    failed = [c for c in checks if not c["match"]]
    if failed:
        raise PairIndexError(
            REASON_COUNT_MISMATCH,
            "expectation mismatch: "
            + ", ".join(f"{c['name']} got {c['got']} expected {c['expected']}" for c in failed),
        )

    pair_space = PairSpace(n_texts=len(text_entries), n_cads=len(cad_entries))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            # Runtime locations differ across Windows and yunnan-share2. The
            # bytes, not the host path, define the frozen pair space.
            "query_jsonl": {"sha256": sha256_file(query_path)},
            "gallery_jsonl": {"sha256": sha256_file(gallery_path)},
        },
        "counts": {
            "n_query": len(query_records),
            "n_unique_text": len(text_entries),
            "n_gallery": len(cad_entries),
            "n_pairs": pair_space.n_pairs,
        },
        "ordering": {
            "texts": "canonical: sorted by (text_sha256, text); dedup by exact string equality on strip()-normalized M0 text",
            "cads": "frozen gallery order (verified ascending cad_id)",
            "pair_formula": PAIR_FORMULA,
        },
        "sharding": {
            "rule": "contiguous text ranges; boundary k = round(k * n_texts / n_shards)",
            "pair_range_of_shard": "[t0 * n_cads, t1 * n_cads)",
            "chunk_rule": "texts_per_chunk consecutive texts per chunk file; default 1",
        },
        "isolation": {
            "worker_inputs": [TEXTS_FILE, CAD_STORE_FILE],
            "evaluation_only": [EVALUATION_MAP_FILE],
            "never_read_by_any_stage": ["qrels.jsonl (evaluation module only)"],
        },
        "expectations_checked": checks,
    }
    return {
        "manifest": manifest,
        "pair_space": pair_space,
        "text_entries": text_entries,
        "cad_entries": cad_entries,
        "cad_ids": [rec["cad_id"] for rec in gallery_records],
        "query_to_text": query_to_text,
    }


def ensure_outside_checkout(out_dir, repo_root) -> None:
    """Refuse to write builder outputs inside the Git checkout."""
    out_resolved = Path(out_dir).resolve()
    repo_resolved = Path(repo_root).resolve()
    if out_resolved == repo_resolved or repo_resolved in out_resolved.parents:
        raise PairIndexError(
            REASON_INSIDE_CHECKOUT,
            f"out_dir {out_resolved} is inside the Git checkout {repo_resolved}",
        )


def write_pair_manifest(
    out_dir,
    built: dict[str, Any],
    *,
    repo_root=None,
    zip_report: Optional[dict] = None,
    cad_length_records: Optional[list[dict[str, int]]] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the four/five deterministic artifacts and finish the manifest.

    Files: ``texts.jsonl`` + ``cad_store.jsonl`` (worker-facing),
    ``evaluation_map.json`` (evaluation-only linkage), ``pair_manifest.json``
    (small binding manifest incl. artifact SHA256s), and optionally
    ``cad_length_index.jsonl`` (zero-forward structural scan facts).
    Existing files are never overwritten unless ``overwrite`` is passed
    (rebuilds are expected to be byte-identical; overwriting a DIFFERENT
    prior build would silently change the pair space).
    """
    out_dir = Path(out_dir)
    if repo_root is not None:
        ensure_outside_checkout(out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        TEXTS_FILE: out_dir / TEXTS_FILE,
        CAD_STORE_FILE: out_dir / CAD_STORE_FILE,
        EVALUATION_MAP_FILE: out_dir / EVALUATION_MAP_FILE,
        PAIR_MANIFEST_FILE: out_dir / PAIR_MANIFEST_FILE,
    }
    if cad_length_records is not None:
        paths[CAD_LENGTH_FILE] = out_dir / CAD_LENGTH_FILE
    for name, path in paths.items():
        if path.exists() and not overwrite:
            raise PairIndexError(
                REASON_OUTPUT_EXISTS, f"{path} exists; refusing to overwrite a prior build"
            )

    text_records = [
        {
            "text_index": e.text_index,
            "text": e.text,
            "text_sha256": e.text_sha256,
            "n_chars": e.n_chars,
        }
        for e in built["text_entries"]
    ]
    cad_records = [{"cad_index": e.cad_index, "member": e.member} for e in built["cad_entries"]]
    evaluation_map = {
        "note": (
            "evaluation-only linkage; the scoring runner never reads this file. "
            "query_to_text maps M0 query_id -> text_index; cad_index_to_cad_id maps "
            "cad_index -> official test UID. Relevance lives ONLY in qrels.jsonl, "
            "which no stage of this pipeline opens."
        ),
        "query_to_text": dict(sorted(built["query_to_text"].items())),
        "cad_index_to_cad_id": list(built["cad_ids"]),
    }

    write_jsonl(paths[TEXTS_FILE], text_records)
    write_jsonl(paths[CAD_STORE_FILE], cad_records)
    if cad_length_records is not None:
        write_jsonl(paths[CAD_LENGTH_FILE], cad_length_records)
    paths[EVALUATION_MAP_FILE].write_text(
        json.dumps(evaluation_map, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )

    manifest = dict(built["manifest"])
    manifest["artifacts"] = {
        name: {"file": name, "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
        if name not in (PAIR_MANIFEST_FILE, EVALUATION_MAP_FILE)
    }
    if zip_report is not None:
        manifest["cad_seq_zip"] = zip_report
    manifest_path = paths[PAIR_MANIFEST_FILE]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )
    # Re-record the manifest's own hash for the summary (not stored inside itself).
    summary = {
        "out_dir": str(out_dir),
        "n_pairs": manifest["counts"]["n_pairs"],
        "artifact_sha256": {name: info["sha256"] for name, info in manifest["artifacts"].items()},
        "evaluation_artifact_sha256": {EVALUATION_MAP_FILE: sha256_file(paths[EVALUATION_MAP_FILE])},
        "pair_manifest_sha256": sha256_file(manifest_path),
    }
    return {"manifest": manifest, "summary": summary}


# ---------------------------------------------------------------------------
# Loader for the committed/verified artifacts (runner side)
# ---------------------------------------------------------------------------


def load_pair_manifest(manifest_dir, expected_sha256: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Load ``pair_manifest.json`` and verify worker artifacts only.

    ``expected_sha256`` (file name -> sha256) is the REVIEWED fingerprint from
    the committed config; when given, any drift between the on-disk artifacts
    and the reviewed values is a hard failure. The worker-facing files are
    validated for schema only -- content checks happen here via SHA.
    """
    manifest_dir = Path(manifest_dir)
    manifest_path = manifest_dir / PAIR_MANIFEST_FILE
    if not manifest_path.is_file():
        raise PairIndexError(REASON_BAD_JSON, f"missing {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PairIndexError(REASON_BAD_JSON, f"unreadable pair manifest: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PairIndexError(
            REASON_SCHEMA_MISMATCH,
            f"pair manifest schema {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}",
        )
    artifacts = manifest.get("artifacts") or {}
    problems: list[str] = []
    if EVALUATION_MAP_FILE in artifacts or EVALUATION_MAP_FILE in (expected_sha256 or {}):
        raise PairIndexError(REASON_SCHEMA_MISMATCH, "evaluation linkage cannot be a scoring artifact")
    for name, info in sorted(artifacts.items()):
        path = manifest_dir / name
        if not path.is_file():
            problems.append(f"artifact {name} missing")
            continue
        actual = sha256_file(path)
        if actual != info.get("sha256"):
            problems.append(f"artifact {name} sha256 drift: {actual} != {info.get('sha256')}")
    if expected_sha256:
        for name, sha in sorted(expected_sha256.items()):
            if name == PAIR_MANIFEST_FILE:
                continue
            pinned = (artifacts.get(name) or {}).get("sha256")
            if pinned != sha:
                problems.append(
                    f"reviewed fingerprint mismatch for {name}: manifest pins {pinned}, "
                    f"reviewed config says {sha}"
                )
        actual_manifest_sha = sha256_file(manifest_path)
        if expected_sha256.get(PAIR_MANIFEST_FILE) not in (None, actual_manifest_sha):
            problems.append(
                f"pair_manifest.json sha256 {actual_manifest_sha} != reviewed "
                f"{expected_sha256.get(PAIR_MANIFEST_FILE)}"
            )
    if problems:
        raise PairIndexError(REASON_ASSET_SHA_MISMATCH, "; ".join(problems))
    return manifest


def load_texts(manifest_dir) -> list[str]:
    """Worker-facing text table: returns texts indexed by text_index.

    Strict schema (exactly the TEXTS_RECORD_KEYS), contiguous indices
    0..n-1, and per-record sha/n_chars consistency re-verified so a corrupted
    or re-ordered file cannot silently change which text a pair scores.
    """
    manifest_dir = Path(manifest_dir)
    records = read_jsonl_strict(manifest_dir / TEXTS_FILE, TEXTS_RECORD_KEYS, TEXTS_FILE)
    texts: list[Optional[str]] = [None] * len(records)
    for rec in records:
        index = rec["text_index"]
        if not isinstance(index, int) or not 0 <= index < len(records):
            raise PairIndexError(REASON_FIELD_MISMATCH, f"bad text_index {index!r}")
        if texts[index] is not None:
            raise PairIndexError(REASON_DUPLICATE_ID, f"duplicate text_index {index}")
        text = rec["text"]
        if not isinstance(text, str) or not text or text != text.strip():
            raise PairIndexError(REASON_TEXT_DRIFT, f"text_index {index}: bad text")
        if rec["text_sha256"] != sha256_text(text) or rec["n_chars"] != len(text):
            raise PairIndexError(REASON_FIELD_MISMATCH, f"text_index {index}: integrity drift")
        texts[index] = text
    if any(t is None for t in texts):
        raise PairIndexError(REASON_COUNT_MISMATCH, "text_index coverage is not contiguous")
    return [t for t in texts if t is not None]


def load_background_texts(manifest_dir, expected_text: str) -> str:
    """E2-only auditable admission for a single designated background text.

    The general `load_texts` contract is UNTOUCHED (it keeps rejecting empty
    strings for query texts). This loader accepts EXACTLY one texts.jsonl
    record whose text equals `expected_text` verbatim -- including the empty
    string for the E2 B0 background -- with consistent text_sha256/n_chars
    and text_index 0. Anything else is a hard refusal.
    """
    manifest_dir = Path(manifest_dir)
    records = read_jsonl_strict(manifest_dir / TEXTS_FILE, TEXTS_RECORD_KEYS, TEXTS_FILE)
    if len(records) != 1:
        raise PairIndexError(
            REASON_COUNT_MISMATCH,
            f"background manifest must contain exactly 1 text record, got {len(records)}",
        )
    rec = records[0]
    if rec["text_index"] != 0:
        raise PairIndexError(REASON_FIELD_MISMATCH, f"background text_index must be 0, got {rec['text_index']!r}")
    text = rec["text"]
    if not isinstance(text, str) or text != expected_text:
        raise PairIndexError(
            REASON_BACKGROUND_TEXT_MISMATCH,
            f"background text mismatch: got {text!r}, expected {expected_text!r}",
        )
    if rec["text_sha256"] != sha256_text(text) or rec["n_chars"] != len(text):
        raise PairIndexError(REASON_FIELD_MISMATCH, "background text integrity drift")
    return text


def load_cad_store(manifest_dir) -> list[str]:
    """Worker-facing CAD store: returns member paths indexed by cad_index."""
    manifest_dir = Path(manifest_dir)
    records = read_jsonl_strict(manifest_dir / CAD_STORE_FILE, CAD_STORE_RECORD_KEYS, CAD_STORE_FILE)
    members: list[Optional[str]] = [None] * len(records)
    for rec in records:
        index = rec["cad_index"]
        if not isinstance(index, int) or not 0 <= index < len(records):
            raise PairIndexError(REASON_FIELD_MISMATCH, f"bad cad_index {index!r}")
        if members[index] is not None:
            raise PairIndexError(REASON_DUPLICATE_ID, f"duplicate cad_index {index}")
        member = rec["member"]
        if not isinstance(member, str) or not _MEMBER_RE.match(member):
            raise PairIndexError(REASON_MEMBER_MISMATCH, f"cad_index {index}: bad member {member!r}")
        members[index] = member
    if any(m is None for m in members):
        raise PairIndexError(REASON_COUNT_MISMATCH, "cad_index coverage is not contiguous")
    return [m for m in members if m is not None]


def load_cad_length_index(manifest_dir) -> list[dict[str, int]]:
    """Optional zero-forward length facts (cad_index order), schema-checked."""
    manifest_dir = Path(manifest_dir)
    records = read_jsonl_strict(manifest_dir / CAD_LENGTH_FILE, CAD_LENGTH_RECORD_KEYS, CAD_LENGTH_FILE)
    out: list[Optional[dict[str, int]]] = [None] * len(records)
    for rec in records:
        index = rec["cad_index"]
        if not isinstance(index, int) or not 0 <= index < len(records) or out[index] is not None:
            raise PairIndexError(REASON_FIELD_MISMATCH, f"bad cad_index in length index: {index!r}")
        for field in ("n_real_tokens", "n_valid_x", "n_valid_y"):
            if not isinstance(rec[field], int) or rec[field] < 0:
                raise PairIndexError(
                    REASON_FIELD_MISMATCH, f"cad_index {index}: bad {field} {rec[field]!r}"
                )
        out[index] = {field: rec[field] for field in CAD_LENGTH_RECORD_KEYS - {"cad_index"}}
        out[index]["cad_index"] = index
    if any(r is None for r in out):
        raise PairIndexError(REASON_COUNT_MISMATCH, "length index coverage is not contiguous")
    return [r for r in out if r is not None]


__all__ = [
    "CAD_SEQ_ZIP_SHA256",
    "CAD_STORE_FILE",
    "CAD_LENGTH_FILE",
    "CHUNK_BINDING_FIELDS",
    "EVALUATION_MAP_FILE",
    "EXPECTED_N_GALLERY",
    "EXPECTED_N_PAIRS",
    "EXPECTED_N_QUERY",
    "EXPECTED_N_UNIQUE_TEXT",
    "PAIR_FORMULA",
    "PAIR_MANIFEST_FILE",
    "SCHEMA_VERSION",
    "TEXTS_FILE",
    "CadEntry",
    "PairIndexError",
    "PairSpace",
    "TextEntry",
    "build_cad_store",
    "build_pair_manifest",
    "build_text_index",
    "chunk_file_name",
    "ensure_outside_checkout",
    "load_cad_length_index",
    "load_cad_store",
    "load_gallery_records",
    "load_pair_manifest",
    "load_query_records",
    "load_texts",
    "read_jsonl_strict",
    "scan_cad_lengths",
    "sha256_file",
    "sha256_text",
    "verify_chunk_records",
    "verify_zip_members",
    "write_jsonl",
    "write_pair_manifest",
]
