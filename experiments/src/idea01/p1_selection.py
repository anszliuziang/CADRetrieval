"""P1 candidate selection: 32 queries x 8 candidates, fixed a priori (seed 20260925).

Plan contract (experiments/plans/IDEA01_P1_CONDITIONAL_SENSITIVITY_2026-09-25.md, sec.2):

  Eligible pool (validation split only), after sorting by uid ascending:
    unique uid, ZIP program byte-identical to JSONL `output`, prefix token
    boundary OK, no EOS inside tokenized output, n_prefix_tokens <= 250,
    n_response_tokens <= 700 (EOS included). Selection never looks at model
    scores. If the pool has fewer than 32 rows, stop (SelectionError).

  rng = random.Random(20260925)
    - 32 distinct query/paired programs drawn with rng.sample(pool_uids, 32);
      the sampled order is kept as the fixed query order.
    - 3 distinct cyclic offsets drawn from 1..31 with the SAME rng instance;
      the same offsets are used for every query i: candidate (i+o) mod 32.
      Each selected program thus serves as an offset candidate for exactly 3
      other queries, enabling the within-candidate margin diagnostic.

  For query i, 8 distinct candidates:
    1 self            the paired program
    3 selected-offset selected[(i+o) % 32]
    2 method-neighbor from pool minus the 32 selected uids, ranked by
                      Jaccard(method_set(self), method_set(cand)) desc,
                      |response token length diff| asc, uid asc
    2 length-neighbor from the remaining pool (pool minus 32 selected minus
                      this query's 2 method neighbors, so all 8 are distinct),
                      ranked by |response token length diff| asc, uid asc
  If any query cannot reach 8 distinct candidates, stop (SelectionError); the
  rules are NOT relaxed.

  Method names: regex `\\.([A-Za-z_][A-Za-z0-9_]*)\\(` over the raw program,
  excluding {"export", "exporters"} (per plan). Jaccard of two empty sets is
  defined as 0.0.

Pair plan: 32x8 = 256 conditional pairs plus one empty-query pair per distinct
candidate program (<= 32 + 4*32 = 160), total forwards <= 416 (< 512 cap).

UIDs/row indices in the selection are bookkeeping only; the scorer receives
only query text and candidate program text.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

SEED = 20260925
N_QUERIES = 32
N_OFFSETS = 3
N_METHOD_NEIGHBORS = 2
N_LENGTH_NEIGHBORS = 2
OFFSET_MIN = 1  # offsets drawn from [1, 31]; 0 excluded so offset != self
MAX_PREFIX_TOKENS = 250
MAX_RESPONSE_TOKENS = 700  # EOS included
MAX_FORWARDS = 416  # 256 conditional + <=160 empty; hard budget cap

METHOD_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\(")
METHOD_EXCLUDE = frozenset({"export", "exporters"})

CANDIDATE_SOURCES = ("self", "selected-offset", "method-neighbor", "length-neighbor")


class SelectionError(Exception):
    """Hard stop: pool too small, candidates not distinct, inconsistent input."""


def extract_method_set(program: str) -> frozenset[str]:
    """CadQuery method-name set from the raw program: `\\.<name>(` matches,
    excluding export/exporters (boilerplate-adjacent, per plan)."""
    return frozenset(m for m in METHOD_RE.findall(program) if m not in METHOD_EXCLUDE)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|A&B| / |A|B|; defined 0.0 when the union is empty (both sets empty)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class RowRecord:
    """One validation row with everything P1 eligibility/selection needs.

    Token counts follow coverage.py conventions: n_response_tokens includes
    the appended EOS; boundary_ok means tok(prefix) is a strict token prefix
    of tok(prefix+output); eos_in_output means the EOS id appeared inside the
    tokenized output (boundary inseparable -> exclude).
    """

    row_index: int
    uid: str | None  # unique CSV mapping; None if unmapped/ambiguous
    program_identical: bool  # ZIP member CQ/{uid}.py byte-identical to JSONL output
    n_prefix_tokens: int
    n_response_tokens: int  # includes EOS
    boundary_ok: bool
    eos_in_output: bool
    method_set: frozenset[str] = field(default_factory=frozenset)


def row_is_eligible(r: RowRecord, max_prefix: int, max_response: int) -> bool:
    return (
        r.uid is not None
        and r.program_identical
        and r.boundary_ok
        and not r.eos_in_output
        and r.n_prefix_tokens <= max_prefix
        and r.n_response_tokens <= max_response
    )


def compute_eligibility(
    records: list[RowRecord], max_prefix: int = MAX_PREFIX_TOKENS, max_response: int = MAX_RESPONSE_TOKENS
) -> tuple[list[bool], dict[str, int]]:
    """Per-row eligibility flags plus exclusion counts (a row is counted under
    every criterion it fails; criteria are independent)."""
    flags: list[bool] = []
    counts = {
        "n_rows": len(records),
        "excl_uid_not_unique": 0,
        "excl_program_not_identical": 0,
        "excl_boundary": 0,
        "excl_eos_in_output": 0,
        "excl_prefix_too_long": 0,
        "excl_response_too_long": 0,
        "n_eligible": 0,
    }
    for r in records:
        ok = True
        if r.uid is None:
            counts["excl_uid_not_unique"] += 1
            ok = False
        if not r.program_identical:
            counts["excl_program_not_identical"] += 1
            ok = False
        if not r.boundary_ok:
            counts["excl_boundary"] += 1
            ok = False
        if r.eos_in_output:
            counts["excl_eos_in_output"] += 1
            ok = False
        if r.n_prefix_tokens > max_prefix:
            counts["excl_prefix_too_long"] += 1
            ok = False
        if r.n_response_tokens > max_response:
            counts["excl_response_too_long"] += 1
            ok = False
        flags.append(ok)
        if ok:
            counts["n_eligible"] += 1
    return flags, counts


@dataclass
class CandidatePick:
    cand_index: int  # 0..7 within the query's candidate list
    source: str  # one of CANDIDATE_SOURCES
    uid: str
    row_index: int
    n_prefix_tokens: int  # prefix length of the candidate's OWN row (info only)
    n_response_tokens: int  # includes EOS
    offset: int | None = None  # cyclic offset for selected-offset picks
    method_similarity: float | None = None  # Jaccard vs the paired program (method-neighbor)
    method_set: tuple[str, ...] = ()
    len_diff: int | None = None  # |n_response_tokens - positive's| (neighbor picks)


@dataclass
class QueryPick:
    query_index: int  # 0..31, the fixed sampled order
    uid: str
    row_index: int
    n_prefix_tokens: int
    n_response_tokens: int  # includes EOS
    method_set: tuple[str, ...]
    n_method_pool: int = 0  # candidates available at the method-neighbor step (pool minus 32 selected)
    n_length_pool: int = 0  # candidates available at the length-neighbor step (also minus this query's method picks)
    candidates: list[CandidatePick] = field(default_factory=list)


@dataclass
class Selection:
    seed: int
    offsets: list[int]
    queries: list[QueryPick]
    pool_uids: list[str]  # full eligible pool, uid ascending
    distinct_candidate_uids: list[str]  # 32 selected first (sampled order), then neighbors by first appearance


def draw_queries_and_offsets(pool_uids_sorted: list[str], n_queries: int, n_offsets: int, seed: int) -> tuple[list[str], list[int]]:
    """Fixed draw: one random.Random(seed) instance, first the n_queries sample
    from the uid-ascending pool (sampled order kept), then n_offsets distinct
    cyclic offsets from [1, n_queries-1]."""
    if len(pool_uids_sorted) < n_queries:
        raise SelectionError(f"eligible pool has {len(pool_uids_sorted)} rows, need >= {n_queries}; stop per plan")
    if len(set(pool_uids_sorted)) != len(pool_uids_sorted):
        raise SelectionError("eligible pool contains duplicate uids")
    rng = random.Random(seed)
    selected = rng.sample(pool_uids_sorted, n_queries)
    offsets = rng.sample(range(OFFSET_MIN, n_queries), n_offsets)
    return selected, offsets


def rank_method_neighbors(self_rec: RowRecord, pool: list[RowRecord], exclude_uids: set[str]) -> list[RowRecord]:
    cands = [r for r in pool if r.uid not in exclude_uids]
    cands.sort(
        key=lambda r: (
            -jaccard(self_rec.method_set, r.method_set),
            abs(r.n_response_tokens - self_rec.n_response_tokens),
            r.uid or "",
        )
    )
    return cands


def rank_length_neighbors(self_rec: RowRecord, pool: list[RowRecord], exclude_uids: set[str]) -> list[RowRecord]:
    cands = [r for r in pool if r.uid not in exclude_uids]
    cands.sort(key=lambda r: (abs(r.n_response_tokens - self_rec.n_response_tokens), r.uid or ""))
    return cands


def select_p1(
    records: list[RowRecord],
    eligible_flags: list[bool],
    seed: int = SEED,
    n_queries: int = N_QUERIES,
    n_offsets: int = N_OFFSETS,
    n_method: int = N_METHOD_NEIGHBORS,
    n_length: int = N_LENGTH_NEIGHBORS,
) -> Selection:
    """Full deterministic P1 selection. Pure function of (records, flags, seed)."""
    if len(records) != len(eligible_flags):
        raise SelectionError("records/eligible_flags length mismatch")
    pool = sorted((r for r, ok in zip(records, eligible_flags) if ok), key=lambda r: (r.uid or "", r.row_index))
    pool_uids = [r.uid or "" for r in pool]
    selected_uids, offsets = draw_queries_and_offsets(pool_uids, n_queries, n_offsets, seed)
    by_uid = {r.uid: r for r in pool}
    selected_set = set(selected_uids)

    queries: list[QueryPick] = []
    distinct: list[str] = list(selected_uids)
    for qi, uid in enumerate(selected_uids):
        rec = by_uid[uid]
        qp = QueryPick(
            query_index=qi,
            uid=uid,
            row_index=rec.row_index,
            n_prefix_tokens=rec.n_prefix_tokens,
            n_response_tokens=rec.n_response_tokens,
            method_set=tuple(sorted(rec.method_set)),
        )
        picks: list[CandidatePick] = [
            CandidatePick(
                cand_index=0,
                source="self",
                uid=uid,
                row_index=rec.row_index,
                n_prefix_tokens=rec.n_prefix_tokens,
                n_response_tokens=rec.n_response_tokens,
                method_set=tuple(sorted(rec.method_set)),
            )
        ]
        for k, off in enumerate(offsets):
            cuid = selected_uids[(qi + off) % n_queries]
            crec = by_uid[cuid]
            picks.append(
                CandidatePick(
                    cand_index=1 + k,
                    source="selected-offset",
                    uid=cuid,
                    row_index=crec.row_index,
                    n_prefix_tokens=crec.n_prefix_tokens,
                    n_response_tokens=crec.n_response_tokens,
                    offset=off,
                    method_set=tuple(sorted(crec.method_set)),
                )
            )
        method_cands = rank_method_neighbors(rec, pool, selected_set)
        qp.n_method_pool = len(method_cands)
        if len(method_cands) < n_method:
            raise SelectionError(
                f"query {qi} ({uid}): only {len(method_cands)} method-neighbor candidates available, need {n_method}; stop per plan"
            )
        method_picked = method_cands[:n_method]
        for k, crec in enumerate(method_picked):
            picks.append(
                CandidatePick(
                    cand_index=1 + n_offsets + k,
                    source="method-neighbor",
                    uid=crec.uid or "",
                    row_index=crec.row_index,
                    n_prefix_tokens=crec.n_prefix_tokens,
                    n_response_tokens=crec.n_response_tokens,
                    method_similarity=jaccard(rec.method_set, crec.method_set),
                    method_set=tuple(sorted(crec.method_set)),
                    len_diff=abs(crec.n_response_tokens - rec.n_response_tokens),
                )
            )
        exclude_for_length = selected_set | {c.uid for c in method_picked}
        length_cands = rank_length_neighbors(rec, pool, exclude_for_length)
        qp.n_length_pool = len(length_cands)
        if len(length_cands) < n_length:
            raise SelectionError(
                f"query {qi} ({uid}): only {len(length_cands)} length-neighbor candidates available, need {n_length}; stop per plan"
            )
        for k, crec in enumerate(length_cands[:n_length]):
            picks.append(
                CandidatePick(
                    cand_index=1 + n_offsets + n_method + k,
                    source="length-neighbor",
                    uid=crec.uid or "",
                    row_index=crec.row_index,
                    n_prefix_tokens=crec.n_prefix_tokens,
                    n_response_tokens=crec.n_response_tokens,
                    method_set=tuple(sorted(crec.method_set)),
                    len_diff=abs(crec.n_response_tokens - rec.n_response_tokens),
                )
            )
        n_want = 1 + n_offsets + n_method + n_length
        pick_uids = [p.uid for p in picks]
        if len(picks) != n_want or len(set(pick_uids)) != n_want:
            raise SelectionError(f"query {qi} ({uid}): candidates not {n_want} distinct uids: {pick_uids}; stop per plan")
        for p in picks:
            if p.uid not in distinct:
                distinct.append(p.uid)
        qp.candidates = picks
        queries.append(qp)

    return Selection(seed=seed, offsets=offsets, queries=queries, pool_uids=pool_uids, distinct_candidate_uids=distinct)


def build_pair_plan(selection: Selection) -> list[dict]:
    """256 conditional pairs + one empty pair per distinct candidate.

    Pair ids: q{query_index:02d}_{cand_index}_{source-tag}; empty: e{idx:03d}
    over distinct_candidate_uids (32 selected first, then neighbors).
    """
    src_tag = {"self": "self", "selected-offset": "off", "method-neighbor": "method", "length-neighbor": "length"}
    pairs: list[dict] = []
    for qp in selection.queries:
        for c in qp.candidates:
            pairs.append(
                {
                    "pair_id": f"q{qp.query_index:02d}_{c.cand_index}_{src_tag[c.source]}",
                    "kind": "conditional",
                    "query_index": qp.query_index,
                    "query_uid": qp.uid,
                    "query_row_index": qp.row_index,
                    "candidate_source": c.source,
                    "offset": c.offset,
                    "candidate_uid": c.uid,
                    "candidate_row_index": c.row_index,
                }
            )
    for idx, uid in enumerate(selection.distinct_candidate_uids):
        row_index = None
        for qp in selection.queries:
            for c in qp.candidates:
                if c.uid == uid:
                    row_index = c.row_index
                    break
            if row_index is not None:
                break
        if row_index is None:
            raise SelectionError(f"distinct candidate {uid} has no row_index (internal inconsistency)")
        pairs.append(
            {
                "pair_id": f"e{idx:03d}",
                "kind": "empty",
                "query_index": None,
                "query_uid": None,
                "query_row_index": None,
                "candidate_source": "empty-query",
                "offset": None,
                "candidate_uid": uid,
                "candidate_row_index": row_index,
            }
        )
    if len(pairs) > MAX_FORWARDS:
        raise SelectionError(f"pair plan has {len(pairs)} pairs, exceeds forward budget {MAX_FORWARDS}")
    ids = [p["pair_id"] for p in pairs]
    if len(set(ids)) != len(ids):
        raise SelectionError("duplicate pair_id in plan")
    return pairs


def offset_coverage_check(selection: Selection, n_offsets: int = N_OFFSETS) -> dict:
    """Every selected program must appear as an offset candidate for exactly
    n_offsets other queries (fixed-offset cyclic design)."""
    n = len(selection.queries)
    counts = {qp.uid: 0 for qp in selection.queries}
    for qp in selection.queries:
        for c in qp.candidates:
            if c.source == "selected-offset":
                counts[c.uid] += 1
    bad = {u: k for u, k in counts.items() if k != n_offsets}
    return {"ok": not bad, "n_queries": n, "violations": bad}
