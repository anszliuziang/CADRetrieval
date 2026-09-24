"""Deterministic P0 candidate and mismatch-query selection.

Selection never looks at model scores (contract). Rule, fixed a priori:
  1. validation rows with a uniquely mapped uid, boundary_ok, and
     full (prefix+output+EOS) token length <= max_len;
  2. sort eligible rows by uid ascending;
  3. take the first n_candidates.

Mismatch query for candidate i: the `input` text of one other selected
candidate, drawn with random.Random(seed) from the other candidates' indices,
so it is reproducible from (uid list, seed) alone. The empty condition uses the
empty string as the query.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Candidate:
    cand_index: int  # 0..n-1 within the P0 selection
    row_index: int  # row index within the split file
    uid: str
    n_full_tokens: int
    mismatch_cand_index: int  # index of the candidate providing the wrong query


def select_candidates(
    row_uids: list[str | None],
    fits: list[bool],
    boundary_oks: list[bool],
    n_full_tokens: list[int],
    n_candidates: int,
    seed: int,
) -> list[Candidate]:
    n = len(row_uids)
    if not (len(fits) == len(boundary_oks) == len(n_full_tokens) == n):
        raise ValueError("length mismatch between audit inputs")

    eligible = [
        (row_uids[i], i)
        for i in range(n)
        if row_uids[i] is not None and fits[i] and boundary_oks[i]
    ]
    eligible.sort(key=lambda t: (t[0], t[1]))  # uid asc, row_index tiebreak
    chosen = eligible[:n_candidates]
    if len(chosen) < 2:
        raise ValueError(f"need >=2 eligible candidates, got {len(chosen)}")

    rng = random.Random(seed)
    cands: list[Candidate] = []
    m = len(chosen)
    for ci, (uid, row_index) in enumerate(chosen):
        others = [j for j in range(m) if j != ci]
        mismatch = rng.choice(others)
        cands.append(
            Candidate(
                cand_index=ci,
                row_index=row_index,
                uid=uid,  # type: ignore[arg-type]
                n_full_tokens=n_full_tokens[row_index],
                mismatch_cand_index=mismatch,
            )
        )
    return cands


CONDITIONS = ("correct", "mismatched", "empty")


def build_pair_plan(cands: list[Candidate]) -> list[dict]:
    """3 pairs per candidate: correct / mismatched / empty query."""
    pairs = []
    for c in cands:
        for cond in CONDITIONS:
            pairs.append(
                {
                    "pair_id": f"c{c.cand_index:02d}_{cond}",
                    "cand_index": c.cand_index,
                    "candidate_uid": c.uid,
                    "candidate_row_index": c.row_index,
                    "condition": cond,
                    "query_uid": (
                        c.uid
                        if cond == "correct"
                        else (cands[c.mismatch_cand_index].uid if cond == "mismatched" else None)
                    ),
                }
            )
    return pairs
