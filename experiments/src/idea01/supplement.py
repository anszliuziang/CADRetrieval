"""Review supplement: CPU/tokenizer-only reconstruction of the 60 P0 pairs'
FULL model input token ids, cross-checked against the original run evidence.

Codex review 2026-09-25 item 1: the original run saved only target token ids in
tokens/<pair_id>.jsonl; the contract requires the complete model input ids. This
module rebuilds them from the pinned data + checkpoint tokenizer (no model
forward, no re-scoring) and verifies every target token id against the original
per-token evidence. Original P0 files are never modified; outputs go to a
separate review_supplement/ directory.

Scoring semantics are unchanged: reconstruction uses the same prefix template,
the same build_scored_sequence boundary assertion, and the same EOS append as
the original run (Git SHA recorded in each original pair record).
"""

from __future__ import annotations

from typing import Callable

from .prompts import build_prefix
from .scoring import build_scored_sequence


def query_for_condition(condition: str, own_input: str, mismatch_input: str | None) -> str:
    if condition == "correct":
        return own_input
    if condition == "mismatched":
        if mismatch_input is None:
            raise ValueError("mismatched condition requires a mismatch input")
        return mismatch_input
    if condition == "empty":
        return ""
    raise ValueError(f"unknown condition {condition!r}")


def reconstruct_input_ids(
    query: str,
    program: str,
    encode: Callable[[str], list[int]],
    eos_id: int,
) -> tuple[list[int], int, list[int]]:
    """Return (input_ids, target_start, prefix_ids) exactly as the scorer builds
    them. Raises ScoringError on prefix-boundary mismatch (same policy as run)."""
    prefix = build_prefix(query)
    prefix_ids = encode(prefix)
    full_ids = encode(prefix + program)
    input_ids, target_start = build_scored_sequence(prefix_ids, full_ids, eos_id)
    return input_ids, target_start, prefix_ids


def crosscheck_pair(
    input_ids: list[int],
    target_start: int,
    eos_id: int,
    pair_rec: dict,
    token_recs: list[dict],
) -> list[str]:
    """Cross-check reconstructed ids against the ORIGINAL pair record and its
    per-token evidence. Returns a list of mismatch descriptions (empty == pass).

    Checks (no logprob recomputation — that would need the model; we verify the
    inputs the original run claims to have scored):
      - n_prefix_tokens / n_input_tokens / n_target_tokens match reconstruction
      - every original target token id equals input_ids at the same position
      - original positions are exactly target_start..L-1, pred_row == position-1
      - last target is the EOS id
    """
    errors: list[str] = []
    n_input = len(input_ids)
    if pair_rec.get("n_prefix_tokens") != target_start:
        errors.append(f"n_prefix_tokens {pair_rec.get('n_prefix_tokens')} != reconstructed {target_start}")
    if pair_rec.get("n_input_tokens") != n_input:
        errors.append(f"n_input_tokens {pair_rec.get('n_input_tokens')} != reconstructed {n_input}")
    if pair_rec.get("n_target_tokens") != n_input - target_start:
        errors.append(
            f"n_target_tokens {pair_rec.get('n_target_tokens')} != reconstructed {n_input - target_start}"
        )
    targets = input_ids[target_start:]
    if len(token_recs) != len(targets):
        errors.append(f"token evidence rows {len(token_recs)} != target count {len(targets)}")
        return errors
    for k, t in enumerate(token_recs):
        if t["token_id"] != targets[k]:
            errors.append(f"target[{k}] id {t['token_id']} != reconstructed {targets[k]}")
        if t["position"] != target_start + k:
            errors.append(f"target[{k}] position {t['position']} != {target_start + k}")
        if t["pred_row"] != t["position"] - 1:
            errors.append(f"target[{k}] pred_row {t['pred_row']} != position-1")
    if token_recs and token_recs[-1]["token_id"] != eos_id:
        errors.append(f"last target id {token_recs[-1]['token_id']} != eos {eos_id}")
    return errors
