"""Teacher-forced conditional scoring core (numpy, no torch required).

The numerically sensitive part is implemented here in float32 numpy so it can be
unit-tested locally without the NPU/torch stack. The remote runner converts
model logits with `logits.float().cpu().numpy()` and calls these functions.

Score definition (per task contract):
  input_ids  = tokenize(prefix + output, add_special_tokens=False) + [EOS]
  target set = positions target_start .. L-1 (response tokens + EOS)
  S_sum      = sum_i log p_theta(input_ids[i] | input_ids[<i])  for i in target set
             = raw conditional sequence log-prob of program+EOS given the query
  S_mean     = S_sum / |target set|   (diagnostic only, NOT a full probability)

Shift convention: logits row j (0-based) predicts token at position j+1, so the
prediction rows for the target set are logits[target_start-1 : L-1].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ScoringError(Exception):
    """Raised for hard contract violations (bad boundaries, shapes)."""


@dataclass
class TokenScore:
    """Per-token record for one scored target position."""

    position: int  # absolute position in input_ids
    pred_row: int  # logits row whose distribution produced this score (position-1)
    token_id: int
    logprob: float


@dataclass
class SequenceScore:
    n_target_tokens: int
    sum_logprob: float
    mean_logprob: float
    all_finite: bool
    tokens: list[TokenScore] = field(default_factory=list)


def log_softmax_f32(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax computed in float32 (task contract: FP32 log-softmax).

    logits: [..., V] array. Returns same shape, float32.
    """
    x = np.asarray(logits, dtype=np.float32)
    m = x.max(axis=-1, keepdims=True)
    z = x - m  # <= 0, so exp(z) in (0, 1]; sum <= V, no overflow in f32
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return (z - lse).astype(np.float32)


def gather_response_logprobs(
    logits: np.ndarray,
    input_ids: np.ndarray,
    target_start: int,
) -> SequenceScore:
    """Gather per-token log-probs for response+EOS targets from full-sequence logits.

    Args:
        logits: [L, V] full-sequence model logits (any dtype; cast to f32).
        input_ids: [L] int token ids, with EOS as the final token.
        target_start: index of the first response token (== len(prefix_ids)).

    Returns:
        SequenceScore with per-token records, sum and mean over target set.

    Raises:
        ScoringError: on shape/boundary violations.
    """
    logits = np.asarray(logits)
    input_ids = np.asarray(input_ids, dtype=np.int64)
    if logits.ndim != 2:
        raise ScoringError(f"logits must be [L, V], got shape {logits.shape}")
    L = int(input_ids.shape[0])
    if logits.shape[0] != L:
        raise ScoringError(f"logits length {logits.shape[0]} != input length {L}")
    if not (0 < target_start < L):
        raise ScoringError(f"target_start {target_start} not in (0, {L})")

    pred_rows = logits[target_start - 1 : L - 1]  # row j predicts token j+1
    targets = input_ids[target_start:]
    V = logits.shape[1]
    if targets.min() < 0 or targets.max() >= V:
        raise ScoringError(f"target token id out of vocab range [0, {V})")

    lsm = log_softmax_f32(pred_rows)
    tok_lp = lsm[np.arange(targets.shape[0]), targets]

    tokens = [
        TokenScore(
            position=target_start + k,
            pred_row=target_start - 1 + k,
            token_id=int(targets[k]),
            logprob=float(tok_lp[k]),
        )
        for k in range(targets.shape[0])
    ]
    all_finite = bool(np.isfinite(tok_lp).all()) and bool(np.isfinite(lsm).all())
    total = float(tok_lp.sum(dtype=np.float64))
    n = int(targets.shape[0])
    return SequenceScore(
        n_target_tokens=n,
        sum_logprob=total,
        mean_logprob=total / n,
        all_finite=all_finite,
        tokens=tokens,
    )


def check_prefix_boundary(prefix_ids: list[int], full_ids: list[int]) -> bool:
    """True iff separately tokenized prefix is a strict token prefix of the full
    sequence (without EOS). If False, the char-level prefix boundary does not map
    to a token boundary and the sample must be marked, not silently scored."""
    n = len(prefix_ids)
    return n < len(full_ids) and list(full_ids[:n]) == list(prefix_ids)


def build_scored_sequence(prefix_ids: list[int], full_ids: list[int], eos_id: int) -> tuple[list[int], int]:
    """Return (input_ids, target_start). Appends EOS at token level; verifies the
    prefix boundary. Raises ScoringError on boundary mismatch (caller records and
    skips the sample per contract)."""
    if eos_id is None:
        raise ScoringError("tokenizer has no eos_token_id")
    if not check_prefix_boundary(prefix_ids, full_ids):
        raise ScoringError("prefix tokenization is not a strict prefix of full tokenization")
    input_ids = list(full_ids) + [int(eos_id)]
    return input_ids, len(prefix_ids)
