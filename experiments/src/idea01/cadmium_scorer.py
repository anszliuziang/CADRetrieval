"""CADmium native JSON conditional scorer (local CPU implementation, M1a).

Task contract: experiments/plans/IDEA01_M1_CADMIUM_SCORER_2026-09-28.md

This module implements the *scoring definition* and *rejection conditions* for
using the frozen CADmium-3B (Qwen2.5-Coder-3B-Instruct + LoRA) as a conditional
scorer over candidate CAD model ``json_desc`` strings given a text query. It is
deliberately restricted to:

  - the tokenizer (CPU, no weights, no model forward),
  - a fixed system prompt whose SHA256 is verified against the contract,
  - the query text and the candidate ``json_desc`` text only.

It does NOT read qrels, audit files, raw CSVs, or CADmium ``annotation``. UID
handling is left to the outer logging / evaluation module.

Score definition (per contract item 3):
  prefix_ids = apply_chat_template([system, user], add_generation_prompt=True)
  full_ids   = apply_chat_template([system, user, assistant=json_desc],
                                   add_generation_prompt=False)
  target_set = full_ids[len(prefix_ids):]      # includes author template's
                                               # <|im_end|> AND trailing newline
  S_sum      = sum_i log p(y_i | q, y_<i)      # primary score
  S_mean     = S_sum / |target_set|            # diagnostic only

No extra EOS is appended (contrast idea01.scoring.build_scored_sequence, which
is for the CadQuery arm and adds EOS -- we must NOT call it here).

The numerically sensitive log-prob gathering reuses
``idea01.scoring.gather_response_logprobs``; this module wraps it with the
CADmium-specific boundary / context / ledger logic.

This file is numpy + tokenizer only; no torch required to import it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .scoring import ScoringError, gather_response_logprobs

# ---------------------------------------------------------------------------
# Fixed constants from the M1a contract / verified assets
# ---------------------------------------------------------------------------

#: Maximum training context length for CADmium (official max_length 4096).
MAX_TRAINING_CONTEXT = 4096

#: Expected SHA256 of the CADmium system message text (contract item in §"固定资产和输入").
EXPECTED_SYSTEM_MESSAGE_SHA256 = (
    "59d623d0c7ca29810eb80ab265891023c749402516701caade11e96a62a18ab3"
)

#: SHA256 of the verified CADmium adapter chat template, computed read-only
#: from ``.cache/cadmium_model/chat_template.jinja`` (== tokenizer.chat_template).
#: The contract requires the adapter tokenizer to match the verified base/adapter
#: chat template; comparing only numeric ids is not enough. This is the only
#: chat template permitted for official scoring.
EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)

#: Verified CADmium adapter tokenizer constants (read-only computed from
#: ``.cache/cadmium_model`` with transformers 4.53.1):
#:   len(tokenizer) == 151665, vocab_size == 151643,
#:   eos_token_id == im_end_token_id == 151645, pad_token_id == 151643,
#:   im_start_token_id == 151644.
EXPECTED_VOCAB_SIZE = 151643
EXPECTED_LEN_TOKENIZER = 151665
EXPECTED_EOS_TOKEN_ID = 151645
EXPECTED_PAD_TOKEN_ID = 151643
EXPECTED_IM_START_TOKEN_ID = 151644

#: Qwen2.5 special tokens used by the chat template. The assistant turn ends
#: with ``<|im_end|>`` followed by a literal ``\n`` (the template appends
#: ``'<|im_end|>' + '\n'``). EOS *is* ``<|im_end|>`` for this tokenizer, so the
#: template's own closing token already provides the sequence terminator; the
#: trailing newline is an additional template token that is part of the target.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# ---------------------------------------------------------------------------
# Reason codes (stable strings for the failure ledger)
# ---------------------------------------------------------------------------

#: candidate json_desc is empty / whitespace only.
REASON_EMPTY_TARGET = "empty_target"
#: separately tokenized prefix is NOT a strict token prefix of full sequence.
REASON_PREFIX_MISMATCH = "prefix_mismatch"
#: tokenizer version does not match the verified CADmium tokenizer.
REASON_TOKENIZER_MISMATCH = "tokenizer_mismatch"
#: full sequence length exceeds the 4096 training context; NOT truncated.
REASON_OUT_OF_TRAINING_CONTEXT = "out_of_training_context"
#: log-prob reduction produced non-finite values (nan/inf).
REASON_NONFINITE = "nonfinite_score"
#: tokenizer reported no EOS / im_end id (cannot validate terminator).
REASON_NO_IM_END = "no_im_end_token"
#: target does not end with the expected <|im_end|> + newline structure.
REASON_BAD_TERMINATOR = "bad_terminator"


@dataclass
class TokenizerFingerprint:
    """Identity fingerprint of a tokenizer for the ledger.

    The contract requires verifying the adapter tokenizer matches the verified
    base/adapter chat template, vocab length and special token ids -- comparing
    only numeric ids is not enough, since two tokenizers can share ids while
    rendering turns differently. ``matches`` is the boolean verdict.
    """

    vocab_size: int
    len_tokenizer: int
    eos_token_id: int
    im_end_token_id: int
    im_start_token_id: int
    pad_token_id: int
    chat_template_present: bool
    chat_template_sha256: str | None
    matches: bool


@dataclass
class ScoredPair:
    """Result of scoring one (query, candidate) pair.

    ``status`` is one of ``"scored"``, ``"rejected"``. ``reason`` is set when
    rejected. ``target_ids`` is the verbatim target token list (for audit);
    ``candidate_json_desc`` is stored verbatim so the outer module can persist
    UID linkage without the scorer reading any UID source.
    """

    query_text: str
    candidate_json_desc: str
    status: str
    reason: str | None = None
    prefix_len: int = 0
    full_len: int = 0
    target_len: int = 0
    sum_logprob: float | None = None
    mean_logprob: float | None = None
    all_finite: bool | None = None
    target_ids: list[int] = field(default_factory=list)
    within_training_context: bool = True


@dataclass
class ScoreLedger:
    """Aggregate counts over a batch of scored pairs."""

    n_total: int = 0
    n_scored: int = 0
    n_rejected: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    n_out_of_context: int = 0
    target_len_min: int | None = None
    target_len_max: int | None = None
    full_len_min: int | None = None
    full_len_max: int | None = None
    prefix_len_min: int | None = None
    prefix_len_max: int | None = None

    def add(self, r: ScoredPair) -> None:
        self.n_total += 1
        self.prefix_len_min = r.prefix_len if self.prefix_len_min is None else min(self.prefix_len_min, r.prefix_len)
        self.prefix_len_max = r.prefix_len if self.prefix_len_max is None else max(self.prefix_len_max, r.prefix_len)
        self.full_len_min = r.full_len if self.full_len_min is None else min(self.full_len_min, r.full_len)
        self.full_len_max = r.full_len if self.full_len_max is None else max(self.full_len_max, r.full_len)
        if r.status == "scored":
            self.n_scored += 1
            self.target_len_min = r.target_len if self.target_len_min is None else min(self.target_len_min, r.target_len)
            self.target_len_max = r.target_len if self.target_len_max is None else max(self.target_len_max, r.target_len)
        else:
            self.n_rejected += 1
            assert r.reason is not None
            self.by_reason[r.reason] = self.by_reason.get(r.reason, 0) + 1
            if r.reason == REASON_OUT_OF_TRAINING_CONTEXT:
                self.n_out_of_context += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_rejected": self.n_rejected,
            "by_reason": dict(self.by_reason),
            "n_out_of_context": self.n_out_of_context,
            "target_len_min": self.target_len_min,
            "target_len_max": self.target_len_max,
            "full_len_min": self.full_len_min,
            "full_len_max": self.full_len_max,
            "prefix_len_min": self.prefix_len_min,
            "prefix_len_max": self.prefix_len_max,
        }


# ---------------------------------------------------------------------------
# System message loading + SHA verification
# ---------------------------------------------------------------------------

def load_system_message(config_path: str | os.PathLike) -> str:
    """Load and SHA-verify the CADmium system message from the JSON config.

    The config file ``experiments/configs/cadmium_system_message.json`` holds
    the verbatim system message and its expected SHA256 (recorded by Codex from
    the original repo git SHA ``2b2a5e7...``). We recompute the SHA here and
    require it to equal the *code-fixed* ``EXPECTED_SYSTEM_MESSAGE_SHA256``
    constant -- the config field, if present, must also equal that constant.
    A mismatch on either side refuses to score, preventing a silent schema edit
    (or a config-only hash bump) from corrupting the prompt.

    Returns the system message text on success.
    """
    import hashlib

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    msg = cfg["system_message"]
    actual = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    if actual != EXPECTED_SYSTEM_MESSAGE_SHA256:
        raise ScoringError(
            f"cadmium system message SHA256 mismatch: got {actual}, expected "
            f"(code constant) {EXPECTED_SYSTEM_MESSAGE_SHA256}"
        )
    cfg_expected = cfg.get("system_message_sha256")
    if cfg_expected is not None and cfg_expected != EXPECTED_SYSTEM_MESSAGE_SHA256:
        raise ScoringError(
            f"cadmium config system_message_sha256 {cfg_expected} != code "
            f"constant {EXPECTED_SYSTEM_MESSAGE_SHA256}"
        )
    return msg


# ---------------------------------------------------------------------------
# Tokenizer verification
# ---------------------------------------------------------------------------

def fingerprint_tokenizer(tokenizer: Any) -> TokenizerFingerprint:
    """Fingerprint a tokenizer against the verified CADmium adapter tokenizer.

    Checks (contract item 1): chat template present, vocab length, EOS / im_end
    id equality, pad token present. The verified CADmium adapter tokenizer
    (cloned to .cache/cadmium_model) has vocab_size 151643 (len 151665 with
    added tokens), eos == im_end == 151645, pad == 151643.
    """
    import hashlib

    try:
        im_end_id = int(tokenizer.convert_tokens_to_ids(IM_END))
        im_start_id = int(tokenizer.convert_tokens_to_ids(IM_START))
    except (AttributeError, TypeError, ValueError):
        im_end_id = im_start_id = -1
    try:
        tokenizer_len = len(tokenizer)
    except (TypeError, AttributeError):
        tokenizer_len = -1
    eos_id = int(getattr(tokenizer, "eos_token_id", -1) or -1)
    pad_id = int(getattr(tokenizer, "pad_token_id", -1) or -1)
    vocab = int(getattr(tokenizer, "vocab_size", -1) or -1)
    chat_tmpl = getattr(tokenizer, "chat_template", None)
    present = bool(chat_tmpl)
    template_sha = hashlib.sha256(chat_tmpl.encode("utf-8")).hexdigest() if isinstance(chat_tmpl, str) else None
    matches = (
        present
        and template_sha == EXPECTED_CHAT_TEMPLATE_SHA256
        and vocab == EXPECTED_VOCAB_SIZE
        and tokenizer_len == EXPECTED_LEN_TOKENIZER
        and eos_id == EXPECTED_EOS_TOKEN_ID
        and im_end_id == EXPECTED_EOS_TOKEN_ID
        and im_start_id == EXPECTED_IM_START_TOKEN_ID
        and pad_id == EXPECTED_PAD_TOKEN_ID
    )
    return TokenizerFingerprint(
        vocab_size=vocab,
        len_tokenizer=tokenizer_len,
        eos_token_id=eos_id,
        im_end_token_id=im_end_id,
        im_start_token_id=im_start_id,
        pad_token_id=pad_id,
        chat_template_present=present,
        chat_template_sha256=template_sha,
        matches=matches,
    )


# ---------------------------------------------------------------------------
# Chat template construction (Qwen2.5 shape, restricted to CADmium usage)
# ---------------------------------------------------------------------------

def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    """Tokenize a chat using the tokenizer's own template; fail closed.

    We always pass ``tokenize=True`` and request ids. Special tokens are NOT
    added on top: the Qwen2.5 template emits the ``<|im_start|>`` / ``<|im_end|>``
    markers as literal text, and the BPE merges them into the registered special
    tokens; adding ``bos``/``eos`` separately would corrupt the boundary.
    """
    if not callable(getattr(tokenizer, "apply_chat_template", None)) or not getattr(tokenizer, "chat_template", None):
        raise ScoringError("CADmium tokenizer has no usable chat template")
    ids = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return [int(t) for t in ids]


# ---------------------------------------------------------------------------
# Core: build prefix / full target token lists with contract checks
# ---------------------------------------------------------------------------

@dataclass
class BuiltSequence:
    """Tokenized prefix/full for one pair plus the validated target slice."""

    prefix_ids: list[int]
    full_ids: list[int]
    target_ids: list[int]
    prefix_len: int
    full_len: int
    target_len: int


def build_cadmium_sequence(
    query_text: str,
    candidate_json_desc: str,
    system_message: str,
    tokenizer: Any,
) -> tuple[BuiltSequence | None, str | None]:
    """Build and validate the tokenized prefix/full sequences for one pair.

    Per contract item 1: construct prefix (add_generation_prompt=True) and full
    (add_generation_prompt=False with the assistant turn = candidate_json_desc),
    require the prefix to be a STRICT token prefix of the full sequence, and the
    target to be ``full_ids[len(prefix_ids):]`` including the template's
    ``<|im_end|>`` and the trailing newline token. No EOS is appended.

    Args:
        query_text: the natural-language query (from the verified M0 query.jsonl
            ``text`` field, only ``strip()``-ed by the caller). The scorer does
            not open the query file.
        candidate_json_desc: the verbatim CAD ``json_desc`` string for the
            candidate. Stored verbatim; the scorer does not parse it as JSON.
        system_message: the SHA-verified CADmium system message text.
        tokenizer: a Qwen2.5-compatible tokenizer (real or fake).

    Returns:
        (BuiltSequence, None) on success, or (None, reason_code) when the pair
        must be rejected. Rejection reasons never raise -- the caller records
        them in the ledger.
    """
    if candidate_json_desc is None or not str(candidate_json_desc).strip():
        return None, REASON_EMPTY_TARGET

    messages_prefix = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query_text},
    ]
    messages_full = messages_prefix + [
        {"role": "assistant", "content": candidate_json_desc},
    ]

    prefix_ids = _apply_chat_template(tokenizer, messages_prefix, add_generation_prompt=True)
    full_ids = _apply_chat_template(tokenizer, messages_full, add_generation_prompt=False)

    # Strict token-prefix check (contract item 1).
    n = len(prefix_ids)
    if not (n < len(full_ids) and list(full_ids[:n]) == list(prefix_ids)):
        return None, REASON_PREFIX_MISMATCH

    target_ids = list(full_ids[n:])

    # Terminator check: target must end with <|im_end|> then a newline token.
    # The Qwen2.5 template appends '<|im_end|>' + '\n' after the assistant
    # content. We verify im_end is present at target[-2] and a newline-style
    # token at target[-1]. We do NOT require a specific newline id (the BPE
    # may merge); we require the last token to decode to whitespace containing
    # a newline, and the second-to-last to be the im_end id.
    im_end_id = int(tokenizer.convert_tokens_to_ids(IM_END)) if hasattr(tokenizer, "convert_tokens_to_ids") else -1
    if im_end_id is None or im_end_id < 0:
        return None, REASON_NO_IM_END
    if len(target_ids) < 2 or target_ids[-2] != im_end_id:
        return None, REASON_BAD_TERMINATOR
    # verify trailing token decodes to something newline-bearing or empty
    try:
        tail_str = tokenizer.decode([target_ids[-1]])
    except Exception:
        tail_str = ""
    if "\n" not in tail_str and tail_str != "":
        # Some tokenizers decode the raw newline token to "" for special
        # handling; accept only if it is whitespace-only.
        if tail_str.strip() != "":
            return None, REASON_BAD_TERMINATOR

    built = BuiltSequence(
        prefix_ids=prefix_ids,
        full_ids=full_ids,
        target_ids=target_ids,
        prefix_len=n,
        full_len=len(full_ids),
        target_len=len(target_ids),
    )
    return built, None


# ---------------------------------------------------------------------------
# Scoring wrapper over raw model logits
# ---------------------------------------------------------------------------

def _score_pair_from_logits_unverified(
    query_text: str,
    candidate_json_desc: str,
    system_message: str,
    tokenizer: Any,
    logits: np.ndarray,
    *,
    input_ids: np.ndarray | None = None,
) -> ScoredPair:
    """Score one pair after tokenizer verification (or in synthetic CPU tests).

    The public entry point verifies tokenizer identity before calling this
    math core. It rebuilds the token sequences (cheap, CPU) to recover
    ``target_start = len(prefix_ids)`` and then delegates the log-prob
    gathering to :func:`idea01.scoring.gather_response_logprobs`.

    Per contract: padding (if any) is the runner's concern and must not enter
    ``logits``/``input_ids`` here; batch=1 and batch>1 must give identical
    per-pair scores, so the runner slices to the unpadded sequence before
    calling this. We do NOT append EOS.

    Args:
        logits: [L, V] float array of raw model logits for the full sequence
            (unpadded). Cast to float32 internally.
        input_ids: optional [L] int array. If None, we re-tokenize the full
            sequence and assert it matches ``logits`` length.

    Returns:
        ScoredPair with status "scored" or "rejected" plus reason.
    """
    built, reason = build_cadmium_sequence(query_text, candidate_json_desc, system_message, tokenizer)
    base = ScoredPair(
        query_text=query_text,
        candidate_json_desc=candidate_json_desc,
        status="rejected",
        reason=reason,
    )
    if built is None:
        return base

    base.prefix_len = built.prefix_len
    base.full_len = built.full_len
    base.target_len = built.target_len
    base.target_ids = built.target_ids

    if built.full_len > MAX_TRAINING_CONTEXT:
        base.reason = REASON_OUT_OF_TRAINING_CONTEXT
        base.within_training_context = False
        return base

    if input_ids is None:
        input_ids_arr = np.asarray(built.full_ids, dtype=np.int64)
    else:
        input_ids_arr = np.asarray(input_ids, dtype=np.int64)

    logits_arr = np.asarray(logits)
    if logits_arr.ndim != 2:
        base.reason = "logits_not_2d"
        return base
    if input_ids_arr.ndim != 1:
        base.reason = "input_ids_not_1d"
        return base
    if logits_arr.shape[0] != input_ids_arr.shape[0]:
        base.reason = "logits_length_mismatch"
        return base
    # A model input must match the sequence whose target boundary was built.
    if not np.array_equal(input_ids_arr, np.asarray(built.full_ids, dtype=np.int64)):
        base.reason = "input_ids_mismatch"
        return base

    try:
        seq = gather_response_logprobs(logits_arr, input_ids_arr, target_start=built.prefix_len)
    except ScoringError as e:
        base.reason = f"scoring_error:{e}"
        return base

    if not seq.all_finite:
        base.reason = REASON_NONFINITE
        return base

    base.status = "scored"
    base.reason = None
    base.sum_logprob = float(seq.sum_logprob)
    base.mean_logprob = float(seq.mean_logprob)
    base.all_finite = True
    return base


def score_pair_from_logits(
    query_text: str,
    candidate_json_desc: str,
    system_message: str,
    tokenizer: Any,
    logits: np.ndarray,
    *,
    input_ids: np.ndarray | None = None,
) -> ScoredPair:
    """Official CADmium entry point; reject any tokenizer mismatch.

    A runner should check ``fingerprint_tokenizer(tokenizer).matches`` before
    model forward to avoid wasting compute; this check also guards the score.
    """
    if not fingerprint_tokenizer(tokenizer).matches:
        return ScoredPair(
            query_text=query_text,
            candidate_json_desc=candidate_json_desc,
            status="rejected",
            reason=REASON_TOKENIZER_MISMATCH,
        )
    return _score_pair_from_logits_unverified(
        query_text,
        candidate_json_desc,
        system_message,
        tokenizer,
        logits,
        input_ids=input_ids,
    )


# ---------------------------------------------------------------------------
# Batch consistency helper
# ---------------------------------------------------------------------------

def score_batch_consistency(
    pair_results_single: Sequence[ScoredPair],
    pair_results_batch: Sequence[ScoredPair],
) -> bool:
    """Assert batch=1 and batch>1 produce identical per-pair primary scores.

    The contract requires that padding (used only for batching) does not change
    per-pair scores. The runner is responsible for actually running both modes;
    this helper compares the returned ScoredPair objects. Returns True iff all
    scored pairs match on (status, sum_logprob) within float tolerance.
    """
    if len(pair_results_single) != len(pair_results_batch):
        return False
    for a, b in zip(pair_results_single, pair_results_batch):
        if a.status != b.status:
            return False
        if a.status == "scored":
            if a.sum_logprob is None or b.sum_logprob is None:
                return False
            if not np.isclose(a.sum_logprob, b.sum_logprob, rtol=1e-5, atol=1e-6):
                return False
    return True


__all__ = [
    "MAX_TRAINING_CONTEXT",
    "EXPECTED_SYSTEM_MESSAGE_SHA256",
    "REASON_EMPTY_TARGET",
    "REASON_PREFIX_MISMATCH",
    "REASON_TOKENIZER_MISMATCH",
    "REASON_OUT_OF_TRAINING_CONTEXT",
    "REASON_NONFINITE",
    "REASON_NO_IM_END",
    "REASON_BAD_TERMINATOR",
    "TokenizerFingerprint",
    "ScoredPair",
    "ScoreLedger",
    "load_system_message",
    "fingerprint_tokenizer",
    "build_cadmium_sequence",
    "score_pair_from_logits",
    "score_batch_consistency",
]
