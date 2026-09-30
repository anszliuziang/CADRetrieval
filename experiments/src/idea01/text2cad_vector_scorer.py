"""Text2CAD two-head factorized conditional scorer -- CPU numeric core (M1c stage A).

Task contract : experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md (G0.1-G0.4)
Source audit  : experiments/submissions/idea01_text2cad_benchmark_20260928/TEXT2CAD_SOURCE_AUDIT.md
Official source (read-only mirror, Git SHA 669a81472ede2269837c22a4d00070eaac360c82,
license CC BY-NC-SA 4.0): token/vocabulary constants in ``CadSeqProc/utility/macro.py``
are reproduced here with attribution; the scoring math is derived from
``Cad_VLM/train.py``, ``Cad_VLM/models/loss.py``, ``Cad_VLM/models/decoder.py`` and
``CadSeqProc/cad_sequence.py`` as documented in the audit. This module is an
adaptation for research evaluation and remains under CC BY-NC-SA 4.0
(attribution: Mohammad Sadil Khan et al., Text2CAD, NeurIPS 2024).

Scope restrictions (stage A):
  * pure CPU numeric core over *given* logits and candidate ``vec``/``mask`` --
    no model weights are loaded and no model forward is run here;
  * inputs are limited to the query text (opaque string, used only by
    :func:`dedup_texts` / :class:`ScoreRequest`) and the candidate
    ``cad_vec`` / ``key_padding_mask`` (+ optional ``flag_vec``/``index_vec``);
  * this module never reads qrels, CSVs, UID mappings or audit files, never
    opens files, and never sees positive-example UIDs;
  * the training loss (``CrossEntropyLoss(label_smoothing=0.1)``, masked mean,
    averaged over the two heads) is NOT used and is NOT a score.

Score definition (exact; evidence in the audit document)
--------------------------------------------------------
Teacher forcing identical to official training (train.py:247-287): with a
padded candidate of ``N`` rows (official data always has ``N == 272``),

  input  = cad_vec[0:N-1]   (flag_vec/index_vec/key_padding_mask sliced alike)
  target = cad_vec[1:N]     i.e. labels (x_t, y_t) for t = 1..N-1
  valid  = target != 0      per (position, head); identical to the inverted
                            official ``key_padding_mask`` (json2vec.py:262)
                            that train.py:285 hands to the loss

Given raw model logits ``pred`` of shape ``(N-1, 2, 267)`` (per candidate;
the two output heads of decoder.py:100-105 over the shared 267-class
vocabulary):

  lp[t, h]  = log_softmax_fp32(pred[t, h, :])[target[t, h]]     (FP32 kernel)
  score_x   = sum_{t: valid[t,0]} lp[t,0]   (float64 accumulation)
  score_y   = sum_{t: valid[t,1]} lp[t,1]   (float64 accumulation)
  score     = score_x + score_y

Which (position, head) factors are scored follows the official mask exactly:

  token row of cad_vec          x label   y label   x scored   y scored
  ----------------------------  --------  --------  ---------  ---------
  initial START [1, 0] (t = 0)  1         0         n/a (never a target)
  curve coordinate [x+11,y+11]  11..266   11..266   yes        yes
  END_CURVE [5, 0]              5         0         yes        no
  END_LOOP  [4, 0]              4         0         yes        no
  END_FACE  [3, 0]              3         0         yes        no
  END_SKETCH [2, 0]             2         0         yes        no
  extrusion param [v+11, 0]     11..266   0         yes        no
  extrusion boolean [b+7, 0]    7..10     0         yes        no
  END_EXTRUSION [6, 0]          6         0         yes        no
  terminal START [1, 0]         1         0         yes        no
  PADDING [0, 0]                0         0         no         no

Termination: there is NO distinct EOS token. A sequence ends with the token
``[START, 0]`` (cad_sequence.py:260); its x factor (label 1) is scored, its y
factor is masked out. Trailing PADDING rows contribute nothing.

Why this is NOT a joint log-probability (audit section "评分数学定位"):
  1. the two heads are independent linear maps sharing one 267-way softmax
     vocabulary, i.e. p(x_t, y_t | h_<t, q) is factorized as
     p_x(x_t | .) * p_y(y_t | .) -- a conditional-independence factorization;
  2. even inside that factorization, factors whose labels are structurally 0
     (y of every non-coordinate token, both columns of PADDING) are excluded:
     they were never supervised (masked out of the training loss) and their
     "labels" are constants of the serialization, not random variables;
  3. the parameters were trained with label_smoothing=0.1 on a masked *mean*
     per head, averaged over heads -- not on this sum -- so the logits are
     not calibrated to it.
  The quantity is therefore reported strictly as a **two-head factorized
  conditional log-probability-sum ranking score**, with per-head sums and
  valid counts stored for every pair. Mean/length-normalized variants are
  diagnostics only, never the primary score.

Truncation guards: official candidates are padded to exactly ``N == 272``
(overlong sequences cannot exist in ``cad_seq.zip`` -- ``add_padding`` raises
on negative pad, utils.py:847-850). This core rejects ``N > 272`` and
``N < 2``. The text side is *not* handled here: the pinned BERT tokenizer
truncates silently at 512 tokens (text_embed.py:36-42), so the stage-B runner
must verify query token length <= 512 with the pinned tokenizer before any
forward; this module only checks that text is a non-empty ``str``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# Constants reproduced from the official Text2CAD source (CC BY-NC-SA 4.0).
# Source: CadSeqProc/utility/macro.py @ 669a81472ede2269837c22a4d00070eaac360c82
# ---------------------------------------------------------------------------

#: macro.py:8-9 -- special token rows of the CAD command vocabulary.
END_TOKEN = [
    "PADDING",
    "START",
    "END_SKETCH",
    "END_FACE",
    "END_LOOP",
    "END_CURVE",
    "END_EXTRUSION",
]

PADDING_INDEX = END_TOKEN.index("PADDING")  # 0
START_INDEX = END_TOKEN.index("START")  # 1
END_SKETCH_INDEX = END_TOKEN.index("END_SKETCH")  # 2
END_FACE_INDEX = END_TOKEN.index("END_FACE")  # 3
END_LOOP_INDEX = END_TOKEN.index("END_LOOP")  # 4
END_CURVE_INDEX = END_TOKEN.index("END_CURVE")  # 5
END_EXTRUSION_INDEX = END_TOKEN.index("END_EXTRUSION")  # 6

#: macro.py:11-12 -- 7 special rows + 4 boolean rows precede quantized values.
END_PAD = 7
BOOLEAN_PAD = 4

#: macro.py:6 -- 8-bit quantization of coordinates/extrusion parameters.
N_BIT = 8

#: macro.py:14 -- padded CAD sequence length used by all official assets.
MAX_CAD_SEQUENCE_LENGTH = 272

#: macro.py:40-44 -- per-head softmax vocabulary: 7 + 4 + 2**8 = 267.
ONE_HOT_SIZE = END_PAD + BOOLEAN_PAD + 2**N_BIT  # 267
FLAG_SIZE = 12  # ONE_EXT_SEQ_LENGTH(10) + 2 (sketch / padding)
INDEX_SIZE = 11  # MAX_EXTRUSION(10) + 1 (padding)

#: Smallest label of a quantized coordinate/parameter value (value + 11).
QUANTIZED_VALUE_OFFSET = END_PAD + BOOLEAN_PAD  # 11
#: Boolean labels occupy END_PAD .. END_PAD + BOOLEAN_PAD - 1 (7..10).
BOOLEAN_VALUE_OFFSET = END_PAD  # 7

SCORE_NAME = "two_head_factorized_logprob_sum"
SCORE_SPEC_VERSION = "text2cad-vector-score-v1"

# ---------------------------------------------------------------------------
# Rejection reason codes (stable strings; asserted in tests and ledgers)
# ---------------------------------------------------------------------------

REASON_BAD_SHAPE = "bad_shape"
REASON_BAD_DTYPE = "bad_dtype"
REASON_LENGTH_OVER_MAX = "length_over_max"
REASON_LENGTH_TOO_SHORT = "length_too_short"
REASON_LABEL_OUT_OF_RANGE = "label_out_of_range"
REASON_BAD_START_TOKEN = "bad_start_token"
REASON_BAD_TERMINAL_TOKEN = "bad_terminal_token"
REASON_NON_TAIL_PADDING = "non_tail_padding"
REASON_INCONSISTENT_ROW = "inconsistent_row"
REASON_MASK_MISMATCH = "mask_mismatch"
REASON_EMPTY_TARGET = "empty_target"
REASON_NONFINITE_LOGITS = "nonfinite_logits"
REASON_VOCAB_MISMATCH = "vocab_mismatch"
REASON_FLAG_MISMATCH = "flag_mismatch"
REASON_INDEX_MISMATCH = "index_mismatch"
REASON_BAD_TEXT = "bad_text"


class ScorerInputError(ValueError):
    """Raised when a candidate vector/mask/logits violate the score contract.

    The ``reason`` attribute is one of the ``REASON_*`` constants above so
    callers can ledger rejections without parsing messages.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CadVecDiagnostics:
    """Structural facts about one validated candidate vector."""

    n_tokens: int  # padded length N (<= MAX_CAD_SEQUENCE_LENGTH)
    n_real_tokens: int  # unpadded length L (2 <= L <= N)
    n_target_positions: int  # N - 1 rows that are prediction targets
    n_valid_x: int  # target positions whose x label is scored (== L - 1)
    n_valid_y: int  # target positions whose y label is scored
    n_valid_total: int  # n_valid_x + n_valid_y


def _as_int_tensor(value: Any, name: str) -> torch.Tensor:
    """Convert array-likes to an integer torch tensor or reject."""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ScorerInputError(REASON_BAD_DTYPE, f"{name}: {exc}") from exc
    if tensor.dtype == torch.bool or not tensor.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ScorerInputError(
            REASON_BAD_DTYPE, f"{name}: expected integer tensor, got {tensor.dtype}"
        )
    return tensor


def _as_bool_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ScorerInputError(REASON_BAD_DTYPE, f"{name}: {exc}") from exc
    if tensor.dtype != torch.bool:
        raise ScorerInputError(
            REASON_BAD_DTYPE, f"{name}: expected bool tensor, got {tensor.dtype}"
        )
    return tensor


def validate_cad_vec(
    cad_vec: Any,
    key_padding_mask: Optional[Any] = None,
    flag_vec: Optional[Any] = None,
    index_vec: Optional[Any] = None,
) -> CadVecDiagnostics:
    """Validate one official-format candidate and return its diagnostics.

    ``cad_vec`` must be an integer ``(N, 2)`` tensor with ``2 <= N <= 272``,
    labels in ``[0, 266]``, first row exactly ``[START, 0] == [1, 0]``, zero
    rows only as a tail run, and the last non-zero row exactly ``[1, 0]``
    (the terminal START token). ``key_padding_mask``, when given, must be the
    official bool ``(N, 2)`` mask and equal ``cad_vec == 0`` elementwise
    (json2vec.py:262); any mismatch is rejected because score validity rests
    on mask/vector self-consistency (task stop condition). ``flag_vec`` /
    ``index_vec`` are optional; when given this function checks their shape,
    range, and padding values from ``CADSequence.to_vec``
    (cad_sequence.py:242-274): integer length-N vectors, flag in [0, 11],
    index in [0, 10], flag == 11 and index == max(real index) + 1 on padding
    rows. It does not reconstruct every non-padding flag/index value; the
    runner must pass the pinned official vectors unchanged.

    Never mutates its inputs.
    """
    vec = _as_int_tensor(cad_vec, "cad_vec")
    if vec.ndim != 2 or vec.shape[1] != 2:
        raise ScorerInputError(
            REASON_BAD_SHAPE, f"cad_vec: expected (N, 2), got {tuple(vec.shape)}"
        )
    n_tokens = vec.shape[0]
    if n_tokens > MAX_CAD_SEQUENCE_LENGTH:
        raise ScorerInputError(
            REASON_LENGTH_OVER_MAX,
            f"cad_vec: {n_tokens} rows > {MAX_CAD_SEQUENCE_LENGTH}",
        )
    if n_tokens < 2:
        raise ScorerInputError(
            REASON_LENGTH_TOO_SHORT, f"cad_vec: {n_tokens} rows < 2 (no target row)"
        )
    if int(vec.min()) < 0 or int(vec.max()) > ONE_HOT_SIZE - 1:
        raise ScorerInputError(
            REASON_LABEL_OUT_OF_RANGE,
            f"cad_vec: labels must be in [0, {ONE_HOT_SIZE - 1}], "
            f"got [{int(vec.min())}, {int(vec.max())}]",
        )
    # A row is either fully zero (PADDING) or has a non-zero x column; the
    # official construction never emits x == 0 with y != 0.
    row_x_zero = vec[:, 0] == PADDING_INDEX
    row_y_zero = vec[:, 1] == PADDING_INDEX
    if bool((row_x_zero & ~row_y_zero).any()):
        raise ScorerInputError(
            REASON_INCONSISTENT_ROW,
            "cad_vec: rows with x == PADDING but y != PADDING are not "
            "produced by the official tokenizer",
        )
    row_is_pad = row_x_zero & row_y_zero
    # Padding must be a tail run: no real row after the first padding row.
    pad_positions = row_is_pad.nonzero(as_tuple=True)[0]
    n_real_tokens = int(pad_positions[0]) if pad_positions.numel() else n_tokens
    if pad_positions.numel() and bool((~row_is_pad[n_real_tokens:]).any()):
        raise ScorerInputError(
            REASON_NON_TAIL_PADDING,
            "cad_vec: PADDING rows must form a tail run",
        )
    if n_real_tokens < 2:
        raise ScorerInputError(
            REASON_EMPTY_TARGET,
            "cad_vec: fewer than 2 real rows -> no scorable target",
        )
    if int(vec[0, 0]) != START_INDEX or int(vec[0, 1]) != PADDING_INDEX:
        raise ScorerInputError(
            REASON_BAD_START_TOKEN,
            f"cad_vec: first row must be [START, 0] == [1, 0], "
            f"got {vec[0].tolist()}",
        )
    terminal = vec[n_real_tokens - 1]
    if int(terminal[0]) != START_INDEX or int(terminal[1]) != PADDING_INDEX:
        raise ScorerInputError(
            REASON_BAD_TERMINAL_TOKEN,
            f"cad_vec: last real row must be the terminal [START, 0], "
            "got {t} at row {r}".format(t=terminal.tolist(), r=n_real_tokens - 1),
        )
    if key_padding_mask is not None:
        mask = _as_bool_tensor(key_padding_mask, "key_padding_mask")
        if mask.shape != vec.shape:
            raise ScorerInputError(
                REASON_BAD_SHAPE,
                f"key_padding_mask: expected {tuple(vec.shape)}, "
                f"got {tuple(mask.shape)}",
            )
        if not bool((mask == (vec == PADDING_INDEX)).all()):
            raise ScorerInputError(
                REASON_MASK_MISMATCH,
                "key_padding_mask != (cad_vec == PADDING); candidate vector "
                "and mask are not self-consistent",
            )
    if flag_vec is not None:
        flags = _as_int_tensor(flag_vec, "flag_vec")
        if flags.ndim != 1 or flags.shape[0] != n_tokens:
            raise ScorerInputError(
                REASON_BAD_SHAPE,
                f"flag_vec: expected ({n_tokens},), got {tuple(flags.shape)}",
            )
        if int(flags.min()) < 0 or int(flags.max()) > FLAG_SIZE - 1:
            raise ScorerInputError(
                REASON_FLAG_MISMATCH,
                f"flag_vec: values must be in [0, {FLAG_SIZE - 1}]",
            )
        if n_real_tokens < n_tokens and not bool(
            (flags[n_real_tokens:] == FLAG_SIZE - 1).all()
        ):
            raise ScorerInputError(
                REASON_FLAG_MISMATCH,
                "flag_vec: padding rows must carry flag 11 (to_vec line 269)",
            )
    if index_vec is not None:
        indices = _as_int_tensor(index_vec, "index_vec")
        if indices.ndim != 1 or indices.shape[0] != n_tokens:
            raise ScorerInputError(
                REASON_BAD_SHAPE,
                f"index_vec: expected ({n_tokens},), got {tuple(indices.shape)}",
            )
        if int(indices.min()) < 0 or int(indices.max()) > INDEX_SIZE - 1:
            raise ScorerInputError(
                REASON_INDEX_MISMATCH,
                f"index_vec: values must be in [0, {INDEX_SIZE - 1}]",
            )
        if n_real_tokens < n_tokens:
            expected_pad_index = int(indices[:n_real_tokens].max()) + 1
            if expected_pad_index > INDEX_SIZE - 1 or not bool(
                (indices[n_real_tokens:] == expected_pad_index).all()
            ):
                raise ScorerInputError(
                    REASON_INDEX_MISMATCH,
                    "index_vec: padding rows must carry max(real index) + 1 "
                    "(to_vec line 270)",
                )
    target = vec[1:]
    n_valid_x = int((target[:, 0] != PADDING_INDEX).sum())
    n_valid_y = int((target[:, 1] != PADDING_INDEX).sum())
    if n_valid_x + n_valid_y == 0:
        raise ScorerInputError(
            REASON_EMPTY_TARGET, "cad_vec: no valid (position, head) label"
        )
    return CadVecDiagnostics(
        n_tokens=n_tokens,
        n_real_tokens=n_real_tokens,
        n_target_positions=n_tokens - 1,
        n_valid_x=n_valid_x,
        n_valid_y=n_valid_y,
        n_valid_total=n_valid_x + n_valid_y,
    )


# ---------------------------------------------------------------------------
# Scoring core
# ---------------------------------------------------------------------------


def _score_single(
    logits: torch.Tensor, vec: torch.Tensor, diag: CadVecDiagnostics
) -> dict:
    """Score one candidate: logits (N-1, 2, V) FP32-castable, vec (N, 2)."""
    target = vec[1:].to(torch.long)  # (N-1, 2), new tensor
    valid = target != PADDING_INDEX  # (N-1, 2) bool, == official ~mask
    # FP32 log_softmax over the shared 267-class vocabulary (never bf16/fp16
    # accumulation inside the kernel).
    log_probs = torch.nn.functional.log_softmax(logits.to(torch.float32), dim=-1)
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (N-1, 2)
    gathered64 = gathered.to(torch.float64)
    valid_x = valid[:, 0]
    valid_y = valid[:, 1]
    score_x = float(gathered64[:, 0][valid_x].sum())
    score_y = float(gathered64[:, 1][valid_y].sum())
    n_x = int(valid_x.sum())
    n_y = int(valid_y.sum())
    n_total = n_x + n_y
    if n_total == 0:  # defensive; validate_cad_vec already rejects this
        raise ScorerInputError(REASON_EMPTY_TARGET, "no valid label after mask")
    score_total = score_x + score_y
    return {
        "score_name": SCORE_NAME,
        "spec_version": SCORE_SPEC_VERSION,
        "score_x": score_x,
        "score_y": score_y,
        "score_total": score_total,
        "n_valid_x": n_x,
        "n_valid_y": n_y,
        "n_valid_total": n_total,
        "valid_seq_len": diag.n_real_tokens - 1,
        "n_real_tokens": diag.n_real_tokens,
        "mean_x": score_x / n_x if n_x else None,
        "mean_y": score_y / n_y if n_y else None,
        "mean_total": score_total / n_total,
    }


def two_head_factorized_score(
    logits: Any,
    cad_vec: Any,
    key_padding_mask: Optional[Any] = None,
) -> "dict | list[dict]":
    """Compute the two-head factorized conditional score from raw logits.

    Args:
        logits: raw model output, shape ``(N-1, 2, 267)`` for one candidate or
            ``(B, N-1, 2, 267)`` for a batch. Any floating dtype is accepted
            and cast to FP32 before ``log_softmax``; NaN/Inf is rejected.
            ``N-1`` must equal ``cad_vec.shape[-2] - 1``.
        cad_vec: integer candidate labels, ``(N, 2)`` or ``(B, N, 2)``,
            official padded format (see :func:`validate_cad_vec`).
        key_padding_mask: optional official full-length bool mask
            ``(N, 2)`` / ``(B, N, 2)`` (True = PADDING). When given it must
            equal ``cad_vec == 0`` elementwise; the score then derives the
            valid mask itself. Never pass a shifted/pre-inverted mask.

    Returns:
        One result dict per candidate (a single dict for unbatched input)
        with the exact keys asserted by the tests: ``score_name``,
        ``spec_version``, ``score_x``, ``score_y``, ``score_total``,
        ``n_valid_x``, ``n_valid_y``, ``n_valid_total``, ``valid_seq_len``,
        ``n_real_tokens``, ``mean_x``, ``mean_y``, ``mean_total``.
        Scores are Python floats accumulated in float64 from FP32 per-token
        log-probabilities. Mean values are diagnostics only.

    Raises:
        ScorerInputError: any contract violation (illegal labels, all-empty
            target, length over 272, mask/vector mismatch, non-finite or
            wrongly shaped logits, ...). Inputs are never mutated.
    """
    if not isinstance(logits, torch.Tensor):
        try:
            logits = torch.as_tensor(logits)
        except Exception as exc:  # pragma: no cover - defensive
            raise ScorerInputError(REASON_BAD_DTYPE, f"logits: {exc}") from exc
    if not logits.dtype.is_floating_point:
        raise ScorerInputError(
            REASON_BAD_DTYPE, f"logits: expected floating tensor, got {logits.dtype}"
        )
    batched = logits.ndim == 4
    if logits.ndim not in (3, 4):
        raise ScorerInputError(
            REASON_BAD_SHAPE,
            f"logits: expected (N-1, 2, V) or (B, N-1, 2, V), "
            f"got {tuple(logits.shape)}",
        )
    if logits.shape[-2] != 2:
        raise ScorerInputError(
            REASON_BAD_SHAPE,
            f"logits: head dim must be 2, got shape {tuple(logits.shape)}",
        )
    if logits.shape[-1] != ONE_HOT_SIZE:
        raise ScorerInputError(
            REASON_VOCAB_MISMATCH,
            f"logits: vocab dim must be {ONE_HOT_SIZE}, got {logits.shape[-1]}",
        )
    if not bool(torch.isfinite(logits).all()):
        raise ScorerInputError(REASON_NONFINITE_LOGITS, "logits contain NaN/Inf")

    vec = _as_int_tensor(cad_vec, "cad_vec")
    if vec.ndim != (3 if batched else 2):
        raise ScorerInputError(
            REASON_BAD_SHAPE,
            f"cad_vec/logits batch mismatch: {tuple(vec.shape)} vs "
            f"{tuple(logits.shape)}",
        )
    if batched and vec.shape[0] != logits.shape[0]:
        raise ScorerInputError(
            REASON_BAD_SHAPE,
            f"batch size mismatch: {vec.shape[0]} vs {logits.shape[0]}",
        )
    if vec.shape[-2] != logits.shape[-3] + 1:
        raise ScorerInputError(
            REASON_BAD_SHAPE,
            f"cad_vec has {vec.shape[-2]} rows but logits predict "
            f"{logits.shape[-3]} steps (need exactly one less)",
        )
    if key_padding_mask is not None:
        mask = _as_bool_tensor(key_padding_mask, "key_padding_mask")
        if mask.shape != vec.shape:
            raise ScorerInputError(
                REASON_BAD_SHAPE,
                f"key_padding_mask: expected {tuple(vec.shape)}, "
                f"got {tuple(mask.shape)}",
            )
    else:
        mask = None

    results = []
    vecs = vec if batched else vec.unsqueeze(0)
    masks = mask if (mask is not None and batched) else (
        mask.unsqueeze(0) if mask is not None else [None] * vecs.shape[0]
    )
    logits_b = logits if batched else logits.unsqueeze(0)
    for i in range(vecs.shape[0]):
        diag = validate_cad_vec(vecs[i], None if masks is None else masks[i])
        results.append(_score_single(logits_b[i], vecs[i], diag))
    return results if batched else results[0]


# ---------------------------------------------------------------------------
# Query-text helpers (text is opaque; never transformed, never tied to UIDs)
# ---------------------------------------------------------------------------


def dedup_texts(texts: Sequence[str]) -> "tuple[list[str], list[int]]":
    """Deduplicate query texts by exact string equality, preserving order.

    Returns ``(unique_texts, mapping)`` where ``mapping[i]`` is the index of
    ``texts[i]`` inside ``unique_texts``. Texts are compared verbatim -- the
    protocol's ``strip()`` normalization happened upstream when the frozen
    query manifest was built; this function must not re-normalize, so that a
    later drift in preprocessing cannot silently change score identity.
    """
    if isinstance(texts, (str, bytes)):
        raise ScorerInputError(REASON_BAD_TEXT, "texts must be a sequence of str")
    seen: dict[str, int] = {}
    unique: list[str] = []
    mapping: list[int] = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            raise ScorerInputError(
                REASON_BAD_TEXT, f"query text must be a non-empty str, got {text!r}"
            )
        index = seen.get(text)
        if index is None:
            index = len(unique)
            seen[text] = index
            unique.append(text)
        mapping.append(index)
    return unique, mapping


@dataclass(frozen=True)
class ScoreRequest:
    """One (query, candidate) scoring request -- the complete input schema.

    The scoring path accepts exactly these fields and nothing else: no UID,
    no qrels, no candidate metadata. Constructing with extra fields (e.g.
    ``uid=...``) is a ``TypeError`` by dataclass semantics; ``validate``
    applies the same candidate checks as :func:`validate_cad_vec`.
    """

    query_text: str
    cad_vec: Any
    key_padding_mask: Optional[Any] = None
    flag_vec: Optional[Any] = None
    index_vec: Optional[Any] = None

    def validate(self) -> CadVecDiagnostics:
        if not isinstance(self.query_text, str) or len(self.query_text) == 0:
            raise ScorerInputError(
                REASON_BAD_TEXT,
                f"query_text must be a non-empty str, got {self.query_text!r}",
            )
        return validate_cad_vec(
            self.cad_vec,
            self.key_padding_mask,
            self.flag_vec,
            self.index_vec,
        )
