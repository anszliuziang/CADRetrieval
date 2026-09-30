"""Unit tests for the Text2CAD two-head factorized vector scorer (M1c stage A).

Covers contract G0.3/G0.4 items with synthetic logits and masks only (no
weights, no model forward, no official assets required):

  * teacher-forcing shift and per-(position, head) valid mask;
  * terminal START handling (x scored, y masked) and padding exclusion;
  * invariance of the score to logits at masked positions;
  * batch == single numerical equality;
  * FP32 log_softmax semantics and float64 accumulation;
  * brute-force float64 reference on random inputs;
  * every documented rejection path (illegal target, all-empty target,
    length > 272, mask mismatch incl. the inverted-mask trap, non-finite
    logits, vocab/shape/dtype mismatches);
  * input non-mutation, output schema / field isolation, text dedup mapping.
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_vector_scorer import (  # noqa: E402
    MAX_CAD_SEQUENCE_LENGTH,
    ONE_HOT_SIZE,
    PADDING_INDEX,
    SCORE_NAME,
    SCORE_SPEC_VERSION,
    START_INDEX,
    REASON_BAD_DTYPE,
    REASON_BAD_SHAPE,
    REASON_BAD_START_TOKEN,
    REASON_BAD_TERMINAL_TOKEN,
    REASON_BAD_TEXT,
    REASON_EMPTY_TARGET,
    REASON_FLAG_MISMATCH,
    REASON_INCONSISTENT_ROW,
    REASON_INDEX_MISMATCH,
    REASON_LABEL_OUT_OF_RANGE,
    REASON_LENGTH_OVER_MAX,
    REASON_LENGTH_TOO_SHORT,
    REASON_MASK_MISMATCH,
    REASON_NONFINITE_LOGITS,
    REASON_NON_TAIL_PADDING,
    REASON_VOCAB_MISMATCH,
    CadVecDiagnostics,
    ScoreRequest,
    ScorerInputError,
    dedup_texts,
    two_head_factorized_score,
    validate_cad_vec,
)

V = ONE_HOT_SIZE  # 267


def toy_cad_vec() -> torch.Tensor:
    """Minimal legal candidate (N=6): START, one coordinate, END_CURVE,
    END_EXTRUSION, terminal START, one PADDING row.

    targets t=1..5: [15,20] (x+y), [5,0] (x), [6,0] (x), [1,0] (x), [0,0] (-)
    => n_valid_x = 4, n_valid_y = 1, n_real_tokens = 5.
    """
    return torch.tensor(
        [[1, 0], [15, 20], [5, 0], [6, 0], [1, 0], [0, 0]], dtype=torch.int32
    )


def make_logits(n_target: int, seed: int = 0, scale: float = 2.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n_target, 2, V, generator=gen) * scale


def ref_logprob_sum(logits: torch.Tensor, cad_vec: torch.Tensor) -> dict:
    """Independent float64 reference over the FP32-rounded logits."""
    x = logits.to(torch.float32).to(torch.float64)
    target = cad_vec[1:].to(torch.long)
    out = {"x": 0.0, "y": 0.0, "nx": 0, "ny": 0}
    for t in range(target.shape[0]):
        for h, key in ((0, "x"), (1, "y")):
            label = int(target[t, h])
            if label == PADDING_INDEX:
                continue
            row = x[t, h]
            m = float(row.max())
            lse = m + math.log(float((row - m).exp().sum()))
            out[key] += float(row[label]) - lse
            out["n" + key] += 1
    return out


class TestShiftAndMask(unittest.TestCase):
    def test_manual_score_matches_hand_computation(self):
        vec = toy_cad_vec()
        logits = make_logits(5, seed=7)
        result = two_head_factorized_score(logits, vec)
        ref = ref_logprob_sum(logits, vec)
        self.assertEqual(ref["nx"], 4)
        self.assertEqual(ref["ny"], 1)
        # kernel: FP32 log_softmax gathered, accumulated in float64; the
        # reference is pure float64, so agree only up to fp32 rounding
        self.assertAlmostEqual(result["score_x"], ref["x"], delta=1e-5)
        self.assertAlmostEqual(result["score_y"], ref["y"], delta=1e-5)
        self.assertAlmostEqual(result["score_total"], ref["x"] + ref["y"], delta=1e-5)
        self.assertEqual(result["n_valid_x"], 4)
        self.assertEqual(result["n_valid_y"], 1)
        self.assertEqual(result["n_valid_total"], 5)
        self.assertEqual(result["valid_seq_len"], 4)
        self.assertEqual(result["n_real_tokens"], 5)
        self.assertAlmostEqual(result["mean_x"], ref["x"] / 4, delta=1e-5)
        self.assertAlmostEqual(result["mean_y"], ref["y"], delta=1e-5)
        self.assertAlmostEqual(result["mean_total"], (ref["x"] + ref["y"]) / 5, delta=1e-5)

    def test_uniform_logits_closed_form(self):
        vec = toy_cad_vec()
        logits = torch.zeros(5, 2, V)
        result = two_head_factorized_score(logits, vec)
        fp32_log_v = float(torch.tensor(-math.log(V), dtype=torch.float32))
        self.assertAlmostEqual(result["score_x"], 4 * fp32_log_v, places=12)
        self.assertAlmostEqual(result["score_y"], 1 * fp32_log_v, places=12)
        self.assertAlmostEqual(result["score_total"], 5 * fp32_log_v, places=12)

    def test_dominant_logits_score_near_zero(self):
        vec = toy_cad_vec()
        logits = torch.full((5, 2, V), -30.0)
        target = vec[1:].to(torch.long)
        for t in range(5):
            for h in range(2):
                logits[t, h, target[t, h]] = 30.0  # certain on every label
        result = two_head_factorized_score(logits, vec)
        self.assertGreater(result["score_total"], -1e-4)
        self.assertLessEqual(result["score_total"], 0.0)

    def test_terminal_start_x_scored_y_masked(self):
        vec = toy_cad_vec()
        logits = make_logits(5, seed=11)
        base = two_head_factorized_score(logits, vec)
        # target t=3 is vec row 4, the terminal START [1, 0]: perturbing its
        # x row must change the score (x of the terminal token is scored)...
        perturbed = logits.clone()
        perturbed[3, 0, START_INDEX] += 5.0
        after = two_head_factorized_score(perturbed, vec)
        self.assertNotEqual(after["score_x"], base["score_x"])
        # ...while perturbing its y row must not (y is structurally masked).
        perturbed_y = logits.clone()
        perturbed_y[3, 1] = torch.arange(V, dtype=torch.float32) * 100.0
        after_y = two_head_factorized_score(perturbed_y, vec)
        self.assertEqual(after_y["score_y"], base["score_y"])
        self.assertEqual(after_y["score_total"], base["score_total"])

    def test_masked_positions_do_not_affect_score(self):
        vec = toy_cad_vec()
        logits = make_logits(5, seed=13)
        base = two_head_factorized_score(logits, vec)
        target = vec[1:].to(torch.long)
        valid = target != PADDING_INDEX
        poisoned = logits.clone()
        poison = torch.full((V,), 1e30)
        for t in range(5):
            for h in range(2):
                if not bool(valid[t, h]):
                    poisoned[t, h] = poison  # includes PADDING rows, y of
                    # END/param/terminal rows
        result = two_head_factorized_score(poisoned, vec)
        self.assertEqual(result["score_x"], base["score_x"])
        self.assertEqual(result["score_y"], base["score_y"])
        self.assertEqual(result["score_total"], base["score_total"])

    def test_explicit_matching_mask_equals_derived(self):
        vec = toy_cad_vec()
        logits = make_logits(5, seed=17)
        implicit = two_head_factorized_score(logits, vec)
        explicit = two_head_factorized_score(logits, vec, vec == PADDING_INDEX)
        self.assertEqual(implicit, explicit)


class TestBoundaries(unittest.TestCase):
    def test_full_272_with_padding_accepted(self):
        rows = [[1, 0]] + [[15, 20]] * 269 + [[1, 0]] + [[0, 0]]
        vec = torch.tensor(rows, dtype=torch.int32)
        self.assertEqual(vec.shape[0], MAX_CAD_SEQUENCE_LENGTH)
        logits = make_logits(MAX_CAD_SEQUENCE_LENGTH - 1, seed=19)
        result = two_head_factorized_score(logits, vec)
        self.assertEqual(result["n_valid_x"], 270)
        self.assertEqual(result["n_valid_y"], 269)
        self.assertEqual(result["valid_seq_len"], 270)

    def test_full_272_without_padding_accepted(self):
        rows = [[1, 0]] + [[15, 20]] * 270 + [[1, 0]]
        vec = torch.tensor(rows, dtype=torch.int32)
        logits = make_logits(MAX_CAD_SEQUENCE_LENGTH - 1, seed=23)
        result = two_head_factorized_score(logits, vec)
        self.assertEqual(result["n_valid_x"], 271)
        self.assertEqual(result["n_valid_y"], 270)
        self.assertEqual(result["n_real_tokens"], MAX_CAD_SEQUENCE_LENGTH)

    def test_reject_length_over_272(self):
        rows = [[1, 0]] + [[15, 20]] * 271 + [[1, 0]]
        vec = torch.tensor(rows, dtype=torch.int32)
        self.assertEqual(vec.shape[0], MAX_CAD_SEQUENCE_LENGTH + 1)
        with self.assertRaises(ScorerInputError) as ctx:
            validate_cad_vec(vec)
        self.assertEqual(ctx.exception.reason, REASON_LENGTH_OVER_MAX)

    def test_reject_length_one(self):
        with self.assertRaises(ScorerInputError) as ctx:
            validate_cad_vec(torch.tensor([[1, 0]], dtype=torch.int32))
        self.assertEqual(ctx.exception.reason, REASON_LENGTH_TOO_SHORT)

    def test_reject_all_padding(self):
        vec = torch.zeros(MAX_CAD_SEQUENCE_LENGTH, 2, dtype=torch.int32)
        with self.assertRaises(ScorerInputError) as ctx:
            validate_cad_vec(vec)
        self.assertEqual(ctx.exception.reason, REASON_EMPTY_TARGET)

    def test_reject_start_then_only_padding(self):
        # first row is a legal START but there is no real target row at all
        vec = torch.zeros(MAX_CAD_SEQUENCE_LENGTH, 2, dtype=torch.int32)
        vec[0] = torch.tensor([1, 0], dtype=torch.int32)
        with self.assertRaises(ScorerInputError) as ctx:
            validate_cad_vec(vec)
        self.assertEqual(ctx.exception.reason, REASON_EMPTY_TARGET)


class TestRejections(unittest.TestCase):
    def assert_reason(self, reason, fn, *args, **kwargs):
        with self.assertRaises(ScorerInputError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.reason, reason, str(ctx.exception))

    def test_reject_label_above_vocab(self):
        vec = toy_cad_vec().clone()
        vec[1, 0] = V  # 267 is outside [0, 266]
        self.assert_reason(REASON_LABEL_OUT_OF_RANGE, validate_cad_vec, vec)

    def test_reject_negative_label(self):
        vec = toy_cad_vec().clone()
        vec[2, 0] = -1
        self.assert_reason(REASON_LABEL_OUT_OF_RANGE, validate_cad_vec, vec)

    def test_reject_bad_initial_token(self):
        vec = toy_cad_vec().clone()
        vec[0] = torch.tensor([2, 0], dtype=torch.int32)
        self.assert_reason(REASON_BAD_START_TOKEN, validate_cad_vec, vec)

    def test_reject_missing_terminal_start(self):
        vec = toy_cad_vec().clone()
        vec[4] = torch.tensor([6, 0], dtype=torch.int32)  # ends on END_EXTRUSION
        self.assert_reason(REASON_BAD_TERMINAL_TOKEN, validate_cad_vec, vec)

    def test_reject_non_tail_padding(self):
        vec = torch.tensor(
            [[1, 0], [5, 0], [0, 0], [15, 20], [1, 0], [0, 0]], dtype=torch.int32
        )
        self.assert_reason(REASON_NON_TAIL_PADDING, validate_cad_vec, vec)

    def test_reject_x_padding_with_y_value(self):
        vec = torch.tensor(
            [[1, 0], [0, 7], [1, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.int32
        )
        self.assert_reason(REASON_INCONSISTENT_ROW, validate_cad_vec, vec)

    def test_reject_mask_vec_mismatch(self):
        vec = toy_cad_vec()
        mask = vec == PADDING_INDEX
        mask[1, 0] = True  # corrupt one bit
        self.assert_reason(
            REASON_MASK_MISMATCH, validate_cad_vec, vec, mask
        )

    def test_reject_inverted_mask_trap(self):
        # the classic bug: passing ~key_padding_mask (the loss-side valid
        # mask) where the official storage mask is expected
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_MASK_MISMATCH,
            two_head_factorized_score,
            make_logits(5, seed=29),
            vec,
            ~(vec == PADDING_INDEX),
        )

    def test_reject_shifted_mask_trap(self):
        # a pre-shifted (N-1, 2) mask has the wrong shape entirely
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_BAD_SHAPE,
            two_head_factorized_score,
            make_logits(5, seed=31),
            vec,
            (vec == PADDING_INDEX)[1:],
        )

    def test_reject_nonfinite_logits(self):
        vec = toy_cad_vec()
        for bad in (float("nan"), float("inf"), float("-inf")):
            logits = make_logits(5, seed=37)
            logits[3, 1, 100] = bad  # at a *masked* position: still rejected
            self.assert_reason(
                REASON_NONFINITE_LOGITS, two_head_factorized_score, logits, vec
            )

    def test_reject_vocab_mismatch(self):
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_VOCAB_MISMATCH,
            two_head_factorized_score,
            torch.zeros(5, 2, V - 1),
            vec,
        )

    def test_reject_head_dim_mismatch(self):
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_BAD_SHAPE,
            two_head_factorized_score,
            torch.zeros(5, 3, V),
            vec,
        )

    def test_reject_logits_rank(self):
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_BAD_SHAPE, two_head_factorized_score, torch.zeros(5, V), vec
        )

    def test_reject_logits_integer_dtype(self):
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_BAD_DTYPE,
            two_head_factorized_score,
            torch.zeros(5, 2, V, dtype=torch.int32),
            vec,
        )

    def test_reject_step_count_mismatch(self):
        vec = toy_cad_vec()
        self.assert_reason(
            REASON_BAD_SHAPE,
            two_head_factorized_score,
            make_logits(4, seed=41),  # needs 5 steps for N=6
            vec,
        )

    def test_reject_bool_cad_vec(self):
        self.assert_reason(
            REASON_BAD_DTYPE,
            validate_cad_vec,
            torch.zeros(6, 2, dtype=torch.bool),
        )

    def test_reject_float_cad_vec(self):
        self.assert_reason(
            REASON_BAD_DTYPE,
            validate_cad_vec,
            torch.zeros(6, 2, dtype=torch.float32),
        )

    def test_reject_bad_flag_padding(self):
        vec = toy_cad_vec()
        flags = torch.tensor([0, 0, 0, 1, 0, 10], dtype=torch.int32)
        self.assert_reason(REASON_FLAG_MISMATCH, validate_cad_vec, vec, None, flags)

    def test_reject_flag_out_of_range(self):
        vec = toy_cad_vec()
        flags = torch.tensor([0, 0, 0, 1, 0, 12], dtype=torch.int32)
        self.assert_reason(REASON_FLAG_MISMATCH, validate_cad_vec, vec, None, flags)

    def test_reject_bad_index_padding(self):
        vec = toy_cad_vec()
        indices = torch.tensor([0, 0, 0, 0, 0, 5], dtype=torch.int32)
        self.assert_reason(
            REASON_INDEX_MISMATCH, validate_cad_vec, vec, None, None, indices
        )

    def test_accept_consistent_flag_and_index(self):
        vec = toy_cad_vec()
        flags = torch.tensor([0, 0, 0, 1, 0, 11], dtype=torch.int32)
        indices = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int32)
        diag = validate_cad_vec(vec, None, flags, indices)
        self.assertIsInstance(diag, CadVecDiagnostics)
        self.assertEqual(diag.n_valid_x, 4)


class TestNumerics(unittest.TestCase):
    def test_batch_equals_single_bitwise(self):
        vecs = torch.stack(
            [
                toy_cad_vec(),
                torch.tensor(
                    [[1, 0], [100, 200], [5, 0], [1, 0], [0, 0], [0, 0]],
                    dtype=torch.int32,
                ),
                torch.tensor(
                    [[1, 0], [11, 12], [13, 14], [5, 0], [6, 0], [1, 0]],
                    dtype=torch.int32,
                ),
            ]
        )
        logits = torch.randn(3, 5, 2, V, generator=torch.Generator().manual_seed(43))
        batched = two_head_factorized_score(logits, vecs)
        self.assertIsInstance(batched, list)
        for i in range(3):
            single = two_head_factorized_score(logits[i], vecs[i])
            self.assertEqual(batched[i], single)

    def test_batch_with_mask(self):
        vecs = torch.stack([toy_cad_vec(), toy_cad_vec()])
        logits = make_logits(5, seed=47).unsqueeze(0).repeat(2, 1, 1, 1)
        result = two_head_factorized_score(logits, vecs, vecs == PADDING_INDEX)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], result[1])
        self.assertEqual(
            result[0], two_head_factorized_score(logits[0], vecs[0], vecs[0] == 0)
        )

    def test_fp64_logits_are_scored_in_fp32(self):
        vec = toy_cad_vec()
        gen = torch.Generator().manual_seed(53)
        logits32 = torch.randn(5, 2, V, generator=gen)
        # perturb below fp32 resolution: fp64 sees a different tensor, the
        # fp32 kernel must not
        logits64 = logits32.to(torch.float64) + 1e-9 * torch.randn(
            5, 2, V, generator=gen
        ).to(torch.float64)
        from32 = two_head_factorized_score(logits32, vec)
        from64 = two_head_factorized_score(logits64, vec)
        self.assertEqual(from32["score_total"], from64["score_total"])
        self.assertEqual(from32["score_x"], from64["score_x"])
        self.assertEqual(from32["score_y"], from64["score_y"])

    def test_bf16_logits_accepted_and_fp32_scored(self):
        vec = toy_cad_vec()
        logits = make_logits(5, seed=59).to(torch.bfloat16)
        result = two_head_factorized_score(logits, vec)
        ref = two_head_factorized_score(logits.to(torch.float32), vec)
        self.assertEqual(result, ref)

    def test_bruteforce_reference_random_batch(self):
        gen = torch.Generator().manual_seed(61)
        vecs = torch.stack(
            [
                toy_cad_vec(),
                torch.tensor(
                    [[1, 0], [250, 260], [5, 0], [6, 0], [1, 0], [0, 0]],
                    dtype=torch.int32,
                ),
            ]
        )
        logits = torch.randn(2, 5, 2, V, generator=gen) * 3.0
        results = two_head_factorized_score(logits, vecs)
        for i in range(2):
            ref = ref_logprob_sum(logits[i], vecs[i])
            self.assertAlmostEqual(
                results[i]["score_x"], ref["x"], delta=1e-4 * max(1.0, abs(ref["x"]))
            )
            self.assertAlmostEqual(
                results[i]["score_y"], ref["y"], delta=1e-4 * max(1.0, abs(ref["y"]))
            )


class TestHygieneAndSchema(unittest.TestCase):
    def test_inputs_not_mutated(self):
        vec = toy_cad_vec()
        mask = vec == PADDING_INDEX
        logits = make_logits(5, seed=67)
        vec0, mask0, logits0 = vec.clone(), mask.clone(), logits.clone()
        two_head_factorized_score(logits, vec, mask)
        self.assertTrue(torch.equal(vec, vec0))
        self.assertTrue(torch.equal(mask, mask0))
        self.assertTrue(torch.equal(logits, logits0))

    def test_output_schema_exact(self):
        result = two_head_factorized_score(make_logits(5, seed=71), toy_cad_vec())
        self.assertEqual(
            set(result.keys()),
            {
                "score_name",
                "spec_version",
                "score_x",
                "score_y",
                "score_total",
                "n_valid_x",
                "n_valid_y",
                "n_valid_total",
                "valid_seq_len",
                "n_real_tokens",
                "mean_x",
                "mean_y",
                "mean_total",
            },
        )
        self.assertEqual(result["score_name"], SCORE_NAME)
        self.assertEqual(result["spec_version"], SCORE_SPEC_VERSION)
        for key in ("score_x", "score_y", "score_total", "mean_x", "mean_y", "mean_total"):
            self.assertIsInstance(result[key], float)
        for key in ("n_valid_x", "n_valid_y", "n_valid_total", "valid_seq_len", "n_real_tokens"):
            self.assertIsInstance(result[key], int)
        # no UID / qrels / metadata fields ever appear in scorer output
        for leaked in ("uid", "qrel", "relevant", "label", "rank"):
            self.assertNotIn(leaked, {k.lower() for k in result})

    def test_score_request_field_isolation(self):
        self.assertEqual(
            {f.name for f in fields(ScoreRequest)},
            {"query_text", "cad_vec", "key_padding_mask", "flag_vec", "index_vec"},
        )
        with self.assertRaises(TypeError):
            ScoreRequest(query_text="q", cad_vec=toy_cad_vec(), uid="0000/00000001")
        with self.assertRaises(TypeError):
            ScoreRequest(query_text="q", cad_vec=toy_cad_vec(), qrel=1)

    def test_score_request_validate(self):
        req = ScoreRequest(
            query_text="a cylindrical bracket",
            cad_vec=toy_cad_vec(),
            key_padding_mask=toy_cad_vec() == PADDING_INDEX,
        )
        diag = req.validate()
        self.assertEqual(diag.n_real_tokens, 5)
        with self.assertRaises(ScorerInputError) as ctx:
            ScoreRequest(query_text="", cad_vec=toy_cad_vec()).validate()
        self.assertEqual(ctx.exception.reason, REASON_BAD_TEXT)

    def test_dedup_texts(self):
        unique, mapping = dedup_texts(["a", "b", "a", "c", "b", "a"])
        self.assertEqual(unique, ["a", "b", "c"])
        self.assertEqual(mapping, [0, 1, 0, 2, 1, 0])

    def test_dedup_texts_verbatim(self):
        # no re-normalization: whitespace/case differences stay distinct
        unique, mapping = dedup_texts(["A part", "a part", "A part "])
        self.assertEqual(len(unique), 3)
        self.assertEqual(mapping, [0, 1, 2])

    def test_dedup_texts_rejects_bad_text(self):
        for bad in (["ok", ""], ["ok", None], ["ok", 3], "not-a-sequence"):
            with self.assertRaises(ScorerInputError) as ctx:
                dedup_texts(bad)  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.reason, REASON_BAD_TEXT)


if __name__ == "__main__":
    unittest.main()
