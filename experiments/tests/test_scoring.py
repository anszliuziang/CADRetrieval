"""Unit tests for the scoring core: shift / mask / EOS handling with
hand-constructed logits, prefix boundary checks, and score arithmetic.

Runs locally without torch (numpy only). These tests are the P0 "constructed
logits" correctness evidence required by the task contract.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.prompts import build_full_text, build_prefix  # noqa: E402
from idea01.scoring import (  # noqa: E402
    ScoringError,
    build_scored_sequence,
    check_prefix_boundary,
    gather_response_logprobs,
    log_softmax_f32,
)


def make_logits(L: int, V: int, value: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.normal(0, 1, size=(L, V)).astype(np.float32) + value


class TestLogSoftmax(unittest.TestCase):
    def test_rows_sum_to_one_in_prob_space(self):
        logits = make_logits(5, 17)
        lsm = log_softmax_f32(logits)
        np.testing.assert_allclose(np.exp(lsm).sum(axis=-1), np.ones(5), rtol=1e-5)

    def test_known_values(self):
        logits = np.array([[0.0, 0.0]], dtype=np.float32)
        lsm = log_softmax_f32(logits)
        self.assertAlmostEqual(float(lsm[0, 0]), -math.log(2), places=6)


class TestGatherShift(unittest.TestCase):
    """Hand-constructed case: prompt = tokens [10, 11, 12], response = [20, 21],
    EOS = 99. input_ids = [10, 11, 12, 20, 21, 99], target_start = 3.

    We set logits so that:
      row 2 is one-hot on 20  -> first response token scored ~0
      row 3 is one-hot on 21  -> second response token ~0
      row 4 is one-hot on 99  -> EOS ~0
    All other target choices get very negative log-probs. If shift is off by
    one in either direction, sums must change drastically.
    """

    def setUp(self):
        self.V = 100
        self.ids = np.array([10, 11, 12, 20, 21, 99])
        self.L = len(self.ids)
        logits = np.full((self.L, self.V), -20.0, dtype=np.float32)
        logits[2, 20] = 20.0
        logits[3, 21] = 20.0
        logits[4, 99] = 20.0
        self.logits = logits

    def test_correct_shift_gives_near_zero_sum(self):
        res = gather_response_logprobs(self.logits, self.ids, target_start=3)
        self.assertEqual(res.n_target_tokens, 3)  # 2 response + EOS
        self.assertAlmostEqual(res.sum_logprob, 0.0, places=3)
        self.assertTrue(res.all_finite)

    def test_positions_and_pred_rows(self):
        res = gather_response_logprobs(self.logits, self.ids, target_start=3)
        self.assertEqual([t.position for t in res.tokens], [3, 4, 5])
        self.assertEqual([t.pred_row for t in res.tokens], [2, 3, 4])
        self.assertEqual([t.token_id for t in res.tokens], [20, 21, 99])

    def test_eos_included_prompt_excluded(self):
        res = gather_response_logprobs(self.logits, self.ids, target_start=3)
        self.assertEqual(res.tokens[-1].token_id, 99)  # EOS scored
        self.assertNotIn(12, [t.token_id for t in res.tokens])  # last prompt token not a target
        self.assertTrue(all(t.position >= 3 for t in res.tokens))

    def test_shift_sensitivity(self):
        # Corrupt the row that would be used with an off-by-one shift: make
        # row 3 predict token 20 instead of 21. Correct shift must notice.
        logits = self.logits.copy()
        logits[3, :] = -20.0
        logits[3, 20] = 20.0
        res = gather_response_logprobs(logits, self.ids, target_start=3)
        self.assertLess(res.sum_logprob, -10.0)  # token 21 now unpredicted

    def test_mean_is_sum_over_n(self):
        res = gather_response_logprobs(self.logits, self.ids, target_start=3)
        self.assertAlmostEqual(res.mean_logprob, res.sum_logprob / 3.0, places=7)

    def test_uniform_logits_match_log_vocab(self):
        logits = np.zeros((self.L, self.V), dtype=np.float32)
        res = gather_response_logprobs(logits, self.ids, target_start=3)
        self.assertAlmostEqual(res.sum_logprob, -3.0 * math.log(self.V), places=3)

    def test_boundary_errors(self):
        with self.assertRaises(ScoringError):
            gather_response_logprobs(self.logits, self.ids, target_start=0)
        with self.assertRaises(ScoringError):
            gather_response_logprobs(self.logits, self.ids, target_start=self.L)
        with self.assertRaises(ScoringError):
            gather_response_logprobs(self.logits[: self.L - 1], self.ids, target_start=3)

    def test_nonfinite_detected(self):
        logits = self.logits.copy()
        logits[3, :] = np.nan
        res = gather_response_logprobs(logits, self.ids, target_start=3)
        self.assertFalse(res.all_finite)


class TestPrefixBoundary(unittest.TestCase):
    def test_exact_template(self):
        self.assertEqual(build_prefix("abc"), "### Instruction:\nabc\n\n### Response:\n")
        self.assertEqual(build_full_text("q", "body"), "### Instruction:\nq\n\n### Response:\nbody")

    def test_query_with_braces_is_verbatim(self):
        q = "use dict {a: 1}"
        self.assertIn(q, build_prefix(q))

    def test_boundary_check(self):
        self.assertTrue(check_prefix_boundary([1, 2], [1, 2, 3, 4]))
        self.assertFalse(check_prefix_boundary([1, 2], [1, 5, 3, 4]))
        self.assertFalse(check_prefix_boundary([1, 2, 3], [1, 2]))  # not strict
        self.assertFalse(check_prefix_boundary([1, 2], [1, 2]))  # no response left

    def test_build_scored_sequence_appends_eos(self):
        ids, ts = build_scored_sequence([1, 2], [1, 2, 3], eos_id=99)
        self.assertEqual(ids, [1, 2, 3, 99])
        self.assertEqual(ts, 2)
        with self.assertRaises(ScoringError):
            build_scored_sequence([1, 9], [1, 2, 3], eos_id=99)
        with self.assertRaises(ScoringError):
            build_scored_sequence([1, 2], [1, 2, 3], eos_id=None)


if __name__ == "__main__":
    unittest.main()
