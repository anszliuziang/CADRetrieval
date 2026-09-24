"""Tests for the P1 matrix-runner pure helpers (budgets, pair-id parsing, stop policy)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.p1_runner import (  # noqa: E402
    EXIT_ANOMALY,
    EXIT_BUDGET,
    EXIT_OK,
    MAX_PAIRS,
    STATUS_BOUNDARY,
    STATUS_ERROR,
    STATUS_NONFINITE,
    STATUS_OK,
    STATUS_OVERLONG,
    parse_pair_ids,
    should_stop,
)


class TestParsePairIds(unittest.TestCase):
    def test_all(self):
        self.assertEqual(parse_pair_ids("all", 416), list(range(416)))

    def test_single_and_range(self):
        self.assertEqual(parse_pair_ids("0", 416), [0])
        self.assertEqual(parse_pair_ids("0-9", 416), list(range(10)))
        self.assertEqual(parse_pair_ids("0,3,5-7", 416), [0, 3, 5, 6, 7])

    def test_dedupes_preserving_order(self):
        self.assertEqual(parse_pair_ids("2,2,1-3,3", 416), [2, 1, 3])

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_pair_ids("416", 416)
        with self.assertRaises(ValueError):
            parse_pair_ids("-1", 416)

    def test_cap_enforced_on_request(self):
        with self.assertRaises(ValueError):
            parse_pair_ids("all", 416, max_pairs=100)

    def test_plan_larger_than_cap_refused(self):
        with self.assertRaises(ValueError):
            parse_pair_ids("all", 500)  # plan itself exceeds MAX_PAIRS


class TestStopPolicy(unittest.TestCase):
    def test_ok_does_not_stop(self):
        self.assertFalse(should_stop(STATUS_OK))

    def test_all_anomalies_stop_including_overlong(self):
        for s in (STATUS_BOUNDARY, STATUS_NONFINITE, STATUS_ERROR, STATUS_OVERLONG):
            self.assertTrue(should_stop(s), s)

    def test_exit_codes_distinct(self):
        self.assertEqual(len({EXIT_OK, EXIT_ANOMALY, EXIT_BUDGET}), 3)
        self.assertEqual(EXIT_OK, 0)

    def test_budget_is_416(self):
        self.assertEqual(MAX_PAIRS, 416)


if __name__ == "__main__":
    unittest.main()
