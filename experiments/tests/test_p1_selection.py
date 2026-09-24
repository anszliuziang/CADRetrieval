"""Tests for the P1 fixed selection logic (plan sec.2) on synthetic pools."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.p1_selection import (  # noqa: E402
    MAX_FORWARDS,
    SEED,
    RowRecord,
    SelectionError,
    build_pair_plan,
    compute_eligibility,
    draw_queries_and_offsets,
    extract_method_set,
    jaccard,
    offset_coverage_check,
    select_p1,
)


def mkrec(row, uid, methods=(), resp=100, prefix=50, identical=True, boundary=True, eos_out=False):
    return RowRecord(
        row_index=row,
        uid=uid,
        program_identical=identical,
        n_prefix_tokens=prefix,
        n_response_tokens=resp,
        boundary_ok=boundary,
        eos_in_output=eos_out,
        method_set=frozenset(methods),
    )


def mkpool(n, resp_base=100):
    """Pool of n eligible rows with distinct uids and varied lengths/methods."""
    recs = []
    for i in range(n):
        methods = {f"m{i % 7}", f"m{i % 5}", "workplane", "extrude"}
        recs.append(mkrec(i, f"u{i:04d}", methods, resp=resp_base + (i * 13) % 97))
    return recs


class TestMethodExtraction(unittest.TestCase):
    def test_realistic_snippet(self):
        prog = (
            "import cadquery as cq\n"
            "result = cq.Workplane('XY').box(1, 2, 3).edges('|Z').fillet(0.1)\n"
            "cq.exporters.export(result, 'out.step')\n"
        )
        ms = extract_method_set(prog)
        self.assertIn("Workplane", ms)
        self.assertIn("box", ms)
        self.assertIn("edges", ms)
        self.assertIn("fillet", ms)
        # export/exporters are excluded per plan
        self.assertNotIn("export", ms)
        self.assertNotIn("exporters", ms)

    def test_no_match_on_bare_calls(self):
        self.assertEqual(extract_method_set("foo(1)\nbar = 2\n"), frozenset())

    def test_jaccard(self):
        a, b = frozenset({"x", "y"}), frozenset({"y", "z"})
        self.assertAlmostEqual(jaccard(a, b), 1 / 3)
        self.assertEqual(jaccard(frozenset(), frozenset()), 0.0)
        self.assertEqual(jaccard(frozenset({"x"}), frozenset()), 0.0)


class TestEligibility(unittest.TestCase):
    def test_per_criterion_exclusion_counts(self):
        recs = [
            mkrec(0, "u0", resp=100),                      # eligible
            mkrec(1, None, resp=100),                      # no uid
            mkrec(2, "u2", resp=701),                      # response too long
            mkrec(3, "u3", resp=100, prefix=251),          # prefix too long
            mkrec(4, "u4", resp=100, identical=False),     # zip differs
            mkrec(5, "u5", resp=100, boundary=False),      # boundary
            mkrec(6, "u6", resp=100, eos_out=True),        # eos in output
            mkrec(7, "u7", resp=800, prefix=900),          # two violations, counted twice
        ]
        flags, counts = compute_eligibility(recs, 250, 700)
        self.assertEqual(flags, [True, False, False, False, False, False, False, False])
        self.assertEqual(counts["n_eligible"], 1)
        self.assertEqual(counts["excl_uid_not_unique"], 1)
        self.assertEqual(counts["excl_response_too_long"], 2)
        self.assertEqual(counts["excl_prefix_too_long"], 2)
        self.assertEqual(counts["excl_program_not_identical"], 1)
        self.assertEqual(counts["excl_boundary"], 1)
        self.assertEqual(counts["excl_eos_in_output"], 1)


class TestDraw(unittest.TestCase):
    def test_too_small_pool_stops(self):
        with self.assertRaises(SelectionError):
            draw_queries_and_offsets([f"u{i}" for i in range(31)], 32, 3, SEED)

    def test_duplicate_uid_pool_stops(self):
        pool = [f"u{i:04d}" for i in range(40)]
        pool[5] = pool[6]
        with self.assertRaises(SelectionError):
            draw_queries_and_offsets(sorted(pool), 32, 3, SEED)

    def test_deterministic_and_offsets_valid(self):
        pool = sorted(f"u{i:04d}" for i in range(100))
        s1, o1 = draw_queries_and_offsets(pool, 32, 3, SEED)
        s2, o2 = draw_queries_and_offsets(pool, 32, 3, SEED)
        self.assertEqual(s1, s2)
        self.assertEqual(o1, o2)
        self.assertEqual(len(set(s1)), 32)
        self.assertEqual(len(o1), 3)
        self.assertEqual(len(set(o1)), 3)
        self.assertTrue(all(1 <= o <= 31 for o in o1))
        # different seed -> (almost surely) different draw
        s3, _ = draw_queries_and_offsets(pool, 32, 3, SEED + 1)
        self.assertNotEqual(s1, s3)


class TestSelectP1(unittest.TestCase):
    def setUp(self):
        self.recs = mkpool(120)
        self.flags = [True] * len(self.recs)

    def test_deterministic_and_structure(self):
        s1 = select_p1(self.recs, self.flags)
        s2 = select_p1(self.recs, self.flags)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1.queries), 32)
        self.assertEqual(len(s1.offsets), 3)
        for qp in s1.queries:
            self.assertEqual(len(qp.candidates), 8)
            uids = [c.uid for c in qp.candidates]
            self.assertEqual(len(set(uids)), 8, f"query {qp.uid}: candidates not distinct")
            self.assertEqual(qp.candidates[0].source, "self")
            self.assertEqual(qp.candidates[0].uid, qp.uid)
            self.assertEqual([c.source for c in qp.candidates],
                             ["self", "selected-offset", "selected-offset", "selected-offset",
                              "method-neighbor", "method-neighbor", "length-neighbor", "length-neighbor"])

    def test_offset_candidates_match_rule(self):
        s = select_p1(self.recs, self.flags)
        selected = [qp.uid for qp in s.queries]
        for qp in s.queries:
            offs = [c for c in qp.candidates if c.source == "selected-offset"]
            self.assertEqual([c.offset for c in offs], s.offsets)
            for c in offs:
                self.assertEqual(c.uid, selected[(qp.query_index + c.offset) % 32])

    def test_offset_coverage_exactly_three(self):
        s = select_p1(self.recs, self.flags)
        cov = offset_coverage_check(s)
        self.assertTrue(cov["ok"], cov)

    def test_neighbors_exclude_selected_32(self):
        s = select_p1(self.recs, self.flags)
        selected = {qp.uid for qp in s.queries}
        for qp in s.queries:
            for c in qp.candidates:
                if c.source in ("method-neighbor", "length-neighbor"):
                    self.assertNotIn(c.uid, selected)

    def test_unfillable_pool_stops(self):
        # 34 eligible: after 32 selected + 2 method picks, length pool is empty
        recs = mkpool(34)
        flags = [True] * len(recs)
        with self.assertRaises(SelectionError):
            select_p1(recs, flags)

    def test_method_neighbor_tiebreak_length_then_uid(self):
        # self methods {a,b}: m3 wins on jaccard (1.0); m1/m2 tie at 1/3 and the
        # smaller response-length diff (m2, 10) ranks before m1 (80), even
        # though uid order would prefer m1.
        from idea01.p1_selection import rank_method_neighbors

        self_rec = mkrec(0, "a_self", ("a", "b"), resp=100)
        pool = [
            self_rec,
            mkrec(1, "a_m1", ("a", "c"), resp=180),   # jac 1/3, diff 80
            mkrec(2, "a_m2", ("a", "d"), resp=110),   # jac 1/3, diff 10 -> before m1
            mkrec(3, "a_m3", ("a", "b"), resp=400),   # jac 1.0 -> first
        ]
        ranked = rank_method_neighbors(self_rec, pool, exclude_uids={"a_self"})
        self.assertEqual([r.uid for r in ranked], ["a_m3", "a_m2", "a_m1"])

    def test_length_neighbor_uid_tiebreak(self):
        from idea01.p1_selection import rank_length_neighbors

        self_rec = mkrec(0, "s", ("a",), resp=100)
        pool = [
            self_rec,
            mkrec(1, "b_hi", ("a",), resp=105),     # excluded (pretend method pick)
            mkrec(2, "c_lo", ("a", "b"), resp=95),  # excluded (pretend method pick)
            mkrec(3, "d_l1", ("z",), resp=102),     # diff 2
            mkrec(4, "e_l2", ("z",), resp=98),      # diff 2, larger uid -> after d_l1
            mkrec(5, "f_far", ("z",), resp=500),
        ]
        ranked = rank_length_neighbors(self_rec, pool, {"s", "b_hi", "c_lo"})
        self.assertEqual([r.uid for r in ranked], ["d_l1", "e_l2", "f_far"])


class TestPairPlan(unittest.TestCase):
    def test_counts_ids_fields(self):
        recs = mkpool(120)
        flags = [True] * len(recs)
        s = select_p1(recs, flags)
        pairs = build_pair_plan(s)
        n_cond = [p for p in pairs if p["kind"] == "conditional"]
        n_empty = [p for p in pairs if p["kind"] == "empty"]
        self.assertEqual(len(n_cond), 256)
        self.assertEqual(len(n_empty), len(s.distinct_candidate_uids))
        self.assertLessEqual(len(n_empty), 160)
        self.assertLessEqual(len(pairs), MAX_FORWARDS)
        ids = [p["pair_id"] for p in pairs]
        self.assertEqual(len(set(ids)), len(ids))
        # self pair: query uid == candidate uid
        for p in n_cond:
            if p["candidate_source"] == "self":
                self.assertEqual(p["query_uid"], p["candidate_uid"])
            else:
                self.assertNotEqual(p["query_uid"], p["candidate_uid"])
            self.assertIsNotNone(p["query_row_index"])
        for p in n_empty:
            self.assertIsNone(p["query_uid"])
            self.assertIsNotNone(p["candidate_row_index"])

    def test_cap_enforced(self):
        # 64 queries x 8 candidates = 512 conditional pairs > 416 budget -> stop
        recs = mkpool(120)
        flags = [True] * len(recs)
        s = select_p1(recs, flags, n_queries=64)
        with self.assertRaises(SelectionError):
            build_pair_plan(s)


if __name__ == "__main__":
    unittest.main()
