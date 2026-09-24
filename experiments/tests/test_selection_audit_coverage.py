"""Tests for deterministic selection and data-audit logic on synthetic fixtures."""

import csv
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.coverage import comment_line_ratio, has_export_boilerplate, measure_row, summarize  # noqa: E402
from idea01.data_audit import audit_split, build_intermediate_index, cross_split_overlap  # noqa: E402
from idea01.prompts import build_prefix  # noqa: E402
from idea01.selection import build_pair_plan, select_candidates  # noqa: E402


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.uids = ["0000/00000003", "0000/00000001", None, "0000/00000002", "0000/00000004"]
        self.fits = [True, True, True, False, True]  # row 3 too long
        self.boundary = [True, True, True, True, True]
        self.lens = [100, 200, 300, 2000, 400]

    def test_sorted_uid_first_n_and_exclusions(self):
        cands = select_candidates(self.uids, self.fits, self.boundary, self.lens, n_candidates=3, seed=7)
        # eligible: rows 0,1,4 (row 2 unmapped, row 3 too long)
        self.assertEqual([c.uid for c in cands], ["0000/00000001", "0000/00000003", "0000/00000004"])
        self.assertEqual([c.row_index for c in cands], [1, 0, 4])

    def test_deterministic_given_seed(self):
        a = select_candidates(self.uids, self.fits, self.boundary, self.lens, 3, seed=42)
        b = select_candidates(self.uids, self.fits, self.boundary, self.lens, 3, seed=42)
        self.assertEqual([c.mismatch_cand_index for c in a], [c.mismatch_cand_index for c in b])
        for c in a:
            self.assertNotEqual(c.mismatch_cand_index, c.cand_index)

    def test_boundary_mismatch_excluded(self):
        boundary = [True, False, True, True, True]
        cands = select_candidates(self.uids, self.fits, boundary, self.lens, 2, seed=1)
        self.assertNotIn(1, [c.row_index for c in cands])

    def test_pair_plan_three_conditions(self):
        cands = select_candidates(self.uids, self.fits, self.boundary, self.lens, 3, seed=7)
        pairs = build_pair_plan(cands)
        self.assertEqual(len(pairs), 9)
        self.assertEqual({p["condition"] for p in pairs}, {"correct", "mismatched", "empty"})
        for p in pairs:
            if p["condition"] == "correct":
                self.assertEqual(p["query_uid"], p["candidate_uid"])
            elif p["condition"] == "mismatched":
                self.assertIsNotNone(p["query_uid"])
                self.assertNotEqual(p["query_uid"], p["candidate_uid"])
            else:
                self.assertIsNone(p["query_uid"])


class TestDataAudit(unittest.TestCase):
    def _csv(self, rows):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["uid", "abstract", "beginner", "expert", "description", "keywords", "intermediate", "all_level_data", "nli_data"])
        for uid, inter in rows:
            w.writerow([uid, "", "", "", "", "", inter, "", ""])
        return buf.getvalue()

    def _zip(self, mapping, path):
        with zipfile.ZipFile(path, "w") as zf:
            for uid, text in mapping.items():
                zf.writestr(f"CQ/{uid}.py", text)

    def test_mapping_and_program_status(self, tmp=None):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "t.csv"
            csv_path.write_text(self._csv([
                ("u1", "text one"),
                ("u2", "text two"),
                ("u3", "text two"),  # ambiguous duplicate intermediate
            ]), encoding="utf-8")
            zip_path = Path(td) / "t.zip"
            self._zip({"u1": "prog A", "u2": "prog B"}, zip_path)

            index, n_csv = build_intermediate_index(csv_path)
            self.assertEqual(n_csv, 3)

            rows = [
                {"input": "text one", "output": "prog A"},      # unique, identical
                {"input": "text two", "output": "prog B"},      # ambiguous
                {"input": "text gone", "output": "prog C"},     # missing
                {"input": "text one", "output": "prog A"},      # duplicate input
            ]
            with zipfile.ZipFile(zip_path) as zf:
                a = audit_split("val", rows, index, zf)
            self.assertEqual(a.n_unique_map, 2)
            self.assertEqual(a.n_ambiguous_map, 1)
            self.assertEqual(a.n_missing_map, 1)
            self.assertEqual(a.program_status.get("identical"), 2)
            self.assertEqual(a.exact_duplicate_inputs, 1)
            self.assertEqual(a.row_uids, ["u1", None, None, "u1"])

    def test_program_difference_recorded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "t.csv"
            csv_path.write_text(self._csv([("u1", "x")]), encoding="utf-8")
            zip_path = Path(td) / "t.zip"
            self._zip({"u1": "zip version"}, zip_path)
            index, _ = build_intermediate_index(csv_path)
            rows = [{"input": "x", "output": "jsonl version  # extra comment"}]
            with zipfile.ZipFile(zip_path) as zf:
                a = audit_split("train", rows, index, zf)
            self.assertEqual(a.program_status.get("different"), 1)
            self.assertEqual(a.program_different_uids, ["u1"])

    def test_cross_split_overlap(self):
        class A:
            def __init__(self, uids):
                self.row_uids = uids
        res = cross_split_overlap({"train": A(["a", "b"]), "val": A(["b", "c"])})
        self.assertEqual(res["train__val"]["uid_intersection"], 1)


class TestCoverageHelpers(unittest.TestCase):
    def test_comment_ratio(self):
        prog = "import cadquery as cq\n# comment\nx = 1\n\n   # indented\n"
        self.assertAlmostEqual(comment_line_ratio(prog), 2 / 4)

    def test_export_boilerplate(self):
        self.assertTrue(has_export_boilerplate("cq.exporters.export(w, 'x.stl')"))
        self.assertTrue(has_export_boilerplate("result.export('x.step')"))
        self.assertFalse(has_export_boilerplate("x = 1"))

    def test_measure_row_and_boundary(self):
        # char-level fake tokenizer: each char is a token id
        encode = lambda s: [ord(c) for c in s]  # noqa: E731
        r = measure_row(0, "q", "body", encode, eos_id=0, max_len=100,
                        prefix_builder=lambda q: f"P[{q}]")
        # prefix = "P[q]" (4 chars), full = "P[q]body" (8) + EOS = 9
        self.assertEqual(r.n_prefix_tokens, 4)
        self.assertEqual(r.n_response_tokens, 5)  # body + EOS
        self.assertEqual(r.n_full_tokens, 9)
        self.assertTrue(r.boundary_ok)
        self.assertTrue(r.fits_max_len)
        self.assertFalse(r.eos_in_output)

    def test_measure_row_detects_eos_inside_output(self):
        encode = lambda s: [ord(c) for c in s]  # noqa: E731
        r = measure_row(0, "q", "bo\x00dy", encode, eos_id=0, max_len=100,
                        prefix_builder=lambda q: f"P[{q}]")
        self.assertTrue(r.eos_in_output)

    def test_summarize(self):
        encode = lambda s: [ord(c) for c in s]  # noqa: E731
        rows = [
            measure_row(i, "q", "b" * (10 * (i + 1)), encode, eos_id=999, max_len=25,
                        prefix_builder=build_prefix)
            for i in range(3)
        ]
        s = summarize(rows, max_len=25)
        self.assertEqual(s["n_rows"], 3)
        self.assertIn("coverage_max_len", s)
        self.assertEqual(s["n_boundary_mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
