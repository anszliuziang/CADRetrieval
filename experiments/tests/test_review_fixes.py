"""Tests for Codex review 2026-09-25 fixes:
- run_p0_smoke: pair-id parsing (dedupe/cap/range), stop policy
- envspec: canonical JSON vs file-bytes hashing
- supplement: input-id reconstruction + cross-check against original evidence
- make_manifest: non-self-referencing manifest
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from idea01.envspec import canonical_json_sha256, file_bytes_sha256  # noqa: E402
from idea01.supplement import crosscheck_pair, query_for_condition, reconstruct_input_ids  # noqa: E402
from make_manifest import build_manifest_lines  # noqa: E402
from run_p0_smoke import MAX_PAIRS, should_stop, parse_pair_ids  # noqa: E402


class TestParsePairIds(unittest.TestCase):
    def test_all(self):
        self.assertEqual(parse_pair_ids("all", 60), list(range(60)))

    def test_single_and_range(self):
        self.assertEqual(parse_pair_ids("0", 60), [0])
        self.assertEqual(parse_pair_ids("1-59", 60), list(range(1, 60)))
        self.assertEqual(parse_pair_ids("0,3,5-7", 60), [0, 3, 5, 6, 7])

    def test_dedupes_preserving_order(self):
        self.assertEqual(parse_pair_ids("2,2,1-3,3", 60), [2, 1, 3])

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_pair_ids("60", 60)
        with self.assertRaises(ValueError):
            parse_pair_ids("-1", 60)

    def test_cap_enforced(self):
        with self.assertRaises(ValueError):
            parse_pair_ids("all", 100)  # 100 unique > MAX_PAIRS
        self.assertEqual(len(parse_pair_ids("all", MAX_PAIRS)), MAX_PAIRS)


class TestStopPolicy(unittest.TestCase):
    def test_ok_and_unsupported_do_not_stop(self):
        self.assertFalse(should_stop("ok"))
        self.assertFalse(should_stop("unsupported_full_sequence"))

    def test_anomalies_stop(self):
        for s in ("boundary_error", "nonfinite", "error"):
            self.assertTrue(should_stop(s), s)


class TestEnvSpecHashes(unittest.TestCase):
    def test_canonical_order_independent(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.json"
            p2 = Path(td) / "b.json"
            p1.write_text('{"x": 1, "y": {"b": 2, "a": 1}}', encoding="utf-8")
            p2.write_text('{ "y": {"a": 1, "b": 2}, "x": 1 }', encoding="utf-8")
            self.assertEqual(canonical_json_sha256(p1), canonical_json_sha256(p2))

    def test_file_bytes_depends_on_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.json"
            p.write_bytes(b'{"x":1}')
            h1 = file_bytes_sha256(p)
            p.write_bytes(b'{"x":1}\n')
            self.assertNotEqual(file_bytes_sha256(p), h1)

    def test_known_env_spec_hashes(self):
        # Review-documented values for the pinned env spec (same content, two algorithms)
        repo = Path(__file__).resolve().parents[2]
        p = repo / "experiments" / "configs" / "idea01_yunnan_share2_env.json"
        canon = canonical_json_sha256(p)
        self.assertTrue(canon.startswith("10c18c17"), canon)
        blob = subprocess.run(["git", "show", f"HEAD:experiments/configs/idea01_yunnan_share2_env.json"],
                              cwd=repo, capture_output=True, check=True).stdout
        import hashlib

        self.assertTrue(hashlib.sha256(blob).hexdigest().startswith("b6cdd884"))


def fake_encode(s: str) -> list[int]:
    return [ord(c) % 5000 + 10 for c in s]


class TestSupplement(unittest.TestCase):
    EOS = 151645

    def _rec_and_tokens(self, input_ids, target_start):
        rec = {
            "n_prefix_tokens": target_start,
            "n_input_tokens": len(input_ids),
            "n_target_tokens": len(input_ids) - target_start,
        }
        tokens = [
            {"position": target_start + k, "pred_row": target_start + k - 1, "token_id": tid}
            for k, tid in enumerate(input_ids[target_start:])
        ]
        return rec, tokens

    def test_query_for_condition(self):
        self.assertEqual(query_for_condition("correct", "own", "other"), "own")
        self.assertEqual(query_for_condition("mismatched", "own", "other"), "other")
        self.assertEqual(query_for_condition("empty", "own", "other"), "")
        with self.assertRaises(ValueError):
            query_for_condition("mismatched", "own", None)
        with self.assertRaises(ValueError):
            query_for_condition("bogus", "own", None)

    def test_reconstruct_and_crosscheck_pass(self):
        ids, ts, _ = reconstruct_input_ids("q", "import cadquery as cq\n", fake_encode, self.EOS)
        self.assertEqual(ids[-1], self.EOS)
        rec, tokens = self._rec_and_tokens(ids, ts)
        self.assertEqual(crosscheck_pair(ids, ts, self.EOS, rec, tokens), [])

    def test_crosscheck_detects_token_mismatch(self):
        ids, ts, _ = reconstruct_input_ids("q", "body", fake_encode, self.EOS)
        rec, tokens = self._rec_and_tokens(ids, ts)
        tokens[0]["token_id"] += 1  # corrupt evidence
        errs = crosscheck_pair(ids, ts, self.EOS, rec, tokens)
        self.assertTrue(any("target[0]" in e for e in errs))

    def test_crosscheck_detects_bad_counts_and_eos(self):
        ids, ts, _ = reconstruct_input_ids("q", "body", fake_encode, self.EOS)
        rec, tokens = self._rec_and_tokens(ids, ts)
        rec["n_input_tokens"] += 1
        tokens[-1]["token_id"] = 42  # not EOS
        errs = crosscheck_pair(ids, ts, self.EOS, rec, tokens)
        self.assertTrue(any("n_input_tokens" in e for e in errs))
        self.assertTrue(any("eos" in e for e in errs))

    def test_crosscheck_detects_shift_violation(self):
        ids, ts, _ = reconstruct_input_ids("q", "body", fake_encode, self.EOS)
        rec, tokens = self._rec_and_tokens(ids, ts)
        tokens[1]["pred_row"] = tokens[1]["position"]  # off by one
        errs = crosscheck_pair(ids, ts, self.EOS, rec, tokens)
        self.assertTrue(any("pred_row" in e for e in errs))


class TestManifest(unittest.TestCase):
    def test_no_self_reference_and_format(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sub").mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "sub" / "b.txt").write_text("beta", encoding="utf-8")
            (root / "MANIFEST_SHA256.txt").write_text("old manifest content", encoding="utf-8")
            lines = build_manifest_lines(root, "MANIFEST_SHA256_v2.txt")
            self.assertEqual(len(lines), 3)  # a, sub/b, old manifest included
            for ln in lines:
                hexpart, path = ln.split("  ")
                self.assertEqual(len(hexpart), 64)
                self.assertTrue(path.startswith("./"))
                self.assertNotIn("MANIFEST_SHA256_v2", path)
            # old manifest is covered as ordinary content
            self.assertTrue(any("MANIFEST_SHA256.txt" in ln for ln in lines))
            # hashes verify
            import hashlib

            for ln in lines:
                hexpart, path = ln.split("  ")
                actual = hashlib.sha256((root / path[2:]).read_bytes()).hexdigest()
                self.assertEqual(hexpart, actual)


if __name__ == "__main__":
    unittest.main()
