"""Tests for the P1 pre-registered analysis (plan sec.3) on synthetic scores."""

import json
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.p1_analysis import (  # noqa: E402
    AnalysisError,
    baseline_empty,
    baseline_length,
    baseline_random,
    baseline_tfidf,
    bootstrap_ci,
    char_ngrams,
    empty_score_map,
    near_duplicate_screen,
    program_token_set,
    rank_of_self,
    ranking_analysis,
    same_candidate_margins,
    score_lookup,
    tfidf_cosine,
    validate_pair_records,
)
from idea01.p1_selection import build_pair_plan, select_p1  # noqa: E402
from idea01.p1_verify import EOS_ID  # noqa: E402
from test_p1_selection import mkpool  # noqa: E402


def synthetic_selection(n_queries=8, n_offsets=3, n_method=1, n_length=1, pool=60):
    recs = mkpool(pool)
    flags = [True] * len(recs)
    sel = select_p1(recs, flags, n_queries=n_queries, n_offsets=n_offsets, n_method=n_method, n_length=n_length)
    pairs = build_pair_plan(sel)
    # manifest-shaped dict, as run_p1_analysis / the analysis functions consume it
    import dataclasses

    manifest = {
        "offsets": sel.offsets,
        "distinct_candidate_uids": sel.distinct_candidate_uids,
        "queries": [dataclasses.asdict(q) for q in sel.queries],
        "pairs": pairs,
    }
    return manifest


def synthetic_records(manifest, own_bonus=100.0):
    """Score fabricator: own pair = base(candidate) + own_bonus; everything
    deterministic from uid strings so tests are reproducible. Records carry
    input_ids so per-token evidence files can be fabricated consistently."""
    def base(uid):
        return -50.0 - (sum(ord(c) for c in uid) % 37)

    recs = []
    for p in manifest["pairs"]:
        if p["kind"] == "empty":
            s = base(p["candidate_uid"]) - 20.0  # empty condition is worst
        else:
            s = base(p["candidate_uid"]) + (own_bonus if p["query_uid"] == p["candidate_uid"] else 0.0)
        ids = [1000 + k for k in range(10)] + [2000 + k for k in range(9)] + [EOS_ID]
        recs.append(
            {
                "pair_id": p["pair_id"],
                "kind": p["kind"],
                "query_uid": p["query_uid"],
                "candidate_uid": p["candidate_uid"],
                "candidate_source": p["candidate_source"],
                "status": "ok",
                "all_finite": True,
                "sum_logprob": s,
                "mean_logprob": s / 10.0,
                "n_prefix_tokens": 10,
                "n_input_tokens": len(ids),
                "n_target_tokens": 10,
                "input_ids": ids,
                "eos_logprob": s / 10.0,
            }
        )
    return recs


def write_token_evidence(tokens_dir, rec):
    """Fabricate the per-token evidence file consistent with rec (needed since
    run_p1_analysis verifies raw evidence before analyzing, R2)."""
    tokens_dir.mkdir(parents=True, exist_ok=True)
    per_tok = rec["sum_logprob"] / rec["n_target_tokens"]
    with open(tokens_dir / f"{rec['pair_id']}.jsonl", "w", encoding="utf-8") as f:
        for k in range(rec["n_target_tokens"]):
            f.write(
                json.dumps(
                    {
                        "position": rec["n_prefix_tokens"] + k,
                        "pred_row": rec["n_prefix_tokens"] + k - 1,
                        "token_id": rec["input_ids"][rec["n_prefix_tokens"] + k],
                        "logprob": per_tok,
                    }
                )
                + "\n"
            )


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.manifest = synthetic_selection()
        self.recs = synthetic_records(self.manifest)
        self.expected = [p["pair_id"] for p in self.manifest["pairs"]]

    def test_ok(self):
        validate_pair_records(self.recs, self.expected)

    def test_missing_pair(self):
        with self.assertRaises(AnalysisError):
            validate_pair_records(self.recs[:-1], self.expected)

    def test_duplicate_pair(self):
        with self.assertRaises(AnalysisError):
            validate_pair_records(self.recs + [self.recs[0]], self.expected)

    def test_failed_status_rejected(self):
        self.recs[0] = {**self.recs[0], "status": "error"}
        with self.assertRaises(AnalysisError):
            validate_pair_records(self.recs, self.expected)


class TestBootstrap(unittest.TestCase):
    def test_deterministic_same_seed(self):
        vals = [float(i) for i in range(32)]
        a = bootstrap_ci(vals, statistics.mean, n_resamples=500, seed=1)
        b = bootstrap_ci(vals, statistics.mean, n_resamples=500, seed=1)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_lo"], a["point"])
        self.assertLessEqual(a["point"], a["ci_hi"])

    def test_seed_changes_ci(self):
        vals = [float(i) for i in range(32)]
        a = bootstrap_ci(vals, statistics.mean, n_resamples=500, seed=1)
        b = bootstrap_ci(vals, statistics.mean, n_resamples=500, seed=2)
        self.assertNotEqual((a["ci_lo"], a["ci_hi"]), (b["ci_lo"], b["ci_hi"]))


class TestRankOfSelf(unittest.TestCase):
    def test_basic(self):
        rank, top1, win = rank_of_self([10.0, 9.0, 11.0], 0)
        self.assertEqual(rank, 2)
        self.assertFalse(top1)
        self.assertAlmostEqual(win, 0.5)

    def test_strict_top1_and_ties(self):
        rank, top1, win = rank_of_self([10.0, 10.0, 9.0], 0)
        self.assertEqual(rank, 1)          # no strictly greater
        self.assertFalse(top1)             # but not STRICTLY greater than all
        self.assertAlmostEqual(win, 0.75)  # tie counts 0.5


class TestMarginsAndRanking(unittest.TestCase):
    def setUp(self):
        self.manifest = synthetic_selection()
        self.recs = synthetic_records(self.manifest, own_bonus=100.0)
        self.table = score_lookup(self.recs)
        self.empty = empty_score_map(self.recs)

    def test_margins_all_positive(self):
        m = same_candidate_margins(self.table, self.manifest)
        self.assertEqual(m["n"], 8)
        self.assertEqual(m["n_positive"], 8)
        for p in m["per_candidate"]:
            self.assertAlmostEqual(p["margin"], 100.0)
            self.assertEqual(len(p["other_sums"]), 3)
        self.assertAlmostEqual(m["mean_margin"]["point"], 100.0)

    def test_margin_uses_exactly_offset_queries(self):
        # corrupt one offset score -> margin changes; corrupt nothing else does
        m0 = same_candidate_margins(self.table, self.manifest)
        uid = m0["per_candidate"][0]["candidate_uid"]
        off_pair = next(
            p for p in self.manifest["pairs"]
            if p["candidate_uid"] == uid and p["candidate_source"] == "selected-offset"
        )
        for r in self.recs:
            if r["pair_id"] == off_pair["pair_id"]:
                r["sum_logprob"] += 30.0
        m1 = same_candidate_margins(self.table, self.manifest)
        self.assertAlmostEqual(m0["per_candidate"][0]["margin"] - 10.0, m1["per_candidate"][0]["margin"])

    def test_ranking_all_arms(self):
        for arm in ("sum", "mean", "corrected"):
            ra = ranking_analysis(self.table, self.manifest, self.empty, arm, n_resamples=200, seed=7)
            self.assertEqual(ra["top1_strict_rate"]["point"], 1.0, arm)
            self.assertEqual(ra["mean_rank"], 1.0, arm)
            self.assertEqual(ra["mean_win_rate"]["point"], 1.0, arm)
            self.assertEqual(len(ra["per_query"]), 8)

    def test_corrected_arm_uses_empty(self):
        # inflate one candidate's empty prior: raw sum arm must be unaffected,
        # corrected arm must demote that candidate to the bottom of the query
        q0 = self.manifest["queries"][0]
        cand1 = q0["candidates"][1]["uid"]
        ra_sum = ranking_analysis(self.table, self.manifest, self.empty, "sum", n_resamples=100, seed=3)
        for r in self.recs:
            if r["kind"] == "empty" and r["candidate_uid"] == cand1:
                r["sum_logprob"] = 1e6
        empty2 = empty_score_map(self.recs)
        table2 = score_lookup(self.recs)
        ra_sum2 = ranking_analysis(table2, self.manifest, empty2, "sum", n_resamples=100, seed=3)
        ra_cor = ranking_analysis(table2, self.manifest, empty2, "corrected", n_resamples=100, seed=3)
        # raw sum arm unchanged by empty-score inflation
        self.assertEqual(ra_sum["per_query"][0], ra_sum2["per_query"][0])
        # self remains strictly top-1 under the corrected arm (own bonus dominates)
        self.assertTrue(ra_cor["per_query"][0]["top1_strict"])
        # and the inflated candidate's corrected score is the minimum of the 8
        scores = {}
        for c in q0["candidates"]:
            rec = table2[("conditional", q0["uid"], c["uid"])]
            scores[c["uid"]] = rec["sum_logprob"] - empty2[c["uid"]]
        self.assertEqual(min(scores, key=scores.get), cand1)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.manifest = synthetic_selection()

    def test_random_deterministic(self):
        a = baseline_random(self.manifest, seed=11)
        b = baseline_random(self.manifest, seed=11)
        self.assertEqual(a["per_query"], b["per_query"])
        ranks = [p["rank"] for p in a["per_query"]]
        self.assertTrue(all(1 <= r <= 8 for r in ranks))

    def test_length_orders_shortest_first(self):
        res = baseline_length(self.manifest)
        for p, qp in zip(res["per_query"], self.manifest["queries"]):
            own = qp["candidates"][0]["n_response_tokens"]
            others = [c["n_response_tokens"] for c in qp["candidates"][1:]]
            self.assertEqual(p["rank"], 1 + sum(x < own for x in others))
            self.assertEqual(p["top1_strict"], all(own < x for x in others))
            self.assertAlmostEqual(p["win_rate"], sum(1 if own < x else .5 if own == x else 0 for x in others) / len(others))

    def test_query_independent_baselines_do_not_use_uid_to_break_score_ties(self):
        qp = self.manifest["queries"][0]
        uids = [c["uid"] for c in qp["candidates"]]
        # Force eight identical length scores; the paired UID must tie for first,
        # not be demoted according to its arbitrary identifier.
        for c in qp["candidates"]:
            c["n_response_tokens"] = 100
        one = {"queries": [qp], "distinct_candidate_uids": uids}
        length = baseline_length(one)["per_query"][0]
        self.assertEqual(length["rank"], 1)
        self.assertFalse(length["top1_strict"])
        self.assertAlmostEqual(length["win_rate"], 0.5)
        empty = baseline_empty(one, {uid: -100.0 for uid in uids})["per_query"][0]
        self.assertEqual(empty["rank"], 1)
        self.assertFalse(empty["top1_strict"])
        self.assertAlmostEqual(empty["win_rate"], 0.5)
        tfidf = baseline_tfidf(one, {qp["uid"]: "ZZZ"}, {uid: "aaa" for uid in uids})["per_query"][0]
        self.assertEqual(tfidf["rank"], 1)
        self.assertFalse(tfidf["top1_strict"])
        self.assertAlmostEqual(tfidf["win_rate"], 0.5)

    def test_tfidf_self_top1_when_texts_match(self):
        # craft texts so each query shares unique 3-grams with its own program only
        query_texts = {}
        programs = {}
        for qp in self.manifest["queries"]:
            tag = qp["uid"]
            query_texts[tag] = f"make a part with zqxfeature{tag}zqx and holes"
            programs[tag] = f"import cadquery as cq\n# zqxfeature{tag}zqx\nresult = cq.Workplane().box(1,1,1)\n"
        for qp in self.manifest["queries"]:
            for c in qp["candidates"]:
                if c["uid"] not in programs:
                    programs[c["uid"]] = f"import cadquery as cq\nresult = cq.Workplane().box(2,2,2)  # {c['uid']}\n"
        res = baseline_tfidf(self.manifest, query_texts, programs)
        self.assertEqual(res["top1_strict_rate"]["point"], 1.0)
        self.assertEqual(res["n_docs_idf"], len(self.manifest["distinct_candidate_uids"]))

    def test_tfidf_cosine_math(self):
        idf = {"abc": 1.0}
        self.assertAlmostEqual(tfidf_cosine("abc", "abc", idf), 1.0)
        self.assertEqual(tfidf_cosine("abc", "xyz", idf), 0.0)
        self.assertEqual(char_ngrams("abcd", 3), ["abc", "bcd"])
        self.assertEqual(char_ngrams("ab", 3), ["ab"])


class TestNearDuplicate(unittest.TestCase):
    def test_exact_same_text_flags_admission_error(self):
        sel = {
            "queries": [
                {"uid": "a", "candidates": [{"uid": "a", "source": "self"}, {"uid": "b", "source": "method-neighbor"}]}
            ]
        }
        programs = {"a": "same program text", "b": "same program text"}
        res = near_duplicate_screen(sel, programs)
        self.assertEqual(res["n_exact_same_text"], 1)
        self.assertTrue(res["data_admission_error"])
        self.assertAlmostEqual(res["pairs"][0]["token_jaccard"], 1.0)

    def test_partial_overlap_sorted_desc(self):
        sel = {
            "queries": [
                {
                    "uid": "a",
                    "candidates": [
                        {"uid": "a", "source": "self"},
                        {"uid": "b", "source": "method-neighbor"},
                        {"uid": "c", "source": "length-neighbor"},
                    ],
                }
            ]
        }
        programs = {"a": "x = workplane box extrude", "b": "x = workplane box", "c": "totally different code"}
        res = near_duplicate_screen(sel, programs, threshold=0.5)
        self.assertEqual(res["n_pairs"], 2)
        self.assertGreater(res["pairs"][0]["token_jaccard"], res["pairs"][1]["token_jaccard"])
        self.assertFalse(res["data_admission_error"])

    def test_program_token_set(self):
        self.assertEqual(program_token_set("a.b(c)"), frozenset({"a", ".", "b", "(", "c", ")"}))


class TestEndToEndDriver(unittest.TestCase):
    def test_run_p1_analysis_main_on_synthetic_files(self):
        import json
        import tempfile
        from unittest import mock

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_p1_analysis  # noqa: E402

        manifest = synthetic_selection(pool=60)
        recs = synthetic_records(manifest)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data_dir"
            (data_dir / "data").mkdir(parents=True)
            with open(data_dir / "data" / "data_val.jsonl", "w", encoding="utf-8") as f:
                for i in range(60):
                    f.write(json.dumps({"input": f"query text {i}", "output": f"program body {i}"}) + "\n")
            sel_path = root / "p1_selection.json"
            sel_path.write_text(json.dumps({**manifest, "task_id": "synthetic"}), encoding="utf-8")
            pairs_path = root / "pairs.jsonl"
            with open(pairs_path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            for r in recs:
                write_token_evidence(root / "tokens", r)
            out_dir = root / "out"
            argv = [
                "run_p1_analysis.py",
                "--pairs", str(pairs_path),
                "--selection", str(sel_path),
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
                "--n-resamples", "200",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = run_p1_analysis.main()
            self.assertEqual(rc, 0)
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["primary_same_candidate_margin"]["n_positive"], 8)
            for arm in ("sum", "mean", "corrected"):
                self.assertEqual(metrics["ranking_by_arm"][arm]["top1_strict_rate"]["point"], 1.0)
            for b in ("random_fixed_seed", "response_length_ascending", "empty_query_score", "char3_tfidf_cosine"):
                self.assertIn(b, metrics["baselines"])
                self.assertIn("elapsed_s", metrics["baselines"][b])
            self.assertFalse(metrics["near_duplicate_screen"]["data_admission_error"])
            self.assertTrue((out_dir / "per_query_table.jsonl").exists())
            self.assertTrue((out_dir / "near_duplicates.jsonl").exists())

    def test_driver_rejects_incomplete_pairs(self):
        import json
        import tempfile
        from unittest import mock

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_p1_analysis  # noqa: E402

        manifest = synthetic_selection(pool=60)
        recs = synthetic_records(manifest)[:-1]  # drop the last pair
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data_dir"
            (data_dir / "data").mkdir(parents=True)
            with open(data_dir / "data" / "data_val.jsonl", "w", encoding="utf-8") as f:
                for i in range(60):
                    f.write(json.dumps({"input": f"query text {i}", "output": f"program body {i}"}) + "\n")
            sel_path = root / "p1_selection.json"
            sel_path.write_text(json.dumps(manifest), encoding="utf-8")
            pairs_path = root / "pairs.jsonl"
            with open(pairs_path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            for r in recs:
                write_token_evidence(root / "tokens", r)
            argv = [
                "run_p1_analysis.py",
                "--pairs", str(pairs_path),
                "--selection", str(sel_path),
                "--data-dir", str(data_dir),
                "--out-dir", str(root / "out"),
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = run_p1_analysis.main()
            self.assertEqual(rc, 2)  # success-only subset refused


if __name__ == "__main__":
    unittest.main()
