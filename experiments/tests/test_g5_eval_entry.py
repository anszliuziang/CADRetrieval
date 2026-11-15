"""Targeted tests for the unified G5 evaluation entry (plan
IDEA01_G5_EXTERNAL_BASELINES_2026-11-15.md section 2, deliverable 3).

Synthetic universe: 3 unique texts, 4 L0 queries (one duplicate-text group),
12 CADs. Covers duplicate-text recovery, exact ties, row/col misalignment,
missing/non-finite scores, an external-format stub through the SAME entry,
sub-library mode, and the shared-draw paired bootstrap. CPU only; no model;
reuses the audited g5.link_inputs to build the protocol so the synthetic
inputs obey the exact frozen linkage rules.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2cad_g5_eval_entry import (  # noqa: E402
    EntryError, ScoreSet, evaluate, load_frozen_protocol)

N_CAD = 12
CAD_IDS = [f"c{i:03d}" for i in range(N_CAD)]
TEXTS = ["alpha bench", "bracket delta", "circuit 200"]
QUERIES = [  # (query_id, text, positive cad_id)  — texts 0 appears twice
    ("q0", TEXTS[0], "c003"),
    ("q1", TEXTS[0], "c007"),
    ("q2", TEXTS[1], "c000"),
    ("q3", TEXTS[2], "c011"),
]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _build_protocol():
    """Synthetic frozen protocol with the exact linkage g5.link_inputs demands."""
    # unique-text axis must be in sorted (sha, text) canonical order
    order = sorted(range(len(TEXTS)), key=lambda i: (_sha(TEXTS[i]), TEXTS[i]))
    idx_of = {TEXTS[i]: rank for rank, i in enumerate(order)}
    texts = [dict(text_index=idx_of[t], text=t, text_sha256=_sha(t), n_chars=len(t))
             for t in TEXTS]
    gallery = [dict(cad_id=cid, cad_seq_member=f"m{i}") for i, cid in enumerate(CAD_IDS)]
    store = [dict(cad_index=i, member=f"m{i}") for i in range(N_CAD)]
    lengths = [dict(cad_index=i, n_valid_x=10 + i, n_valid_y=5, n_real_tokens=16 + i)
               for i in range(N_CAD)]
    queries, qrels, q2t = [], [], {}
    for qid, text, cad_id in QUERIES:
        queries.append(dict(query_id=qid, text=text, level="L0",
                            text_sha256=_sha(text), n_chars=len(text)))
        qrels.append(dict(query_id=qid, relevance=1, cad_id=cad_id))
        q2t[qid] = idx_of[text]
    mapping = dict(cad_index_to_cad_id=list(CAD_IDS), query_to_text=q2t)
    linked = dict(queries=queries, gallery=gallery, qrels=qrels, texts=texts,
                  store=store, lengths=lengths)
    return linked, mapping, idx_of


def _linked_dict(linked_kwargs, mapping):
    """Wrap synthetic inputs through the audited link_inputs linkage rules."""
    import text2cad_g5_cpu as g5
    return g5.link_inputs(**linked_kwargs, mapping=mapping)


def _protocol():
    linked_kwargs, mapping, idx_of = _build_protocol()
    linked = _linked_dict(linked_kwargs, mapping)
    return linked, idx_of


def _cad_index_of(cad_id: str) -> int:
    return CAD_IDS.index(cad_id)


def _score_set_for_texts(linked, idx_of, rows_by_text, direction="higher", name="stub",
                         cad_axis=None):
    """rows_by_text: {text: score list over cad_axis (default cad_index 0..11)}.

    Iterates rows_by_text, so a missing text is a real omission (the entry
    must refuse it) rather than a KeyError in the helper."""
    axis = list(range(N_CAD)) if cad_axis is None else list(cad_axis)
    return ScoreSet(direction=direction, dtype="float64",
                    meta=dict(name=name, format="synthetic"),
                    rows=((idx_of[t], axis, list(scores))
                          for t, scores in rows_by_text.items()))


class TestEvalEntry(unittest.TestCase):
    def setUp(self):
        self.linked, self.idx_of = _protocol()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    # -- 1. duplicate text recovery --------------------------------------
    def test_duplicate_text_queries_share_ranking(self):
        # text 0: make c007 best, c003 second best, others worse
        scores = [10.0 - i for i in range(N_CAD)]
        scores[_cad_index_of("c007")] = 99.0
        scores[_cad_index_of("c003")] = 50.0
        ss = _score_set_for_texts(self.linked, self.idx_of, {t: scores for t in TEXTS})
        out = Path(self.tmp.name) / "dup"
        res = evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                       output=out, method_meta=dict(name="stub"))
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q0"]["text_index"], qr["q1"]["text_index"])
        self.assertEqual(qr["q0"]["cad_id"], "c003")
        self.assertEqual(qr["q1"]["cad_id"], "c007")
        self.assertEqual(qr["q0"]["ranks"]["stub"], 2)   # c003 second best
        self.assertEqual(qr["q1"]["ranks"]["stub"], 1)   # c007 best
        self.assertEqual(res["hit_counts"]["stub"]["hit@1"], 1)
        self.assertEqual(res["denominators"]["stub"], 4)

    # -- 2. exact ties -> cad_index ascending ----------------------------
    def test_exact_ties_rank_by_cad_index(self):
        scores = [0.0] * N_CAD
        ss = _score_set_for_texts(self.linked, self.idx_of, {t: scores for t in TEXTS})
        out = Path(self.tmp.name) / "ties"
        evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                 output=out, method_meta=dict(name="stub"))
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q0"]["ranks"]["stub"], _cad_index_of("c003") + 1)
        self.assertEqual(qr["q2"]["ranks"]["stub"], _cad_index_of("c000") + 1)
        self.assertEqual(qr["q3"]["ranks"]["stub"], _cad_index_of("c011") + 1)

    # -- 3. row/col misalignment -----------------------------------------
    def test_matrix_shape_misalignment_refused(self):
        with self.assertRaises(EntryError):
            ScoreSet.from_matrix(np.zeros((2, N_CAD)), [0, 1, 2], list(range(N_CAD)))
        with self.assertRaises(EntryError):
            ScoreSet.from_matrix(np.zeros((3, 5)), [0, 1, 2], list(range(N_CAD)))

    def test_jsonl_row_identity_mismatch_refused(self):
        path = Path(self.tmp.name) / "bad_axis.jsonl"
        header = dict(record_type="method_score_header", schema="text2cad-method-scores-v1",
                      method="stub", direction="higher", dtype="float64",
                      n_texts=3, n_cads_per_text=N_CAD)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(header) + "\n")
            for t in range(3):
                ci = list(range(N_CAD))
                sc = [1.0] * N_CAD
                if t == 1:
                    sc = sc[:-1]          # scores shorter than cad_indices
                f.write(json.dumps(dict(record_type="method_scores",
                                        text_index=t, cad_indices=ci, scores=sc)) + "\n")
        with self.assertRaises(EntryError):
            ScoreSet.from_jsonl(path)

    def test_jsonl_row_permuted_axis_placed_correctly(self):
        # row 2 declares a REVERSED cad axis with correspondingly reversed
        # scores — review fix 1: the entry places scores at their declared
        # cad's canonical position, so ranks must equal the dense-order run
        path = Path(self.tmp.name) / "perm.jsonl"
        header = dict(record_type="method_score_header", schema="text2cad-method-scores-v1",
                      method="stub", direction="higher", dtype="float64",
                      n_texts=3, n_cads_per_text=N_CAD)
        scores = [10.0 - i for i in range(N_CAD)]
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(header) + "\n")
            for t in range(3):
                ci, sc = list(range(N_CAD)), list(scores)
                if t == 2:
                    ci = list(reversed(ci))
                    sc = list(reversed(sc))
                f.write(json.dumps(dict(record_type="method_scores",
                                        text_index=t, cad_indices=ci, scores=sc)) + "\n")
        ss = ScoreSet.from_jsonl(path)
        out = Path(self.tmp.name) / "perm_eval"
        evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                 output=out, method_meta=dict(name="stub"))
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q2"]["ranks"]["stub"], 1)      # c000 best either way
        self.assertEqual(qr["q0"]["ranks"]["stub"], 4)
        self.assertEqual(qr["q1"]["ranks"]["stub"], 8)
        self.assertEqual(qr["q3"]["ranks"]["stub"], 12)

    def test_jsonl_row_set_mismatch_refused(self):
        # a row that DROPS a cad from its identity set is a real misalignment
        path = Path(self.tmp.name) / "bad_set.jsonl"
        header = dict(record_type="method_score_header", schema="text2cad-method-scores-v1",
                      method="stub", direction="higher", dtype="float64",
                      n_texts=3, n_cads_per_text=N_CAD)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(header) + "\n")
            for t in range(3):
                ci, sc = list(range(N_CAD)), [1.0] * N_CAD
                if t == 2:
                    ci = ci[:-1]
                    sc = sc[:-1]
                f.write(json.dumps(dict(record_type="method_scores",
                                        text_index=t, cad_indices=ci, scores=sc)) + "\n")
        with self.assertRaises(EntryError):
            ScoreSet.from_jsonl(path)      # header n_cads_per_text mismatch

    def test_unsorted_axis_tie_break_and_kept_ceilings(self):
        # review fix 1 + 2 in one: axis declared [3, 0] with EQUAL scores —
        # canonical order [0, 3] means c000 ranks 1 and c003 ranks 2; and the
        # duplicate ceiling must use KEPT group sizes (each kept group has
        # exactly 1 query here -> R@1 ceiling 100%, not the original-library
        # 2-query group's 50%)
        subset = [3, 0]                    # unsorted on purpose
        scores = [5.0, 5.0]                # exact tie on the unsorted axis
        ss = _score_set_for_texts(self.linked, self.idx_of,
                                  {t: scores for t in TEXTS}, cad_axis=subset)
        out = Path(self.tmp.name) / "ties_subset"
        res = evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                       output=out, method_meta=dict(name="stub"), cad_subset=subset)
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q2"]["ranks"]["stub"], 1)          # cad 0 wins the tie
        self.assertEqual(qr["q0"]["ranks"]["stub"], 2)          # cad 3 second
        self.assertEqual(res["duplicate_uid_ceilings"]["R@1"], 1.0)
        self.assertEqual(res["axis"]["cad_subset"], [0, 3])     # saved canonically
        self.assertIsNotNone(res["axis"]["cad_subset_sha256"])
        self.assertEqual(res["kept_queries"], 2)
        self.assertEqual(res["dropped_queries"], 2)

    def test_paired_name_collision_refused(self):
        scores = [1.0] * N_CAD
        s_method = _score_set_for_texts(self.linked, self.idx_of,
                                        {t: scores for t in TEXTS}, name="same")
        s_sum = _score_set_for_texts(self.linked, self.idx_of,
                                     {t: scores for t in TEXTS}, name="same")
        with self.assertRaises(EntryError):
            evaluate(s_method, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "collide", method_meta=dict(name="same"),
                     paired_sum_set=s_sum)

    def test_nonintegral_index_refused(self):
        ss = ScoreSet(direction="higher", dtype="float64", meta=dict(name="frac"),
                      rows=((1.5, list(range(N_CAD)), [1.0] * N_CAD),))
        with self.assertRaises(EntryError):
            evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "frac_text",
                     method_meta=dict(name="frac"))
        ss2 = _score_set_for_texts(self.linked, self.idx_of,
                                   {t: [1.0] * N_CAD for t in TEXTS})
        ss2._rows = ((self.idx_of[t], [0, 1.5] + list(range(2, N_CAD)), [1.0] * N_CAD)
                     for t in TEXTS)
        with self.assertRaises(EntryError):
            evaluate(ss2, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "frac_cad",
                     method_meta=dict(name="frac"))

    def test_binary_artifact_roundtrip_and_tamper(self):
        from text2cad_g5_eval_entry import ARTIFACT_SIDECAR_SCHEMA
        d = Path(self.tmp.name) / "artifact"
        d.mkdir()
        npy, side = d / "m.f64", d / "m.sidecar.json"
        scores = [10.0 - i for i in range(N_CAD)]
        written = ScoreSet.write_binary_artifact(
            np.array([scores] * 3, dtype=np.float64), side, npy,
            direction="higher", text_indices=[0, 1, 2], cad_indices=list(range(N_CAD)))
        self.assertEqual(written["schema"], ARTIFACT_SIDECAR_SCHEMA)
        ss = ScoreSet.from_binary_artifact(npy, side)
        out = Path(self.tmp.name) / "artifact_eval"
        res = evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                       output=out, method_meta=dict(name="artifact"))
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q2"]["ranks"]["artifact"], 1)
        self.assertEqual(res["fingerprints"]["score_source"]["format"], "binary_artifact")
        # tampering with the matrix bytes must be refused via the sidecar pin
        raw = bytearray(npy.read_bytes())
        raw[0] ^= 0xFF
        npy.write_bytes(bytes(raw))
        with self.assertRaises(EntryError):
            ScoreSet.from_binary_artifact(npy, side)

    def test_binary_artifact_shape_mismatch_refused(self):
        d = Path(self.tmp.name) / "artifact_bad"
        d.mkdir()
        npy, side = d / "m.f64", d / "m.sidecar.json"
        ScoreSet.write_binary_artifact(
            np.zeros((3, N_CAD)), side, npy,
            direction="higher", text_indices=[0, 1, 2], cad_indices=list(range(N_CAD)))
        side_obj = json.loads(side.read_text(encoding="utf-8"))
        side_obj["cad_indices"] = list(range(N_CAD - 1))   # axis implies fewer cells
        side.write_text(json.dumps(side_obj), encoding="utf-8")
        with self.assertRaises(EntryError):
            ScoreSet.from_binary_artifact(npy, side)

    # -- 4. missing / non-finite ------------------------------------------
    def test_missing_text_refused(self):
        ss = _score_set_for_texts(self.linked, self.idx_of,
                                  {TEXTS[0]: [1.0] * N_CAD, TEXTS[1]: [1.0] * N_CAD})
        with self.assertRaises(EntryError):
            evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "missing", method_meta=dict(name="stub"))

    def test_nonfinite_refused(self):
        bad = [float("nan")] * N_CAD
        ss = _score_set_for_texts(self.linked, self.idx_of, {t: bad for t in TEXTS})
        with self.assertRaises(EntryError):
            evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "nan", method_meta=dict(name="stub"))
        inf = [float("inf")] + [1.0] * (N_CAD - 1)
        ss2 = _score_set_for_texts(self.linked, self.idx_of, {t: inf for t in TEXTS})
        with self.assertRaises(EntryError):
            evaluate(ss2, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "inf", method_meta=dict(name="stub"))

    # -- 5. external-format stub through the SAME entry --------------------
    def test_external_jsonl_stub_lower_better(self):
        # direction='lower': c000 best ... c011 worst. positives: q2 (c000)
        # hits @1; q0 (c003) rank 4; q1 (c007) rank 8; q3 (c011) rank 12.
        path = Path(self.tmp.name) / "stub_scores.jsonl"
        header = dict(record_type="method_score_header", schema="text2cad-method-scores-v1",
                      method="stub_lower", direction="lower", dtype="float64",
                      n_texts=3, n_cads_per_text=N_CAD, meta=dict(note="stub"))
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(header) + "\n")
            for t in range(3):
                f.write(json.dumps(dict(record_type="method_scores", text_index=t,
                                        cad_indices=list(range(N_CAD)),
                                        scores=[float(i) for i in range(N_CAD)])) + "\n")
        ss = ScoreSet.from_jsonl(path)
        self.assertEqual(ss.direction, "lower")
        out = Path(self.tmp.name) / "stub_eval"
        res = evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                       output=out, method_meta=dict(name="stub_lower"))
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q2"]["ranks"]["stub_lower"], 1)
        self.assertEqual(qr["q0"]["ranks"]["stub_lower"], 4)
        self.assertEqual(qr["q1"]["ranks"]["stub_lower"], 8)
        self.assertEqual(qr["q3"]["ranks"]["stub_lower"], 12)
        self.assertEqual(res["hit_counts"]["stub_lower"],
                         {"hit@1": 1, "hit@5": 2, "hit@10": 3, "hit@20": 4})
        self.assertTrue("file_sha256" in res["fingerprints"]["score_source"])

    # -- 6. sub-library mode ------------------------------------------------
    def test_subset_mode_filters_queries_and_reranks(self):
        subset = list(range(6))          # c000..c005 stay; c007/c011 dropped
        scores = [10.0 - i for i in range(N_CAD)][:6]
        ss = _score_set_for_texts(self.linked, self.idx_of, {t: scores for t in TEXTS},
                                  cad_axis=subset)
        out = Path(self.tmp.name) / "subset"
        res = evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                       output=out, method_meta=dict(name="stub"), cad_subset=subset)
        self.assertEqual(res["kept_queries"], 2)      # q0 (c003) and q2 (c000)
        self.assertEqual(res["dropped_queries"], 2)
        qr = {json.loads(l)["query_id"]: json.loads(l)
              for l in (out / "query_ranks.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(qr["q2"]["ranks"]["stub"], 1)          # c000 best in subset
        self.assertEqual(qr["q0"]["ranks"]["stub"], 4)          # c003 4th in subset
        self.assertNotIn("q1", qr)
        self.assertNotIn("q3", qr)
        self.assertTrue(res["axis"]["full_library"] is False)

    def test_subset_mismatch_refused(self):
        # score rows declare the FULL dense axis while cad_subset declares a
        # 6-cad sub-library — refused
        scores = [1.0] * N_CAD
        ss = _score_set_for_texts(self.linked, self.idx_of, {t: scores for t in TEXTS})
        with self.assertRaises(EntryError):
            evaluate(ss, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "badsubset", method_meta=dict(name="s"),
                     cad_subset=list(range(6)))

    # -- 7. paired arm + shared-draw bootstrap ------------------------------
    def test_paired_bootstrap_deterministic(self):
        good = [10.0 - i for i in range(N_CAD)]
        bad = [float(i) for i in range(N_CAD)]      # reversed quality
        s_method = _score_set_for_texts(self.linked, self.idx_of,
                                        {t: good for t in TEXTS}, name="good")
        s_sum = _score_set_for_texts(self.linked, self.idx_of,
                                     {t: bad for t in TEXTS}, name="sum_ref")
        boot = dict(seed=1234, replicates=50)
        common = dict(protocol=dict(linked=self.linked, fingerprints={}),
                      method_meta=dict(name="good"), paired_sum_set=s_sum, bootstrap=boot)
        out1 = Path(self.tmp.name) / "boot1"
        res1 = evaluate(s_method, output=out1, **common)
        out2 = Path(self.tmp.name) / "boot2"
        res2 = evaluate(s_method, output=out2, **common)
        st = res1["statistics"]
        self.assertEqual(st["paired_vs"], "sum_ref")
        self.assertEqual(len(st["paired_ci95"]["R@1"]["ci95"]), 2)
        self.assertEqual(res1["statistics"]["draw_counts_sha256"],
                         res2["statistics"]["draw_counts_sha256"])
        self.assertEqual(st["paired_ci95"], res2["statistics"]["paired_ci95"])
        # method arm (higher_better, descending quality) must beat the
        # reversed arm on MedR; R@20 is trivially 1.0 on both sides of a
        # 12-cad gallery so it cannot separate them
        self.assertLess(st["paired_ci95"]["MedR"]["point"], 0.0)
        self.assertGreater(st["paired_ci95"]["MedR"]["ci95"][1],
                           st["paired_ci95"]["MedR"]["ci95"][0])

    def test_paired_axis_mismatch_refused(self):
        good = [1.0] * N_CAD
        s_method = _score_set_for_texts(self.linked, self.idx_of, {t: good for t in TEXTS})
        # paired arm declares a different cad axis
        s_sum = ScoreSet(direction="higher", dtype="float64", meta=dict(name="sum_ref"),
                         rows=((self.idx_of[t], list(range(6)), [1.0] * 6) for t in TEXTS))
        with self.assertRaises(EntryError):
            evaluate(s_method, protocol=dict(linked=self.linked, fingerprints={}),
                     output=Path(self.tmp.name) / "bootbad", method_meta=dict(name="good"),
                     paired_sum_set=s_sum)


if __name__ == "__main__":
    unittest.main()
