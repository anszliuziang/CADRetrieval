"""CPU tests for the Text2CAD G1/G2 benchmark + equivalence gates (M1c B2).

Covers, without real weights / NPU / network:

  * selection generation: determinism, stratification correctness against
    hand-computed ranks, shortfall bookkeeping, pair_index/coords coherence;
  * selection-vs-manifest verification: acceptance and every drift refusal;
  * full-matrix cell counts: exact 29,319,624 coverage;
  * ETA: exact-sum gate, missing/non-positive rate refusal, safety floor,
    the 63.0 pairs/s seven-day boundary (independently recomputed);
  * two-card combining: overlapping-window requirement, per-cell presence,
    no ideal-2x scaling of a single measurement;
  * compare_gate_results: pass within tolerance, logit/score/count/shape
    violations fail, and -- key non-mirror property -- differences OUTSIDE
    the valid (real-token target) rows are ignored by construction;
  * score_pairs_detailed + gates on a tiny random ported model: batch=1 vs
    batch>1 equality, trim=batch_max vs full-272 equality, BERT cache with
    zero-pad reuse vs direct pad-embedding path equality -- the masking
    chain is exercised end to end (the fake BERT's pad embedding is NOT
    zero, so equality can only hold if padding is truly masked);
  * run_throughput_tiers: exact budget charges matching
    plan_throughput_forwards, warmup floor enforcement, per-rep/per-cell
    timing records, record-file publishing, OOM escalation stop;
  * logits_stats finiteness.

Runs on CPU only with small random weights.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_frozen_runner import (  # noqa: E402
    RunnerConfig,
    ZipCandidateStore,
    assemble_batch,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_LENGTH_FILE,
    PAIR_MANIFEST_FILE,
    TEXTS_FILE,
    PairSpace,
    sha256_file,
    sha256_text,
)
from idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256, PINNED_CHECKPOINT_SHA256, build_text2cad  # noqa: E402
from idea01.text2cad_port.model import generate_attention_mask  # noqa: E402
from idea01.text2cad_port.text_embed import TextEmbedder  # noqa: E402
from idea01.text2cad_speed_bench import (  # noqa: E402
    BENCH_RESULT_SCHEMA_VERSION,
    ETA_SAFETY_FACTOR,
    FULL_MATRIX_PAIRS,
    G1_SCORE_ATOL,
    N_CAD_DECILES,
    N_TEXT_TERCILES,
    RunnerError,
    SEVEN_DAY_SECONDS,
    TEN_PAIR_QUANTILES,
    THROUGHPUT_PAIRS_PER_CELL,
    TWO_CARD_MIN_PAIRS_PER_S,
    combine_two_card_results,
    parse_batch_tiers,
    validate_eta_card_results,
    compare_gate_results,
    estimate_full_matrix_seconds,
    full_matrix_cell_counts,
    generate_selection,
    logits_stats,
    plan_throughput_forwards,
    run_throughput_tiers,
    score_pairs_detailed,
    verify_selection_against_manifest,
    _gated_setup,
)


class ForwardGateBoundaryTest(unittest.TestCase):
    def test_external_fingerprints_and_bad_budget_dtype_stop_before_manifest_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for fields in (
                {"expected_pair_hashes": str(root / "external.json")},
                {"cards_used": 2},
                {"torch_dtype": "bfloat16"},
                {"device": "cpu"},
            ):
                values = dict(repo_root=str(root), expected_pair_hashes=None,
                              cards_used=1, torch_dtype="float32", device="npu:0")
                values.update(fields)
                args = SimpleNamespace(**values)
                with self.subTest(fields=fields), mock.patch(
                    "idea01.text2cad_speed_bench.load_pair_manifest"
                ) as load:
                    with self.assertRaises(RunnerError):
                        _gated_setup(args)
                    load.assert_not_called()

    def test_g2_and_legacy_selection_paths_not_interchangeable(self):
        from idea01.text2cad_speed_bench import G2_SELECTION_CONFIG_REL, SELECTION_CONFIG_REL

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mocked_manifest = {"counts": {"n_unique_text": 5, "n_gallery": 37}}
            for g2, selection_rel, expected_name in (
                (True, SELECTION_CONFIG_REL, "text2cad_g2_pair_selection_20261001.json"),
                (False, G2_SELECTION_CONFIG_REL, "text2cad_g1g2_pair_selection_20260930.json"),
            ):
                args = SimpleNamespace(
                    repo_root=str(root), expected_pair_hashes=None, cards_used=1,
                    torch_dtype="float32", device="npu:0",
                    pair_manifest_dir=str(root), selection=str(root / selection_rel),
                )
                with self.subTest(g2=g2), mock.patch.multiple(
                    "idea01.text2cad_speed_bench",
                    _load_expected_pair_hashes=mock.Mock(return_value={}),
                    load_pair_manifest=mock.Mock(return_value=mocked_manifest),
                    load_texts=mock.Mock(return_value=[]),
                    load_cad_store=mock.Mock(return_value=[]),
                ):
                    with self.assertRaisesRegex(RunnerError, expected_name):
                        _gated_setup(args, g2=g2)

# Reuse the runner-test fixtures style (kept local for self-containedness).
from test_text2cad_frozen_runner import (  # noqa: E402
    FakeBatchEncoding,
    FakeTokenizer,
    _candidate_tensors,
    _make_zip,
)


# ---------------------------------------------------------------------------
# Synthetic stratification fixtures
# ---------------------------------------------------------------------------


def _text_records(n: int) -> list[dict]:
    # deterministic spread of lengths; some ties to exercise index tiebreaks
    out = []
    for i in range(n):
        n_chars = (i * 37) % 180 + 10 + (5 if i % 7 == 0 else 0)
        text = f"t{i} " + "x" * (n_chars - len(f"t{i} "))
        out.append({
            "text_index": i, "text": text, "text_sha256": sha256_text(text),
            "n_chars": len(text),
        })
    return out


def _cad_length_records(n: int) -> list[dict]:
    return [
        {"cad_index": i, "n_real_tokens": (i * 13) % 190 + 19,
         "n_valid_x": (i * 13) % 190 + 18, "n_valid_y": 2 + (i % 5)}
        for i in range(n)
    ]


class SelectionTest(unittest.TestCase):
    def test_deterministic_and_coherent(self):
        texts = _text_records(60)
        cads = _cad_length_records(40)
        a = generate_selection(texts, cads, texts_sha256="a", cad_length_sha256="b",
                               pair_manifest_sha256="c")
        b = generate_selection(texts, cads, texts_sha256="a", cad_length_sha256="b",
                               pair_manifest_sha256="c")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        space = PairSpace(60, 40)
        for entry in [a["spot"], *a["ten_pair"]]:
            self.assertEqual(
                space.coords_of(entry["pair_index"]),
                (entry["text_index"], entry["cad_index"]),
            )
        self.assertEqual(len(a["throughput_strata"]), N_TEXT_TERCILES * N_CAD_DECILES)
        total = sum(len(v) for v in a["throughput_strata"].values())
        self.assertEqual(total, a["counts"]["n_throughput_pairs"])
        for key, entries in a["throughput_strata"].items():
            ordered = [e["pair_index"] for e in entries]
            self.assertEqual(ordered, sorted(ordered))

    def test_quantile_picks_match_hand_computation(self):
        texts = _text_records(11)
        cads = _cad_length_records(9)
        sel = generate_selection(texts, cads, texts_sha256="a", cad_length_sha256="b",
                                 pair_manifest_sha256="c")
        n_chars = {r["text_index"]: r["n_chars"] for r in texts}
        n_real = {r["cad_index"]: r["n_real_tokens"] for r in cads}
        text_order = sorted(range(11), key=lambda i: (n_chars[i], i))
        cad_order = sorted(range(9), key=lambda i: (n_real[i], i))
        # spot = median x median
        self.assertEqual(sel["spot"]["text_index"], text_order[5])
        self.assertEqual(sel["spot"]["cad_index"], cad_order[4])
        # ten-pair table entries follow TEN_PAIR_QUANTILES exactly
        for entry, (tq, cq) in zip(sel["ten_pair"], TEN_PAIR_QUANTILES):
            self.assertEqual(
                entry["text_index"], text_order[round(10 * tq / 100)]
            )
            self.assertEqual(entry["cad_index"], cad_order[round(8 * cq / 100)])
        # strata membership: tercile/decile by rank
        tercile = {idx: min(2, 3 * r // 11) for r, idx in enumerate(text_order)}
        decile = {idx: min(9, 10 * r // 9) for r, idx in enumerate(cad_order)}
        for key, entries in sel["throughput_strata"].items():
            t = int(key.split("_")[1][1:])
            d = int(key.split("_")[3][1:])
            for e in entries:
                self.assertEqual(tercile[e["text_index"]], t, key)
                self.assertEqual(decile[e["cad_index"]], d, key)

    def test_shortfall_recorded_for_tiny_inputs(self):
        texts = _text_records(3)
        cads = _cad_length_records(2)
        sel = generate_selection(texts, cads, texts_sha256="a", cad_length_sha256="b",
                                 pair_manifest_sha256="c")
        self.assertTrue(sel["counts"]["strata_shortfalls"])
        # every pair still coherent
        space = PairSpace(3, 2)
        for entries in sel["throughput_strata"].values():
            for e in entries:
                self.assertEqual(
                    space.coords_of(e["pair_index"]), (e["text_index"], e["cad_index"])
                )


class SelectionVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.texts = _text_records(12)
        self.cads = _cad_length_records(6)
        (self.root / TEXTS_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.texts), encoding="utf-8"
        )
        (self.root / CAD_LENGTH_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.cads), encoding="utf-8"
        )
        (self.root / PAIR_MANIFEST_FILE).write_text("{}", encoding="utf-8")
        self.selection = generate_selection(
            self.texts, self.cads,
            texts_sha256=sha256_file(self.root / TEXTS_FILE),
            cad_length_sha256=sha256_file(self.root / CAD_LENGTH_FILE),
            pair_manifest_sha256=sha256_file(self.root / PAIR_MANIFEST_FILE),
        )
        self.space = PairSpace(12, 6)

    def test_accepts_matching_manifest(self):
        verify_selection_against_manifest(self.selection, self.root, self.space)

    def test_schema_drift_refused(self):
        bad = json.loads(json.dumps(self.selection))
        bad["schema_version"] = "v0"
        with self.assertRaises(RunnerError):
            verify_selection_against_manifest(bad, self.root, self.space)

    def test_provenance_drift_refused(self):
        bad = json.loads(json.dumps(self.selection))
        bad["provenance"]["texts_jsonl_sha256"] = "0" * 64
        with self.assertRaisesRegex(RunnerError, "provenance"):
            verify_selection_against_manifest(bad, self.root, self.space)

    def test_pair_fact_drift_refused(self):
        bad = json.loads(json.dumps(self.selection))
        bad["spot"]["n_chars"] += 1
        with self.assertRaisesRegex(RunnerError, "n_chars"):
            verify_selection_against_manifest(bad, self.root, self.space)
        bad = json.loads(json.dumps(self.selection))
        bad["ten_pair"][0]["pair_index"] += 1
        with self.assertRaisesRegex(RunnerError, "coords"):
            verify_selection_against_manifest(bad, self.root, self.space)


class CellCountsAndETATest(unittest.TestCase):
    def test_cell_counts_cover_exact_matrix(self):
        texts = _text_records(3644)
        cads = _cad_length_records(8046)
        counts = full_matrix_cell_counts(texts, cads)
        self.assertEqual(len(counts), 30)
        self.assertEqual(sum(counts.values()), FULL_MATRIX_PAIRS)
        # tercile boundaries via min(2, int(3r/3644)): tercile 0 = ranks
        # 0..1214 (1215 texts), tercile 1 = 1215..2429 (1215), tercile 2 = rest
        # (1214) -- hand-derived, independent of the implementation.
        terciles = [1215, 1215, 1214]
        for t in range(3):
            self.assertEqual(
                sum(v for k, v in counts.items() if k.startswith(f"text_t{t}_")),
                terciles[t] * 8046,
            )

    def test_eta_boundary_at_63_pairs_per_s(self):
        counts = {f"text_t{t}_cad_d{d}": 0 for t in range(3) for d in range(10)}
        per_cell = FULL_MATRIX_PAIRS // 30
        for k in counts:
            counts[k] = per_cell
        counts["text_t0_cad_d0"] += FULL_MATRIX_PAIRS - sum(counts.values())
        # the EXACT gate rate: 29,319,624 * 1.30 / 604,800 = 63.0200...
        exact_threshold = FULL_MATRIX_PAIRS * ETA_SAFETY_FACTOR / SEVEN_DAY_SECONDS
        passing = estimate_full_matrix_seconds(
            counts, {k: exact_threshold * 1.0000001 for k in counts}, overhead_seconds=0.0
        )
        self.assertTrue(passing["passes_seven_day_gate"])
        self.assertLessEqual(passing["eta_seconds"], SEVEN_DAY_SECONDS)
        self.assertAlmostEqual(exact_threshold, TWO_CARD_MIN_PAIRS_PER_S)
        # at the exact threshold FP rounding lands marginally OVER the gate
        # (conservative direction; the fail-safe side)
        borderline = estimate_full_matrix_seconds(
            counts, {k: exact_threshold for k in counts}, overhead_seconds=0.0
        )
        self.assertLess(abs(borderline["eta_seconds"] - SEVEN_DAY_SECONDS), 0.01)
        # The old rounded 63.0 is marginally below the exact threshold.
        marginal = estimate_full_matrix_seconds(
            counts, {k: 63.0 for k in counts}, overhead_seconds=0.0
        )
        self.assertFalse(marginal["passes_seven_day_gate"])
        failing = estimate_full_matrix_seconds(
            counts, {k: 62.9 for k in counts}, overhead_seconds=0.0
        )
        self.assertFalse(failing["passes_seven_day_gate"])
        # overhead pushes a borderline rate over the gate
        over = estimate_full_matrix_seconds(
            counts, {k: exact_threshold for k in counts}, overhead_seconds=1000.0
        )
        self.assertFalse(over["passes_seven_day_gate"])

    def test_eta_fail_closed_inputs(self):
        counts = {"a": FULL_MATRIX_PAIRS}
        with self.assertRaisesRegex(RunnerError, "no positive"):
            estimate_full_matrix_seconds(counts, {"a": 0.0}, overhead_seconds=0.0)
        with self.assertRaisesRegex(RunnerError, "no positive"):
            estimate_full_matrix_seconds(counts, {}, overhead_seconds=0.0)
        with self.assertRaisesRegex(RunnerError, "full matrix"):
            estimate_full_matrix_seconds({"a": 5}, {"a": 1.0}, overhead_seconds=0.0)
        with self.assertRaisesRegex(RunnerError, "safety"):
            estimate_full_matrix_seconds(counts, {"a": 1.0}, overhead_seconds=0.0,
                                         safety_factor=1.0)
        with self.assertRaisesRegex(RunnerError, "overhead"):
            estimate_full_matrix_seconds(counts, {"a": 1.0}, overhead_seconds=-1.0)
        ok = estimate_full_matrix_seconds(counts, {"a": 10_000.0}, overhead_seconds=60.0)
        self.assertAlmostEqual(
            ok["eta_seconds"], (FULL_MATRIX_PAIRS / 10_000.0 + 60.0) * ETA_SAFETY_FACTOR
        )

    def test_two_card_combination_rules(self):
        def result(start, end, rate=30.0, cells=("c0", "c1")):
            return {
                "schema_version": BENCH_RESULT_SCHEMA_VERSION,
                "measurement_window": {"start_epoch": start, "end_epoch": end},
                "cell_e2e_pairs_per_s": {c: rate for c in cells},
            }

        combined = combine_two_card_results(result(0, 100), result(20, 120))
        self.assertEqual(combined, {"c0": 60.0, "c1": 60.0})
        with self.assertRaisesRegex(RunnerError, "overlap"):
            combine_two_card_results(result(0, 100), result(200, 300))
        with self.assertRaisesRegex(RunnerError, "missing"):
            combine_two_card_results(result(0, 100), result(20, 120, cells=("c0",)))
        with self.assertRaisesRegex(RunnerError, "not a"):
            combine_two_card_results({"schema_version": "x"}, result(0, 1))

    def test_large_tiers_require_explicit_gate(self):
        self.assertEqual(parse_batch_tiers("1,2,4,8", False), [1, 2, 4, 8])
        with self.assertRaisesRegex(RunnerError, "not approved"):
            parse_batch_tiers("16,32", False)
        with self.assertRaisesRegex(RunnerError, "not approved"):
            parse_batch_tiers("64", True)
        with self.assertRaisesRegex(RunnerError, "not approved"):
            parse_batch_tiers("3", True)
        # G2 repair: --allow-large-batch only PERMITS explicitly requested
        # 16/32; it never auto-appends unrequested tiers
        self.assertEqual(parse_batch_tiers("1,2,4,8", True), [1, 2, 4, 8])
        self.assertEqual(parse_batch_tiers("16", True), [16])
        self.assertEqual(parse_batch_tiers("8,32", True), [8, 32])

    def test_eta_refuses_duplicate_card_and_mismatched_bindings(self):
        def result(device, invocation):
            return {
                "schema_version": BENCH_RESULT_SCHEMA_VERSION,
                "mode": "throughput",
                "invocation_id": invocation,
                "measurement_window": {"start_epoch": 0, "end_epoch": 100},
                "selection_sha256": "selection",
                "environment": {
                    "device": device, "git_sha": "a" * 40,
                    "pair_manifest_sha256": "manifest",
                    "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
                    "torch_dtype": "float32", "trim_mode": "none", "text_cache": False,
                    "score_name": "two_head_factorized_logprob_sum", "spec_version": "text2cad-vector-score-v1",
                },
                "authorization": {"approved_sha": "a" * 40},
                "budget": {"approved_sha": "a" * 40},
                "tiers": [{"batch_size": 8, "status": "ok"}],
                "cell_e2e_pairs_per_s_by_tier": {"cell": {"8": 32.0}},
            }
        a, b = result("npu:0", "a"), result("npu:1", "b")
        kwargs = dict(expected_manifest_sha256="manifest", expected_selection_sha256="selection", expected_cells={"cell"})
        validate_eta_card_results([a, b], ["a.json", "b.json"], 8, **kwargs)
        with self.assertRaisesRegex(RunnerError, "distinct card result"):
            validate_eta_card_results([a, b], ["a.json", "a.json"], 8, **kwargs)
        with self.assertRaisesRegex(RunnerError, "distinct throughput"):
            validate_eta_card_results([a, a], ["a.json", "b.json"], 8, **kwargs)
        with self.assertRaisesRegex(RunnerError, "each NPU"):
            validate_eta_card_results([a, result("npu:0", "b")], ["a.json", "b.json"], 8, **kwargs)
        b["selection_sha256"] = "other"
        with self.assertRaisesRegex(RunnerError, "reviewed pair manifest"):
            validate_eta_card_results([a, b], ["a.json", "b.json"], 8, **kwargs)


# ---------------------------------------------------------------------------
# Gate comparison semantics
# ---------------------------------------------------------------------------


def _fake_store_with_lengths(lengths: list[int]) -> ZipCandidateStore:
    """Store stub carrying only n_real_tokens (no tensors needed)."""
    store = ZipCandidateStore.__new__(ZipCandidateStore)
    store._candidates = [
        {"n_real_tokens": n, "cad_vec": None} for n in lengths
    ]
    store.rejections = {}
    return store


class CompareGateTest(unittest.TestCase):
    def _scores(self, base: float, n_valid: int = 5) -> dict:
        return {
            "score_x": base * 0.6, "score_y": base * 0.4, "score_total": base,
            "n_valid_x": 3, "n_valid_y": 2, "n_valid_total": n_valid,
            "valid_seq_len": 18, "n_real_tokens": 19,
        }

    def test_identical_passes(self):
        logits = [torch.randn(271, 2, 267, dtype=torch.float64) for _ in range(2)]
        scores = [self._scores(-10.0), self._scores(-20.5)]
        store = _fake_store_with_lengths([19, 19])
        report = compare_gate_results(
            (logits, scores), ([l.clone() for l in logits], [dict(s) for s in scores]),
            store, [(0, 0), (0, 1)],
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["per_pair"][0]["n_rows_compared"], 18)

    def test_padding_row_differences_are_ignored(self):
        # Non-mirror property: corruption BEYOND the valid target rows must
        # not affect the gate (those logits are never scored).
        ref = torch.randn(271, 2, 267, dtype=torch.float64)
        alt = ref.clone()
        alt[18:] = 9999.0  # rows >= n_real-1 are padding-position predictions
        scores = [self._scores(-3.0)]
        store = _fake_store_with_lengths([19])
        report = compare_gate_results(([ref], scores), ([alt], [dict(scores[0])]),
                                      store, [(0, 0)])
        self.assertTrue(report["passed"])
        # but corrupting a VALID row fails the gate
        alt2 = ref.clone()
        alt2[3, 0, 5] += 1.0
        report2 = compare_gate_results(([ref], scores), ([alt2], [dict(scores[0])]),
                                       store, [(0, 0)])
        self.assertFalse(report2["passed"])
        self.assertEqual(report2["per_pair"][0]["n_logit_violations"], 1)

    def test_score_and_count_drift_fail(self):
        logits = torch.randn(271, 2, 267, dtype=torch.float64)
        store = _fake_store_with_lengths([19])
        good = self._scores(-10.0)
        bad_score = self._scores(-10.0 + 10 * G1_SCORE_ATOL)
        report = compare_gate_results(
            ([logits], [good]), ([logits], [bad_score]), store, [(0, 0)]
        )
        self.assertFalse(report["passed"])
        bad_counts = self._scores(-10.0)
        bad_counts["n_valid_y"] = 99
        report = compare_gate_results(
            ([logits], [good]), ([logits], [bad_counts]), store, [(0, 0)]
        )
        self.assertFalse(report["passed"])

    def test_shape_mismatch_fails_closed(self):
        ref = torch.randn(271, 2, 267, dtype=torch.float64)
        alt = torch.randn(100, 2, 267, dtype=torch.float64)
        store = _fake_store_with_lengths([19])
        report = compare_gate_results(
            ([ref], [self._scores(-1.0)]), ([alt], [self._scores(-1.0)]), store, [(0, 0)]
        )
        self.assertFalse(report["passed"])


# ---------------------------------------------------------------------------
# End-to-end gates on a tiny random ported model
# ---------------------------------------------------------------------------


class _FakeBertForGates(torch.nn.Module):
    """Pad token 0 has a NONZERO embedding: cache-vs-direct equality can only
    hold if padded positions are truly masked downstream."""

    def __init__(self, hidden=32, vocab=100):
        super().__init__()
        torch.manual_seed(11)
        self.emb = torch.nn.Embedding(vocab, hidden)
        with torch.no_grad():
            self.emb.weight[0] = 5.0  # blatantly nonzero pad embedding

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return (self.emb(input_ids),)


def _tiny_model():
    torch.manual_seed(3)
    config = {
        "text_encoder": {
            "text_embedder": {"max_seq_len": 512},
            "adaptive_layer": {"in_dim": 32, "out_dim": 32, "num_heads": 4, "dropout": 0.0},
        },
        "cad_decoder": {
            "tdim": 32, "cdim": 16, "num_layers": 3, "num_heads": 2,
            "dropout": 0.0, "ca_level_start": 1, "cad_seq_len": 272,
        },
    }
    embedder = TextEmbedder(
        max_seq_len=512, device="cpu", model=_FakeBertForGates(), tokenizer=FakeTokenizer()
    )
    model = build_text2cad(embedder, device="cpu", config=config)
    model.eval()
    return model


class GateIntegrationTest(unittest.TestCase):
    """Real forward through the ported architecture (random weights, CPU)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        specs = [(f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth", n)
                 for i, n in enumerate((19, 36, 53, 19))]
        _make_zip(self.root / "cad_seq.zip", specs)
        zip_path = self.root / "cad_seq.zip"
        self.store = ZipCandidateStore(
            zip_path, [m for m, _ in specs], expected_sha256=sha256_file(zip_path)
        )
        self.store.preload()
        self.texts = [
            "short query",
            "a medium length query with several tokens inside",
            "a much longer query " + "with many descriptive words " * 4,
        ]
        self.model = _tiny_model()
        self.pairs = [(0, 0), (1, 1), (2, 2), (1, 3), (0, 2)]

    def test_batch_consistency_gate_passes(self):
        ref = score_pairs_detailed(self.model, self.store, self.texts, self.pairs,
                                   torch.device("cpu"), batch_size=1)
        alt = score_pairs_detailed(self.model, self.store, self.texts, self.pairs,
                                   torch.device("cpu"), batch_size=5)
        report = compare_gate_results(ref, alt, self.store, self.pairs)
        self.assertTrue(report["passed"], json.dumps(report["per_pair"], indent=1))

    def test_trim_gate_passes_and_preserves_official_math(self):
        ref = score_pairs_detailed(self.model, self.store, self.texts, self.pairs,
                                   torch.device("cpu"), batch_size=1, trim_mode="none")
        alt = score_pairs_detailed(self.model, self.store, self.texts, self.pairs,
                                   torch.device("cpu"), batch_size=5, trim_mode="batch_max")
        report = compare_gate_results(ref, alt, self.store, self.pairs)
        self.assertTrue(report["passed"], json.dumps(report["per_pair"], indent=1))
        # and trimming actually shrank the input (lengths 19/36/53 -> max 53)
        vec_dict, mask_dict, cad_seq_len = assemble_batch(
            self.store, [0, 2], torch.device("cpu"), trim_mode="batch_max"
        )
        self.assertEqual(vec_dict["cad_vec"].shape[1], 53)
        self.assertEqual(cad_seq_len, 52)
        self.assertEqual(tuple(mask_dict["attn_mask"].shape), (52, 52))
        # the trimmed mask is exactly the official mask's top-left submatrix
        canonical = generate_attention_mask(271)
        self.assertTrue(torch.equal(mask_dict["attn_mask"], canonical[:52, :52]))

    def test_cache_gate_passes_despite_nonzero_pad_embedding(self):
        ref = score_pairs_detailed(self.model, self.store, self.texts, self.pairs,
                                   torch.device("cpu"), batch_size=5, text_cache=False)
        alt = score_pairs_detailed(self.model, self.store, self.texts, self.pairs,
                                   torch.device("cpu"), batch_size=5, text_cache=True)
        report = compare_gate_results(ref, alt, self.store, self.pairs)
        self.assertTrue(report["passed"], json.dumps(report["per_pair"], indent=1))
        # the embedder was restored after the gated run
        self.assertIsInstance(self.model.base_text_embedder, TextEmbedder)

    def test_trimmed_full_length_candidate_is_identity(self):
        # a candidate padded to the full 272 trims to itself
        vec_dict, _mask, cad_seq_len = assemble_batch(
            self.store, [0], torch.device("cpu"), trim_mode="batch_max"
        )
        self.assertEqual(vec_dict["cad_vec"].shape[1], 19)
        self.assertEqual(cad_seq_len, 18)


# ---------------------------------------------------------------------------
# Throughput driver
# ---------------------------------------------------------------------------


class ThroughputDriverTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cells = {
            "text_t0_cad_d0": [(0, 0), (0, 1), (0, 2), (0, 3)],
            "text_t1_cad_d1": [(1, 0), (1, 1)],
        }

    def _score_fn(self, charges: list[int], fail_at_call: int | None = None):
        calls = {"n": 0}

        def fn(batch):
            calls["n"] += 1
            if fail_at_call is not None and calls["n"] == fail_at_call:
                raise RuntimeError("NPU out of memory: fake")
            scores = [
                {"score_total": -1.0, "n_valid_total": 10, "pair_index": i}
                for i, _ in enumerate(batch)
            ]
            return scores, {"preprocess_s": 0.001, "forward_score_s": 0.01}

        return fn

    def test_charges_match_plan_exactly(self):
        charges: list[int] = []
        tiers = [1, 2, 4]
        planned = plan_throughput_forwards(self.cells, tiers, warmup_passes=2, reps=3)
        result = run_throughput_tiers(
            self._score_fn(charges), self.cells,
            batch_tiers=tiers, warmup_passes=2, reps=3, device="cpu",
            charge_fn=charges.append, out_dir=self.root,
        )
        self.assertEqual(sum(charges), planned)
        # structure: tiers/cells/reps recorded; record files published
        self.assertEqual(len(result["tiers"]), 3)
        for tier in result["tiers"]:
            self.assertEqual(tier["status"], "ok")
            for cell in self.cells:
                reps = tier["cells"][cell]["reps"]
                self.assertEqual(len(reps), 3)
                for rep in reps:
                    self.assertGreater(rep["e2e_s"], 0)
                    self.assertGreater(rep["forward_score_s"], 0)
                    self.assertIn("write_s", rep)
                    self.assertIsNotNone(rep["pairs_per_s_e2e"])
        n_files = len(list(self.root.glob("bench_records_*.jsonl")))
        self.assertEqual(n_files, 3 * 3 * 2)  # tiers x reps x cells

    def test_warmup_floor_enforced(self):
        with self.assertRaisesRegex(RunnerError, "warmup"):
            run_throughput_tiers(self._score_fn([]), self.cells, batch_tiers=[1],
                                 warmup_passes=1, reps=1)
        with self.assertRaisesRegex(RunnerError, "reps"):
            run_throughput_tiers(self._score_fn([]), self.cells, batch_tiers=[1],
                                 warmup_passes=2, reps=0)

    def test_oom_stops_escalation_and_is_recorded(self):
        # calls per tier: warmup 2 passes x 2 batches = 4 calls at tier 1;
        # make tier 2's first rep batch OOM
        charges: list[int] = []
        # tier1: warmup 2*2 + reps 2*3 batches(cell0 4/1=4, cell1 2) per rep...
        # simply fail on the first call whose batch size is 2
        def fn(batch):
            if len(batch) == 2:
                raise RuntimeError("CUDA out of memory: fake")
            scores = [{"score_total": -1.0, "n_valid_total": 5} for _ in batch]
            return scores, {"preprocess_s": 0.0, "forward_score_s": 0.01}

        result = run_throughput_tiers(
            fn, self.cells, batch_tiers=[1, 2, 4], warmup_passes=2, reps=2, device="cpu",
            charge_fn=charges.append,
        )
        statuses = [t["status"] for t in result["tiers"]]
        self.assertEqual(statuses[0], "ok")
        self.assertEqual(statuses[1], "oom")
        self.assertEqual(statuses[2], "skipped_after_oom")
        self.assertEqual(result["escalation_stop"], "oom")
        self.assertIn("memory", result["tiers"][1]["error"])

    def test_non_oom_runtime_error_aborts(self):
        def fn(batch):
            raise RuntimeError("some other failure")

        with self.assertRaisesRegex(RuntimeError, "some other failure"):
            run_throughput_tiers(fn, self.cells, batch_tiers=[1], warmup_passes=2, reps=1)


class LogitsStatsTest(unittest.TestCase):
    def test_stats_and_nonfinite(self):
        logits = torch.zeros(4, 2, 267)
        stats = logits_stats(logits)
        self.assertTrue(stats["all_finite"])
        self.assertEqual(stats["abs_max"], 0.0)
        logits[0, 0, 0] = float("nan")
        self.assertFalse(logits_stats(logits)["all_finite"])


class LogitsStatsTest(unittest.TestCase):
    def test_stats_and_nonfinite(self):
        logits = torch.zeros(4, 2, 267)
        stats = logits_stats(logits)
        self.assertTrue(stats["all_finite"])
        self.assertEqual(stats["abs_max"], 0.0)
        logits[0, 0, 0] = float("nan")
        self.assertFalse(logits_stats(logits)["all_finite"])


class EtaG2CliTest(unittest.TestCase):
    """eta-g2 through the real CLI over a production-shape synthetic repo."""

    @classmethod
    def setUpClass(cls) -> None:
        from idea01.text2cad_bench_protocol import (
            build_g2_selection, estimate_sharded_eta,
        )
        from idea01.text2cad_speed_bench import (
            EXPECTED_PAIR_HASHES_REL, G2_SELECTION_CONFIG_REL,
        )

        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        root = Path(cls._tmp.name)
        cls.repo = root / "repo"
        configs = cls.repo / "experiments" / "configs"
        configs.mkdir(parents=True)
        cls.manifest_dir = root / "pair_manifest"
        cls.manifest_dir.mkdir()
        cls.out_dir = root / "out"
        cls.out_dir.mkdir()

        # production-shape synthetic records (no real assets involved)
        texts = []
        for i in range(3644):
            n_chars = (i * 37) % 180 + 10
            text = f"t{i} " + "x" * (n_chars - len(f"t{i} "))
            texts.append({"text_index": i, "text": text,
                          "text_sha256": sha256_text(text), "n_chars": len(text)})
        cads = [{"cad_index": i, "n_real_tokens": (i * 13) % 190 + 19,
                 "n_valid_x": (i * 13) % 190 + 18, "n_valid_y": 2 + (i % 5)}
                for i in range(8046)]
        (cls.manifest_dir / TEXTS_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in texts), encoding="utf-8")
        (cls.manifest_dir / CAD_LENGTH_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in cads), encoding="utf-8")
        texts_sha = sha256_file(cls.manifest_dir / TEXTS_FILE)
        cad_sha = sha256_file(cls.manifest_dir / CAD_LENGTH_FILE)
        manifest = {
            "schema_version": "text2cad-pair-manifest-v1",
            "counts": {"n_unique_text": 3644, "n_gallery": 8046, "n_pairs": 29319624},
            "artifacts": {
                TEXTS_FILE: {"file": TEXTS_FILE, "sha256": texts_sha},
                CAD_LENGTH_FILE: {"file": CAD_LENGTH_FILE, "sha256": cad_sha},
            },
        }
        (cls.manifest_dir / PAIR_MANIFEST_FILE).write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8")
        manifest_sha = sha256_file(cls.manifest_dir / PAIR_MANIFEST_FILE)
        (configs / Path(EXPECTED_PAIR_HASHES_REL).name).write_text(json.dumps({
            "artifact_sha256": {TEXTS_FILE: texts_sha, CAD_LENGTH_FILE: cad_sha},
            "pair_manifest_sha256": manifest_sha,
        }), encoding="utf-8")

        tokens = {i: (i * 11) % 90 + 5 for i in range(3644)}
        selection = build_g2_selection(texts, cads, tokens)
        selection["provenance"] = {
            "texts_jsonl_sha256": texts_sha,
            "cad_length_index_sha256": cad_sha,
            "pair_manifest_sha256": manifest_sha,
            "token_audit_sha256": "f" * 64,
            "bert_file_sha256": dict(sorted(PINNED_BERT_FILE_SHA256.items())),
            "generated_by": "test fixture",
        }
        cls.selection_path = configs / Path(G2_SELECTION_CONFIG_REL).name
        cls.selection_path.write_text(
            json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        cls.selection = selection
        cls.git_sha = "d" * 40

        for device, per_batch in (("npu:0", 0.8), ("npu:1", 1.6)):
            result = cls._fabricate_result(cls, selection, device, per_batch)
            (cls.out_dir / f"card_{device.replace(':', '')}.json").write_text(
                json.dumps(result, sort_keys=True), encoding="utf-8")

    def _fabricate_result(cls, selection, device, per_batch_s):
        shard = {"npu:0": 0, "npu:1": 1}[device]
        cells = {}
        for cell_key, cell in selection["tiers"]["32"]["cells"].items():
            if cell["shard_id"] != shard:
                continue
            n_sample = len(cell["samples"]["texts"]) * len(cell["samples"]["windows"])
            reps = []
            for r in range(3):
                start = 1000.0 + r * 100.0
                reps.append({
                    "rep": r,
                    "point": f"tt{cell['text_tier']}_{cell['layer']}",
                    "window": {"start_epoch": start, "end_epoch": start + per_batch_s * n_sample},
                    "n_batches": n_sample,
                    "n_pairs": n_sample * cell["batch_size"],
                    "e2e_s": per_batch_s * n_sample,
                })
            cells[cell_key] = {
                "point": f"tt{cell['text_tier']}_{cell['layer']}",
                "batch_size": cell["batch_size"],
                "reps": reps,
                "per_batch_e2e_s_max": per_batch_s,
            }
        return {
            "schema_version": "text2cad-bench-result-v2",
            "mode": "throughput",
            "invocation_id": f"inv-{device}",
            "cold_start_s": 10.0,
            "selection_sha256": sha256_file(cls.selection_path),
            "environment": {
                "device": device,
                "git_sha": cls.git_sha,
                "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
                "pair_manifest_sha256": sha256_file(cls.manifest_dir / PAIR_MANIFEST_FILE),
                "torch_dtype": "float32",
                "trim_mode": "none",
                "text_cache": False,
                "score_name": "two_head_factorized_logprob_sum",
                "spec_version": "text2cad-vector-score-v1",
            },
            "authorization": {"approved_sha": cls.git_sha},
            "budget": {"approved_sha": cls.git_sha},
            "tiers": [{"batch_size": 32, "status": "ok", "cells": cells}],
        }

    def _argv(self, *extra):
        return [
            "--mode", "eta-g2",
            "--pair-manifest-dir", str(self.manifest_dir),
            "--production-batch-size", "32",
            "--card-result", str(self.out_dir / "card_npu0.json"),
            "--card-result", str(self.out_dir / "card_npu1.json"),
            "--repo-root", str(self.repo),
            *extra,
        ]

    def test_happy_path_writes_estimate_and_fails_gate_honestly(self):
        from idea01.text2cad_speed_bench import main

        out = self.out_dir / "eta_g2.json"
        rc = main(self._argv("--out", str(out)))
        self.assertEqual(rc, 3)  # 20 pairs/s slow card honestly fails the gate
        estimate = json.loads(out.read_text(encoding="utf-8"))
        self.assertFalse(estimate["passes_seven_day_gate"])
        self.assertEqual(estimate["slower_card"], "npu:1")
        self.assertEqual(estimate["full_matrix_pairs"], 29319624)
        n_batches = 1822 * 252
        self.assertAlmostEqual(
            estimate["per_card"]["npu:1"]["compute_seconds"], n_batches * 1.6
        )
        # a second invocation must not overwrite the first estimate
        rc2 = main(self._argv("--out", str(out)))
        self.assertEqual(rc2, 2)

    def test_refusals(self):
        from idea01.text2cad_speed_bench import main

        # foreign manifest (provenance mismatch)
        foreign = self.out_dir / "foreign_manifest"
        foreign.mkdir()
        (foreign / PAIR_MANIFEST_FILE).write_text("{}", encoding="utf-8")
        argv = self._argv()
        argv[argv.index("--pair-manifest-dir") + 1] = str(foreign)
        self.assertEqual(main(argv), 2)
        # one card only
        self.assertEqual(main(self._argv()[:6]), 2)
        # v1-schema card result refused
        bad = json.loads((self.out_dir / "card_npu1.json").read_text(encoding="utf-8"))
        bad["schema_version"] = "text2cad-bench-result-v1"
        (self.out_dir / "card_npu1_v1.json").write_text(json.dumps(bad), encoding="utf-8")
        argv = self._argv()
        argv[argv.index(str(self.out_dir / "card_npu1.json"))] = str(self.out_dir / "card_npu1_v1.json")
        self.assertEqual(main(argv), 2)


# ---------------------------------------------------------------------------
# G2 repair: real production-batch throughput driver
# ---------------------------------------------------------------------------


def _g2_fixture():
    """5 texts x 37 CADs; token lengths crafted to differ from char lengths."""
    from idea01.text2cad_bench_protocol import build_g2_selection

    char_lens = [10, 50, 20, 90, 30]
    token_lens = [5, 12, 8, 40, 20]
    cad_reals = [(i * 13) % 190 + 19 for i in range(37)]
    texts = []
    for i, n_chars in enumerate(char_lens):
        text = f"t{i} " + "x" * (n_chars - len(f"t{i} "))
        texts.append({"text_index": i, "text": text, "text_sha256": sha256_text(text),
                      "n_chars": len(text)})
    cads = [{"cad_index": i, "n_real_tokens": n, "n_valid_x": n - 1, "n_valid_y": 2}
            for i, n in enumerate(cad_reals)]
    tokens = {i: token_lens[i] for i in range(5)}
    selection = build_g2_selection(texts, cads, tokens)
    return selection, token_lens, cad_reals


class ProductionBatchDriverTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01.text2cad_bench_protocol import build_tier_plans
        from idea01.text2cad_speed_bench import run_production_batch_tiers

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.selection, self.token_lens, self.cad_reals = _g2_fixture()
        self.plans = build_tier_plans(self.selection, [1, 32], shard_id=0)
        self.run_driver = run_production_batch_tiers

    def _facts(self, batch):
        return {
            "max_cad_len": max(self.cad_reals[c] for _, c in batch),
            "text_tokens": [self.token_lens[t] for t, _ in batch],
            "text_pad_width": max(self.token_lens[t] for t, _ in batch),
        }

    def _score_fn(self, seen_batches):
        def fn(batch):
            seen_batches.append(len(batch))
            scores = [{"score_total": -1.0, "n_valid_total": 10} for _ in batch]
            return scores, {"preprocess_s": 0.001, "forward_score_s": 0.01}

        return fn

    def test_legacy_symptom_fake32_then_new_driver_refuses_it(self):
        """Symptom (Codex review): the legacy driver ran a 20-pair cell at
        nominal tier 32. The legacy function is kept only to document the
        symptom; the new driver REFUSES undersized full-layer batches."""
        cells = {"cell": [(0, i) for i in range(20)]}
        seen: list[int] = []

        def fn(batch):
            seen.append(len(batch))
            return ([{"score_total": -1.0, "n_valid_total": 5} for _ in batch],
                    {"preprocess_s": 0.0, "forward_score_s": 0.01})

        run_throughput_tiers(fn, cells, batch_tiers=[32], warmup_passes=2, reps=1)
        self.assertTrue(seen)
        self.assertEqual(max(seen), 20)  # the old symptom: nominal 32, real 20

        from idea01.text2cad_bench_protocol import build_tier_plans

        plans = build_tier_plans(self.selection, [32], shard_id=0)
        plans[0]["cells"][0]["batches"][0] = plans[0]["cells"][0]["batches"][0][:20]
        with self.assertRaisesRegex(RunnerError, "batch size"):
            self.run_driver(
                self._score_fn([]), plans, warmup_passes=2, reps=3,
                batch_facts_fn=self._facts,
            )

    def test_real_batches_records_and_budget_same_source(self):
        from idea01.text2cad_bench_protocol import plan_production_forwards

        charges: list[int] = []
        seen: list[int] = []
        planned = plan_production_forwards(self.plans, warmup_passes=2, reps=3)
        result = self.run_driver(
            self._score_fn(seen), self.plans, warmup_passes=2, reps=3,
            charge_fn=charges.append, out_dir=self.root, batch_facts_fn=self._facts,
        )
        self.assertEqual(sum(charges), planned)
        self.assertEqual(planned, 410)  # hand-computed from the 5x37 fixture
        # every charged batch was a REAL batch of its tier (or the honest tail):
        # tier 1 -> 1, tier 32 -> 32 full + 5 tail, nothing else
        self.assertEqual(sorted(set(charges)), [1, 5, 32])
        # per-rep records: real sizes, windows, batch facts
        tier32 = next(t for t in result["tiers"] if t["batch_size"] == 32)
        d0 = tier32["cells"]["shard0_tt0_d0"]
        self.assertEqual(len(d0["reps"]), 3)
        for rep in d0["reps"]:
            self.assertEqual(rep["n_batches"], 1)
            self.assertEqual(rep["n_pairs"], 32)
            self.assertGreaterEqual(rep["window"]["end_epoch"], rep["window"]["start_epoch"])
            self.assertEqual(rep["sync_wait_s"], 0.0)
            batch = rep["batches"][0]
            self.assertEqual(batch["size"], 32)
            self.assertEqual(batch["max_cad_len"], max(self.cad_reals[:32]))
            self.assertEqual(batch["text_pad_width"], batch["text_tokens"][0])
            self.assertEqual(len(batch["pairs"]), 32)
        self.assertEqual(d0["per_batch_e2e_s_max"], max(r["e2e_s"] for r in d0["reps"]))
        # record files published per tier/cell/rep
        n_files = len(list(self.root.glob("bench_records_*.jsonl")))
        n_cells = sum(len(p["cells"]) for p in self.plans)
        self.assertEqual(n_files, 3 * n_cells)  # reps(3) x cells, both tiers
        self.assertIn("plan_sha256", result)

    def test_reps_and_warmup_floors(self):
        with self.assertRaisesRegex(RunnerError, "reps"):
            self.run_driver(self._score_fn([]), self.plans, warmup_passes=2, reps=2)
        with self.assertRaisesRegex(RunnerError, "warmup"):
            self.run_driver(self._score_fn([]), self.plans, warmup_passes=1, reps=3)

    def test_sync_wait_separated_from_measured_window_and_failure_signalled(self):
        from idea01.text2cad_bench_protocol import SyncBarrier, build_tier_plans

        plans = build_tier_plans(self.selection, [32], shard_id=0)

        class FakeSync:
            def __init__(self):
                self.points: list[str] = []
                self.failures: list[tuple] = []

            def arrive(self, point, payload):
                self.points.append(point)
                time.sleep(0.01)
                return {"peer": True}

            def signal_failure(self, point, kind, detail):
                self.failures.append((point, kind, detail))

        sync = FakeSync()
        result = self.run_driver(
            self._score_fn([]), plans, warmup_passes=2, reps=3,
            out_dir=self.root, batch_facts_fn=self._facts, sync=sync,
        )
        tier = result["tiers"][0]
        n_cells = len(plans[0]["cells"])
        self.assertEqual(len(sync.points), n_cells * 3)
        for cell in tier["cells"].values():
            for rep in cell["reps"]:
                self.assertGreater(rep["sync_wait_s"], 0.0)
                # measured window excludes the barrier wait by construction
                self.assertGreater(rep["e2e_s"], 0.0)

        def oom_fn(batch):
            raise RuntimeError("NPU out of memory: fake")

        sync2 = FakeSync()
        result2 = self.run_driver(
            oom_fn, plans, warmup_passes=2, reps=3,
            batch_facts_fn=self._facts, sync=sync2,
        )
        self.assertEqual(result2["tiers"][0]["status"], "oom")
        self.assertTrue(sync2.failures)
        self.assertEqual(sync2.failures[0][1], "oom")


# ---------------------------------------------------------------------------
# G2 repair: profile-consistency (704-pair extended G1 equivalence entry)
# ---------------------------------------------------------------------------


class ProfileConsistencyTest(unittest.TestCase):
    """The 64-pair reference vs 10 alternate variants on a tiny ported model."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.selection, _tokens, _reals = _g2_fixture()
        self.pairs = [(e["text_index"], e["cad_index"]) for e in self.selection["profile64"]]
        self.assertEqual(len(self.pairs), 64)
        # store with 37 short candidates; 5 texts matching the fixture indices
        specs = [(f"cad_seq/0000/{i:08d}/seq/{i:08d}.pth", 19 + (i % 3) * 17)
                 for i in range(37)]
        _make_zip(self.root / "cad_seq.zip", specs)
        self.store = ZipCandidateStore(
            self.root / "cad_seq.zip", [m for m, _ in specs],
            expected_sha256=sha256_file(self.root / "cad_seq.zip"),
        )
        self.store.preload()
        self.texts = [
            "alpha beta gamma",
            "a longer descriptive query about the mechanical part " + "with more words " * 3,
            "short",
            "medium length query text here",
            "fallback query text",
        ]
        self.model = _tiny_model()

    def _run(self, charges):
        from idea01.text2cad_speed_bench import run_profile_consistency

        return run_profile_consistency(
            self.model, self.store, self.texts, self.pairs, torch.device("cpu"),
            charge_fn=charges.append, out_dir=self.root,
        )

    def test_704_scored_pairs_across_reference_and_ten_variants(self):
        charges: list[int] = []
        result = self._run(charges)
        self.assertTrue(result["passed"], json.dumps(result.get("failure"), indent=1)[:2000])
        # 64 pairs x (1 reference + 10 variants) = 704, pre-charged per variant
        self.assertEqual(charges, [64] * 11)
        self.assertEqual(result["total_scored_pairs"], 704)
        self.assertEqual(result["n_pairs"], 64)
        expected_variants = [
            "batch16", "batch32", "trim32", "cache32",
            "trim_cache_b1", "trim_cache_b2", "trim_cache_b4",
            "trim_cache_b8", "trim_cache_b16", "trim_cache_b32",
        ]
        self.assertEqual([v["variant"] for v in result["variants"]], expected_variants)
        by_name = {v["variant"]: v for v in result["variants"]}
        self.assertEqual(by_name["batch16"]["real_batch_sizes"], [16, 16, 16, 16])
        self.assertEqual(by_name["batch32"]["real_batch_sizes"], [32, 32])
        self.assertEqual(by_name["trim_cache_b1"]["real_batch_sizes"], [1] * 64)
        self.assertEqual(by_name["trim32"]["trim_mode"], "batch_max")
        self.assertFalse(by_name["trim32"]["text_cache"])
        self.assertTrue(by_name["cache32"]["text_cache"])
        self.assertEqual(by_name["cache32"]["cache_stats"]["entries"], 5)
        self.assertEqual(
            by_name["cache32"]["cache_stats"]["hits"]
            + by_name["cache32"]["cache_stats"]["misses"],
            64,
        )
        for variant in result["variants"]:
            self.assertEqual(len(variant["per_pair"]), 64)
            self.assertTrue(variant["all_finite"])
            self.assertTrue(math.isfinite(variant["max_abs_score_diff"]))
            self.assertTrue(math.isfinite(variant["max_normalized_score_diff"]))
            self.assertTrue(math.isfinite(variant["max_normalized_logit_diff"]))
            entry = variant["per_pair"][0]
            for key in ("score_x_ref", "score_y_ref", "score_total_ref",
                        "score_total_alt", "n_valid_total_ref"):
                self.assertIn(key, entry)
        # valid-row logits saved for EVERY variant + reference, weights_only-loadable
        for name in ["reference_b1", *expected_variants]:
            path = self.root / f"profile_logits_{name}.pt"
            self.assertTrue(path.is_file(), name)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["variant"], name)
            self.assertEqual(len(payload["pairs"]), 64)
            self.assertEqual(len(payload["logits_valid_rows"]), 64)
            for tensor, (t, c) in zip(payload["logits_valid_rows"], payload["pairs"]):
                n_real = int(self.store.get(c)["n_real_tokens"])
                self.assertEqual(tuple(tensor.shape), (n_real - 1, 2, 267))

    def test_gate_failure_stops_at_first_variant_with_evidence(self):
        from unittest import mock

        import idea01.text2cad_speed_bench as bench

        def failed_compare(*args, **kwargs):
            return {"passed": False, "tolerances": {}, "per_pair": []}

        charges: list[int] = []
        with mock.patch.object(bench, "compare_gate_results", failed_compare):
            result = self._run(charges)
        self.assertFalse(result["passed"])
        # reference + exactly one variant were scored before the stop
        self.assertEqual(charges, [64, 64])
        self.assertEqual(len(result["variants"]), 1)
        self.assertTrue((self.root / "profile_logits_reference_b1.pt").is_file())
        self.assertTrue((self.root / "profile_logits_batch16.pt").is_file())
        self.assertFalse((self.root / "profile_logits_batch32.pt").exists())

    def test_planned_forwards(self):
        from idea01.text2cad_speed_bench import plan_profile_consistency_forwards

        self.assertEqual(plan_profile_consistency_forwards(64), 704)


if __name__ == "__main__":
    unittest.main()
