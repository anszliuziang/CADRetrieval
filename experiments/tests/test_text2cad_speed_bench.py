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
from idea01.text2cad_port.loader import PINNED_CHECKPOINT_SHA256, build_text2cad  # noqa: E402
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
        self.assertEqual(parse_batch_tiers("1,2,4,8", True), [1, 2, 4, 8, 16, 32])

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


if __name__ == "__main__":
    unittest.main()
