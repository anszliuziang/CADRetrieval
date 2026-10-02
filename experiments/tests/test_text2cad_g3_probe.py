"""CPU tests for the Text2CAD G3 full-gallery probe (2026-10-02).

Contract: experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md.
Covers, with small fake data / fake models and NO model or BERT forward:

  * pick/layer math: the pre-registered evenly spaced rank picks (endpoints,
    exact count, dedup-shrink refusal) and the global token-length layers
    (identical to the G2 terciles);
  * the blind selector against the REAL frozen worker artifacts (skipUnless
    guarded): 50+50 probes including each shard's shortest/longest text,
    per-shard-per-layer coverage, deterministic regeneration byte-identity,
    provenance/asset/dimension tamper refusals, LF output, no overwrite;
  * runner probe scope on a mini manifest: probe filtering keeps the global
    indices and the full gallery, full-matrix chunks refuse under the probe
    scope and vice versa, out-of-plan chunks refuse, resume requires the
    sidecar timing line of every completed chunk;
  * scorer header: the probe scope stamp appears only in scoped runs;
  * timing sidecar: append/read roundtrip, torn-line refusal, minimal
    resume-side scope/shard/duplicate checks;
  * the two-card ETA on a synthetic OFFICIAL-SIZE selection: hand-computed
    per-layer slowest-query pricing, slower-card wall clock, cold starts,
    overhead rules, 80% window overlap, the 604,800 s gate both sides, and
    the fail-closed refusals (device/shard/host/ledger/scope/count/window);
  * runner main() probe gates exit 2 before authorization;
  * eta-g3 CLI end-to-end against the real committed selection.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: F401  (parity with the runner test module)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.cadmium_runner import RunnerError  # noqa: E402
from idea01.text2cad_g3_probe import (  # noqa: E402
    G3_BATCH_SIZE,
    G3_PROBE_SELECTION_REL,
    G3_PROVENANCE_KEYS,
    G3_SELECTION_SCHEMA_VERSION,
    G3_TIMING_SCHEMA_VERSION,
    G2_SELECTION_CONFIG_REL,
    EXPECTED_PAIR_HASHES_REL,
    PINNED_BERT_FILE_SHA256,
    PINNED_CHECKPOINT_SHA256,
    PROBE_SCOPE_FIELD,
    RUN_SUMMARY_SCHEMA_VERSION,
    SCORE_NAME,
    SCORE_SPEC_VERSION,
    append_timing_record,
    build_g3_probe_selection,
    build_query_timing_record,
    estimate_g3_probe_eta,
    global_text_layer_map,
    main as g3_main,
    probe_facts_for_shard,
    probe_texts_for_shard,
    read_timing_records,
    run_select_g3,
    shard_pick_positions,
    text_layer_of,
    timing_sidecar_name,
    verified_sidecar_text_set,
    verify_g3_g2_config_binding,
    verify_g3_probe_selection,
    verify_g3_probe_selection_against_manifest,
    verify_g3_token_audit,
)
from idea01.text2cad_bench_protocol import SHARD_BY_DEVICE  # noqa: E402
from idea01.text2cad_frozen_runner import (  # noqa: E402
    CHUNK_SCHEMA_VERSION,
    FrozenMatrixScorer,
    RunnerConfig,
    SharedBudgetLedger,
    ZipCandidateStore,
    atomic_publish,
    main as runner_main,
    plan_chunks,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    EXPECTED_N_GALLERY,
    EXPECTED_N_UNIQUE_TEXT,
    PairSpace,
    chunk_file_name,
    sha256_file,
)

from test_text2cad_frozen_runner import (  # noqa: E402
    FULL_SHA,
    TASK_ID,
    _make_zip,
    _mini_manifest_dir,
    build_tiny_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker")
TOKEN_AUDIT = Path(
    "D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/derived/text2cad_text_token_lengths_20261001.json"
)
G2_CONFIG = REPO_ROOT / G2_SELECTION_CONFIG_REL
G3_CONFIG = REPO_ROOT / G3_PROBE_SELECTION_REL
REAL_ASSETS_PRESENT = (
    (WORKER_DIR / "texts.jsonl").is_file()
    and (WORKER_DIR / "cad_store.jsonl").is_file()
    and (WORKER_DIR / "cad_length_index.jsonl").is_file()
    and (WORKER_DIR / "pair_manifest.json").is_file()
    and TOKEN_AUDIT.is_file()
    and G2_CONFIG.is_file()
    and G3_CONFIG.is_file()
)

PROBE_SHA = "b" * 64
MANIFEST_SHA = "c" * 64
G3_N_PAIRS_PER_QUERY = EXPECTED_N_GALLERY
G3_N_BATCHES_PER_QUERY = (EXPECTED_N_GALLERY + G3_BATCH_SIZE - 1) // G3_BATCH_SIZE  # 503


# ---------------------------------------------------------------------------
# Duplicated-constant cross-checks
# ---------------------------------------------------------------------------


class G3ConstantsCrossCheckTest(unittest.TestCase):
    def test_pinned_constants_match_their_audited_sources(self):
        from idea01.text2cad_port import loader
        from idea01 import text2cad_speed_bench, text2cad_vector_scorer
        from idea01 import text2cad_frozen_runner

        self.assertEqual(PINNED_CHECKPOINT_SHA256, loader.PINNED_CHECKPOINT_SHA256)
        self.assertEqual(PINNED_BERT_FILE_SHA256, loader.PINNED_BERT_FILE_SHA256)
        self.assertEqual(SCORE_NAME, text2cad_vector_scorer.SCORE_NAME)
        self.assertEqual(SCORE_SPEC_VERSION, text2cad_vector_scorer.SCORE_SPEC_VERSION)
        self.assertEqual(G2_SELECTION_CONFIG_REL, text2cad_speed_bench.G2_SELECTION_CONFIG_REL)
        self.assertEqual(EXPECTED_PAIR_HASHES_REL, text2cad_speed_bench.EXPECTED_PAIR_HASHES_REL)
        self.assertEqual(RUN_SUMMARY_SCHEMA_VERSION, text2cad_frozen_runner.SUMMARY_SCHEMA_VERSION)

    def test_g3_module_imports_without_torch(self):
        code = (
            "import sys; import idea01.text2cad_g3_probe; "
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT / "experiments" / "src"),
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode()[:200])

    def test_expected_hashes_loader_matches_runner_loader(self):
        from idea01.text2cad_frozen_runner import _load_expected_pair_hashes as runner_loader
        from idea01.text2cad_g3_probe import _load_expected_pair_hashes as g3_loader

        path = REPO_ROOT / EXPECTED_PAIR_HASHES_REL
        if not path.is_file():
            self.skipTest("reviewed fingerprint config not present")
        self.assertEqual(runner_loader(path), g3_loader(path))


# ---------------------------------------------------------------------------
# Pick / layer math (small synthetic inputs)
# ---------------------------------------------------------------------------


class G3PickMathTest(unittest.TestCase):
    def test_positions_endpoints_exact_count(self):
        positions = shard_pick_positions(1822, 50)
        self.assertEqual(len(positions), 50)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 1821)
        self.assertEqual(positions, sorted(set(positions)))
        for j in (1, 7, 25, 48):
            self.assertEqual(positions[j], round(1821 * j / 49))

    def test_positions_refuse_impossible_requests(self):
        with self.assertRaises(RunnerError):
            shard_pick_positions(10, 50)  # more picks than texts
        with self.assertRaises(RunnerError):
            shard_pick_positions(1822, 1)  # need at least two endpoints
        with self.assertRaises(RunnerError):
            shard_pick_positions(0, 50)

    def test_layer_boundaries_match_g2_terciles(self):
        n = EXPECTED_N_UNIQUE_TEXT  # 3644
        self.assertEqual(text_layer_of(0, n), 0)
        self.assertEqual(text_layer_of(1214, n), 0)
        self.assertEqual(text_layer_of(1215, n), 1)
        self.assertEqual(text_layer_of(2429, n), 1)
        self.assertEqual(text_layer_of(2430, n), 2)
        self.assertEqual(text_layer_of(3643, n), 2)
        with self.assertRaises(RunnerError):
            text_layer_of(3644, n)

    def test_global_layer_map_tiebreaks_by_index(self):
        layers = global_text_layer_map({0: 5, 1: 1, 2: 9, 3: 1})
        # order by (length, index): [1, 3, 0, 2]; n=4 -> layers 0,0,1,2
        self.assertEqual(layers, {1: 0, 3: 0, 0: 1, 2: 2})


# ---------------------------------------------------------------------------
# Selector against the REAL frozen artifacts (skipUnless)
# ---------------------------------------------------------------------------


def _real_inputs():
    from idea01.text2cad_g3_probe import load_g3_manifest_records
    from idea01.text2cad_bench_protocol import token_lengths_from_audit

    text_records, n_cads = load_g3_manifest_records(WORKER_DIR)
    audit = json.loads(TOKEN_AUDIT.read_text(encoding="utf-8"))
    lengths = token_lengths_from_audit(
        audit,
        expected_texts_sha256=sha256_file(WORKER_DIR / "texts.jsonl"),
        expected_bert_file_sha256=dict(PINNED_BERT_FILE_SHA256),
        n_texts=len(text_records),
    )
    return text_records, n_cads, lengths


@unittest.skipUnless(REAL_ASSETS_PRESENT, "real frozen worker artifacts required")
class G3SelectionRealDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text_records, self.n_cads, self.lengths = _real_inputs()
        self.selection = build_g3_probe_selection(self.text_records, self.n_cads, self.lengths)

    def test_contract_shape_50_50_endpoints_layers(self):
        sel = self.selection
        self.assertEqual(sel["counts"]["n_texts"], EXPECTED_N_UNIQUE_TEXT)
        self.assertEqual(sel["counts"]["n_cads"], EXPECTED_N_GALLERY)
        self.assertEqual(sel["counts"]["probe_pairs_total"], 804600)
        self.assertEqual(sel["counts"]["probe_pairs_per_shard"], 402300)
        probes = sel["probes"]
        self.assertEqual(len(probes), 100)
        self.assertEqual(len({e["text_index"] for e in probes}), 100)
        layer_map = global_text_layer_map(self.lengths)
        for shard_id, (t0, t1) in enumerate([(0, 1822), (1822, 3644)]):
            entries = [e for e in probes if e["shard_id"] == shard_id]
            self.assertEqual(len(entries), 50)
            self.assertEqual([e["shard_pick"] for e in entries], list(range(50)))
            indices = {e["text_index"] for e in entries}
            self.assertTrue(all(t0 <= i < t1 for i in indices))
            # endpoints of the shard's (length, index) order are always probed
            ordered = sorted(range(t0, t1), key=lambda i: (self.lengths[i], i))
            self.assertEqual(entries[0]["text_index"], ordered[0])
            self.assertEqual(entries[-1]["text_index"], ordered[-1])
            self.assertEqual(
                {e["text_index"] for e in entries},
                {ordered[round(1821 * j / 49)] for j in range(50)},
            )
            per_layer = {0: 0, 1: 0, 2: 0}
            for e in entries:
                self.assertEqual(e["text_layer"], layer_map[e["text_index"]])
                self.assertEqual(e["n_bert_tokens"], self.lengths[e["text_index"]])
                per_layer[e["text_layer"]] += 1
            self.assertTrue(all(v >= 1 for v in per_layer.values()))
            shard_entry = next(s for s in sel["shards"] if s["shard_id"] == shard_id)
            self.assertEqual({int(k): v for k, v in shard_entry["layer_probe_counts"].items()}, per_layer)
            full = {int(k): v for k, v in shard_entry["layer_text_counts"].items()}
            self.assertEqual(sum(full.values()), 1822)
            expected_full = {0: 0, 1: 0, 2: 0}
            for t in range(t0, t1):
                expected_full[layer_map[t]] += 1
            self.assertEqual(full, expected_full)

    def test_verify_roundtrip_and_tamper_refusals(self):
        sel = json.loads(G3_CONFIG.read_text(encoding="utf-8"))
        verify_g3_probe_selection_against_manifest(sel, WORKER_DIR)
        verify_g3_token_audit(sel, TOKEN_AUDIT, WORKER_DIR)
        verify_g3_g2_config_binding(sel, REPO_ROOT)
        # re-derived content equals the committed file's derived sections
        for key in ("rules", "counts", "text_token_lengths", "text_shards", "shards", "probes"):
            self.assertEqual(sel[key], self.selection[key])
        self.assertEqual(set(sel["provenance"]), set(G3_PROVENANCE_KEYS))

        import copy

        def refused(mutate, pattern):
            bad = copy.deepcopy(sel)
            mutate(bad)
            with self.assertRaisesRegex(RunnerError, pattern):
                verify_g3_probe_selection_against_manifest(bad, WORKER_DIR)

        refused(lambda b: b["probes"][0].update(text_index=9999), "drift")
        refused(lambda b: b["probes"][1].update(text_index=b["probes"][0]["text_index"]), "drift")
        refused(lambda b: b["probes"][0].update(shard_id=1), "drift")
        refused(lambda b: b["probes"][0].update(text_layer=(b["probes"][0]["text_layer"] + 1) % 3), "drift")
        refused(lambda b: b["probes"][0].update(n_bert_tokens=b["probes"][0]["n_bert_tokens"] + 1), "drift")
        refused(lambda b: b.update(schema_version="text2cad-g2-selection-v1"), "schema")
        refused(lambda b: b["provenance"].update(texts_jsonl_sha256="0" * 64), "provenance")
        refused(lambda b: b["provenance"].update(evil="x"), "provenance keys")
        refused(lambda b: b.pop("probes"), "missing section")
        refused(lambda b: b.update(injected=[]), "non-allowlisted")
        refused(lambda b: b["text_token_lengths"].__setitem__(0, b["text_token_lengths"][0] + 1), "drift")

    def test_committed_config_is_lf_and_regeneration_is_byte_identical(self):
        raw = (REPO_ROOT / G3_PROBE_SELECTION_REL).read_bytes()
        self.assertNotIn(b"\r", raw)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        out = Path(self._tmp.name) / "regen.json"
        import types

        args = types.SimpleNamespace(
            token_audit=str(TOKEN_AUDIT), out=str(out),
            repo_root=str(REPO_ROOT), pair_manifest_dir=str(WORKER_DIR),
        )
        run_select_g3(args)
        self.assertEqual(out.read_bytes(), raw)
        with self.assertRaisesRegex(RunnerError, "overwrite"):
            run_select_g3(args)

    def test_select_g3_refuses_nonofficial_dimensions(self):
        # a tiny pair manifest is not the official pair space
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        manifest_dir = _mini_manifest_dir(root)
        audit = {
            "schema_version": "text2cad-text-token-lengths-v1",
            "provenance": {
                "texts_jsonl_sha256": sha256_file(manifest_dir / "texts.jsonl"),
                "bert_file_sha256": dict(sorted(PINNED_BERT_FILE_SHA256.items())),
                "method": "test",
                "model_loaded": False,
            },
            "max_seq_len": 512,
            "lengths": [
                {"text_index": 0, "n_bert_tokens": 5},
                {"text_index": 1, "n_bert_tokens": 6},
                {"text_index": 2, "n_bert_tokens": 7},
            ],
        }
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        import types

        args = types.SimpleNamespace(
            token_audit=str(audit_path), out=str(root / "sel.json"),
            repo_root=str(REPO_ROOT), pair_manifest_dir=str(manifest_dir),
        )
        with self.assertRaisesRegex(RunnerError, "official"):
            run_select_g3(args)
        self.assertFalse((root / "sel.json").exists())

    def test_cli_select_g3_happy_and_refusals(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        out = Path(self._tmp.name) / "g3.json"
        argv = [
            "--mode", "select-g3",
            "--pair-manifest-dir", str(WORKER_DIR),
            "--token-audit", str(TOKEN_AUDIT),
            "--repo-root", str(REPO_ROOT),
            "--out", str(out),
        ]
        self.assertEqual(g3_main(argv), 0)
        self.assertNotIn(b"\r", out.read_bytes())
        self.assertEqual(g3_main(argv), 2)  # refuses overwrite
        self.assertEqual(
            g3_main([
                "--mode", "select-g3", "--pair-manifest-dir", str(WORKER_DIR),
                "--repo-root", str(REPO_ROOT), "--out", str(Path(self._tmp.name) / "x.json"),
            ]),
            2,  # missing --token-audit
        )


# ---------------------------------------------------------------------------
# Runner probe scope: plan_chunks + scorer header (mini manifest, tiny model)
# ---------------------------------------------------------------------------


class _ProbeChunkMixin:
    def _binding(self, probe_sha=None) -> dict:
        binding = {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "pair_manifest_sha256": MANIFEST_SHA,
            "git_sha": FULL_SHA,
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "device": "cpu",
            "torch_dtype": "float32",
            "trim_mode": "none",
            "text_cache": False,
            "score_name": SCORE_NAME,
            "spec_version": SCORE_SPEC_VERSION,
        }
        if probe_sha is not None:
            binding[PROBE_SCOPE_FIELD] = probe_sha
        return binding

    def _write_chunk(self, out_dir, pair_space, t0, t1, *, probe_sha=None, binding=None) -> Path:
        binding = binding if binding is not None else self._binding()
        header = {
            "record_type": "chunk_header", "shard_id": 0, "text_range": [t0, t1],
            "n_pairs": (t1 - t0) * pair_space.n_cads,
            "n_scored": (t1 - t0) * pair_space.n_cads, "n_rejected": 0,
            "pair_index_first": t0 * pair_space.n_cads,
            "pair_index_last": t1 * pair_space.n_cads - 1,
            **binding,
        }
        if probe_sha is not None:
            header[PROBE_SCOPE_FIELD] = probe_sha
        lines = [json.dumps(header)]
        for pair_index in pair_space.chunk_pair_indices(t0, t1):
            text_index, cad_index = pair_space.coords_of(pair_index)
            lines.append(json.dumps({
                "record_type": "pair", "pair_index": pair_index,
                "text_index": text_index, "cad_index": cad_index, "status": "scored",
                "score_x": -1.0, "score_y": -2.0, "score_total": -3.0,
                "mean_x": -1.0, "mean_y": -2.0, "mean_total": -1.5,
                "n_valid_x": 1, "n_valid_y": 1, "n_valid_total": 2,
                "valid_seq_len": 2, "n_real_tokens": 2,
            }))
        path = out_dir / chunk_file_name(0, t0, t1)
        atomic_publish(path, lines)
        return path


class G3ProbePlanChunksTest(_ProbeChunkMixin, unittest.TestCase):
    """Mini space: 3 texts x 4 CADs, one shard; probe = texts {0, 2}."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        self.pair_space = PairSpace(3, 4)
        self.out_dir = self.root / "out"
        self.cfg = RunnerConfig(
            checkpoint_path="x", bert_dir="x",
            cad_seq_zip=str(self.root / "cad_seq.zip"),
            pair_manifest_dir=str(self.manifest_dir), device="cpu", batch_size=2,
        )

    def _plan(self, probe: bool = False, **kwargs):
        binding = self._binding(probe_sha=PROBE_SHA) if probe else self._binding()
        return plan_chunks(self.out_dir, self.pair_space, self.cfg, binding, **kwargs)

    def test_probe_filter_keeps_global_indices_full_gallery(self):
        todo, completed = self._plan(
            resume=False, probe_text_indices={0, 2},
            extra_binding_fields=(PROBE_SCOPE_FIELD,),
        )
        self.assertEqual(completed, [])
        self.assertEqual(todo, [(0, 1, 0), (2, 3, 8)])  # global pair offsets preserved
        # the text-1 chunk stays planned for the UNSCOPED path
        todo_full, _ = self._plan(resume=False, extra_binding_fields=(PROBE_SCOPE_FIELD,))
        self.assertEqual(todo_full, [(0, 1, 0), (1, 2, 4), (2, 3, 8)])

    def test_probe_filter_requires_single_text_chunks(self):
        cfg = RunnerConfig(
            checkpoint_path="x", bert_dir="x", cad_seq_zip="x",
            pair_manifest_dir=str(self.manifest_dir), texts_per_chunk=2,
        )
        with self.assertRaisesRegex(RunnerError, "texts_per_chunk"):
            plan_chunks(self.out_dir, self.pair_space, cfg, self._binding(),
                        resume=False, probe_text_indices={0, 2})

    def test_full_matrix_chunk_refused_under_probe_scope(self):
        self._write_chunk(self.out_dir, self.pair_space, 0, 1)  # no probe stamp
        with self.assertRaisesRegex(RunnerError, "chunk verification failed"):
            self._plan(probe=True, resume=True, probe_text_indices={0, 2},
                       extra_binding_fields=(PROBE_SCOPE_FIELD,))
        # a chunk for a non-probed text is outside the probe plan entirely
        self._write_chunk(self.out_dir, self.pair_space, 1, 2, probe_sha=PROBE_SHA)
        with self.assertRaisesRegex(RunnerError, "outside this shard's plan"):
            self._plan(probe=True, resume=True, probe_text_indices={0, 2},
                       extra_binding_fields=(PROBE_SCOPE_FIELD,))

    def test_probe_chunk_refused_under_full_matrix_scope(self):
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        with self.assertRaisesRegex(RunnerError, "chunk verification failed"):
            self._plan(resume=True, extra_binding_fields=(PROBE_SCOPE_FIELD,))

    def test_probe_resume_skips_verified_and_requires_timing(self):
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        binding = self._binding(probe_sha=PROBE_SHA)
        # completed chunk has a timing line -> skipped exactly once
        todo, completed = plan_chunks(
            self.out_dir, self.pair_space, self.cfg, binding, resume=True,
            probe_text_indices={0, 2}, extra_binding_fields=(PROBE_SCOPE_FIELD,),
            timed_text_indices={0},
        )
        self.assertEqual(completed, [chunk_file_name(0, 0, 1)])
        self.assertEqual(todo, [(2, 3, 8)])
        # ... but without its timing line the same chunk refuses the resume
        with self.assertRaisesRegex(RunnerError, "no sidecar timing record"):
            plan_chunks(
                self.out_dir, self.pair_space, self.cfg, binding, resume=True,
                probe_text_indices={0, 2}, extra_binding_fields=(PROBE_SCOPE_FIELD,),
                timed_text_indices=set(),
            )


class G3ProbeScorerHeaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        from idea01.text2cad_pair_index import load_cad_store, load_texts

        self.texts = load_texts(self.manifest_dir)
        self.members = load_cad_store(self.manifest_dir)

    def _score_header(self, probe_sha) -> dict:
        cfg = RunnerConfig(
            checkpoint_path="x", bert_dir="x",
            cad_seq_zip=str(self.root / "cad_seq.zip"),
            pair_manifest_dir=str(self.manifest_dir), device="cpu", batch_size=2,
        )
        ledger = SharedBudgetLedger(
            self.root / f"ledger-{probe_sha is not None}.json",
            approved_sha=FULL_SHA, max_pairs=100, max_card_seconds=1000.0,
        )
        store = ZipCandidateStore(
            self.root / "cad_seq.zip", self.members,
            expected_sha256=sha256_file(self.root / "cad_seq.zip"),
        )
        store.preload()
        ledger.begin_invocation("inv-1", shard_id=0, cards_used=1)
        binding = _ProbeChunkMixin._binding(self, probe_sha=probe_sha)
        scorer = FrozenMatrixScorer(
            cfg, build_tiny_model(), self.texts, store, PairSpace(3, 4),
            ledger, "inv-1", binding,
        )
        header = scorer.score_chunk(0, 1)[0]
        ledger.finalize_invocation("inv-1", exit_code=0)
        return header

    def test_scope_stamp_only_when_scoped(self):
        scoped = self._score_header(PROBE_SHA)
        self.assertEqual(scoped.get(PROBE_SCOPE_FIELD), PROBE_SHA)
        plain = self._score_header(None)
        self.assertNotIn(PROBE_SCOPE_FIELD, plain)
        # every other header field is unchanged between the two runs
        self.assertEqual(
            {k: v for k, v in scoped.items() if k != PROBE_SCOPE_FIELD}, plain
        )


# ---------------------------------------------------------------------------
# Timing sidecar
# ---------------------------------------------------------------------------


def _scope(**overrides) -> dict:
    scope = {
        PROBE_SCOPE_FIELD: PROBE_SHA,
        "git_sha": FULL_SHA,
        "pair_manifest_sha256": MANIFEST_SHA,
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "score_name": SCORE_NAME,
        "spec_version": SCORE_SPEC_VERSION,
        "device": "npu:0",
    }
    scope.update(overrides)
    return scope


def _timing_record(text_index: int, *, shard_id: int = 0, start: float = 100.0, e2e: float = 5.0,
                   inv: str = "inv-0", device: str = "npu:0", layer: int = 0, tokens: int = 10,
                   pick: int = 0, scope=None) -> dict:
    return build_query_timing_record(
        invocation_id=inv, shard_id=shard_id, device=device, text_index=text_index,
        probe_facts={"text_layer": layer, "n_bert_tokens": tokens, "shard_pick": pick},
        start_epoch=start, end_epoch=start + e2e,
        n_pairs=G3_N_PAIRS_PER_QUERY, n_batches=G3_N_BATCHES_PER_QUERY,
        n_scored=G3_N_PAIRS_PER_QUERY, n_rejected=0, n_valid_total=1000,
        chunk_file=chunk_file_name(shard_id, text_index, text_index + 1),
        scope=scope or _scope(device=device),
    )


class G3TimingSidecarTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.path = self.root / timing_sidecar_name(0)

    def test_append_read_roundtrip_fsync(self):
        for i, t in enumerate((69, 111, 933)):
            append_timing_record(self.path, _timing_record(t, start=100.0 + 10 * i, pick=i))
        records = read_timing_records(self.path)
        self.assertEqual([r["text_index"] for r in records], [69, 111, 933])
        for i, r in enumerate(records):
            self.assertEqual(r["schema_version"], G3_TIMING_SCHEMA_VERSION)
            self.assertEqual(r["window"]["start_epoch"], 100.0 + 10 * i)
            self.assertAlmostEqual(r["e2e_s"], r["window"]["end_epoch"] - r["window"]["start_epoch"])
            self.assertEqual(r["chunk_file"], chunk_file_name(0, r["text_index"], r["text_index"] + 1))
        self.assertNotIn(b"\r", self.path.read_bytes())

    def test_torn_last_line_fails_closed(self):
        append_timing_record(self.path, _timing_record(69))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write('{"record_type": "query_timing", "text_in')  # crash mid-line
        with self.assertRaisesRegex(RunnerError, "torn|complete JSON"):
            read_timing_records(self.path)

    def test_missing_sidecar_refused(self):
        with self.assertRaisesRegex(RunnerError, "does not exist"):
            read_timing_records(self.root / "nope.jsonl")

    def test_resume_side_text_set_scope_shard_duplicates(self):
        records = [_timing_record(69), _timing_record(111, pick=1)]
        texts = verified_sidecar_text_set(
            records, shard_id=0, scope=_scope(device="npu:0"), path_label="sidecar"
        )
        self.assertEqual(texts, {69, 111})
        with self.assertRaisesRegex(RunnerError, "scope"):
            verified_sidecar_text_set(
                records, shard_id=0, scope=_scope(device="npu:0", git_sha="d" * 40),
                path_label="sidecar",
            )
        with self.assertRaisesRegex(RunnerError, "shard"):
            verified_sidecar_text_set(
                records, shard_id=1, scope=_scope(device="npu:0"), path_label="sidecar"
            )
        with self.assertRaisesRegex(RunnerError, "twice"):
            verified_sidecar_text_set(
                [records[0], records[0]], shard_id=0, scope=_scope(device="npu:0"),
                path_label="sidecar",
            )
        with self.assertRaisesRegex(RunnerError, "query_timing"):
            verified_sidecar_text_set(
                [{"record_type": "other"}], shard_id=0, scope=_scope(device="npu:0"),
                path_label="sidecar",
            )


# ---------------------------------------------------------------------------
# Two-card ETA (synthetic official-size selection)
# ---------------------------------------------------------------------------


def _synthetic_selection() -> dict:
    lengths = {i: (i * 37) % 140 + 4 for i in range(EXPECTED_N_UNIQUE_TEXT)}
    text_records = [{"text_index": i} for i in range(EXPECTED_N_UNIQUE_TEXT)]
    return build_g3_probe_selection(text_records, EXPECTED_N_GALLERY, lengths)


def _probe_summary(device: str, *, cold: float, inv: str, host: str = "hostA",
                   ledger: str = "/budget/g3.json", git_sha: str = FULL_SHA,
                   probe_sha: str = PROBE_SHA, manifest_sha: str = MANIFEST_SHA,
                   batch_size: int = G3_BATCH_SIZE, dtype: str = "float32",
                   trim: str = "none", cache: bool = False, cards_used: int = 1,
                   finalized: bool = True, exit_code: int = 0,
                   completed_now: int = 50, completed_before: int = 0,
                   status: str = "ok", failure=None, peak: int = 1_900_000_000) -> dict:
    shard = SHARD_BY_DEVICE[device]
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "time_utc": "2026-10-02T00:00:00Z", "platform": "test", "python_version": "3.11",
        "torch_version": "2.7.1+cpu", "torch_npu_version": "2.7.1.post1",
        "transformers_version": "4.53.1",
        "device": device, "torch_dtype": dtype, "batch_size": batch_size,
        "trim_mode": trim, "text_cache": cache, "shard_id": shard, "n_shards": 2,
        "texts_per_chunk": 1, "checkpoint_path": "ckpt",
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256, "bert_dir": "bert",
        "bert_file_sha256": {}, "cad_seq_zip": "zip",
        "score_name": SCORE_NAME, "spec_version": SCORE_SPEC_VERSION,
        "max_cad_sequence_length": 272, "env_vars_present": [],
        "git_sha": git_sha, "pair_manifest_sha256": manifest_sha,
        PROBE_SCOPE_FIELD: probe_sha, "timing_file": timing_sidecar_name(shard),
        "cold_start_s": cold, "peak_device_memory_bytes": peak, "hbm_device": device,
        "probe_queries_total": 50, "invocation_id": inv,
        "authorization": {"approved_sha": git_sha, "task_id": TASK_ID},
        "chunks_completed_now": completed_now, "chunks_previously_completed": completed_before,
        "pairs_planned": 402300,
        "stats": {"batches": G3_N_BATCHES_PER_QUERY * completed_now,
                  "pairs_forwarded": G3_N_PAIRS_PER_QUERY * completed_now,
                  "pairs_rejected": 0},
        "load_report": {}, "candidate_preload": {},
        "budget": {
            "budget_file": ledger, "approved_sha": git_sha,
            "max_pairs": 804600, "max_card_seconds": 43200.0,
            "pairs_charged": 804600, "remaining_pairs": 0, "card_seconds_spent": 50.0,
            "invocations": [{
                "invocation_id": inv, "shard_id": shard, "cards_used": cards_used,
                "pid": 1234, "host": host, "started_epoch": 1.0,
                "last_heartbeat_epoch": 2.0, "finalized": finalized,
                "ended_epoch": 3.0, "card_seconds": 2.0, "exit_code": exit_code,
            }],
        },
        "stale_invocations_auto_finalized": [], "status": status, "failure": failure,
    }


def _card_records(selection: dict, device: str, *, base: float, step: float,
                  e2e_fn, inv: str, git_sha: str = FULL_SHA,
                  probe_sha: str = PROBE_SHA, manifest_sha: str = MANIFEST_SHA) -> list[dict]:
    shard = SHARD_BY_DEVICE[device]
    facts = probe_facts_for_shard(selection, shard)
    records = []
    for k, text_index in enumerate(sorted(facts)):
        e2e = float(e2e_fn(facts[text_index], k))
        start = base + k * step
        records.append(build_query_timing_record(
            invocation_id=inv, shard_id=shard, device=device, text_index=text_index,
            probe_facts=facts[text_index], start_epoch=start, end_epoch=start + e2e,
            n_pairs=G3_N_PAIRS_PER_QUERY, n_batches=G3_N_BATCHES_PER_QUERY,
            n_scored=G3_N_PAIRS_PER_QUERY, n_rejected=0, n_valid_total=1000,
            chunk_file=chunk_file_name(shard, text_index, text_index + 1),
            scope=_scope(device=device, git_sha=git_sha, probe_selection_sha256=probe_sha,
                         pair_manifest_sha256=manifest_sha),
        ))
    return records


def _evidence(selection: dict, *, cold0=10.0, cold1=20.0, e2e0=None, e2e1=None,
              base0=1000.0, base1=1000.0, step=10.0, **summary_kw) -> dict:
    e2e0 = e2e0 or (lambda facts, k: {0: 8.0, 1: 5.0, 2: 3.0}[facts["text_layer"]])
    e2e1 = e2e1 or (lambda facts, k: {0: 6.0, 1: 6.0, 2: 6.0}[facts["text_layer"]])
    kw0 = {k: v for k, v in summary_kw.items() if not k.endswith("__1")}
    kw1 = {k[:-3]: v for k, v in summary_kw.items() if k.endswith("__1")}
    return {
        "npu:0": {
            "summary": _probe_summary("npu:0", cold=cold0, inv="inv-card0", **kw0),
            "timing_records": _card_records(selection, "npu:0", base=base0, step=step,
                                            e2e_fn=e2e0, inv="inv-card0",
                                            **{k: kw0[k] for k in ("git_sha", "probe_sha", "manifest_sha") if k in kw0}),
        },
        "npu:1": {
            "summary": _probe_summary("npu:1", cold=cold1, inv="inv-card1", **kw1),
            "timing_records": _card_records(selection, "npu:1", base=base1, step=step,
                                            e2e_fn=e2e1, inv="inv-card1",
                                            **{k: kw1[k] for k in ("git_sha", "probe_sha", "manifest_sha") if k in kw1}),
        },
    }


def _expected_compute(selection: dict, shard: int, slowest_by_layer: dict[int, float]) -> float:
    entry = next(s for s in selection["shards"] if s["shard_id"] == shard)
    return sum(
        slowest_by_layer[int(layer)] * int(count)
        for layer, count in entry["layer_text_counts"].items()
    )


class G3ETATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = _synthetic_selection()

    def _estimate(self, evidence, **kw):
        return estimate_g3_probe_eta(
            self.selection, evidence, expected_probe_selection_sha256=PROBE_SHA, **kw
        )

    def test_hand_computed_slower_card_per_layer_slowest(self):
        evidence = _evidence(self.selection)
        estimate = self._estimate(evidence)
        compute0 = _expected_compute(self.selection, 0, {0: 8.0, 1: 5.0, 2: 3.0})
        compute1 = _expected_compute(self.selection, 1, {0: 6.0, 1: 6.0, 2: 6.0})
        self.assertAlmostEqual(estimate["per_card"]["npu:0"]["compute_seconds"], compute0)
        self.assertAlmostEqual(estimate["per_card"]["npu:1"]["compute_seconds"], compute1)
        wall = max(10.0 + compute0, 20.0 + compute1)
        self.assertAlmostEqual(estimate["wall_seconds_presafety"], wall)
        self.assertAlmostEqual(estimate["eta_seconds"], wall * 1.30)
        self.assertEqual(estimate["slower_card"], "npu:0" if 10.0 + compute0 >= 20.0 + compute1 else "npu:1")
        self.assertEqual(estimate["window_overlap_fraction"], 1.0)
        self.assertEqual(estimate["probe_pairs_total"], 804600)
        for device, shard in (("npu:0", 0), ("npu:1", 1)):
            card = estimate["per_card"][device]
            self.assertEqual(card["n_probe_queries"], 50)
            self.assertEqual(card["n_pairs_scored"], 50 * EXPECTED_N_GALLERY)
            entry = next(s for s in self.selection["shards"] if s["shard_id"] == shard)
            for layer, info in card["layers"].items():
                self.assertEqual(info["n_full_texts"], entry["layer_text_counts"][layer])
                self.assertEqual(info["n_probe"], entry["layer_probe_counts"][layer])
                self.assertGreater(info["slowest_e2e_s"], 0)
                self.assertAlmostEqual(info["seconds"], info["slowest_e2e_s"] * info["n_full_texts"])

    def test_slowest_query_prices_the_layer_not_the_mean(self):
        # one 8s query among 0.5s queries in every layer of card 0
        def e2e0(facts, k):
            return 8.0 if k == 0 else 0.5

        def e2e1(facts, k):
            return 0.5

        evidence = _evidence(self.selection, e2e0=e2e0, e2e1=e2e1)
        estimate = self._estimate(evidence)
        entry = next(s for s in self.selection["shards"] if s["shard_id"] == 0)
        first_layer = probe_facts_for_shard(self.selection, 0)[
            sorted(probe_facts_for_shard(self.selection, 0))[0]]["text_layer"]
        for layer, info in estimate["per_card"]["npu:0"]["layers"].items():
            expected = 8.0 if int(layer) == first_layer else 0.5
            self.assertAlmostEqual(info["slowest_e2e_s"], expected)
            self.assertAlmostEqual(info["seconds"], expected * info["n_full_texts"])

    def test_seven_day_gate_both_sides(self):
        # eta <= 604800 passes; eta > 604800 fails -- never widened
        compute0 = _expected_compute(self.selection, 0, {0: 8.0, 1: 5.0, 2: 3.0})
        compute1 = _expected_compute(self.selection, 1, {0: 6.0, 1: 6.0, 2: 6.0})
        wall = max(10.0 + compute0, 20.0 + compute1)
        for target_wall, expect_pass in ((465230.0, True), (465231.0, False)):
            evidence = _evidence(self.selection)
            overhead = 30.0 + (target_wall - wall)  # cold_sum = 30
            estimate = self._estimate(evidence, overhead_seconds=overhead)
            self.assertAlmostEqual(estimate["wall_seconds_presafety"], target_wall)
            self.assertEqual(estimate["passes_seven_day_gate"], expect_pass)
            self.assertEqual(estimate["eta_seconds"] <= 604800.0, expect_pass)

    def test_overhead_rules(self):
        evidence = _evidence(self.selection)
        # None keeps the measured cold starts as the only overhead
        estimate = self._estimate(evidence, overhead_seconds=None)
        self.assertEqual(estimate["extra_overhead_seconds"], 0.0)
        self.assertEqual(estimate["cold_start_seconds_sum"], 30.0)
        # explicit overhead below the cold-start sum is refused
        with self.assertRaisesRegex(RunnerError, "cold start"):
            self._estimate(evidence, overhead_seconds=29.0)
        with self.assertRaisesRegex(RunnerError, "safety_factor"):
            self._estimate(evidence, safety_factor=1.29)

    def test_window_overlap_gate(self):
        # card windows: length = 49*10 + max e2e = 498; overlap >= 0.8 needs
        # the stagger < ~99.6 s
        estimate = self._estimate(_evidence(self.selection, base1=1000.0 + 50.0))
        self.assertGreaterEqual(estimate["window_overlap_fraction"], 0.8)
        with self.assertRaisesRegex(RunnerError, "overlap"):
            self._estimate(_evidence(self.selection, base1=1000.0 + 200.0))

    def test_fail_closed_refusals(self):
        base = _evidence(self.selection)

        def refused(mutate, pattern):
            import copy

            evidence = copy.deepcopy(base)
            mutate(evidence)
            with self.assertRaisesRegex(RunnerError, pattern):
                self._estimate(evidence)

        # device / shard / host / ledger identity
        refused(lambda e: e["npu:0"]["summary"].update(device="npu:1"), "device")
        refused(lambda e: e["npu:0"]["summary"].update(shard_id=1), "correspondence")
        refused(lambda e: e["npu:1"]["summary"]["budget"]["invocations"][0].update(host="hostB"),
                "different hosts")
        refused(lambda e: e["npu:1"]["summary"]["budget"].update(budget_file="/other.json"),
                "different ledgers")

        def same_invocation_twice(e):
            e["npu:1"]["summary"].update(invocation_id="inv-card0")
            e["npu:1"]["summary"]["budget"]["invocations"][0].update(invocation_id="inv-card0")

        refused(same_invocation_twice, "distinct")
        # ledger outcome
        refused(lambda e: e["npu:0"]["summary"]["budget"]["invocations"][0].update(finalized=False),
                "finalized")
        refused(lambda e: e["npu:0"]["summary"]["budget"]["invocations"][0].update(exit_code=1),
                "exit_code")
        refused(lambda e: e["npu:0"]["summary"]["budget"]["invocations"][0].update(cards_used=2),
                "cards_used")
        refused(lambda e: e["npu:0"]["summary"].update(status="failed", failure="boom"), "successful")
        # production shape and frozen reference configuration
        refused(lambda e: e["npu:0"]["summary"].update(batch_size=8), "batch_size")
        refused(lambda e: e["npu:0"]["summary"].update(torch_dtype="bfloat16"), "reference")
        refused(lambda e: e["npu:0"]["summary"].update(trim_mode="batch_max"), "reference")
        refused(lambda e: e["npu:0"]["summary"].update(text_cache=True), "reference")
        refused(lambda e: e["npu:0"]["summary"].update(cold_start_s=-1.0), "cold_start")
        refused(lambda e: e["npu:0"]["summary"].update(peak_device_memory_bytes=None), "HBM")
        refused(lambda e: e["npu:0"]["summary"].update(hbm_device="npu:1"), "HBM")
        # scope / fingerprint agreement
        refused(lambda e: e["npu:1"]["summary"].update(**{PROBE_SCOPE_FIELD: "e" * 64}), "scope|bound")

        def drift_git_sha(e):
            # consistent auth/budget so the failure is the scope disagreement
            e["npu:1"]["summary"].update(git_sha="f" * 40)
            e["npu:1"]["summary"]["authorization"].update(approved_sha="f" * 40)
            e["npu:1"]["summary"]["budget"].update(approved_sha="f" * 40)

        refused(drift_git_sha, "git_sha|scope|disagree")
        refused(lambda e: e["npu:1"]["summary"].update(git_sha="short"), "git_sha")
        refused(lambda e: e["npu:0"]["summary"].update(authorization={"approved_sha": "0" * 40}),
                "authorization")
        refused(lambda e: e["npu:1"]["summary"].update(pair_manifest_sha256="0" * 64),
                "disagree|scope")
        # chunk accounting vs timed queries
        refused(lambda e: e["npu:0"]["summary"].update(chunks_completed_now=49,
                                                       chunks_previously_completed=0),
                "accounting")
        # single device / one card's evidence filed under both devices
        with self.assertRaisesRegex(RunnerError, "per device"):
            self._estimate({"npu:0": base["npu:0"]})
        with self.assertRaisesRegex(RunnerError, "device"):
            self._estimate({"npu:0": base["npu:0"], "npu:1": base["npu:0"]})

    def test_timing_record_refusals(self):
        base = _evidence(self.selection)

        def refused(mutate, pattern):
            import copy

            evidence = copy.deepcopy(base)
            mutate(evidence["npu:0"]["timing_records"])
            with self.assertRaisesRegex(RunnerError, pattern):
                self._estimate(evidence)

        refused(lambda r: r.pop(0), "no timing record")
        refused(lambda r: r.append(dict(r[0])), "twice")
        refused(lambda r: r[0].update(n_pairs=8045), "complete gallery")
        refused(lambda r: r[0].update(n_batches=502), "production count")
        refused(lambda r: r[0].update(n_scored=8045), "not complete")
        refused(lambda r: r[0].update(n_rejected=1), "not complete")
        refused(lambda r: r[0].update(e2e_s=r[0]["e2e_s"] + 1.0), "end-start")
        refused(lambda r: r[0].update(text_layer=(r[0]["text_layer"] + 1) % 3), "facts")
        refused(lambda r: r[0].update(n_bert_tokens=r[0]["n_bert_tokens"] + 1), "facts")
        refused(lambda r: r[0].update(text_index=9999), "not one of this shard")
        refused(lambda r: r[0].update(shard_id=1), "mismatch")
        refused(lambda r: r[0].update(**{PROBE_SCOPE_FIELD: "e" * 64}), "scope")
        refused(lambda r: r[0].update(chunk_file="chunk_s00_t09998_09999.jsonl"), "chunk_file")
        refused(lambda r: r[0].update(n_valid_total=0), "n_valid_total")
        refused(lambda r: r[0]["window"].update(end_epoch=r[0]["window"]["start_epoch"] - 1.0),
                "ordered|end-start")

    def test_missing_layer_refused(self):
        def e2e0(facts, k):
            return 5.0

        evidence = _evidence(self.selection, e2e0=e2e0)
        # drop every layer-2 record of card 0 and re-tag them into layer 1
        layer2 = [r for r in evidence["npu:0"]["timing_records"] if r["text_layer"] == 2]
        self.assertTrue(layer2)
        records = [r for r in evidence["npu:0"]["timing_records"] if r["text_layer"] != 2]
        for r in layer2:
            r["text_layer"] = 1
        records.extend(layer2)
        evidence["npu:0"]["timing_records"] = records
        with self.assertRaisesRegex(RunnerError, "facts"):
            self._estimate(evidence)
        # simply deleting them instead leaves the layer unpriced
        evidence2 = _evidence(self.selection, e2e0=e2e0)
        evidence2["npu:0"]["timing_records"] = [
            r for r in evidence2["npu:0"]["timing_records"] if r["text_layer"] != 2
        ]
        with self.assertRaisesRegex(RunnerError, "no timing record"):
            self._estimate(evidence2)


# ---------------------------------------------------------------------------
# eta-g3 CLI end-to-end against the real committed selection
# ---------------------------------------------------------------------------


@unittest.skipUnless(REAL_ASSETS_PRESENT, "real frozen worker artifacts required")
class G3EtaCliRealDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.selection = json.loads(G3_CONFIG.read_text(encoding="utf-8"))
        self.probe_sha = sha256_file(G3_CONFIG)
        self.manifest_sha = "62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557"

    def _card_dir(self, device: str, *, name: str = None, base: float, cold: float, slow: float) -> Path:
        shard = SHARD_BY_DEVICE[device]
        card_dir = self.root / (name or f"card{shard}")
        card_dir.mkdir()
        facts = probe_facts_for_shard(self.selection, shard)
        records = []
        for k, text_index in enumerate(sorted(facts)):
            e2e = slow if k == 0 else 1.0
            start = base + k * 10.0
            record = _timing_record(
                text_index, shard_id=shard, start=start, e2e=e2e,
                inv=f"inv-card{shard}", device=device,
                layer=facts[text_index]["text_layer"],
                tokens=facts[text_index]["n_bert_tokens"],
                pick=facts[text_index]["shard_pick"],
                scope=_scope(device=device, pair_manifest_sha256=self.manifest_sha),
            )
            record[PROBE_SCOPE_FIELD] = self.probe_sha
            append_timing_record(card_dir / timing_sidecar_name(shard), record)
            (card_dir / record["chunk_file"]).write_text("", encoding="utf-8")
        summary = _probe_summary(
            device, cold=cold, inv=f"inv-card{shard}", probe_sha=self.probe_sha,
            manifest_sha=self.manifest_sha,
        )
        (card_dir / f"summary_{shard}.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return card_dir

    def _argv(self, s0: Path, s1: Path, out: Path) -> list[str]:
        return [
            "--mode", "eta-g3",
            "--pair-manifest-dir", str(WORKER_DIR),
            "--token-audit", str(TOKEN_AUDIT),
            "--repo-root", str(REPO_ROOT),
            "--card-summary", str(s0),
            "--card-summary", str(s1),
            "--out", str(out),
        ]

    def test_happy_path_hand_computed(self):
        d0 = self._card_dir("npu:0", base=1000.0, cold=10.0, slow=8.0)
        d1 = self._card_dir("npu:1", base=1000.0, cold=20.0, slow=6.0)
        out = self.root / "eta.json"
        rc = g3_main(self._argv(d0 / "summary_0.json", d1 / "summary_1.json", out))
        self.assertEqual(rc, 0)
        estimate = json.loads(out.read_text(encoding="utf-8"))
        self.assertNotIn(b"\r", out.read_bytes())
        first_layer0 = probe_facts_for_shard(self.selection, 0)[
            sorted(probe_facts_for_shard(self.selection, 0))[0]]["text_layer"]
        first_layer1 = probe_facts_for_shard(self.selection, 1)[
            sorted(probe_facts_for_shard(self.selection, 1))[0]]["text_layer"]
        slow0 = {0: 1.0, 1: 1.0, 2: 1.0, first_layer0: 8.0}
        slow1 = {0: 1.0, 1: 1.0, 2: 1.0, first_layer1: 6.0}
        compute0 = _expected_compute(self.selection, 0, slow0)
        compute1 = _expected_compute(self.selection, 1, slow1)
        self.assertAlmostEqual(estimate["per_card"]["npu:0"]["compute_seconds"], compute0)
        self.assertAlmostEqual(estimate["per_card"]["npu:1"]["compute_seconds"], compute1)
        wall = max(10.0 + compute0, 20.0 + compute1)
        self.assertAlmostEqual(estimate["eta_seconds"], wall * 1.30)
        self.assertTrue(estimate["passes_seven_day_gate"])
        self.assertEqual(estimate["inputs"]["g3_probe_selection_sha256"], self.probe_sha)
        # refuses to overwrite the estimate
        self.assertEqual(
            g3_main(self._argv(d0 / "summary_0.json", d1 / "summary_1.json", out)), 2
        )

    def test_cli_refusals(self):
        d0 = self._card_dir("npu:0", base=1000.0, cold=10.0, slow=8.0)
        d1 = self._card_dir("npu:1", base=1000.0 + 400.0, cold=20.0, slow=6.0)
        # staggered windows (400s >> 0.2 * ~498s shorter window) refuse
        self.assertEqual(
            g3_main(self._argv(d0 / "summary_0.json", d1 / "summary_1.json", self.root / "x.json")),
            2,
        )
        # aligned windows pass
        d1b = self._card_dir("npu:1", name="card1b", base=1000.0, cold=20.0, slow=6.0)
        self.assertEqual(
            g3_main(self._argv(d0 / "summary_0.json", d1b / "summary_1.json", self.root / "y.json")),
            0,
        )
        # a missing chunk file behind a timing line refuses
        first_text = sorted(probe_facts_for_shard(self.selection, 1))[0]
        (d1b / chunk_file_name(1, first_text, first_text + 1)).unlink()
        self.assertEqual(
            g3_main(self._argv(d0 / "summary_0.json", d1b / "summary_1.json", self.root / "z.json")),
            2,
        )
        # one summary only
        self.assertEqual(
            g3_main([
                "--mode", "eta-g3", "--pair-manifest-dir", str(WORKER_DIR),
                "--token-audit", str(TOKEN_AUDIT), "--repo-root", str(REPO_ROOT),
                "--card-summary", str(d0 / "summary_0.json"),
            ]),
            2,
        )


# ---------------------------------------------------------------------------
# Runner main() probe gates (refuse before authorization)
# ---------------------------------------------------------------------------


class G3RunnerMainGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        self.repo = self.root / "repo"
        cfg_dir = self.repo / "experiments" / "configs"
        cfg_dir.mkdir(parents=True)
        shutil.copyfile(
            self.root / "expected_hashes.json",
            cfg_dir / "text2cad_pair_manifest_expected_20260930.json",
        )

    def _args(self, *extra: str, mode: str = "score") -> list[str]:
        return [
            "--mode", mode,
            "--checkpoint", str(self.root / "ckpt.pth"),
            "--bert-dir", str(self.root / "bert"),
            "--cad-seq-zip", str(self.root / "cad_seq.zip"),
            "--pair-manifest-dir", str(self.manifest_dir),
            "--expected-keys-manifest", str(self.root / "keys.json"),
            "--out-dir", str(self.root / "out"),
            "--repo-root", str(self.repo),
            "--n-shards", "2", "--shard-id", "0", "--device", "npu:0",
            "--batch-size", "16",
            *extra,
        ]

    def _committed_probe_path(self) -> Path:
        return self.repo / G3_PROBE_SELECTION_REL

    def test_probe_gates_fire_before_authorization(self):
        cases = [
            (("--probe-selection", str(self._committed_probe_path())), "does not exist"),
            (("--probe-selection", str(self.root / "foreign.json")), "committed reviewed probe selection"),
            (("--probe-selection", "x", "--texts-per-chunk", "2"), "texts_per_chunk"),
            (("--probe-selection", "x", "--batch-size", "8"), "batch"),
            (("--token-audit", "x"), "only meaningful"),
        ]
        for extra, pattern in cases:
            with self.subTest(extra=extra), mock.patch(
                "idea01.text2cad_frozen_runner.authorize_text2cad_run"
            ) as auth:
                self.assertEqual(runner_main(self._args(*extra)), 2)
                auth.assert_not_called()

    def test_probe_refused_in_dry_run(self):
        with mock.patch("idea01.text2cad_frozen_runner.authorize_text2cad_run") as auth:
            self.assertEqual(
                runner_main(self._args("--probe-selection", "x", mode="dry-run")), 2
            )
            auth.assert_not_called()

    @unittest.skipUnless(G3_CONFIG.is_file(), "committed G3 config required")
    def test_committed_probe_config_verified_against_live_manifest(self):
        # the real committed config binds the REAL artifacts; the mini
        # manifest here cannot satisfy its provenance -> refuse pre-auth
        shutil.copyfile(G3_CONFIG, self._committed_probe_path())
        with mock.patch("idea01.text2cad_frozen_runner.authorize_text2cad_run") as auth:
            self.assertEqual(
                runner_main(self._args(
                    "--probe-selection", str(self._committed_probe_path()),
                    "--token-audit", str(TOKEN_AUDIT),
                )),
                2,
            )
            auth.assert_not_called()
        # missing --token-audit refuses at the same gate (config now present)
        with mock.patch("idea01.text2cad_frozen_runner.authorize_text2cad_run") as auth:
            self.assertEqual(
                runner_main(self._args(
                    "--probe-selection", str(self._committed_probe_path()),
                )),
                2,
            )
            auth.assert_not_called()


# ---------------------------------------------------------------------------
# Source hygiene: the G3 module never references relevance artifacts
# ---------------------------------------------------------------------------


class G3SourceHygieneTest(unittest.TestCase):
    def test_g3_module_never_reads_relevance_artifacts(self):
        import ast

        path = REPO_ROOT / "experiments" / "src" / "idea01" / "text2cad_g3_probe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tokens: list[str] = []

        def is_docstring(node) -> bool:
            return (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )

        def walk(node):
            if is_docstring(node):
                return
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                tokens.append(node.value)
            if isinstance(node, (ast.Name, ast.Attribute)):
                tokens.append(node.id if isinstance(node, ast.Name) else node.attr)
            for child in ast.iter_child_nodes(node):
                walk(child)

        walk(tree)
        for token in tokens:
            self.assertNotIn("qrels", token)
            self.assertNotIn("evaluation_map", token)
            self.assertNotIn(".csv", token.lower())
            # the selector binds the audited token lengths by SHA; it never
            # imports the tokenizer library itself
            self.assertNotIn("transformers", token.lower())


if __name__ == "__main__":
    unittest.main()
