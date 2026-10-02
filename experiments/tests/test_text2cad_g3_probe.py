"""CPU tests for the Text2CAD G3 full-gallery probe (2026-10-02, R1 hardened).

Contract: experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md and
experiments/plans/IDEA01_M1C_G3_R1_FIX_2026-10-02.md. Covers, with small fake
data / fake models and NO model or BERT forward:

  * pick/layer math: the pre-registered evenly spaced rank picks (endpoints,
    exact count, dedup-shrink refusal) and the global token-length layers
    (identical to the G2 terciles);
  * the blind selector against the REAL frozen worker artifacts (skipUnless
    guarded): 50+50 probes including each shard's shortest/longest text,
    per-shard-per-layer coverage, deterministic regeneration byte-identity,
    the pinned reviewed selection SHA, provenance/asset/dimension tamper
    refusals, LF output, no overwrite;
  * runner probe scope on a mini manifest: probe filtering keeps the global
    indices and the full gallery, full-matrix chunks refuse under the probe
    scope and vice versa, out-of-plan chunks refuse, resume requires the
    sidecar timing line of every completed chunk;
  * scorer header: the probe scope stamp appears only in scoped runs;
  * timing sidecar: append/read roundtrip, torn-line refusal, minimal
    resume-side scope/shard/duplicate checks plus the STRICT R1 resume
    validation (foreign probe line, bad window/counts);
  * probe chunk content: real read + dense pair sequence + binding/scope +
    finite complete rows + audited valid-item counts + timing cross-checks
    (empty/missing/duplicate/non-finite/wrong-scope/torn chunks refuse);
  * the two-card ETA on a synthetic OFFICIAL-SIZE selection: hand-computed
    per-layer slowest-query pricing, slower-card wall clock, cold starts,
    overhead rules, 80% window overlap, the 604,800 s gate both sides, the
    fail-closed refusals (device/shard/host/ledger/scope/count/window) and
    the R1 evidence gates: fixed-asset binding (pair manifest/BERT/full-272/
    measured load+preload reports), real accounting (stats/planned/charged/
    caps/card-seconds), timing<->invocation binding, the FINAL shared ledger
    (missing/extra/failed/drifted ledgers refuse), fresh-run-only accounting,
    and a legitimate pending-peer snapshot accepted;
  * the R2 bindings: authorization.card_hours*3600 == the card's ledger
    snapshot cap == the FINAL ledger cap, both cards authorized equally
    (0.1 h against a 43,200 s cap, a 3,001 s final cap against 43,200 s
    snapshots, and cross-card disagreement all refuse), and the same-card
    SERIAL query timeline in production text_index order (all-50-equal
    starts, adjacent overlap and reversed windows refuse; back-to-back and
    legally spaced large-duration timelines with matching ledger intervals
    pass);
  * the runner resume gate: fresh run refuses an existing sidecar; orphan /
    foreign / duplicate / invalid sidecar lines refuse BEFORE any forward;
    sidecar<->chunk one-to-one correspondence;
  * a CPU end-to-end producer->reader path: the REAL runner loop helpers
    (fake model, real chunk publish / timing sidecar / budget ledger / run
    summary) feed the REAL ETA reader verifiers;
  * runner main() probe gates exit 2 before authorization;
  * eta-g3 CLI end-to-end against the real committed selection, with REAL
    8,046-row chunks, the --final-ledger requirement and content refusals.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch  # noqa: F401  (parity with the runner test module)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.cadmium_runner import RunAuthorization, RunnerError  # noqa: E402
from idea01.text2cad_g3_probe import (  # noqa: E402
    CHUNK_SCHEMA_VERSION as G3_CHUNK_SCHEMA_VERSION,
    EXPECTED_PAIR_HASHES_REL,
    G2_SELECTION_CONFIG_REL,
    G3_APPROVED_MAX_CARD_HOURS,
    G3_APPROVED_MAX_CARD_SECONDS,
    G3_APPROVED_MAX_PAIRS,
    G3_BATCH_SIZE,
    G3_EXPECTED_NONBERT_TENSORS,
    G3_MAX_CAD_SEQUENCE_LENGTH,
    G3_PROBE_SELECTION_REL,
    G3_PROVENANCE_KEYS,
    G3_SELECTION_SCHEMA_VERSION,
    G3_TASK_ID,
    G3_TIMING_SCHEMA_VERSION,
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
    verify_final_probe_ledger,
    verify_g3_g2_config_binding,
    verify_g3_probe_selection,
    verify_g3_probe_selection_against_manifest,
    verify_g3_token_audit,
    verify_probe_budget_evidence,
    verify_probe_chunk_content,
    verify_probe_resume_records,
    verify_probe_summary,
    verify_probe_timing_records,
)
from idea01.text2cad_bench_protocol import SHARD_BY_DEVICE  # noqa: E402
from idea01.text2cad_frozen_runner import (  # noqa: E402
    CHUNK_SCHEMA_VERSION,
    FrozenMatrixScorer,
    RunnerConfig,
    SharedBudgetLedger,
    ZipCandidateStore,
    atomic_publish,
    build_run_summary,
    execute_probe_query,
    load_probe_sidecar_timed_texts,
    main as runner_main,
    plan_chunks,
    require_sidecar_chunk_correspondence,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    EXPECTED_N_GALLERY,
    EXPECTED_N_UNIQUE_TEXT,
    PairSpace,
    chunk_file_name,
    load_cad_length_index,
    sha256_file,
)
from idea01.text2cad_vector_scorer import validate_cad_vec  # noqa: E402

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

    def test_r1_duplicated_constants_match_their_audited_sources(self):
        from idea01.text2cad_port import model as port_model
        from idea01 import text2cad_frozen_runner

        self.assertEqual(G3_MAX_CAD_SEQUENCE_LENGTH, port_model.MAX_CAD_SEQUENCE_LENGTH)
        self.assertEqual(G3_TASK_ID, text2cad_frozen_runner.TASK_ID)
        self.assertEqual(G3_CHUNK_SCHEMA_VERSION, text2cad_frozen_runner.CHUNK_SCHEMA_VERSION)
        self.assertEqual(G3_APPROVED_MAX_PAIRS, 100 * EXPECTED_N_GALLERY)
        self.assertEqual(G3_APPROVED_MAX_CARD_HOURS, 12.0)  # "at most", never "at least"
        self.assertEqual(G3_APPROVED_MAX_CARD_SECONDS, G3_APPROVED_MAX_CARD_HOURS * 3600.0)
        keys_manifest = REPO_ROOT / "experiments" / "configs" / "text2cad_checkpoint_expected_keys_20260930.json"
        if not keys_manifest.is_file():
            self.skipTest("committed 172-key manifest not present")
        keys = json.loads(keys_manifest.read_text(encoding="utf-8"))
        self.assertEqual(G3_EXPECTED_NONBERT_TENSORS, keys["n_tensors"])
        self.assertEqual(G3_EXPECTED_NONBERT_TENSORS, len(keys["tensors"]))

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
        # the committed selection is exactly the reviewed pin
        self.assertEqual(
            sha256_file(G3_CONFIG),
            "f8521598b1bf517f21491bf8185cdd23cae39a34a7019389d4a05c6fcbdd4985",
        )
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
                "score_name": SCORE_NAME, "spec_version": SCORE_SPEC_VERSION,
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
                   pick: int = 0, scope=None, n_valid: int = 1000) -> dict:
    return build_query_timing_record(
        invocation_id=inv, shard_id=shard_id, device=device, text_index=text_index,
        probe_facts={"text_layer": layer, "n_bert_tokens": tokens, "shard_pick": pick},
        start_epoch=start, end_epoch=start + e2e,
        n_pairs=G3_N_PAIRS_PER_QUERY, n_batches=G3_N_BATCHES_PER_QUERY,
        n_scored=G3_N_PAIRS_PER_QUERY, n_rejected=0, n_valid_total=n_valid,
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
# Probe chunk content verification (R1: read, never trust existence)
# ---------------------------------------------------------------------------


class G3ChunkContentTest(_ProbeChunkMixin, unittest.TestCase):
    """Mini space 3 texts x 4 CADs; one single-query probe chunk for text 0."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.pair_space = PairSpace(3, 4)
        self.out_dir = self.root / "out"
        self.binding = self._binding(probe_sha=PROBE_SHA)
        self.cad_valid = [
            {"cad_index": c, "n_real_tokens": 2, "n_valid_x": 1, "n_valid_y": 1}
            for c in range(4)
        ]
        self.timing = {
            "n_scored": 4, "n_rejected": 0, "n_valid_total": 8,
            "chunk_file": chunk_file_name(0, 0, 1),
        }

    def _verify(self, timing=None):
        return verify_probe_chunk_content(
            self.out_dir / chunk_file_name(0, 0, 1),
            pair_space=self.pair_space, shard_id=0, text_index=0,
            expected_binding=self.binding,
            timing_record=timing or self.timing,
            cad_valid_counts=self.cad_valid,
            path_label="test",
        )

    def _rewrite(self, mutate_lines):
        # each mutation starts from a FRESH valid chunk (mutations are independent)
        path = self.out_dir / chunk_file_name(0, 0, 1)
        if path.exists():
            path.unlink()
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mutate_lines(lines)
        path.unlink()
        atomic_publish(path, [json.dumps(r) for r in lines])

    def test_valid_chunk_passes_with_facts(self):
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        facts = self._verify()
        self.assertEqual(facts, {"n_scored": 4, "n_valid_total": 8})

    def test_missing_chunk_file_refused(self):
        with self.assertRaisesRegex(RunnerError, "does not exist"):
            self._verify()

    def test_empty_chunk_refused(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / chunk_file_name(0, 0, 1)).write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "empty chunk file"):
            self._verify()

    def test_torn_line_refused(self):
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        path = self.out_dir / chunk_file_name(0, 0, 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"record_type": "pair", "pair_in')
        with self.assertRaisesRegex(RunnerError, "torn|complete JSON"):
            self._verify()

    def test_missing_duplicate_nonfinite_rows_refused(self):
        cases = [
            (lambda lines: lines.pop(), "expected 4|pair records"),       # missing row
            (lambda lines: lines.append(dict(lines[-1])), "pair records|pair_index"),  # duplicated row
            (lambda lines: lines[1].update(score_total=float("inf")), "nonfinite"),
            (lambda lines: lines[2].update(status="error"), "scored"),
        ]
        for mutate, pattern in cases:
            with self.subTest(pattern=pattern):
                self._rewrite(mutate)
                with self.assertRaisesRegex(RunnerError, pattern):
                    self._verify()

    def test_binding_and_scope_refused(self):
        # full-matrix chunk (no probe stamp) refuses under the probe scope
        self._write_chunk(self.out_dir, self.pair_space, 0, 1)
        with self.assertRaisesRegex(RunnerError, "scope"):
            self._verify()
        # a foreign probe scope refuses
        path = self.out_dir / chunk_file_name(0, 0, 1)
        path.unlink()
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha="e" * 64)
        with self.assertRaisesRegex(RunnerError, "scope"):
            self._verify()
        # a drifted binding field refuses
        path.unlink()
        bad = self._binding(probe_sha=PROBE_SHA)
        bad["device"] = "npu:9"
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, binding=bad)
        with self.assertRaisesRegex(RunnerError, "binding"):
            self._verify()

    def test_row_score_identity_and_audited_counts_refused(self):
        cases = [
            (lambda lines: lines[1].update(score_name="other_score"), "score identity"),
            # internally consistent row whose counts drift from the AUDITED index
            (lambda lines: lines[1].update(n_valid_x=2, n_valid_total=3),
             "audited worker length index"),
            (lambda lines: lines[3].update(n_real_tokens=9), "audited worker length index"),
        ]
        for mutate, pattern in cases:
            with self.subTest(pattern=pattern):
                self._rewrite(mutate)
                with self.assertRaisesRegex(RunnerError, pattern):
                    self._verify()

    def test_timing_cross_checks_refused(self):
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        with self.assertRaisesRegex(RunnerError, "header counts"):
            self._verify(timing=dict(self.timing, n_scored=3))
        with self.assertRaisesRegex(RunnerError, "valid items"):
            self._verify(timing=dict(self.timing, n_valid_total=7))


# ---------------------------------------------------------------------------
# Runner resume gate (R1): fresh-run sidecar refusal, strict resume validation
# ---------------------------------------------------------------------------


def _mini_probe_selection() -> dict:
    """Hand-built minimal selection dict for the 3x4 mini space (probe={0,2}).

    Only the sections the resume/timing verifiers read are provided; the real
    builder output shape is mirrored.
    """
    return {
        "schema_version": G3_SELECTION_SCHEMA_VERSION,
        "counts": {
            "n_texts": 3, "n_cads": 4, "n_pairs": 12, "n_shards": 2,
            "picks_per_shard": 2, "n_probe_texts": 2, "gallery_per_query": 4,
            "probe_pairs_per_shard": 8, "probe_pairs_total": 16,
        },
        "probes": [
            {"text_index": 0, "shard_id": 0, "shard_pick": 0,
             "n_bert_tokens": 5, "text_layer": 0},
            {"text_index": 2, "shard_id": 0, "shard_pick": 1,
             "n_bert_tokens": 7, "text_layer": 1},
        ],
    }


def _mini_timing_record(text_index: int, *, pick: int, scope, **overrides) -> dict:
    facts = {"text_layer": pick, "n_bert_tokens": 5 + 2 * pick, "shard_pick": pick}
    record = build_query_timing_record(
        invocation_id="inv-mini", shard_id=0, device="cpu", text_index=text_index,
        probe_facts=facts, start_epoch=100.0 + 10 * pick, end_epoch=105.0 + 10 * pick,
        n_pairs=4, n_batches=1, n_scored=4, n_rejected=0, n_valid_total=8,
        chunk_file=chunk_file_name(0, text_index, text_index + 1),
        scope=scope,
    )
    record.update(overrides)
    return record


class G3ResumeGateTest(_ProbeChunkMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        self.pair_space = PairSpace(3, 4)
        self.out_dir = self.root / "out"
        self.out_dir.mkdir()
        self.selection = _mini_probe_selection()
        self.cfg = RunnerConfig(
            checkpoint_path="x", bert_dir="x",
            cad_seq_zip=str(self.root / "cad_seq.zip"),
            pair_manifest_dir=str(self.manifest_dir), device="cpu", batch_size=16,
        )
        self.scope = _scope(device="cpu")

    def _timed(self, **kw):
        return load_probe_sidecar_timed_texts(
            self.out_dir, self.cfg, self.selection, self.scope, **kw
        )

    def _write_sidecar(self, records):
        path = self.out_dir / timing_sidecar_name(0)
        for record in records:
            append_timing_record(path, record)
        return path

    def test_fresh_run_refuses_existing_sidecar(self):
        self._write_sidecar([_mini_timing_record(0, pick=0, scope=self.scope)])
        with self.assertRaisesRegex(RunnerError, "mix timing evidence"):
            self._timed(resume=False)
        # the same fresh run on an empty directory is fine
        self.assertIsNone(
            load_probe_sidecar_timed_texts(
                self.root / "empty", self.cfg, self.selection, self.scope, resume=False
            )
        )

    def test_resume_missing_sidecar_is_empty_set(self):
        self.assertEqual(self._timed(resume=True), set())

    def test_resume_valid_sidecar_returns_timed_set(self):
        self._write_sidecar([_mini_timing_record(0, pick=0, scope=self.scope)])
        self.assertEqual(self._timed(resume=True), {0})

    def test_resume_strict_record_refusals(self):
        good = _mini_timing_record(0, pick=0, scope=self.scope)
        cases = [
            # foreign probe line: not one of this shard's probes
            (dict(good, text_index=1, chunk_file=chunk_file_name(0, 1, 2)),
             "not one of this shard"),
            # drifted selection facts
            (dict(good, text_layer=2), "facts"),
            # invalid window / counts
            (dict(good, n_scored=3), "not complete"),
            (dict(good, n_pairs=3), "complete gallery"),
            (dict(good, e2e_s=99.0), "end-start"),
            (dict(good, chunk_file="chunk_s00_t00001_00002.jsonl"), "chunk_file"),
            (dict(good, **{PROBE_SCOPE_FIELD: "e" * 64}), "scope"),
        ]
        for record, pattern in cases:
            with self.subTest(pattern=pattern):
                self._write_sidecar([record])
                with self.assertRaisesRegex(RunnerError, pattern):
                    self._timed(resume=True)
                (self.out_dir / timing_sidecar_name(0)).unlink()
        # duplicated line refuses
        self._write_sidecar([good, dict(good)])
        with self.assertRaisesRegex(RunnerError, "twice"):
            self._timed(resume=True)
        # torn final line refuses at the reader
        path = self.out_dir / timing_sidecar_name(0)
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"record_type": "query_timing", "text_in')
        with self.assertRaisesRegex(RunnerError, "torn|complete JSON"):
            self._timed(resume=True)

    def test_orphan_and_correspondence_refusals(self):
        # reviewer case: an orphan timing line (no chunk) must refuse the
        # resume, never send the query back into the todo list
        self._write_sidecar([_mini_timing_record(0, pick=0, scope=self.scope)])
        timed = self._timed(resume=True)
        binding = self._binding(probe_sha=PROBE_SHA)
        todo, completed = plan_chunks(
            self.out_dir, self.pair_space, self.cfg, binding, resume=True,
            probe_text_indices={0, 2}, extra_binding_fields=(PROBE_SCOPE_FIELD,),
            timed_text_indices=timed,
        )
        self.assertEqual(completed, [])
        self.assertEqual(todo, [(0, 1, 0), (2, 3, 8)])
        with self.assertRaisesRegex(RunnerError, "no verified chunk"):
            require_sidecar_chunk_correspondence(timed, completed, path_label="resume")
        # with the verified chunk present the correspondence holds and only
        # the untimed probe query stays todo
        self._write_chunk(self.out_dir, self.pair_space, 0, 1, probe_sha=PROBE_SHA)
        todo, completed = plan_chunks(
            self.out_dir, self.pair_space, self.cfg, binding, resume=True,
            probe_text_indices={0, 2}, extra_binding_fields=(PROBE_SCOPE_FIELD,),
            timed_text_indices=timed,
        )
        require_sidecar_chunk_correspondence(timed, completed, path_label="resume")
        self.assertEqual(completed, [chunk_file_name(0, 0, 1)])
        self.assertEqual(todo, [(2, 3, 8)])


# ---------------------------------------------------------------------------
# CPU end-to-end: REAL runner producer (fake model) -> REAL ETA reader
# ---------------------------------------------------------------------------


def _genuine_valid_counts(zip_path: Path, members: list[str]) -> list[dict]:
    """Per-candidate valid-item facts computed from the real zip tensors."""
    counts = []
    with zipfile.ZipFile(zip_path) as archive:
        for cad_index, member in enumerate(members):
            sample = torch.load(
                io.BytesIO(archive.read(member)), map_location="cpu", weights_only=True
            )
            diag = validate_cad_vec(
                sample["vec"]["cad_vec"],
                sample["mask_cad_dict"]["key_padding_mask"],
                sample["vec"]["flag_vec"],
                sample["vec"]["index_vec"],
            )
            counts.append({
                "cad_index": cad_index, "n_real_tokens": diag.n_real_tokens,
                "n_valid_x": diag.n_valid_x, "n_valid_y": diag.n_valid_y,
            })
    return counts


class G3ProducerReaderEndToEndTest(_ProbeChunkMixin, unittest.TestCase):
    """100 texts x 12 CADs synthetic space, two interleaved fake cards on CPU.

    The producer is the REAL runner machinery (FrozenMatrixScorer +
    execute_probe_query + SharedBudgetLedger + build_run_summary over a fake
    tiny model and a real candidate zip); the reader is the REAL ETA
    verification (summary/budget/timing/chunk/final-ledger verifiers). Only
    the model-load report is a schema-mirroring fixture (the real checkpoint
    load is not possible on this CPU host); the candidate preload report is
    the store's REAL output.
    """

    N_TEXTS = 100
    N_CADS = 12
    PAIRS_PER_CARD = 50 * 12
    MAX_PAIRS = 100 * 12

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        root = Path(cls._tmp.name)
        cls.root = root
        text_records = [{"text_index": i} for i in range(cls.N_TEXTS)]
        lengths = {i: (i % 3) + 1 for i in range(cls.N_TEXTS)}
        cls.selection = build_g3_probe_selection(text_records, cls.N_CADS, lengths)
        cls.texts = [f"synthetic query text number {i}" for i in range(cls.N_TEXTS)]
        cls.pair_space = PairSpace(cls.N_TEXTS, cls.N_CADS)
        cad_ids = [f"0000/{i:08d}" for i in range(1, cls.N_CADS + 1)]
        members = [f"cad_seq/{u}/seq/{u.split('/')[-1]}.pth" for u in cad_ids]
        zip_path = root / "cad_seq.zip"
        _make_zip(zip_path, [(m, 19 + 5 * i) for i, m in enumerate(members)])
        cls.zip_path = zip_path
        cls.members = members
        cls.cad_valid = _genuine_valid_counts(zip_path, members)
        cls.ledger_path = root / "ledger.json"
        cls.summaries: dict[int, dict] = {}
        cls.timing_records: dict[int, list[dict]] = {}
        cls.out_dirs: dict[int, Path] = {}
        cls._preloads: dict[int, dict] = {}
        cls._produce()

    @classmethod
    def _produce(cls) -> None:
        root = cls.root
        ledger = SharedBudgetLedger(
            cls.ledger_path, approved_sha=FULL_SHA,
            max_pairs=cls.MAX_PAIRS, max_card_seconds=3600.0,
        )
        authorization = RunAuthorization(
            task_state="pilot_authorized", task_id=TASK_ID, approved_sha=FULL_SHA,
            actual_sha=FULL_SHA, max_pairs=cls.MAX_PAIRS, card_hours=1.0,
            cards_used=1, max_wall_seconds=3600.0, active_task_source="test",
            authorization_ref_sha="e" * 40, active_task_blob_sha256="f" * 64,
            origin_url="test", repo_root="test",
        )
        mixin = _ProbeChunkMixin()
        binding = _ProbeChunkMixin._binding(mixin, probe_sha=PROBE_SHA)
        invocations = {0: "inv-card0", 1: "inv-card1"}
        scorers: dict[int, FrozenMatrixScorer] = {}
        cfgs: dict[int, RunnerConfig] = {}
        probe_facts = {
            shard: probe_facts_for_shard(cls.selection, shard) for shard in (0, 1)
        }
        for shard in (0, 1):
            cfg = RunnerConfig(
                checkpoint_path="x", bert_dir="x", cad_seq_zip=str(cls.zip_path),
                pair_manifest_dir="unused", device="cpu", batch_size=G3_BATCH_SIZE,
                shard_id=shard, n_shards=2, texts_per_chunk=1,
            )
            store = ZipCandidateStore(
                cls.zip_path, cls.members, expected_sha256=sha256_file(cls.zip_path)
            )
            preload = store.preload()
            assert preload["n_rejected"] == 0, preload
            cfgs[shard] = cfg
            cls._preloads[shard] = preload
            ledger.begin_invocation(invocations[shard], shard_id=shard, cards_used=1)
            scorers[shard] = FrozenMatrixScorer(
                cfg, build_tiny_model(seed=shard), cls.texts, store, cls.pair_space,
                ledger, invocations[shard], binding,
            )
            cls.out_dirs[shard] = root / f"card{shard}"
        # interleave the two cards' queries so the measured windows overlap
        queries = {
            shard: probe_texts_for_shard(cls.selection, shard) for shard in (0, 1)
        }
        for k in range(50):
            for shard in (0, 1):
                execute_probe_query(
                    scorers[shard], cls.out_dirs[shard],
                    cls.out_dirs[shard] / timing_sidecar_name(shard),
                    probe_facts[shard], queries[shard][k], queries[shard][k] + 1,
                )
        # first finisher: card 0's summary legitimately shows the peer pending
        ledger.finalize_invocation(invocations[0], exit_code=0)
        cls.summaries[0] = build_run_summary(
            cfgs[0], invocation_id=invocations[0], authorization=authorization,
            published=50, n_previously_completed=0, planned_pairs=cls.PAIRS_PER_CARD,
            stats=scorers[0].stats, load_report=_load_report_fixture(),
            preload_report=cls._preloads[0], ledger=ledger,
            stale_finalized=[], failure=None, probe_selection=cls.selection,
            probe_sha=PROBE_SHA, pair_manifest_sha256=MANIFEST_SHA, cold_start_s=0.5,
        )
        ledger.finalize_invocation(invocations[1], exit_code=0)
        cls.summaries[1] = build_run_summary(
            cfgs[1], invocation_id=invocations[1], authorization=authorization,
            published=50, n_previously_completed=0, planned_pairs=cls.PAIRS_PER_CARD,
            stats=scorers[1].stats, load_report=_load_report_fixture(),
            preload_report=cls._preloads[1], ledger=ledger,
            stale_finalized=[], failure=None, probe_selection=cls.selection,
            probe_sha=PROBE_SHA, pair_manifest_sha256=MANIFEST_SHA, cold_start_s=0.5,
        )
        for shard in (0, 1):
            atomic_publish(
                cls.out_dirs[shard] / f"summary_{shard}.json",
                [json.dumps(cls.summaries[shard], indent=1, sort_keys=True)],
            )
            cls.timing_records[shard] = read_timing_records(
                cls.out_dirs[shard] / timing_sidecar_name(shard)
            )

    def _budget_info(self, shard: int) -> dict:
        return verify_probe_budget_evidence(
            self.summaries[shard], shard_id=shard,
            expected_pairs_per_card=self.PAIRS_PER_CARD,
            expected_max_pairs=self.MAX_PAIRS,
            path_label=f"card{shard} summary",
        )

    def _scope_for(self, shard: int) -> dict:
        return _scope(device="cpu")

    def test_real_produced_artifacts_pass_the_real_reader(self):
        hosts = set()
        expected_invocations: dict[int, dict] = {}
        windows: dict[int, dict] = {}
        for shard in (0, 1):
            info = verify_probe_summary(
                self.summaries[shard], shard_id=shard, device="cpu",
                selection=self.selection,
                expected_pair_manifest_sha256=MANIFEST_SHA,
                expected_max_pairs=self.MAX_PAIRS,
                require_hbm=False, path_label=f"card{shard} summary",
            )
            self.assertEqual(info["git_sha"], FULL_SHA)
            budget = self._budget_info(shard)
            hosts.add(budget["host"])
            ordered = verify_probe_timing_records(
                self.timing_records[shard], shard_id=shard, device="cpu",
                selection=self.selection, expected_scope=self._scope_for(shard),
                expected_invocation_id=budget["invocation_id"],
                invocation_window={
                    "started_epoch": budget["started_epoch"],
                    "ended_epoch": budget["ended_epoch"],
                },
                path_label=f"card{shard} timing",
            )
            self.assertEqual(len(ordered), 50)
            binding = _ProbeChunkMixin._binding(self, probe_sha=PROBE_SHA)
            for record in ordered:
                facts = verify_probe_chunk_content(
                    self.out_dirs[shard] / record["chunk_file"],
                    pair_space=self.pair_space, shard_id=shard,
                    text_index=int(record["text_index"]),
                    expected_binding=binding, timing_record=record,
                    cad_valid_counts=self.cad_valid, path_label=f"card{shard} chunk",
                )
                self.assertEqual(facts["n_scored"], self.N_CADS)
            windows[shard] = {
                "start_epoch": min(r["window"]["start_epoch"] for r in ordered),
                "end_epoch": max(r["window"]["end_epoch"] for r in ordered),
            }
            expected_invocations[shard] = {
                "invocation_id": budget["invocation_id"],
                "started_epoch": budget["started_epoch"],
                "ended_epoch": budget["ended_epoch"],
                "card_seconds": budget["card_seconds"],
                "window_start_epoch": windows[shard]["start_epoch"],
                "window_end_epoch": windows[shard]["end_epoch"],
            }
        self.assertEqual(len(hosts), 1)
        # the interleaved queries actually overlapped in time
        from idea01.text2cad_bench_protocol import window_overlap_fraction

        self.assertGreaterEqual(
            window_overlap_fraction(windows[0], windows[1]), 0.8
        )
        # card 0's snapshot legitimately shows the peer still pending
        peer = next(
            i for i in self.summaries[0]["budget"]["invocations"]
            if i["invocation_id"] == "inv-card1"
        )
        self.assertFalse(peer["finalized"])
        # the FINAL on-disk ledger proves the complete accounting
        final_ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        verify_final_probe_ledger(
            final_ledger, approved_sha=FULL_SHA, host=next(iter(hosts)),
            expected_invocations=expected_invocations,
            expected_pairs_per_card=self.PAIRS_PER_CARD,
            expected_max_card_seconds=3600.0,
            expected_max_pairs=self.MAX_PAIRS,
            path_label="final ledger",
        )

    def test_cpu_summary_still_fails_the_npu_hbm_gate(self):
        # the production gate is not relaxed: a CPU run cannot fabricate HBM
        with self.assertRaisesRegex(RunnerError, "HBM"):
            verify_probe_summary(
                self.summaries[0], shard_id=0, device="cpu",
                selection=self.selection,
                expected_pair_manifest_sha256=MANIFEST_SHA,
                expected_max_pairs=self.MAX_PAIRS,
                require_hbm=True, path_label="card0 summary",
            )

    def test_budget_evidence_binds_authorization_to_the_ledger_cap(self):
        # R2: the REAL produced summary (authorization 1.0 card-hours, ledger
        # cap 3,600 s) passes and reports the bound authorized seconds...
        self.assertEqual(self._budget_info(0)["authorized_card_seconds"], 3600.0)
        self.assertEqual(self._budget_info(1)["authorized_card_seconds"], 3600.0)
        # ...while an authorization that disagrees with the ledger cap it was
        # charged under refuses, even though every other fact is untouched
        import copy

        summary = copy.deepcopy(self.summaries[0])
        summary["authorization"].update(card_hours=0.5)  # 1,800 s != 3,600 s cap
        with self.assertRaisesRegex(RunnerError, "must equal the ledger cap"):
            verify_probe_budget_evidence(
                summary, shard_id=0, expected_pairs_per_card=self.PAIRS_PER_CARD,
                expected_max_pairs=self.MAX_PAIRS, path_label="mutated summary",
            )
        # a missing/non-numeric authorization cannot bind at all
        summary = copy.deepcopy(self.summaries[0])
        summary["authorization"].pop("card_hours")
        with self.assertRaisesRegex(RunnerError, "card_hours"):
            verify_probe_budget_evidence(
                summary, shard_id=0, expected_pairs_per_card=self.PAIRS_PER_CARD,
                expected_max_pairs=self.MAX_PAIRS, path_label="mutated summary",
            )

    def test_reader_refuses_corrupted_produced_artifacts(self):
        shard = 0
        record = self.timing_records[shard][0]
        chunk_path = self.out_dirs[shard] / record["chunk_file"]
        binding = _ProbeChunkMixin._binding(self, probe_sha=PROBE_SHA)

        def verify_chunk(path, timing=None):
            return verify_probe_chunk_content(
                path, pair_space=self.pair_space, shard_id=shard,
                text_index=int(record["text_index"]), expected_binding=binding,
                timing_record=timing or record, cad_valid_counts=self.cad_valid,
                path_label="corrupted",
            )

        # a corrupted copy of the real chunk refuses (non-finite score)
        lines = [
            json.loads(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        lines[1]["score_total"] = float("nan")
        bad = self.root / "bad_chunk.jsonl"
        atomic_publish(bad, [json.dumps(r) for r in lines])
        with self.assertRaisesRegex(RunnerError, "nonfinite"):
            verify_chunk(bad)
        # an empty file refuses
        empty = self.root / "empty_chunk.jsonl"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "empty chunk file"):
            verify_chunk(empty)
        # an extra orphan timing line makes the resume validator refuse
        orphan = build_query_timing_record(
            invocation_id="inv-card0", shard_id=0, device="cpu",
            text_index=int(record["text_index"]),
            probe_facts={"text_layer": record["text_layer"],
                         "n_bert_tokens": record["n_bert_tokens"],
                         "shard_pick": record["shard_pick"]},
            start_epoch=record["window"]["end_epoch"] + 1.0,
            end_epoch=record["window"]["end_epoch"] + 2.0,
            n_pairs=self.N_CADS, n_batches=1, n_scored=self.N_CADS, n_rejected=0,
            n_valid_total=record["n_valid_total"], chunk_file=record["chunk_file"],
            scope=binding,
        )
        with self.assertRaisesRegex(RunnerError, "twice"):
            verify_probe_resume_records(
                [*self.timing_records[shard], orphan], shard_id=shard, device="cpu",
                selection=self.selection, expected_scope=self._scope_for(shard),
                path_label="orphan sidecar",
            )
        # a drifted final ledger refuses (zero total charge)
        final_ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        final_ledger["pairs_charged"] = 0
        expected_invocations = {}
        hosts = set()
        for s in (0, 1):
            budget = self._budget_info(s)
            hosts.add(budget["host"])
            expected_invocations[s] = {
                "invocation_id": budget["invocation_id"],
                "started_epoch": budget["started_epoch"],
                "ended_epoch": budget["ended_epoch"],
                "card_seconds": budget["card_seconds"],
                "window_start_epoch": budget["started_epoch"],
                "window_end_epoch": budget["ended_epoch"],
            }
        with self.assertRaisesRegex(RunnerError, "pairs_charged"):
            verify_final_probe_ledger(
                final_ledger, approved_sha=FULL_SHA, host=next(iter(hosts)),
                expected_invocations=expected_invocations,
                expected_pairs_per_card=self.PAIRS_PER_CARD,
                expected_max_card_seconds=3600.0,
                expected_max_pairs=self.MAX_PAIRS,
                path_label="drifted ledger",
            )


# ---------------------------------------------------------------------------
# Two-card ETA (synthetic official-size selection)
# ---------------------------------------------------------------------------


def _synthetic_selection() -> dict:
    lengths = {i: (i * 37) % 140 + 4 for i in range(EXPECTED_N_UNIQUE_TEXT)}
    text_records = [{"text_index": i} for i in range(EXPECTED_N_UNIQUE_TEXT)]
    return build_g3_probe_selection(text_records, EXPECTED_N_GALLERY, lengths)


def _authorization_fixture(git_sha: str = FULL_SHA, max_pairs: int = G3_APPROVED_MAX_PAIRS,
                           card_hours: float = G3_APPROVED_MAX_CARD_HOURS) -> dict:
    """RunAuthorization.to_dict() shape the runner embeds in every summary."""
    return {
        "task_state": "pilot_authorized",
        "task_id": TASK_ID,
        "approved_sha": git_sha,
        "actual_sha": git_sha,
        "max_pairs": max_pairs,
        "card_hours": card_hours,
        "cards_used": 1,
        "max_wall_seconds": card_hours * 3600.0,
        "active_task_source": "test",
        "authorization_ref_sha": "e" * 40,
        "active_task_blob_sha256": "f" * 64,
        "origin_url": "test",
        "repo_root": "test",
    }


def _load_report_fixture() -> dict:
    """The loader's report shape (measured SHAs; mirrors text2cad_port.loader)."""
    return {
        "checkpoint": {
            "checkpoint": {"path": "ckpt", "sha256": PINNED_CHECKPOINT_SHA256},
            "n_tensors": G3_EXPECTED_NONBERT_TENSORS,
            "n_zero_tensors": 0,
            "zero_tensor_keys": [],
            "dtypes": ["torch.float32"],
        },
        "bert_files": {
            name: {"path": f"bert/{name}", "sha256": sha}
            for name, sha in sorted(PINNED_BERT_FILE_SHA256.items())
        },
        "bert_load": {"word_embeddings_exact_equal": True},
        "strict_load": {
            "n_loaded_tensors": G3_EXPECTED_NONBERT_TENSORS,
            "n_missing_keys": 0,
            "n_unexpected_keys": 0,
            "zero_tensor_keys": [],
            "numeric_equality": True,
            "strict": True,
        },
        "n_nonbert_tensors": G3_EXPECTED_NONBERT_TENSORS,
        "eval_mode": True,
        "all_parameters_frozen": True,
    }


def _preload_fixture(n_cads: int = EXPECTED_N_GALLERY) -> dict:
    return {"n_candidates": n_cads, "n_loaded": n_cads, "n_rejected": 0, "rejection_reasons": []}


def _probe_summary(device: str, *, cold: float, inv: str, host: str = "hostA",
                   ledger: str = "/budget/g3.json", git_sha: str = FULL_SHA,
                   probe_sha: str = PROBE_SHA, manifest_sha: str = MANIFEST_SHA,
                   batch_size: int = G3_BATCH_SIZE, dtype: str = "float32",
                   trim: str = "none", cache: bool = False, cards_used: int = 1,
                   finalized: bool = True, exit_code: int = 0,
                   completed_now: int = 50, completed_before: int = 0,
                   started: float = 500.0, ended: float = 2000.0,
                   status: str = "ok", failure=None, peak: int = 1_900_000_000) -> dict:
    shard = SHARD_BY_DEVICE[device]
    card_seconds = round(ended - started, 2) * cards_used
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "time_utc": "2026-10-02T00:00:00Z", "platform": "test", "python_version": "3.11",
        "torch_version": "2.7.1+cpu", "torch_npu_version": "2.7.1.post1",
        "transformers_version": "4.53.1",
        "device": device, "torch_dtype": dtype, "batch_size": batch_size,
        "trim_mode": trim, "text_cache": cache, "shard_id": shard, "n_shards": 2,
        "texts_per_chunk": 1, "checkpoint_path": "ckpt",
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256, "bert_dir": "bert",
        "bert_file_sha256": dict(PINNED_BERT_FILE_SHA256), "cad_seq_zip": "zip",
        "score_name": SCORE_NAME, "spec_version": SCORE_SPEC_VERSION,
        "max_cad_sequence_length": G3_MAX_CAD_SEQUENCE_LENGTH, "env_vars_present": [],
        "git_sha": git_sha, "pair_manifest_sha256": manifest_sha,
        PROBE_SCOPE_FIELD: probe_sha, "timing_file": timing_sidecar_name(shard),
        "cold_start_s": cold, "peak_device_memory_bytes": peak, "hbm_device": device,
        "probe_queries_total": 50, "invocation_id": inv,
        "authorization": _authorization_fixture(git_sha),
        "chunks_completed_now": completed_now, "chunks_previously_completed": completed_before,
        "pairs_planned": 402300,
        "stats": {"batches": G3_N_BATCHES_PER_QUERY * completed_now,
                  "pairs_forwarded": G3_N_PAIRS_PER_QUERY * completed_now,
                  "pairs_rejected": 0},
        "load_report": _load_report_fixture(),
        "candidate_preload": _preload_fixture(),
        "budget": {
            "budget_file": ledger, "approved_sha": git_sha,
            "max_pairs": G3_APPROVED_MAX_PAIRS,
            "max_card_seconds": G3_APPROVED_MAX_CARD_SECONDS,
            "pairs_charged": G3_N_PAIRS_PER_QUERY * completed_now,
            "remaining_pairs": G3_APPROVED_MAX_PAIRS - G3_N_PAIRS_PER_QUERY * completed_now,
            "card_seconds_spent": card_seconds if finalized else 0.0,
            "invocations": [{
                "invocation_id": inv, "shard_id": shard, "cards_used": cards_used,
                "pid": 1234, "host": host, "started_epoch": started,
                "last_heartbeat_epoch": ended - 1.0, "finalized": finalized,
                "ended_epoch": ended, "card_seconds": card_seconds, "exit_code": exit_code,
                "pairs_charged": G3_N_PAIRS_PER_QUERY * completed_now,
            }],
        },
        "stale_invocations_auto_finalized": [], "status": status, "failure": failure,
    }


def _final_ledger(summaries: list[dict], **overrides) -> dict:
    """The FINAL shared ledger consistent with the cards' own invocations."""
    invs = []
    spent = 0.0
    for summary in summaries:
        own_id = summary["invocation_id"]
        own = next(
            i for i in summary["budget"]["invocations"] if i["invocation_id"] == own_id
        )
        invs.append(dict(own))
        spent += float(own.get("card_seconds") or 0.0)
    ledger = {
        "version": 1,
        "approved_sha": summaries[0]["budget"]["approved_sha"],
        "max_pairs": summaries[0]["budget"]["max_pairs"],
        "max_card_seconds": summaries[0]["budget"]["max_card_seconds"],
        "pairs_charged": sum(int(i.get("pairs_charged") or 0) for i in invs),
        "card_seconds_spent": round(spent, 2),
        "heartbeat_interval_s": 30.0,
        "invocations": invs,
    }
    ledger.update(overrides)
    return ledger


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

    def _estimate(self, evidence, final_ledger=None, **kw):
        if final_ledger is None:
            final_ledger = _final_ledger(
                [evidence[d]["summary"] for d in ("npu:0", "npu:1") if d in evidence]
                or [list(evidence.values())[0]["summary"]]
            )
        return estimate_g3_probe_eta(
            self.selection,
            evidence,
            expected_probe_selection_sha256=PROBE_SHA,
            expected_pair_manifest_sha256=MANIFEST_SHA,
            final_ledger=final_ledger,
            **kw,
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
            for record in e["npu:1"]["timing_records"]:
                record.update(invocation_id="inv-card0")

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
            e["npu:1"]["summary"]["authorization"].update(approved_sha="f" * 40,
                                                          actual_sha="f" * 40)
            e["npu:1"]["summary"]["budget"].update(approved_sha="f" * 40)

        refused(drift_git_sha, "git_sha|scope|disagree")
        refused(lambda e: e["npu:1"]["summary"].update(git_sha="short"), "git_sha")
        refused(lambda e: e["npu:0"]["summary"].update(authorization={"approved_sha": "0" * 40}),
                "authorization")
        refused(lambda e: e["npu:1"]["summary"].update(pair_manifest_sha256="0" * 64),
                "reviewed fingerprint")
        # chunk accounting vs timed queries
        refused(lambda e: e["npu:0"]["summary"].update(chunks_completed_now=49,
                                                       chunks_previously_completed=0),
                "chunks_completed_now")
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

    # -- R1: the final shared ledger is mandatory evidence -------------------

    def test_final_ledger_required(self):
        evidence = _evidence(self.selection)
        with self.assertRaisesRegex(RunnerError, "FINAL shared budget ledger"):
            estimate_g3_probe_eta(
                self.selection, evidence,
                expected_probe_selection_sha256=PROBE_SHA,
                expected_pair_manifest_sha256=MANIFEST_SHA,
                final_ledger=None,
            )
        with self.assertRaisesRegex(RunnerError, "FINAL shared budget ledger"):
            self._estimate(evidence, final_ledger={})

    # -- R1: real accounting, not just success flags (Root witness cases) ----

    def test_zero_stats_planned_charged_refused(self):
        base = _evidence(self.selection)

        def refused(mutate, pattern, ledger_overrides=None):
            import copy

            evidence = copy.deepcopy(base)
            mutate(evidence)
            final = None
            if ledger_overrides is not None:
                final = _final_ledger(
                    [evidence["npu:0"]["summary"], evidence["npu:1"]["summary"]],
                    **ledger_overrides,
                )
            with self.assertRaisesRegex(RunnerError, pattern):
                self._estimate(evidence, final_ledger=final)

        refused(lambda e: e["npu:0"]["summary"]["stats"].update(pairs_forwarded=0),
                "pairs_forwarded")
        refused(lambda e: e["npu:0"]["summary"]["stats"].update(batches=0), "stats.batches")
        refused(lambda e: e["npu:0"]["summary"].update(pairs_planned=0), "pairs_planned")
        refused(lambda e: e["npu:0"]["summary"]["budget"].update(pairs_charged=0),
                "pairs_charged")
        refused(lambda e: e["npu:0"]["summary"]["budget"]["invocations"][0].update(
            pairs_charged=0), "pairs_charged")
        refused(lambda e: None, "pairs_charged", ledger_overrides={"pairs_charged": 0})

    def test_over_card_budget_refused(self):
        base = _evidence(self.selection)

        # Root case: spent 99999 vs cap 1 (and inconsistent accounting); the
        # authorization follows the cap so the refusal is the spend itself
        import copy

        evidence = copy.deepcopy(base)
        evidence["npu:0"]["summary"]["budget"].update(
            card_seconds_spent=99999.0, max_card_seconds=1.0
        )
        evidence["npu:0"]["summary"]["authorization"].update(card_hours=1.0 / 3600.0)
        with self.assertRaisesRegex(RunnerError, "card_seconds_spent|exceeds"):
            self._estimate(evidence)
        # consistent but over the approved cap: cap below the spent seconds
        evidence = copy.deepcopy(base)
        evidence["npu:0"]["summary"]["budget"].update(max_card_seconds=1000.0)
        evidence["npu:0"]["summary"]["authorization"].update(card_hours=1000.0 / 3600.0)
        with self.assertRaisesRegex(RunnerError, "exceeds the approved cap"):
            self._estimate(evidence)
        # a cap beyond the approved 12 card-hours refuses even with room
        evidence = copy.deepcopy(base)
        evidence["npu:0"]["summary"]["budget"].update(max_card_seconds=99999.0)
        with self.assertRaisesRegex(RunnerError, "at most"):
            self._estimate(evidence)
        # the final ledger itself over budget
        final = _final_ledger(
            [base["npu:0"]["summary"], base["npu:1"]["summary"]],
            card_seconds_spent=99999.0,
        )
        with self.assertRaisesRegex(RunnerError, "card_seconds_spent"):
            self._estimate(base, final_ledger=final)
        # R2 reviewer case: a final cap of 3,001 s against the two 43,200 s
        # snapshots is contradictory evidence and refuses at the binding
        final = _final_ledger(
            [base["npu:0"]["summary"], base["npu:1"]["summary"]], max_card_seconds=3001.0
        )
        with self.assertRaisesRegex(RunnerError, "SAME approved"):
            self._estimate(base, final_ledger=final)

    def test_timing_invocation_binding(self):
        base = _evidence(self.selection)
        import copy

        # Root case: timing lines naming an invocation that is not the ledger's
        evidence = copy.deepcopy(base)
        for record in evidence["npu:0"]["timing_records"]:
            record.update(invocation_id="not-in-ledger")
        with self.assertRaisesRegex(RunnerError, "charged run"):
            self._estimate(evidence)
        # Root case: ledger epochs 1-3 vs query windows around epoch 1000
        evidence = copy.deepcopy(base)
        own = evidence["npu:0"]["summary"]["budget"]["invocations"][0]
        own.update(started_epoch=1.0, ended_epoch=3.0, card_seconds=2.0,
                   last_heartbeat_epoch=2.0)
        evidence["npu:0"]["summary"]["budget"].update(card_seconds_spent=2.0)
        evidence["npu:0"]["summary"].update(cold_start_s=1.0)
        with self.assertRaisesRegex(RunnerError, "outside the ledger invocation"):
            self._estimate(evidence)
        # cold start beyond the invocation interval contradicts the timeline
        evidence = copy.deepcopy(base)
        evidence["npu:0"]["summary"].update(cold_start_s=99999.0)
        with self.assertRaisesRegex(RunnerError, "timeline disagrees"):
            self._estimate(evidence)

    def test_pending_peer_snapshot_accepted(self):
        import copy

        base = _evidence(self.selection)
        # first finisher: card 0's snapshot shows the peer legitimately pending
        evidence = copy.deepcopy(base)
        evidence["npu:0"]["summary"]["budget"]["invocations"].append({
            "invocation_id": "inv-card1", "shard_id": 1, "cards_used": 1,
            "pid": 4321, "host": "hostA", "started_epoch": 510.0,
            "last_heartbeat_epoch": 1990.0, "finalized": False,
            "pairs_charged": 402300,
        })
        evidence["npu:0"]["summary"]["budget"].update(pairs_charged=804600,
                                                      remaining_pairs=0)
        estimate = self._estimate(evidence)
        self.assertTrue(estimate["passes_seven_day_gate"])
        # second finisher: the peer appears finalized-successful in the snapshot
        evidence = copy.deepcopy(base)
        peer_done = dict(evidence["npu:1"]["summary"]["budget"]["invocations"][0])
        evidence["npu:0"]["summary"]["budget"]["invocations"].append(peer_done)
        evidence["npu:0"]["summary"]["budget"].update(
            pairs_charged=804600, remaining_pairs=0, card_seconds_spent=3000.0
        )
        estimate = self._estimate(evidence)
        self.assertTrue(estimate["passes_seven_day_gate"])

    def test_extra_failed_old_invocations_refused(self):
        import copy

        base = _evidence(self.selection)

        def with_extra(inv):
            evidence = copy.deepcopy(base)
            evidence["npu:0"]["summary"]["budget"]["invocations"].append(inv)
            return evidence

        # a failed peer attempt in the snapshot refuses
        failed = {
            "invocation_id": "inv-old", "shard_id": 1, "cards_used": 1, "pid": 99,
            "host": "hostA", "started_epoch": 100.0, "last_heartbeat_epoch": 150.0,
            "finalized": True, "ended_epoch": 150.0, "card_seconds": 50.0,
            "exit_code": 2, "pairs_charged": 16092,
        }
        with self.assertRaisesRegex(RunnerError, "failed"):
            self._estimate(with_extra(failed))
        # a second invocation on THIS shard refuses (double-run evidence)
        same_shard = dict(failed, invocation_id="inv-dup", shard_id=0, exit_code=0)
        with self.assertRaisesRegex(RunnerError, "double-run|shard"):
            self._estimate(with_extra(same_shard))
        # a third invocation (even a plausible peer) refuses the fresh round
        evidence = copy.deepcopy(base)
        pending_peer = {
            "invocation_id": "inv-card1", "shard_id": 1, "cards_used": 1, "pid": 5,
            "host": "hostA", "started_epoch": 510.0, "last_heartbeat_epoch": 600.0,
            "finalized": False, "pairs_charged": 16092,
        }
        third = dict(failed, invocation_id="inv-third", exit_code=0)
        evidence["npu:0"]["summary"]["budget"]["invocations"].extend([pending_peer, third])
        with self.assertRaisesRegex(RunnerError, "invocations in the snapshot"):
            self._estimate(evidence)
        # the final ledger with an old/extra invocation refuses
        final = _final_ledger([base["npu:0"]["summary"], base["npu:1"]["summary"]])
        final["invocations"].append(dict(failed))
        with self.assertRaisesRegex(RunnerError, "old/extra/failed"):
            self._estimate(base, final_ledger=final)
        # the final ledger missing one card refuses
        final = _final_ledger([base["npu:0"]["summary"], base["npu:1"]["summary"]])
        final["invocations"] = final["invocations"][:1]
        with self.assertRaisesRegex(RunnerError, "invocation"):
            self._estimate(base, final_ledger=final)
        # the final ledger recording a failed card refuses
        final = _final_ledger([base["npu:0"]["summary"], base["npu:1"]["summary"]])
        final["invocations"][1].update(exit_code=1)
        with self.assertRaisesRegex(RunnerError, "finalized success"):
            self._estimate(base, final_ledger=final)

    def test_final_ledger_snapshot_disagreement_refused(self):
        import copy

        base = _evidence(self.selection)
        stale = _final_ledger([base["npu:0"]["summary"], base["npu:1"]["summary"]])
        evidence = copy.deepcopy(base)
        own = evidence["npu:0"]["summary"]["budget"]["invocations"][0]
        own.update(started_epoch=400.0, card_seconds=1600.0)
        evidence["npu:0"]["summary"]["budget"].update(card_seconds_spent=1600.0)
        with self.assertRaisesRegex(RunnerError, "disagree"):
            self._estimate(evidence, final_ledger=stale)
        # a final ledger bound to another approval refuses
        stale = _final_ledger([base["npu:0"]["summary"], base["npu:1"]["summary"]],
                              approved_sha="0" * 40)
        with self.assertRaisesRegex(RunnerError, "bound to"):
            self._estimate(base, final_ledger=stale)
        # a final ledger with a drifted pair cap refuses
        stale = _final_ledger([base["npu:0"]["summary"], base["npu:1"]["summary"]],
                              max_pairs=805000)
        with self.assertRaisesRegex(RunnerError, "max_pairs"):
            self._estimate(base, final_ledger=stale)

    def test_resume_shaped_evidence_refused(self):
        import copy

        evidence = _evidence(self.selection)
        evidence["npu:0"]["summary"].update(chunks_previously_completed=10)
        with self.assertRaisesRegex(RunnerError, "chunks_previously_completed"):
            self._estimate(evidence)

    # -- R1: fixed-asset binding, never mutual agreement ---------------------

    def test_fixed_asset_binding_refused(self):
        base = _evidence(self.selection)

        def refused(mutate, pattern):
            import copy

            evidence = copy.deepcopy(base)
            mutate(evidence)
            with self.assertRaisesRegex(RunnerError, pattern):
                self._estimate(evidence)

        # reviewer case: BOTH cards agree on the wrong manifest SHA -- refuses
        def zero_manifest(e):
            for device in ("npu:0", "npu:1"):
                e[device]["summary"].update(pair_manifest_sha256="0" * 64)

        refused(zero_manifest, "reviewed fingerprint")
        # reviewer case: BERT fingerprints zeroed consistently
        def zero_bert(e):
            for device in ("npu:0", "npu:1"):
                e[device]["summary"].update(bert_file_sha256={
                    name: "0" * 64 for name in PINNED_BERT_FILE_SHA256
                })

        refused(zero_bert, "five pinned")
        # reviewer case: the CAD length shrunk to 128 consistently
        def short_cad(e):
            for device in ("npu:0", "npu:1"):
                e[device]["summary"].update(max_cad_sequence_length=128)

        refused(short_cad, "272-row|full")
        refused(lambda e: e["npu:0"]["summary"].update(checkpoint_sha256="0" * 64),
                "pinned checkpoint")
        # measured load/preload evidence, not claimed constants
        refused(lambda e: e["npu:0"]["summary"].update(load_report={}), "load_report")
        refused(lambda e: e["npu:0"]["summary"]["load_report"]["bert_files"].update(
            config_json={"path": "x", "sha256": "0" * 64}), "BERT")
        refused(lambda e: e["npu:0"]["summary"]["load_report"]["strict_load"].update(
            n_loaded_tensors=171), "n_loaded_tensors")
        refused(lambda e: e["npu:0"]["summary"]["load_report"]["strict_load"].update(
            numeric_equality=False), "numeric equality")
        refused(lambda e: e["npu:0"]["summary"].update(candidate_preload={
            "n_candidates": 8046, "n_loaded": 8045, "n_rejected": 1,
            "rejection_reasons": ["bad_shape"]}), "n_rejected|n_loaded")
        # the approval itself must be the exact G3 pilot authorization
        refused(lambda e: e["npu:0"]["summary"]["authorization"].update(max_pairs=805000),
                "max_pairs")
        refused(lambda e: e["npu:0"]["summary"]["authorization"].update(card_hours=13.0),
                "card-hours")
        refused(lambda e: e["npu:0"]["summary"]["authorization"].update(
            task_state="implementation"), "pilot_authorized")
        refused(lambda e: e["npu:0"]["summary"]["authorization"].update(actual_sha="0" * 40),
                "approved commit")

    # -- R2: card-time authorization bound to the ledger caps ----------------

    def test_authorization_snapshot_cap_binding(self):
        import copy

        base = _evidence(self.selection)
        # R2 reviewer witness: BOTH authorizations shrunk to 0.1 card-hours
        # (360 s) while the snapshots keep the 43,200 s cap and the real
        # 3,000 s spend -- contradictory evidence, must refuse now
        evidence = copy.deepcopy(base)
        for device in ("npu:0", "npu:1"):
            evidence[device]["summary"]["authorization"].update(card_hours=0.1)
        with self.assertRaisesRegex(RunnerError, "must equal the ledger cap"):
            self._estimate(evidence)
        # one card's authorization alone disagreeing with its snapshot refuses
        evidence = copy.deepcopy(base)
        evidence["npu:0"]["summary"]["authorization"].update(card_hours=11.0)
        with self.assertRaisesRegex(RunnerError, "must equal the ledger cap"):
            self._estimate(evidence)
        # the two cards authorized differently (each consistently charged
        # under its own snapshot cap) refuse the shared-budget evidence
        evidence = copy.deepcopy(base)
        evidence["npu:1"]["summary"]["authorization"].update(card_hours=11.0)
        evidence["npu:1"]["summary"]["budget"].update(max_card_seconds=39600.0)
        with self.assertRaisesRegex(RunnerError, "different card-second caps"):
            self._estimate(evidence)
        # the final ledger cap disagreeing with the snapshots refuses, both
        # below and above the snapshot cap (3,001 s is the reviewer case)
        for bad_cap in (3001.0, 43199.0):
            final = _final_ledger(
                [base["npu:0"]["summary"], base["npu:1"]["summary"]],
                max_card_seconds=bad_cap,
            )
            with self.assertRaisesRegex(RunnerError, "SAME approved"):
                self._estimate(base, final_ledger=final)
        # the unmutated evidence remains legal under the binding
        self.assertTrue(self._estimate(copy.deepcopy(base))["passes_seven_day_gate"])

    # -- R2: same-card serial query timeline in production order --------------

    def test_same_card_serial_timeline_required(self):
        import copy

        base = _evidence(self.selection)
        # R2 reviewer witness: all 50 query starts identical, e2e preserved
        evidence = copy.deepcopy(base)
        for record in evidence["npu:0"]["timing_records"]:
            e2e = record["e2e_s"]
            record["window"] = {"start_epoch": 1000.0, "end_epoch": 1000.0 + e2e}
        with self.assertRaisesRegex(RunnerError, "serial"):
            self._estimate(evidence)
        # an adjacent overlap refuses (record 1 starts inside record 0)
        evidence = copy.deepcopy(base)
        records = evidence["npu:0"]["timing_records"]
        start = records[0]["window"]["start_epoch"] + 1.0
        e2e = records[1]["e2e_s"]
        records[1]["window"] = {"start_epoch": start, "end_epoch": start + e2e}
        with self.assertRaisesRegex(RunnerError, "serial"):
            self._estimate(evidence)
        # reversed windows refuse: production order is ascending text_index,
        # never re-sorted by time (window+e2e swapped together to keep every
        # record internally consistent)
        evidence = copy.deepcopy(base)
        records = evidence["npu:0"]["timing_records"]
        windows = [dict(r["window"]) for r in records]
        for record, window in zip(records, reversed(windows)):
            record["window"] = window
            record["e2e_s"] = window["end_epoch"] - window["start_epoch"]
        with self.assertRaisesRegex(RunnerError, "serial"):
            self._estimate(evidence)
        # a legal back-to-back timeline (next start == previous end) passes
        evidence = _evidence(self.selection, step=8.0,
                             e2e0=lambda facts, k: 8.0, e2e1=lambda facts, k: 8.0)
        estimate = self._estimate(evidence)
        self.assertTrue(estimate["passes_seven_day_gate"])
        # a legal large-duration timeline passes too: the spacing is serial
        # with gaps and the ledger invocation interval / card-seconds are
        # rebuilt to match (both cards: 20,900 s each, 41,800 s <= the
        # 43,200 s shared cap); the ETA math itself is unchanged
        evidence = _evidence(
            self.selection, step=400.0, started=500.0, ended=21400.0,
            started__1=500.0, ended__1=21400.0,
            e2e0=lambda facts, k: 300.0, e2e1=lambda facts, k: 300.0,
        )
        estimate = self._estimate(evidence)
        for device in ("npu:0", "npu:1"):
            self.assertAlmostEqual(
                estimate["per_card"][device]["compute_seconds"], 300.0 * 1822
            )


# ---------------------------------------------------------------------------
# eta-g3 CLI end-to-end against the real committed selection
# ---------------------------------------------------------------------------


@unittest.skipUnless(REAL_ASSETS_PRESENT, "real frozen worker artifacts required")
class G3EtaCliRealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._cls_tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._cls_tmp.cleanup)
        cls.root = Path(cls._cls_tmp.name)
        cls.selection = json.loads(G3_CONFIG.read_text(encoding="utf-8"))
        cls.probe_sha = sha256_file(G3_CONFIG)
        cls.manifest_sha = "62086beb064af7b738665825edbf1a3cb63ebfaceddc7e227c0aa45390902557"
        cls.cad_valid = load_cad_length_index(WORKER_DIR)
        cls.total_valid = sum(r["n_valid_x"] + r["n_valid_y"] for r in cls.cad_valid)
        # REAL 8,046-row chunks for both cards (the empty-file fixture of the
        # pre-R1 test is gone: chunks carry the audited per-CAD valid counts)
        cls.card0 = cls._build_card("npu:0", base=1000.0, cold=10.0, slow=8.0)
        cls.card1 = cls._build_card("npu:1", base=1000.0, cold=20.0, slow=6.0)
        cls.final_ledger = cls.root / "final_ledger.json"
        cls._write_final_ledger(cls.card0 / "summary_0.json", cls.card1 / "summary_1.json",
                                cls.final_ledger)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)

    @classmethod
    def _chunk_binding(cls, device: str) -> dict:
        return {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "pair_manifest_sha256": cls.manifest_sha,
            "git_sha": FULL_SHA,
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "device": device,
            "torch_dtype": "float32",
            "trim_mode": "none",
            "text_cache": False,
            "score_name": SCORE_NAME,
            "spec_version": SCORE_SPEC_VERSION,
            PROBE_SCOPE_FIELD: cls.probe_sha,
        }

    @classmethod
    def _write_real_chunk(cls, card_dir: Path, shard: int, text_index: int,
                          binding: dict) -> None:
        header = {
            "record_type": "chunk_header", "shard_id": shard,
            "text_range": [text_index, text_index + 1],
            "n_pairs": EXPECTED_N_GALLERY, "n_scored": EXPECTED_N_GALLERY,
            "n_rejected": 0,
            "pair_index_first": text_index * EXPECTED_N_GALLERY,
            "pair_index_last": (text_index + 1) * EXPECTED_N_GALLERY - 1,
            **binding,
        }
        path = card_dir / chunk_file_name(shard, text_index, text_index + 1)
        lines = [json.dumps(header, ensure_ascii=False)]
        for cad_index, valid in enumerate(cls.cad_valid):
            lines.append(json.dumps({
                "record_type": "pair",
                "pair_index": text_index * EXPECTED_N_GALLERY + cad_index,
                "text_index": text_index, "cad_index": cad_index,
                "status": "scored", "reason": None,
                "score_name": SCORE_NAME, "spec_version": SCORE_SPEC_VERSION,
                "score_x": -1.0, "score_y": -2.0, "score_total": -3.0,
                "mean_x": -1.0, "mean_y": -2.0, "mean_total": -1.5,
                "n_valid_x": valid["n_valid_x"], "n_valid_y": valid["n_valid_y"],
                "n_valid_total": valid["n_valid_x"] + valid["n_valid_y"],
                "valid_seq_len": valid["n_real_tokens"],
                "n_real_tokens": valid["n_real_tokens"],
            }, ensure_ascii=False, separators=(",", ":")))
        atomic_publish(path, lines)

    @classmethod
    def _build_card(cls, device: str, *, base: float, cold: float, slow: float,
                    name: str = None) -> Path:
        shard = SHARD_BY_DEVICE[device]
        card_dir = cls.root / (name or f"card{shard}")
        card_dir.mkdir(exist_ok=True)
        binding = cls._chunk_binding(device)
        facts = probe_facts_for_shard(cls.selection, shard)
        for k, text_index in enumerate(sorted(facts)):
            e2e = slow if k == 0 else 1.0
            start = base + k * 10.0
            record = _timing_record(
                text_index, shard_id=shard, start=start, e2e=e2e,
                inv=f"inv-card{shard}", device=device,
                layer=facts[text_index]["text_layer"],
                tokens=facts[text_index]["n_bert_tokens"],
                pick=facts[text_index]["shard_pick"],
                scope=_scope(device=device, pair_manifest_sha256=cls.manifest_sha),
                n_valid=cls.total_valid,
            )
            record[PROBE_SCOPE_FIELD] = cls.probe_sha
            append_timing_record(card_dir / timing_sidecar_name(shard), record)
            cls._write_real_chunk(card_dir, shard, text_index, binding)
        summary = _probe_summary(
            device, cold=cold, inv=f"inv-card{shard}", probe_sha=cls.probe_sha,
            manifest_sha=cls.manifest_sha,
        )
        (card_dir / f"summary_{shard}.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return card_dir

    @classmethod
    def _write_final_ledger(cls, s0: Path, s1: Path, out: Path) -> None:
        summaries = [json.loads(p.read_text(encoding="utf-8")) for p in (s0, s1)]
        out.write_text(json.dumps(_final_ledger(summaries)), encoding="utf-8")

    def _argv(self, s0: Path, s1: Path, out: Path, ledger: Path = None) -> list[str]:
        argv = [
            "--mode", "eta-g3",
            "--pair-manifest-dir", str(WORKER_DIR),
            "--token-audit", str(TOKEN_AUDIT),
            "--repo-root", str(REPO_ROOT),
            "--card-summary", str(s0),
            "--card-summary", str(s1),
            "--out", str(out),
        ]
        if ledger is not None:
            argv += ["--final-ledger", str(ledger)]
        return argv

    def _clone_card(self, src: Path, shard: int, name: str) -> Path:
        """Hard-link the big chunks into a fresh dir; COPY summary/sidecar so
        mutations of the clone never touch the shared class fixture."""
        dst = self.work / name
        dst.mkdir()
        for entry in src.iterdir():
            target = dst / entry.name
            if entry.name.startswith("chunk_"):
                try:
                    os.link(entry, target)
                except OSError:
                    shutil.copyfile(entry, target)
            else:
                shutil.copyfile(entry, target)
        return dst

    def test_happy_path_hand_computed(self):
        out = self.work / "eta.json"
        rc = g3_main(self._argv(self.card0 / "summary_0.json", self.card1 / "summary_1.json",
                                out, ledger=self.final_ledger))
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
        self.assertEqual(estimate["inputs"]["final_ledger"], str(self.final_ledger.resolve()))
        # refuses to overwrite the estimate
        self.assertEqual(
            g3_main(self._argv(self.card0 / "summary_0.json", self.card1 / "summary_1.json",
                               out, ledger=self.final_ledger)),
            2,
        )

    def test_cli_requires_the_final_ledger(self):
        self.assertEqual(
            g3_main(self._argv(self.card0 / "summary_0.json", self.card1 / "summary_1.json",
                               self.work / "x.json")),
            2,
        )
        self.assertFalse((self.work / "x.json").exists())

    def test_cli_refusals(self):
        # staggered scoring windows (sidecar shifted +400s) refuse the
        # concurrency gate even though everything else is consistent
        d1 = self._clone_card(self.card1, 1, "card1_staggered")
        records = read_timing_records(d1 / timing_sidecar_name(1))
        (d1 / timing_sidecar_name(1)).unlink()
        for record in records:
            start = record["window"]["start_epoch"] + 400.0
            e2e = record["e2e_s"]
            record["window"] = {"start_epoch": start, "end_epoch": start + e2e}
            append_timing_record(d1 / timing_sidecar_name(1), record)
        summary = json.loads((d1 / "summary_1.json").read_text(encoding="utf-8"))
        # widen the invocation interval so the shifted windows stay inside it
        own = summary["budget"]["invocations"][0]
        own.update(ended_epoch=2600.0, card_seconds=2100.0)
        summary["budget"].update(card_seconds_spent=2100.0)
        (d1 / "summary_1.json").write_text(json.dumps(summary), encoding="utf-8")
        ledger = self.work / "ledger_staggered.json"
        self._write_final_ledger(self.card0 / "summary_0.json", d1 / "summary_1.json", ledger)
        self.assertEqual(
            g3_main(self._argv(self.card0 / "summary_0.json", d1 / "summary_1.json",
                               self.work / "x.json", ledger=ledger)),
            2,
        )
        # a missing chunk file behind a timing line refuses
        d1b = self._clone_card(self.card1, 1, "card1_missing")
        first_text = sorted(probe_facts_for_shard(self.selection, 1))[0]
        (d1b / chunk_file_name(1, first_text, first_text + 1)).unlink()
        self.assertEqual(
            g3_main(self._argv(self.card0 / "summary_0.json", d1b / "summary_1.json",
                               self.work / "y.json", ledger=self.final_ledger)),
            2,
        )
        # a zero-byte chunk (the pre-R1 fixture!) now refuses
        d1c = self._clone_card(self.card1, 1, "card1_empty")
        (d1c / chunk_file_name(1, first_text, first_text + 1)).unlink()
        (d1c / chunk_file_name(1, first_text, first_text + 1)).write_text("", encoding="utf-8")
        self.assertEqual(
            g3_main(self._argv(self.card0 / "summary_0.json", d1c / "summary_1.json",
                               self.work / "z.json", ledger=self.final_ledger)),
            2,
        )
        # an extra foreign chunk without a timing line refuses (one-to-one)
        d0 = self._clone_card(self.card0, 0, "card0_extra")
        foreign = d0 / chunk_file_name(0, 9999, 10000)
        foreign.write_text("{}\n", encoding="utf-8")
        self.assertEqual(
            g3_main(self._argv(d0 / "summary_0.json", self.card1 / "summary_1.json",
                               self.work / "w.json", ledger=self.final_ledger)),
            2,
        )
        # one summary only
        self.assertEqual(
            g3_main([
                "--mode", "eta-g3", "--pair-manifest-dir", str(WORKER_DIR),
                "--token-audit", str(TOKEN_AUDIT), "--repo-root", str(REPO_ROOT),
                "--card-summary", str(self.card0 / "summary_0.json"),
                "--final-ledger", str(self.final_ledger),
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
