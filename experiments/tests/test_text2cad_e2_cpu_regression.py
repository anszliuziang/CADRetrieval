"""REAL-ledger, REAL-gate, REAL-model-load CPU regression for E2 (review
round 4 final demand): everything is the reviewed production code — the ONLY
replacement is ``idea01.text2cad_frozen_runner.forward_and_score_batch``, the
model forward itself. Covers the full mandatory order
(smoke -> REPLAY -> tolerance-gate record -> b0 -> b1) against the REAL
SharedBudgetLedger with the exact pair arithmetic, plus the timeout-failure
paths (past-deadline refusal and a deadline firing mid-run).

Requires torch + the local pinned assets + the local G4 raw matrix; skips
cleanly when any of these is missing. Slow (~10 min): four real model loads,
real candidate preload, real CPU scoring of all 32,461 pairs.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO.parent / "CADRetrievalIdea1_runtime"

sys.path.insert(0, str(REPO / "experiments" / "src"))

from idea01 import text2cad_e2_runner as e2  # noqa: E402

MANIFESTS = RUNTIME / "text2cad_e2_manifests_20261112"
G4_RAW = REPO / "experiments" / "runs" / "idea01_text2cad_g4_full_20261002"
LOCAL_V4 = RUNTIME / "pair_manifest_v4"
LOCAL_T2C = REPO / ".cache" / "text2cad_official"
LOCAL_BERT = REPO / ".cache" / "bert_large_uncased"


def _assets_present() -> bool:
    need = [
        MANIFESTS / "b0" / "texts.jsonl",
        MANIFESTS / "smoke_b0" / "texts.jsonl",
        MANIFESTS / "replay" / "texts.jsonl",
        G4_RAW / "SHA256SUMS",
        LOCAL_V4 / "cad_store.jsonl",
        LOCAL_T2C / "text2cad_v1.0" / "Text2CAD_1.0.pth",
        LOCAL_T2C / "cad_seq.zip",
        LOCAL_BERT / "model.safetensors",
    ]
    return all(p.exists() for p in need)


try:
    import torch  # noqa: F401

    TORCH_OK = True
except Exception:  # noqa: BLE001
    TORCH_OK = False


@unittest.skipUnless(TORCH_OK and _assets_present(), "needs torch + local pinned assets + G4 raw matrix")
class TestE2CpuRegression(unittest.TestCase):
    """Full-order run with the real ledger/gates/loader; forward stubbed to
    mimic the certified G4 scores so the REPLAY tolerance gate is meaningful."""

    RUN_ROOT = RUNTIME / "runs" / "idea01_text2cad_e2_cpu_regression_20261112"
    ENV_ROOT = RUNTIME / "e2_regression_env_20261112"
    LEDGER_ROOT = RUNTIME / "e2_regression_ledgers_20261112"
    CFG = json.loads((REPO / "experiments" / "configs" / "text2cad_e2_background_expected_20261112.json")
                     .read_text(encoding="utf-8"))

    @classmethod
    def setUpClass(cls):
        import idea01.text2cad_frozen_runner as fr

        # real module, real ledger class; only the ledger ROOT and the forward
        # are redirected/ replaced (restored in tearDownClass)
        cls.fr = fr
        cls._original_forward = fr.forward_and_score_batch
        cls._original_budget_root_value = fr.PRODUCTION_BUDGET_ROOT
        if cls.ENV_ROOT.exists():
            shutil.rmtree(cls.ENV_ROOT)
        if cls.LEDGER_ROOT.exists():
            shutil.rmtree(cls.LEDGER_ROOT)
        if cls.RUN_ROOT.exists():
            shutil.rmtree(cls.RUN_ROOT)
        fr.PRODUCTION_BUDGET_ROOT = cls.LEDGER_ROOT
        cls._build_env_tree()
        cls._load_g4_scores()
        cls._install_forward_stub()

    @classmethod
    def tearDownClass(cls):
        cls.fr.forward_and_score_batch = cls._original_forward
        cls.fr.PRODUCTION_BUDGET_ROOT = cls._original_budget_root_value

    # ---------- fixtures ----------

    @classmethod
    def _original_budget_root(cls):
        import importlib

        m = importlib.import_module("idea01.text2cad_frozen_runner")
        return m.__dict__.get("_original_production_budget_root", m.__dict__["PRODUCTION_BUDGET_ROOT"])

    @classmethod
    def _build_env_tree(cls):
        pairs = [
            (LOCAL_T2C / "text2cad_v1.0" / "Text2CAD_1.0.pth",
             cls.ENV_ROOT / "models" / "Text2CAD-v1.0" / "422d4c93198c3cd98d5c2092c0ab214b3a76a68d" / "Text2CAD_1.0.pth"),
            (LOCAL_BERT / "model.safetensors",
             cls.ENV_ROOT / "models" / "bert-large-uncased" / "6da4b6a26a1877e173fca3225479512db81a5e5b" / "model.safetensors"),
            (LOCAL_T2C / "cad_seq.zip",
             cls.ENV_ROOT / "datasets" / "Text2CAD-v1.0" / "422d4c93198c3cd98d5c2092c0ab214b3a76a68d" / "cad_seq.zip"),
        ]
        small = [
            (LOCAL_BERT / name,
             cls.ENV_ROOT / "models" / "bert-large-uncased" / "6da4b6a26a1877e173fca3225479512db81a5e5b" / name)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt")
        ]
        for src, dst in pairs + small:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        cls.CHECKPOINT = pairs[0][1]
        cls.BERT = pairs[1][1].parent  # loader expects the bert DIRECTORY
        cls.ZIP = pairs[2][1]

    @classmethod
    def _load_g4_scores(cls):
        """(orig_text_index, orig_cad_index) -> full G4 score dict, for the 8
        replay texts limited to the selected candidates."""
        cfg = cls.CFG
        dense_to_orig = list(cfg["replay_dense_to_original_cad_index"])
        cls.g4_scores = {}
        cls.orig_by_text = {}
        for k, meta in enumerate(cfg["replay_texts"]):
            orig_t = int(meta["original_text_index"])
            text = meta["text"]
            cls.orig_by_text[text] = orig_t
            card = 0 if orig_t < 1822 else 1
            name = f"chunk_s{card:02d}_t{orig_t:05d}_{orig_t + 1:05d}.jsonl"
            with open(G4_RAW / f"card{card}" / name, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    r = json.loads(line)
                    cad = int(r["cad_index"])
                    if cad in dense_to_orig:
                        cls.g4_scores[(orig_t, cad)] = {
                            "score_x": r["score_x"], "score_y": r["score_y"],
                            "score_total": r["score_total"],
                            "n_valid_x": r["n_valid_x"], "n_valid_y": r["n_valid_y"],
                            "n_valid_total": r["n_valid_total"],
                            "valid_seq_len": r["valid_seq_len"],
                            "n_real_tokens": r["n_real_tokens"],
                            "mean_x": r["mean_x"], "mean_y": r["mean_y"],
                            "mean_total": r["mean_total"],
                        }
        cls.dense_to_orig = dense_to_orig
        expected = len(cfg["replay_texts"]) * len(dense_to_orig)
        assert len(cls.g4_scores) == expected, f"G4 mimic table {len(cls.g4_scores)} != {expected}"

    @classmethod
    def _install_forward_stub(cls):
        """Replace ONLY the model forward. Replay batches return the certified
        G4 scores for the same (text, candidate) pairs; other backgrounds get
        deterministic synthetic scores."""

        def stub_forward(model, store, texts, cad_indices, device, trim_mode="none"):
            out = []
            for text, cad_dense in zip(texts, cad_indices):
                if text in cls.orig_by_text:
                    orig_t = cls.orig_by_text[text]
                    orig_cad = cls.dense_to_orig[int(cad_dense)]
                    out.append(dict(cls.g4_scores[(orig_t, orig_cad)]))
                else:
                    s = -1.0 - int(cad_dense)
                    out.append({
                        "score_x": s / 2, "score_y": s / 2, "score_total": s,
                        "n_valid_x": 8, "n_valid_y": 2, "n_valid_total": 10,
                        "valid_seq_len": 8, "n_real_tokens": 10,
                        "mean_x": s / 20, "mean_y": s / 20, "mean_total": s / 10,
                    })
            time.sleep(cls.forward_delay)
            return out

        cls.fr.forward_and_score_batch = stub_forward
        cls.forward_delay = 0.0

    # ---------- helpers ----------

    def _auth(self, sha, deadline_epoch=None, max_pairs=32461):
        return {
            "approved_sha": sha,
            "max_pairs": max_pairs,
            "max_card_seconds": 900.0,
            "deadline_epoch": deadline_epoch if deadline_epoch is not None else time.time() + 3600,
            "authorization_meta": {"problems": [], "origin_url": "test", "origin_ref_sha": "0" * 40},
            "authorization_ref": "origin/glm-night-20261112",
        }

    def _dirs(self):
        return {
            "b0": MANIFESTS / "b0", "b1": MANIFESTS / "b1",
            "smoke_b0": MANIFESTS / "smoke_b0", "replay": MANIFESTS / "replay",
            "_original": LOCAL_V4,
        }

    def _patch_runner_edges(self, auth):
        e2.git_dirty = lambda repo_root: ""
        e2.head_is_detached = lambda repo_root: (True, "")
        e2.authorize_e2_run = lambda repo_root, cfg: auth

    def _ledger(self, sha):
        return self.fr.SharedBudgetLedger(
            self.LEDGER_ROOT / f"text2cad_e2_{sha}.json",
            approved_sha=sha, max_pairs=32461, max_card_seconds=900.0,
        )

    # ---------- the full-order regression ----------

    def test_full_order_smoke_replay_gate_b0_b1(self):
        sha = "f" * 40
        auth = self._auth(sha)
        self._patch_runner_edges(auth)
        dirs = self._dirs()
        run = self.RUN_ROOT / "full_order"

        def rc(which):
            return e2.score_e2(REPO, self.CFG, which, dirs, self.CHECKPOINT, self.BERT,
                               self.ZIP, run, "cpu", resume=False)

        # 1. smoke (1 pair)
        self.assertEqual(rc("smoke_b0"), 0)
        # 2. replay (8 x 2,046)
        self.assertEqual(rc("replay"), 0)
        # 3. tolerance gate vs the certified local G4 records
        rec = e2.compare_replay_to_g4(
            run / "replay", G4_RAW, self.CFG, sha, run / "replay_gate_record.json")
        self.assertTrue(rec["pass"], f"replay gate failed: {rec['violations'][:3]}")
        self.assertEqual(rec["n_pairs_compared"], 16368)
        # 4. b0 (gate must now accept), 5. b1 (additionally requires b0 done)
        self.assertEqual(rc("b0"), 0)
        self.assertEqual(rc("b1"), 0)

        ledger = self._ledger(sha)
        self.assertEqual(ledger.data["pairs_charged"], 32461)
        notes = [i["note"] for i in ledger.data["invocations"] if i["finalized"] and i["exit_code"] == 0]
        self.assertEqual(notes, [e2.SUCCESS_NOTE[w] for w in ("smoke_b0", "replay", "b0", "b1")])
        for inv in ledger.data["invocations"]:
            self.assertTrue(inv["finalized"])
            self.assertEqual(inv["exit_code"], 0)

    def test_past_deadline_refused_before_any_forward(self):
        sha = "e" * 40
        auth = self._auth(sha, deadline_epoch=time.time() - 1)
        self._patch_runner_edges(auth)
        with self.assertRaises(e2.E2Error) as ctx:
            e2.score_e2(REPO, self.CFG, "smoke_b0", self._dirs(), self.CHECKPOINT, self.BERT,
                        self.ZIP, self.RUN_ROOT / "past_deadline", "cpu")
        self.assertIn("deadline", str(ctx.exception))

    def test_midrun_deadline_fails_the_ledger_too(self):
        # deadline fires while the (stubbed) forward is running: the settle
        # point MUST convert this into a failure in both the process and the
        # ledger, with the success note suppressed (review round 4, item 4)
        sha = "d" * 40
        auth = self._auth(sha, deadline_epoch=time.time() + 2.0)
        self._patch_runner_edges(auth)
        self.forward_delay = 3.0  # the single smoke batch outlives the deadline
        try:
            rc = e2.score_e2(REPO, self.CFG, "smoke_b0", self._dirs(), self.CHECKPOINT, self.BERT,
                             self.ZIP, self.RUN_ROOT / "midrun_deadline", "cpu", resume=False)
        finally:
            self.forward_delay = 0.0
        self.assertEqual(rc, 2)
        ledger = self._ledger(sha)
        invs = ledger.data["invocations"]
        self.assertTrue(all(i["finalized"] for i in invs))
        self.assertTrue(all(i["exit_code"] == 2 for i in invs))
        self.assertFalse(any(str(i["note"]).startswith("e2_") and i["note"].endswith("_ok")
                             for i in invs))

    def test_evidence_failure_real_ledger_blocks_next_stage(self):
        # review round 5, item 2 with the REAL ledger: an environment-write
        # failure settles the invocation as a failure and the next stage is
        # refused (the success note never exists)
        sha = "c" * 40
        auth = self._auth(sha)
        self._patch_runner_edges(auth)
        original_env = self.fr.environment_record

        def failing_env(cfg, extra):
            raise OSError("simulated environment record write failure")

        self.fr.environment_record = failing_env
        try:
            rc = e2.score_e2(REPO, self.CFG, "smoke_b0", self._dirs(), self.CHECKPOINT, self.BERT,
                             self.ZIP, self.RUN_ROOT / "evidence_failure", "cpu", resume=False)
        finally:
            self.fr.environment_record = original_env
        self.assertEqual(rc, 2)
        ledger = self._ledger(sha)
        inv = ledger.data["invocations"][-1]
        self.assertTrue(inv["finalized"])
        self.assertEqual(inv["exit_code"], 2)
        self.assertEqual(inv["note"], "evidence_write_failed")
        with self.assertRaises(e2.E2Error) as ctx:
            e2.score_e2(REPO, self.CFG, "replay", self._dirs(), self.CHECKPOINT, self.BERT,
                        self.ZIP, self.RUN_ROOT / "evidence_failure", "cpu", resume=False)
        self.assertIn("smoke_b0", str(ctx.exception))

    def test_evidence_phase_wallclock_crossing_real_ledger(self):
        # review round 6, item 2 with the REAL ledger and a controllable clock:
        # the wall-clock deadline crosses DURING the evidence writes - settle #2
        # must catch it, settle process AND ledger as exit 2, rewrite the
        # evidence as failure evidence, and the next stage must be refused
        sha = "9" * 40
        auth = self._auth(sha)
        t0 = time.time()
        auth["deadline_epoch"] = t0 + 2.5
        self._patch_runner_edges(auth)

        class FakeClock:
            def __init__(self):
                self.t = t0

            def __call__(self):
                self.t += 1.0
                return self.t

        clock = FakeClock()
        original_now = e2.NOW_FN
        e2.NOW_FN = clock
        try:
            rc = e2.score_e2(REPO, self.CFG, "smoke_b0", self._dirs(), self.CHECKPOINT, self.BERT,
                             self.ZIP, self.RUN_ROOT / "evidence_phase_crossing", "cpu",
                             resume=False)
        finally:
            e2.NOW_FN = original_now
        self.assertEqual(rc, 2)
        ledger = self._ledger(sha)
        inv = ledger.data["invocations"][-1]
        self.assertEqual(inv["exit_code"], 2)
        # the deadline is crossed either in the loop (aborted) or at settle #2;
        # EITHER WAY the invariant holds: process exit == ledger exit == 2 and
        # no success note is written
        self.assertIn(inv["note"], ("aborted", "deadline_exceeded", "card_seconds_exceeded"))
        # the FINAL SETTLEMENT on disk must agree with the ledger verdict
        fin = json.loads((self.RUN_ROOT / "evidence_phase_crossing"
                          / f"final_settlement_{inv['invocation_id']}.json").read_text(encoding="utf-8"))
        self.assertIs(fin["settle_ok"], False)
        self.assertEqual(fin["exit_code"], 2)
        self.assertGreaterEqual(fin["ledger_card_seconds"], 1.0)  # real model load billed
        # the next stage is refused by the smoke gate
        self._patch_runner_edges(self._auth(sha))
        with self.assertRaises(e2.E2Error) as ctx:
            e2.score_e2(REPO, self.CFG, "replay", self._dirs(), self.CHECKPOINT, self.BERT,
                        self.ZIP, self.RUN_ROOT / "evidence_phase_crossing", "cpu",
                        resume=False)
        self.assertIn("smoke_b0", str(ctx.exception))

    def test_wall_crossing_at_finalize_real_ledger(self):
        # review round 8, item 2: the ledger's OWN billing clock (ended_epoch)
        # crosses the wall deadline AFTER settle #2 passed - the authoritative
        # settlement must amend the ledger to exit 2 and the next stage must
        # be refused. The ledger's clock is injected to run 500s ahead; the
        # supervisor is disabled (this path is enforced in-process).
        sha = "8" * 40
        base = time.time()
        auth = self._auth(sha)
        auth["deadline_epoch"] = base + 400.0
        self._patch_runner_edges(auth)
        e2.start_deadline_supervision = lambda a, l, ready_event=None: None

        import idea01.text2cad_frozen_runner as fr

        class FastClockLedger(fr.SharedBudgetLedger):
            def __init__(self, path, **kw):
                super().__init__(path, now_fn=lambda: base + 500.0, **kw)

        original_ledger_cls = fr.SharedBudgetLedger
        fr.SharedBudgetLedger = FastClockLedger
        original_sup = e2.start_deadline_supervision
        e2.start_deadline_supervision = lambda a, l, ready_event=None: None
        try:
            rc = e2.score_e2(REPO, self.CFG, "smoke_b0", self._dirs(), self.CHECKPOINT, self.BERT,
                             self.ZIP, self.RUN_ROOT / "wall_cross_finalize", "cpu", resume=False)
        finally:
            fr.SharedBudgetLedger = original_ledger_cls
            e2.start_deadline_supervision = original_sup
        self.assertEqual(rc, 2)
        ledger = self._ledger(sha)
        inv = ledger.data["invocations"][-1]
        self.assertEqual(inv["exit_code"], 2)
        self.assertEqual(inv["note"], "deadline_exceeded")
        fin = json.loads((self.RUN_ROOT / "wall_cross_finalize"
                          / f"final_settlement_{inv['invocation_id']}.json").read_text(encoding="utf-8"))
        self.assertIs(fin["wall_crossed"], True)
        self.assertIs(fin["settle_ok"], False)
        # replay refused by the failed smoke settlement
        self._patch_runner_edges(self._auth(sha))
        e2.start_deadline_supervision = lambda a, l, ready_event=None: None
        with self.assertRaises(e2.E2Error) as ctx:
            e2.score_e2(REPO, self.CFG, "replay", self._dirs(), self.CHECKPOINT, self.BERT,
                        self.ZIP, self.RUN_ROOT / "wall_cross_finalize", "cpu", resume=False)
        self.assertIn("smoke_b0", str(ctx.exception))

    def test_final_settlement_write_failure_real_ledger(self):
        # review round 8, item 2 (second repro): if the final settlement record
        # cannot be published, the ledger verdict is amended to a failure and
        # the next stage is refused - no provisional success survives
        sha = "7" * 40
        auth = self._auth(sha)
        self._patch_runner_edges(auth)
        import idea01.text2cad_frozen_runner as fr

        original_publish = fr.atomic_publish

        def failing_publish(path, lines):
            if "final_settlement_" in str(path):
                raise OSError("simulated final settlement write failure")
            return original_publish(path, lines)

        fr.atomic_publish = failing_publish
        e2_target = sys.modules["idea01.text2cad_e2_runner"]
        # score_e2 binds atomic_publish from frozen_runner at call time
        try:
            rc = e2.score_e2(REPO, self.CFG, "smoke_b0", self._dirs(), self.CHECKPOINT, self.BERT,
                             self.ZIP, self.RUN_ROOT / "settlement_write_fail", "cpu",
                             resume=False)
        finally:
            fr.atomic_publish = original_publish
        self.assertEqual(rc, 2)
        ledger = self._ledger(sha)
        inv = ledger.data["invocations"][-1]
        self.assertEqual(inv["exit_code"], 2)
        self.assertEqual(inv["note"], "final_settlement_write_failed")
        # next stage refused
        self._patch_runner_edges(self._auth(sha))
        import idea01.text2cad_e2_runner as e2mod
        original_sup = e2mod.start_deadline_supervision
        e2mod.start_deadline_supervision = lambda a, l, ready_event=None: None
        try:
            with self.assertRaises(e2.E2Error) as ctx:
                e2mod.score_e2(REPO, self.CFG, "replay", self._dirs(), self.CHECKPOINT,
                               self.BERT, self.ZIP,
                               self.RUN_ROOT / "settlement_write_fail", "cpu", resume=False)
        finally:
            e2mod.start_deadline_supervision = original_sup
        self.assertIn("smoke_b0", str(ctx.exception))

    def test_amend_preserves_concurrent_charges_real_ledger(self):
        # review round 9, item 1 with TWO REAL ledger instances: the amendment
        # must re-read the file fresh and patch only the matching invocation,
        # preserving the concurrent instance's pending entry and the
        # cumulative pairs_charged totals
        import idea01.text2cad_frozen_runner as fr

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "led.json"
            L1 = fr.SharedBudgetLedger(path, approved_sha="a" * 40,
                                       max_pairs=416, max_card_seconds=900.0)
            L2 = fr.SharedBudgetLedger(path, approved_sha="a" * 40,
                                       max_pairs=416, max_card_seconds=900.0)
            L1.begin_invocation("I1", 0, 1)
            L1.charge_pairs("I1", 1)
            L1.finalize_invocation("I1", exit_code=0,
                                    note=e2.SUCCESS_NOTE["smoke_b0"])
            L2.begin_invocation("I2", 0, 1)
            L2.charge_pairs("I2", 1)  # pending: concurrent stage mid-flight

            e2.amend_ledger_verdict_on_disk(path, L1, "I1", 2,
                                             "card_seconds_exceeded")

            fresh = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(fresh["pairs_charged"], 2)
            i1, i2 = fresh["invocations"]
            self.assertEqual(i1["exit_code"], 2)
            self.assertEqual(i1["note"], "card_seconds_exceeded")
            self.assertIs(i2["finalized"], False)  # pending preserved
            self.assertEqual(i2["pairs_charged"], 1)
            # in-memory view refreshed so downstream gates read the truth
            self.assertEqual(L1.data["invocations"][1]["finalized"], False)

    def test_g4_sums_pin_mismatch_refuses_replay(self):
        # review round 5, item 3: a comparator run against files that do not
        # match the PINNED certified G4 SHA256SUMS is refused outright
        auth = self._auth("b" * 40)
        bad_cfg = dict(self.CFG)
        bad_cfg["g4_sha256sums_sha256"] = "0" * 64
        with self.assertRaises(e2.E2Error) as ctx:
            e2.compare_replay_to_g4(
                self.RUN_ROOT / "full_order" / "replay", G4_RAW, bad_cfg,
                "b" * 40, self.RUN_ROOT / "g4_pin_mismatch_gate.json")
        self.assertIn("pinned certified", str(ctx.exception))

    def test_g4_chunk_drift_from_certified_list_refused(self):
        # a G4 chunk whose bytes differ from the certified list is refused even
        # when the sums FILE itself matches the pin
        import hashlib

        def _sha_text(text):
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        def _sha_bytes(data):
            return hashlib.sha256(data).hexdigest()

        def _jl(rows):
            return b"".join(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n"
                            for r in rows)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            g4 = root / "card0"
            g4.mkdir(parents=True)
            payload = _jl([{"record_type": "chunk_header"}] + [
                {"record_type": "pair", "cad_index": c, "status": "scored",
                 "score_total": -100.0 - c, "n_valid_total": 10} for c in range(2)])
            (g4 / "chunk_s00_t00000_00001.jsonl").write_bytes(payload)
            # the certified list pins a DIFFERENT sha than the on-disk chunk
            sums = "0" * 64 + "  card0/chunk_s00_t00000_00001.jsonl\n"
            (root / "SHA256SUMS").write_bytes(sums.encode("utf-8"))
            cfg = {
                "backgrounds": {"replay": {"n_unique_text": 1, "n_gallery": 2}},
                "replay_texts": [{"original_text_index": 0, "text": "t", "text_sha256": _sha_text("t")}],
                "g4_sha256sums_sha256": _sha_bytes(sums.encode("utf-8")),
                "replay_dense_to_original_cad_index": [0, 1],
            }
            rep = root / "replay"
            rep.mkdir()
            (rep / "chunk_s00_t00000_00001.jsonl").write_bytes(payload)
            with self.assertRaises(e2.E2Error) as ctx:
                e2.compare_replay_to_g4(rep, root, cfg, "a" * 40, root / "gate.json")
            self.assertIn("certified", str(ctx.exception))

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "fork"),
                         "POSIX process-group supervision: runs on the Linux preflight host")
    def test_posix_supervisor_kills_stuck_child_by_deadline(self):
        """REAL stuck-process regression (review round 5, item 1): an
        independent supervisor process must terminate a child that is blocked
        forever, BY the deadline (grace counted into the caps)."""
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "led.json"
            ledger_path.write_text(json.dumps({
                "invocations": [{"invocation_id": "X", "finalized": False,
                                 "started_epoch": time.time()}],
                "card_seconds_spent": 0.0,
            }), encoding="utf-8")
            deadline = time.time() + 35.0  # TERM at +5, KILL at +35
            pid = os.fork()
            if pid == 0:
                try:
                    while True:
                        time.sleep(1)
                finally:
                    os._exit(9)
            sup = ctx.Process(target=e2._supervise_loop,
                              args=(pid, str(ledger_path), deadline, 900.0, 1))
            sup.start()
            died_at = None
            t_end = deadline + 10.0
            while time.time() < t_end:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    died_at = time.time()
                    break
                time.sleep(0.2)
            if died_at is None:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
            if sup.is_alive():
                sup.terminate()
            self.assertIsNotNone(died_at, "stuck child survived past the deadline")
            self.assertLessEqual(died_at, deadline + 1.0, "KILL must land BY the deadline")


if __name__ == "__main__":
    unittest.main()
