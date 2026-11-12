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


if __name__ == "__main__":
    unittest.main()
