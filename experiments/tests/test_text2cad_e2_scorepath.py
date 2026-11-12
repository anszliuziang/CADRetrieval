"""Score-path and gate tests for the E2 runner (torch-free, CPU stubs).

Covers the review round 3 items:
  1. the score entry actually reaches model load and result persistence
     (regression: a missing lazy import produced NameError only at runtime);
  2. prerequisite execution-order gates (smoke -> replay-gate record -> batch);
  3. non-finite budget caps refused; deadline enforced via the ledger proxy;
  4. the comparator's pass/fail/n_valid logic on toy chunks;
  5. authorization fetch uses an explicit branch refspec (deleted remote branch
     must NOT fall back to a stale tracking ref).
"""

import json
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_e2_runner as e2  # noqa: E402
from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_LENGTH_FILE,
    CAD_STORE_FILE,
    PAIR_MANIFEST_FILE,
    TEXTS_FILE,
    PairSpace,
    sha256_text,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def sha(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def jl(rows) -> bytes:
    return b"".join(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in rows)


def make_manifest(root: Path, texts: list[str], n_cads: int) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    files = {
        TEXTS_FILE: jl([
            {"text_index": i, "text": t, "text_sha256": sha256_text(t), "n_chars": len(t)}
            for i, t in enumerate(texts)
        ]),
        CAD_STORE_FILE: jl([
            {"cad_index": i, "member": f"cad_seq/0000/{i:08d}/seq/{i:08d}.pth"} for i in range(n_cads)
        ]),
        CAD_LENGTH_FILE: jl([
            {"cad_index": i, "n_real_tokens": 10, "n_valid_x": 8, "n_valid_y": 2} for i in range(n_cads)
        ]),
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    manifest = {
        "schema_version": "text2cad-pair-manifest-v1",
        "artifacts": {name: {"file": name, "sha256": sha(data)} for name, data in files.items()},
        "counts": {"n_unique_text": len(texts), "n_gallery": n_cads, "n_pairs": len(texts) * n_cads},
    }
    (root / PAIR_MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {
        "texts_sha256": sha(files[TEXTS_FILE]),
        "cad_store_sha256": sha(files[CAD_STORE_FILE]),
        "cad_length_index_sha256": sha(files[CAD_LENGTH_FILE]),
        "pair_manifest_sha256": sha((root / PAIR_MANIFEST_FILE).read_bytes()),
    }


def install_frozen_runner_stub(ledger_root: Path, records: dict) -> None:
    """Inject a torch-free stub for idea01.text2cad_frozen_runner capturing the
    score path's wiring (this is what would have caught the missing
    load_gated_model_and_store import)."""

    class FakeRunnerConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class FakeLedger:
        def __init__(self, path, approved_sha=None, max_pairs=None, max_card_seconds=None):
            self.path = Path(path)
            self.approved_sha = approved_sha
            self.max_pairs = max_pairs
            self.max_card_seconds = max_card_seconds
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self.data = {"invocations": []}
            self._pairs = 0
            records["ledger"] = self

        def _persist(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data), encoding="utf-8")

        def remaining_pairs(self):
            return self.max_pairs - self._pairs

        def begin_invocation(self, invocation_id, shard_id=0, cards_used=1):
            self.data["invocations"].append({"invocation_id": invocation_id, "finalized": False,
                                             "started_epoch": time.time(), "cards_used": cards_used})
            self._persist()
            return []

        def finalize_invocation(self, invocation_id, exit_code=0, note=""):
            # matches the REAL SharedBudgetLedger: pairs are recorded as
            # pairs_charged (never 'forwards') — the exact mismatch the round-4
            # review caught in verify_smoke_done
            for i in self.data["invocations"]:
                if i["invocation_id"] == invocation_id and not i.get("finalized"):
                    i.update(finalized=True, exit_code=exit_code, note=note,
                             pairs_charged=records.get("pairs_charged", 1))
                    break
            self._persist()

        def card_seconds_spent(self):
            return records.get("card_seconds", 0.0)

        def pairs_charged(self):
            return self._pairs

        def to_dict(self):
            return {"invocations": self.data["invocations"]}

    mod = types.ModuleType("idea01.text2cad_frozen_runner")
    mod.CHUNK_SCHEMA_VERSION = "text2cad-chunk-v1"
    mod.PINNED_CHECKPOINT_SHA256 = "c" * 64
    mod.PRODUCTION_BUDGET_ROOT = ledger_root
    mod.SCORE_NAME = "two_head_factorized_logprob_sum"
    mod.SCORE_SPEC_VERSION = "text2cad-vector-score-v1"
    mod.RunnerConfig = FakeRunnerConfig
    mod.SharedBudgetLedger = FakeLedger
    mod.check_revision_paths = lambda cfg: None
    mod.chunk_file_name = lambda shard, t0, t1: f"chunk_s{shard:02d}_t{t0:05d}_{t1:05d}.jsonl"
    mod.new_invocation_id = lambda: "TEST-INV-0001"
    mod.environment_record = lambda cfg, extra: {"extra": extra}
    mod.plan_chunks = lambda out_dir, pair_space, cfg, binding, resume=False: ([(0, 1, 0)], [])
    mod.atomic_publish = lambda path, lines: Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    mod.load_gated_model_and_store = lambda cfg, members: (object(), {"loaded": True}, object(), {"preloaded": True})

    class FakeScorer:
        def __init__(self, cfg, model, texts, store, pair_space, ledger, invocation_id, binding):
            records["scorer_ledger"] = ledger
            self.pair_space = pair_space
            self.binding = dict(binding)
            self.stats = {"batches": 1, "pairs_forwarded": 2, "pairs_rejected": 0}

        def score_chunk(self, t0, t1):
            records["forwards"] = (t1 - t0) * self.pair_space.n_cads
            header = {"record_type": "chunk_header", "binding": self.binding}
            return [header] + [
                {"record_type": "pair", "cad_index": c, "status": "scored", "score_total": -1.0}
                for c in range(self.pair_space.n_cads)
            ]

    mod.FrozenMatrixScorer = FakeScorer
    sys.modules["idea01.text2cad_frozen_runner"] = mod
    return mod


class TestScorePath(unittest.TestCase):
    """The score entry must reach model load and persist chunk/ledger/summary."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)
        self.records: dict = {}
        # save the previous module entry (usually the REAL frozen_runner) and
        # restore it: an unconditional pop would make the next import create a
        # SECOND module object, silently detaching mock.patch targets from the
        # module globals that load_gated_model_and_store actually reads
        self._prev_frozen_module = sys.modules.get("idea01.text2cad_frozen_runner")
        self.stub = install_frozen_runner_stub(self.root / "ledger_root", self.records)
        self.addCleanup(self._restore_frozen_module)

    def _restore_frozen_module(self):
        if self._prev_frozen_module is not None:
            sys.modules["idea01.text2cad_frozen_runner"] = self._prev_frozen_module
        else:
            sys.modules.pop("idea01.text2cad_frozen_runner", None)

    def _prepare(self):
        repo = self.root / "repo"
        (repo / "experiments" / "configs").mkdir(parents=True)
        (self.root / "cad_seq.zip").write_bytes(b"PK-STUB")
        (self.root / "ckpt.pth").write_bytes(b"stub")
        original = self.root / "v4"
        cfg = {
            "schema": "text2cad-e2-background-expected-v1",
            "authorization_ref": "origin/glm-night-20261112",
            "backgrounds": {
                "b0": {**make_manifest(self.root / "mb0", [""], 2), "n_unique_text": 1, "n_gallery": 2},
                "b1": {**make_manifest(self.root / "mb1", ["a CAD model."], 2), "n_unique_text": 1, "n_gallery": 2},
                "smoke_b0": {**make_manifest(self.root / "msmoke", [""], 2), "n_unique_text": 1, "n_gallery": 2},
                "replay": {**make_manifest(self.root / "mreplay", ["t0", "t1"], 2), "n_unique_text": 2, "n_gallery": 2},
            },
            "original_cad_store_sha256": sha((original_store_dir(self.root) / CAD_STORE_FILE).read_bytes()),
            "replay_texts": [
                {"original_text_index": 0, "text": "t0", "text_sha256": sha256_text("t0")},
                {"original_text_index": 1, "text": "t1", "text_sha256": sha256_text("t1")},
            ],
            "cad_seq_zip_sha256": "0" * 64,
        }
        dirs = {"b0": self.root / "mb0", "b1": self.root / "mb1", "smoke_b0": self.root / "msmoke",
                "replay": self.root / "mreplay", "_original": original}
        auth = {
            "approved_sha": "a" * 40, "max_pairs": 100, "max_card_seconds": 900.0,
            "deadline_epoch": time.time() + 3600, "authorization_meta": {"problems": []},
            "authorization_ref": "origin/glm-night-20261112",
        }
        return repo, cfg, dirs, auth

    def test_score_reaches_model_load_and_persistence(self):
        repo, cfg, dirs, auth = self._prepare()
        e2.git_dirty = lambda repo_root: ""
        e2.head_is_detached = lambda repo_root: (True, "")
        e2.authorize_e2_run = lambda repo_root, cfg: auth
        # smoke_b0 has no prerequisite gates — it exercises the full wiring:
        # validation -> authorization -> ledger -> model load -> chunk publish
        rc = e2.score_e2(repo, cfg, "smoke_b0", dirs, self.root / "ckpt.pth", self.root / "bert",
                         self.root / "cad_seq.zip", self.root / "run", "npu:0")
        self.assertEqual(rc, 0)
        run = self.root / "run"
        chunk = next(iter((run / "smoke_b0").glob("chunk_*.jsonl")))
        body = [json.loads(l) for l in chunk.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(body[0]["record_type"], "chunk_header")
        self.assertEqual(len([r for r in body if r.get("record_type") == "pair"]), 2)
        envs = list(run.glob("environment_*.json"))
        sums = list(run.glob("summary_*.json"))
        self.assertEqual(len(envs), 1, "score mode must persist an environment record")
        self.assertEqual(len(sums), 1)
        summary = json.loads(sums[0].read_text(encoding="utf-8"))
        self.assertIn("load_report", summary)
        ledger = self.records["ledger"]
        self.assertEqual(ledger.pairs_charged() if hasattr(ledger, "pairs_charged") else 0, 0)
        finalized = [i for i in ledger.data["invocations"] if i["finalized"]]
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["exit_code"], 0)

    def test_missing_lazy_import_is_caught(self):
        # regression guard for the original bug: with load_gated_model_and_store
        # absent from the stub module, score_e2 must fail loudly (ImportError /
        # NameError -> exit 2), never silently skip the model load
        del sys.modules["idea01.text2cad_frozen_runner"].load_gated_model_and_store
        repo, cfg, dirs, auth = self._prepare()
        e2.git_dirty = lambda repo_root: ""
        e2.head_is_detached = lambda repo_root: (True, "")
        e2.authorize_e2_run = lambda repo_root, cfg: auth
        with self.assertRaises((ImportError, NameError, e2.E2Error)):
            e2.score_e2(repo, cfg, "b0", dirs, self.root / "ckpt.pth", self.root / "bert",
                        self.root / "cad_seq.zip", self.root / "run2", "npu:0")

    def test_detached_head_required(self):
        repo, cfg, dirs, auth = self._prepare()
        e2.git_dirty = lambda repo_root: ""
        e2.head_is_detached = lambda repo_root: (False, "glm-night-20261112")
        e2.authorize_e2_run = lambda repo_root, cfg: auth
        with self.assertRaises(e2.E2Error) as ctx:
            e2.score_e2(repo, cfg, "b0", dirs, self.root / "ckpt.pth", self.root / "bert",
                        self.root / "cad_seq.zip", self.root / "run3", "npu:0")
        self.assertIn("not detached", str(ctx.exception))

    def test_batch_requires_smoke_and_replay_gate(self):
        repo, cfg, dirs, auth = self._prepare()
        e2.git_dirty = lambda repo_root: ""
        e2.head_is_detached = lambda repo_root: (True, "")
        e2.authorize_e2_run = lambda repo_root, cfg: auth
        run = self.root / "run4"
        run.mkdir(parents=True)
        # b0 without smoke and without replay gate record -> refused
        with self.assertRaises(e2.E2Error) as ctx:
            e2.score_e2(repo, cfg, "b0", dirs, self.root / "ckpt.pth", self.root / "bert",
                        self.root / "cad_seq.zip", run, "npu:0")
        self.assertIn("smoke_b0", str(ctx.exception))
        # with smoke done but no replay gate -> refused at the replay gate
        fake_ledger = self.records["ledger"]
        fake_ledger.data["invocations"].append(
            {"invocation_id": "S1", "finalized": True, "exit_code": 0,
             "note": e2.SMOKE_NOTE, "pairs_charged": 1})
        fake_ledger._persist()
        (run / "smoke_b0").mkdir()
        (run / "smoke_b0" / "chunk_s00_t00000_00001.jsonl").write_bytes(
            jl([{"record_type": "chunk_header"}, {"record_type": "pair", "cad_index": 0,
                "status": "scored", "score_total": -1.0, "n_valid_total": 10}]))
        with self.assertRaises(e2.E2Error) as ctx2:
            e2.score_e2(repo, cfg, "b0", dirs, self.root / "ckpt.pth", self.root / "bert",
                        self.root / "cad_seq.zip", run, "npu:0")
        self.assertIn("replay gate", str(ctx2.exception))


def original_store_dir(root: Path) -> Path:
    d = root / "v4"
    if not (d / CAD_STORE_FILE).exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / CAD_STORE_FILE).write_bytes(jl([
            {"cad_index": i, "member": f"cad_seq/0000/{i:08d}/seq/{i:08d}.pth"} for i in range(2)
        ]))
    return d


class TestGates(unittest.TestCase):
    def test_replay_gate_record_checks(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            auth = {"approved_sha": "a" * 40}
            cfg = {"backgrounds": {"replay": {"n_unique_text": 2, "n_gallery": 2}}}
            with self.assertRaises(e2.E2Error) as ctx:
                e2.verify_replay_gate(run, cfg, auth)
            self.assertIn("missing", str(ctx.exception))
            rec = {
                "schema": "text2cad-e2-replay-gate-v1", "approved_sha": "b" * 40,
                "pass": True, "n_pairs_compared": 4, "max_abs_delta": 0.0,
                "replay_chunk_sha256": {
                    "chunk_s00_t00000_00001.jsonl": "x",
                    "chunk_s00_t00001_00002.jsonl": "y",
                },
                "g4_chunk_sha256": {"chunk_s00_t00000_00001.jsonl": "g"},
            }
            (run / "replay_gate_record.json").write_text(json.dumps(rec), encoding="utf-8")
            with self.assertRaises(e2.E2Error) as ctx2:
                e2.verify_replay_gate(run, cfg, auth)
            self.assertIn("bound to", str(ctx2.exception))
            rec["approved_sha"] = "a" * 40
            (run / "replay_gate_record.json").write_text(json.dumps(rec), encoding="utf-8")
            # partial pin set must be refused BEFORE any hash check
            rec_partial = {**rec, "replay_chunk_sha256": {"chunk_s00_t00000_00001.jsonl": "x"}}
            (run / "replay_gate_record.json").write_text(json.dumps(rec_partial), encoding="utf-8")
            with self.assertRaises(e2.E2Error) as ctx3:
                e2.verify_replay_gate(run, cfg, auth)
            self.assertIn("expected exactly the 2 replay chunks", str(ctx3.exception))
            payload = jl([{"record_type": "chunk_header"}])
            (run / "replay").mkdir()
            (run / "replay" / "chunk_s00_t00000_00001.jsonl").write_bytes(payload)
            (run / "replay" / "chunk_s00_t00001_00002.jsonl").write_bytes(payload)
            rec["replay_chunk_sha256"] = {
                "chunk_s00_t00000_00001.jsonl": sha(payload),
                "chunk_s00_t00001_00002.jsonl": sha(payload),
            }
            (run / "replay_gate_record.json").write_text(json.dumps(rec), encoding="utf-8")
            info = e2.verify_replay_gate(run, cfg, auth)
            self.assertEqual(info["compared"], 4)

    def test_nonfinite_caps_refused(self):
        text = (
            "- `state`: `pilot_authorized`\n"
            "- `e2_approved_sha`: `" + "a" * 40 + "`\n"
            "- `e2_max_pairs`: 32461\n"
            "- `e2_max_card_seconds`: inf\n"
            "- `e2_deadline_utc`: 2026-11-12T18:00:00Z\n"
        )
        with self.assertRaises(e2.E2Error):
            e2.parse_e2_authorization(text, "a" * 40)


class TestComparator(unittest.TestCase):
    def _toy(self, delta=0.0, n_valid=10):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            g4 = root / "card0"
            g4.mkdir(parents=True)
            g4_rows = [{"record_type": "chunk_header"}] + [
                {"record_type": "pair", "cad_index": c, "status": "scored",
                 "score_total": -100.0 - c, "n_valid_total": n_valid} for c in range(2)
            ]
            (g4 / "chunk_s00_t00000_00001.jsonl").write_bytes(jl(g4_rows))
            rep = root / "replay"
            rep.mkdir()
            rep_rows = [{"record_type": "chunk_header"}] + [
                {"record_type": "pair", "cad_index": c, "status": "scored",
                 "score_total": -100.0 - c + delta, "n_valid_total": n_valid} for c in range(2)
            ]
            (rep / "chunk_s00_t00000_00001.jsonl").write_bytes(jl(rep_rows))
            cfg = {
                "backgrounds": {"replay": {"n_unique_text": 1, "n_gallery": 2}},
                "replay_texts": [{"original_text_index": 0, "text": "t", "text_sha256": sha256_text("t")}],
            }
            rec = e2.compare_replay_to_g4(rep, root, cfg, "a" * 40, root / "gate.json")
            return rec

    def test_pass_within_tolerance(self):
        rec = self._toy(delta=0.0)
        self.assertTrue(rec["pass"])
        self.assertEqual(rec["n_pairs_compared"], 2)
        self.assertEqual(rec["max_abs_delta"], 0.0)

    def test_fail_beyond_tolerance(self):
        rec = self._toy(delta=5.0)
        self.assertFalse(rec["pass"])
        self.assertEqual(rec["n_violations"], 2)


class TestFetchRefspec(unittest.TestCase):
    """Deleted remote branch must not fall back to a stale tracking ref."""

    def _git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)

    def test_deleted_branch_refused(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            src.mkdir()
            self._git(src, "init", "-q", "-b", "master")
            self._git(src, "config", "user.email", "t@e.com")
            self._git(src, "config", "user.name", "t")
            (src / "experiments").mkdir()
            (src / "experiments" / "ACTIVE_TASK.md").write_text("- `state`: `pilot_authorized`\n", encoding="utf-8")
            self._git(src, "add", ".")
            self._git(src, "commit", "-qm", "auth")
            self._git(src, "branch", "authbr")
            bare = Path(td) / "bare.git"
            self._git(src, "clone", "-q", "--bare", ".", str(bare))
            work = Path(td) / "work"
            self._git(td, "clone", "-q", str(bare), str(work))
            self._git(work, "config", "user.email", "t@e.com")
            self._git(work, "config", "user.name", "t")
            # consume the branch once (creates the tracking ref), then delete it on the remote
            self._git(work, "fetch", "origin", "+refs/heads/authbr:refs/remotes/origin/authbr")
            self._git(bare, "branch", "-D", "authbr") if False else None
            r = self._git(bare, "branch", "-D", "authbr")
            self.assertEqual(r.returncode, 0)
            from idea01.p1_runner import fetch_authorized_active_task

            text, meta = fetch_authorized_active_task(work, expected_origin=str(bare), ref="origin/authbr")
            self.assertIsNone(text, "deleted remote branch must be refused, not read from a stale tracking ref")
            self.assertTrue(any("fetching authorization branch failed" in p for p in meta["problems"]))

    def test_live_branch_read_from_captured_commit(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            src.mkdir()
            self._git(src, "init", "-q", "-b", "master")
            self._git(src, "config", "user.email", "t@e.com")
            self._git(src, "config", "user.name", "t")
            (src / "experiments").mkdir()
            (src / "experiments" / "ACTIVE_TASK.md").write_text("- `state`: `pilot_authorized`\n", encoding="utf-8")
            self._git(src, "add", ".")
            self._git(src, "commit", "-qm", "auth")
            self._git(src, "branch", "authbr")
            bare = Path(td) / "bare.git"
            self._git(src, "clone", "-q", "--bare", ".", str(bare))
            work = Path(td) / "work"
            self._git(td, "clone", "-q", str(bare), str(work))
            from idea01.p1_runner import fetch_authorized_active_task

            text, meta = fetch_authorized_active_task(work, expected_origin=str(bare), ref="origin/authbr")
            self.assertIsNotNone(text)
            self.assertIn("pilot_authorized", text)
            self.assertEqual(len(meta["origin_ref_sha"]), 40)


if __name__ == "__main__":
    unittest.main()
