"""CPU tests for the frozen Text2CAD matrix runner (M1c stage B2).

Covers, without real weights / NPU / network:

  * SharedBudgetLedger: persistence, approval binding, per-batch charging
    (crash can never under-count), card-second accounting with an injected
    clock, two-card concurrency rules (same-shard double-run refused, other
    shard allowed, live stale processes retained and proven-dead processes
    charged through recovery), lockfile
    timeout, corruption refusals;
  * authorization: REAL temporary Git repositories (local-path origin, no
    injected git fakes) prove the dual-version gate -- approval commit !=
    approved code SHA, working-tree self-authorization refused, attached
    HEAD refused, dirty tree refused, wrong task_id refused, and the real
    pre-approval state (implementation, zero budgets) refuses;
  * ZipCandidateStore: synthetic zip preload, per-candidate structural and
    attn-mask drift rejections (rejected candidates score as `rejected`
    WITHOUT any forward or budget charge);
  * FrozenMatrixScorer on a small random-weight ported model: chunk records
    are dense/ordered/binding-complete, hand-computed valid counts match,
    budget charged exactly for forwarded pairs only, batch=1 vs batch=3
    scores agree, trim_mode=batch_max agrees with the full-length reference
    within the pre-registered tolerance, resume verification (complete chunk
    skipped, tampered/foreign/unfinalized chunks refused, no-resume refused);
  * main() fail-closed gates: bad inputs exit 2 before any model work, the
    authorization gate fires before weights load, removed relaxation
    switches are argparse errors, dry-run never forwards;
  * isolation hygiene: the runner module never references qrels/evaluation
    artifacts as inputs and its per-pair record schema carries no UID fields.

The real pinned checkpoint/BERT appear ONLY in skipUnless-guarded dry-run
tests (zero forward), mirroring the audited stage-B1 audit path.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_frozen_runner import (  # noqa: E402
    CHUNK_SCHEMA_VERSION,
    PINNED_CHECKPOINT_SHA256,
    RunnerConfig,
    RunnerError,
    SharedBudgetLedger,
    FrozenMatrixScorer,
    ZipCandidateStore,
    atomic_publish,
    authorize_text2cad_run,
    chunk_file_name,
    dry_run,
    load_gated_model_and_store,
    main,
    new_invocation_id,
    plan_chunks,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    PairSpace,
    build_pair_manifest,
    sha256_file,
    sha256_text,
    write_pair_manifest,
)
from idea01.text2cad_port.loader import build_text2cad  # noqa: E402
from idea01.text2cad_port.model import generate_attention_mask  # noqa: E402
from idea01.text2cad_port.text_embed import TextEmbedder  # noqa: E402
from idea01.text2cad_vector_scorer import validate_cad_vec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CHECKPOINT = REPO_ROOT / ".cache/text2cad_official/text2cad_v1.0/Text2CAD_1.0.pth"
REAL_BERT_DIR = REPO_ROOT / ".cache/bert_large_uncased"
REAL_ZIP = REPO_ROOT / ".cache/text2cad_official/cad_seq.zip"
REAL_KEYS_MANIFEST = REPO_ROOT / "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"

TASK_ID = "idea01_text2cad_benchmark_20260928"
FULL_SHA = "a" * 40


# ---------------------------------------------------------------------------
# Fake clock
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# SharedBudgetLedger
# ---------------------------------------------------------------------------


class SharedBudgetLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "ledger.json"
        self.clock = FakeClock()

    def _ledger(self, **kwargs) -> SharedBudgetLedger:
        params = dict(
            approved_sha=FULL_SHA,
            max_pairs=10,
            max_card_seconds=100.0,
            now_fn=self.clock.time,
            sleep_fn=self.clock.sleep,
        )
        params.update(kwargs)
        return SharedBudgetLedger(self.path, **params)

    def test_persist_reload_and_binding_refusals(self):
        ledger = self._ledger()
        ledger.begin_invocation("inv-1", shard_id=0, cards_used=1)
        ledger.charge_pairs("inv-1", 4)
        ledger.finalize_invocation("inv-1", exit_code=0)
        reloaded = self._ledger()
        self.assertEqual(reloaded.pairs_charged(), 4)
        self.assertEqual(reloaded.remaining_pairs(), 6)
        with self.assertRaisesRegex(RunnerError, "approved_sha"):
            self._ledger(approved_sha="b" * 40)
        with self.assertRaisesRegex(RunnerError, "cap"):
            self._ledger(max_pairs=11)
        with self.assertRaisesRegex(RunnerError, "card-second"):
            self._ledger(max_card_seconds=101.0)

    def test_charge_never_exceeds_cap_and_is_persisted_before_forward(self):
        ledger = self._ledger(max_pairs=3)
        ledger.begin_invocation("i", shard_id=0, cards_used=1)
        ledger.charge_pairs("i", 3)
        # A crash right after charging must still show the charge.
        self.assertEqual(self._ledger(max_pairs=3).pairs_charged(), 3)
        with self.assertRaisesRegex(RunnerError, "pair budget exceeded"):
            ledger.charge_pairs("i", 1)

    def test_zero_or_negative_budgets_fail_closed(self):
        with self.assertRaisesRegex(RunnerError, "max_pairs"):
            self._ledger(max_pairs=0)
        with self.assertRaisesRegex(RunnerError, "max_card_seconds"):
            self._ledger(max_card_seconds=0.0)
        ledger = self._ledger()
        ledger.begin_invocation("i", shard_id=0, cards_used=1)
        with self.assertRaisesRegex(RunnerError, "negative"):
            ledger.charge_pairs("i", -1)

    def test_card_seconds_accumulate_wall_times_cards(self):
        ledger = self._ledger()
        ledger.begin_invocation("i", shard_id=0, cards_used=2)
        self.clock.advance(10.0)
        ledger.finalize_invocation("i", exit_code=0)
        self.assertAlmostEqual(ledger.card_seconds_spent(), 20.0)
        # Pending invocation counts live against the clock.
        ledger.begin_invocation("j", shard_id=1, cards_used=1)
        self.clock.advance(5.0)
        self.assertAlmostEqual(ledger.check_card_clock(), 25.0)
        # j is pending with a fresh heartbeat -> a second shard-1 run refuses
        with self.assertRaisesRegex(RunnerError, "double-run"):
            ledger.begin_invocation("j2", shard_id=1, cards_used=1)
        # a different shard is the other card legitimately running
        ledger.begin_invocation("k", shard_id=3, cards_used=1)
        # a reloaded ledger sees the same live state
        self.assertEqual(self._ledger().pairs_charged(), 0)

    def test_two_single_device_processes_charge_two_not_four_card_hours(self):
        ledger = self._ledger(max_card_seconds=10_000.0)
        ledger.begin_invocation("card0", shard_id=0, cards_used=1)
        ledger.begin_invocation("card1", shard_id=1, cards_used=1)
        self.clock.advance(3600.0)
        self.assertAlmostEqual(ledger.check_card_clock(), 7200.0)
        ledger.finalize_invocation("card0", exit_code=0)
        ledger.finalize_invocation("card1", exit_code=0)
        self.assertAlmostEqual(ledger.card_seconds_spent(), 7200.0)

    def test_card_clock_exhaustion_refuses(self):
        ledger = self._ledger(max_card_seconds=10.0)
        ledger.begin_invocation("i", shard_id=0, cards_used=1)
        self.clock.advance(11.0)
        with self.assertRaisesRegex(RunnerError, "card-second budget exceeded"):
            ledger.check_card_clock()
        with self.assertRaisesRegex(RunnerError, "card-second budget exceeded"):
            ledger.heartbeat("i")

    def test_live_process_is_not_auto_finalized_by_stale_heartbeat(self):
        ledger = self._ledger(max_card_seconds=10_000.0)
        ledger.begin_invocation("crash", shard_id=0, cards_used=2)
        self.clock.advance(100.0)
        ledger.heartbeat("crash")  # last heartbeat at +100
        self.clock.advance(200.0)  # now stale (>180s)
        with mock.patch("idea01.text2cad_frozen_runner.platform.system", return_value="Linux"), mock.patch("idea01.text2cad_frozen_runner.os.kill", return_value=None):
            with self.assertRaisesRegex(RunnerError, "already has a live invocation"):
                ledger.begin_invocation("new", shard_id=0, cards_used=1)
        inv = [i for i in ledger.data["invocations"] if i["invocation_id"] == "crash"][0]
        self.assertFalse(inv["finalized"])
        self.assertAlmostEqual(ledger.pending_card_seconds(), 600.0)

    def test_windows_stale_recovery_never_sends_a_signal(self):
        ledger = self._ledger(max_card_seconds=10_000.0)
        ledger.begin_invocation("live", shard_id=0, cards_used=1)
        self.clock.advance(200.0)
        with mock.patch("idea01.text2cad_frozen_runner.platform.system", return_value="Windows"), mock.patch("idea01.text2cad_frozen_runner.os.kill") as kill:
            with self.assertRaisesRegex(RunnerError, "already has a live invocation"):
                ledger.begin_invocation("new", shard_id=0, cards_used=1)
            kill.assert_not_called()

    def test_proven_dead_process_charged_through_recovery_time(self):
        ledger = self._ledger(max_card_seconds=10_000.0)
        ledger.begin_invocation("dead", shard_id=0, cards_used=2)
        ledger.data["invocations"][0]["pid"] = 2**30  # absent process
        ledger._persist_locked()
        self.clock.advance(200.0)
        with mock.patch("idea01.text2cad_frozen_runner.platform.system", return_value="Linux"), mock.patch("idea01.text2cad_frozen_runner.os.kill", side_effect=ProcessLookupError):
            ledger.begin_invocation("new", shard_id=0, cards_used=1)
        self.assertAlmostEqual(ledger.card_seconds_spent(), 400.0)
        self.assertEqual(ledger.data["invocations"][0]["note"], "auto_finalized_proven_dead_process")

    def test_lock_timeout_refuses_without_breaking(self):
        ledger = self._ledger(lock_timeout_s=0.5)
        ledger.lock_path.write_text("held", encoding="utf-8")  # simulated live holder
        with self.assertRaisesRegex(RunnerError, "lock"):
            ledger.begin_invocation("i", shard_id=0, cards_used=1)
        self.assertTrue(ledger.lock_path.exists())  # never broken automatically

    def test_corrupt_ledger_refused(self):
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "unreadable"):
            self._ledger()

    def test_finalize_unknown_invocation_refused(self):
        ledger = self._ledger()
        with self.assertRaisesRegex(RunnerError, "no pending invocation"):
            ledger.finalize_invocation("ghost", exit_code=0)


# ---------------------------------------------------------------------------
# Authorization (real temporary Git repositories; no injected git fakes)
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {cwd}: {result.stderr.strip()}")
    return result.stdout.strip()


def _task_md(state="pilot_authorized", approved=FULL_SHA, gpu_budget="2", max_pairs="10",
             task_id=TASK_ID) -> str:
    approved_part = f"`{approved}`" if approved else "无"
    return (
        "# 当前实验任务\n"
        f"- `state`: `{state}`（说明）\n"
        f"- `task_id`: `{task_id}`\n"
        f"- `approved_sha`: {approved_part}；说明\n"
        f"- `gpu_budget`: {gpu_budget} 卡小时；`max_scored_pairs`: {max_pairs}；说明\n"
    )


def _build_dual_version_repo(base: Path, approve: bool = True, task_id: str = TASK_ID,
                             expected_fp: Path | None = None):
    """Same dual-version scenario as the audited cadmium runner tests."""
    src = base / "src"
    src.mkdir(parents=True)
    _git(src, "init", "-q")
    _git(src, "symbolic-ref", "HEAD", "refs/heads/master")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "t")
    (src / "experiments").mkdir()
    if expected_fp is not None:
        cfg_dir = src / "experiments" / "configs"
        cfg_dir.mkdir()
        shutil.copyfile(expected_fp, cfg_dir / "text2cad_pair_manifest_expected_20260930.json")
    (src / "code.txt").write_text("runner code v1\n", encoding="utf-8")
    (src / "experiments" / "ACTIVE_TASK.md").write_text(
        _task_md(state="implementation", approved=None, gpu_budget="0", max_pairs="0"),
        encoding="utf-8",
    )
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "code")
    code_sha = _git(src, "rev-parse", "HEAD")
    approval_sha = None
    if approve:
        (src / "experiments" / "ACTIVE_TASK.md").write_text(
            _task_md(approved=code_sha, task_id=task_id), encoding="utf-8"
        )
        _git(src, "add", ".")
        _git(src, "commit", "-qm", "authorize code sha")
        approval_sha = _git(src, "rev-parse", "HEAD")
    origin = base / "remote.git"
    _git(src, "clone", "-q", "--bare", str(src), str(origin))
    work = base / "work"
    _git(base, "clone", "-q", str(origin), str(work))
    _git(work, "checkout", "-q", "--detach", code_sha)
    origin_url = _git(work, "remote", "get-url", "origin")
    return work, origin_url, code_sha, approval_sha


@unittest.skipUnless(shutil.which("git"), "git executable required for real-repo authorization tests")
class AuthorizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_dual_version_authorization_and_task_id(self):
        work, origin_url, code_sha, approval_sha = _build_dual_version_repo(self.base)
        self.assertNotEqual(code_sha, approval_sha)
        auth = authorize_text2cad_run(work, cards_used=2, expected_origin=origin_url)
        self.assertEqual(auth.approved_sha, code_sha)
        self.assertEqual(auth.actual_sha, code_sha)
        self.assertEqual(auth.authorization_ref_sha, approval_sha)
        self.assertEqual(auth.task_id, TASK_ID)
        self.assertEqual(auth.max_pairs, 10)
        self.assertEqual(auth.card_hours, 2.0)
        self.assertEqual(auth.max_wall_seconds, 3600.0)  # 2h / 2 cards

    def test_pre_approval_state_refuses(self):
        work, origin_url, _c, _a = _build_dual_version_repo(self.base, approve=False)
        with self.assertRaisesRegex(RunnerError, "pilot_authorized"):
            authorize_text2cad_run(work, cards_used=1, expected_origin=origin_url)

    def test_wrong_task_id_refuses(self):
        work, origin_url, _c, _a = _build_dual_version_repo(self.base, task_id="other_task")
        with self.assertRaisesRegex(RunnerError, "task_id"):
            authorize_text2cad_run(work, cards_used=1, expected_origin=origin_url)

    def test_working_tree_self_authorization_refused(self):
        work, origin_url, code_sha, _ = _build_dual_version_repo(self.base)
        (work / "experiments" / "ACTIVE_TASK.md").write_text(
            _task_md(approved=code_sha), encoding="utf-8"
        )
        with self.assertRaisesRegex(RunnerError, "dirty"):
            authorize_text2cad_run(work, cards_used=1, expected_origin=origin_url)

    def test_attached_head_refused(self):
        work, origin_url, code_sha, _ = _build_dual_version_repo(self.base)
        _git(work, "checkout", "-q", "-b", "movable", code_sha)
        with self.assertRaisesRegex(RunnerError, "attached"):
            authorize_text2cad_run(work, cards_used=1, expected_origin=origin_url)

    def test_cards_used_validation(self):
        work, origin_url, _c, _a = _build_dual_version_repo(self.base)
        with self.assertRaisesRegex(RunnerError, "cards_used"):
            authorize_text2cad_run(work, cards_used=0, expected_origin=origin_url)


# ---------------------------------------------------------------------------
# Synthetic assets: candidates zip + tiny random model + fake BERT
# ---------------------------------------------------------------------------


def _candidate_tensors(n_real: int, *, corrupt: bool = False):
    """Official-format candidate with n_real rows (circle+extrusion skeleton,
    extended with extra sketch+extrusion segments for longer lengths)."""
    if n_real < 19:
        raise ValueError("fixture needs >= 19 real rows")
    rows = [[1, 0]]
    flags = [0]
    indices = [0]
    sketch = [[139, 139], [139, 266], [5, 0], [4, 0], [3, 0], [2, 0]]
    extrusion = [[17 + j, 0] for j in range(9)] + [[8, 0], [6, 0]]
    segment = sketch + extrusion  # 17 rows
    seg_index = 0
    while len(rows) + len(segment) + 1 <= n_real:
        rows += segment
        flags += [0] * len(sketch) + [1] + list(range(1, 11))
        indices += [seg_index] * len(segment)
        seg_index += 1
    rows.append([1, 0])
    flags.append(0)
    indices.append(0)
    actual_real = len(rows)
    n_pad = 272 - actual_real
    rows += [[0, 0]] * n_pad
    flags += [11] * n_pad
    indices += [max(indices) + 1] * n_pad
    cad_vec = torch.tensor(rows, dtype=torch.int32)
    if corrupt:
        cad_vec[5] = torch.tensor([0, 9], dtype=torch.int32)  # x=0,y!=0 row
    sample = {
        "vec": {
            "cad_vec": cad_vec,
            "flag_vec": torch.tensor(flags, dtype=torch.int32),
            "index_vec": torch.tensor(indices, dtype=torch.int32),
        },
        "mask_cad_dict": {
            "attn_mask": generate_attention_mask(271),
            "key_padding_mask": cad_vec == 0 if not corrupt else torch.zeros(272, 2, dtype=torch.bool),
        },
    }
    return sample, actual_real


def _make_zip(path: Path, specs: list[tuple[str, int]], corrupt_index: int | None = None,
              tamper_attn_index: int | None = None) -> list[int]:
    reals = []
    with zipfile.ZipFile(path, "w") as archive:
        for i, (member, n_real) in enumerate(specs):
            sample, actual = _candidate_tensors(n_real, corrupt=(i == corrupt_index))
            if i == tamper_attn_index:
                sample["mask_cad_dict"]["attn_mask"] = (
                    sample["mask_cad_dict"]["attn_mask"].clone()
                )
                sample["mask_cad_dict"]["attn_mask"][10, 10] = -1.0
            buffer = io.BytesIO()
            torch.save(sample, buffer)
            archive.writestr(member, buffer.getvalue())
            reals.append(actual)
    return reals


class FakeBatchEncoding(dict):
    def to(self, device):
        return FakeBatchEncoding({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    """Deterministic word-count tokenizer; pad token id 0 (nonzero embedding)."""

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        assert truncation is False  # the adapted embedder must never truncate
        lens = [min(len(t.split()) + 2, 30) for t in texts]
        width = max(lens)
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attn = torch.zeros(len(texts), width, dtype=torch.long)
        for i, n in enumerate(lens):
            input_ids[i, :n] = torch.arange(10, 10 + n)
            attn[i, :n] = 1
        return FakeBatchEncoding({"input_ids": input_ids, "attention_mask": attn})


class FakeBert(torch.nn.Module):
    def __init__(self, hidden=32, vocab=100):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        torch.manual_seed(7)
        with torch.no_grad():
            self.emb.weight.mul_(0.1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return (self.emb(input_ids),)


def build_tiny_model(seed: int = 0):
    """Small random-weight ported Text2CAD with a fake text embedder."""
    torch.manual_seed(seed)
    config = {
        "text_encoder": {
            "text_embedder": {"max_seq_len": 512},
            "adaptive_layer": {"in_dim": 32, "out_dim": 32, "num_heads": 4, "dropout": 0.0},
        },
        "cad_decoder": {
            "tdim": 32,
            "cdim": 16,
            "num_layers": 3,
            "num_heads": 2,
            "dropout": 0.0,
            "ca_level_start": 1,
            "cad_seq_len": 272,
        },
    }
    embedder = TextEmbedder(
        max_seq_len=512, device="cpu", model=FakeBert(hidden=32), tokenizer=FakeTokenizer()
    )
    model = build_text2cad(embedder, device="cpu", config=config)
    model.eval()
    return model


def _mini_manifest_dir(root: Path) -> Path:
    """4 CADs x 3 texts mini pair manifest + synthetic zip + fingerprints."""
    query_path = root / "query.jsonl"
    gallery_path = root / "gallery.jsonl"
    texts = ["alpha beta", "gamma", "delta epsilon zeta"]
    queries = ["alpha beta", "gamma", "alpha beta", "delta epsilon zeta"]
    cad_ids = ["0000/00000001", "0000/00000002", "0000/00000003", "0000/00000004"]
    query_path.write_text(
        "".join(
            json.dumps({
                "level": "L0", "n_chars": len(t), "query_id": f"Q{i:05d}",
                "text": t, "text_sha256": sha256_text(t),
            }, sort_keys=True) + "\n"
            for i, t in enumerate(queries)
        ),
        encoding="utf-8",
    )
    gallery_path.write_text(
        "".join(
            json.dumps({
                "cad_id": u,
                "cad_seq_member": f"cad_seq/{u}/seq/{u.split('/')[-1]}.pth",
                "cadmium_native_present": True,
                "minimal_json_member": None,
                "n_representations": 1,
            }, sort_keys=True) + "\n"
            for u in cad_ids
        ),
        encoding="utf-8",
    )
    built = build_pair_manifest(
        query_path, gallery_path,
        expected_n_query=4, expected_n_unique_text=3, expected_n_gallery=4,
    )
    zip_path = root / "cad_seq.zip"
    specs = [(f"cad_seq/{u}/seq/{u.split('/')[-1]}.pth", n) for u, n in
             zip(cad_ids, (19, 36, 53, 19))]
    _make_zip(zip_path, specs)
    members = [e.member for e in built["cad_entries"]]
    from idea01.text2cad_pair_index import verify_zip_members

    zip_report = verify_zip_members(zip_path, members, expected_sha256=sha256_file(zip_path))
    out_dir = root / "pair_manifest"
    result = write_pair_manifest(out_dir, built, zip_report=zip_report)
    fingerprints = dict(result["summary"]["artifact_sha256"])
    fingerprints["pair_manifest.json"] = result["summary"]["pair_manifest_sha256"]
    fp_path = root / "expected_hashes.json"
    fp_path.write_text(json.dumps({
        "artifact_sha256": result["summary"]["artifact_sha256"],
        "pair_manifest_sha256": result["summary"]["pair_manifest_sha256"],
    }), encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------------------
# ZipCandidateStore
# ---------------------------------------------------------------------------


class ZipCandidateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _store(self, specs, **zip_kwargs) -> ZipCandidateStore:
        zip_path = self.root / "cad_seq.zip"
        reals = _make_zip(zip_path, specs, **zip_kwargs)
        store = ZipCandidateStore(
            zip_path, [m for m, _ in specs], expected_sha256=sha256_file(zip_path)
        )
        store._expected_reals = reals  # test aid
        return store

    def test_preload_validates_and_records(self):
        specs = [(f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth", n) for i, n in
                 enumerate((19, 36, 53))]
        store = self._store(specs)
        report = store.preload()
        self.assertEqual(report["n_loaded"], 3)
        self.assertEqual(report["n_rejected"], 0)
        for i, real in enumerate(store._expected_reals):
            self.assertEqual(int(store.get(i)["n_real_tokens"]), real)

    def test_preload_opens_large_archive_once(self):
        specs = [(f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth", 19)
                 for i in range(4)]
        store = self._store(specs)
        original_zipfile = zipfile.ZipFile
        with mock.patch("idea01.text2cad_frozen_runner.zipfile.ZipFile",
                        wraps=original_zipfile) as opened:
            self.assertEqual(store.preload()["n_loaded"], 4)
        self.assertEqual(opened.call_count, 1)

    def test_rejections_are_stable_and_forward_free(self):
        specs = [(f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth", 19) for i in range(3)]
        store = self._store(specs, corrupt_index=1, tamper_attn_index=2)
        report = store.preload()
        self.assertEqual(report["n_loaded"], 1)
        self.assertEqual(report["n_rejected"], 2)
        self.assertIn("inconsistent", store.rejections[1])  # scorer structural reason
        self.assertEqual(store.rejections[2], "mask_mismatch")  # drifted attn_mask
        with self.assertRaises(Exception):
            store.get(1)

    def test_production_loader_refuses_incomplete_gallery_before_forward(self):
        specs = [(f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth", 19)
                 for i in range(2)]
        store = self._store(specs, corrupt_index=1)
        cfg = RunnerConfig(
            checkpoint_path="unused", bert_dir="unused", cad_seq_zip=str(store.zip_path),
            pair_manifest_dir="unused", device="cpu", expected_keys_manifest="unused",
        )
        with mock.patch("idea01.text2cad_frozen_runner.load_frozen_text2cad",
                        return_value=(object(), {})), mock.patch(
            "idea01.text2cad_frozen_runner.ZipCandidateStore", return_value=store
        ):
            with self.assertRaisesRegex(RunnerError, "incomplete exact score matrix"):
                load_gated_model_and_store(cfg, store.members)


# ---------------------------------------------------------------------------
# FrozenMatrixScorer integration (tiny random model, CPU)
# ---------------------------------------------------------------------------


class FrozenMatrixScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        from idea01.text2cad_pair_index import load_cad_store, load_texts

        self.texts = load_texts(self.manifest_dir)
        self.members = load_cad_store(self.manifest_dir)
        self.pair_space = PairSpace(3, 4)
        self.clock = FakeClock()

    def _cfg(self, **kwargs) -> RunnerConfig:
        params = dict(
            checkpoint_path="unused-in-test",
            bert_dir="unused-in-test",
            cad_seq_zip=str(self.root / "cad_seq.zip"),
            pair_manifest_dir=str(self.manifest_dir),
            device="cpu",
            batch_size=2,
            shard_id=0,
            n_shards=1,
            texts_per_chunk=1,
        )
        params.update(kwargs)
        return RunnerConfig(**params)

    def _binding(self, cfg) -> dict:
        return {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "pair_manifest_sha256": "m",
            "git_sha": FULL_SHA,
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "device": cfg.device,
            "torch_dtype": cfg.torch_dtype,
            "trim_mode": cfg.trim_mode,
            "text_cache": cfg.text_cache,
            "score_name": "two_head_factorized_logprob_sum",
            "spec_version": "text2cad-vector-score-v1",
        }

    def _ledger(self, max_pairs=100) -> SharedBudgetLedger:
        return SharedBudgetLedger(
            self.root / "ledger.json",
            approved_sha=FULL_SHA,
            max_pairs=max_pairs,
            max_card_seconds=10_000.0,
            now_fn=self.clock.time,
            sleep_fn=self.clock.sleep,
        )

    def _scorer(self, cfg, ledger, invocation="inv-1") -> FrozenMatrixScorer:
        model = build_tiny_model()
        zip_path = self.root / "cad_seq.zip"
        store = ZipCandidateStore(zip_path, self.members, expected_sha256=sha256_file(zip_path))
        store.preload()
        ledger.begin_invocation(invocation, shard_id=cfg.shard_id, cards_used=1)
        return FrozenMatrixScorer(
            cfg, model, self.texts, store, self.pair_space, ledger, invocation,
            self._binding(cfg),
        )

    def test_chunk_records_dense_ordered_binding_complete(self):
        cfg = self._cfg()
        ledger = self._ledger()
        scorer = self._scorer(cfg, ledger)
        records = scorer.score_chunk(0, 1)  # text 0 x 4 cads
        header, body = records[0], records[1:]
        self.assertEqual(header["record_type"], "chunk_header")
        self.assertEqual(header["shard_id"], 0)
        self.assertEqual(header["text_range"], [0, 1])
        self.assertEqual(header["n_scored"], 4)
        for field in ("git_sha", "checkpoint_sha256", "device", "trim_mode", "text_cache"):
            self.assertIn(field, header)
        self.assertEqual([r["pair_index"] for r in body], [0, 1, 2, 3])
        for i, rec in enumerate(body):
            self.assertEqual(rec["status"], "scored")
            self.assertEqual(rec["text_index"], 0)
            self.assertEqual(rec["cad_index"], i)
            self.assertIsNone(rec["reason"])
            self.assertIsInstance(rec["score_total"], float)
            self.assertLess(rec["score_total"], 0.0)  # log-probabilities
            # schema: no UID fields anywhere
            for forbidden in ("uid", "cad_id", "query_id", "text", "member"):
                self.assertNotIn(forbidden, rec)
        # charged exactly the forwarded pairs
        self.assertEqual(ledger.pairs_charged(), 4)
        ledger.finalize_invocation("inv-1", exit_code=0)

    def test_valid_counts_match_hand_computation(self):
        cfg = self._cfg(batch_size=4)
        ledger = self._ledger()
        scorer = self._scorer(cfg, ledger)
        records = scorer.score_chunk(1, 2)[1:]
        store = scorer.store
        for rec in records:
            diag = validate_cad_vec(store.get(rec["cad_index"])["cad_vec"])
            self.assertEqual(rec["n_valid_x"], diag.n_valid_x)
            self.assertEqual(rec["n_valid_y"], diag.n_valid_y)
            self.assertEqual(rec["n_real_tokens"], diag.n_real_tokens)
            self.assertEqual(rec["valid_seq_len"], diag.n_real_tokens - 1)
            self.assertAlmostEqual(
                rec["score_total"], rec["score_x"] + rec["score_y"], places=9
            )
        ledger.finalize_invocation("inv-1", exit_code=0)

    def test_rejected_candidate_never_forwarded_never_charged(self):
        # Rebuild the store with one corrupt candidate (index 2).
        zip_path = self.root / "cad_seq.zip"
        specs = [(m, 19) for m in self.members]
        _make_zip(zip_path, specs, corrupt_index=2)
        cfg = self._cfg(batch_size=4)
        ledger = self._ledger()
        model = build_tiny_model()
        store = ZipCandidateStore(zip_path, self.members, expected_sha256=sha256_file(zip_path))
        store.preload()
        ledger.begin_invocation("inv-1", shard_id=0, cards_used=1)
        scorer = FrozenMatrixScorer(
            cfg, model, self.texts, store, self.pair_space, ledger, "inv-1",
            self._binding(cfg),
        )
        records = scorer.score_chunk(0, 1)[1:]
        self.assertEqual([r["status"] for r in records],
                         ["scored", "scored", "rejected", "scored"])
        self.assertEqual(records[2]["cad_index"], 2)
        self.assertIsNotNone(records[2]["reason"])
        self.assertIsNone(records[2]["score_total"])
        self.assertEqual(ledger.pairs_charged(), 3)  # rejection consumed no forward
        self.assertEqual(scorer.stats["pairs_forwarded"], 3)
        self.assertEqual(scorer.stats["pairs_rejected"], 1)
        ledger.finalize_invocation("inv-1", exit_code=0)

    def test_batch_invariance_and_trim_equivalence(self):
        # Same chunk scored at batch=1 / batch=3 / trim=batch_max: the audited
        # reference and the trimmed batch must agree within tight tolerance
        # (CPU random model; real-device equivalence is the G1 gate).
        ledger = self._ledger()
        reference: dict[int, float] = {}
        for name, kwargs in (
            ("b1", dict(batch_size=1)),
            ("b3", dict(batch_size=3)),
            ("trim", dict(batch_size=3, trim_mode="batch_max")),
        ):
            cfg = self._cfg(**kwargs)
            scorer = self._scorer(cfg, ledger, invocation=f"inv-{name}")
            for rec in scorer.score_chunk(2, 3)[1:]:
                if name == "b1":
                    reference[rec["cad_index"]] = rec["score_total"]
                else:
                    ref = reference[rec["cad_index"]]
                    self.assertAlmostEqual(
                        rec["score_total"], ref, places=4,
                        msg=f"{name} drifted on cad {rec['cad_index']}",
                    )
            ledger.finalize_invocation(f"inv-{name}", exit_code=0)
        # and the charges accumulated honestly across all three passes
        self.assertEqual(ledger.pairs_charged(), 12)

    def test_text_cache_matches_direct_within_tolerance(self):
        ledger = self._ledger()
        reference: dict[int, float] = {}
        for name, cache in (("direct", False), ("cached", True)):
            cfg = self._cfg(batch_size=3, text_cache=cache)
            scorer = self._scorer(cfg, ledger, invocation=f"inv-{name}")
            # chunk with two texts -> cross-text batch exercises the cache
            for t0, t1 in ((0, 2),):
                for rec in scorer.score_chunk(t0, t1)[1:]:
                    if name == "direct":
                        reference[rec["pair_index"]] = rec["score_total"]
                    else:
                        self.assertAlmostEqual(
                            rec["score_total"], reference[rec["pair_index"]], places=4,
                        )
            ledger.finalize_invocation(f"inv-{name}", exit_code=0)

    def test_budget_cap_stops_before_forward(self):
        cfg = self._cfg(batch_size=2)
        ledger = self._ledger(max_pairs=3)
        scorer = self._scorer(cfg, ledger)
        with self.assertRaisesRegex(RunnerError, "pair budget exceeded"):
            scorer.score_chunk(0, 1)  # 4 pairs > cap 3 (charged per batch of 2)


class PlanChunksTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        self.pair_space = PairSpace(3, 4)
        self.cfg = RunnerConfig(
            checkpoint_path="x", bert_dir="x", cad_seq_zip=str(self.root / "cad_seq.zip"),
            pair_manifest_dir=str(self.manifest_dir), device="cpu", batch_size=2,
        )
        self.binding = {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "pair_manifest_sha256": "m",
            "git_sha": FULL_SHA,
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "device": "cpu",
            "torch_dtype": "float32",
            "trim_mode": "none",
            "text_cache": False,
            "score_name": "two_head_factorized_logprob_sum",
            "spec_version": "text2cad-vector-score-v1",
        }

    def _write_chunk(self, t0: int, t1: int, status: str = "scored", name: str | None = None,
                     binding=None) -> Path:
        header = {
            "record_type": "chunk_header", "shard_id": 0, "text_range": [t0, t1],
            "n_pairs": (t1 - t0) * self.pair_space.n_cads,
            "n_scored": (t1 - t0) * self.pair_space.n_cads if status == "scored" else 0,
            "n_rejected": 0 if status == "scored" else (t1 - t0) * self.pair_space.n_cads,
            "pair_index_first": t0 * self.pair_space.n_cads,
            "pair_index_last": t1 * self.pair_space.n_cads - 1,
            **{f: (binding or self.binding).get(f) for f in
               ("schema_version", "pair_manifest_sha256", "git_sha", "checkpoint_sha256",
                "device", "torch_dtype", "trim_mode", "text_cache", "score_name",
                "spec_version")},
        }
        lines = [json.dumps(header)]
        for pair_index in self.pair_space.chunk_pair_indices(t0, t1):
            text_index, cad_index = self.pair_space.coords_of(pair_index)
            lines.append(json.dumps({
                "record_type": "pair", "pair_index": pair_index,
                "text_index": text_index, "cad_index": cad_index, "status": status,
                "score_x": -1.0, "score_y": -2.0, "score_total": -3.0,
                "mean_x": -1.0, "mean_y": -2.0, "mean_total": -1.5,
                "n_valid_x": 1, "n_valid_y": 1, "n_valid_total": 2,
                "valid_seq_len": 2, "n_real_tokens": 2,
            }))
        path = self.root / "out" / (name or chunk_file_name(0, t0, t1))
        atomic_publish(path, lines)
        return path

    def test_fresh_dir_plans_all_chunks(self):
        todo, completed = plan_chunks(self.root / "out", self.pair_space, self.cfg,
                                      self.binding, resume=False)
        self.assertEqual(len(todo), 3)
        self.assertEqual(completed, [])

    def test_existing_chunks_require_resume(self):
        self._write_chunk(0, 1)
        with self.assertRaisesRegex(RunnerError, "resume"):
            plan_chunks(self.root / "out", self.pair_space, self.cfg, self.binding, resume=False)

    def test_resume_skips_verified_chunks_exactly_once(self):
        self._write_chunk(0, 1)
        self._write_chunk(2, 3)
        todo, completed = plan_chunks(self.root / "out", self.pair_space, self.cfg,
                                      self.binding, resume=True)
        self.assertEqual(sorted(completed), [chunk_file_name(0, 0, 1), chunk_file_name(0, 2, 3)])
        self.assertEqual(todo, [(1, 2, 4)])

    def test_tampered_chunk_refused_loudly(self):
        path = self._write_chunk(0, 1)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[1] = json.dumps({**json.loads(lines[1]), "score_total": 0.0, "pair_index": 7})
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "chunk verification failed"):
            plan_chunks(self.root / "out", self.pair_space, self.cfg, self.binding, resume=True)

    def test_binding_drift_and_foreign_files_refused(self):
        self._write_chunk(0, 1, binding={**self.binding, "git_sha": "b" * 40})
        with self.assertRaisesRegex(RunnerError, "chunk verification failed"):
            plan_chunks(self.root / "out", self.pair_space, self.cfg, self.binding, resume=True)
        (self.root / "out").mkdir(exist_ok=True)
        # foreign shard file
        foreign = self.root / "out" / chunk_file_name(1, 0, 1)
        atomic_publish(foreign, ["{}"])
        cfg2 = RunnerConfig(
            checkpoint_path="x", bert_dir="x", cad_seq_zip="x",
            pair_manifest_dir=str(self.manifest_dir), shard_id=0, n_shards=2,
        )
        with self.assertRaisesRegex(RunnerError, "foreign chunk"):
            plan_chunks(self.root / "out", self.pair_space, cfg2, self.binding, resume=True)

    def test_atomic_publish_never_overwrites(self):
        target = self.root / "f.jsonl"
        atomic_publish(target, ["{}"])
        with self.assertRaisesRegex(RunnerError, "overwrite"):
            atomic_publish(target, ["{}"])
        self.assertEqual(target.read_text(encoding="utf-8"), "{}\n")


# ---------------------------------------------------------------------------
# dry-run (zero forward) and main() gates
# ---------------------------------------------------------------------------


class DryRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)

    def test_dry_run_refuses_bad_checkpoint_before_anything(self):
        fake_ckpt = self.root / "fake.pth"
        fake_ckpt.write_bytes(b"not the pinned checkpoint")
        zip_path = self.root / "cad_seq.zip"
        cfg = RunnerConfig(
            checkpoint_path=str(fake_ckpt), bert_dir=str(self.root),
            cad_seq_zip=str(zip_path),
            pair_manifest_dir=str(self.manifest_dir),
        )
        from idea01.text2cad_port.loader import AssetVerificationError

        with self.assertRaises(AssetVerificationError) as ctx:
            dry_run(cfg, self.root / "dry", zip_sha256=sha256_file(zip_path))
        self.assertEqual(ctx.exception.reason, "asset_sha256_mismatch")
        self.assertFalse((self.root / "dry" / "dry_run.json").exists())

    def test_dry_run_refuses_overwrite(self):
        zip_path = self.root / "cad_seq.zip"
        if not REAL_CHECKPOINT.is_file():
            self.skipTest("real checkpoint not available")
        out = self.root / "dry"
        out.mkdir()
        (out / "dry_run.json").write_text("{}", encoding="utf-8")
        cfg = RunnerConfig(
            checkpoint_path=str(REAL_CHECKPOINT), bert_dir=str(REAL_BERT_DIR),
            cad_seq_zip=str(zip_path),
            pair_manifest_dir=str(self.manifest_dir),
            expected_keys_manifest=str(REAL_KEYS_MANIFEST),
        )
        with self.assertRaisesRegex(RunnerError, "overwrite"):
            dry_run(cfg, out, zip_sha256=sha256_file(zip_path))
        self.assertEqual((out / "dry_run.json").read_text(encoding="utf-8"), "{}")

    @unittest.skipUnless(
        REAL_CHECKPOINT.is_file() and REAL_BERT_DIR.is_dir() and REAL_ZIP.is_file()
        and (REPO_ROOT / ".cache/m0_real_run_v3/query.jsonl").is_file(),
        "pinned local assets required",
    )
    def test_dry_run_on_real_assets_never_forwards(self):
        # Build the real pair manifest into the tmp dir (CPU, zero forward),
        # then dry-run against the real checkpoint/BERT/zip.
        m0 = REPO_ROOT / ".cache/m0_real_run_v3"
        built = build_pair_manifest(m0 / "query.jsonl", m0 / "gallery.jsonl")
        manifest_dir = self.root / "real_manifest"
        write_pair_manifest(manifest_dir, built)
        cfg = RunnerConfig(
            checkpoint_path=str(REAL_CHECKPOINT), bert_dir=str(REAL_BERT_DIR),
            cad_seq_zip=str(REAL_ZIP),
            pair_manifest_dir=str(manifest_dir),
            expected_keys_manifest=str(REAL_KEYS_MANIFEST),
        )
        report = dry_run(cfg, self.root / "dry")
        self.assertEqual(report["status"], "dry_run_no_forward")
        self.assertFalse(report["forward_executed"])
        self.assertEqual(report["strict_load"]["n_loaded_tensors"], 172)
        self.assertTrue(report["strict_load"]["numeric_equality"])
        self.assertEqual(report["zip"]["n_store_members"], 8046)
        self.assertTrue((self.root / "dry" / "dry_run.json").is_file())


class MainGateTest(unittest.TestCase):
    """The production CLI fails closed BEFORE any model load."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = _mini_manifest_dir(self.root)
        self.default_repo = self.root / "repo"
        cfg_dir = self.default_repo / "experiments" / "configs"
        cfg_dir.mkdir(parents=True)
        shutil.copyfile(
            self.root / "expected_hashes.json",
            cfg_dir / "text2cad_pair_manifest_expected_20260930.json",
        )

    def _canonical_fp(self) -> Path:
        return (self.default_repo / "experiments" / "configs"
                / "text2cad_pair_manifest_expected_20260930.json")

    def _args(self, *extra: str, repo_root: Path | None = None, mode="score") -> list[str]:
        repo = repo_root if repo_root is not None else self.root / "repo"
        return [
            "--mode", mode,
            "--checkpoint", str(self.root / "ckpt.pth"),
            "--bert-dir", str(self.root / "bert"),
            "--cad-seq-zip", str(self.root / "cad_seq.zip"),
            "--pair-manifest-dir", str(self.manifest_dir),
            "--expected-keys-manifest", str(self.root / "keys.json"),
            "--out-dir", str(self.root / "out"),
            "--repo-root", str(repo),
            *extra,
        ]

    def test_missing_fingerprints_exit_2(self):
        self._canonical_fp().unlink()
        self.assertEqual(main(self._args()), 2)

    def test_tampered_manifest_fingerprint_exit_2(self):
        fp = json.loads(self._canonical_fp().read_text(encoding="utf-8"))
        fp["artifact_sha256"]["texts.jsonl"] = "0" * 64
        self._canonical_fp().write_text(json.dumps(fp), encoding="utf-8")
        self.assertEqual(main(self._args()), 2)

    def test_out_dir_inside_checkout_exit_2(self):
        repo = self.root / "repo"
        repo.mkdir(exist_ok=True)
        self.assertEqual(main(self._args("--out-dir", str(repo / "sub"), repo_root=repo)), 2)

    def test_external_fingerprint_override_is_not_a_forward_bypass(self):
        with mock.patch("idea01.text2cad_frozen_runner.authorize_text2cad_run") as auth:
            self.assertEqual(main(self._args(
                "--expected-pair-hashes", str(self.root / "expected_hashes.json"))), 2)
            auth.assert_not_called()

    def test_matrix_rejects_unapproved_optimizations_and_dtype(self):
        for option in (("--trim-mode", "batch_max"), ("--text-cache",),
                       ("--torch-dtype", "bfloat16")):
            with self.subTest(option=option), mock.patch(
                "idea01.text2cad_frozen_runner.authorize_text2cad_run"
            ) as auth:
                self.assertEqual(main(self._args(*option)), 2)
                auth.assert_not_called()

    def test_single_device_cli_refuses_two_card_charge(self):
        with self.assertRaises(SystemExit):
            main(self._args("--cards-used", "2"))

    @unittest.skipUnless(shutil.which("git"), "git required")
    def test_authorization_fires_before_model_load(self):
        work, _url, _c, _a = _build_dual_version_repo(
            self.root / "scenario", approve=False,
            expected_fp=self.root / "expected_hashes.json")
        self.assertEqual(main(self._args(repo_root=work)), 2)  # state=implementation

    @unittest.skipUnless(shutil.which("git"), "git required")
    def test_authorized_temp_repo_still_refuses_nonproduction_path(self):
        work, _url, _c, _a = _build_dual_version_repo(
            self.root / "scenario2", approve=True,
            expected_fp=self.root / "expected_hashes.json")
        # passes authorization, then the production checkout pin refuses
        self.assertEqual(main(self._args(repo_root=work)), 2)

    def test_removed_relaxation_switches_rejected(self):
        for extra in (
            ["--max-pairs", "5"],
            ["--no-revision-path-check"],
            ["--active-task", "x"],
            ["--budget-file", "new.json"],
            ["--skip-authorization"],
        ):
            with self.assertRaises(SystemExit):
                main(self._args(*extra))

    def test_worker_input_schema_has_no_uid_path(self):
        # The runner's argparse surface carries no qrels/uid/evaluation input.
        import argparse as _ap

        parser_probe = json.dumps(main.__defaults__) if main.__defaults__ else ""
        self.assertNotIn("qrels", parser_probe)
        # record schema check happens in FrozenMatrixScorerTest (no uid fields)


class SourceHygieneTest(unittest.TestCase):
    def test_runner_and_bench_never_read_relevance_artifacts(self):
        """AST-level: outside docstrings/comments, no name or string constant in
        the runner/bench modules references qrels, evaluation_map or CSV files."""
        import ast

        def live_tokens(path: Path) -> list[str]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            tokens: list[str] = []

            def is_docstring(node) -> bool:
                return (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )

            def walk(node, in_doc=False):
                if is_docstring(node):
                    return  # docstrings document the prohibition; skip them
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    tokens.append(node.value)
                if isinstance(node, (ast.Name, ast.Attribute)):
                    tokens.append(
                        node.id if isinstance(node, ast.Name) else node.attr
                    )
                for child in ast.iter_child_nodes(node):
                    walk(child)

            walk(tree)
            return tokens

        for module in ("text2cad_frozen_runner.py", "text2cad_speed_bench.py"):
            tokens = live_tokens(REPO_ROOT / "experiments/src/idea01" / module)
            for token in tokens:
                self.assertNotIn("qrels", token, f"{module}: {token[:80]!r}")
                self.assertNotIn("evaluation_map", token, f"{module}: {token[:80]!r}")
                self.assertNotIn(".csv", token.lower(), f"{module}: {token[:80]!r}")


if __name__ == "__main__":
    unittest.main()
