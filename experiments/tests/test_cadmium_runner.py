"""CPU tests for cadmium_runner review fixes of 2026-09-30 (review items 1-3)
and the 2026-09-30 review addendum (authorization SOURCE).

Covers, without torch / weights / NPU / network:

  * item 1 (asset integrity): the SHA256 manifest is REQUIRED, must cover
    every file the load actually reads (base config, safetensors index and
    all shards / single file, adapter config + exactly one weight file +
    tokenizer files), hashes must match, and load-influencing small files
    present on disk may not exist unmanifested. The pinned-revision path
    check has no production bypass.
  * item 2 (run authorization): the approved SHA / pair cap / derived wall
    clock are bound to the persisted BudgetLedger, which survives restarts
    and refuses cross-approval reuse.
  * addendum (authorization SOURCE): the authorization text is read from the
    freshly fetched ``origin/master:experiments/ACTIVE_TASK.md`` (origin URL
    validated first) -- never from the detached HEAD's working tree, because
    a commit cannot contain its own approved_sha. REAL temporary Git
    repositories (no injected fake task files or fake HEADs) prove the
    approval commit SHA differs from the approved code SHA, that a stale or
    self-authored working-tree file cannot authorize, and that an attached
    HEAD, a dirty tree, a wrong origin URL or a HEAD != approved_sha all
    refuse. Field-level parsing (pilot_authorized, full 40-hex approved_sha,
    positive pair/card-hour budgets) is checked as a pure text function with
    no Git involved. Git remotes in these tests are local paths only.
  * item 3 (input isolation): the pairs JSONL must carry exactly the two
    string fields per line; UID/qrels/annotation/extra/duplicate keys or
    non-string values exit non-zero before any other gate.

Runs on CPU only; never imports torch, never touches the network or SSH.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.cadmium_runner import (  # noqa: E402
    EXPECTED_ADAPTER_REVISION,
    EXPECTED_BASE_REVISION,
    BudgetLedger,
    RunnerConfig,
    RunnerError,
    _atomic_write_new,
    authorize_run,
    load_pair_requests,
    main,
    parse_active_task_authorization,
    preflight_assets,
    remap_cadmium_adapter_keys,
)

FULL_SHA = "a" * 40
OTHER_SHA = "b" * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_asset_tree(root: Path) -> tuple[Path, Path, dict]:
    """Minimal fake base/adapter layout plus a complete, correct manifest."""
    base = root / "base"
    adapter = root / "adapter"
    manifest: dict[str, dict[str, str]] = {"base": {}, "adapter": {}}

    def put(section: str, dirpath: Path, rel: str, data: bytes) -> None:
        path = dirpath / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[section][rel] = _sha256(data)

    put("base", base, "config.json", json.dumps({"vocab_size": 999, "model_type": "qwen2"}).encode())
    put(
        "base",
        base,
        "model.safetensors.index.json",
        json.dumps(
            {"weight_map": {"w1": "model-00001-of-00002.safetensors", "w2": "model-00002-of-00002.safetensors"}}
        ).encode(),
    )
    put("base", base, "model-00001-of-00002.safetensors", b"shard-one")
    put("base", base, "model-00002-of-00002.safetensors", b"shard-two")
    put("adapter", adapter, "adapter_config.json", json.dumps({"peft_type": "LORA", "r": 64}).encode())
    put("adapter", adapter, "tokenizer_config.json", b"{}")
    put("adapter", adapter, "tokenizer.json", b"{}")
    put("adapter", adapter, "adapter_model.safetensors", b"adapter-weights")
    return base, adapter, manifest


def _cfg(base: Path, adapter: Path, manifest_path, **overrides) -> RunnerConfig:
    kwargs = dict(
        base_model_path=str(base),
        adapter_path=str(adapter),
        system_message_config="unused-system-message.json",  # preflight never reads it
        require_revision_in_path=False,  # test-only entry; the CLI has no such switch
        asset_manifest=str(manifest_path) if manifest_path else None,
    )
    kwargs.update(overrides)
    return RunnerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Item 1: preflight asset-integrity closure
# ---------------------------------------------------------------------------

class PreflightAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.base, self.adapter, self.manifest = _build_asset_tree(self.root)
        self.manifest_path = self.root / "manifest.json"
        self._write_manifest(self.manifest)

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _preflight(self, **overrides):
        return preflight_assets(_cfg(self.base, self.adapter, self.manifest_path, **overrides))

    def test_complete_manifest_passes(self):
        record = self._preflight()
        self.assertEqual(record["manifest_checked_files"], 8)
        self.assertEqual(
            record["base_weight_shards"],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )
        self.assertEqual(record["adapter_weight_file"], "adapter_model.safetensors")
        self.assertEqual(record["asset_manifest_sha256"], _sha256(self.manifest_path.read_bytes()))

    def test_manifest_is_required(self):
        with self.assertRaisesRegex(RunnerError, "REQUIRED"):
            self._preflight(asset_manifest=None)

    def test_sha_mismatch_refused(self):
        self.manifest["base"]["model-00001-of-00002.safetensors"] = "0" * 64
        self._write_manifest(self.manifest)
        with self.assertRaisesRegex(RunnerError, "SHA256 mismatch"):
            self._preflight()

    def test_uncovered_weight_shard_refused(self):
        del self.manifest["base"]["model-00002-of-00002.safetensors"]
        self._write_manifest(self.manifest)
        with self.assertRaisesRegex(RunnerError, "does not cover"):
            self._preflight()

    def test_uncovered_adapter_weight_refused(self):
        del self.manifest["adapter"]["adapter_model.safetensors"]
        self._write_manifest(self.manifest)
        with self.assertRaisesRegex(RunnerError, "does not cover"):
            self._preflight()

    def test_manifested_file_missing_on_disk_refused(self):
        (self.base / "model-00001-of-00002.safetensors").unlink()
        with self.assertRaisesRegex(RunnerError, "missing"):
            self._preflight()

    def test_unmanifested_load_influencing_file_refused(self):
        # A stray tokenizer override the load would silently pick up.
        (self.adapter / "special_tokens_map.json").write_bytes(b"{}")
        with self.assertRaisesRegex(RunnerError, "silently influence"):
            self._preflight()

    def test_manifested_conditional_file_passes(self):
        data = b'{"do_sample": false}'
        (self.base / "generation_config.json").write_bytes(data)
        self.manifest["base"]["generation_config.json"] = _sha256(data)
        self._write_manifest(self.manifest)
        record = self._preflight()
        self.assertEqual(record["manifest_checked_files"], 9)

    def test_ambiguous_adapter_weights_refused(self):
        (self.adapter / "adapter_model.bin").write_bytes(b"bin-weights")
        with self.assertRaisesRegex(RunnerError, "ambiguous adapter weights"):
            self._preflight()

    def test_single_file_base_layout_passes(self):
        for rel in (
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ):
            (self.base / rel).unlink()
            del self.manifest["base"][rel]
        data = b"single-shard-weights"
        (self.base / "model.safetensors").write_bytes(data)
        self.manifest["base"]["model.safetensors"] = _sha256(data)
        self._write_manifest(self.manifest)
        record = self._preflight()
        self.assertEqual(record["base_weight_shards"], ["model.safetensors"])
        self.assertEqual(record["manifest_checked_files"], 6)

    def test_manifest_shape_errors_refused(self):
        for bad in (
            {"base": {}},  # missing/empty sections
            {"base": {"config.json": "xyz"}, "adapter": {"tokenizer.json": "0" * 64}},  # bad sha
            {"base": {"../escape": "0" * 64}, "adapter": {"tokenizer.json": "0" * 64}},  # traversal
        ):
            self._write_manifest(bad)
            with self.assertRaises(RunnerError):
                self._preflight()

    def test_revision_path_check_refuses_plain_paths(self):
        with self.assertRaisesRegex(RunnerError, "pinned revision"):
            self._preflight(require_revision_in_path=True)

    def test_revision_path_check_passes_with_pinned_layout(self):
        pinned = self.root / f"models/{EXPECTED_BASE_REVISION}_{EXPECTED_ADAPTER_REVISION}"
        base2, adapter2, manifest2 = _build_asset_tree(pinned)
        mp2 = self.root / "manifest2.json"
        mp2.write_text(json.dumps(manifest2), encoding="utf-8")
        record = preflight_assets(_cfg(base2, adapter2, mp2, require_revision_in_path=True))
        self.assertEqual(record["manifest_checked_files"], 8)


# ---------------------------------------------------------------------------
# Item 2: run authorization gate + ledger binding
# ---------------------------------------------------------------------------

def _task_md(
    state: str = "pilot_authorized",
    approved: str | None = FULL_SHA,
    gpu_budget: str = "2",
    max_pairs: str = "10",
) -> str:
    approved_part = f"`{approved}`" if approved else "无"
    return (
        "# 当前实验任务\n"
        f"- `state`: `{state}`（说明文字）\n"
        "- `task_id`: `idea01_text2cad_benchmark_20260928`\n"
        f"- `approved_sha`: {approved_part}；其余说明\n"
        f"- `gpu_budget`: {gpu_budget} 卡小时；`max_scored_pairs`: {max_pairs}；不得启动任何新的模型前向评分\n"
    )


class ParseActiveTaskAuthorizationTest(unittest.TestCase):
    """Pure text-field checks of the authorization text -- fail closed, no Git.

    The authorization-SOURCE problem (2026-09-30 addendum) is solved
    structurally: authorize_run obtains this text ONLY from the freshly
    fetched origin/master blob. These tests pin the field-level refusals.
    """

    def test_happy_path_fields(self):
        f = parse_active_task_authorization(_task_md())
        self.assertEqual(f["task_state"], "pilot_authorized")
        self.assertEqual(f["task_id"], "idea01_text2cad_benchmark_20260928")
        self.assertEqual(f["approved_sha"], FULL_SHA)
        self.assertEqual(f["max_pairs"], 10)
        self.assertEqual(f["card_hours"], 2.0)

    def test_state_not_pilot_authorized(self):
        with self.assertRaisesRegex(RunnerError, "pilot_authorized"):
            parse_active_task_authorization(_task_md(state="implementation"))

    def test_missing_state_field(self):
        with self.assertRaisesRegex(RunnerError, "pilot_authorized"):
            parse_active_task_authorization("# no state here\n")

    def test_placeholder_approved_sha(self):
        with self.assertRaisesRegex(RunnerError, "approved_sha"):
            parse_active_task_authorization(_task_md(approved=None))

    def test_short_approved_sha(self):
        with self.assertRaisesRegex(RunnerError, "approved_sha"):
            parse_active_task_authorization(_task_md(approved="abcdef1"))

    def test_zero_pair_budget(self):
        with self.assertRaisesRegex(RunnerError, "max_scored_pairs"):
            parse_active_task_authorization(_task_md(max_pairs="0"))

    def test_zero_card_hours(self):
        with self.assertRaisesRegex(RunnerError, "gpu_budget"):
            parse_active_task_authorization(_task_md(gpu_budget="0"))


# -- real temporary Git repositories (2026-09-30 addendum) -------------------
# The pre-addendum tests injected a fake ACTIVE_TASK.md path plus fake git
# HEAD/status callables, which masked the self-reference timing problem (a
# commit cannot contain its own approved_sha). These tests instead build
# REAL repositories with a local-path "origin" remote -- no network, no
# injected git fakes -- and exercise authorize_run end to end.


def _git(cwd, *args):
    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {cwd}: {r.stderr.strip()}")
    return r.stdout.strip()


def _build_dual_version_repo(base: Path, approve: bool = True):
    """Build the real dual-version Git scenario; returns
    (work, origin_url, code_sha, approval_sha_or_None).

    * ``src`` (branch master): commit 1 holds code plus an ACTIVE_TASK.md
      whose state is ``implementation`` with placeholder approved_sha and
      zero budgets -- the real project state as of 2026-09-30 -> CODE_SHA.
      If ``approve``, commit 2 rewrites ACTIVE_TASK.md to
      ``pilot_authorized`` approving CODE_SHA with positive budgets ->
      APPROVAL_SHA (always != CODE_SHA: the approval commit names the code
      commit, never itself).
    * ``remote.git`` is a bare clone standing in for origin; ``work`` is
      cloned from it and checked out DETACHED at CODE_SHA (the production
      protocol). The working-tree ACTIVE_TASK.md in ``work`` therefore still
      says ``implementation`` even when origin/master authorizes HEAD.
    """
    src = base / "src"
    src.mkdir(parents=True)
    _git(src, "init", "-q")
    _git(src, "symbolic-ref", "HEAD", "refs/heads/master")  # portable across init.defaultBranch
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "t")
    (src / "experiments").mkdir()
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
            _task_md(approved=code_sha), encoding="utf-8"
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
class RealGitAuthorizationTest(unittest.TestCase):
    """Dual-version authorization end to end in real Git repositories."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_approval_commit_differs_from_approved_code_sha(self):
        """THE addendum proof: the origin/master approval commit SHA differs
        from the approved code SHA; the detached HEAD's working tree still
        carries the UNAPPROVED text, yet the run is authorized from the
        freshly fetched origin blob."""
        work, origin_url, code_sha, approval_sha = _build_dual_version_repo(self.base)
        self.assertNotEqual(code_sha, approval_sha)
        wt = (work / "experiments" / "ACTIVE_TASK.md").read_text(encoding="utf-8")
        self.assertIn("`state`: `implementation`", wt)  # working tree is stale...
        self.assertNotIn("pilot_authorized", wt)  # ...and can never self-authorize

        auth = authorize_run(work, expected_origin=origin_url)

        self.assertEqual(auth.task_state, "pilot_authorized")
        self.assertEqual(auth.task_id, "idea01_text2cad_benchmark_20260928")
        self.assertEqual(auth.approved_sha, code_sha)
        self.assertEqual(auth.actual_sha, code_sha)
        self.assertEqual(auth.authorization_ref_sha, approval_sha)
        self.assertNotEqual(auth.authorization_ref_sha, auth.approved_sha)
        self.assertEqual(
            auth.active_task_source,
            f"origin/master:experiments/ACTIVE_TASK.md @ {approval_sha}",
        )
        self.assertEqual(auth.origin_url, origin_url)
        self.assertEqual(auth.max_pairs, 10)
        self.assertEqual(auth.card_hours, 2.0)
        self.assertEqual(auth.max_wall_seconds, 7200.0)

    def test_pre_approval_origin_state_refuses(self):
        """Before the Codex approval commit reaches origin/master (the real
        project state as of 2026-09-30), the gate refuses."""
        work, origin_url, _code_sha, _ = _build_dual_version_repo(self.base, approve=False)
        with self.assertRaisesRegex(RunnerError, "pilot_authorized"):
            authorize_run(work, expected_origin=origin_url)

    def test_working_tree_self_authorization_refused(self):
        """Editing the detached HEAD's own ACTIVE_TASK.md to claim approval
        cannot authorize: the file is never read as an authorization source,
        and the edit leaves the worktree dirty -> refuse."""
        work, origin_url, code_sha, _ = _build_dual_version_repo(self.base)
        (work / "experiments" / "ACTIVE_TASK.md").write_text(
            _task_md(approved=code_sha), encoding="utf-8"
        )
        with self.assertRaisesRegex(RunnerError, "dirty"):
            authorize_run(work, expected_origin=origin_url)

    def test_attached_head_at_approved_sha_refused(self):
        """Even at the RIGHT commit, an attached (movable) branch refuses."""
        work, origin_url, code_sha, _ = _build_dual_version_repo(self.base)
        _git(work, "checkout", "-q", "-B", "master", code_sha)
        with self.assertRaisesRegex(RunnerError, "attached"):
            authorize_run(work, expected_origin=origin_url)

    def test_head_differs_from_approved_sha_refused(self):
        work, origin_url, code_sha, _ = _build_dual_version_repo(self.base)
        (work / "code.txt").write_text("unreviewed change\n", encoding="utf-8")
        _git(work, "config", "user.email", "t@example.com")
        _git(work, "config", "user.name", "t")
        _git(work, "add", ".")
        _git(work, "commit", "-qm", "unreviewed local change")
        self.assertNotEqual(_git(work, "rev-parse", "HEAD"), code_sha)
        with self.assertRaisesRegex(RunnerError, "!= approved_sha"):
            authorize_run(work, expected_origin=origin_url)

    def test_origin_url_mismatch_refused(self):
        work, _origin_url, _code_sha, _ = _build_dual_version_repo(self.base)
        with self.assertRaisesRegex(RunnerError, "expected"):
            authorize_run(work, expected_origin="https://example.com/attacker")

    def test_not_a_git_repo_refused(self):
        plain = self.base / "plain"
        plain.mkdir()
        with self.assertRaises(RunnerError):
            authorize_run(plain, expected_origin="https://example.com/whatever")

    def test_cards_used_divides_wall_clock(self):
        work, origin_url, _code_sha, _ = _build_dual_version_repo(self.base)
        auth = authorize_run(work, cards_used=2, expected_origin=origin_url)
        self.assertEqual(auth.max_wall_seconds, 3600.0)
        self.assertEqual(auth.cards_used, 2)


class BudgetLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger_path = Path(self._tmp.name) / "budget.json"

    def test_charge_persist_resume_and_refuse_overrun(self):
        led = BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=100.0)
        led.charge(2)
        led2 = BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=100.0)
        self.assertEqual(led2.pairs_forwarded, 2)
        self.assertEqual(led2.remaining(), 1)
        with self.assertRaises(RunnerError):
            led2.charge(2)  # 2 + 2 > 3
        self.assertEqual(led2.pairs_forwarded, 2)  # failed charge not persisted
        state = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(state["pairs_forwarded"], 2)

    def test_zero_cap_fails_closed(self):
        led = BudgetLedger(self.ledger_path, 0, approved_sha=FULL_SHA, max_wall_seconds=100.0)
        with self.assertRaises(RunnerError):
            led.charge(1)
        self.assertEqual(led.pairs_forwarded, 0)

    def test_cross_approval_sha_refused(self):
        BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=100.0).charge(1)
        with self.assertRaisesRegex(RunnerError, "approved_sha"):
            BudgetLedger(self.ledger_path, 3, approved_sha=OTHER_SHA, max_wall_seconds=100.0)

    def test_cap_change_refused(self):
        BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=100.0).charge(1)
        with self.assertRaisesRegex(RunnerError, "cap"):
            BudgetLedger(self.ledger_path, 5, approved_sha=FULL_SHA, max_wall_seconds=100.0)

    def test_wall_clock_change_refused(self):
        BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=100.0).charge(1)
        with self.assertRaisesRegex(RunnerError, "wall-clock"):
            BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=200.0)

    def test_wall_clock_survives_restart(self):
        led = BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=50.0)
        led.charge(1)  # persists started_at_epoch
        state = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        state["started_at_epoch"] = time.time() - 5000.0  # simulate long-crashed run
        self.ledger_path.write_text(json.dumps(state), encoding="utf-8")
        led2 = BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=50.0)
        with self.assertRaisesRegex(RunnerError, "wall-clock"):
            led2.check_wall_clock()

    def test_wall_clock_within_budget(self):
        led = BudgetLedger(self.ledger_path, 3, approved_sha=FULL_SHA, max_wall_seconds=5000.0)
        self.assertGreaterEqual(led.check_wall_clock(), 0.0)

    def test_in_memory_ledger_without_binding(self):
        led = BudgetLedger(None, 3)
        led.charge(3)
        self.assertEqual(led.check_wall_clock(), 0.0)
        with self.assertRaises(RunnerError):
            led.charge(1)


# ---------------------------------------------------------------------------
# Item 3: strict pair-input isolation
# ---------------------------------------------------------------------------

class LoadPairRequestsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pairs = Path(self._tmp.name) / "pairs.jsonl"

    def _write(self, text: str) -> None:
        self.pairs.write_text(text, encoding="utf-8")

    def test_valid_pairs_preserve_order_and_verbatim_values(self):
        self._write(
            '{"query_text": "a bracket", "candidate_json_desc": "{\\"a\\": 1}"}\n'
            "\n"  # blank lines skipped
            '{"candidate_json_desc": "{\\"b\\": 2}", "query_text": "a gear"}\n'
        )
        reqs = load_pair_requests(self.pairs)
        self.assertEqual(len(reqs), 2)
        self.assertEqual(reqs[0].query_text, "a bracket")
        self.assertEqual(reqs[0].candidate_json_desc, '{"a": 1}')
        self.assertEqual(reqs[1].query_text, "a gear")
        self.assertEqual(reqs[1].candidate_json_desc, '{"b": 2}')

    def test_extra_uid_key_refused(self):
        self._write('{"query_text": "q", "candidate_json_desc": "c", "uid": "0000/abcd"}\n')
        with self.assertRaisesRegex(RunnerError, "extra"):
            load_pair_requests(self.pairs)

    def test_qrels_like_key_refused(self):
        self._write('{"query_text": "q", "candidate_json_desc": "c", "qrel": 1}\n')
        with self.assertRaises(RunnerError):
            load_pair_requests(self.pairs)

    def test_missing_key_refused(self):
        self._write('{"query_text": "q"}\n')
        with self.assertRaisesRegex(RunnerError, "missing"):
            load_pair_requests(self.pairs)

    def test_non_string_value_refused(self):
        self._write('{"query_text": "q", "candidate_json_desc": 123}\n')
        with self.assertRaisesRegex(RunnerError, "strings"):
            load_pair_requests(self.pairs)

    def test_duplicate_key_refused(self):
        self._write('{"query_text": "q1", "query_text": "q2", "candidate_json_desc": "c"}\n')
        with self.assertRaisesRegex(RunnerError, "duplicate"):
            load_pair_requests(self.pairs)

    def test_non_object_line_refused(self):
        self._write('["query_text", "candidate_json_desc"]\n')
        with self.assertRaisesRegex(RunnerError, "object"):
            load_pair_requests(self.pairs)

    def test_invalid_json_refused(self):
        self._write('{not json}\n')
        with self.assertRaises(RunnerError):
            load_pair_requests(self.pairs)

    def test_missing_file_refused(self):
        with self.assertRaises(RunnerError):
            load_pair_requests(Path(self._tmp.name) / "absent.jsonl")


class MainGateTest(unittest.TestCase):
    """The production CLI must fail closed BEFORE any model import/load."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.pairs = self.root / "pairs.jsonl"

    def _args(self, *extra: str, repo_root: Path | None = None) -> list[str]:
        return [
            "--base-model-path", str(self.root / "base"),
            "--adapter-path", str(self.root / "adapter"),
            "--system-message-config", str(self.root / "sys.json"),
            "--pairs-jsonl", str(self.pairs),
            "--out-jsonl", str(self.root / "out.jsonl"),
            "--asset-manifest", str(self.root / "manifest.json"),
            "--repo-root", str(repo_root if repo_root is not None else self.repo),
            *extra,
        ]

    def _write_valid_pairs(self) -> None:
        self.pairs.write_text('{"query_text": "q", "candidate_json_desc": "c"}\n', encoding="utf-8")

    def test_bad_input_exits_2_before_any_other_gate(self):
        # No git repo, no assets, no manifest: the input gate fires first.
        self.pairs.write_text(
            '{"query_text": "q", "candidate_json_desc": "c", "uid": "x"}\n', encoding="utf-8"
        )
        self.assertEqual(main(self._args()), 2)

    @unittest.skipUnless(shutil.which("git"), "git executable required for real-repo gate test")
    def test_authorization_gate_exits_2_before_model_load(self):
        # Valid input and a REAL git checkout, but its origin is not the
        # pinned project remote and its origin/master ACTIVE_TASK.md says
        # implementation anyway (the real project state as of 2026-09-30):
        # the gate refuses BEFORE touching assets or importing torch.
        self._write_valid_pairs()
        work, _url, _code, _approval = _build_dual_version_repo(self.root / "scenario", approve=False)
        self.assertEqual(main(self._args(repo_root=work)), 2)

    def test_non_git_repo_root_fails_closed(self):
        # The checkout is not a git repo: the fetched-origin authorization
        # cannot even be attempted -> refuse.
        self._write_valid_pairs()
        self.assertEqual(main(self._args()), 2)

    def test_removed_relaxation_switches_rejected(self):
        # Review items 1-2 + 2026-09-30 addendum: no production switches to
        # loosen revision checks, hand in an arbitrary pair cap, or point the
        # authorization at a working-tree file.
        self._write_valid_pairs()
        for extra in (["--max-pairs", "5"], ["--no-revision-path-check"], ["--active-task", "x"], ["--budget-file", "new.json"]):
            with self.assertRaises(SystemExit):
                main(self._args(*extra))


class AtomicOutputTest(unittest.TestCase):
    def test_completed_file_is_published_once_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "result.jsonl"
            _atomic_write_new(target, ['{"pair_index":0}', '{"pair_index":1}'])
            original = target.read_bytes()
            self.assertEqual(original, b'{"pair_index":0}\n{"pair_index":1}\n')
            with self.assertRaisesRegex(RunnerError, "overwrite"):
                _atomic_write_new(target, ['{"pair_index":999}'])
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(list(Path(tmp).glob(".*.tmp")))


class CadmiumAdapterKeyTest(unittest.TestCase):
    def test_official_504_key_layout_maps_one_to_one(self):
        modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        source = {
            f"base_model.model.layers.{layer}.self_attn.{module}.lora_{part}.weight": 1
            for layer in range(36) for module in modules for part in ("A", "B")
        }
        mapped = remap_cadmium_adapter_keys(source)
        self.assertEqual(len(mapped), 504)
        self.assertEqual(len(set(mapped)), 504)
        self.assertIn(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", mapped
        )

    def test_missing_key_is_fatal(self):
        with self.assertRaisesRegex(RunnerError, "504"):
            remap_cadmium_adapter_keys({"base_model.model.layers.0.lora_A.weight": 1})


if __name__ == "__main__":
    unittest.main()
