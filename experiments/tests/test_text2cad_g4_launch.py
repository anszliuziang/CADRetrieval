"""CPU tests for the Text2CAD G4 full-matrix launch controller (2026-10-02).

Contract: experiments/plans/IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md sec.5 and
experiments/ACTIVE_TASK.md (implementation; GPU budget 0). Every test is pure
CPU: no model, no BERT, no NPU, no network. Covers:

  * the fixed full-matrix plan bound to the REAL producer math
    (text2cad_pair_index.PairSpace): shard ranges [0,1822)/[1822,3644), union
    covers all 3,644 texts with empty intersection, 1,822 chunks per shard,
    14,659,812 pairs / 916,466 batches per shard (8,046 = 502x16 + 14 tail),
    totals 29,319,624 pairs / 1,832,932 batches;
  * the FIXED scorer argv: exact flag set of the audited G3 template minus
    probe flags, exactly three per-card differences (device/shard/out-dir),
    no --probe-selection/--token-audit/--resume/--text-cache/
    --scan-candidates, revision-pinned asset paths;
  * the child environment: pinned offline/thread overrides, cache dirs
    outside the checkout, ASCEND_RT_VISIBLE_DEVICES removed and recorded;
  * the read-only npu-smi preflight parser against the REAL G3 evidence
    output, plus process-row/unhealthy/single-chip/format-drift refusals;
  * Git/authorization gates on REAL temporary repositories (no injected
    fakes): pass, HEAD mismatch, dirty worktree, attached HEAD, non-pilot
    state, wrong task_id/run_id/max_pairs/card_hours, and the read-only
    git whitelist itself; plus a live fail-closed check of the real project
    repository in its current (non-pilot, attached) state;
  * fresh-run refusals: existing run root / card dirs / shared ledger / stale
    ledger lock; second controller invocation refuses and spawns nothing;
  * the launch engine with REAL short-lived subprocesses (fake scorers with
    real exit codes): two-card success, one-card failure recorded promptly
    while the peer succeeds, real timeout termination under the common
    deadline, spawn failure of card1 keeping card0, and preflight/disk
    refusals leaving no residue;
  * POST-SPAWN FAULT SUPERVISION (R1, Root probe repro): an injected write
    failure for card0.pid.txt or controller_launch.json AFTER a successful
    Popen no longer orphans the live children, never writes a fake exit,
    reaps both real exit codes, writes controller_exit.json and ends
    "failed"; event-log / final-Git / exit-record failures are likewise
    contained (children reaped first, non-zero honest outcome); R2 adds the
    Root repro where ONLY the final controller_exit event append fails --
    the returned AND persisted status must both be "failed" with the error
    visible in both;
  * PROCESS-GROUP LIFECYCLE (R1/R2): deadline TERM/KILL addresses the child's
    whole process group (pgid == pid, injected POSIX-semantics mock on this
    Windows host -- NOT claimed as Linux evidence) and the post-exit drain
    terminates surviving group members; REAL parent+descendant trees run
    both the normal-parent-exit and deadline paths, and the REAL group /
    descendant PID liveness (live vs zombie distinguished) is collected and
    asserted AT controller return, BEFORE the test's own finally cleanup --
    the cleanup can never launder the controller's drain evidence; POSIX
    asserts real group cleanup, Windows asserts the recorded PID-only
    degradation (and that the leaked descendant is really still alive at
    controller return, i.e. the degradation is not vacuous);
  * DESCENDANT DRAIN vs COMMON DEADLINE (R2, Root repro): drain settle /
    TERM / KILL are scheduled INSIDE the original common deadline -- a
    parent reaped at deadline-1 gets its group TERM'd at deadline-1, not at
    deadline+9; the settle window shrinks/skips near the deadline; a drain
    starting past the deadline signals immediately and records the breach
    (failed); deadline+kill_wait bounds only the post-KILL confirmation
    wait; a non-ESRCH group-probe fault is recorded as probe_error (failed)
    instead of being claimed empty; launch-level fake-clock integration
    pins the near-deadline parent exit, the early-exit/peer-alive case and
    the probe-error case;
  * the common-deadline arithmetic with a deterministic fake clock: the
    staggered second card does NOT extend the wall clock, TERM/KILL grace
    sits inside the budget, per-card exits are reaped with their real codes;
  * the REAL producer binding: the controller's exact fixed argv (only path
    values adapted to the local real worker manifest and a temporary
    non-approving repo) drives the REAL audited scorer CLI through every
    pre-authorization gate (committed fingerprints vs the REAL frozen worker
    four files, fixed-arg pins) and stops exactly at the expected
    authorization refusal -- zero forward, zero model load;
  * AST stability: the seven audited scorer functions/classes
    (assemble_batch, forward_and_score_batch, score_pair_token_diagnostic,
    load_gated_model_and_store, SharedBudgetLedger, authorize_text2cad_run,
    ZipCandidateStore) are AST-identical to the reviewed G3 commit
    487d02b9ac92f5e2338718e45b1f023c308dd332;
  * source hygiene: the controller never references qrels/evaluation
    linkage/CSV outside docstrings, and imports torch-free;
  * constant cross-checks against the audited modules (task id, revisions,
    pair dimensions, ledger path formula, seven-day wall).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_g4_launch as g4  # noqa: E402
from idea01.cadmium_runner import PRODUCTION_BUDGET_ROOT, RunnerError  # noqa: E402
from idea01.text2cad_pair_index import (  # noqa: E402  (REAL producer math)
    EXPECTED_N_GALLERY,
    EXPECTED_N_PAIRS,
    EXPECTED_N_UNIQUE_TEXT,
    PairSpace,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "experiments" / "src"
G3_REVIEWED_SHA = "487d02b9ac92f5e2338718e45b1f023c308dd332"
LOCAL_WORKER_DIR = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/pair_manifest_v4_worker")
G3_PREFLIGHT_EVIDENCE = (
    REPO_ROOT / "experiments/runs/idea01_text2cad_g3_probe_20261002/pre_launch_git_npu.txt"
)

GIT_AVAILABLE = shutil.which("git") is not None


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


def _task_md(
    state="pilot_authorized",
    approved=None,
    max_pairs=g4.G4_MAX_PAIRS,
    card_hours="336",
    task_id=g4.TASK_ID,
    run_id=g4.G4_RUN_ID,
) -> str:
    approved_part = f"`{approved}`" if approved else "无"
    return (
        "# 当前实验任务\n"
        f"- `state`: `{state}`（说明）\n"
        f"- `task_id`: `{task_id}`\n"
        f"- `approved_sha`: {approved_part}；说明\n"
        f"- `gpu_budget`: {card_hours} 卡小时；`max_scored_pairs`: {max_pairs}；说明\n"
        f"- `run_id`: `{run_id}`（说明）\n"
    )


def _build_dual_version_repo(base: Path, *, approve: bool = True, **task_kwargs):
    """Real temporary dual-version repo (same pattern as the audited runner
    tests): code commit, then a separate approval commit on origin/master."""
    src = base / "src"
    src.mkdir(parents=True)
    _git(src, "init", "-q")
    _git(src, "symbolic-ref", "HEAD", "refs/heads/master")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "t")
    (src / "experiments").mkdir()
    (src / "code.txt").write_text("controller code v1\n", encoding="utf-8")
    (src / "experiments" / "ACTIVE_TASK.md").write_text(
        _task_md(state="implementation", approved=None, max_pairs=0, card_hours="0"),
        encoding="utf-8",
    )
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "code")
    code_sha = _git(src, "rev-parse", "HEAD")
    if approve:
        (src / "experiments" / "ACTIVE_TASK.md").write_text(
            _task_md(approved=code_sha, **task_kwargs), encoding="utf-8"
        )
        _git(src, "add", ".")
        _git(src, "commit", "-qm", "authorize code sha")
    origin = base / "remote.git"
    _git(src, "clone", "-q", "--bare", str(src), str(origin))
    work = base / "work"
    _git(base, "clone", "-q", str(origin), str(work))
    _git(work, "checkout", "-q", "--detach", code_sha)
    origin_url = _git(work, "remote", "get-url", "origin")
    return work, origin_url, code_sha


def _make_runtime(base: Path) -> Path:
    rt = base / "runtime"
    (rt / "runs").mkdir(parents=True)
    (rt / "budgets").mkdir(parents=True)
    return rt


def _launch_cfg(work, origin_url, code_sha, rt: Path, **overrides) -> g4.LaunchConfig:
    values = {
        "approved_sha": code_sha,
        "repo_root": work,
        "runtime_root": rt,
        "expected_origin": origin_url,
        "stagger_s": 0.0,
        "poll_interval_s": 0.02,
        "wall_budget_s": 600.0,
    }
    values.update(overrides)
    return g4.LaunchConfig(**values)


FAKE_PREFLIGHT = lambda: {"chips": [{"chip": "2 Ascend910", "health": "OK"}] * 2, "n_chips": 2, "running_processes": 0, "raw_output": "fake"}  # noqa: E731
FAKE_FREE_BYTES = lambda _path: 1 << 40  # noqa: E731  (1 TiB; the low-disk refusal is tested separately)


def _real_popen_factory(scripts: dict, received: dict):
    """Spawn REAL short-lived processes as fake scorers; record what the
    controller actually passed (argv must be the pinned scorer argv)."""

    def factory(argv, **kwargs):
        shard = int(argv[argv.index("--shard-id") + 1])
        received[shard] = {"argv": list(argv), "kwargs": kwargs}
        return subprocess.Popen([sys.executable, "-c", scripts[shard]], **kwargs)

    return factory


# ---------------------------------------------------------------------------
# Fixed plan math (REAL PairSpace producer)
# ---------------------------------------------------------------------------


class PlanMathTest(unittest.TestCase):
    def test_shard_ranges_union_disjoint(self):
        space = g4.g4_pair_space()
        self.assertIsInstance(space, PairSpace)
        r0 = space.shard_text_range(2, 0)
        r1 = space.shard_text_range(2, 1)
        self.assertEqual(r0, (0, 1822))
        self.assertEqual(r1, (1822, 3644))
        covered = set(range(*r0)) | set(range(*r1))
        self.assertEqual(covered, set(range(3644)))
        self.assertEqual(set(range(*r0)) & set(range(*r1)), set())
        chunks0 = list(space.iter_chunks(2, 0, 1))
        chunks1 = list(space.iter_chunks(2, 1, 1))
        self.assertEqual(len(chunks0), 1822)
        self.assertEqual(len(chunks1), 1822)
        texts = [t0 for _, t0, _, _ in chunks0] + [t0 for _, t0, _, _ in chunks1]
        self.assertEqual(sorted(texts), list(range(3644)))

    def test_pairs_and_batches_exact(self):
        space = g4.g4_pair_space()
        self.assertEqual((space.n_texts, space.n_cads), (3644, 8046))
        self.assertEqual(space.n_pairs, 29319624)
        self.assertEqual(8046, 502 * 16 + 14)  # full batches + honest tail
        total_pairs = total_batches = 0
        for shard in (0, 1):
            plan = g4.shard_plan(space, shard)
            self.assertEqual(plan["n_chunks"], 1822)
            self.assertEqual(plan["n_pairs"], 14659812)
            self.assertEqual(plan["n_batches"], 916466)
            self.assertEqual(plan["tail_batch_size"], 14)
            self.assertEqual(plan["device"], f"npu:{shard}")
            self.assertEqual(plan["texts_per_chunk"], 1)
            self.assertEqual(plan["cards_used"], 1)
            self.assertEqual(plan["n_shards"], 2)
            total_pairs += plan["n_pairs"]
            total_batches += plan["n_batches"]
        self.assertEqual(total_pairs, 29319624)
        self.assertEqual(total_batches, 1832932)
        self.assertEqual(g4.G4_MAX_PAIRS, 29319624)

    def test_shard_pair_ranges_dense(self):
        space = g4.g4_pair_space()
        p0 = space.shard_pair_range(2, 0)
        p1 = space.shard_pair_range(2, 1)
        self.assertEqual(p0, (0, 14659812))
        self.assertEqual(p1, (14659812, 29319624))


# ---------------------------------------------------------------------------
# Fixed argv / environment
# ---------------------------------------------------------------------------


def _argv_map(argv):
    out = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            out[argv[i]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


class FixedArgvTest(unittest.TestCase):
    def test_argv_exact_both_cards(self):
        repo, rt = Path("/home/ma-user/work/CADRetrieval"), Path("/home/ma-user/work/CADRetrieval_runtime")
        for shard in (0, 1):
            argv = g4.build_card_argv(shard, repo_root=repo, runtime_root=rt)
            self.assertEqual(argv[0], g4.PRODUCTION_PYTHON)
            self.assertEqual(argv[1], str(repo / "experiments/scripts/run_text2cad_score_matrix.py"))
            flags = _argv_map(argv[2:])
            expected = {
                "--mode": "score",
                "--checkpoint": str(rt / "models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth"),
                "--bert-dir": str(rt / "models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b"),
                "--cad-seq-zip": str(rt / "datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"),
                "--pair-manifest-dir": str(rt / "pair_manifest_v4"),
                "--expected-keys-manifest": str(repo / "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"),
                "--batch-size": "16",
                "--texts-per-chunk": "1",
                "--cards-used": "1",
                "--device": f"npu:{shard}",
                "--shard-id": str(shard),
                "--n-shards": "2",
                "--out-dir": str(rt / f"runs/idea01_text2cad_g4_full_20261002/card{shard}"),
                "--repo-root": str(repo),
            }
            self.assertEqual(flags, expected)
            # no flag value may be another flag (parser shape sanity)
            for value in flags.values():
                self.assertFalse(value.startswith("--"), value)

    def test_only_three_differences_between_cards(self):
        repo, rt = Path("/repo"), Path("/rt")
        a0 = g4.build_card_argv(0, repo_root=repo, runtime_root=rt)
        a1 = g4.build_card_argv(1, repo_root=repo, runtime_root=rt)
        self.assertEqual(len(a0), len(a1))
        diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a0, a1)) if x != y]
        self.assertEqual(len(diffs), 3)
        changed = {a0[i - 1] for i, _, _ in diffs}
        self.assertEqual(changed, {"--device", "--shard-id", "--out-dir"})

    def test_no_forbidden_or_probe_flags(self):
        argv = g4.build_card_argv(1, repo_root=Path("/repo"), runtime_root=Path("/rt"))
        for flag in g4.FORBIDDEN_SCORER_FLAGS:
            self.assertNotIn(flag, argv)
        for flag in ("--trim-mode", "--torch-dtype", "--dry-run", "--token-audit"):
            self.assertNotIn(flag, argv)
        self.assertEqual(g4.FORBIDDEN_SCORER_FLAGS,
                         ("--probe-selection", "--token-audit", "--resume",
                          "--text-cache", "--scan-candidates"))

    def test_shard_validation(self):
        with self.assertRaises(RunnerError):
            g4.build_card_argv(2, repo_root=Path("/repo"), runtime_root=Path("/rt"))
        with self.assertRaises(RunnerError):
            g4.shard_plan(g4.g4_pair_space(), 5)


class ChildEnvTest(unittest.TestCase):
    def test_pinned_overrides_and_mask_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp) / "rt"
            repo = Path(tmp) / "repo"
            base = {"PATH": "/usr/bin", "ASCEND_RT_VISIBLE_DEVICES": "1", "KEEP_ME": "x"}
            env, removed = g4.build_child_env(rt, repo, base_env=base)
            self.assertEqual(removed, ["ASCEND_RT_VISIBLE_DEVICES"])
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", env)
            self.assertEqual(env["KEEP_ME"], "x")
            for key, value in g4.PINNED_ENV_OVERRIDES.items():
                self.assertEqual(env[key], value)
            self.assertEqual(env["HF_HOME"], str(rt / "hf"))
            self.assertEqual(env["XDG_CACHE_HOME"], str(rt / "xdg"))
            self.assertEqual(env["TMPDIR"], str(rt / "tmp"))

    def test_cache_dirs_inside_checkout_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            with self.assertRaises(RunnerError):
                g4.build_child_env(repo, repo, base_env={})


# ---------------------------------------------------------------------------
# npu-smi read-only preflight parser
# ---------------------------------------------------------------------------


class NpuPreflightTest(unittest.TestCase):
    SAMPLE = (
        "+----+\n"
        "| 2     Ascend910           | OK            | 162.2       41                0    / 0             |\n"
        "| 0     4                   | 0000:95:00.0  | 0           0    / 0          3102 / 65536         |\n"
        "+----+\n"
        "| 2     Ascend910           | OK            | -           40                0    / 0             |\n"
        "| 1     5                   | 0000:97:00.0  | 0           0    / 0          2873 / 65536         |\n"
        "+----+\n"
        "| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |\n"
        "+----+\n"
        "| No running processes found in NPU 2                                                            |\n"
        "+----+\n"
    )

    def test_real_g3_evidence_output_accepted(self):
        if not G3_PREFLIGHT_EVIDENCE.is_file():
            self.skipTest("G3 pre-launch evidence mirror not present")
        text = G3_PREFLIGHT_EVIDENCE.read_text(encoding="utf-8", errors="replace")
        report = g4.parse_npu_smi_health(text)
        self.assertEqual(report["n_chips"], 2)
        self.assertEqual(report["running_processes"], 0)

    def test_sample_accepted(self):
        report = g4.parse_npu_smi_health(self.SAMPLE)
        self.assertEqual(report["n_chips"], 2)

    def test_process_row_refused(self):
        busy = self.SAMPLE.replace(
            "| No running processes found in NPU 2                                                            |",
            "| 2     0                   | 3294246       | python                     | 1849                  |",
        )
        with self.assertRaisesRegex(RunnerError, "process"):
            g4.parse_npu_smi_health(busy)

    def test_busy_row_with_marker_present_still_refused(self):
        """REAL pipe-column format: a busy process row must refuse the launch
        EVEN WHEN a 'No running processes found' marker is also present in
        the mixed output (the old space-separated-triples regex missed the
        row entirely and would have ACCEPTED the busy card here)."""
        mixed = (
            "+----+\n"
            "| 2     Ascend910           | OK            | 162.2       41                0    / 0             |\n"
            "| 0     4                   | 0000:95:00.0  | 0           0    / 0          3102 / 65536         |\n"
            "+----+\n"
            "| 2     Ascend910           | OK            | -           40                0    / 0             |\n"
            "| 1     5                   | 0000:97:00.0  | 0           0    / 0          2873 / 65536         |\n"
            "+----+\n"
            "| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |\n"
            "+----+\n"
            "| 2     0                   | 3294246       | python                     | 1849                  |\n"
            "+----+\n"
            "| No running processes found in NPU 2                                                            |\n"
            "+----+\n"
        )
        with self.assertRaisesRegex(RunnerError, "running process"):
            g4.parse_npu_smi_health(mixed)

    def test_chip_detail_row_not_misread_as_process(self):
        """The second per-chip line ('| 0     4  | 0000:95:00.0 | ...') has a
        numeric pair in column 1 but a PCI bus id (not a pid) in column 2 --
        it must NOT count as a process row."""
        report = g4.parse_npu_smi_health(self.SAMPLE)
        self.assertEqual(report["running_processes"], 0)

    def test_unhealthy_refused(self):
        with self.assertRaisesRegex(RunnerError, "unhealthy"):
            g4.parse_npu_smi_health(self.SAMPLE.replace("| OK", "| Error", 1))

    def test_single_chip_refused(self):
        lines = [ln for ln in self.SAMPLE.splitlines() if "1     5" not in ln and "OK            | -" not in ln]
        with self.assertRaisesRegex(RunnerError, "chip rows"):
            g4.parse_npu_smi_health("\n".join(lines))

    def test_missing_marker_refused(self):
        with self.assertRaisesRegex(RunnerError, "format drifted"):
            g4.parse_npu_smi_health(self.SAMPLE.replace("No running processes found", "table empty"))


# ---------------------------------------------------------------------------
# Git/authorization gates (real temporary repositories)
# ---------------------------------------------------------------------------


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class GitStateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_pass_returns_source_facts(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        facts = g4.check_git_state(work, code_sha, expected_origin=origin_url)
        self.assertEqual(facts["head_sha"], code_sha)
        self.assertEqual(facts["approved_sha"], code_sha)
        self.assertTrue(facts["detached"])
        self.assertTrue(facts["worktree_clean"])
        self.assertEqual(facts["task_id"], g4.TASK_ID)
        self.assertEqual(facts["run_id"], g4.G4_RUN_ID)
        self.assertEqual(facts["max_scored_pairs"], 29319624)
        self.assertEqual(facts["gpu_budget_card_hours"], 336.0)
        self.assertEqual(facts["task_state"], "pilot_authorized")
        self.assertIn("origin/master", facts["authorization_source"])
        self.assertEqual(len(facts["active_task_blob_sha256"]), 64)
        blob = subprocess.run(
            ["git", "show", "origin/master:experiments/ACTIVE_TASK.md"],
            cwd=str(work), capture_output=True, timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        self.assertEqual(blob.returncode, 0, blob.stderr)
        self.assertEqual(
            facts["active_task_blob_sha256"],
            hashlib.sha256(blob.stdout).hexdigest(),
        )

    def test_head_mismatch_refused(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        other = "0" if code_sha[0] != "0" else "1"
        with self.assertRaisesRegex(RunnerError, "approved-sha"):
            g4.check_git_state(work, other + code_sha[1:], expected_origin=origin_url)

    def test_short_sha_refused(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        with self.assertRaisesRegex(RunnerError, "40-hex"):
            g4.validate_approved_sha(code_sha[:7])
        with self.assertRaisesRegex(RunnerError, "40-hex"):
            g4.validate_approved_sha("")
        with self.assertRaisesRegex(RunnerError, "40-hex"):
            g4.validate_approved_sha("HEAD")

    def test_dirty_worktree_refused(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        (work / "code.txt").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "dirty"):
            g4.check_git_state(work, code_sha, expected_origin=origin_url)

    def test_attached_head_refused(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        _git(work, "checkout", "-q", "-b", "movable")
        with self.assertRaisesRegex(RunnerError, "attached"):
            g4.check_git_state(work, code_sha, expected_origin=origin_url)

    def test_non_pilot_state_refused(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base, approve=False)
        with self.assertRaisesRegex(RunnerError, "pilot_authorized"):
            g4.check_git_state(work, code_sha, expected_origin=origin_url)

    def test_wrong_scope_refused(self):
        for kwargs, pattern in (
            ({"task_id": "other_task"}, "task_id"),
            ({"run_id": "idea01_text2cad_g3_probe_20261002"}, "run_id"),
            ({"max_pairs": 804600}, "max_scored_pairs"),
            ({"card_hours": "12"}, "gpu_budget"),
        ):
            with self.subTest(**kwargs):
                base = self.base / f"case_{len(list(self.base.iterdir()))}"
                base.mkdir()
                work, origin_url, code_sha = _build_dual_version_repo(base, **kwargs)
                with self.assertRaisesRegex(RunnerError, pattern):
                    g4.check_git_state(work, code_sha, expected_origin=origin_url)

    def test_wrong_origin_refused(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        with self.assertRaisesRegex(RunnerError, "origin"):
            g4.check_git_state(work, code_sha)  # production origin != temp path

    def test_readonly_git_whitelist(self):
        work, origin_url, code_sha = _build_dual_version_repo(self.base)
        for args in (("fetch", "origin"), ("checkout", "x"), ("config", "x", "y"),
                     ("reset", "--hard"), ("clean", "-fd"), ("push",), ("pull",)):
            with self.assertRaisesRegex(RunnerError, "read-only"):
                g4._git_readonly(work, *args)

    def test_real_project_repo_currently_refuses(self):
        """Live fail-closed evidence: the REAL local repo today is attached to
        master and origin/master ACTIVE_TASK is implementation/gpu 0 -- the
        controller must refuse. Read-only; no fetch, no network."""
        head = _git(REPO_ROOT, "rev-parse", "HEAD")
        with self.assertRaises(RunnerError):
            g4.check_git_state(REPO_ROOT, head)


# ---------------------------------------------------------------------------
# Fresh-run / ledger refusals
# ---------------------------------------------------------------------------


class FreshRunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.rt = _make_runtime(Path(self._tmp.name))
        self.sha = "a" * 40

    def test_clean_pass(self):
        fresh = g4.check_fresh_run(self.rt, g4.G4_RUN_ID, self.sha)
        self.assertTrue(fresh["run_root"].endswith(g4.G4_RUN_ID))

    def test_run_root_exists_refused(self):
        (self.rt / "runs" / g4.G4_RUN_ID).mkdir()
        with self.assertRaisesRegex(RunnerError, "already exists"):
            g4.check_fresh_run(self.rt, g4.G4_RUN_ID, self.sha)

    def test_card_dir_exists_refused(self):
        (self.rt / "runs" / g4.G4_RUN_ID / "card0").mkdir(parents=True)
        with self.assertRaisesRegex(RunnerError, "card0"):
            g4.check_fresh_run(self.rt, g4.G4_RUN_ID, self.sha)

    def test_ledger_exists_refused(self):
        ledger = g4.canonical_budget_path(self.rt, self.sha)
        ledger.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "ledger"):
            g4.check_fresh_run(self.rt, g4.G4_RUN_ID, self.sha)

    def test_stale_ledger_lock_refused(self):
        ledger = g4.canonical_budget_path(self.rt, self.sha)
        ledger.with_suffix(".json.lock").write_text("held", encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "lock"):
            g4.check_fresh_run(self.rt, g4.G4_RUN_ID, self.sha)

    def test_g3_ledger_name_never_collides(self):
        g3 = g4.canonical_budget_path(self.rt, G3_REVIEWED_SHA)
        g3.write_text("{}", encoding="utf-8")  # old G3 ledger stays untouched
        fresh = g4.check_fresh_run(self.rt, g4.G4_RUN_ID, self.sha)  # different approval
        self.assertIn(self.sha, fresh["budget_ledger"])

    def test_budget_path_matches_runner_formula(self):
        from idea01.text2cad_frozen_runner import canonical_budget_path as runner_path

        self.assertEqual(
            g4.canonical_budget_path(g4.PRODUCTION_RUNTIME_ROOT, self.sha),
            runner_path(self.sha),
        )
        self.assertEqual(
            g4.PRODUCTION_RUNTIME_ROOT / "budgets", PRODUCTION_BUDGET_ROOT
        )


# ---------------------------------------------------------------------------
# Launch engine: real subprocesses, temp repos, injected preflight
# ---------------------------------------------------------------------------


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class LaunchEngineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.work, self.origin_url, self.code_sha = _build_dual_version_repo(self.base)
        self.rt = _make_runtime(self.base)

    def _run_launch(self, scripts, cfg_overrides=None, **launch_kwargs):
        received = {}
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt,
                          **(cfg_overrides or {}))
        record = g4.launch(
            cfg,
            popen_fn=_real_popen_factory(scripts, received),
            preflight_fn=FAKE_PREFLIGHT,
            free_bytes_fn=FAKE_FREE_BYTES,
            **launch_kwargs,
        )
        return record, received, cfg

    def test_two_card_success_real_exit(self):
        scripts = {
            0: "import sys; print('out0'); print('err0', file=sys.stderr); sys.exit(0)",
            1: "import sys; print('out1'); print('err1', file=sys.stderr); sys.exit(0)",
        }
        record, received, cfg = self._run_launch(scripts)
        run_root = self.rt / "runs" / g4.G4_RUN_ID
        self.assertEqual(record["status"], "ok")
        for shard in (0, 1):
            # real exit code, reaped from the real process and written bare
            self.assertEqual((run_root / f"card{shard}.exit_code.txt").read_text().strip(), "0")
            self.assertEqual(record["cards"][str(shard)]["exit_code"], 0)
            self.assertFalse(record["cards"][str(shard)]["deadline_enforced"])
            # real captured stdout/stderr of the child
            self.assertIn(f"out{shard}", (run_root / f"card{shard}.stdout.log").read_text())
            self.assertIn(f"err{shard}", (run_root / f"card{shard}.stderr.log").read_text())
            # the machinery spawned EXACTLY the pinned argv
            expected_argv = g4.build_card_argv(
                shard, repo_root=self.work, runtime_root=self.rt
            )
            self.assertEqual(received[shard]["argv"], expected_argv)
            self.assertEqual(
                (run_root / f"launch_command_card{shard}.txt").read_text().strip(),
                shlex.join(expected_argv),
            )
            kw = received[shard]["kwargs"]
            self.assertEqual(kw["cwd"], str(self.work))
            self.assertTrue(kw["start_new_session"])
            self.assertEqual(kw["stdin"], subprocess.DEVNULL)
            env = kw["env"]
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", env)
            self.assertEqual(env["TMPDIR"], str(self.rt / "tmp"))
            # real PID recorded and matches the launched process
            pid = int((run_root / f"card{shard}.pid.txt").read_text().strip())
            self.assertEqual(pid, record["cards"][str(shard)]["pid"])
            self.assertGreater(pid, 0)
        launch_record = json.loads((run_root / g4.LAUNCH_RECORD_NAME).read_text())
        self.assertEqual(
            launch_record["common_deadline_epoch"] - launch_record["common_start_epoch"],
            cfg.wall_budget_s,
        )
        self.assertEqual(launch_record["git"]["approved_sha"], self.code_sha)
        self.assertEqual(launch_record["plans"]["0"]["n_pairs"], 14659812)
        exit_record = json.loads((run_root / g4.EXIT_RECORD_NAME).read_text())
        self.assertEqual(exit_record["status"], "ok")
        self.assertTrue(exit_record["post_launch_git"]["head_matches_approved"])
        events = [json.loads(ln) for ln in (run_root / g4.EVENTS_LOG_NAME).read_text().splitlines()]
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "common_start")
        self.assertEqual(kinds.count("spawn"), 2)
        self.assertEqual(kinds.count("card_exit"), 2)
        self.assertEqual(kinds[-1], "controller_exit")

    def test_one_card_fails_peer_not_misreported(self):
        scripts = {0: "import sys; sys.exit(3)", 1: "import sys; sys.exit(0)"}
        record, _, _ = self._run_launch(scripts)
        run_root = self.rt / "runs" / g4.G4_RUN_ID
        self.assertEqual(record["status"], "failed")
        self.assertEqual((run_root / "card0.exit_code.txt").read_text().strip(), "3")
        self.assertEqual((run_root / "card1.exit_code.txt").read_text().strip(), "0")
        self.assertEqual(record["cards"]["0"]["exit_code"], 3)
        self.assertEqual(record["cards"]["1"]["exit_code"], 0)
        # the failing card's exit is recorded no later than the peer's
        self.assertIsNotNone(record["cards"]["0"]["exit_epoch"])

    def test_second_invocation_refused_and_spawns_nothing(self):
        scripts = {0: "import sys; sys.exit(0)", 1: "import sys; sys.exit(0)"}
        self._run_launch(scripts)
        calls = []

        def counting_popen(argv, **kwargs):
            calls.append(argv)
            raise AssertionError("must never spawn on a refused second launch")

        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with self.assertRaisesRegex(RunnerError, "already exists"):
            g4.launch(cfg, popen_fn=counting_popen, preflight_fn=FAKE_PREFLIGHT,
                      free_bytes_fn=FAKE_FREE_BYTES)
        self.assertEqual(calls, [])

    def test_real_timeout_terminates_children(self):
        scripts = {0: "import time; time.sleep(60)", 1: "import time; time.sleep(60)"}
        record, _, cfg = self._run_launch(
            scripts,
            cfg_overrides={
                "wall_budget_s": 0.8,
                "term_grace_s": 0.2,
                "kill_wait_s": 0.3,
                "stagger_s": 0.01,
            },
        )
        self.assertEqual(record["status"], "failed")
        for shard in ("0", "1"):
            card = record["cards"][shard]
            self.assertTrue(card["deadline_enforced"])
            self.assertTrue(card["term_sent"] or card["kill_sent"])
            self.assertNotEqual(card["exit_code"], 0)
            self.assertIsNotNone(card["exit_code"])
        run_root = self.rt / "runs" / g4.G4_RUN_ID
        for shard in (0, 1):
            self.assertNotEqual(
                (run_root / f"card{shard}.exit_code.txt").read_text().strip(), "0"
            )
        used = record["ended_epoch"] - record["common_start_epoch"]
        self.assertLess(used, cfg.wall_budget_s + cfg.kill_wait_s + 5.0)

    def test_spawn_failure_card1_keeps_card0(self):
        scripts = {0: "import sys; sys.exit(0)"}
        received = {}

        def flaky_popen(argv, **kwargs):
            shard = int(argv[argv.index("--shard-id") + 1])
            received[shard] = list(argv)
            if shard == 1:
                raise OSError("device busy")
            return subprocess.Popen([sys.executable, "-c", scripts[0]], **kwargs)

        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        record = g4.launch(cfg, popen_fn=flaky_popen, preflight_fn=FAKE_PREFLIGHT,
                           free_bytes_fn=FAKE_FREE_BYTES)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["cards"]["0"]["exit_code"], 0)
        self.assertIsNone(record["cards"]["1"]["exit_code"])
        self.assertIn("device busy", record["cards"]["1"]["spawn_error"])
        run_root = self.rt / "runs" / g4.G4_RUN_ID
        self.assertEqual((run_root / "card1.exit_code.txt").read_text().strip(), "NONE")
        self.assertEqual((run_root / "card0.exit_code.txt").read_text().strip(), "0")
        self.assertEqual(sorted(received), [0, 1])  # exactly two attempts, no retry

    def test_preflight_refusal_leaves_no_residue(self):
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)

        def busy():
            raise RunnerError("npu busy")

        with self.assertRaisesRegex(RunnerError, "npu busy"):
            g4.launch(cfg, popen_fn=lambda *a, **k: None, preflight_fn=busy)
        self.assertFalse((self.rt / "runs" / g4.G4_RUN_ID).exists())

    def test_low_disk_refused_before_mkdir(self):
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with self.assertRaisesRegex(RunnerError, "bytes free"):
            g4.launch(cfg, popen_fn=lambda *a, **k: None,
                      preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=lambda p: 1)
        self.assertFalse((self.rt / "runs" / g4.G4_RUN_ID).exists())

    def test_config_validation(self):
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt,
                          wall_budget_s=5.0, term_grace_s=4.0, kill_wait_s=2.0)
        with self.assertRaisesRegex(RunnerError, "wall budget"):
            g4.launch(cfg, popen_fn=lambda *a, **k: None, preflight_fn=FAKE_PREFLIGHT)
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt,
                          approved_sha="nothex")
        with self.assertRaisesRegex(RunnerError, "40-hex"):
            g4.launch(cfg, popen_fn=lambda *a, **k: None, preflight_fn=FAKE_PREFLIGHT)


# ---------------------------------------------------------------------------
# Common-deadline arithmetic with a deterministic fake clock
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += float(dt)


class _FakeProc:
    """Scripted fake process. This fake IGNORES terminate() (a wedged
    process), so the TERM -> KILL escalation path is exercised; the real
    termination of a compliant process is covered by the real-subprocess
    timeout test."""

    def __init__(self, clock, pid, exit_at=None, exit_code=0):
        self.clock = clock
        self.pid = pid
        self.exit_at = exit_at
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        if self.killed:
            return -9
        if self.exit_at is not None and self.clock.now() >= self.exit_at:
            return self.exit_code
        return None

    def wait(self, timeout=None):
        rc = self.poll()
        return rc if rc is not None else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _fake_group_signal(procs_by_pgid: dict):
    """Injected POSIX-semantics group signal for FAKE processes (MOCK -- not
    real Linux evidence): maps pgid == pid to the fake proc, delivers TERM /
    KILL through its handles, probes with signal 0, and records the calls.
    Fake-proc tests MUST inject this: the real default would address real
    process groups that do not belong to the test."""
    calls = []

    def fn(pgid, sig):
        proc = procs_by_pgid.get(pgid)
        if proc is None:
            return False
        if sig == 0:
            return proc.poll() is None
        if sig == g4._SIGTERM:
            calls.append((pgid, sig))
            proc.terminate()
            return True
        if sig == g4._SIGKILL:
            calls.append((pgid, sig))
            proc.kill()
            return True
        return True

    fn.calls = calls
    return fn


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class CommonDeadlineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.work, self.origin_url, self.code_sha = _build_dual_version_repo(self.base)
        self.rt = _make_runtime(self.base)

    def test_staggered_second_card_gets_no_extra_wall_clock(self):
        clock = _FakeClock(1000.0)
        procs = {
            0: _FakeProc(clock, pid=4201, exit_at=1050.0, exit_code=0),
            1: _FakeProc(clock, pid=4202, exit_at=None),  # runs until killed
        }

        def fake_popen(argv, **kwargs):
            shard = int(argv[argv.index("--shard-id") + 1])
            return procs[shard]

        gsig = _fake_group_signal({4201: procs[0], 4202: procs[1]})
        cfg = _launch_cfg(
            self.work, self.origin_url, self.code_sha, self.rt,
            wall_budget_s=100.0, stagger_s=3.0, poll_interval_s=0.5,
            term_grace_s=6.0, kill_wait_s=4.0,
        )
        record = g4.launch(
            cfg, now_fn=clock.now, sleep_fn=clock.sleep,
            popen_fn=fake_popen, preflight_fn=FAKE_PREFLIGHT,
            free_bytes_fn=FAKE_FREE_BYTES, group_signal_fn=gsig,
        )
        # common start 1000 -> deadline 1100 for BOTH cards, even though
        # card1 launched at 1003 (stagger never extends the wall clock).
        self.assertEqual(record["common_start_epoch"], 1000.0)
        self.assertEqual(record["common_deadline_epoch"], 1100.0)
        self.assertEqual(record["cards"]["1"]["launch_epoch"], 1003.0)
        card0, card1 = record["cards"]["0"], record["cards"]["1"]
        self.assertEqual((card0["exit_code"], card0["deadline_enforced"]), (0, False))
        self.assertEqual(card0["exit_epoch"], 1050.0)
        # TERM at 1100-10=1090, KILL at 1100-4=1096: grace sits INSIDE the budget.
        self.assertTrue(card1["term_sent"])
        self.assertTrue(card1["kill_sent"])
        self.assertTrue(card1["deadline_enforced"])
        self.assertEqual(card1["exit_code"], -9)
        self.assertGreaterEqual(card1["exit_epoch"], 1096.0)
        self.assertLessEqual(record["ended_epoch"], 1100.0 + cfg.kill_wait_s)
        self.assertEqual(record["status"], "failed")  # honest, never spawn-success
        # the deadline signals addressed the GROUP (pgid == pid), in order
        self.assertEqual(
            gsig.calls, [(4202, g4._SIGTERM), (4202, g4._SIGKILL)]
        )
        self.assertEqual(card0["group_cleanup"], "empty")
        self.assertEqual(card1["group_cleanup"], "empty")
        events = [
            json.loads(ln)
            for ln in (self.rt / "runs" / g4.G4_RUN_ID / g4.EVENTS_LOG_NAME)
            .read_text().splitlines()
        ]
        by_event = {}
        for e in events:
            by_event.setdefault(e["event"], e)
        self.assertEqual(by_event["common_start"]["common_deadline_epoch"], 1100.0)
        self.assertEqual(by_event["common_start"]["term_at_epoch"], 1090.0)
        self.assertEqual(by_event["common_start"]["kill_at_epoch"], 1096.0)
        terms = [e for e in events if e["event"] == "term_sent"]
        kills = [e for e in events if e["event"] == "kill_sent"]
        self.assertEqual([e["shard_id"] for e in terms], [1])
        self.assertEqual([e["shard_id"] for e in kills], [1])
        self.assertEqual([e["signal_scope"] for e in terms], ["process_group"])
        self.assertEqual([e["signal_scope"] for e in kills], ["process_group"])
        exits = {e["shard_id"]: e for e in events if e["event"] == "card_exit"}
        self.assertEqual(exits[0]["exit_code"], 0)
        self.assertEqual(exits[1]["exit_code"], -9)
        self.assertTrue(exits[1]["deadline_enforced"])

    def test_both_cards_exit_before_deadline_no_enforcement(self):
        clock = _FakeClock(2000.0)
        procs = {
            0: _FakeProc(clock, pid=1, exit_at=2005.0, exit_code=0),
            1: _FakeProc(clock, pid=2, exit_at=2007.0, exit_code=0),
        }

        def fake_popen(argv, **kwargs):
            return procs[int(argv[argv.index("--shard-id") + 1])]

        gsig = _fake_group_signal({1: procs[0], 2: procs[1]})
        cfg = _launch_cfg(
            self.work, self.origin_url, self.code_sha, self.rt,
            wall_budget_s=100.0, stagger_s=1.0, poll_interval_s=0.5,
        )
        record = g4.launch(
            cfg, now_fn=clock.now, sleep_fn=clock.sleep,
            popen_fn=fake_popen, preflight_fn=FAKE_PREFLIGHT,
            free_bytes_fn=FAKE_FREE_BYTES, group_signal_fn=gsig,
        )
        self.assertEqual(record["status"], "ok")
        self.assertFalse(record["cards"]["1"]["deadline_enforced"])
        self.assertLessEqual(record["ended_epoch"], 2100.0)
        self.assertEqual(gsig.calls, [])  # no deadline signals were needed
        self.assertEqual(record["cards"]["0"]["group_cleanup"], "empty")
        self.assertEqual(record["cards"]["1"]["group_cleanup"], "empty")


# ---------------------------------------------------------------------------
# R1 Important 1: post-spawn evidence-I/O faults (Root probe reproduction)
# ---------------------------------------------------------------------------


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class PostSpawnFaultTest(unittest.TestCase):
    """Root fault-probe scenarios against the FIXED controller: an evidence
    write failure AFTER a successful Popen must never orphan the live
    children, never write a fake exit for a live process, and never
    masquerade as ok. REAL short-lived child processes; only the evidence
    writes are fault-injected."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.work, self.origin_url, self.code_sha = _build_dual_version_repo(self.base)
        self.rt = _make_runtime(self.base)
        self.run_root = self.rt / "runs" / g4.G4_RUN_ID

    def _factory(self, scripts, procs):
        def factory(argv, **kwargs):
            shard = int(argv[argv.index("--shard-id") + 1])
            proc = subprocess.Popen([sys.executable, "-c", scripts[shard]], **kwargs)
            procs.append(proc)
            return proc

        return factory

    def _launch_with_atomic_fault(self, fail_name, scripts=None):
        scripts = scripts or {0: "import sys; sys.exit(0)", 1: "import sys; sys.exit(0)"}
        procs = []
        original = g4._atomic_write_text

        def faulty(path, text):
            if Path(path).name == fail_name:
                raise OSError(f"injected evidence write failure for {fail_name}")
            return original(path, text)

        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with mock.patch.object(g4, "_atomic_write_text", side_effect=faulty):
            record = g4.launch(
                cfg, popen_fn=self._factory(scripts, procs),
                preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=FAKE_FREE_BYTES,
            )
        return record, procs

    def test_pid_write_failure_keeps_real_exit_and_peer(self):
        """Root case 1: card0.pid.txt write fails AFTER spawn. Old code wrote
        a fake NONE exit, then crashed on the real reap (exit file exists),
        killing the controller with the peer alive. Fixed: real exits reaped
        for BOTH cards, no fake exit, controller_exit written, run failed."""
        record, procs = self._launch_with_atomic_fault("card0.pid.txt")
        self.assertEqual(record["status"], "failed")
        # no fake exit for the LIVE child: its REAL exit 0 was reaped+written
        self.assertEqual(record["cards"]["0"]["exit_code"], 0)
        self.assertEqual(record["cards"]["1"]["exit_code"], 0)
        self.assertEqual((self.run_root / "card0.exit_code.txt").read_text().strip(), "0")
        self.assertEqual((self.run_root / "card1.exit_code.txt").read_text().strip(), "0")
        self.assertFalse((self.run_root / "card0.pid.txt").exists())
        self.assertIn("card0.pid.txt", record["cards"]["0"]["record_error"])
        self.assertIsNone(record["cards"]["1"]["record_error"])
        # every spawned child was reaped before the controller returned
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertEqual(len(procs), 2)
        # the final record exists and the launch record still carries the pid
        self.assertTrue((self.run_root / g4.EXIT_RECORD_NAME).exists())
        launch_record = json.loads((self.run_root / g4.LAUNCH_RECORD_NAME).read_text())
        self.assertGreater(launch_record["cards"]["0"]["pid"], 0)
        self.assertIn("card0.pid.txt", launch_record["cards"]["0"]["record_error"])
        events = [
            json.loads(ln)
            for ln in (self.run_root / g4.EVENTS_LOG_NAME).read_text().splitlines()
        ]
        spawn0 = [e for e in events if e["event"] == "spawn" and e["shard_id"] == 0][0]
        self.assertFalse(spawn0["pid_recorded"])
        self.assertEqual(events[-1]["event"], "controller_exit")

    def test_launch_record_write_failure_children_still_supervised(self):
        """Root case 2: controller_launch.json write fails with BOTH children
        already running. Old code raised past the wait loop and left two live
        children unsupervised. Fixed: both reaped with real exits, run failed."""
        record, procs = self._launch_with_atomic_fault(g4.LAUNCH_RECORD_NAME)
        self.assertEqual(record["status"], "failed")
        self.assertIn(g4.LAUNCH_RECORD_NAME, record["launch_record_error"])
        for shard in (0, 1):
            self.assertEqual(record["cards"][str(shard)]["exit_code"], 0)
            self.assertEqual(
                (self.run_root / f"card{shard}.exit_code.txt").read_text().strip(), "0"
            )
        self.assertFalse((self.run_root / g4.LAUNCH_RECORD_NAME).exists())
        self.assertTrue((self.run_root / g4.EXIT_RECORD_NAME).exists())
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertEqual(len(procs), 2)

    def test_event_log_failure_keeps_supervision(self):
        """A permanently failing event log must not abort supervision either:
        children reaped, exit files written, controller_exit written, run
        failed with the event errors remembered."""
        scripts = {0: "import sys; sys.exit(0)", 1: "import sys; sys.exit(0)"}
        procs = []
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with mock.patch.object(
            g4, "_append_event", side_effect=OSError("injected event log failure")
        ):
            record = g4.launch(
                cfg, popen_fn=self._factory(scripts, procs),
                preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=FAKE_FREE_BYTES,
            )
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["event_log_errors"])
        self.assertIn("injected event log failure", record["event_log_errors"][0])
        for shard in (0, 1):
            self.assertEqual(record["cards"][str(shard)]["exit_code"], 0)
            self.assertEqual(
                (self.run_root / f"card{shard}.exit_code.txt").read_text().strip(), "0"
            )
        self.assertTrue((self.run_root / g4.EXIT_RECORD_NAME).exists())
        self.assertTrue(all(p.poll() is not None for p in procs))

    def test_final_git_recheck_failure_still_records_exit(self):
        """The post-launch Git re-check fails AFTER the children are reaped:
        the controller still writes controller_exit.json and fails honestly."""
        scripts = {0: "import sys; sys.exit(0)", 1: "import sys; sys.exit(0)"}
        procs = []
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with mock.patch.object(
            g4, "recheck_git_state", side_effect=OSError("injected final git read failure")
        ):
            record = g4.launch(
                cfg, popen_fn=self._factory(scripts, procs),
                preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=FAKE_FREE_BYTES,
            )
        self.assertEqual(record["status"], "failed")
        self.assertIn("injected final git read failure", record["post_launch_git"]["error"])
        self.assertFalse(record["post_launch_git"]["head_matches_approved"])
        for shard in (0, 1):
            self.assertEqual(record["cards"][str(shard)]["exit_code"], 0)
        self.assertTrue((self.run_root / g4.EXIT_RECORD_NAME).exists())
        self.assertTrue(all(p.poll() is not None for p in procs))

    def test_exit_record_write_failure_raises_after_children_reaped(self):
        """Total disk fault at controller_exit.json: the write may fail, but
        ONLY after every child is reaped and every real exit recorded -- and
        the controller exits non-zero (RunnerError) instead of claiming ok."""
        scripts = {0: "import sys; sys.exit(0)", 1: "import sys; sys.exit(0)"}
        procs = []
        original = g4._atomic_write_text

        def faulty(path, text):
            if Path(path).name == g4.EXIT_RECORD_NAME:
                raise OSError("injected controller_exit write failure")
            return original(path, text)

        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with mock.patch.object(g4, "_atomic_write_text", side_effect=faulty):
            with self.assertRaisesRegex(RunnerError, "controller_exit.json"):
                g4.launch(
                    cfg, popen_fn=self._factory(scripts, procs),
                    preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=FAKE_FREE_BYTES,
                )
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertEqual(len(procs), 2)
        for shard in (0, 1):
            self.assertEqual(
                (self.run_root / f"card{shard}.exit_code.txt").read_text().strip(), "0"
            )
        self.assertFalse((self.run_root / g4.EXIT_RECORD_NAME).exists())

    def test_final_event_write_failure_fails_honestly(self):
        """R2 Root repro: two REAL CPU children exit 0 and ONLY the final
        controller_exit event append fails. Old code computed ok and persisted
        controller_exit.json BEFORE that append: it returned ok with the error
        in memory and persisted ok with an EMPTY error list. Fixed: the final
        event is appended before the record is finalized, so the returned AND
        persisted status are both "failed" and BOTH error lists carry the
        failure; the real exits and fail-if-exists semantics are unchanged."""
        scripts = {0: "import sys; sys.exit(0)", 1: "import sys; sys.exit(0)"}
        procs = []
        original = g4._append_event

        def faulty(root, record):
            if record.get("event") == "controller_exit":
                raise OSError("injected final controller_exit event failure")
            return original(root, record)

        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt)
        with mock.patch.object(g4, "_append_event", side_effect=faulty):
            record = g4.launch(
                cfg, popen_fn=self._factory(scripts, procs),
                preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=FAKE_FREE_BYTES,
            )
        # returned state: honest failure with the error visible
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["event_log_errors"])
        self.assertIn("final controller_exit event failure", record["event_log_errors"][-1])
        # persisted state: the SAME honest failure, error NOT hidden
        persisted = json.loads((self.run_root / g4.EXIT_RECORD_NAME).read_text())
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["event_log_errors"], record["event_log_errors"])
        # real exits reaped and recorded; no fake success anywhere
        for shard in (0, 1):
            self.assertEqual(record["cards"][str(shard)]["exit_code"], 0)
            self.assertEqual(
                (self.run_root / f"card{shard}.exit_code.txt").read_text().strip(), "0"
            )
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertEqual(len(procs), 2)
        # the failed final event never landed in the log; every earlier event did
        events = [
            json.loads(ln)
            for ln in (self.run_root / g4.EVENTS_LOG_NAME).read_text().splitlines()
        ]
        self.assertNotIn("controller_exit", [e["event"] for e in events])
        self.assertEqual([e["event"] for e in events].count("card_exit"), 2)


# ---------------------------------------------------------------------------
# R1 Important 2: process-group signaling + descendant drain
# ---------------------------------------------------------------------------


def _parent_descendant_scripts(base: Path, mode: str):
    """REAL parent (card0) spawning ONE REAL descendant (a 60 s sleeper that
    would write a marker only if it survived that long). The parent records
    the descendant pid so the TEST can clean up its own leaked child on
    platforms where the controller honestly degrades to PID-only signaling.
    mode "deadline": the parent sleeps 60 s (deadline path); mode "exit0":
    the parent exits 0 immediately (normal-completion path)."""
    marker = base / f"descendant_marker_{mode}.txt"
    pid_file = base / f"descendant_pid_{mode}.txt"
    descendant = (
        "import time,pathlib;"
        "time.sleep(60);"
        f"pathlib.Path({str(marker)!r}).write_text('survived controller')"
    )
    parent = (
        "import subprocess,sys,time,pathlib;"
        f"_d=subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(_d.pid));"
        + ("time.sleep(60)" if mode == "deadline" else "sys.exit(0)")
    )
    return parent, marker, pid_file


def _pid_state(pid: int) -> str:
    """Best-effort REAL liveness of a pid the TEST created: "live" / "zombie"
    / "gone" / "unknown:<why>". POSIX probes with kill(pid, 0) -- a ZOMBIE
    still answers signal 0, so /proc classifies it when present (a zombie is
    dead and must never be counted as a live leftover). Windows uses a
    read-only OpenProcess probe (os.kill(pid, 0) is not a safe probe there);
    Windows has no user-visible zombie state."""
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "gone"
        except PermissionError:
            return "unknown:perm"
        if Path("/proc").is_dir():
            try:
                rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                return "zombie" if rest[0] == "Z" else "live"
            except (OSError, IndexError):
                return "gone"  # died between the probe and the read
        return "live"  # POSIX without /proc: cannot distinguish a zombie
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return "live"
            return "gone"
        except Exception as exc:  # noqa: BLE001 -- evidence probe, never fatal
            return f"unknown:{type(exc).__name__}"
    return "unknown:platform"


def _group_members_posix(pgid: int):
    """REAL members of a process group with live/zombie state via /proc
    (Linux). Returns None when /proc is unavailable -- the killpg probe is
    then the only evidence and the limitation is recorded, not hidden."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    members = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            rest = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            state, pgrp = rest[0], int(rest[2])
        except (OSError, IndexError, ValueError):
            continue  # raced with an exit -- simply not a confirmed member
        if pgrp == pgid:
            members.append((int(entry.name), "zombie" if state == "Z" else "live"))
    return members


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class ProcessGroupTest(unittest.TestCase):
    """The controller must not leave any member of the process groups IT
    created running after it returns -- on either the deadline path or the
    normal parent-exit path. REAL parent+descendant trees run on any
    platform, and the REAL group/descendant liveness is COLLECTED AND
    ASSERTED AT CONTROLLER RETURN, before the test's own finally cleanup
    runs: the cleanup reaps only test-created PIDs and can never launder the
    controller's own drain evidence. POSIX asserts the real group cleanup
    (live vs zombie distinguished via /proc; the server-side POSIX witness
    runs the same file); Windows asserts the recorded PID-only degradation
    AND that the leaked descendant is really still alive at controller
    return -- the degradation is evidence, not a vacuous claim; the test
    then reaps its own leaked descendant itself. An injected POSIX-semantics
    mock separately pins the exact pgid/signal sequence -- mock evidence is
    never claimed as real Linux evidence."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.work, self.origin_url, self.code_sha = _build_dual_version_repo(self.base)
        self.rt = _make_runtime(self.base)

    def _kill_own_descendant(self, pid_file: Path):
        """Reap the descendant THIS test leaked (never anything else). On
        POSIX the fixed controller already drained it, so this is a no-op."""
        if not pid_file.exists():
            return
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            return
        for _ in range(40):  # up to ~2 s: TERM until the pid is gone
            try:
                os.kill(pid, g4._SIGTERM)  # Windows: TerminateProcess
            except OSError:
                return
            time.sleep(0.05)
        try:
            os.kill(pid, g4._SIGKILL)
        except OSError:
            pass

    def _run_tree(self, mode, scripts, **cfg_overrides):
        received = {}
        cfg = _launch_cfg(self.work, self.origin_url, self.code_sha, self.rt,
                          **cfg_overrides)
        record = g4.launch(
            cfg, popen_fn=_real_popen_factory(scripts, received),
            preflight_fn=FAKE_PREFLIGHT, free_bytes_fn=FAKE_FREE_BYTES,
        )
        return record

    def _observe_tree_state(self, record, pid_file: Path) -> dict:
        """Collect the REAL group/descendant state AT controller return. This
        MUST run before the finally cleanup: only evidence gathered here can
        be attributed to the controller's own drain; anything measured after
        the test killed its own leaked PIDs would be laundered."""
        card0 = record["cards"]["0"]
        observed: dict = {"card0_pid": card0["pid"]}
        if g4._PROCESS_GROUP_SUPPORTED:
            pgid = card0["pid"]
            try:
                os.killpg(pgid, 0)
                observed["group_probe"] = "members_present"
            except ProcessLookupError:
                observed["group_probe"] = "empty"
            except OSError as exc:
                observed["group_probe"] = f"unknown:{type(exc).__name__}"
            observed["group_members"] = _group_members_posix(pgid)
        if pid_file.exists():
            try:
                desc_pid = int(pid_file.read_text().strip())
            except ValueError:
                desc_pid = None
            if desc_pid is not None:
                observed["descendant_pid"] = desc_pid
                observed["descendant_state"] = _pid_state(desc_pid)
        return observed

    def _assert_no_live_tree_at_return(self, observed: dict):
        """POSIX: at controller return the group it created has NO LIVE
        member and the descendant is not alive (a transient zombie awaiting
        init's reap is dead, not a leak; it is recorded, not claimed live)."""
        members = observed.get("group_members")
        if members is not None:
            live = [pid for pid, state in members if state == "live"]
            self.assertEqual(live, [], f"controller returned with live group members: {observed}")
        else:
            # no /proc to classify: the killpg probe is the only evidence
            self.assertEqual(
                observed.get("group_probe"), "empty",
                f"cannot classify group members (no /proc) and probe is not empty: {observed}",
            )
        self.assertIn(
            observed.get("descendant_state"), ("gone", "zombie"),
            f"descendant still alive at controller return: {observed}",
        )

    def test_deadline_terminates_parent_and_descendant(self):
        parent0, marker, pid_file = _parent_descendant_scripts(self.base, "deadline")
        scripts = {0: parent0, 1: "import time; time.sleep(60)"}
        try:
            record = self._run_tree(
                "deadline", scripts,
                wall_budget_s=1.2, term_grace_s=0.3, kill_wait_s=0.3,
                stagger_s=0.0, poll_interval_s=0.02,
                descendant_settle_s=0.3, descendant_term_grace_s=0.3,
                descendant_kill_wait_s=0.5,
            )
            # BEFORE any test cleanup: the controller's own end state.
            observed = self._observe_tree_state(record, pid_file)
            if g4._PROCESS_GROUP_SUPPORTED:
                # REAL POSIX witness, measured at controller return.
                self._assert_no_live_tree_at_return(observed)
            else:
                # Windows degradation is REAL: the leaked descendant is still
                # alive at controller return -- the controller did NOT (and
                # honestly never claimed to) clean the group.
                self.assertEqual(
                    observed.get("descendant_state"), "live",
                    f"expected the leaked descendant alive at return (degradation): {observed}",
                )
        finally:
            # The TEST reaps only the PIDs it created; this is never claimed
            # as controller cleanup and runs AFTER the assertions above.
            self._kill_own_descendant(pid_file)
        card0 = record["cards"]["0"]
        self.assertEqual(record["status"], "failed")
        self.assertTrue(card0["deadline_enforced"])
        self.assertNotIn(card0["exit_code"], (0, None))
        used = record["ended_epoch"] - record["common_start_epoch"]
        self.assertLess(used, 10.0)  # nothing lingers for the 60 s sleeps
        self.assertFalse(marker.exists())
        if g4._PROCESS_GROUP_SUPPORTED:
            # TERM reached the whole group at the deadline; the descendant
            # died with its parent before the controller returned (asserted
            # above from pre-cleanup evidence).
            self.assertFalse(card0["leftover_descendants"])
            self.assertIn(card0["group_cleanup"], ("empty", "drained_within_grace"))
        else:
            # Windows degradation: PID-only scope recorded honestly; the
            # leaked descendant was reaped by the TEST in `finally`.
            self.assertFalse(record["process_group_supported"])
            self.assertEqual(card0["group_cleanup"], "unsupported_platform")
            events = [
                json.loads(ln)
                for ln in (self.rt / "runs" / g4.G4_RUN_ID / g4.EVENTS_LOG_NAME)
                .read_text().splitlines()
            ]
            scopes = [e.get("signal_scope") for e in events
                      if e["event"] in ("term_sent", "kill_sent")]
            self.assertTrue(scopes)
            self.assertEqual(set(scopes), {"pid_only"})

    def test_parent_exit0_descendant_drained_posix_or_degradation_recorded(self):
        parent0, marker, pid_file = _parent_descendant_scripts(self.base, "exit0")
        scripts = {0: parent0, 1: "import sys; sys.exit(0)"}
        try:
            record = self._run_tree(
                "exit0", scripts,
                wall_budget_s=600.0, stagger_s=0.0, poll_interval_s=0.02,
                descendant_settle_s=0.3, descendant_term_grace_s=0.4,
                descendant_kill_wait_s=0.5,
            )
            # BEFORE any test cleanup: the controller's own end state.
            observed = self._observe_tree_state(record, pid_file)
            if g4._PROCESS_GROUP_SUPPORTED:
                self._assert_no_live_tree_at_return(observed)
            else:
                self.assertEqual(
                    observed.get("descendant_state"), "live",
                    f"expected the leaked descendant alive at return (degradation): {observed}",
                )
        finally:
            self._kill_own_descendant(pid_file)
        card0 = record["cards"]["0"]
        self.assertEqual(card0["exit_code"], 0)  # the PARENT really exited 0
        self.assertFalse(marker.exists())
        if g4._PROCESS_GROUP_SUPPORTED:
            # REAL POSIX witness: the 60 s descendant outlived its parent;
            # the controller settled, TERMed the group and reported the
            # anomaly instead of returning ok with a live tree behind it.
            self.assertTrue(card0["leftover_descendants"])
            self.assertEqual(card0["group_cleanup"], "terminated")
            self.assertEqual(record["status"], "failed")
            used = record["ended_epoch"] - record["common_start_epoch"]
            self.assertLess(used, 10.0)
        else:
            # Windows degradation: the record says plainly that NO group
            # cleanup happened; the test reaped the leaked descendant.
            self.assertFalse(record["process_group_supported"])
            self.assertEqual(card0["group_cleanup"], "unsupported_platform")
            self.assertFalse(card0["leftover_descendants"])

    def test_drain_terminates_surviving_group_mock(self):
        """MOCK POSIX semantics (NOT real Linux evidence): the parent exits 0
        but its group stays non-empty; the controller must settle-probe, TERM
        the group (pgid == pid) and report leftover_descendants -> failed."""
        clock = _FakeClock(4000.0)
        procs = {
            0: _FakeProc(clock, pid=6101, exit_at=4002.0, exit_code=0),
            1: _FakeProc(clock, pid=6102, exit_at=4003.0, exit_code=0),
        }
        group_members = {6101: True}
        calls = []

        def gsig(pgid, sig):
            if pgid == 6101:
                if sig == 0:
                    return group_members[6101]
                calls.append((pgid, sig))
                if sig == g4._SIGTERM:
                    group_members[6101] = False
                return True
            if sig == 0:
                return procs[1].poll() is None
            return True

        def fake_popen(argv, **kwargs):
            return procs[int(argv[argv.index("--shard-id") + 1])]

        cfg = _launch_cfg(
            self.work, self.origin_url, self.code_sha, self.rt,
            wall_budget_s=600.0, stagger_s=0.0, poll_interval_s=0.5,
            descendant_settle_s=1.0, descendant_term_grace_s=1.0,
            descendant_kill_wait_s=1.0,
        )
        record = g4.launch(
            cfg, now_fn=clock.now, sleep_fn=clock.sleep,
            popen_fn=fake_popen, preflight_fn=FAKE_PREFLIGHT,
            free_bytes_fn=FAKE_FREE_BYTES, group_signal_fn=gsig,
        )
        card0 = record["cards"]["0"]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(card0["exit_code"], 0)  # the PARENT really exited 0
        self.assertTrue(card0["leftover_descendants"])
        self.assertEqual(card0["group_cleanup"], "terminated")
        self.assertEqual(calls, [(6101, g4._SIGTERM)])  # TERM sufficed, no KILL
        self.assertEqual(record["cards"]["1"]["group_cleanup"], "empty")


# ---------------------------------------------------------------------------
# R2 Important 1: descendant drain signals stay INSIDE the common deadline
# ---------------------------------------------------------------------------


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class DescendantDeadlineTest(unittest.TestCase):
    """R2 (Root late-parent repro + boundaries): the post-exit descendant
    drain must schedule settle/TERM/KILL INSIDE the original common deadline
    -- deadline+kill_wait_s bounds ONLY the post-KILL confirmation wait. All
    fake-clock; the group signal is an injected POSIX-semantics MOCK (never
    claimed as real Linux evidence). Covers: parent reaped at deadline-1
    (Root repro), near-deadline settle skip with compressed TERM/KILL grace,
    normal-timing full settle, a drain STARTING past the deadline (signals
    immediately + breach recorded, never re-scheduled), survivors after KILL,
    and a non-ESRCH probe fault (recorded, failed, never claimed empty)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.work, self.origin_url, self.code_sha = _build_dual_version_repo(self.base)
        self.rt = _make_runtime(self.base)

    @staticmethod
    def _card(pid=31415):
        return g4._CardState(shard_id=0, argv=[], env={}, proc=object(), pid=pid, exit_code=0)

    def _drain(self, card, sig, clock, *, settle=10.0, grace=5.0, wait=5.0,
               deadline=1600.0, confirm=1630.0):
        g4._drain_card_group(
            card, group_signal_fn=sig, now_fn=clock.now, sleep_fn=clock.sleep,
            settle_s=settle, term_grace_s=grace, kill_wait_s=wait,
            deadline=deadline, confirm_end=confirm, events=lambda x: None,
        )

    def test_late_parent_drain_term_inside_deadline(self):
        """ROOT REPRO (regression): parent reaped at 1599, deadline 1600. Old
        code passed hard_end=deadline+30 and TERMed the real group at 1609 --
        9 s past the common deadline. Fixed: settle is skipped (its cutoff
        deadline-(grace+wait)=1590 already passed) and TERM goes out at 1599,
        INSIDE the deadline; the +30 is not a grace for running descendants."""
        clock = _FakeClock(1599.0)
        alive = [True]
        calls = []

        def sig(pgid, s):
            if s == 0:
                return alive[0]
            calls.append({"signal": s, "at": clock.now()})
            alive[0] = False
            return True

        card = self._card()
        self._drain(card, sig, clock)
        self.assertEqual(calls, [{"signal": g4._SIGTERM, "at": 1599.0}])
        self.assertLessEqual(calls[0]["at"], 1600.0)  # NEVER deadline+9 again
        self.assertEqual(card.group_cleanup, "terminated")
        self.assertTrue(card.leftover_descendants)
        self.assertFalse(card.group_deadline_breach)  # 1599 < deadline: inside
        self.assertIsNone(card.group_probe_error)

    def test_drain_near_deadline_settle_skipped_kill_bounded(self):
        """Parent reaped at deadline-6 with a TERM-ignoring group: the settle
        window is SKIPPED (not scheduled to now+10), TERM goes immediately,
        its grace is compressed to the kill cutoff (deadline-5), and KILL is
        sent there -- every signal inside the original deadline."""
        clock = _FakeClock(1594.0)  # deadline 1600; settle cutoff 1590 passed
        alive = [True]
        calls = []

        def sig(pgid, s):
            if s == 0:
                return alive[0]
            calls.append({"signal": s, "at": clock.now()})
            if s == g4._SIGKILL:
                alive[0] = False
            return True

        card = self._card()
        self._drain(card, sig, clock)
        self.assertEqual(
            calls,
            [{"signal": g4._SIGTERM, "at": 1594.0}, {"signal": g4._SIGKILL, "at": 1595.0}],
        )
        self.assertTrue(all(c["at"] <= 1600.0 for c in calls))
        self.assertEqual(card.group_cleanup, "killed")
        self.assertFalse(card.group_deadline_breach)

    def test_drain_normal_timing_full_grace_drained(self):
        """Far from the deadline the full settle window applies and a group
        draining on its own inside it is NOT an anomaly (unchanged R1
        semantics: drained_within_grace, no signals, no leftover)."""
        clock = _FakeClock(1500.0)
        alive = [True]
        calls = []

        def sig(pgid, s):
            if s == 0:
                return alive[0] and clock.now() < 1504.0
            calls.append({"signal": s, "at": clock.now()})
            return True

        card = self._card()
        self._drain(card, sig, clock)
        self.assertEqual(calls, [])
        self.assertEqual(card.group_cleanup, "drained_within_grace")
        self.assertFalse(card.leftover_descendants)
        self.assertFalse(card.group_deadline_breach)

    def test_drain_starting_after_deadline_signals_immediately_breach(self):
        """Pathological late start (e.g. a parent reaped only at the hard
        accounting bound): the remaining TERM/KILL go out IMMEDIATELY
        back-to-back -- never re-scheduled further out -- the cleanup still
        happens (a live job is never abandoned for a prettier log), and every
        post-deadline signal is recorded as a breach so the run can never
        claim the common budget held."""
        clock = _FakeClock(1631.0)  # deadline 1600, confirm_end 1630 passed
        alive = [True]
        calls = []

        def sig(pgid, s):
            if s == 0:
                return alive[0]
            calls.append({"signal": s, "at": clock.now()})
            if s == g4._SIGKILL:
                alive[0] = False
            return True

        card = self._card()
        self._drain(card, sig, clock)
        self.assertEqual(
            calls,
            [{"signal": g4._SIGTERM, "at": 1631.0}, {"signal": g4._SIGKILL, "at": 1631.0}],
        )
        self.assertEqual(card.group_cleanup, "killed")
        self.assertTrue(card.leftover_descendants)
        self.assertTrue(card.group_deadline_breach)  # honest: budget did NOT hold

    def test_drain_after_deadline_survivors_recorded(self):
        """Even an unkillable (e.g. D-state) group past the deadline is
        signaled immediately and the survivor state is recorded honestly --
        the controller never hangs and never claims empty."""
        clock = _FakeClock(1631.0)
        calls = []

        def sig(pgid, s):
            if s == 0:
                return True  # never dies
            calls.append({"signal": s, "at": clock.now()})
            return True

        card = self._card()
        self._drain(card, sig, clock)
        self.assertEqual(
            calls,
            [{"signal": g4._SIGTERM, "at": 1631.0}, {"signal": g4._SIGKILL, "at": 1631.0}],
        )
        self.assertEqual(card.group_cleanup, "survivors_after_kill")
        self.assertTrue(card.group_deadline_breach)

    def test_drain_probe_error_recorded_not_empty(self):
        """Minor (R2): a non-ESRCH probe fault (EPERM) is recorded as a probe
        error -- the group is NOT claimed empty, no signals are sent on an
        unconfirmable group, and the card carries the error for ok=False."""
        clock = _FakeClock(1500.0)

        def sig(pgid, s):
            if s == 0:
                raise PermissionError("operation not permitted")
            raise AssertionError("must never signal an unconfirmable group")

        card = self._card()
        self._drain(card, sig, clock)
        self.assertIsNotNone(card.group_probe_error)
        self.assertIn("PermissionError", card.group_probe_error)
        self.assertTrue(card.group_cleanup.startswith("probe_error:"))
        self.assertNotEqual(card.group_cleanup, "empty")
        self.assertFalse(card.leftover_descendants)

    def _launch_with_residue(self, card0_exit_at, card1_exit_at, clock,
                             *, wall=100.0, term_grace=0.4, kill_wait=0.4,
                             settle=10.0, grace=5.0, wait=5.0):
        """launch() with fake clock/procs; card0's GROUP keeps one surviving
        member until the drain TERMs it; card1's group mirrors its own proc."""
        procs = {
            0: _FakeProc(clock, pid=4201, exit_at=card0_exit_at, exit_code=0),
            1: _FakeProc(clock, pid=4202, exit_at=card1_exit_at, exit_code=0),
        }
        residue_alive = [True]
        calls = []

        def gsig(pgid, sig):
            if pgid == 4201:
                if sig == 0:
                    return residue_alive[0]
                calls.append((pgid, sig, clock.now()))
                if sig == g4._SIGTERM:
                    residue_alive[0] = False
                return True
            if sig == 0:
                return procs[1].poll() is None
            return True

        def fake_popen(argv, **kwargs):
            return procs[int(argv[argv.index("--shard-id") + 1])]

        cfg = _launch_cfg(
            self.work, self.origin_url, self.code_sha, self.rt,
            wall_budget_s=wall, stagger_s=0.0, poll_interval_s=0.5,
            term_grace_s=term_grace, kill_wait_s=kill_wait,
            descendant_settle_s=settle, descendant_term_grace_s=grace,
            descendant_kill_wait_s=wait,
        )
        record = g4.launch(
            cfg, now_fn=clock.now, sleep_fn=clock.sleep,
            popen_fn=fake_popen, preflight_fn=FAKE_PREFLIGHT,
            free_bytes_fn=FAKE_FREE_BYTES, group_signal_fn=gsig,
        )
        return record, calls

    def test_end_to_end_parent_reaped_at_deadline_minus_1(self):
        """Launch-level Root scenario: BOTH parents exit naturally at
        deadline-2/deadline-1 and the last-reaped card's group has residue.
        The drain TERMs it at deadline-1 -- inside the budget -- and the run
        fails honestly (leftover), without deadline enforcement on the
        parents and without any breach."""
        clock = _FakeClock(1000.0)  # deadline 1100; term_at 1099.2, kill_at 1099.6
        record, calls = self._launch_with_residue(1099.0, 1098.5, clock)
        card0, card1 = record["cards"]["0"], record["cards"]["1"]
        self.assertEqual((card0["exit_code"], card1["exit_code"]), (0, 0))
        self.assertFalse(card0["deadline_enforced"] or card1["deadline_enforced"])
        self.assertEqual(calls, [(4201, g4._SIGTERM, 1099.0)])
        self.assertLessEqual(calls[0][2], record["common_deadline_epoch"])
        self.assertTrue(card0["leftover_descendants"])
        self.assertFalse(card0["group_deadline_breach"])
        self.assertEqual(card0["group_cleanup"], "terminated")
        self.assertEqual(record["status"], "failed")
        self.assertLessEqual(record["ended_epoch"], record["common_deadline_epoch"] + 0.4)
        events = [
            json.loads(ln)
            for ln in (self.rt / "runs" / g4.G4_RUN_ID / g4.EVENTS_LOG_NAME)
            .read_text().splitlines()
        ]
        term_events = [e for e in events if e["event"] == "descendant_term_sent"]
        self.assertEqual(len(term_events), 1)
        self.assertLessEqual(term_events[0]["time_epoch"], record["common_deadline_epoch"])
        self.assertFalse(term_events[0]["after_deadline"])

    def test_end_to_end_early_exit_peer_alive_not_delayed(self):
        """Card0 exits EARLY (t=1050) leaving residue while card1 keeps its
        authorized run to t=1089: card1's reap/supervision is never delayed
        by card0's leftover (drain runs only after every parent is reaped),
        and the delayed drain still TERMs the residue inside the deadline."""
        clock = _FakeClock(1000.0)  # deadline 1100; drain settle capped at 1090
        record, calls = self._launch_with_residue(
            1050.0, 1089.0, clock, term_grace=6.0, kill_wait=4.0,
        )
        card0, card1 = record["cards"]["0"], record["cards"]["1"]
        self.assertEqual(card0["exit_epoch"], 1050.0)  # recorded immediately
        self.assertEqual(card1["exit_epoch"], 1089.0)  # peer reap NOT delayed
        self.assertFalse(card1["deadline_enforced"])
        self.assertEqual(calls, [(4201, g4._SIGTERM, 1090.0)])
        self.assertLessEqual(calls[0][2], record["common_deadline_epoch"])
        self.assertTrue(card0["leftover_descendants"])
        self.assertFalse(card0["group_deadline_breach"])
        self.assertEqual(record["status"], "failed")
        events = [
            json.loads(ln)
            for ln in (self.rt / "runs" / g4.G4_RUN_ID / g4.EVENTS_LOG_NAME)
            .read_text().splitlines()
        ]
        kinds = [e["event"] for e in events]
        exits = [i for i, e in enumerate(events) if e["event"] == "card_exit"]
        self.assertEqual(events[exits[0]]["shard_id"], 0)
        self.assertEqual(events[exits[1]]["shard_id"], 1)
        # the drain happened only AFTER both parents were reaped
        self.assertLess(exits[1], kinds.index("descendants_detected"))

    def test_end_to_end_probe_error_fails_run(self):
        """Launch-level Minor: a non-ESRCH group-probe fault on card0 is
        recorded per card, persists into controller_exit.json and fails the
        run -- never misreported as an empty group."""
        clock = _FakeClock(4000.0)
        procs = {
            0: _FakeProc(clock, pid=6101, exit_at=4002.0, exit_code=0),
            1: _FakeProc(clock, pid=6102, exit_at=4003.0, exit_code=0),
        }

        def gsig(pgid, sig):
            if pgid == 6101:
                if sig == 0:
                    raise PermissionError("operation not permitted")
                raise AssertionError("must never signal an unconfirmable group")
            if sig == 0:
                return procs[1].poll() is None
            return True

        def fake_popen(argv, **kwargs):
            return procs[int(argv[argv.index("--shard-id") + 1])]

        cfg = _launch_cfg(
            self.work, self.origin_url, self.code_sha, self.rt,
            wall_budget_s=600.0, stagger_s=0.0, poll_interval_s=0.5,
        )
        record = g4.launch(
            cfg, now_fn=clock.now, sleep_fn=clock.sleep,
            popen_fn=fake_popen, preflight_fn=FAKE_PREFLIGHT,
            free_bytes_fn=FAKE_FREE_BYTES, group_signal_fn=gsig,
        )
        card0, card1 = record["cards"]["0"], record["cards"]["1"]
        self.assertEqual(record["status"], "failed")
        self.assertIn("PermissionError", card0["group_probe_error"])
        self.assertTrue(card0["group_cleanup"].startswith("probe_error:"))
        self.assertFalse(card0["leftover_descendants"])
        self.assertEqual(card1["group_cleanup"], "empty")
        persisted = json.loads(
            (self.rt / "runs" / g4.G4_RUN_ID / g4.EXIT_RECORD_NAME).read_text()
        )
        self.assertEqual(persisted["status"], "failed")
        self.assertIn("PermissionError", persisted["cards"]["0"]["group_probe_error"])


# ---------------------------------------------------------------------------
# REAL producer binding: the fixed argv drives the real audited scorer CLI
# through every pre-authorization gate against the REAL frozen worker files
# ---------------------------------------------------------------------------


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
@unittest.skipUnless(LOCAL_WORKER_DIR.is_dir(), "local frozen worker manifest not present")
class ProducerBindingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.work, self.origin_url, self.code_sha = _build_dual_version_repo(self.base)
        cfg_dir = self.work / "experiments" / "configs"
        cfg_dir.mkdir(parents=True)
        shutil.copyfile(
            REPO_ROOT / "experiments/configs/text2cad_pair_manifest_expected_20260930.json",
            cfg_dir / "text2cad_pair_manifest_expected_20260930.json",
        )

    def _adapted_argv(self, shard: int, out_dir: Path) -> list:
        """The controller's EXACT fixed scorer arguments with ONLY path values
        adapted to this host (repo root, worker manifest dir, out dir); every
        flag and non-path value is untouched. Returns the CLI argument list
        (what follows the script name) for the runner's ``main(argv)``."""
        argv = g4.build_card_argv(
            shard,
            repo_root=g4.PRODUCTION_REPO_ROOT,
            runtime_root=g4.PRODUCTION_RUNTIME_ROOT,
        )
        flags = _argv_map(argv[2:])
        original = dict(flags)
        flags["--repo-root"] = str(self.work)
        flags["--pair-manifest-dir"] = str(LOCAL_WORKER_DIR)
        flags["--out-dir"] = str(out_dir)
        changed = {k for k in flags if flags[k] != original[k]}
        self.assertEqual(changed, {"--repo-root", "--pair-manifest-dir", "--out-dir"})
        adapted: list = []
        for flag, value in flags.items():
            adapted += [flag, value]
        return adapted

    def test_fixed_argv_reaches_authorization_gate_on_real_producer(self):
        from idea01.text2cad_frozen_runner import main as runner_main

        for shard in (0, 1):
            with self.subTest(shard=shard):
                out_dir = self.base / f"out{shard}"
                argv = self._adapted_argv(shard, out_dir)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    rc = runner_main(argv)
                self.assertEqual(rc, 2)
                text = stderr.getvalue()
                # Gate 1 (REAL worker files vs committed reviewed fingerprints,
                # fixed-arg pins, two-card pin) passed; the run stops exactly at
                # the fail-closed authorization gate (on this host the temporary
                # repo's origin path is not the production remote -- in
                # production the same gate refuses a non-pilot ACTIVE_TASK).
                self.assertIn("authorization", text)
                self.assertIn("refusing any forward", text)
                for bad in ("unrecognized arguments", "two-card matrix",
                            "forward mode requires", "expected-keys",
                            "reviewed fingerprint", "sha256 drift",
                            "asset_sha256_mismatch"):
                    self.assertNotIn(bad, text)

    def test_real_worker_counts_match_fixed_plan(self):
        from idea01.text2cad_pair_index import load_pair_manifest

        manifest = load_pair_manifest(LOCAL_WORKER_DIR)
        counts = manifest["counts"]
        self.assertEqual(counts["n_unique_text"], g4.G4_N_TEXTS)
        self.assertEqual(counts["n_gallery"], g4.G4_N_CADS)
        self.assertEqual(counts["n_pairs"], g4.G4_MAX_PAIRS)


# ---------------------------------------------------------------------------
# AST stability of the seven audited scorer definitions (reviewed G3 commit)
# ---------------------------------------------------------------------------


@unittest.skipUnless(GIT_AVAILABLE, "git executable required")
class AstStabilityTest(unittest.TestCase):
    SEVEN = (
        "assemble_batch",
        "forward_and_score_batch",
        "score_pair_token_diagnostic",
        "load_gated_model_and_store",
        "SharedBudgetLedger",
        "authorize_text2cad_run",
        "ZipCandidateStore",
    )
    REL = "experiments/src/idea01/text2cad_frozen_runner.py"

    @staticmethod
    def _def_dumps(source: str) -> dict:
        tree = ast.parse(source)
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in AstStabilityTest.SEVEN:
                    out[node.name] = ast.dump(node)
        return out

    def test_seven_defs_ast_identical_to_reviewed_g3_commit(self):
        current = (REPO_ROOT / self.REL).read_text(encoding="utf-8")
        proc = subprocess.run(
            ["git", "show", f"{G3_REVIEWED_SHA}:{self.REL}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        cur, old = self._def_dumps(current), self._def_dumps(proc.stdout)
        self.assertEqual(set(cur), set(self.SEVEN))
        self.assertEqual(set(old), set(self.SEVEN))
        for name in self.SEVEN:
            self.assertEqual(cur[name], old[name], f"{name} drifted from {G3_REVIEWED_SHA}")


# ---------------------------------------------------------------------------
# Constant cross-checks against the audited modules
# ---------------------------------------------------------------------------


class ConstantCrossCheckTest(unittest.TestCase):
    def test_duplicated_constants_match_audited_modules(self):
        from idea01 import text2cad_frozen_runner as runner
        from idea01.text2cad_bench_protocol import SEVEN_DAY_SECONDS, SHARD_BY_DEVICE
        from idea01.text2cad_g3_probe import G3_BATCH_SIZE

        self.assertEqual(g4.TASK_ID, runner.TASK_ID)
        self.assertEqual(g4.EXPECTED_MODEL_REVISION, runner.EXPECTED_MODEL_REVISION)
        self.assertEqual(g4.EXPECTED_BERT_REVISION, runner.EXPECTED_BERT_REVISION)
        self.assertEqual(g4.EXPECTED_DATA_REVISION, runner.EXPECTED_DATA_REVISION)
        self.assertEqual(g4.EXPECTED_KEYS_MANIFEST_REL, runner.EXPECTED_KEYS_MANIFEST_REL)
        self.assertEqual(g4.G4_N_TEXTS, EXPECTED_N_UNIQUE_TEXT)
        self.assertEqual(g4.G4_N_CADS, EXPECTED_N_GALLERY)
        self.assertEqual(g4.G4_MAX_PAIRS, EXPECTED_N_PAIRS)
        self.assertEqual(g4.G4_BATCH_SIZE, G3_BATCH_SIZE)  # both pin 16 (G2 evidence)
        self.assertEqual(g4.G4_WALL_BUDGET_S, float(SEVEN_DAY_SECONDS))
        for shard in (0, 1):
            self.assertEqual(SHARD_BY_DEVICE[f"npu:{shard}"], shard)
        self.assertEqual(g4.G4_CARD_HOURS * 3600.0, 1209600.0)


# ---------------------------------------------------------------------------
# Source hygiene + import surface
# ---------------------------------------------------------------------------


class SourceHygieneTest(unittest.TestCase):
    def test_controller_never_references_relevance_artifacts(self):
        def live_tokens(path: Path) -> list:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            tokens = []

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
            return tokens

        for rel in (
            "experiments/src/idea01/text2cad_g4_launch.py",
            "experiments/scripts/run_text2cad_g4_launch.py",
        ):
            for token in live_tokens(REPO_ROOT / rel):
                self.assertNotIn("qrels", token, f"{rel}: {token[:80]!r}")
                self.assertNotIn("evaluation_map", token, f"{rel}: {token[:80]!r}")
                self.assertNotIn(".csv", token.lower(), f"{rel}: {token[:80]!r}")

    def test_controller_imports_torch_free(self):
        code = (
            "import sys; sys.path.insert(0, {src!r}); "
            "import idea01.text2cad_g4_launch; "
            "print('torch' in sys.modules)"
        ).format(src=str(SRC_DIR))
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "False")

    def test_controller_has_no_scorer_passthrough(self):
        # The argparse surface carries exactly three options; no extra-args
        # passthrough (``parse_known_args``/``REMAINDER``) exists.
        source = (REPO_ROOT / "experiments/src/idea01/text2cad_g4_launch.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("parse_known_args", source)
        self.assertNotIn("REMAINDER", source)


# ---------------------------------------------------------------------------
# CLI argument validation (production path pins; launch itself tested above)
# ---------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    def test_missing_approved_sha_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            g4.main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_bad_sha_format_exits_2(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = g4.main(["--approved-sha", "abc123"])
        self.assertEqual(rc, 2)
        self.assertIn("40-hex", stderr.getvalue())

    def test_nonproduction_paths_refused(self):
        sha = "b" * 40
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = g4.main(["--approved-sha", sha, "--repo-root", "/tmp/elsewhere"])
        self.assertEqual(rc, 2)
        self.assertIn("production checkout", stderr.getvalue())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = g4.main(["--approved-sha", sha, "--runtime-root", "/tmp/elsewhere"])
        self.assertEqual(rc, 2)
        self.assertIn("production runtime root", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
