"""Text2CAD G4 full-matrix two-card launch controller (2026-10-02) -- fail-closed.

Task contract: experiments/plans/IDEA01_M1C_G4_FULL_MATRIX_2026-10-02.md sec.5
(the minimal local launch-control implementation contract) and
experiments/ACTIVE_TASK.md (state ``implementation``; GPU 0; no model/BERT/NPU
forward in this round). This module is pure CPU and performs NO model forward
anywhere -- it is torch-free at import (the asset revision constants are
duplicated from the audited ``text2cad_frozen_runner`` with a unit-test
cross-check, the same pattern as ``text2cad_g3_probe``). It never fetches,
checks out, edits or otherwise modifies Git state, never writes inside the
checkout, and never touches qrels / UID linkage / the raw CSV / prior scores.

WHAT THE CONTROLLER DOES (and nothing more)
-------------------------------------------
1. Pre-launch verification, all read-only, all BEFORE any run artifact is
   created:

   * ``--approved-sha`` is REQUIRED on the CLI and must be a full 40-hex
     string. The controller NEVER guesses it, never defaults it to HEAD and
     never reads it from a movable ref; Codex fills it after code review.
   * the checkout HEAD must equal ``--approved-sha`` exactly, must be
     DETACHED (the production protocol is ``git checkout --detach <sha>``)
     and the worktree must be CLEAN (``git status --porcelain`` empty). The
     only git commands the controller may run are an explicit read-only
     whitelist (``rev-parse`` / ``symbolic-ref`` / ``status`` /
     ``remote get-url`` / ``show``) -- no fetch, no detach, no config.
   * the authorization text is read from the ALREADY-FETCHED
     ``origin/master:experiments/ACTIVE_TASK.md`` blob (origin URL validated;
     the operator fetches before launch per protocol -- the controller itself
     never fetches). The audited ``parse_active_task_authorization`` must
     accept it (``pilot_authorized`` + full 40-hex approved_sha + positive
     pair/card budgets), and the G4-fixed scope must match exactly:
     ``task_id`` == idea01_text2cad_benchmark_20260928, approved_sha == HEAD,
     ``max_scored_pairs`` == 29,319,624, ``gpu_budget`` == 336 card-hours
     (= 1,209,600 card-seconds) and ``run_id`` ==
     idea01_text2cad_g4_full_20261002. The origin ref SHA and the blob SHA256
     are recorded as the authorization-source evidence.
   * FRESH checks: the run root ``$RT/runs/<run_id>`` and the shared budget
     ledger ``$RT/budgets/text2cad_m1c_<approved_sha>.json`` (and a leftover
     ``.lock`` sidecar) must NOT exist -- the G3 run directory/ledger belong
     to a different run_id/approval and are never reused, and existing
     outputs are never overwritten.
   * a read-only ``npu-smi info`` preflight must show both chips healthy
     (``OK``) and NO running processes (other people's jobs are never
     terminated -- a busy card refuses the launch). The raw output is
     recorded.
   * the runtime partition must have >= 100 GiB free (the expected raw JSONL
     volume is ~20-25 GB per contract sec.4).

2. Commit point: the run root is created with ``os.mkdir`` (no parents), so a
   second controller invocation -- or a leftover directory -- fails loudly
   instead of overwriting or double-launching.

3. Two FIXED full-matrix scorer command lines are generated from pinned
   constants (the audited G3 template minus the probe flags): ``--mode
   score``, the pinned checkpoint/BERT/cad_seq.zip revisions, the committed
   expected-keys manifest, ``--batch-size 16 --texts-per-chunk 1
   --cards-used 1 --n-shards 2``, shard0 on ``npu:0`` / shard1 on ``npu:1``,
   each with its own ``--out-dir``. There is NO controller option that adds
   scorer arguments: ``--probe-selection``, ``--token-audit``, ``--resume``,
   ``--text-cache`` and ``--scan-candidates`` can never appear (self-checked
   at build time); dtype/trim stay at the runner's pinned float32/none
   defaults (the runner's own score-mode gate refuses anything else). The
   child environment inherits the operator session minus any device-masking
   variable (``ASCEND_RT_VISIBLE_DEVICES`` is removed and the removal is
   recorded) plus the pinned offline/cache overrides pointing OUTSIDE the
   checkout.

4. Background-persistent execution: each scorer is spawned with
   ``start_new_session=True`` (setsid on Linux -- an SSH disconnect does not
   kill the job), stdin from /dev/null, stdout/stderr to separate per-card
   log files. The real PID, launch time and exact shell-quoted command are
   written per card immediately after spawn; every event is also appended
   (fsync) to ``controller_events.jsonl``. A post-spawn evidence-write
   failure NEVER abandons or fake-exits the live child: the failure is
   remembered, the child keeps its deadline supervision, and the run ends
   ``failed`` -- a successful spawn is never reported as experimental
   success.

5. ONE common deadline: ``common_start`` is taken once before card0 spawns;
   ``deadline = common_start + 604800 s`` (the seven-day hard wall). The two
   cards may start a few seconds apart (default 3 s, avoiding the scorer
   authorization fetch lock) but share the SAME deadline -- a later card
   never gets extra wall clock. Termination grace is INSIDE the budget:
   SIGTERM at ``deadline - (term_grace + kill_wait)`` (60 s + 30 s), SIGKILL
   at ``deadline - kill_wait``, so both processes are dead by the deadline.
   On POSIX the signals go to the WHOLE process group the controller created
   (the child leads it via ``start_new_session``, pgid == pid, so no other
   job can be reached); after each parent's REAL exit is recorded, surviving
   group members are drained (settle window, TERM, KILL) with every signal
   scheduled INSIDE the original common deadline -- the settle window shrinks
   and is skipped as the deadline approaches, TERM/KILL cutoffs never move
   past it, and only the post-KILL confirmation wait may extend to
   ``deadline + kill_wait_s`` (pure accounting after the KILL was sent). A
   drain that itself starts past the deadline sends the remaining signals
   immediately and records the breach (``group_deadline_breach``); a card
   that needed that drain is an honest anomaly that fails the run. Platforms
   without process groups (Windows dev host) degrade to PID-only signaling
   and RECORD the degradation instead of claiming cleanup. Each card's real
   exit code is reaped and written to ``card{s}.exit_code.txt`` IMMEDIATELY
   when the process exits (no pipe masks it, no waiting for the peer). A
   failed card is recorded and handed over; the surviving card keeps its
   authorized run -- the controller never restarts a failed card, never
   retries, and reports ``ok`` ONLY when both cards really exited 0 without
   deadline enforcement, without leftover descendants, without a deadline
   breach or unconfirmable group, and without any evidence-record failure --
   including the final ``controller_exit`` event, whose write result is
   folded into the returned AND persisted status.

The scorer's own audited gates (fresh origin/master authorization, committed
fingerprints, strict load, shared budget ledger) remain the final word; this
controller adds pre-launch proof, fixed commands and honest lifecycle
evidence on top. CPU tests inject fake clocks / fake process factories /
temporary directories and drive REAL short-lived subprocesses for the exit
and timeout paths; no test loads a model, BERT or any NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .cadmium_runner import (
    EXPECTED_ORIGIN_URL,
    PRODUCTION_REPO_ROOT,
    RunnerError,
    parse_active_task_authorization,
)
from .p1_runner import (
    ACTIVE_TASK_BLOB,
    AUTHORIZATION_REF,
    active_task_field,
    normalize_remote_url,
)
from .text2cad_pair_index import PairIndexError, PairSpace, ensure_outside_checkout

# ---------------------------------------------------------------------------
# Pinned G4 identities (contract sec.2-4; never inferred at runtime)
# ---------------------------------------------------------------------------

#: Duplicated from the audited text2cad_frozen_runner (torch-bound); a unit
#: test cross-checks every value against the real module.
TASK_ID = "idea01_text2cad_benchmark_20260928"
EXPECTED_MODEL_REVISION = "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"
EXPECTED_BERT_REVISION = "6da4b6a26a1877e173fca3225479512db81a5e5b"
EXPECTED_DATA_REVISION = "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"
EXPECTED_KEYS_MANIFEST_REL = "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"

G4_RUN_ID = "idea01_text2cad_g4_full_20261002"
SCORER_SCRIPT_REL = "experiments/scripts/run_text2cad_score_matrix.py"
PRODUCTION_PYTHON = "/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python"
PRODUCTION_RUNTIME_ROOT = Path("/home/ma-user/work/CADRetrieval_runtime")

#: Fixed exact workload (contract sec.3): 3,644 unique texts x 8,046 CAD.
G4_N_TEXTS = 3644
G4_N_CADS = 8046
G4_N_SHARDS = 2
G4_BATCH_SIZE = 16
G4_TEXTS_PER_CHUNK = 1
G4_CARDS_USED = 1
G4_MAX_PAIRS = G4_N_TEXTS * G4_N_CADS  # 29,319,624
#: Fixed hard budgets (contract sec.4): seven-day COMMON wall clock and
#: 336 combined card-hours = 1,209,600 card-seconds (distinct from the
#: 38.28394 h conservative ESTIMATE, which is not a stop line).
G4_WALL_BUDGET_S = 604800.0
G4_CARD_HOURS = 336.0
#: Termination grace sits INSIDE the wall budget: TERM at
#: deadline - (G4_TERM_GRACE_S + G4_KILL_WAIT_S), KILL at
#: deadline - G4_KILL_WAIT_S.
G4_TERM_GRACE_S = 60.0
G4_KILL_WAIT_S = 30.0
#: The two cards may start this far apart (scorer authorization fetch lock);
#: the common deadline does not move.
G4_STAGGER_S = 3.0
G4_POLL_INTERVAL_S = 5.0
#: Descendant drain (POSIX process groups): after a card's REAL exit is
#: recorded, surviving members of the group the controller created get a
#: settle window, then TERM, then KILL. Every SIGNAL is scheduled inside the
#: ORIGINAL common deadline: the settle window ends no later than
#: deadline - (term grace + kill wait), and the TERM grace is capped at
#: deadline - kill wait, so no descendant termination is ever scheduled past
#: the deadline; only the post-KILL confirmation wait may extend to
#: deadline + kill_wait (pure accounting). A drain starting past the deadline
#: signals immediately and records the breach. The audited production scorer
#: spawns no children, so any survivor is an anomaly that fails the run
#: honestly instead of leaking past the controller.
G4_DESCENDANT_SETTLE_S = 10.0
G4_DESCENDANT_TERM_GRACE_S = 5.0
G4_DESCENDANT_KILL_WAIT_S = 5.0
#: Contract sec.4: confirm >= 100 GB free on the runtime partition before
#: launch (raw JSONL is expected at ~20-25 GB).
G4_MIN_FREE_BYTES = 100 * 1024**3

#: Device-masking variables are removed from the child environment (the
#: removal is recorded); device numbering must stay unmasked.
DEVICE_MASK_ENV_VARS = ("ASCEND_RT_VISIBLE_DEVICES",)

#: Scorer flags the full matrix must never carry (contract sec.3).
FORBIDDEN_SCORER_FLAGS = (
    "--probe-selection",
    "--token-audit",
    "--resume",
    "--text-cache",
    "--scan-candidates",
)

#: Pinned environment overrides for the scorer children (identical to the
#: audited G3 launch; cache/tmp values are filled from the runtime root).
PINNED_ENV_OVERRIDES = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
}

LAUNCH_RECORD_NAME = "controller_launch.json"
EXIT_RECORD_NAME = "controller_exit.json"
EVENTS_LOG_NAME = "controller_events.jsonl"
LAUNCH_SCHEMA_VERSION = "text2cad-g4-launch-v1"

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")

#: Read-only git subcommands the controller may run (enforced in _git_readonly;
#: the controller NEVER fetches, checks out, resets, cleans or configures).
_READ_ONLY_GIT = ("rev-parse", "symbolic-ref", "status", "show", "remote")

#: Signal numbers: ``signal.SIGKILL`` does not exist on Windows, so resolve
#: both by name at import time (the numeric values are the POSIX ABI).
_SIGTERM = int(getattr(signal, "SIGTERM", 15))
_SIGKILL = int(getattr(signal, "SIGKILL", 9))

#: POSIX process-group support. The controller spawns every card with
#: ``start_new_session=True``, so each child leads its OWN session/group
#: (pgid == pid) containing only itself and its descendants -- signaling
#: that group can never reach another job. Platforms without ``killpg``
#: (Windows dev host) degrade to PID-only signaling and record the
#: degradation honestly; they are NEVER cited as POSIX evidence.
_PROCESS_GROUP_SUPPORTED = os.name == "posix" and hasattr(os, "killpg")


def _posix_group_signal(pgid: int, sig: int) -> bool:
    """Signal the process group this controller created; False when empty.

    Only ever called with ``pgid == card.pid`` of a child this controller
    spawned with ``start_new_session=True`` (the child leads the group), so
    it cannot reach processes outside that card's own tree. ``sig == 0``
    probes group existence without delivering a signal.
    """
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


# ---------------------------------------------------------------------------
# Read-only Git verification + authorization-source binding
# ---------------------------------------------------------------------------


def _git_readonly(repo_root, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    """Run a whitelisted READ-ONLY git command; anything else refuses."""
    if not args or args[0] not in _READ_ONLY_GIT:
        raise RunnerError(
            f"controller git command {args!r} is not in the read-only whitelist "
            f"{_READ_ONLY_GIT}; the controller never modifies Git state"
        )
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=text,
            timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerError(f"controller: cannot run git {' '.join(args)}: {exc}") from exc


def _git_output(repo_root, *args: str) -> str:
    proc = _git_readonly(repo_root, *args)
    if proc.returncode != 0:
        raise RunnerError(
            f"controller: git {' '.join(args)} failed in {repo_root}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def validate_approved_sha(approved_sha: str) -> str:
    """The CLI-supplied approval must be a full 40-hex string (never HEAD)."""
    value = (approved_sha or "").strip().lower()
    if not _SHA40_RE.match(value):
        raise RunnerError(
            f"--approved-sha must be a full 40-hex commit SHA supplied by Codex "
            f"after code review, got {approved_sha!r}; the controller never "
            "guesses it from HEAD or any ref"
        )
    return value


def check_git_state(
    repo_root,
    approved_sha: str,
    *,
    task_id: str = TASK_ID,
    run_id: str = G4_RUN_ID,
    max_pairs: int = G4_MAX_PAIRS,
    card_hours: float = G4_CARD_HOURS,
    expected_origin: str = EXPECTED_ORIGIN_URL,
    authorization_ref: str = AUTHORIZATION_REF,
) -> dict[str, Any]:
    """All read-only pre-launch Git/authorization checks; returns the facts.

    Order: HEAD equality + detached + clean -> origin URL -> the ALREADY
    fetched ``<authorization_ref>:ACTIVE_TASK_BLOB`` text -> the audited
    authorization parser -> the fixed G4 scope (task/run/budget). Any
    deviation raises RunnerError BEFORE anything is created.
    """
    repo_root = Path(repo_root)
    head = _git_output(repo_root, "rev-parse", "HEAD").lower()
    if not _SHA40_RE.match(head):
        raise RunnerError(f"controller: git HEAD {head!r} is not a full 40-hex SHA")
    if head != approved_sha:
        raise RunnerError(
            f"controller: checked-out HEAD {head} != --approved-sha {approved_sha}; "
            "detach the reviewed commit before launching"
        )
    attached = _git_readonly(repo_root, "symbolic-ref", "-q", "HEAD")
    if attached.returncode == 0:
        raise RunnerError(
            f"controller: HEAD is attached to {attached.stdout.strip()!r}; the "
            "approved protocol is `git checkout --detach <approved_sha>` -- "
            "refusing to launch on a movable ref"
        )
    if attached.returncode != 1:
        raise RunnerError(
            f"controller: cannot determine HEAD attachment: {attached.stderr.strip()}"
        )
    dirty = _git_output(repo_root, "status", "--porcelain")
    if dirty:
        raise RunnerError(
            f"controller: worktree at {repo_root} is dirty; refusing to launch:\n{dirty}"
        )
    origin_url = _git_output(repo_root, "remote", "get-url", "origin")
    if normalize_remote_url(origin_url) != normalize_remote_url(expected_origin):
        raise RunnerError(
            f"controller: origin remote {origin_url!r} != expected {expected_origin!r}"
        )
    origin_ref_sha = _git_output(repo_root, "rev-parse", authorization_ref)
    blob = _git_readonly(
        repo_root, "show", f"{authorization_ref}:{ACTIVE_TASK_BLOB}", text=False
    )
    if blob.returncode != 0:
        raise RunnerError(
            f"controller: cannot read {authorization_ref}:{ACTIVE_TASK_BLOB}: "
            f"{blob.stderr.decode('utf-8', 'replace').strip()}"
        )
    active_text = blob.stdout.decode("utf-8")
    blob_sha256 = hashlib.sha256(blob.stdout).hexdigest()
    fields = parse_active_task_authorization(active_text)  # audited fail-closed parser
    if fields["task_id"] != task_id:
        raise RunnerError(
            f"controller: ACTIVE_TASK task_id {fields['task_id']!r} != {task_id!r}"
        )
    if fields["approved_sha"] != approved_sha:
        raise RunnerError(
            f"controller: fetched ACTIVE_TASK approved_sha {fields['approved_sha']} "
            f"!= --approved-sha {approved_sha}"
        )
    if int(fields["max_pairs"]) != int(max_pairs):
        raise RunnerError(
            f"controller: approved max_scored_pairs {fields['max_pairs']} != the "
            f"fixed G4 workload {max_pairs}"
        )
    if float(fields["card_hours"]) != float(card_hours):
        raise RunnerError(
            f"controller: approved gpu_budget {fields['card_hours']} card-hours != "
            f"the fixed G4 budget {card_hours}"
        )
    active_run_id = active_task_field(active_text, "run_id")
    if active_run_id != run_id:
        raise RunnerError(
            f"controller: ACTIVE_TASK run_id {active_run_id!r} != {run_id!r}"
        )
    return {
        "head_sha": head,
        "approved_sha": approved_sha,
        "detached": True,
        "worktree_clean": True,
        "origin_url": origin_url,
        "authorization_source": f"{authorization_ref}:{ACTIVE_TASK_BLOB} @ {origin_ref_sha}",
        "authorization_ref_sha": origin_ref_sha,
        "active_task_blob_sha256": blob_sha256,
        "task_id": task_id,
        "run_id": run_id,
        "max_scored_pairs": int(max_pairs),
        "gpu_budget_card_hours": float(card_hours),
        "task_state": fields["task_state"],
    }


def recheck_git_state(repo_root, approved_sha: str) -> dict[str, Any]:
    """Post-launch read-only re-check (recorded as evidence, never a fix)."""
    head = _git_output(repo_root, "rev-parse", "HEAD").lower()
    dirty = _git_output(repo_root, "status", "--porcelain")
    return {
        "head_sha": head,
        "head_matches_approved": head == approved_sha,
        "worktree_clean": not dirty,
        "porcelain": dirty,
    }


# ---------------------------------------------------------------------------
# Fixed full-matrix plan (REAL PairSpace math) and pinned scorer argv/env
# ---------------------------------------------------------------------------


def g4_pair_space() -> PairSpace:
    """The real implicit pair space of the fixed G4 workload."""
    return PairSpace(n_texts=G4_N_TEXTS, n_cads=G4_N_CADS)


def shard_plan(pair_space: PairSpace, shard_id: int) -> dict[str, Any]:
    """Per-card exact plan from the REAL producer math (text2cad_pair_index).

    shard0 -> texts [0,1822), shard1 -> [1822,3644); each shard is 1,822
    single-text chunks, 14,659,812 pairs and 916,466 batches (8,046 pairs per
    text = 502 full batches of 16 + the honest tail of 14).
    """
    if shard_id not in (0, 1):
        raise RunnerError(f"shard_id must be 0 or 1, got {shard_id}")
    t0, t1 = pair_space.shard_text_range(G4_N_SHARDS, shard_id)
    chunks = list(pair_space.iter_chunks(G4_N_SHARDS, shard_id, G4_TEXTS_PER_CHUNK))
    n_pairs = sum((b - a) * pair_space.n_cads for _, a, b, _ in chunks)
    n_batches = sum(
        -(-((b - a) * pair_space.n_cads) // G4_BATCH_SIZE) for _, a, b, _ in chunks
    )
    return {
        "shard_id": shard_id,
        "device": f"npu:{shard_id}",
        "text_range": [t0, t1],
        "n_chunks": len(chunks),
        "n_pairs": n_pairs,
        "n_batches": n_batches,
        "batch_size": G4_BATCH_SIZE,
        "tail_batch_size": pair_space.n_cads % G4_BATCH_SIZE,
        "texts_per_chunk": G4_TEXTS_PER_CHUNK,
        "cards_used": G4_CARDS_USED,
        "n_shards": G4_N_SHARDS,
    }


def build_card_argv(
    shard_id: int,
    *,
    repo_root,
    runtime_root,
    run_id: str = G4_RUN_ID,
    python_path: str = PRODUCTION_PYTHON,
) -> list[str]:
    """The FIXED full-matrix scorer command for one card (no free parameters).

    Identical in shape to the audited G3 production template minus the probe
    flags; only ``--device`` / ``--shard-id`` / ``--out-dir`` differ between
    the two cards. The forbidden probe/resume/cache flags are self-checked.
    """
    if shard_id not in (0, 1):
        raise RunnerError(f"shard_id must be 0 or 1, got {shard_id}")
    repo = Path(repo_root)
    rt = Path(runtime_root)
    argv = [
        str(python_path),
        str(repo / SCORER_SCRIPT_REL),
        "--mode", "score",
        "--checkpoint",
        str(rt / "models" / "Text2CAD-v1.0" / EXPECTED_MODEL_REVISION / "Text2CAD_1.0.pth"),
        "--bert-dir",
        str(rt / "models" / "bert-large-uncased" / EXPECTED_BERT_REVISION),
        "--cad-seq-zip",
        str(rt / "datasets" / "Text2CAD-v1.0" / EXPECTED_DATA_REVISION / "cad_seq.zip"),
        "--pair-manifest-dir",
        str(rt / "pair_manifest_v4"),
        "--expected-keys-manifest",
        str(repo / EXPECTED_KEYS_MANIFEST_REL),
        "--batch-size", str(G4_BATCH_SIZE),
        "--texts-per-chunk", str(G4_TEXTS_PER_CHUNK),
        "--cards-used", str(G4_CARDS_USED),
        "--device", f"npu:{shard_id}",
        "--shard-id", str(shard_id),
        "--n-shards", str(G4_N_SHARDS),
        "--out-dir",
        str(rt / "runs" / run_id / f"card{shard_id}"),
        "--repo-root",
        str(repo),
    ]
    for flag in FORBIDDEN_SCORER_FLAGS:
        if flag in argv:
            raise RunnerError(f"fixed full-matrix argv must never carry {flag}")
    return argv


def build_child_env(
    runtime_root,
    repo_root,
    *,
    base_env: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], list[str]]:
    """Child environment: inherited session minus device masks + pinned vars.

    Returns ``(env, removed_mask_vars)``. The cache/tmp directories must
    resolve OUTSIDE the Git checkout (source-tree caches stay external).
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = [name for name in DEVICE_MASK_ENV_VARS if name in env]
    for name in removed:
        env.pop(name)
    rt = Path(runtime_root)
    overrides = dict(PINNED_ENV_OVERRIDES)
    for name, path in (
        ("HF_HOME", rt / "hf"),
        ("XDG_CACHE_HOME", rt / "xdg"),
        ("TMPDIR", rt / "tmp"),
    ):
        try:
            ensure_outside_checkout(path, repo_root)
        except PairIndexError as exc:
            raise RunnerError(f"child env {name} must be outside the checkout: {exc}") from exc
        overrides[name] = str(path)
    env.update(overrides)
    return env, removed


# ---------------------------------------------------------------------------
# Read-only NPU preflight (never terminates anything)
# ---------------------------------------------------------------------------


def parse_npu_smi_health(text: str) -> dict[str, Any]:
    """Fail-closed parse of ``npu-smi info``: both chips OK, zero processes.

    Any ambiguity (fewer than two chip rows, a non-OK health field, a process
    row in the process table, or a missing 'No running processes found'
    marker) refuses the launch; the raw output is always recorded. Process
    rows are detected by the REAL pipe-separated column layout
    (``| NPU  Chip | Process id | Process name | Process memory(MB) |``), so
    a busy card is refused even when a 'No running processes found' marker
    is also present in a mixed output.
    """
    chips: list[dict[str, str]] = []
    process_rows: list[str] = []
    no_process_markers = 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "No running processes found" in line:
            no_process_markers += 1
            continue
        fields = [f.strip() for f in line.strip().strip("|").split("|")]
        if fields and re.match(r"^\d+\s+Ascend", fields[0]):
            chips.append({"chip": fields[0], "health": fields[1] if len(fields) > 1 else ""})
            continue
        if (
            len(fields) >= 2
            and re.match(r"^\d+\s+\d+$", fields[0])
            and re.match(r"^\d+$", fields[1])
        ):
            # process table data row, e.g. "| 2     0  | 3294246 | python | 1849 |"
            process_rows.append(line.strip())
    problems: list[str] = []
    if len(chips) < 2:
        problems.append(f"expected at least 2 chip rows, parsed {len(chips)}")
    unhealthy = [c for c in chips if c["health"] != "OK"]
    if unhealthy:
        problems.append(f"unhealthy chip(s): {unhealthy}")
    if process_rows:
        problems.append(f"NPU has {len(process_rows)} running process row(s), e.g. {process_rows[:2]}")
    if no_process_markers < 1 and not process_rows:
        problems.append("no 'No running processes found' marker; npu-smi output format drifted")
    if problems:
        raise RunnerError("npu-smi preflight refused: " + "; ".join(problems))
    return {"chips": chips, "n_chips": len(chips), "running_processes": 0}


def npu_smi_preflight(timeout: float = 60.0) -> dict[str, Any]:
    """Run the READ-ONLY ``npu-smi info`` and verify health/idleness."""
    try:
        proc = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerError(f"npu-smi preflight cannot run: {exc}") from exc
    if proc.returncode != 0:
        raise RunnerError(f"npu-smi info exited {proc.returncode}: {proc.stderr.strip()}")
    report = parse_npu_smi_health(proc.stdout)
    report["raw_output"] = proc.stdout
    return report


# ---------------------------------------------------------------------------
# Fresh-run / budget-ledger freshness
# ---------------------------------------------------------------------------


def canonical_budget_path(runtime_root, approved_sha: str) -> Path:
    """The scorer's shared ledger path formula (cross-checked by a unit test
    against the audited ``text2cad_frozen_runner.canonical_budget_path``)."""
    return Path(runtime_root) / "budgets" / f"text2cad_m1c_{approved_sha}.json"


def check_fresh_run(runtime_root, run_id: str, approved_sha: str) -> dict[str, Any]:
    """The run root, both card dirs and the shared ledger must NOT exist."""
    run_root = Path(runtime_root) / "runs" / run_id
    ledger = canonical_budget_path(runtime_root, approved_sha)
    ledger_lock = ledger.with_suffix(ledger.suffix + ".lock")
    problems: list[str] = []
    if run_root.exists():
        problems.append(
            f"run root {run_root} already exists; refusing to overwrite or "
            "double-launch (a previous controller invocation, completed or "
            "not, must be reviewed manually first)"
        )
    for shard in (0, 1):
        if (run_root / f"card{shard}").exists():
            problems.append(f"card{shard} directory already exists under {run_root}")
    if ledger.exists():
        problems.append(
            f"budget ledger {ledger} already exists; a fresh G4 run requires a "
            "fresh ledger bound to this approval"
        )
    if ledger_lock.exists():
        problems.append(f"stale ledger lock {ledger_lock} exists; resolve manually")
    if problems:
        raise RunnerError("fresh-run check refused: " + "; ".join(problems))
    return {"run_root": str(run_root), "budget_ledger": str(ledger)}


# ---------------------------------------------------------------------------
# Launch configuration + the launch/wait engine
# ---------------------------------------------------------------------------


@dataclass
class LaunchConfig:
    """Everything the controller needs; production values are the defaults.

    ``repo_root``/``runtime_root``/``run_id`` are pinned by the production
    CLI (``main`` refuses any other value); tests parameterize them together
    with the injected clock/process/preflight hooks. ``wall_budget_s`` is the
    COMMON seven-day wall clock shared by both cards.
    """

    approved_sha: str
    repo_root: Path = PRODUCTION_REPO_ROOT
    runtime_root: Path = PRODUCTION_RUNTIME_ROOT
    run_id: str = G4_RUN_ID
    python_path: str = PRODUCTION_PYTHON
    task_id: str = TASK_ID
    max_pairs: int = G4_MAX_PAIRS
    card_hours: float = G4_CARD_HOURS
    wall_budget_s: float = G4_WALL_BUDGET_S
    term_grace_s: float = G4_TERM_GRACE_S
    kill_wait_s: float = G4_KILL_WAIT_S
    stagger_s: float = G4_STAGGER_S
    poll_interval_s: float = G4_POLL_INTERVAL_S
    min_free_bytes: int = G4_MIN_FREE_BYTES
    expected_origin: str = EXPECTED_ORIGIN_URL
    authorization_ref: str = AUTHORIZATION_REF
    descendant_settle_s: float = G4_DESCENDANT_SETTLE_S
    descendant_term_grace_s: float = G4_DESCENDANT_TERM_GRACE_S
    descendant_kill_wait_s: float = G4_DESCENDANT_KILL_WAIT_S


def _validate_config(cfg: LaunchConfig) -> None:
    cfg.approved_sha = validate_approved_sha(cfg.approved_sha)
    if not cfg.run_id or re.search(r"[/\\]|\.\.", cfg.run_id):
        raise RunnerError(f"bad run_id {cfg.run_id!r}")
    if cfg.wall_budget_s <= cfg.term_grace_s + cfg.kill_wait_s:
        raise RunnerError(
            f"wall budget {cfg.wall_budget_s}s must exceed the in-budget kill "
            f"grace {cfg.term_grace_s + cfg.kill_wait_s}s"
        )
    if cfg.stagger_s < 0 or cfg.poll_interval_s <= 0:
        raise RunnerError("stagger must be >= 0 and the poll interval > 0")
    if cfg.term_grace_s < 0 or cfg.kill_wait_s <= 0:
        raise RunnerError("kill grace values must be positive")
    if (
        cfg.descendant_settle_s < 0
        or cfg.descendant_term_grace_s < 0
        or cfg.descendant_kill_wait_s <= 0
    ):
        raise RunnerError("descendant drain values must be >= 0 (kill wait > 0)")
    argv_probe = build_card_argv(
        0,
        repo_root=cfg.repo_root,
        runtime_root=cfg.runtime_root,
        run_id=cfg.run_id,
        python_path=cfg.python_path,
    )
    for label, revision in (
        ("checkpoint", EXPECTED_MODEL_REVISION),
        ("bert", EXPECTED_BERT_REVISION),
        ("cad_seq_zip", EXPECTED_DATA_REVISION),
    ):
        if not any(revision in arg for arg in argv_probe):
            raise RunnerError(f"fixed argv lost the pinned {label} revision {revision}")


def _atomic_write_text(path: Path, text: str) -> None:
    """Fail-if-exists one-file evidence write (same discipline as the runner)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RunnerError(f"evidence file already exists, refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RunnerError(f"evidence file appeared during launch: {path}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _append_event(run_root: Path, record: dict[str, Any]) -> None:
    with (run_root / EVENTS_LOG_NAME).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _default_popen(argv, **kwargs):
    return subprocess.Popen(argv, **kwargs)


@dataclass
class _CardState:
    shard_id: int
    argv: list[str]
    env: dict[str, str]
    proc: Any = None
    pid: Optional[int] = None
    launch_epoch: Optional[float] = None
    spawn_error: Optional[str] = None
    exit_code: Optional[int] = None
    exit_epoch: Optional[float] = None
    term_sent: bool = False
    kill_sent: bool = False
    #: Post-spawn evidence-I/O failure(s) (pid/time/exit file writes). The
    #: live child is never abandoned and never fake-exited because of these;
    #: any value here forces the run to end "failed".
    record_error: Optional[str] = None
    #: True when members of the card's own process group outlived the parent
    #: past the settle window and had to be terminated (POSIX only).
    leftover_descendants: bool = False
    #: Outcome of the post-exit group drain: not_spawned /
    #: unsupported_platform / empty / drained_within_grace / terminated /
    #: killed / survivors_after_kill / *_error:... .
    group_cleanup: Optional[str] = None
    #: True when a descendant TERM/KILL had to be sent AT or AFTER the
    #: original common deadline (only possible when the drain itself started
    #: late, e.g. a parent reaped at the pathological hard bound). The signal
    #: is still sent IMMEDIATELY -- never re-scheduled further out -- but the
    #: lateness is recorded here so the run can never claim the common budget
    #: held; it forces status "failed".
    group_deadline_breach: bool = False
    #: Non-ESRCH fault while probing the group (signal 0): the group's state
    #: could not be confirmed, so it is recorded honestly and forces status
    #: "failed" -- it is NEVER misreported as an empty group.
    group_probe_error: Optional[str] = None
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    _stdout_stream: Any = field(default=None, repr=False)
    _stderr_stream: Any = field(default=None, repr=False)

    @property
    def deadline_enforced(self) -> bool:
        return self.term_sent or self.kill_sent


def _note_card_error(card: _CardState, message: str) -> None:
    """Remember a post-spawn evidence-I/O failure; forces status failed."""
    card.record_error = (
        message if card.record_error is None else f"{card.record_error}; {message}"
    )


def _signal_card(card: _CardState, signum: int, group_signal_fn) -> str:
    """Signal a LIVE card at the deadline; returns the delivery scope.

    POSIX: the WHOLE process group the controller created (the child leads it
    via ``start_new_session``, so no other job can be reached). Elsewhere the
    bare PID only -- the degradation is recorded in the returned scope and in
    the exit record, never silently claimed as group cleanup.
    """
    if group_signal_fn is not None and card.pid is not None:
        try:
            if group_signal_fn(card.pid, signum):
                return "process_group"
            return "already_gone"  # the whole group exited between poll and signal
        except OSError:
            pass  # unexpected group-signal fault: fall back to the PID handle
    try:
        if signum == _SIGTERM:
            card.proc.terminate()
        else:
            card.proc.kill()
    except OSError:
        return "already_gone"
    return "pid_only"


def _drain_card_group(
    card: _CardState,
    *,
    group_signal_fn,
    now_fn,
    sleep_fn,
    settle_s: float,
    term_grace_s: float,
    kill_wait_s: float,
    deadline: float,
    confirm_end: float,
    events,
) -> None:
    """No member of the group WE created may outlive the controller (POSIX).

    Runs only AFTER the card's real exit is already reaped and recorded, and
    after the wait loop ended for every card, so peer supervision is never
    delayed. Every descendant SIGNAL is scheduled inside the ORIGINAL common
    ``deadline`` -- the +outer-kill-wait bound (``confirm_end``) is pure
    accounting and may cap ONLY the post-KILL confirmation wait:

      * the settle window ends no later than
        ``deadline - (term_grace_s + kill_wait_s)`` so the full TERM grace
        and KILL still fit inside the deadline; as the deadline approaches
        the window shrinks and is finally SKIPPED, never moved past it;
      * TERM is sent immediately after the settle window and its grace is
        capped at ``deadline - kill_wait_s`` so KILL, when needed, is also
        sent inside the deadline;
      * the post-KILL confirmation wait is capped by ``confirm_end`` --
        bookkeeping AFTER the KILL was already sent, never a new grace that
        lets descendants run past the common deadline.

    If the drain itself STARTS past those cutoffs (e.g. a parent was only
    reaped at the pathological hard bound), the remaining TERM/KILL are sent
    IMMEDIATELY back-to-back -- never re-scheduled further out -- and every
    signal sent at/after the original deadline flips
    ``group_deadline_breach``: the cleanup still happens (a live job is never
    abandoned just to keep the log pretty) but the run honestly fails and can
    never claim the budget held. Survivors past the settle window flip
    ``leftover_descendants`` -- an honest anomaly that fails the run. A probe
    fault other than ESRCH is recorded as ``group_probe_error`` (and fails
    the run) instead of being misreported as an empty group. Platforms
    without process groups record the degradation instead of claiming
    cleanup.
    """
    if card.proc is None or card.pid is None:
        card.group_cleanup = "not_spawned"
        return
    if group_signal_fn is None:
        card.group_cleanup = "unsupported_platform"
        return
    pgid = card.pid  # start_new_session=True: our child leads this group

    def group_alive() -> bool:
        try:
            return bool(group_signal_fn(pgid, 0))
        except ProcessLookupError:
            return False
        # Any other OSError (EPERM, ...) propagates: an unconfirmable group
        # must NEVER be misreported as empty.

    term_cutoff = deadline - (term_grace_s + kill_wait_s)
    kill_cutoff = deadline - kill_wait_s
    try:
        if not group_alive():
            card.group_cleanup = "empty"
            return
        settle_end = min(now_fn() + settle_s, term_cutoff)
        while group_alive():
            now = now_fn()
            if now >= settle_end:
                break
            sleep_fn(min(0.05, max(0.0, settle_end - now)))
        if not group_alive():
            card.group_cleanup = "drained_within_grace"
            return
    except OSError as exc:
        card.group_probe_error = f"{type(exc).__name__}: {exc}"
        card.group_cleanup = f"probe_error:{type(exc).__name__}: {exc}"
        return
    card.leftover_descendants = True
    events({"event": "descendants_detected", "shard_id": card.shard_id, "pgid": pgid})
    sent_at = now_fn()
    try:
        group_signal_fn(pgid, _SIGTERM)
    except OSError as exc:
        card.group_cleanup = f"term_error:{type(exc).__name__}:{exc}"
        return
    if sent_at >= deadline:
        card.group_deadline_breach = True
    card.group_cleanup = "terminated"
    events(
        {
            "event": "descendant_term_sent",
            "shard_id": card.shard_id,
            "pgid": pgid,
            "after_deadline": sent_at >= deadline,
        }
    )
    term_end = min(now_fn() + term_grace_s, kill_cutoff)
    try:
        while group_alive():
            now = now_fn()
            if now >= term_end:
                break
            sleep_fn(min(0.05, max(0.0, term_end - now)))
        if not group_alive():
            return
    except OSError as exc:
        card.group_probe_error = f"{type(exc).__name__}: {exc}"
        card.group_cleanup = f"probe_error:{type(exc).__name__}: {exc}"
        return
    sent_at = now_fn()
    try:
        group_signal_fn(pgid, _SIGKILL)
    except OSError as exc:
        card.group_cleanup = f"kill_error:{type(exc).__name__}:{exc}"
        return
    if sent_at >= deadline:
        card.group_deadline_breach = True
    card.group_cleanup = "killed"
    events(
        {
            "event": "descendant_kill_sent",
            "shard_id": card.shard_id,
            "pgid": pgid,
            "after_deadline": sent_at >= deadline,
        }
    )
    # confirm_end (deadline + the outer kill wait) bounds ONLY this
    # confirmation wait -- accounting AFTER the KILL was already sent.
    kill_end = min(now_fn() + kill_wait_s, confirm_end)
    try:
        while group_alive():
            now = now_fn()
            if now >= kill_end:
                break
            sleep_fn(min(0.05, max(0.0, kill_end - now)))
        if group_alive():
            card.group_cleanup = "survivors_after_kill"
            events(
                {"event": "descendants_unreaped", "shard_id": card.shard_id, "pgid": pgid}
            )
    except OSError as exc:
        card.group_probe_error = f"{type(exc).__name__}: {exc}"
        card.group_cleanup = f"probe_error:{type(exc).__name__}: {exc}"
        return


def launch(
    cfg: LaunchConfig,
    *,
    now_fn=time.time,
    sleep_fn=time.sleep,
    popen_fn=_default_popen,
    preflight_fn=npu_smi_preflight,
    free_bytes_fn=None,
    group_signal_fn=None,
) -> dict[str, Any]:
    """Verify, spawn both cards, wait under the COMMON deadline, record all.

    Pre-launch refusals raise RunnerError BEFORE the run root is created (a
    refused launch leaves no residue and may be re-attempted after the
    operator fixes the cause). Once the run root exists the function always
    finishes with a written ``controller_exit.json``; it never relaunches,
    never retries a failed card and never deletes evidence.

    Lifecycle guarantees (R1): after a SUCCESSFUL Popen the live child stays
    under the common deadline no matter what happens to any evidence write,
    the event log or the final Git re-check -- post-spawn I/O failures are
    remembered and force ``failed`` instead of abandoning the child or
    writing it a fake exit. If even ``controller_exit.json`` cannot be
    written, the function still reaps every child first and then raises
    RunnerError (a non-zero controller exit), never a false ``ok``.
    ``group_signal_fn`` delivers signals to the process groups the
    controller created (POSIX default ``os.killpg``); ``None`` means the
    platform has no process groups and the controller degrades to PID-only
    signaling, recorded honestly.
    """
    _validate_config(cfg)
    repo_root = Path(cfg.repo_root)
    runtime_root = Path(cfg.runtime_root)
    if free_bytes_fn is None:
        free_bytes_fn = lambda p: shutil.disk_usage(p).free  # noqa: E731
    if group_signal_fn is None and _PROCESS_GROUP_SUPPORTED:
        group_signal_fn = _posix_group_signal

    # 1. read-only Git + authorization-source verification.
    git_facts = check_git_state(
        repo_root,
        cfg.approved_sha,
        task_id=cfg.task_id,
        run_id=cfg.run_id,
        max_pairs=cfg.max_pairs,
        card_hours=cfg.card_hours,
        expected_origin=cfg.expected_origin,
        authorization_ref=cfg.authorization_ref,
    )
    # 2. fresh run root / card dirs / shared ledger.
    fresh = check_fresh_run(runtime_root, cfg.run_id, cfg.approved_sha)
    run_root = Path(fresh["run_root"])
    # 3. read-only NPU health/idleness preflight.
    npu_report = preflight_fn()
    # 4. runtime partition free space.
    free_bytes = int(free_bytes_fn(runtime_root))
    if free_bytes < cfg.min_free_bytes:
        raise RunnerError(
            f"runtime partition has {free_bytes} bytes free, need >= "
            f"{cfg.min_free_bytes} (contract sec.4)"
        )
    if not run_root.parent.is_dir():
        raise RunnerError(f"run parent {run_root.parent} does not exist; wrong runtime root?")

    # 5. commit point: the run root mkdir IS the single-launch guard.
    try:
        os.mkdir(run_root)
    except FileExistsError as exc:
        raise RunnerError(
            f"run root {run_root} already exists; refusing a second launch"
        ) from exc

    event_errors: list[str] = []

    def events(record: dict[str, Any]) -> None:
        """Append one controller event (best-effort). A failing event log
        NEVER aborts the supervision of live children; the failure is
        remembered and forces the run to end "failed"."""
        try:
            _append_event(run_root, {"time_epoch": now_fn(), **record})
        except Exception as exc:  # noqa: BLE001 -- evidence I/O must not kill supervision
            if len(event_errors) < 5:
                event_errors.append(f"{type(exc).__name__}: {exc}")
    pair_space = g4_pair_space()
    plans = {s: shard_plan(pair_space, s) for s in (0, 1)}
    cards: dict[int, _CardState] = {}
    common_start = now_fn()
    deadline = common_start + cfg.wall_budget_s
    term_at = deadline - (cfg.term_grace_s + cfg.kill_wait_s)
    kill_at = deadline - cfg.kill_wait_s
    events(
        {
            "event": "common_start",
            "run_id": cfg.run_id,
            "common_start_epoch": common_start,
            "common_deadline_epoch": deadline,
            "wall_budget_s": cfg.wall_budget_s,
            "term_at_epoch": term_at,
            "kill_at_epoch": kill_at,
        }
    )

    # 6. spawn both cards (staggered, SAME deadline).
    for shard in (0, 1):
        if shard == 1 and cfg.stagger_s:
            sleep_fn(cfg.stagger_s)
        argv = build_card_argv(
            shard,
            repo_root=repo_root,
            runtime_root=runtime_root,
            run_id=cfg.run_id,
            python_path=cfg.python_path,
        )
        env, removed_masks = build_child_env(runtime_root, repo_root)
        card = _CardState(shard_id=shard, argv=argv, env=env)
        cards[shard] = card
        try:
            _atomic_write_text(
                run_root / f"launch_command_card{shard}.txt", shlex.join(argv) + "\n"
            )
            for name in ("HF_HOME", "XDG_CACHE_HOME", "TMPDIR"):
                os.makedirs(env[name], exist_ok=True)
            card._stdout_stream = (run_root / f"card{shard}.stdout.log").open("xb")
            card._stderr_stream = (run_root / f"card{shard}.stderr.log").open("xb")
            card.stdout_path = run_root / f"card{shard}.stdout.log"
            card.stderr_path = run_root / f"card{shard}.stderr.log"
            card.proc = popen_fn(
                argv,
                env=env,
                cwd=str(repo_root),
                stdin=subprocess.DEVNULL,
                stdout=card._stdout_stream,
                stderr=card._stderr_stream,
                start_new_session=True,  # setsid: SSH disconnect must not kill the job
            )
        except Exception as exc:  # noqa: BLE001 -- record and keep the peer card
            # GENUINE spawn failure: Popen never returned, so this card owns
            # NO live process -- recording its NONE exit is honest and final.
            card.spawn_error = f"{type(exc).__name__}: {exc}"
            for stream in (card._stdout_stream, card._stderr_stream):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            _record_card_exit(run_root, card, None, now_fn(), events, note="spawn_failed")
            continue
        # Popen SUCCEEDED: from here on the live child stays under the common
        # deadline NO MATTER what happens to the evidence writes below. A
        # post-spawn record failure is remembered (and fails the run at the
        # end) -- it never writes a fake exit for the live process and never
        # aborts the supervision of either card.
        card.pid = int(card.proc.pid)
        card.launch_epoch = now_fn()
        try:
            _atomic_write_text(run_root / f"card{shard}.pid.txt", f"{card.pid}\n")
            _atomic_write_text(
                run_root / f"card{shard}.launch_time.txt",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(card.launch_epoch))
                + f" epoch={card.launch_epoch}\n",
            )
        except Exception as exc:  # noqa: BLE001 -- child alive; keep supervising
            _note_card_error(
                card, f"post-spawn pid/time record failed: {type(exc).__name__}: {exc}"
            )
        events(
            {
                "event": "spawn",
                "shard_id": shard,
                "pid": card.pid,
                "device": f"npu:{shard}",
                "removed_device_mask_vars": removed_masks,
                "pid_recorded": card.record_error is None,
            }
        )

    # 7. the complete launch record (fail-if-exists).
    launch_record = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "run_id": cfg.run_id,
        "approved_sha": cfg.approved_sha,
        "git": git_facts,
        "fresh": fresh,
        "npu_preflight": npu_report,
        "runtime_free_bytes": free_bytes,
        "common_start_epoch": common_start,
        "common_deadline_epoch": deadline,
        "wall_budget_s": cfg.wall_budget_s,
        "term_grace_s": cfg.term_grace_s,
        "kill_wait_s": cfg.kill_wait_s,
        "stagger_s": cfg.stagger_s,
        "plans": {str(s): plans[s] for s in (0, 1)},
        "cards": {
            str(s): {
                "shard_id": s,
                "device": f"npu:{s}",
                "argv": cards[s].argv,
                "command": shlex.join(cards[s].argv),
                "pid": cards[s].pid,
                "launch_epoch": cards[s].launch_epoch,
                "spawn_error": cards[s].spawn_error,
                "record_error": cards[s].record_error,
                "out_dir": str(runtime_root / "runs" / cfg.run_id / f"card{s}"),
                "stdout_log": str(cards[s].stdout_path),
                "stderr_log": str(cards[s].stderr_path),
            }
            for s in (0, 1)
        },
    }
    # Its failure is NOT a reason to abandon the already-spawned children:
    # remember it, force the run to end "failed", and keep supervising under
    # the common deadline.
    launch_record_error: Optional[str] = None
    try:
        _atomic_write_text(
            run_root / LAUNCH_RECORD_NAME, json.dumps(launch_record, indent=2, sort_keys=True) + "\n"
        )
    except Exception as exc:  # noqa: BLE001 -- children stay supervised below
        launch_record_error = f"{type(exc).__name__}: {exc}"

    # 8. wait under the COMMON deadline; reap + record each exit IMMEDIATELY.
    pending = {s for s in (0, 1) if cards[s].proc is not None}
    while pending:
        now = now_fn()
        for shard in sorted(pending):
            card = cards[shard]
            rc = card.proc.poll()
            if rc is not None:
                _record_card_exit(run_root, card, rc, now, events)
                pending.discard(shard)
                continue
            if now >= kill_at and not card.kill_sent:
                scope = _signal_card(card, _SIGKILL, group_signal_fn)
                card.kill_sent = True
                events({"event": "kill_sent", "shard_id": shard, "pid": card.pid,
                        "signal_scope": scope})
            elif now >= term_at and not card.term_sent:
                scope = _signal_card(card, _SIGTERM, group_signal_fn)
                card.term_sent = True
                events({"event": "term_sent", "shard_id": shard, "pid": card.pid,
                        "signal_scope": scope})
        if not pending:
            break
        if now_fn() >= deadline + cfg.kill_wait_s:
            # Pathological: SIGKILL not reaped within the hard bound. Record
            # the unknown honestly and stop -- never hang the controller.
            for shard in sorted(pending):
                card = cards[shard]
                card.kill_sent = True
                _record_card_exit(run_root, card, None, now_fn(), events,
                                  note="process_unreaped_after_kill")
            pending.clear()
            break
        sleep_fn(cfg.poll_interval_s)

    for card in cards.values():
        for stream in (card._stdout_stream, card._stderr_stream):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass

    # 8b. drain leftover members of the process groups WE created (POSIX).
    # Runs only after EVERY parent is reaped (or honestly recorded as
    # unreaped), so peer supervision is never delayed. Descendant SIGNALS
    # (settle/TERM/KILL) are scheduled inside the ORIGINAL common deadline;
    # deadline + kill_wait_s bounds ONLY the post-KILL confirmation wait --
    # accounting after termination was already sent, never an extra grace
    # that lets descendants run past the deadline. On platforms without
    # process groups this records the degradation.
    confirm_end = deadline + cfg.kill_wait_s
    for shard in (0, 1):
        _drain_card_group(
            cards[shard],
            group_signal_fn=group_signal_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            settle_s=cfg.descendant_settle_s,
            term_grace_s=cfg.descendant_term_grace_s,
            kill_wait_s=cfg.descendant_kill_wait_s,
            deadline=deadline,
            confirm_end=confirm_end,
            events=events,
        )

    # 9. final exit record + post-launch Git evidence. The Git re-check is
    # evidence only: by now every child is reaped, so its failure cannot
    # orphan anything -- it is remembered and fails the run honestly.
    post_git_error: Optional[str] = None
    try:
        post_git = recheck_git_state(repo_root, cfg.approved_sha)
    except Exception as exc:  # noqa: BLE001 -- children already reaped above
        post_git_error = f"{type(exc).__name__}: {exc}"
        post_git = {
            "error": post_git_error,
            "head_matches_approved": False,
            "worktree_clean": False,
        }
    ok = (
        all(
            cards[s].exit_code == 0
            and not cards[s].deadline_enforced
            and cards[s].spawn_error is None
            and cards[s].record_error is None
            and not cards[s].leftover_descendants
            and not cards[s].group_deadline_breach
            and cards[s].group_probe_error is None
            for s in (0, 1)
        )
        and launch_record_error is None
        and not event_errors
        and post_git_error is None
        and post_git["head_matches_approved"]
        and post_git["worktree_clean"]
    )
    exit_record = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "run_id": cfg.run_id,
        "approved_sha": cfg.approved_sha,
        "common_start_epoch": common_start,
        "common_deadline_epoch": deadline,
        "ended_epoch": now_fn(),
        "wall_budget_s": cfg.wall_budget_s,
        "post_launch_git": post_git,
        "launch_record_error": launch_record_error,
        "event_log_errors": event_errors,
        "process_group_supported": _PROCESS_GROUP_SUPPORTED,
        "cards": {
            str(s): {
                "shard_id": s,
                "device": f"npu:{s}",
                "pid": cards[s].pid,
                "launch_epoch": cards[s].launch_epoch,
                "exit_epoch": cards[s].exit_epoch,
                "exit_code": cards[s].exit_code,
                "deadline_enforced": cards[s].deadline_enforced,
                "term_sent": cards[s].term_sent,
                "kill_sent": cards[s].kill_sent,
                "spawn_error": cards[s].spawn_error,
                "record_error": cards[s].record_error,
                "leftover_descendants": cards[s].leftover_descendants,
                "group_cleanup": cards[s].group_cleanup,
                "group_deadline_breach": cards[s].group_deadline_breach,
                "group_probe_error": cards[s].group_probe_error,
            }
            for s in (0, 1)
        },
        "status": "ok" if ok else "failed",
    }
    # The final controller_exit event is REQUIRED completion evidence, so it
    # is appended BEFORE the exit record is persisted: if even this last
    # event write fails, the failure is folded into the SAME record the
    # function returns and the disk keeps -- the run can never return ok with
    # an event error, and the persisted JSON never hides one.
    events({"event": "controller_exit", "status": exit_record["status"]})
    exit_record["event_log_errors"] = list(event_errors)
    if event_errors:
        exit_record["status"] = "failed"
    try:
        _atomic_write_text(
            run_root / EXIT_RECORD_NAME, json.dumps(exit_record, indent=2, sort_keys=True) + "\n"
        )
    except Exception as exc:  # noqa: BLE001 -- every child is already reaped;
        # a total disk fault may lose this file, but the controller must exit
        # non-zero instead of leaving a false "ok" ambiguous.
        raise RunnerError(
            f"controller_exit.json could not be written ({type(exc).__name__}: {exc}); "
            "all children were reaped and their real exits recorded where the "
            "disk allowed, but this run is NOT ok"
        ) from exc
    return exit_record


def _record_card_exit(run_root: Path, card: _CardState, rc, when: float, events, note: str = "") -> None:
    """Record the card's REAL exit the moment it is known.

    In-memory first -- supervision decisions never depend on the disk. The
    fail-if-exists file write and the event append are best-effort: their
    failure is remembered on the card (forcing the run to end "failed") but
    NEVER raises out of the wait loop, so the peer keeps its supervision.
    Called at most once per card, so the exit file is written exactly once:
    a live process never gets a fake exit and a real exit is never
    overwritten.
    """
    card.exit_code = None if rc is None else int(rc)
    card.exit_epoch = when
    try:
        _atomic_write_text(
            run_root / f"card{card.shard_id}.exit_code.txt",
            f"{card.exit_code if card.exit_code is not None else 'NONE'}\n",
        )
    except Exception as exc:  # noqa: BLE001 -- peer supervision must go on
        _note_card_error(card, f"exit record write failed: {type(exc).__name__}: {exc}")
    events(
        {
            "event": "card_exit",
            "shard_id": card.shard_id,
            "pid": card.pid,
            "exit_code": card.exit_code,
            "deadline_enforced": card.deadline_enforced,
            "note": note,
        }
    )


# ---------------------------------------------------------------------------
# CLI (production-pinned; no scorer pass-through exists)
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Text2CAD G4 full-matrix two-card launch controller. Verifies the "
            "Codex-approved full 40-hex SHA against the clean detached HEAD "
            "and the already-fetched origin/master ACTIVE_TASK (fixed task / "
            "run / 29,319,624 pairs / 336 card-hours), refuses any non-fresh "
            "run root or budget ledger, runs a read-only npu-smi preflight, "
            "then spawns the two FIXED full-matrix scorers under ONE common "
            "seven-day deadline and records real PIDs/stdout/stderr/exit "
            "codes. It never fetches/checks out Git, never adds scorer "
            "arguments, never restarts a failed card and never overwrites "
            "evidence."
        )
    )
    parser.add_argument(
        "--approved-sha",
        required=True,
        help="REQUIRED full 40-hex commit SHA approved by Codex; never "
        "defaulted to HEAD or any ref",
    )
    parser.add_argument("--repo-root", default=str(PRODUCTION_REPO_ROOT))
    parser.add_argument("--runtime-root", default=str(PRODUCTION_RUNTIME_ROOT))
    args = parser.parse_args(argv)
    try:
        if Path(args.repo_root) != PRODUCTION_REPO_ROOT:
            raise RunnerError(f"production checkout must be {PRODUCTION_REPO_ROOT}")
        if Path(args.runtime_root) != PRODUCTION_RUNTIME_ROOT:
            raise RunnerError(f"production runtime root must be {PRODUCTION_RUNTIME_ROOT}")
        cfg = LaunchConfig(
            approved_sha=args.approved_sha,
            repo_root=PRODUCTION_REPO_ROOT,
            runtime_root=PRODUCTION_RUNTIME_ROOT,
        )
        record = launch(cfg)
    except RunnerError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": record["status"],
                "run_id": record["run_id"],
                "cards": {
                    s: {"exit_code": c["exit_code"], "pid": c["pid"]}
                    for s, c in record["cards"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
