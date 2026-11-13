"""Frozen Text2CAD full-matrix scoring runner (M1c stage B2) -- fail-closed.

Task contract: experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md
(G0.2-G0.5) and experiments/reviews/IDEA01_M1C_STAGE_B1_CODEX_REVIEW_2026-09-30.md
(B2 conditions 1-4). Scoring math is EXCLUSIVELY the audited stage-A core
``text2cad_vector_scorer.two_head_factorized_score`` (two-head factorized
conditional log-probability sum -- never the training loss, never called a
CAD-geometry NLL) applied to raw logits produced by the audited stage-B1 port
(``text2cad_port``) under official teacher forcing.

AUTHORIZATION GATE (fail-closed; mirrors the audited cadmium_runner gate)
-------------------------------------------------------------------------
The production CLI (``main``, mode ``score``) refuses any model forward unless
ALL of the following hold, checked in this order BEFORE weights load:

  1. pair-manifest artifacts verify against the REVIEWED fingerprints
     (``--expected-pair-hashes``, default the committed
     ``experiments/configs/text2cad_pair_manifest_expected_20260930.json``);
  2. the authorization text is read from the FRESHLY FETCHED
     ``origin/master:experiments/ACTIVE_TASK.md`` via the audited
     ``p1_runner.fetch_authorized_active_task`` (origin URL validated) --
     never the working-tree file, because a commit cannot contain its own
     ``approved_sha``; it must state ``pilot_authorized`` with a full 40-hex
     ``approved_sha`` equal to the live DETACHED HEAD, a clean worktree,
     ``task_id`` == ``idea01_text2cad_benchmark_20260928``, and positive
     ``max_scored_pairs`` / ``gpu_budget`` (card-hours). As of 2026-09-30 the
     real origin/master file fails these checks, so the runner refuses;
  3. ``--repo-root`` is the production checkout and ``--expected-keys-manifest``
     resolves to the COMMITTED 172-key manifest inside it (the strict loader
     is never invoked without the independent manifest -- no silent omission);
  4. a persistent two-card shared budget ledger (``SharedBudgetLedger``)
     bound to ``approved_sha`` + the approved pair cap + the approved
     card-second cap has room for every batch BEFORE its forward; charges are
     persisted per batch so a crash never under-counts; an unfinished
     invocation can be reclaimed only after its heartbeat is stale AND its
     process is proved dead, charging card time through recovery.

There is no CLI switch that relaxes these gates. ``--dry-run`` is the only
forward-free mode: it performs the zero-forward manifest/asset audit and
writes a report, consumes no budget, and never builds the scoring loop.

ISOLATION
---------
The worker contract is exactly: ``text`` (str), official candidate tensors
(``cad_vec``/``flag_vec``/``index_vec``/``key_padding_mask`` + canonical
``attn_mask``) and the internally meaningless integer ``pair_index`` /
``text_index`` / ``cad_index``. This module NEVER imports or opens qrels,
the raw CSV, audit files or ``evaluation_map.json``; a source-hygiene test
locks this. UID linkage lives exclusively in the outer layer
(``text2cad_pair_index``) and the independent evaluation stage.

OUTPUT DISCIPLINE
-----------------
All outputs go to an ``--out-dir`` OUTSIDE the Git checkout (tmp/cache/log
likewise, per CLAUDE.md sec.3-4); chunk files are published atomically
(hard-link, fail-if-exists) as ``chunk_s{SS}_t{TTTTT}_{TTTTT}.jsonl`` with a
``chunk_header`` record carrying the binding fields (git SHA, pair-manifest
SHA, checkpoint SHA, device, dtype, trim/cache flags, score spec). A resume
(``--resume``) re-verifies every existing chunk (dense exact pair sequence +
binding equality + final statuses) and re-scores only incomplete chunks --
every pair is scored exactly once. Existing outputs are never overwritten.

OPTIMIZATION GATES (default OFF; B1 review condition 2)
-------------------------------------------------------
``trim_mode="batch_max"`` (slices trailing padding to the batch's max real
length, using the Stage-B2 ``cad_seq_len`` override) and ``text_cache=True``
(unique-text BERT embedding cache) are implemented but DISABLED by default.
Each must first pass its same-real-pair equivalence gate against the
full-length per-pair reference on the real device (G1 modes in
``text2cad_speed_bench.py``) within the pre-registered tolerances; this
runner records the flags in every chunk header so a binding drift on resume
is a hard mismatch.

PROBE SCOPE (G3, 2026-10-02; default OFF)
-----------------------------------------
``--probe-selection`` scopes a scoring run to the committed reviewed G3
probe (experiments/plans/IDEA01_M1C_G3_FULL_GALLERY_PROBE_2026-10-02.md): the
path must be exactly the in-repo committed config, which is re-derived from
the live verified artifacts and bound to the actual token audit and the
frozen G2 config before any load. The PairSpace, global text/pair indices,
gallery order and adjacent-CAD batches are UNCHANGED -- the scope only
filters this shard's single-text chunks to the probed queries, each still
scoring the complete gallery. Probe chunks carry ``probe_selection_sha256``
in their headers; chunk verification checks that scope field in BOTH
directions, so full-matrix and probe outputs can never verify under each
other's resume. A per-query timing sidecar (``timing_s{SS}.jsonl``) is
appended AFTER each chunk's atomic publish (true start/end epochs, E2E =
score + serialization + disk, real pair/batch/valid counts). R1 (2026-10-02):
a FRESH probe run refuses an existing sidecar outright; a RESUME strictly
re-validates every sidecar line against this shard's fixed probe and the
current scope BEFORE any forward, a completed chunk without its timing line
refuses, and an orphan timing line without a verified chunk refuses too --
chunks and sidecar must pair one-to-one. Probe runs are fixed to
batch 16 (+honest tail), texts_per_chunk=1; the summary additionally records
cold start, per-device peak HBM, the timing sidecar name and all scope
fingerprints. Without ``--probe-selection`` every output is byte-identical
to the pre-G3 runner.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import torch

from .cadmium_runner import (  # audited authorization gate (M1b, 2026-09-30)
    EXPECTED_ORIGIN_URL,
    PRODUCTION_BUDGET_ROOT,
    PRODUCTION_REPO_ROOT,
    RunAuthorization,
    RunnerError,
    authorize_run,
)
from .p1_runner import AUTHORIZATION_REF  # audited ref constant
from .text2cad_g3_probe import (  # G3 probe scope (pure-CPU module, no forward)
    G3_BATCH_SIZE,
    G3_PROBE_SELECTION_REL,
    G3_TEXTS_PER_CHUNK,
    PROBE_SCOPE_FIELD,
    append_timing_record,
    build_query_timing_record,
    probe_facts_for_shard,
    probe_texts_for_shard,
    read_timing_records,
    timing_sidecar_name,
    verify_g3_g2_config_binding,
    verify_g3_probe_selection_against_manifest,
    verify_g3_token_audit,
    verify_probe_resume_records,
)
from .text2cad_pair_index import (
    CAD_SEQ_ZIP_SHA256,
    CHUNK_BINDING_FIELDS,
    PAIR_MANIFEST_FILE,
    PairIndexError,
    PairSpace,
    chunk_file_name,
    load_cad_store,
    load_pair_manifest,
    load_texts,
    sha256_file,
    verify_chunk_records,
    verify_zip_members,
)
from .text2cad_port.loader import (  # audited stage-B1 strict loader
    PINNED_BERT_FILE_SHA256,
    PINNED_CHECKPOINT_SHA256,
    load_frozen_text2cad,
)
from .text2cad_port.model import (
    MAX_CAD_SEQUENCE_LENGTH,
    generate_attention_mask,
    teacher_forced_logits,
)
from .text2cad_vector_scorer import (  # audited stage-A scoring core
    SCORE_NAME,
    SCORE_SPEC_VERSION,
    ScorerInputError,
    two_head_factorized_score,
    validate_cad_vec,
)

# ---------------------------------------------------------------------------
# Pinned identities and production constants
# ---------------------------------------------------------------------------

TASK_ID = "idea01_text2cad_benchmark_20260928"

#: Pinned asset revisions (directories on yunnan-share2 carry these strings;
#: experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md).
EXPECTED_MODEL_REVISION = "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"
EXPECTED_BERT_REVISION = "6da4b6a26a1877e173fca3225479512db81a5e5b"
EXPECTED_DATA_REVISION = "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"

#: Committed (reviewed) in-repo files the production CLI pins by path.
EXPECTED_KEYS_MANIFEST_REL = "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"
EXPECTED_PAIR_HASHES_REL = "experiments/configs/text2cad_pair_manifest_expected_20260930.json"

CHUNK_SCHEMA_VERSION = "text2cad-chunk-v1"
SUMMARY_SCHEMA_VERSION = "text2cad-run-summary-v1"

#: Canonical per-approval budget ledger (shared by both cards and the
#: benchmark; the file name binds the approved code SHA).
def canonical_budget_path(approved_sha: str) -> Path:
    return PRODUCTION_BUDGET_ROOT / f"text2cad_m1c_{approved_sha}.json"


#: Candidate zip member statuses.
STATUS_SCORED = "scored"
STATUS_REJECTED = "rejected"
STATUS_ERROR = "error"

REASON_AMBIGUOUS_ZIP_MEMBER = "zip_member_ambiguous"

#: Two-card ledger timing defaults (seconds).
HEARTBEAT_INTERVAL_S = 30.0
STALE_AFTER_S = 180.0
LOCK_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Two-card shared budget ledger (persistent, locked, heartbeat-tracked)
# ---------------------------------------------------------------------------


class SharedBudgetLedger:
    """Cumulative pair + card-second budget shared by BOTH cards and all
    benchmark invocations under ONE approval.

    Persistence: one JSON document, written atomically (tmp + os.replace)
    while holding an adjacent lockfile (O_CREAT|O_EXCL). The ledger is BOUND
    to (approved_sha, max_pairs, max_card_seconds) on first write; a reload
    under different values refuses. Concurrency rules:

      * ``charge_pairs`` persists BEFORE the charged batch forwards, so a
        crash never under-counts (a crashed charged-but-unscored batch is
        over-counted on resume -- the safe direction);
      * card-seconds accumulate per invocation as ``wall_s * cards_used`` at
        finalize; a stale invocation is reclaimed only if its same-host
        process is proved dead, charging elapsed time through recovery;
      * a pending invocation on the SAME device/shard refuses a second
        invocation for that shard (double-run protection); pending
        invocations on OTHER shards are the other card legitimately running;
      * a stale lockfile (holder crashed) is NOT broken automatically --
        the next invocation refuses with a manual-resolution error.

    ``now_fn``/``sleep_fn`` are injectable for tests.
    """

    def __init__(
        self,
        path,
        *,
        approved_sha: str,
        max_pairs: int,
        max_card_seconds: float,
        now_fn=time.time,
        sleep_fn=time.sleep,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        stale_after_s: float = STALE_AFTER_S,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
    ) -> None:
        if max_pairs <= 0:
            raise RunnerError(f"ledger: max_pairs must be > 0, got {max_pairs}")
        if max_card_seconds <= 0:
            raise RunnerError(
                f"ledger: max_card_seconds must be > 0, got {max_card_seconds}"
            )
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.approved_sha = approved_sha
        self.max_pairs = int(max_pairs)
        self.max_card_seconds = float(max_card_seconds)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.stale_after_s = float(stale_after_s)
        self.lock_timeout_s = float(lock_timeout_s)
        self.data = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "approved_sha": self.approved_sha,
                "max_pairs": self.max_pairs,
                "max_card_seconds": self.max_card_seconds,
                "pairs_charged": 0,
                "card_seconds_spent": 0.0,
                "heartbeat_interval_s": self.heartbeat_interval_s,
                "invocations": [],
            }
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RunnerError(f"ledger: unreadable {self.path}: {exc}") from exc
        if data.get("approved_sha") != self.approved_sha:
            raise RunnerError(
                f"ledger: {self.path} is bound to approved_sha "
                f"{data.get('approved_sha')!r}, not {self.approved_sha!r}"
            )
        if int(data.get("max_pairs", -1)) != self.max_pairs:
            raise RunnerError(
                f"ledger: cap {data.get('max_pairs')} != approved {self.max_pairs}"
            )
        if float(data.get("max_card_seconds", -1.0)) != self.max_card_seconds:
            raise RunnerError(
                f"ledger: card-second cap {data.get('max_card_seconds')} != "
                f"approved {self.max_card_seconds}"
            )
        return data

    @contextmanager
    def _locked(self) -> Iterator[None]:
        deadline = self.now_fn() + self.lock_timeout_s
        fd: Optional[int] = None
        while fd is None:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self.now_fn() > deadline:
                    raise RunnerError(
                        f"ledger: lock {self.lock_path} held beyond "
                        f"{self.lock_timeout_s}s; if a run crashed while holding "
                        "it, remove the lock manually after verifying no runner "
                        "is alive -- refusing to break it automatically"
                    )
                self.sleep_fn(0.2)
            except OSError as exc:
                raise RunnerError(f"ledger: cannot take lock {self.lock_path}: {exc}") from exc
        try:
            os.close(fd)
            if self.path.exists():  # re-read inside the lock
                self.data = self._load_or_init()
            yield
        finally:
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(self.data, indent=2, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    # -- queries ------------------------------------------------------------

    def pairs_charged(self) -> int:
        return int(self.data["pairs_charged"])

    def remaining_pairs(self) -> int:
        return self.max_pairs - self.pairs_charged()

    def card_seconds_spent(self) -> float:
        return float(self.data["card_seconds_spent"])

    def pending_card_seconds(self, now: Optional[float] = None) -> float:
        now = self.now_fn() if now is None else now
        total = 0.0
        for inv in self.data["invocations"]:
            if not inv.get("finalized"):
                total += max(0.0, now - float(inv["started_epoch"])) * int(inv["cards_used"])
        return total

    def check_card_clock(self, now: Optional[float] = None) -> float:
        """Total card-seconds (spent + pending elapsed); refuse past the cap."""
        total = self.card_seconds_spent() + self.pending_card_seconds(now)
        if total > self.max_card_seconds:
            raise RunnerError(
                f"ledger: approved card-second budget exceeded: {total:.1f} > "
                f"{self.max_card_seconds}; stop and wait for an updated "
                "ACTIVE_TASK.md authorization"
            )
        return total

    # -- mutations (all under lock) ------------------------------------------

    def _auto_finalize_stale_locked(self, now: float) -> list[str]:
        finalized: list[str] = []
        for inv in self.data["invocations"]:
            if inv.get("finalized"):
                continue
            last = float(inv.get("last_heartbeat_epoch", inv["started_epoch"]))
            if now - last <= self.stale_after_s:
                continue
            # A long model forward may have no heartbeat. A timeout alone is
            # never proof that the process has released the NPU. On this
            # single-host ledger, reclaim only a process proved absent.
            if inv.get("host") != socket.gethostname() or not isinstance(inv.get("pid"), int):
                continue
            # os.kill(pid, 0) is a harmless liveness probe only on Unix.
            # Windows interprets it as a console event/termination request.
            # Local tests never need automatic process reclamation.
            if platform.system() != "Linux" or inv["pid"] <= 0:
                continue
            try:
                os.kill(inv["pid"], 0)
            except ProcessLookupError:
                pass  # proven absent
            except (PermissionError, OSError):
                continue  # alive or uncertain: fail closed
            else:
                continue  # still alive, even if its heartbeat is stale
            conservative = max(0.0, now - float(inv["started_epoch"])) * int(inv["cards_used"])
            inv.update(
                finalized=True,
                ended_epoch=now,
                card_seconds=round(conservative, 2),
                exit_code=None,
                note="auto_finalized_proven_dead_process",
            )
            self.data["card_seconds_spent"] = round(
                float(self.data["card_seconds_spent"]) + conservative, 2
            )
            finalized.append(inv["invocation_id"])
        return finalized

    def begin_invocation(self, invocation_id: str, shard_id: int, cards_used: int) -> list[str]:
        """Register a new invocation; returns ids of auto-finalized stale ones."""
        now = self.now_fn()
        with self._locked():
            finalized = self._auto_finalize_stale_locked(now)
            for inv in self.data["invocations"]:
                if inv.get("finalized"):
                    continue
                if inv["invocation_id"] == invocation_id:
                    raise RunnerError(
                        f"ledger: invocation {invocation_id} is already pending"
                    )
                if int(inv["shard_id"]) == int(shard_id):
                    raise RunnerError(
                        f"ledger: shard {shard_id} already has a live invocation "
                        f"{inv['invocation_id']} (fresh heartbeat); refusing a "
                        "double-run of the same shard"
                    )
            self.data["invocations"].append(
                {
                    "invocation_id": invocation_id,
                    "shard_id": int(shard_id),
                    "cards_used": int(cards_used),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_epoch": now,
                    "last_heartbeat_epoch": now,
                    "finalized": False,
                }
            )
            self.check_card_clock(now)
            self._persist_locked()
        return finalized

    def charge_pairs(self, invocation_id: str, n: int) -> None:
        """Charge ``n`` pairs BEFORE they forward; refuse beyond the cap."""
        if n < 0:
            raise RunnerError(f"ledger: cannot charge negative pair count {n}")
        if n == 0:
            return
        with self._locked():
            inv = self._find_pending(invocation_id)
            if self.pairs_charged() + n > self.max_pairs:
                raise RunnerError(
                    f"ledger: pair budget exceeded: charged {self.pairs_charged()} "
                    f"+ requested {n} > approved cap {self.max_pairs}; stop and "
                    "wait for an updated ACTIVE_TASK.md authorization"
                )
            self.data["pairs_charged"] += n
            inv["pairs_charged"] = int(inv.get("pairs_charged", 0)) + n
            self._persist_locked()

    def heartbeat(self, invocation_id: str) -> None:
        with self._locked():
            inv = self._find_pending(invocation_id)
            inv["last_heartbeat_epoch"] = self.now_fn()
            self.check_card_clock()
            self._persist_locked()

    def finalize_invocation(
        self, invocation_id: str, *, exit_code: int, note: str = ""
    ) -> None:
        now = self.now_fn()
        with self._locked():
            inv = self._find_pending(invocation_id)
            card_seconds = max(0.0, now - float(inv["started_epoch"])) * int(inv["cards_used"])
            inv.update(
                finalized=True,
                ended_epoch=now,
                card_seconds=round(card_seconds, 2),
                exit_code=int(exit_code),
                note=note,
            )
            self.data["card_seconds_spent"] = round(
                float(self.data["card_seconds_spent"]) + card_seconds, 2
            )
            self._persist_locked()

    def _find_pending(self, invocation_id: str) -> dict[str, Any]:
        for inv in self.data["invocations"]:
            if inv["invocation_id"] == invocation_id and not inv.get("finalized"):
                return inv
        raise RunnerError(f"ledger: no pending invocation {invocation_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_file": str(self.path),
            "approved_sha": self.approved_sha,
            "max_pairs": self.max_pairs,
            "max_card_seconds": self.max_card_seconds,
            "pairs_charged": self.pairs_charged(),
            "remaining_pairs": self.remaining_pairs(),
            "card_seconds_spent": self.card_seconds_spent(),
            "invocations": [
                {k: v for k, v in inv.items() if k != "note"} for inv in self.data["invocations"]
            ],
        }


# ---------------------------------------------------------------------------
# Runner configuration / authorization
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Everything the frozen runner needs; all paths are LOCAL pinned copies.

    ``expected_keys_manifest`` is REQUIRED for any real load (the production
    CLI resolves it to the committed in-repo file); ``require_revision_in_path``
    is a TEST-ONLY escape the production CLI never disables. ``trim_mode`` /
    ``text_cache`` are optimization experiments, default OFF, each behind its
    same-real-pair equivalence gate (see module docstring).
    """

    checkpoint_path: str
    bert_dir: str
    cad_seq_zip: str
    pair_manifest_dir: str
    device: str = "npu:0"
    torch_dtype: str = "float32"
    batch_size: int = 8
    expected_keys_manifest: Optional[str] = None
    trim_mode: str = "none"  # "none" (audited reference) | "batch_max" (gated)
    text_cache: bool = False  # gated
    require_revision_in_path: bool = True
    shard_id: int = 0
    n_shards: int = 1
    texts_per_chunk: int = 1


def authorize_text2cad_run(
    repo_root,
    *,
    cards_used: int,
    expected_origin: str = EXPECTED_ORIGIN_URL,
    authorization_ref: str = AUTHORIZATION_REF,
) -> RunAuthorization:
    """Fail-closed authorization for the Text2CAD runner.

    Delegates the dual-version origin/master fetch + detached/clean HEAD +
    approved-sha equality checks to the audited ``cadmium_runner.authorize_run``
    and additionally pins the task id. ``expected_origin``/``authorization_ref``
    exist so CPU tests can use a real temporary Git remote; the production CLI
    never overrides them.
    """
    auth = authorize_run(
        repo_root,
        cards_used=cards_used,
        expected_origin=expected_origin,
        authorization_ref=authorization_ref,
    )
    if auth.task_id != TASK_ID:
        raise RunnerError(
            f"authorization: ACTIVE_TASK task_id {auth.task_id!r} != {TASK_ID!r}"
        )
    return auth


# ---------------------------------------------------------------------------
# Candidate store (zip -> validated CPU tensors; NO model forward)
# ---------------------------------------------------------------------------


class ZipCandidateStore:
    """Loads and validates all official candidates from the pinned cad_seq.zip.

    The zip SHA256 is verified at open. ``preload()`` reads every store member
    with ``torch.load(..., weights_only=True)``, requires the exact official
    structure -- ``vec:{cad_vec int32 (272,2), flag_vec int32 (272,),
    index_vec int32 (272,)}`` + ``mask_cad_dict:{attn_mask float32 (271,271),
    key_padding_mask bool (272,2)}`` -- verifies ``attn_mask`` equals the
    canonical ``generate_attention_mask(271)`` elementwise (a per-candidate
    drifted mask is rejected, never used), and runs the audited stage-A
    ``validate_cad_vec``. Candidates are kept on CPU; the shared canonical
    attn_mask is stored once. A candidate failing validation is recorded with
    its stable reason and its pairs are ledgered as ``rejected`` WITHOUT any
    forward or budget charge.
    """

    def __init__(self, zip_path, members: Sequence[str], expected_sha256: str = CAD_SEQ_ZIP_SHA256) -> None:
        self.zip_path = Path(zip_path)
        self.members = list(members)
        # ``expected_sha256`` defaults to the pinned official zip; overriding
        # it is a TEST-ONLY escape (the production CLI never passes it).
        self._zip_report = verify_zip_members(self.zip_path, self.members, expected_sha256)
        self.canonical_attn_mask = generate_attention_mask(MAX_CAD_SEQUENCE_LENGTH - 1)
        self._candidates: list[Optional[dict[str, torch.Tensor]]] = [None] * len(self.members)
        self.rejections: dict[int, str] = {}

    @property
    def zip_report(self) -> dict[str, Any]:
        return dict(self._zip_report)

    def preload(self) -> dict[str, Any]:
        # This archive has 528,161 entries. Parsing its central directory
        # once per candidate takes hours even before the first model forward.
        with zipfile.ZipFile(self.zip_path) as archive:
            for cad_index, member in enumerate(self.members):
                try:
                    self._candidates[cad_index] = self._load_one(member, archive)
                except ScorerInputError as exc:
                    self.rejections[cad_index] = exc.reason
                except (ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as exc:
                    self.rejections[cad_index] = f"candidate_structure:{type(exc).__name__}"
        return {
            "n_candidates": len(self.members),
            "n_loaded": sum(1 for c in self._candidates if c is not None),
            "n_rejected": len(self.rejections),
            "rejection_reasons": sorted(set(self.rejections.values())),
        }

    def _load_one(self, member: str, archive: zipfile.ZipFile) -> dict[str, torch.Tensor]:
        blob = archive.read(member)
        sample = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        if not isinstance(sample, dict) or set(sample.keys()) != {"vec", "mask_cad_dict"}:
            raise ScorerInputError("bad_shape", f"{member}: top-level keys must be vec+mask_cad_dict")
        vec_dict = sample["vec"]
        mask_dict = sample["mask_cad_dict"]
        if set(vec_dict.keys()) != {"cad_vec", "flag_vec", "index_vec"}:
            raise ScorerInputError("bad_shape", f"{member}: vec keys drift")
        if set(mask_dict.keys()) != {"attn_mask", "key_padding_mask"}:
            raise ScorerInputError("bad_shape", f"{member}: mask_cad_dict keys drift")
        cad_vec = vec_dict["cad_vec"]
        flag_vec = vec_dict["flag_vec"]
        index_vec = vec_dict["index_vec"]
        kpm = mask_dict["key_padding_mask"]
        attn = mask_dict["attn_mask"]
        expected = (
            (cad_vec, torch.int32, (MAX_CAD_SEQUENCE_LENGTH, 2)),
            (flag_vec, torch.int32, (MAX_CAD_SEQUENCE_LENGTH,)),
            (index_vec, torch.int32, (MAX_CAD_SEQUENCE_LENGTH,)),
            (kpm, torch.bool, (MAX_CAD_SEQUENCE_LENGTH, 2)),
            (attn, torch.float32, (MAX_CAD_SEQUENCE_LENGTH - 1, MAX_CAD_SEQUENCE_LENGTH - 1)),
        )
        for tensor, dtype, shape in expected:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ScorerInputError(
                    "bad_shape",
                    f"{member}: expected {dtype} {shape}, got "
                    f"{getattr(tensor, 'dtype', type(tensor).__name__)} "
                    f"{tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else ''}",
                )
        if not torch.equal(attn, self.canonical_attn_mask):
            raise ScorerInputError(
                "mask_mismatch",
                f"{member}: attn_mask differs from the canonical causal+START-shield mask",
            )
        diag = validate_cad_vec(cad_vec, kpm, flag_vec, index_vec)
        return {
            "cad_vec": cad_vec,
            "flag_vec": flag_vec,
            "index_vec": index_vec,
            "key_padding_mask": kpm,
            "n_real_tokens": diag.n_real_tokens,
        }

    def get(self, cad_index: int) -> dict[str, torch.Tensor]:
        candidate = self._candidates[cad_index]
        if candidate is None:
            raise ScorerInputError(
                self.rejections.get(cad_index, "candidate_structure:not_loaded"),
                f"cad_index {cad_index} was rejected at preload",
            )
        return candidate


# ---------------------------------------------------------------------------
# Text embedding cache (optimization; default OFF, equivalence-gated)
# ---------------------------------------------------------------------------


class BertTextCache:
    """Per-unique-text cache of frozen BERT outputs (Stage-B2 experiment).

    ``get(text)`` encodes a single text (no padding) on first use and caches
    the REAL-token slice ``embedding[:, :n_real]`` plus the token count.
    ``embed_batch(texts)`` re-pads cached embeddings with ZEROS to the batch
    width and builds the boolean key padding mask -- downstream, padded
    positions are suppressed by the additive cross-attention mask and the
    adaptive-layer key padding mask, so real-position outputs are unchanged
    in exact arithmetic. Equivalence on the real device within the
    pre-registered tolerances is a G1 gate (``text2cad_speed_bench``
    ``cache-check``); until it passes, production runs keep ``text_cache``
    off. The cache is per-process and holds at most n_unique_text entries.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, tuple[torch.Tensor, int]] = {}
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> dict[str, int]:
        """Cache counters for gate reports (additive; no scoring-math role)."""
        return {"entries": len(self._cache), "hits": self._hits, "misses": self._misses}

    def get(self, text: str) -> tuple[torch.Tensor, int]:
        hit = self._cache.get(text)
        if hit is None:
            self._misses += 1
            embedding, kpm = self._embedder.get_embedding([text])
            n_real = int((~kpm[0]).sum())
            hit = (embedding[0, :n_real].clone(), n_real)
            self._cache[text] = hit
        else:
            self._hits += 1
        return hit

    def embed_batch(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        entries = [self.get(t) for t in texts]
        width = max(n for _, n in entries)
        dim = entries[0][0].shape[-1]
        device = entries[0][0].device
        dtype = entries[0][0].dtype
        batch = torch.zeros(len(entries), width, dim, dtype=dtype, device=device)
        kpm = torch.ones(len(entries), width, dtype=torch.bool, device=device)
        for row, (emb, n_real) in enumerate(entries):
            batch[row, :n_real] = emb
            kpm[row, :n_real] = False
        return batch, kpm


class _CachedEmbedder(torch.nn.Module):
    """Adapter giving a BertTextCache the TextEmbedder.get_embedding protocol.

    Registered as a plain (parameter-free) submodule so it can replace
    ``model.base_text_embedder`` without disturbing nn.Module bookkeeping.
    """

    def __init__(self, cache: BertTextCache) -> None:
        super().__init__()
        self._cache = cache

    def get_embedding(self, texts):
        return self._cache.embed_batch(list(texts))


# ---------------------------------------------------------------------------
# Batch assembly + scoring (the only forward path)
# ---------------------------------------------------------------------------


def assemble_batch(
    store: ZipCandidateStore,
    cad_indices: Sequence[int],
    device: torch.device,
    trim_mode: str = "none",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Build the official teacher-forcing input tensors for one batch.

    Returns ``(vec_dict, mask_cad_dict, cad_seq_len)`` where ``cad_seq_len``
    is the number of INPUT rows (N-1). ``trim_mode="none`` keeps the official
    padded length 272 -> 271 input rows (the audited reference path);
    ``trim_mode="batch_max"`` slices every tensor to the batch's maximum real
    length L (>= 2) and the canonical attn_mask to its top-left (L-1)x(L-1)
    submatrix -- mathematically the same submatrix as
    ``generate_attention_mask(L-1)`` -- for the gated tail-trim experiment.
    """
    candidates = [store.get(c) for c in cad_indices]
    n_rows = MAX_CAD_SEQUENCE_LENGTH
    if trim_mode == "batch_max":
        n_rows = max(int(c["n_real_tokens"]) for c in candidates)
        n_rows = max(n_rows, 2)
    elif trim_mode != "none":
        raise RunnerError(f"unknown trim_mode {trim_mode!r}")
    vec_dict = {
        "cad_vec": torch.stack([c["cad_vec"][:n_rows] for c in candidates]).to(device),
        "flag_vec": torch.stack([c["flag_vec"][:n_rows] for c in candidates]).to(device),
        "index_vec": torch.stack([c["index_vec"][:n_rows] for c in candidates]).to(device),
    }
    mask_cad_dict = {
        "attn_mask": store.canonical_attn_mask[: n_rows - 1, : n_rows - 1].to(device),
        "key_padding_mask": torch.stack(
            [c["key_padding_mask"][:n_rows] for c in candidates]
        ).to(device),
    }
    return vec_dict, mask_cad_dict, n_rows - 1


def forward_and_score_batch(
    model,
    store: ZipCandidateStore,
    texts: Sequence[str],
    cad_indices: Sequence[int],
    device: torch.device,
    trim_mode: str = "none",
) -> list[dict[str, Any]]:
    """One teacher-forcing forward + audited CPU scoring for one batch.

    The forward runs under ``torch.inference_mode()`` on ``device``; raw
    logits are cast to FP32 on device and transferred to CPU (at most
    B*271*2*267*4 B -- ~0.6 MB per pair -- so the AUDITED stage-A CPU scoring
    kernel is used verbatim; there is no separate device-side gather to
    review). Returns one audited score dict per pair, in input order. The
    text side is whatever ``model.base_text_embedder`` provides (the direct
    pinned embedder, or the gated BertTextCache adapter installed by
    ``FrozenMatrixScorer``).
    """
    vec_dict, mask_cad_dict, cad_seq_len = assemble_batch(
        store, cad_indices, device, trim_mode=trim_mode
    )
    with torch.inference_mode():
        pred, _cad_vec_target, _shifted = teacher_forced_logits(
            model, list(texts), vec_dict, mask_cad_dict, cad_seq_len=cad_seq_len
        )
    logits_cpu = pred.to(torch.float32).cpu()
    cad_vec_cpu = vec_dict["cad_vec"].cpu()
    kpm_cpu = mask_cad_dict["key_padding_mask"].cpu()
    results = two_head_factorized_score(logits_cpu, cad_vec_cpu, kpm_cpu)
    assert isinstance(results, list)
    return results


def score_pair_token_diagnostic(
    logits_cpu: torch.Tensor, cad_vec: torch.Tensor
) -> dict[str, Any]:
    """Per-valid-token FP32 log-prob detail for the G1 one-pair spot check.

    Diagnostic only (the primary score is the audited aggregate); recomputed
    from the SAME transferred logits with the same FP32 log_softmax kernel.
    """
    target = cad_vec[1:].to(torch.long)
    valid = target != 0
    log_probs = torch.nn.functional.log_softmax(logits_cpu.to(torch.float32), dim=-1)
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    out: dict[str, Any] = {"x": [], "y": []}
    for head, name in ((0, "x"), (1, "y")):
        positions = valid[:, head].nonzero(as_tuple=True)[0].tolist()
        out[name] = [
            {
                "position": int(pos),
                "label": int(target[pos, head]),
                "logprob": float(gathered[pos, head].to(torch.float64)),
            }
            for pos in positions
        ]
    return out


# ---------------------------------------------------------------------------
# Chunk planning, atomic publish, resume
# ---------------------------------------------------------------------------


def new_invocation_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]


def ensure_outside_checkout(path, repo_root, label: str) -> None:
    resolved = Path(path).resolve()
    root = Path(repo_root).resolve()
    if resolved == root or root in resolved.parents:
        raise RunnerError(f"{label} must be outside the Git checkout: {resolved}")


def atomic_publish(path, lines: Sequence[str]) -> None:
    """Publish a completed chunk exactly once (hard-link, fail-if-exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RunnerError(f"output already exists, refusing overwrite: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RunnerError(
                f"output appeared during run, refusing overwrite: {path}"
            ) from exc
        except OSError as exc:
            raise RunnerError(f"cannot atomically publish {path}: {exc}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


_CHUNK_NAME_RE = re.compile(r"^chunk_s(\d+)_t(\d+)_(\d+)\.jsonl$")


def plan_chunks(
    out_dir,
    pair_space: PairSpace,
    cfg: RunnerConfig,
    binding: dict[str, Any],
    resume: bool,
    *,
    probe_text_indices: Optional[set[int]] = None,
    extra_binding_fields: Sequence[str] = (),
    timed_text_indices: Optional[set[int]] = None,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Decide which chunks still need scoring; verify completed ones.

    Returns ``(todo, completed_names)`` where ``todo`` is a list of
    ``(t0, t1, pair_start)``. Existing chunk files must ALL belong to this
    shard, parse as valid chunks, carry the current binding, and hold only
    final statuses; anything else is a loud refusal (never silently skipped
    or re-scored). Without ``resume`` any pre-existing chunk file refuses.

    Probe scope (G3; defaults keep the full-matrix behavior EXACTLY):
    ``probe_text_indices`` restricts the plan to the single-text chunks whose
    global text index is probed (requires ``texts_per_chunk == 1``; a chunk
    for any other text -- e.g. a full-matrix leftover -- is "outside this
    shard's plan" and refuses); ``extra_binding_fields`` makes the scope
    field part of chunk verification in BOTH directions (a full-matrix chunk
    lacks the probe SHA, a probe chunk carries it, and the two never verify
    under the other's binding); ``timed_text_indices`` is the set of query
    text indices with a complete sidecar timing line -- a completed chunk
    without one is unverifiable evidence and refuses (stop, never fabricate
    times).
    """
    if probe_text_indices is not None and cfg.texts_per_chunk != 1:
        raise RunnerError(
            f"probe filtering requires texts_per_chunk=1, got {cfg.texts_per_chunk}"
        )
    out_dir = Path(out_dir)
    existing = sorted(p.name for p in out_dir.glob("chunk_s*_t*_*.jsonl")) if out_dir.is_dir() else []
    if existing and not resume:
        raise RunnerError(
            f"output dir {out_dir} already contains {len(existing)} chunk file(s); "
            "re-running would duplicate scores and double-charge the budget. "
            "Pass --resume to continue the SAME shard after verification, or stop."
        )
    planned = {
        chunk_file_name(cfg.shard_id, t0, t1): (t0, t1, pair_start)
        for _ordinal, t0, t1, pair_start in pair_space.iter_chunks(
            cfg.n_shards, cfg.shard_id, cfg.texts_per_chunk
        )
    }
    if probe_text_indices is not None:
        planned = {name: value for name, value in planned.items() if value[0] in probe_text_indices}
    completed: list[str] = []
    problems: list[str] = []
    for name in existing:
        match = _CHUNK_NAME_RE.match(name)
        if not match or int(match.group(1)) != cfg.shard_id:
            problems.append(f"foreign chunk file {name} (not this shard's naming)")
            continue
        t0, t1 = int(match.group(2)), int(match.group(3))
        if name not in planned:
            problems.append(f"chunk file {name} is outside this shard's plan")
            continue
        records = []
        with (out_dir / name).open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        issues = verify_chunk_records(
            records, pair_space, cfg.shard_id, t0, t1, binding,
            extra_binding_fields=extra_binding_fields,
        )
        if issues:
            problems.append(f"chunk {name} failed verification: {issues[:3]}")
            continue
        if timed_text_indices is not None and t0 not in timed_text_indices:
            problems.append(
                f"chunk {name} is complete but has no sidecar timing record; "
                "the timing evidence is incomplete -- refusing to reuse it"
            )
            continue
        completed.append(name)
    if problems:
        raise RunnerError("chunk verification failed: " + "; ".join(problems[:5]))
    todo = [planned[name] for name in sorted(planned) if name not in set(completed)]
    return todo, completed


def load_probe_sidecar_timed_texts(
    out_dir,
    cfg: RunnerConfig,
    selection: dict[str, Any],
    scope: dict[str, Any],
    *,
    resume: bool,
) -> Optional[set[int]]:
    """G3 R1: the probe sidecar gate, always BEFORE any new forward.

    A FRESH probe run refuses an existing timing sidecar outright -- appending
    to leftover timing lines would mix evidence across runs. A RESUME reads
    the sidecar with the STRICT per-record validation (this shard's fixed
    probe under the current scope; foreign/drifted/duplicated/torn or
    count-invalid lines refuse) and returns the timed query text set (empty
    when no sidecar exists yet). Returns None for a fresh run (no timing
    requirements). The caller then cross-checks the returned set one-to-one
    against the verified chunks.
    """
    sidecar = Path(out_dir) / timing_sidecar_name(cfg.shard_id)
    if not resume:
        if sidecar.exists():
            raise RunnerError(
                f"fresh probe run but timing sidecar {sidecar} already exists; "
                "refusing to mix timing evidence across runs -- pass --resume to "
                "continue the SAME shard after verification, or stop"
            )
        return None
    if not sidecar.exists():
        return set()
    return verify_probe_resume_records(
        read_timing_records(sidecar),
        shard_id=cfg.shard_id,
        device=cfg.device,
        selection=selection,
        expected_scope=scope,
        path_label=str(sidecar),
    )


def require_sidecar_chunk_correspondence(
    timed_texts: set[int],
    completed_names: Sequence[str],
    *,
    path_label: str,
) -> None:
    """G3 R1 resume: sidecar lines and verified chunks must pair one-to-one.

    ``plan_chunks`` already refuses a completed chunk WITHOUT its timing
    line; this refuses the reverse -- an orphan timing line whose query has
    no verified chunk. Such a line can never send the query back into the
    todo list (no re-scoring of timed-but-chunkless queries, no fabricated
    reconstruction of missing outputs).
    """
    completed_texts = set()
    for name in completed_names:
        match = _CHUNK_NAME_RE.match(name)
        if match is None:  # plan_chunks only returns verified same-shard names
            raise RunnerError(f"{path_label}: unexpected completed chunk name {name!r}")
        completed_texts.add(int(match.group(2)))
    orphans = sorted(set(timed_texts) - completed_texts)
    if orphans:
        raise RunnerError(
            f"{path_label}: sidecar timing has no verified chunk for "
            f"{len(orphans)} query(ies), e.g. text_index {orphans[:3]}; the "
            "timing evidence and the published chunks disagree -- refusing to "
            "re-score them"
        )


# ---------------------------------------------------------------------------
# Environment / summary records
# ---------------------------------------------------------------------------


def environment_record(cfg: RunnerConfig, extra: dict[str, Any]) -> dict[str, Any]:
    """Version/binding facts for the run ledger (never secret VALUES)."""

    def _version(dist: str) -> Optional[str]:
        try:
            from importlib.metadata import version

            return version(dist)
        except Exception:
            return None

    record = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_npu_version": _version("torch-npu"),
        "transformers_version": _version("transformers"),
        "device": cfg.device,
        "torch_dtype": cfg.torch_dtype,
        "batch_size": cfg.batch_size,
        "trim_mode": cfg.trim_mode,
        "text_cache": cfg.text_cache,
        "shard_id": cfg.shard_id,
        "n_shards": cfg.n_shards,
        "texts_per_chunk": cfg.texts_per_chunk,
        "checkpoint_path": str(cfg.checkpoint_path),
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "bert_dir": str(cfg.bert_dir),
        "bert_file_sha256": dict(PINNED_BERT_FILE_SHA256),
        "cad_seq_zip": str(cfg.cad_seq_zip),
        "score_name": SCORE_NAME,
        "spec_version": SCORE_SPEC_VERSION,
        "max_cad_sequence_length": MAX_CAD_SEQUENCE_LENGTH,
        "env_vars_present": sorted(
            name
            for name in (
                "HF_HOME",
                "XDG_CACHE_HOME",
                "TMPDIR",
                "PYTHONDONTWRITEBYTECODE",
                "ASCEND_HOME",
                "ASCEND_TOOLKIT_HOME",
            )
            if name in os.environ
        ),
    }
    record.update(extra)
    return record


def git_sha_of(repo_root) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if proc.returncode != 0:
        raise RunnerError(f"cannot read git HEAD of {repo_root}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _peak_device_memory_bytes(device: str) -> Optional[int]:
    """Peak HBM of the EXPLICIT device; None only on non-NPU (CPU test) hosts.

    Mirrors the benchmark's helper (kept separate to avoid a circular import:
    the bench imports this module, never the reverse).
    """
    if device.startswith("npu") and hasattr(torch, "npu"):
        return int(torch.npu.max_memory_allocated(device))
    return None


def _reset_peak_device_memory(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# The scoring engine (model in hand; authorization already done by caller)
# ---------------------------------------------------------------------------


class FrozenMatrixScorer:
    """Scores one shard's chunks with a loaded frozen model.

    Constructed only after every gate passed. ``binding`` is the chunk-header
    binding dict (CHUNK_BINDING_FIELDS values); ``on_heartbeat`` is invoked
    between batches so the CLI can beat the ledger clock.
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        model,
        texts: Sequence[str],
        store: ZipCandidateStore,
        pair_space: PairSpace,
        ledger: SharedBudgetLedger,
        invocation_id: str,
        binding: dict[str, Any],
    ) -> None:
        if cfg.batch_size < 1:
            raise RunnerError(f"batch_size must be >= 1, got {cfg.batch_size}")
        self.cfg = cfg
        self.model = model
        self.texts = list(texts)
        self.store = store
        self.pair_space = pair_space
        self.ledger = ledger
        self.invocation_id = invocation_id
        self.binding = dict(binding)
        self.device = torch.device(cfg.device)
        # Gated optimization: with text_cache enabled, wrap the model's pinned
        # embedder ONCE with the unique-text cache adapter (deterministic;
        # recorded in every chunk header via binding["text_cache"]).
        self._text_cache = BertTextCache(model.base_text_embedder) if cfg.text_cache else None
        if self._text_cache is not None:
            self.model.base_text_embedder = _CachedEmbedder(self._text_cache)
        self.stats = {"batches": 0, "pairs_forwarded": 0, "pairs_rejected": 0}

    def score_chunk(self, t0: int, t1: int) -> list[dict[str, Any]]:
        """Score every pair of text range [t0, t1); returns header+records."""
        n_cads = self.pair_space.n_cads
        records: list[dict[str, Any]] = []
        pending: list[tuple[int, int, int]] = []  # (pair_index, text_index, cad_index)
        for pair_index in self.pair_space.chunk_pair_indices(t0, t1):
            text_index, cad_index = self.pair_space.coords_of(pair_index)
            if cad_index in self.store.rejections:
                records.append(
                    self._record(pair_index, text_index, cad_index, STATUS_REJECTED,
                                 self.store.rejections[cad_index], None)
                )
                self.stats["pairs_rejected"] += 1
            else:
                pending.append((pair_index, text_index, cad_index))
        for start in range(0, len(pending), self.cfg.batch_size):
            batch = pending[start : start + self.cfg.batch_size]
            self.ledger.charge_pairs(self.invocation_id, len(batch))
            self.ledger.heartbeat(self.invocation_id)
            batch_texts = [self.texts[text_index] for _, text_index, _ in batch]
            batch_cads = [cad_index for _, _, cad_index in batch]
            results = forward_and_score_batch(
                self.model,
                self.store,
                batch_texts,
                batch_cads,
                self.device,
                trim_mode=self.cfg.trim_mode,
            )
            for (pair_index, text_index, cad_index), score in zip(batch, results):
                records.append(
                    self._record(pair_index, text_index, cad_index, STATUS_SCORED, None, score)
                )
            self.stats["batches"] += 1
            self.stats["pairs_forwarded"] += len(batch)
        # Records must be in dense pair_index order regardless of rejection
        # interleaving with forwarded batches.
        records.sort(key=lambda r: r["pair_index"])
        header = {
            "record_type": "chunk_header",
            "shard_id": self.cfg.shard_id,
            "text_range": [t0, t1],
            "n_pairs": (t1 - t0) * n_cads,
            "pair_index_first": t0 * n_cads,
            "pair_index_last": t1 * n_cads - 1,
            "n_scored": sum(1 for r in records if r["status"] == STATUS_SCORED),
            "n_rejected": sum(1 for r in records if r["status"] == STATUS_REJECTED),
            **{field: self.binding.get(field) for field in CHUNK_BINDING_FIELDS},
        }
        # Probe scope (G3): a scoped run stamps every chunk with the probe
        # selection SHA; the unscoped full-matrix header is byte-unchanged.
        if self.binding.get(PROBE_SCOPE_FIELD) is not None:
            header[PROBE_SCOPE_FIELD] = self.binding[PROBE_SCOPE_FIELD]
        return [header, *records]

    def _record(
        self,
        pair_index: int,
        text_index: int,
        cad_index: int,
        status: str,
        reason: Optional[str],
        score: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "pair",
            "pair_index": pair_index,
            "text_index": text_index,
            "cad_index": cad_index,
            "status": status,
            "reason": reason,
            "score_name": SCORE_NAME,
            "spec_version": SCORE_SPEC_VERSION,
            "score_x": None,
            "score_y": None,
            "score_total": None,
            "n_valid_x": None,
            "n_valid_y": None,
            "n_valid_total": None,
            "valid_seq_len": None,
            "n_real_tokens": None,
            "mean_x": None,
            "mean_y": None,
            "mean_total": None,
        }
        if score is not None:
            for key in (
                "score_x",
                "score_y",
                "score_total",
                "n_valid_x",
                "n_valid_y",
                "n_valid_total",
                "valid_seq_len",
                "n_real_tokens",
                "mean_x",
                "mean_y",
                "mean_total",
            ):
                record[key] = score[key]
        return record


def execute_probe_query(
    scorer: FrozenMatrixScorer,
    out_dir,
    timing_path,
    probe_facts: dict[int, dict[str, int]],
    t0: int,
    t1: int,
) -> None:
    """G3 probe: score ONE query, publish its chunk, THEN append its timing line.

    The measured window starts at the real scoring start and ends only AFTER
    the chunk's atomic publish completed (score + serialization + disk). No
    warmup query is added; a complete timing line always implies its chunk
    exists. This is the exact loop body the production CLI runs per probe
    query, factored so CPU tests drive the REAL producer.
    """
    query_start_epoch = time.time()
    batches_before = scorer.stats["batches"]
    chunk_records = scorer.score_chunk(t0, t1)
    chunk_name = chunk_file_name(scorer.cfg.shard_id, t0, t1)
    atomic_publish(
        Path(out_dir) / chunk_name,
        [json.dumps(r, ensure_ascii=False) for r in chunk_records],
    )
    query_end_epoch = time.time()
    header = chunk_records[0]
    append_timing_record(
        timing_path,
        build_query_timing_record(
            invocation_id=scorer.invocation_id,
            shard_id=scorer.cfg.shard_id,
            device=scorer.cfg.device,
            text_index=t0,
            probe_facts=probe_facts[t0],
            start_epoch=query_start_epoch,
            end_epoch=query_end_epoch,
            n_pairs=header["n_pairs"],
            n_batches=scorer.stats["batches"] - batches_before,
            n_scored=header["n_scored"],
            n_rejected=header["n_rejected"],
            n_valid_total=sum(
                int(r["n_valid_total"])
                for r in chunk_records[1:]
                if r["status"] == STATUS_SCORED
            ),
            chunk_file=chunk_name,
            scope=scorer.binding,
        ),
    )


def build_run_summary(
    cfg: RunnerConfig,
    *,
    invocation_id: str,
    authorization: RunAuthorization,
    published: int,
    n_previously_completed: int,
    planned_pairs: int,
    stats: dict[str, Any],
    load_report: dict[str, Any],
    preload_report: dict[str, Any],
    ledger: SharedBudgetLedger,
    stale_finalized: Sequence[str],
    failure: Optional[str],
    probe_selection: Optional[dict[str, Any]] = None,
    probe_sha: Optional[str] = None,
    pair_manifest_sha256: Optional[str] = None,
    cold_start_s: Optional[float] = None,
) -> dict[str, Any]:
    """The exact run-summary assembly of the production CLI (factored for tests).

    Probe runs additionally record the code/asset/scope fingerprints, the
    timing sidecar name, the measured cold start and the per-device peak HBM
    reading. ``environment_record`` stamps the fixed configuration identity.
    """
    summary_extra: dict[str, Any] = {
        "invocation_id": invocation_id,
        "authorization": authorization.to_dict(),
        "chunks_completed_now": published,
        "chunks_previously_completed": n_previously_completed,
        "pairs_planned": planned_pairs,
        "stats": stats,
        "load_report": load_report,
        "candidate_preload": preload_report,
        "budget": ledger.to_dict(),
        "stale_invocations_auto_finalized": stale_finalized,
        "status": "failed" if failure else "ok",
        "failure": failure,
    }
    if probe_selection is not None:
        summary_extra.update(
            {
                "git_sha": authorization.approved_sha,
                "pair_manifest_sha256": pair_manifest_sha256,
                PROBE_SCOPE_FIELD: probe_sha,
                "timing_file": timing_sidecar_name(cfg.shard_id),
                "cold_start_s": cold_start_s,
                "peak_device_memory_bytes": _peak_device_memory_bytes(cfg.device),
                "hbm_device": cfg.device,
                "probe_queries_total": published + n_previously_completed,
            }
        )
    return environment_record(cfg, summary_extra)


# ---------------------------------------------------------------------------
# Shared gated loaders (used by the matrix CLI and the benchmark CLI alike --
# one implementation of the asset gates so the two entries cannot diverge)
# ---------------------------------------------------------------------------


def check_revision_paths(cfg: RunnerConfig) -> None:
    """Pinned-revision path check (production always on; test-only escape)."""
    if not cfg.require_revision_in_path:
        return
    for label, path, revision in (
        ("checkpoint", cfg.checkpoint_path, EXPECTED_MODEL_REVISION),
        ("bert", cfg.bert_dir, EXPECTED_BERT_REVISION),
        ("cad_seq_zip", cfg.cad_seq_zip, EXPECTED_DATA_REVISION),
    ):
        if revision not in str(path):
            raise RunnerError(
                f"preflight: {label} path {path} lacks pinned revision {revision}"
            )


def load_gated_model_and_store(cfg: RunnerConfig, members: Optional[Sequence[str]] = None):
    """Gate 5: strict model load + candidate preload (all asset SHA gates).

    ``load_frozen_text2cad`` verifies checkpoint/BERT SHAs, the 172-key
    three-way manifest, zero-tensor policy and per-tensor numeric equality;
    ``ZipCandidateStore.preload`` verifies the zip SHA, member presence and
    every candidate's structure/mask. Returns
    ``(model, load_report, store, preload_report)``.
    """
    model, load_report = load_frozen_text2cad(
        cfg.checkpoint_path,
        cfg.bert_dir,
        cfg.device,
        expected_keys_manifest=cfg.expected_keys_manifest,
    )
    if cfg.torch_dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    if members is None:
        members = load_cad_store(cfg.pair_manifest_dir)
    store = ZipCandidateStore(cfg.cad_seq_zip, members)
    preload_report = store.preload()
    if preload_report["n_rejected"]:
        raise RunnerError(
            f"candidate preload rejected {preload_report['n_rejected']} of "
            f"{preload_report['n_candidates']} official gallery CADs; "
            f"reasons={preload_report['rejection_reasons']}. "
            "Refusing an incomplete exact score matrix before any forward."
        )
    return model, load_report, store, preload_report


# ---------------------------------------------------------------------------
# Dry-run (zero real forward; allowed at forward budget 0)
# ---------------------------------------------------------------------------


def dry_run(
    cfg: RunnerConfig,
    out_dir,
    *,
    scan_candidates: bool = False,
    expected_sha256: Optional[dict[str, str]] = None,
    zip_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Zero-forward manifest/asset audit; NEVER runs a model forward.

    Verifies: pair-manifest artifacts against reviewed fingerprints, text/CAD
    store schemas and counts, shard/chunk math, zip SHA + member presence
    (optionally the full per-candidate structural scan), checkpoint file SHA,
    the strict zero-forward checkpoint load against the committed 172-key
    manifest (structure only, random-init model overwritten by the load,
    StubEmbedder -- the BERT model itself is not needed for this), BERT file
    SHAs, and -- when transformers is importable -- the per-text BERT token
    length audit with the pinned tokenizer (any text > 512 tokens fails the
    dry run). Writes ``dry_run.json`` (refuses to overwrite).
    """
    manifest = load_pair_manifest(cfg.pair_manifest_dir, expected_sha256=expected_sha256)
    texts = load_texts(cfg.pair_manifest_dir)
    members = load_cad_store(cfg.pair_manifest_dir)
    pair_space = PairSpace(
        n_texts=manifest["counts"]["n_unique_text"], n_cads=manifest["counts"]["n_gallery"]
    )
    if pair_space.n_pairs != manifest["counts"]["n_pairs"]:
        raise RunnerError("dry-run: pair count in manifest is not n_texts*n_cads")
    # ``zip_sha256`` override is TEST-ONLY (the CLI never exposes it);
    # production dry-runs verify the pinned official zip.
    zip_report = verify_zip_members(cfg.cad_seq_zip, members, zip_sha256 or CAD_SEQ_ZIP_SHA256)

    report: dict[str, Any] = {
        "status": "dry_run_no_forward",
        "pair_manifest": {
            "dir": str(cfg.pair_manifest_dir),
            "counts": manifest["counts"],
            "artifacts": manifest["artifacts"],
        },
        "zip": zip_report,
        "shard": {
            "shard_id": cfg.shard_id,
            "n_shards": cfg.n_shards,
            "text_range": list(pair_space.shard_text_range(cfg.n_shards, cfg.shard_id)),
            "pair_range": list(pair_space.shard_pair_range(cfg.n_shards, cfg.shard_id)),
            "n_chunks": sum(
                1
                for _ in pair_space.iter_chunks(cfg.n_shards, cfg.shard_id, cfg.texts_per_chunk)
            ),
        },
        "forward_executed": False,
    }

    # Zero-forward checkpoint structure audit (audited stage-B1 loader).
    from .text2cad_port.loader import (
        build_text2cad,
        load_verified_checkpoint,
        nonbert_state_dict_keys,
        strict_load_frozen_weights,
        verify_bert_asset_shas,
    )

    class _StubEmbedder(torch.nn.Module):
        def get_embedding(self, texts):  # pragma: no cover - must never be called
            raise NotImplementedError("stub embedder must not be called")

    state, checkpoint_report = load_verified_checkpoint(cfg.checkpoint_path)
    torch.manual_seed(0)
    stub_model = build_text2cad(_StubEmbedder(), device="cpu")
    report["checkpoint"] = checkpoint_report
    report["model_nonbert_keys"] = len(nonbert_state_dict_keys(stub_model))
    report["strict_load"] = strict_load_frozen_weights(
        stub_model, state, expected_keys_manifest=cfg.expected_keys_manifest
    )
    report["bert_files"] = {
        name: info["sha256"]
        for name, info in verify_bert_asset_shas(cfg.bert_dir).items()
    }
    del stub_model, state

    if scan_candidates:
        store = ZipCandidateStore(
            cfg.cad_seq_zip, members, zip_sha256 or CAD_SEQ_ZIP_SHA256
        )
        report["candidate_scan"] = store.preload()

    try:
        from transformers import BertTokenizer  # lazy: torch-only hosts skip
    except ImportError:
        report["text_token_audit"] = {
            "skipped": "transformers not importable on this host; run on yunnan-share2"
        }
    else:
        tokenizer = BertTokenizer.from_pretrained(str(cfg.bert_dir), local_files_only=True)
        lengths = [int(sum(tokenizer(t)["attention_mask"])) for t in texts]
        over = [(i, n) for i, n in enumerate(lengths) if n > 512]
        report["text_token_audit"] = {
            "n_texts": len(texts),
            "max_tokens": max(lengths),
            "n_over_512": len(over),
        }
        if over:
            raise RunnerError(
                f"dry-run: {len(over)} text(s) exceed 512 BERT tokens, e.g. {over[:3]}; "
                "refusing (no silent truncation)"
            )

    out_path = Path(out_dir) / "dry_run.json"
    atomic_publish(out_path, [json.dumps(report, indent=2, sort_keys=True)])
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_expected_pair_hashes(path) -> dict[str, str]:
    """Reviewed artifact fingerprints (committed config -> {file: sha256}).

    The config carries ``artifact_sha256`` (worker files only) plus
    ``pair_manifest_sha256``; both are returned so ``load_pair_manifest`` can
    verify the manifest file itself against the reviewed value.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RunnerError(f"cannot read expected pair-manifest hashes {path}: {exc}") from exc
    artifacts = data.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RunnerError(f"{path}: missing artifact_sha256 object")
    out = {str(k): str(v) for k, v in artifacts.items()}
    manifest_sha = data.get("pair_manifest_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise RunnerError(f"{path}: missing/invalid pair_manifest_sha256")
    out[PAIR_MANIFEST_FILE] = manifest_sha
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Text2CAD full-matrix scorer (M1c B2). Fail-closed BEFORE "
            "any model forward: (1) pair-manifest artifacts must match the "
            "reviewed committed fingerprints; (2) the FRESHLY FETCHED "
            "origin/master ACTIVE_TASK.md (origin URL validated; never the "
            "working tree) must state pilot_authorized with a full 40-hex "
            "approved_sha equal to the live DETACHED clean HEAD, task_id "
            "idea01_text2cad_benchmark_20260928, and positive "
            "max_scored_pairs/gpu_budget; (3) --expected-keys-manifest must "
            "be the committed 172-key manifest; (4) the shared two-card "
            "budget ledger must fit every batch before it forwards. "
            "--dry-run is the only forward-free mode (zero budget, zero "
            "forward). There are no switches to relax these gates."
        )
    )
    parser.add_argument("--mode", choices=["score", "dry-run"], default="dry-run")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bert-dir", required=True)
    parser.add_argument("--cad-seq-zip", required=True)
    parser.add_argument("--pair-manifest-dir", required=True)
    parser.add_argument("--expected-keys-manifest", required=True,
                        help="REQUIRED committed 172-key manifest (never omitted)")
    parser.add_argument("--expected-pair-hashes", default=None,
                        help="reviewed artifact fingerprints; production default is "
                             "the committed configs/text2cad_pair_manifest_expected_20260930.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--texts-per-chunk", type=int, default=1)
    parser.add_argument("--cards-used", type=int, default=1, choices=[1],
                        help="one device per process; two concurrent processes each charge one card")
    parser.add_argument("--trim-mode", default="none", choices=["none", "batch_max"])
    parser.add_argument("--text-cache", action="store_true",
                        help="gated experiment: unique-text BERT cache (default off)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-selection", default=None,
                        help="G3 probe scope: must be the committed reviewed in-repo "
                             "probe selection config; without it the full-matrix "
                             "behavior is unchanged")
    parser.add_argument("--token-audit", default=None,
                        help="G3 probe scope: reviewed derived token-length audit JSON "
                             "the committed probe selection is bound to")
    parser.add_argument("--scan-candidates", action="store_true",
                        help="dry-run: also re-validate all candidates (CPU, zero forward)")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    try:
        repo_root = Path(args.repo_root).resolve()
        expected_pair_hashes = args.expected_pair_hashes
        if expected_pair_hashes is None:
            expected_pair_hashes = str(repo_root / EXPECTED_PAIR_HASHES_REL)
        if args.probe_selection is not None and args.mode != "score":
            raise RunnerError(
                "--probe-selection scopes a scoring run; --mode dry-run never forwards"
            )
        if args.token_audit is not None and args.probe_selection is None:
            raise RunnerError("--token-audit is only meaningful with --probe-selection")
        if args.mode == "score":
            canonical_hashes = (repo_root / EXPECTED_PAIR_HASHES_REL).resolve()
            if Path(expected_pair_hashes).resolve() != canonical_hashes:
                raise RunnerError(
                    f"forward mode requires the committed reviewed pair fingerprints "
                    f"{canonical_hashes}"
                )
            if args.trim_mode != "none" or args.text_cache or args.torch_dtype != "float32":
                raise RunnerError(
                    "matrix forward is restricted to the reviewed float32, "
                    "full-length, uncached reference path until G1 equivalence "
                    "and dtype evidence are separately approved"
                )
            if args.n_shards != 2 or args.shard_id not in (0, 1) or args.device != f"npu:{args.shard_id}":
                raise RunnerError("two-card matrix requires shard 0 on npu:0 and shard 1 on npu:1")
            if args.probe_selection is not None:
                if args.texts_per_chunk != G3_TEXTS_PER_CHUNK:
                    raise RunnerError(
                        f"G3 probe keeps texts_per_chunk={G3_TEXTS_PER_CHUNK}, "
                        f"got {args.texts_per_chunk}"
                    )
                if args.batch_size != G3_BATCH_SIZE:
                    raise RunnerError(
                        f"G3 probe is fixed to batch {G3_BATCH_SIZE} (plus the honest "
                        f"tail), got {args.batch_size}"
                    )
        fingerprints = _load_expected_pair_hashes(expected_pair_hashes)

        cfg = RunnerConfig(
            checkpoint_path=args.checkpoint,
            bert_dir=args.bert_dir,
            cad_seq_zip=args.cad_seq_zip,
            pair_manifest_dir=args.pair_manifest_dir,
            device=args.device,
            torch_dtype=args.torch_dtype,
            batch_size=args.batch_size,
            expected_keys_manifest=args.expected_keys_manifest,
            trim_mode=args.trim_mode,
            text_cache=bool(args.text_cache),
            shard_id=args.shard_id,
            n_shards=args.n_shards,
            texts_per_chunk=args.texts_per_chunk,
        )

        # Gate 1: pair-manifest artifacts match the reviewed fingerprints.
        manifest = load_pair_manifest(cfg.pair_manifest_dir, expected_sha256=fingerprints)
        texts = load_texts(cfg.pair_manifest_dir)
        members = load_cad_store(cfg.pair_manifest_dir)
        pair_space = PairSpace(
            n_texts=manifest["counts"]["n_unique_text"], n_cads=manifest["counts"]["n_gallery"]
        )

        out_dir = Path(args.out_dir).resolve()
        ensure_outside_checkout(out_dir, repo_root, "out_dir")

        # Gate 1b (G3 probe scope only): the probe selection must be the
        # committed reviewed in-repo file, re-derived from the LIVE verified
        # artifacts and bound to the actual token audit and the frozen G2
        # config. Without --probe-selection nothing in this run changes.
        probe_selection: Optional[dict[str, Any]] = None
        probe_sha: Optional[str] = None
        if args.probe_selection is not None:
            canonical_probe = (repo_root / G3_PROBE_SELECTION_REL).resolve()
            if Path(args.probe_selection).resolve() != canonical_probe:
                raise RunnerError(
                    f"probe scoring requires the committed reviewed probe selection "
                    f"{canonical_probe}"
                )
            if not canonical_probe.is_file():
                raise RunnerError(
                    f"committed probe selection {canonical_probe} does not exist"
                )
            try:
                probe_selection = json.loads(canonical_probe.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RunnerError(
                    f"committed probe selection {canonical_probe} is not valid JSON: {exc}"
                ) from exc
            verify_g3_probe_selection_against_manifest(probe_selection, cfg.pair_manifest_dir)
            verify_g3_token_audit(probe_selection, args.token_audit, cfg.pair_manifest_dir)
            verify_g3_g2_config_binding(probe_selection, repo_root)
            probe_sha = sha256_file(canonical_probe)

        if args.mode == "dry-run":
            report = dry_run(
                cfg, out_dir,
                scan_candidates=bool(args.scan_candidates),
                expected_sha256=fingerprints,
            )
            print(json.dumps({"status": report["status"], "counts": report["pair_manifest"]["counts"]}))
            return 0

        # Gate 2: authorization (fetched origin/master; detached clean HEAD).
        # Checked BEFORE the production-path pin so the gate itself is
        # exercisable from tests with real temporary repositories.
        authorization = authorize_text2cad_run(repo_root, cards_used=args.cards_used)
        if repo_root != PRODUCTION_REPO_ROOT:
            raise RunnerError(f"production checkout must be {PRODUCTION_REPO_ROOT}")
        keys_manifest = Path(args.expected_keys_manifest).resolve()
        if keys_manifest != (repo_root / EXPECTED_KEYS_MANIFEST_REL).resolve():
            raise RunnerError(
                f"expected-keys manifest must be the committed reviewed file "
                f"{repo_root / EXPECTED_KEYS_MANIFEST_REL}"
            )
        ensure_outside_checkout(
            canonical_budget_path(authorization.approved_sha), repo_root, "budget ledger"
        )

        # Gate 3: asset layout pins (revision-in-path) before any load.
        check_revision_paths(cfg)

        # Gate 4: budget ledger + plan before weights load.
        ledger = SharedBudgetLedger(
            canonical_budget_path(authorization.approved_sha),
            approved_sha=authorization.approved_sha,
            max_pairs=authorization.max_pairs,
            max_card_seconds=authorization.card_hours * 3600.0,
        )
        binding = {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "pair_manifest_sha256": fingerprints.get(PAIR_MANIFEST_FILE),
            "git_sha": authorization.approved_sha,
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "device": cfg.device,
            "torch_dtype": cfg.torch_dtype,
            "trim_mode": cfg.trim_mode,
            "text_cache": cfg.text_cache,
            "score_name": SCORE_NAME,
            "spec_version": SCORE_SPEC_VERSION,
        }
        if probe_sha is not None:
            binding[PROBE_SCOPE_FIELD] = probe_sha
        # Probe sidecar gate (G3 R1; BEFORE any new forward): a fresh run
        # refuses an existing sidecar; a resume strictly re-validates every
        # timing line of THIS shard's fixed probe under the current scope.
        timed_texts: Optional[set[int]] = None
        if probe_selection is not None:
            scope = {
                PROBE_SCOPE_FIELD: probe_sha,
                "git_sha": authorization.approved_sha,
                "pair_manifest_sha256": fingerprints.get(PAIR_MANIFEST_FILE),
                "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
                "score_name": SCORE_NAME,
                "spec_version": SCORE_SPEC_VERSION,
                "device": cfg.device,
            }
            timed_texts = load_probe_sidecar_timed_texts(
                out_dir, cfg, probe_selection, scope, resume=bool(args.resume)
            )
        todo, completed = plan_chunks(
            out_dir,
            pair_space,
            cfg,
            binding,
            resume=bool(args.resume),
            probe_text_indices=(
                set(probe_texts_for_shard(probe_selection, cfg.shard_id))
                if probe_selection is not None
                else None
            ),
            extra_binding_fields=(PROBE_SCOPE_FIELD,),
            timed_text_indices=timed_texts,
        )
        if probe_selection is not None:
            n_probe = len(probe_texts_for_shard(probe_selection, cfg.shard_id))
            if len(todo) + len(completed) != n_probe:
                raise RunnerError(
                    f"probe plan covers {len(todo) + len(completed)} chunks, expected "
                    f"exactly the shard's {n_probe} probe queries"
                )
            if args.resume:
                # sidecar lines and verified chunks must pair one-to-one: an
                # orphan timing line (no verified chunk) refuses here; a chunk
                # without its timing line already refused inside plan_chunks
                require_sidecar_chunk_correspondence(
                    timed_texts or set(), completed, path_label="probe resume"
                )
        planned_pairs = sum((t1 - t0) * pair_space.n_cads for t0, t1, _ in todo)
        if planned_pairs > ledger.remaining_pairs():
            raise RunnerError(
                f"planned {planned_pairs} pairs exceed remaining approved budget "
                f"{ledger.remaining_pairs()}"
            )
        ledger.check_card_clock()
        invocation_id = new_invocation_id()

        # Gate 5: strict load (all SHA/key/shape/zero/numeric gates; the
        # manifest is REQUIRED and pinned to the committed file above).
        stale_finalized = ledger.begin_invocation(invocation_id, cfg.shard_id, args.cards_used)
        cold_start_s: Optional[float] = None
        try:
            load_t0 = time.perf_counter()
            model, load_report, store, preload_report = load_gated_model_and_store(cfg, members)
            if probe_selection is not None:
                # G3: model + candidate load cold start, then a clean HBM
                # baseline for the scoring phase (formal path untouched).
                cold_start_s = time.perf_counter() - load_t0
                _reset_peak_device_memory(cfg.device)
        except Exception:
            ledger.finalize_invocation(invocation_id, exit_code=2, note="model_or_candidate_load_failed")
            raise
        scorer = FrozenMatrixScorer(
            cfg, model, texts, store, pair_space, ledger, invocation_id, binding
        )
        probe_facts = (
            probe_facts_for_shard(probe_selection, cfg.shard_id)
            if probe_selection is not None
            else None
        )
        timing_path = (
            out_dir / timing_sidecar_name(cfg.shard_id)
            if probe_selection is not None
            else None
        )
        exit_code = 0
        note = "ok"
        failure: Optional[str] = None
        published = 0
        try:
            for t0, t1, _pair_start in todo:
                if probe_selection is not None:
                    # G3 probe: real per-query window (score + serialization +
                    # atomic publish), timing line appended after the publish
                    execute_probe_query(scorer, out_dir, timing_path, probe_facts, t0, t1)
                else:
                    chunk_records = scorer.score_chunk(t0, t1)
                    atomic_publish(
                        out_dir / chunk_file_name(cfg.shard_id, t0, t1),
                        [json.dumps(r, ensure_ascii=False) for r in chunk_records],
                    )
                published += 1
        except Exception as exc:  # noqa: BLE001 -- ledger finalize must record ANY abort
            exit_code = 2
            note = "aborted"
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            ledger.finalize_invocation(invocation_id, exit_code=exit_code, note=note)

        summary = build_run_summary(
            cfg,
            invocation_id=invocation_id,
            authorization=authorization,
            published=published,
            n_previously_completed=len(completed),
            planned_pairs=planned_pairs,
            stats=scorer.stats,
            load_report=load_report,
            preload_report=preload_report,
            ledger=ledger,
            stale_finalized=stale_finalized,
            failure=failure,
            probe_selection=probe_selection,
            probe_sha=probe_sha,
            pair_manifest_sha256=fingerprints.get(PAIR_MANIFEST_FILE),
            cold_start_s=cold_start_s,
        )
        atomic_publish(
            out_dir / f"summary_{invocation_id}.json",
            [json.dumps(summary, indent=2, sort_keys=True)],
        )
        if failure:
            print(f"FATAL: {failure}", file=sys.stderr)
            return 2
        print(json.dumps({"status": "ok", "invocation_id": invocation_id,
                          "chunks": len(todo), "stats": scorer.stats}))
        return 0
    except (RunnerError, ScorerInputError, PairIndexError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
