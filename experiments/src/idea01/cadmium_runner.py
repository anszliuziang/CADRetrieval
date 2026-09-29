"""CADmium-3B NPU inference runner adapter (M1b).

Task contract: experiments/plans/IDEA01_M1B_CADMIUM_NPU_BENCH_2026-09-30.md
Scientific protocol: experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md
Scoring definition: experiments/src/idea01/cadmium_scorer.py (audited M1a) and
experiments/src/idea01/scoring.py (audited numpy math core).

AUTHORIZATION GATE (fail closed, machine-checked; review item 2 and the
2026-09-30 review addendum on the authorization SOURCE)
----------------------------------------------------------------
The production CLI (``main``) cannot be run on caller-supplied trust:

  * The authorization text is read from the Codex-controlled
    ``origin/master:experiments/ACTIVE_TASK.md`` AFTER ``git fetch origin``
    -- NEVER from the detached HEAD's working tree: a commit cannot contain
    its own ``approved_sha``, so a working-tree file can never prove
    authorization for the checked-out code. This reuses the audited
    ``p1_runner.fetch_authorized_active_task``: the ``origin`` remote URL is
    validated against the expected project remote before any fetch, and the
    approval commit SHA therefore differs from the approved code SHA by
    construction (both SHAs are recorded for audit). HEAD must be DETACHED
    (the production protocol is ``git checkout --detach <approved sha>``; an
    attached, movable branch refuses) and the worktree CLEAN. The fetched
    text must state ``state: pilot_authorized`` with a full 40-hex
    ``approved_sha`` equal to the live ``git rev-parse HEAD``, plus
    ``max_scored_pairs > 0`` and ``gpu_budget > 0`` card-hours; any parse
    ambiguity refuses the run. As of 2026-09-30 the real origin/master task
    file fails these checks. There is no CLI switch to relax the gate or to
    supply the authorization text; CPU tests build real temporary Git
    repositories (no injected fake task files or fake HEADs) and only
    construct ``RunAuthorization`` directly for scorer-level tests.
  * The approved pair cap and the derived wall-clock limit (approved
    card-hours * 3600 / cards in use) are bound to the persisted
    ``BudgetLedger`` together with the approved SHA; a ledger file written
    under a different approval refuses to load, and the wall clock keeps
    counting across restarts instead of resetting.
  * The module imports cleanly WITHOUT torch / transformers / peft (lazy
    imports inside the loaders) so local CPU tests can exercise every gate.

WHAT THIS RUNNER DOES (per contract items 2-3)
----------------------------------------------
  1. Preflight: verify pinned local asset layout for
     ``Qwen/Qwen2.5-Coder-3B-Instruct@488639f1ff808d1d3d0ba301aef8c11461451ec5``
     (base) and ``chandar-lab/CADmium-3B@6c6a79d160595207804171be56a037883b369abd``
     (LoRA adapter holding the verified tokenizer). The resolved paths must
     contain those revision strings (the production CLI always requires this;
     only direct ``RunnerConfig`` construction in tests may disable it). A
     SHA256 manifest (``--asset-manifest``, REQUIRED) must cover EVERY file
     the load actually reads -- base ``config.json``, the safetensors index
     and every weight shard (or the single ``model.safetensors``), adapter
     config/weights/tokenizer -- plus any known load-influencing small file
     found on disk (``generation_config.json``, extra tokenizer files).
     Missing, uncovered or mismatched files abort (non-zero CLI exit) BEFORE
     any forward; no file is existence-only.
  2. Load tokenizer from the adapter directory with ``local_files_only=True``
     and require ``cadmium_scorer.fingerprint_tokenizer(...).matches`` -- a
     fingerprint mismatch refuses to score.
  3. Load base weights + LoRA via transformers/peft with
     ``local_files_only=True``; ``eval()``, frozen grads, ``use_cache=False``.
  4. Score (query_text, candidate_json_desc) pairs by reusing the audited
     ``cadmium_scorer.build_cadmium_sequence`` for tokenization, boundary,
     terminator and 4096-context rejection (``len(full_ids) > 4096`` is
     recorded as ``out_of_training_context``; NEVER truncated). The forward
     runs on NPU; target-token log-probs are computed ON DEVICE in FP32
     (``log_softmax`` over the causal-shift prediction rows, then gather),
     chunked over rows to bound the FP32 workspace. Only the per-pair scalars
     (sum/mean/finite), lengths, target ids and (optionally) the per-target
     token log-probs needed for spot checks cross to CPU -- never the full
     batch's [B, L, V] logits.
  5. Consistency with the audited CPU math core: the device path implements
     exactly ``log_softmax_f32(logits[t-1])[y_t]`` summed in float64, the same
     math as ``scoring.gather_response_logprobs`` on the same real logits.
     ``spot_check_with_cpu_core`` uses the SAME single forward's logits for
     the device gather and audited CPU core, then compares within a fixed
     tolerance (pilot step 1). ``verify_batch_consistency``
     scores the same fixed pair list at batch=1 and batch=B and raises on any
     drift beyond tolerance (contract: stop, never silently switch kernels).

ISOLATION
---------
Inputs are UID-free ``PairRequest(query_text, candidate_json_desc)`` items.
The CLI machine-checks that every input JSONL line carries EXACTLY those two
string fields -- no UID, qrel, annotation or duplicate keys; any deviation
exits non-zero before the authorization gate or any model load (review
item 3). This module never opens qrels, audit files, CADmium ``annotation``
or the raw CSV; UID linkage and qrels-based evaluation belong to outer
scripts running in separate processes, and the pair_index <-> UID mapping is
owned by the outer manifest/logger. Right padding is used only for batching;
with causal attention pad tokens sit strictly to the right of real tokens and
cannot influence their logits (numerically re-verified by
``verify_batch_consistency`` before any scale-up).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cadmium_scorer import (
    MAX_TRAINING_CONTEXT,
    REASON_NONFINITE,
    REASON_OUT_OF_TRAINING_CONTEXT,
    REASON_TOKENIZER_MISMATCH,
    ScoredPair,
    ScoreLedger,
    TokenizerFingerprint,
    build_cadmium_sequence,
    fingerprint_tokenizer,
    load_system_message,
    score_batch_consistency,
    score_pair_from_logits,
)
from .p1_runner import (  # audited dual-version authorization fetch (R3)
    ACTIVE_TASK_BLOB,
    AUTHORIZATION_REF,
    EXPECTED_ORIGIN_URL,
    fetch_authorized_active_task,
)
from .scoring import ScoringError

# ---------------------------------------------------------------------------
# Pinned model identities (contract item 2; SHA-verified runtime copies on
# yunnan-share2 live under directories named by these revisions -- see
# experiments/assets/TEXT2CAD_BENCHMARK_ASSET_AUDIT_2026-09-28.md)
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_MODEL_ID = "chandar-lab/CADmium-3B"
EXPECTED_BASE_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
EXPECTED_ADAPTER_REVISION = "6c6a79d160595207804171be56a037883b369abd"

#: Pre-agreed numerical tolerance for batch=1 vs batch>1 and for the
#: device-vs-CPU-core spot check (matches cadmium_scorer.score_batch_consistency).
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6

#: Reason code when a target token id exceeds the loaded model's output vocab.
#: (The audited numpy core raises ScoringError in this case; the runner records
#: and skips instead of crashing the batch.)
REASON_TARGET_OUT_OF_VOCAB = "target_out_of_vocab"


class RunnerError(Exception):
    """Hard failure: preflight, authorization/budget, or consistency violation."""


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------

@dataclass
class RunnerConfig:
    """Everything the runner needs; paths are LOCAL pinned copies.

    ``asset_manifest`` (REQUIRED JSON) maps
    ``{"base": {relpath: sha256}, "adapter": {relpath: sha256}}`` and must
    SHA256-cover every file the load actually reads -- weight shards, the
    safetensors index, configs, tokenizer files -- plus any load-influencing
    small file present on disk. ``require_revision_in_path`` is a TEST-ONLY
    escape; the production CLI never disables it. The approved pair cap and
    wall-clock limit do NOT live here: they come from
    :class:`RunAuthorization`, parsed from the freshly fetched
    ``origin/master`` ACTIVE_TASK.md.
    """

    base_model_path: str
    adapter_path: str
    system_message_config: str
    device: str = "npu:0"
    torch_dtype: str = "bfloat16"
    expected_base_revision: str = EXPECTED_BASE_REVISION
    expected_adapter_revision: str = EXPECTED_ADAPTER_REVISION
    require_revision_in_path: bool = True
    asset_manifest: str | None = None
    logits_chunk_rows: int = 512
    budget_file: str | None = None
    keep_token_scores: bool = False


@dataclass
class PairRequest:
    """One UID-free scoring input. Only strip()-normalized query text and the
    verbatim CADmium ``json_desc`` candidate string; never a qrel/annotation."""

    query_text: str
    candidate_json_desc: str


@dataclass
class RunnerPairResult:
    """Runner-side wrapper around the audited ScoredPair.

    ``token_logprobs`` (FP32 per-target-token, positions prefix_len..full_len-1)
    is populated only when ``keep_token_scores`` is set (spot checks / pilot
    step 1 full-token comparison); otherwise None to keep transfers tiny.
    """

    pair_index: int
    pair: ScoredPair
    token_logprobs: list[float] | None = None


# ---------------------------------------------------------------------------
# Budget ledger: persisted cumulative pair cap + wall clock bound to ONE
# approval (review item 2); restart-idempotent
# ---------------------------------------------------------------------------

class BudgetLedger:
    """Append-only cumulative forward budget persisted as JSON.

    The file survives restarts, so a crashed run resumes charging against the
    same cumulative cap instead of resetting it. The ledger is BOUND to one
    approval: ``approved_sha``, ``max_pairs`` and ``max_wall_seconds`` are
    persisted on first write, and a reload under different values refuses
    (start a new ledger file for a new approval). ``started_at_epoch`` is
    persisted too, so the approved wall clock keeps counting across restarts
    rather than resetting. ``charge`` refuses (raises RunnerError) when the
    requested forwards would exceed the approved cap; nothing is forwarded in
    that case and the failed charge is not persisted.
    """

    def __init__(
        self,
        path: str | os.PathLike | None,
        max_pairs: int,
        *,
        approved_sha: str | None = None,
        max_wall_seconds: float | None = None,
    ) -> None:
        if max_pairs < 0:
            raise RunnerError(f"max_pairs must be >= 0, got {max_pairs}")
        if max_wall_seconds is not None and max_wall_seconds <= 0:
            raise RunnerError(f"max_wall_seconds must be > 0 when given, got {max_wall_seconds}")
        self.path = Path(path) if path else None
        self.max_pairs = int(max_pairs)
        self.approved_sha = approved_sha
        self.max_wall_seconds = max_wall_seconds
        self.pairs_forwarded = 0
        self.started_at_epoch = time.time()
        if self.path and self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, TypeError, OSError) as e:
                raise RunnerError(f"unreadable budget ledger {self.path}: {e}") from e
            if state.get("approved_sha") != self.approved_sha:
                raise RunnerError(
                    f"budget ledger {self.path} is bound to approved_sha "
                    f"{state.get('approved_sha')!r}, not {self.approved_sha!r}; "
                    "use a fresh ledger file for a new approval"
                )
            if int(state.get("max_pairs", -1)) != self.max_pairs:
                raise RunnerError(
                    f"budget ledger {self.path} cap {state.get('max_pairs')} != "
                    f"current approved cap {self.max_pairs}; use a fresh ledger file"
                )
            if state.get("max_wall_seconds") != self.max_wall_seconds:
                raise RunnerError(
                    f"budget ledger {self.path} wall-clock limit "
                    f"{state.get('max_wall_seconds')} != current "
                    f"{self.max_wall_seconds}; use a fresh ledger file"
                )
            try:
                self.pairs_forwarded = int(state.get("pairs_forwarded", 0))
                self.started_at_epoch = float(state.get("started_at_epoch", self.started_at_epoch))
            except (ValueError, TypeError) as e:
                raise RunnerError(f"corrupt budget ledger {self.path}: {e}") from e

    def remaining(self) -> int:
        return self.max_pairs - self.pairs_forwarded

    def charge(self, n: int) -> None:
        if n < 0:
            raise RunnerError(f"cannot charge negative pair count {n}")
        if n == 0:
            return
        if self.pairs_forwarded + n > self.max_pairs:
            raise RunnerError(
                f"pair budget exceeded: forwarded {self.pairs_forwarded} + "
                f"requested {n} > approved cap {self.max_pairs}; stop and wait "
                f"for an updated ACTIVE_TASK.md authorization"
            )
        self.pairs_forwarded += n
        self._persist()

    def check_wall_clock(self, now: float | None = None) -> float:
        """Elapsed seconds against the approved wall clock; refuse past the cap.

        Measured from the ledger's persisted ``started_at_epoch``, so restarts
        do not reset the clock. No-op (returns 0.0) when no wall-clock limit
        is bound (test-only in-memory ledgers).
        """
        if self.max_wall_seconds is None:
            return 0.0
        elapsed = (time.time() if now is None else now) - self.started_at_epoch
        if elapsed > self.max_wall_seconds:
            raise RunnerError(
                f"approved wall-clock budget exceeded: {elapsed:.1f}s elapsed > "
                f"{self.max_wall_seconds:.1f}s approved; stop and wait for an "
                f"updated ACTIVE_TASK.md authorization"
            )
        return elapsed

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "pairs_forwarded": self.pairs_forwarded,
                    "max_pairs": self.max_pairs,
                    "approved_sha": self.approved_sha,
                    "max_wall_seconds": self.max_wall_seconds,
                    "started_at_epoch": self.started_at_epoch,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_file": str(self.path) if self.path else None,
            "pairs_forwarded": self.pairs_forwarded,
            "max_pairs": self.max_pairs,
            "remaining": self.remaining(),
            "approved_sha": self.approved_sha,
            "max_wall_seconds": self.max_wall_seconds,
            "started_at_epoch": self.started_at_epoch,
        }


# ---------------------------------------------------------------------------
# Preflight asset verification (before ANY model forward; contract item 2)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Adapter weight files; exactly one must exist so the manifest provably
#: covers the file peft actually loads (safetensors is preferred by peft).
ADAPTER_WEIGHT_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")

#: Small files that silently influence the model/tokenizer load IF PRESENT.
#: The contract forbids existence-only checks, so any of these found on disk
#: must be SHA256-covered by the manifest too.
BASE_CONDITIONAL_FILES = ("generation_config.json",)
ADAPTER_CONDITIONAL_FILES = (
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_files(root: Path, relpaths: Sequence[str], label: str) -> None:
    for rel in relpaths:
        if not (root / rel).is_file():
            raise RunnerError(f"preflight: missing {label} file {root / rel}")


def _load_asset_manifest(manifest_path: str) -> dict[str, dict[str, str]]:
    """Parse and shape-check the required SHA256 manifest (fail closed)."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise RunnerError(f"preflight: unreadable asset manifest {manifest_path}: {e}") from e
    if not isinstance(manifest, dict):
        raise RunnerError("preflight: asset manifest must be a JSON object")
    out: dict[str, dict[str, str]] = {}
    for section in ("base", "adapter"):
        entries = manifest.get(section)
        if not isinstance(entries, dict) or not entries:
            raise RunnerError(
                f"preflight: asset manifest section {section!r} must be a "
                "non-empty object mapping relpath -> sha256"
            )
        checked: dict[str, str] = {}
        for rel, sha in entries.items():
            if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
                raise RunnerError(f"preflight: bad manifest relpath {rel!r}")
            if not isinstance(sha, str) or not _SHA256_RE.match(sha):
                raise RunnerError(f"preflight: bad sha256 for manifest entry {section}:{rel}")
            checked[rel] = sha.lower()
        out[section] = checked
    return out


def preflight_assets(cfg: RunnerConfig) -> dict[str, Any]:
    """Verify the pinned local asset layout; raise RunnerError on any gap.

    Fail-closed rules (contract item 2; review item 1):
      * directories exist and (unless ``require_revision_in_path`` is False --
        a test-only entry the production CLI never sets) embed the pinned
        revisions;
      * ``cfg.asset_manifest`` is REQUIRED and must SHA256-cover EVERY file
        the load actually reads: base ``config.json``, the safetensors index
        and every shard (or the single ``model.safetensors``), adapter
        ``adapter_config.json`` + exactly one adapter weight file +
        ``tokenizer_config.json`` / ``tokenizer.json``;
      * any ``BASE_CONDITIONAL_FILES`` / ``ADAPTER_CONDITIONAL_FILES`` member
        present on disk must also be manifest-covered -- no small file is
        existence-only;
      * every manifest entry must exist on disk and hash-match.

    Returns a JSON-serializable record for environment.json.
    """
    base = Path(cfg.base_model_path).resolve()
    adapter = Path(cfg.adapter_path).resolve()
    if not base.is_dir():
        raise RunnerError(f"preflight: base model dir not found: {base}")
    if not adapter.is_dir():
        raise RunnerError(f"preflight: adapter dir not found: {adapter}")
    if cfg.require_revision_in_path:
        if cfg.expected_base_revision not in str(base):
            raise RunnerError(
                f"preflight: base path {base} does not contain pinned revision "
                f"{cfg.expected_base_revision} ({BASE_MODEL_ID})"
            )
        if cfg.expected_adapter_revision not in str(adapter):
            raise RunnerError(
                f"preflight: adapter path {adapter} does not contain pinned "
                f"revision {cfg.expected_adapter_revision} ({ADAPTER_MODEL_ID})"
            )

    if not cfg.asset_manifest:
        raise RunnerError(
            "preflight: asset_manifest is REQUIRED -- a complete SHA256 manifest "
            "of every file the load reads is not optional (review item 1)"
        )
    manifest = _load_asset_manifest(cfg.asset_manifest)

    record: dict[str, Any] = {
        "base_model_id": BASE_MODEL_ID,
        "adapter_model_id": ADAPTER_MODEL_ID,
        "expected_base_revision": cfg.expected_base_revision,
        "expected_adapter_revision": cfg.expected_adapter_revision,
        "base_model_path": str(base),
        "adapter_path": str(adapter),
        "asset_manifest_path": str(Path(cfg.asset_manifest).resolve()),
        "asset_manifest_sha256": _sha256_file(Path(cfg.asset_manifest)),
        "manifest_checked_files": 0,
    }

    # Required base files: config + (index + shards | single safetensors).
    base_required = ["config.json"]
    try:
        base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise RunnerError(f"preflight: unreadable base config.json: {e}") from e
    record["base_config_vocab_size"] = base_config.get("vocab_size")
    record["base_config_model_type"] = base_config.get("model_type")
    index = base / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (ValueError, KeyError, OSError) as e:
            raise RunnerError(f"preflight: unreadable safetensors index: {e}") from e
        shards = sorted(set(weight_map.values()))
        base_required.append("model.safetensors.index.json")
        base_required.extend(shards)
        record["base_weight_shards"] = shards
    else:
        base_required.append("model.safetensors")
        record["base_weight_shards"] = ["model.safetensors"]
    _require_files(base, base_required, "base")

    # Required adapter files: config + tokenizer + exactly one weight file.
    adapter_required = ["adapter_config.json", "tokenizer_config.json", "tokenizer.json"]
    present_weights = [w for w in ADAPTER_WEIGHT_CANDIDATES if (adapter / w).is_file()]
    if not present_weights:
        raise RunnerError(f"preflight: no adapter weights in {adapter} (tried {ADAPTER_WEIGHT_CANDIDATES})")
    if len(present_weights) > 1:
        raise RunnerError(
            f"preflight: ambiguous adapter weights in {adapter}: {present_weights}; "
            "keep exactly one so the manifest provably covers the file peft loads"
        )
    adapter_required.append(present_weights[0])
    record["adapter_weight_file"] = present_weights[0]
    _require_files(adapter, adapter_required, "adapter")
    try:
        adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise RunnerError(f"preflight: unreadable adapter_config.json: {e}") from e
    record["adapter_peft_type"] = adapter_config.get("peft_type")
    record["adapter_r"] = adapter_config.get("r")
    record["adapter_base_model_name_or_path"] = adapter_config.get("base_model_name_or_path")

    # Coverage + integrity: hash every manifest entry; nothing used is skipped.
    n_checked = 0
    plan = (
        ("base", base, base_required, BASE_CONDITIONAL_FILES),
        ("adapter", adapter, adapter_required, ADAPTER_CONDITIONAL_FILES),
    )
    for section, root, required, conditional in plan:
        entries = manifest[section]
        uncovered = [rel for rel in required if rel not in entries]
        if uncovered:
            raise RunnerError(
                f"preflight: manifest section {section!r} does not cover required "
                f"files {uncovered}; the load would read unverified bytes"
            )
        silent = [rel for rel in conditional if (root / rel).is_file() and rel not in entries]
        if silent:
            raise RunnerError(
                f"preflight: {section} files {silent} exist on disk and can "
                "silently influence the load but are not SHA256-covered by the "
                "manifest; add them to the manifest or remove them"
            )
        for rel, expected_sha in sorted(entries.items()):
            path = root / rel
            if not path.is_file():
                raise RunnerError(f"preflight: manifest file missing: {path}")
            actual = _sha256_file(path)
            if actual != expected_sha:
                raise RunnerError(
                    f"preflight: SHA256 mismatch for {path}: got {actual}, "
                    f"expected {expected_sha}"
                )
            n_checked += 1
    record["manifest_checked_files"] = n_checked
    record["manifest_required_files"] = {
        "base": sorted(set(base_required)),
        "adapter": sorted(set(adapter_required)),
    }
    return record


# ---------------------------------------------------------------------------
# Run authorization gate (review item 2 + 2026-09-30 addendum): the FRESHLY
# FETCHED origin/master ACTIVE_TASK.md + live detached/clean git state,
# fail closed BEFORE any weight load
# ---------------------------------------------------------------------------

#: Machine-checked ACTIVE_TASK.md field patterns. Codex fills these fields in
#: the fixed ``- `key`: value`` bullet format; any parse ambiguity refuses.
_STATE_RE = re.compile(r"`state`:\s*`([^`]+)`")
_TASK_ID_RE = re.compile(r"`task_id`:\s*`([^`]+)`")
_APPROVED_SHA_RE = re.compile(r"`approved_sha`:\s*`?([0-9a-fA-F]{40})`?")
_MAX_PAIRS_RE = re.compile(r"`max_scored_pairs`:\s*(\d+)")
_GPU_BUDGET_RE = re.compile(r"`gpu_budget`:\s*(\d+(?:\.\d+)?)")
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class RunAuthorization:
    """A validated run approval (review item 2; 2026-09-30 addendum).

    Built by ``authorize_run`` in production from the FETCHED
    ``origin/master`` ACTIVE_TASK.md; tests may construct it directly to
    inject a fake approval for scorer-level tests (the CLI exposes no such
    switch, and no test hook can supply the authorization text itself).
    Binds the approved pair cap, card-hours and derived wall-clock limit to
    one run and its ledger: ``max_wall_seconds = card_hours * 3600 /
    cards_used``. Audit fields record WHERE the authorization came from:
    ``authorization_ref_sha`` is the origin/master approval commit, which by
    construction differs from ``approved_sha`` (the approved code commit a
    commit cannot name inside itself).
    """

    task_state: str
    task_id: str | None
    approved_sha: str
    actual_sha: str
    max_pairs: int
    card_hours: float
    cards_used: int
    max_wall_seconds: float
    active_task_source: str
    authorization_ref_sha: str
    active_task_blob_sha256: str
    origin_url: str
    repo_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_state": self.task_state,
            "task_id": self.task_id,
            "approved_sha": self.approved_sha,
            "actual_sha": self.actual_sha,
            "max_pairs": self.max_pairs,
            "card_hours": self.card_hours,
            "cards_used": self.cards_used,
            "max_wall_seconds": self.max_wall_seconds,
            "active_task_source": self.active_task_source,
            "authorization_ref_sha": self.authorization_ref_sha,
            "active_task_blob_sha256": self.active_task_blob_sha256,
            "origin_url": self.origin_url,
            "repo_root": self.repo_root,
        }


def _git_proc(repo_root: Path, *git_args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *git_args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RunnerError(
            f"authorization: cannot run git {' '.join(git_args)} in {repo_root}: {e}"
        ) from e


def _git_output(repo_root: Path, *git_args: str) -> str:
    proc = _git_proc(repo_root, *git_args)
    if proc.returncode != 0:
        raise RunnerError(
            f"authorization: git {' '.join(git_args)} failed in {repo_root}: "
            f"{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _git_head_detached(repo_root: Path) -> bool:
    """True iff HEAD is detached (production protocol: ``git checkout
    --detach <approved sha>``). ``git symbolic-ref -q HEAD`` exits 0 when
    HEAD names a branch (attached -> refuse) and 1 when detached."""
    proc = _git_proc(repo_root, "symbolic-ref", "-q", "HEAD")
    if proc.returncode == 0:
        return False
    if proc.returncode == 1:
        return True
    raise RunnerError(
        f"authorization: cannot determine HEAD attachment in {repo_root}: "
        f"{proc.stderr.strip()}"
    )


def parse_active_task_authorization(text: str) -> dict[str, Any]:
    """Validate the authorization FIELDS of an ACTIVE_TASK.md text.

    Fail-closed: ``state`` must be exactly ``pilot_authorized``; a full
    40-hex ``approved_sha`` must be present (a placeholder like ``无`` or a
    short SHA refuses); ``max_scored_pairs`` and ``gpu_budget`` (card-hours)
    must parse and both be > 0 -- the task state as of 2026-09-30 has 0/0,
    so it refuses. Returns the parsed fields.

    This is a pure text check. The authorization-SOURCE problem (Codex
    review addendum 2026-09-30) is solved structurally: ``authorize_run``
    obtains ``text`` ONLY from the freshly fetched ``origin/master`` blob,
    never from a working-tree file, so parsing can never be fooled by a
    self-authored checkout.
    """
    m = _STATE_RE.search(text)
    state = m.group(1).strip() if m else None
    if state != "pilot_authorized":
        raise RunnerError(
            f"authorization: ACTIVE_TASK.md state is {state!r}, not "
            "'pilot_authorized'; no model forward is permitted"
        )
    m = _APPROVED_SHA_RE.search(text)
    if not m:
        raise RunnerError(
            "authorization: ACTIVE_TASK.md has no full 40-hex approved_sha; "
            "Codex must fill it after code review"
        )
    approved_sha = m.group(1).lower()
    m = _MAX_PAIRS_RE.search(text)
    max_pairs = int(m.group(1)) if m else 0
    if max_pairs <= 0:
        raise RunnerError(
            f"authorization: max_scored_pairs is {max_pairs}; refusing any forward"
        )
    m = _GPU_BUDGET_RE.search(text)
    card_hours = float(m.group(1)) if m else 0.0
    if card_hours <= 0:
        raise RunnerError(
            f"authorization: gpu_budget is {card_hours} card-hours; refusing any forward"
        )
    task_id_match = _TASK_ID_RE.search(text)
    return {
        "task_state": state,
        "task_id": task_id_match.group(1).strip() if task_id_match else None,
        "approved_sha": approved_sha,
        "max_pairs": max_pairs,
        "card_hours": card_hours,
    }


def authorize_run(
    repo_root: str | os.PathLike,
    *,
    cards_used: int = 1,
    expected_origin: str = EXPECTED_ORIGIN_URL,
    authorization_ref: str = AUTHORIZATION_REF,
) -> RunAuthorization:
    """Fail-closed run gate; every check passes or nothing loads.

    The authorization text is read from
    ``<authorization_ref>:experiments/ACTIVE_TASK.md`` on the FRESHLY fetched
    origin via the audited ``p1_runner.fetch_authorized_active_task`` --
    never from the detached HEAD's working tree: a commit cannot contain its
    own ``approved_sha``, so the working-tree file could be stale or
    self-authored (Codex review addendum 2026-09-30). The approval commit
    (the ``origin/master`` tip) therefore differs from the approved code
    commit by construction; both SHAs are recorded for audit.

    Checks, in order (all refusals raise RunnerError):
      1. the ``origin`` remote URL matches the expected project remote and
         ``git fetch origin`` succeeds, so ``authorization_ref`` is fresh;
      2. the fetched ACTIVE_TASK.md passes
         :func:`parse_active_task_authorization` (``pilot_authorized``, full
         40-hex ``approved_sha``, ``max_scored_pairs`` > 0, ``gpu_budget``
         > 0);
      3. HEAD is DETACHED at a full 40-hex SHA equal to ``approved_sha``
         (the production protocol is ``git checkout --detach <sha>``; a HEAD
         attached to a movable branch refuses even at the right commit) and
         ``git status --porcelain`` is empty.

    ``cards_used`` is how many NPU cards this run occupies (this runner is
    single-device); approved card-hours convert to a wall-clock limit
    ``card_hours*3600/cards_used``. ``expected_origin``/``authorization_ref``
    exist so CPU tests can point at a real temporary Git remote; the
    production CLI never overrides them, and no test-only switch can supply
    the authorization text or fake the live Git state.
    """
    if int(cards_used) < 1:
        raise RunnerError(f"authorization: cards_used must be >= 1, got {cards_used}")
    cards_used = int(cards_used)
    root = Path(repo_root).resolve()

    text, meta = fetch_authorized_active_task(root, expected_origin=expected_origin, ref=authorization_ref)
    if text is None:
        problems = "; ".join(meta.get("problems", []))
        raise RunnerError(
            f"authorization: cannot read a fresh "
            f"{authorization_ref}:{ACTIVE_TASK_BLOB} from the validated "
            f"origin; refusing any forward ({problems})"
        )
    fields = parse_active_task_authorization(text)

    if not _git_head_detached(root):
        raise RunnerError(
            f"authorization: HEAD in {root} is attached to a branch; the "
            "approved protocol is `git checkout --detach <approved_sha>` -- "
            "refusing to run on a movable ref"
        )
    actual_sha = _git_output(root, "rev-parse", "HEAD").strip().lower()
    if not _SHA40_RE.match(actual_sha):
        raise RunnerError(f"authorization: git HEAD {actual_sha!r} is not a full 40-hex SHA")
    if actual_sha != fields["approved_sha"]:
        raise RunnerError(
            f"authorization: checked-out HEAD {actual_sha} != approved_sha "
            f"{fields['approved_sha']}; detach the reviewed commit before running"
        )
    dirty = _git_output(root, "status", "--porcelain")
    if dirty:
        raise RunnerError(
            f"authorization: worktree at {root} is dirty; refusing to run:\n{dirty}"
        )

    return RunAuthorization(
        task_state=fields["task_state"],
        task_id=fields["task_id"],
        approved_sha=fields["approved_sha"],
        actual_sha=actual_sha,
        max_pairs=fields["max_pairs"],
        card_hours=fields["card_hours"],
        cards_used=cards_used,
        max_wall_seconds=fields["card_hours"] * 3600.0 / cards_used,
        active_task_source=str(meta.get("active_task_source", f"{authorization_ref}:{ACTIVE_TASK_BLOB}")),
        authorization_ref_sha=str(meta.get("origin_ref_sha", "")),
        active_task_blob_sha256=str(meta.get("active_task_blob_sha256", "")),
        origin_url=str(meta.get("origin_url", "")),
        repo_root=str(root),
    )


# ---------------------------------------------------------------------------
# Lazy loaders (torch / transformers / peft import only when really loading)
# ---------------------------------------------------------------------------

def _import_stack(device: str) -> Any:
    """Import torch (+ torch_npu on Ascend); RunnerError with pinned versions."""
    try:
        import torch
    except ImportError as e:
        raise RunnerError(
            "torch is required to load the model; use the pinned clone env "
            "(torch 2.7.1+cpu, torch_npu 2.7.1.post1, transformers 4.53.1, "
            "peft 0.14.0 -- experiments/configs/idea01_yunnan_share2_env.json)"
        ) from e
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401  (registers the NPU backend)
        except ImportError as e:
            raise RunnerError(
                f"device {device!r} requested but torch_npu is not importable"
            ) from e
    return torch


def load_tokenizer(cfg: RunnerConfig) -> tuple[Any, TokenizerFingerprint]:
    """Load the adapter tokenizer locally and enforce the M1a fingerprint."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RunnerError("transformers (4.53.1 pinned) is required") from e
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.adapter_path, local_files_only=True)
    except Exception as e:  # missing files, bad json, etc.
        raise RunnerError(f"tokenizer load failed from {cfg.adapter_path}: {e}") from e
    fingerprint = fingerprint_tokenizer(tokenizer)
    if not fingerprint.matches:
        raise RunnerError(
            f"{REASON_TOKENIZER_MISMATCH}: adapter tokenizer fingerprint failed "
            f"(vocab_size={fingerprint.vocab_size}, len={fingerprint.len_tokenizer}, "
            f"eos={fingerprint.eos_token_id}, im_end={fingerprint.im_end_token_id}, "
            f"pad={fingerprint.pad_token_id}, chat_template_sha256="
            f"{fingerprint.chat_template_sha256}); refusing to score"
        )
    # Runner pads manually (right); set the side anyway so any incidental
    # tokenizer-driven padding cannot silently become left-padding.
    tokenizer.padding_side = "right"
    return tokenizer, fingerprint


def load_model(cfg: RunnerConfig) -> Any:
    """Load pinned base + CADmium LoRA locally, frozen, eval, no cache."""
    torch = _import_stack(cfg.device)
    try:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        raise RunnerError("transformers + peft (0.14.0 pinned) are required") from e

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if cfg.torch_dtype not in dtypes:
        raise RunnerError(f"unsupported torch_dtype {cfg.torch_dtype!r}; choose from {sorted(dtypes)}")
    dtype = dtypes[cfg.torch_dtype]

    try:
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
    except Exception as e:
        raise RunnerError(f"base model load failed from {cfg.base_model_path}: {e}") from e
    try:
        model = PeftModel.from_pretrained(
            base,
            cfg.adapter_path,
            local_files_only=True,
            is_trainable=False,
        )
    except Exception as e:
        raise RunnerError(f"LoRA adapter load failed from {cfg.adapter_path}: {e}") from e

    device = torch.device(cfg.device)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    return model


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class CadmiumNpuScorer:
    """Batch-capable CADmium conditional scorer on NPU.

    Construction requires a validated ``RunAuthorization`` (review item 2)
    and runs preflight + tokenizer fingerprint + model load; any failure
    raises RunnerError before a single forward. The persisted
    ``BudgetLedger`` is bound to the authorization's SHA / pair cap /
    wall-clock limit, and the wall clock is re-checked before every batch.
    ``score`` preserves the input order in its output list and never forwards
    pairs that the audited sequence builder already rejected (empty target,
    prefix mismatch, bad terminator, >4096 context -- recorded, not
    truncated, not forwarded).
    """

    def __init__(self, cfg: RunnerConfig, authorization: RunAuthorization) -> None:
        self.cfg = cfg
        self.authorization = authorization
        self.budget = BudgetLedger(
            cfg.budget_file,
            authorization.max_pairs,
            approved_sha=authorization.approved_sha,
            max_wall_seconds=authorization.max_wall_seconds,
        )
        self.budget.check_wall_clock()
        self.preflight_record = preflight_assets(cfg)
        self.system_message = load_system_message(cfg.system_message_config)
        self.tokenizer, self.fingerprint = load_tokenizer(cfg)
        self.model = load_model(cfg)
        self.budget.check_wall_clock()
        self._torch = _import_stack(cfg.device)
        self._device = self._torch.device(cfg.device)
        if self._device.type == "npu":
            npu = getattr(self._torch, "npu", None)
            visible = int(npu.device_count()) if npu is not None and hasattr(npu, "device_count") else 0
            index = self._device.index if self._device.index is not None else 0
            if index >= visible:
                raise RunnerError(
                    f"authorization: device {self._device} requested but only "
                    f"{visible} NPU device(s) visible; refusing to run"
                )
        out_emb = self.model.get_output_embeddings()
        self._model_vocab = int(out_emb.weight.shape[0]) if out_emb is not None else -1
        self.ledger = ScoreLedger()

    # -- on-device FP32 target-token log-probs (chunked over rows) ----------

    def _gather_target_logprobs(self, pred_rows: Any, targets: Any) -> tuple[np.ndarray, bool]:
        """FP32 log_softmax + gather for [R, V] prediction rows, on device.

        Returns (float32 numpy [R], all_finite). Chunking over rows bounds the
        FP32 workspace to ``logits_chunk_rows * V * 4`` bytes; each chunk is the
        same row-wise math as scoring.log_softmax_f32, so results are identical
        to an unchunked computation on the same logits.
        """
        torch = self._torch
        chunk = max(1, int(self.cfg.logits_chunk_rows))
        R = int(pred_rows.shape[0])
        pieces: list[Any] = []
        finite_flags: list[Any] = []
        for start in range(0, R, chunk):
            lsm = torch.log_softmax(pred_rows[start : start + chunk].float(), dim=-1)
            finite_flags.append(torch.isfinite(lsm).all())
            pieces.append(lsm.gather(1, targets[start : start + chunk].unsqueeze(1)).squeeze(1))
        tok_lp = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float32, device=pred_rows.device)
        all_finite = bool(torch.stack(finite_flags).all().item()) if finite_flags else True
        tok_lp_np = tok_lp.cpu().numpy().astype(np.float32, copy=False)
        all_finite = all_finite and bool(np.isfinite(tok_lp_np).all())
        return tok_lp_np, all_finite

    def _forward_batch(self, batch_full_ids: Sequence[Sequence[int]]) -> Any:
        """Right-pad and forward one batch; returns raw [B, Lmax, V] logits."""
        torch = self._torch
        pad_id = int(self.tokenizer.pad_token_id)
        lengths = [len(s) for s in batch_full_ids]
        max_len = max(lengths)
        ids = torch.full((len(lengths), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(lengths), max_len), dtype=torch.long)
        for i, seq in enumerate(batch_full_ids):
            ids[i, : lengths[i]] = torch.tensor(seq, dtype=torch.long)
            mask[i, : lengths[i]] = 1
        ids = ids.to(self._device)
        mask = mask.to(self._device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return out.logits, lengths

    def _empty_cache(self) -> None:
        if self._device.type == "npu" and hasattr(self._torch, "npu"):
            self._torch.npu.empty_cache()

    # -- public scoring entry ------------------------------------------------

    def score(self, requests: Sequence[PairRequest], batch_size: int = 1) -> list[RunnerPairResult]:
        """Score pairs in fixed input order; budget-charged BEFORE forward.

        Rejections from the audited builder consume no budget and no forward.
        Scorable pairs are charged as a whole before the first batch runs, so a
        budget violation stops the run before any NPU work. The approved
        wall-clock limit is re-checked before every batch (review item 2).
        """
        if batch_size < 1:
            raise RunnerError(f"batch_size must be >= 1, got {batch_size}")
        results: list[RunnerPairResult | None] = [None] * len(requests)
        pending: list[tuple[int, Any]] = []  # (pair_index, BuiltSequence)

        # Phase A (CPU only): audited tokenization / boundary / context checks.
        for idx, req in enumerate(requests):
            base = ScoredPair(
                query_text=req.query_text,
                candidate_json_desc=req.candidate_json_desc,
                status="rejected",
            )
            built, reason = build_cadmium_sequence(
                req.query_text, req.candidate_json_desc, self.system_message, self.tokenizer
            )
            if built is None:
                base.reason = reason
                results[idx] = RunnerPairResult(pair_index=idx, pair=base)
                continue
            base.prefix_len = built.prefix_len
            base.full_len = built.full_len
            base.target_len = built.target_len
            base.target_ids = built.target_ids
            if built.full_len > MAX_TRAINING_CONTEXT:
                base.reason = REASON_OUT_OF_TRAINING_CONTEXT
                base.within_training_context = False
                results[idx] = RunnerPairResult(pair_index=idx, pair=base)
                continue
            if self._model_vocab > 0 and max(built.target_ids) >= self._model_vocab:
                base.reason = REASON_TARGET_OUT_OF_VOCAB
                results[idx] = RunnerPairResult(pair_index=idx, pair=base)
                continue
            pending.append((idx, built))
            results[idx] = RunnerPairResult(pair_index=idx, pair=base)  # filled below

        # Budget gate: refuse BEFORE any forward (contract stop condition).
        self.budget.charge(len(pending))

        # Phase B (NPU): forward in fixed order; per-sample causal-shift gather.
        torch = self._torch
        for start in range(0, len(pending), batch_size):
            self.budget.check_wall_clock()
            chunk = pending[start : start + batch_size]
            logits, _lengths = self._forward_batch([b.full_ids for _, b in chunk])
            for row, (idx, built) in enumerate(chunk):
                res = results[idx]
                assert res is not None
                L = built.full_len
                t0 = built.prefix_len
                pred_rows = logits[row, t0 - 1 : L - 1, :]
                targets = torch.tensor(built.target_ids, dtype=torch.long, device=logits.device)
                tok_lp_np, all_finite = self._gather_target_logprobs(pred_rows, targets)
                if not all_finite:
                    res.pair.reason = REASON_NONFINITE
                    continue
                total = float(tok_lp_np.sum(dtype=np.float64))
                res.pair.status = "scored"
                res.pair.reason = None
                res.pair.sum_logprob = total
                res.pair.mean_logprob = total / built.target_len
                res.pair.all_finite = True
                if self.cfg.keep_token_scores:
                    res.token_logprobs = [float(x) for x in tok_lp_np]
            del logits
            self._empty_cache()

        final: list[RunnerPairResult] = []
        for res in results:
            assert res is not None
            self.ledger.add(res.pair)
            final.append(res)
        return final

    # -- contract consistency gates ------------------------------------------

    def verify_batch_consistency(
        self,
        requests: Sequence[PairRequest],
        batch_size: int,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
    ) -> bool:
        """Score the SAME fixed pair list at batch=1 and batch=B; raise on drift.

        Consumes budget twice (two real forwards per pair) by design -- the
        comparison is only meaningful on real logits. Any mismatch beyond the
        pre-agreed tolerance raises RunnerError (contract: stop, do not tune
        around it).
        """
        single = [r.pair for r in self.score(requests, batch_size=1)]
        batched = [r.pair for r in self.score(requests, batch_size=batch_size)]
        if not score_batch_consistency(single, batched):
            raise RunnerError(
                f"batch consistency violation: batch=1 vs batch={batch_size} "
                f"differ beyond rtol={rtol}, atol={atol}; STOP per contract item 3"
            )
        # score_batch_consistency uses the scorer's fixed tolerance; enforce the
        # caller's (possibly stricter) tolerance explicitly as well.
        for a, b in zip(single, batched):
            if a.status == "scored" and b.status == "scored":
                if not np.isclose(a.sum_logprob, b.sum_logprob, rtol=rtol, atol=atol):
                    raise RunnerError(
                        f"batch consistency violation at tolerance rtol={rtol} "
                        f"atol={atol}: {a.sum_logprob} vs {b.sum_logprob}"
                    )
        return True

    def spot_check_with_cpu_core(
        self,
        request: PairRequest,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
    ) -> dict[str, Any]:
        """Compare device and audited CPU math on ONE real forward's logits.

        Only this bounded pilot check transfers full [L, V] logits to CPU.
        It charges exactly one pair before forwarding and refuses on drift.
        """
        built, reason = build_cadmium_sequence(
            request.query_text, request.candidate_json_desc, self.system_message, self.tokenizer
        )
        if built is None or built.full_len > MAX_TRAINING_CONTEXT:
            return {
                "status": "rejected",
                "reason": reason if built is None else REASON_OUT_OF_TRAINING_CONTEXT,
                "compared": False,
            }
        if self._model_vocab > 0 and max(built.target_ids) >= self._model_vocab:
            return {"status": "rejected", "reason": REASON_TARGET_OUT_OF_VOCAB, "compared": False}

        self.budget.charge(1)
        self.budget.check_wall_clock()
        logits, _ = self._forward_batch([built.full_ids])
        torch = self._torch
        pred_rows = logits[0, built.prefix_len - 1 : built.full_len - 1, :]
        targets = torch.tensor(built.target_ids, dtype=torch.long, device=logits.device)
        device_tokens, all_finite = self._gather_target_logprobs(pred_rows, targets)
        if not all_finite:
            del logits
            self._empty_cache()
            raise RunnerError("spot check device target log-probs are nonfinite")
        device_sum = float(device_tokens.sum(dtype=np.float64))
        full_logits_cpu = logits[0, : built.full_len, :].float().cpu().numpy()
        del logits
        self._empty_cache()
        cpu_pair = score_pair_from_logits(
            request.query_text,
            request.candidate_json_desc,
            self.system_message,
            self.tokenizer,
            full_logits_cpu,
            input_ids=np.asarray(built.full_ids, dtype=np.int64),
        )
        record: dict[str, Any] = {
            "status": cpu_pair.status,
            "reason": cpu_pair.reason,
            "device_sum_logprob": device_sum,
            "cpu_core_sum_logprob": cpu_pair.sum_logprob,
            "target_len": built.target_len,
            "rtol": rtol,
            "atol": atol,
            "compared": cpu_pair.status == "scored",
            "forwards_charged": 1,
        }
        if cpu_pair.status != "scored":
            raise RunnerError(f"CPU core rejected a device-scored pair: {cpu_pair.reason}")
        if not np.isclose(device_sum, cpu_pair.sum_logprob, rtol=rtol, atol=atol):
            raise RunnerError(
                f"device vs audited CPU core mismatch: {device_sum} "
                f"vs {cpu_pair.sum_logprob} (rtol={rtol}, atol={atol}); STOP"
            )
        self.ledger.add(cpu_pair)
        record["device_token_logprobs"] = [float(x) for x in device_tokens]
        return record

    # -- reporting -------------------------------------------------------------

    def summary_record(self) -> dict[str, Any]:
        """JSON-serializable run summary for environment.json / RESULTS.md."""
        fp = self.fingerprint
        return {
            "preflight": self.preflight_record,
            "tokenizer_fingerprint": {
                "vocab_size": fp.vocab_size,
                "len_tokenizer": fp.len_tokenizer,
                "eos_token_id": fp.eos_token_id,
                "im_end_token_id": fp.im_end_token_id,
                "im_start_token_id": fp.im_start_token_id,
                "pad_token_id": fp.pad_token_id,
                "chat_template_sha256": fp.chat_template_sha256,
                "matches": fp.matches,
            },
            "model_vocab_size": self._model_vocab,
            "device": self.cfg.device,
            "torch_dtype": self.cfg.torch_dtype,
            "logits_chunk_rows": self.cfg.logits_chunk_rows,
            "max_training_context": MAX_TRAINING_CONTEXT,
            "authorization": self.authorization.to_dict(),
            "budget": self.budget.to_dict(),
            "ledger": self.ledger.to_dict(),
        }


# ---------------------------------------------------------------------------
# CLI (server-side entry; fail-closed gates run BEFORE any model load)
# ---------------------------------------------------------------------------

#: The ONLY keys a scoring input line may carry (review item 3). Anything
#: else -- uid, qrel, annotation, cad_id, ... -- refuses the whole run.
PAIR_INPUT_KEYS = ("query_text", "candidate_json_desc")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise RunnerError(f"duplicate key {key!r} in pairs JSONL line")
        obj[key] = value
    return obj


def load_pair_requests(pairs_path: str | os.PathLike) -> list[PairRequest]:
    """Read the UID-free pairs JSONL with strict key isolation (review item 3).

    Every non-blank line must be a JSON object with EXACTLY the keys
    ``query_text`` and ``candidate_json_desc``, both strings, no duplicates.
    Any deviation raises RunnerError (the CLI exits non-zero) BEFORE the
    authorization gate or any model work, so a qrels/annotation/CSV-derived
    file can never reach the scorer. The pair_index <-> UID mapping stays
    with the outer manifest/logger; this function never sees UIDs.
    """
    requests: list[PairRequest] = []
    try:
        with Path(pairs_path).open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line, object_pairs_hook=_no_duplicate_keys)
                except json.JSONDecodeError as e:
                    raise RunnerError(f"pairs jsonl line {lineno}: invalid JSON: {e}") from e
                if not isinstance(obj, dict):
                    raise RunnerError(
                        f"pairs jsonl line {lineno}: expected a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                keys = set(obj)
                allowed = set(PAIR_INPUT_KEYS)
                missing = sorted(allowed - keys)
                extra = sorted(keys - allowed)
                if missing or extra:
                    raise RunnerError(
                        f"pairs jsonl line {lineno}: keys must be exactly "
                        f"{list(PAIR_INPUT_KEYS)}; missing={missing} extra={extra} "
                        "(UID/qrels/annotation fields are forbidden)"
                    )
                if not isinstance(obj["query_text"], str) or not isinstance(obj["candidate_json_desc"], str):
                    raise RunnerError(
                        f"pairs jsonl line {lineno}: both fields must be strings"
                    )
                requests.append(
                    PairRequest(
                        query_text=obj["query_text"],
                        candidate_json_desc=obj["candidate_json_desc"],
                    )
                )
    except OSError as e:
        raise RunnerError(f"cannot read pairs jsonl {pairs_path}: {e}") from e
    return requests

def _serialize_result(res: RunnerPairResult, keep_token_scores: bool) -> dict[str, Any]:
    """Compact per-pair record. Deliberately omits query_text / json_desc (the
    input JSONL + pair_index already identify the pair; the outer logger owns
    UID linkage)."""
    p = res.pair
    out: dict[str, Any] = {
        "pair_index": res.pair_index,
        "status": p.status,
        "reason": p.reason,
        "prefix_len": p.prefix_len,
        "full_len": p.full_len,
        "target_len": p.target_len,
        "within_training_context": p.within_training_context,
        "sum_logprob": p.sum_logprob,
        "mean_logprob": p.mean_logprob,
        "all_finite": p.all_finite,
        "target_ids": p.target_ids,
    }
    if keep_token_scores and res.token_logprobs is not None:
        out["token_logprobs"] = res.token_logprobs
    return out


def _atomic_write_new(path: Path, lines: Sequence[str]) -> None:
    """Publish a completed result exactly once, without replacing prior runs.

    The temporary file is on the destination filesystem. A hard link creates
    the final name atomically and fails if it already exists. A failed run may
    leave a charged ledger but cannot silently truncate a previous result.
    """
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
            raise RunnerError(f"output appeared during run, refusing overwrite: {path}") from exc
        except OSError as exc:
            raise RunnerError(f"cannot atomically publish {path}: {exc}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CADmium-3B NPU conditional scorer (M1b). Fail-closed gates run "
            "BEFORE any model load: (1) the input JSONL must contain ONLY the "
            "string fields {\"query_text\", \"candidate_json_desc\"} per line "
            "-- no UIDs, no qrels; (2) the FRESHLY FETCHED origin/master "
            "ACTIVE_TASK.md (origin URL validated, re-fetched every "
            "invocation; never the working-tree file) must state "
            "pilot_authorized with a full approved_sha equal to the live, "
            "DETACHED, clean git HEAD, plus max_scored_pairs > 0 and "
            "gpu_budget > 0; (3) --asset-manifest must SHA256-cover every "
            "file the load reads. There are no switches to relax these "
            "gates."
        )
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--system-message-config", required=True)
    parser.add_argument("--pairs-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--budget-file", default=None,
                        help="persisted cumulative budget ledger (default: "
                             "<out-jsonl>.budget.json); bound to the approval's "
                             "SHA / pair cap / wall-clock limit")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--torch-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--logits-chunk-rows", type=int, default=512)
    parser.add_argument("--asset-manifest", required=True,
                        help="REQUIRED JSON {base|adapter: {relpath: sha256}} "
                             "covering every file the load reads (weights, "
                             "index, configs, tokenizer); verified before any "
                             "forward")
    parser.add_argument("--repo-root", default=".",
                        help="git checkout root for the authorization gate; "
                             "ACTIVE_TASK.md is read from its FRESHLY FETCHED "
                             "origin/master (origin URL validated), never from "
                             "the working tree, and HEAD must be detached at "
                             "the approved SHA with a clean tree")
    parser.add_argument("--keep-token-scores", action="store_true",
                        help="also write per-target-token FP32 log-probs (spot checks)")
    parser.add_argument("--spot-check", action="store_true",
                        help="one pair, one forward: compare NPU target log-probs with audited CPU core")
    args = parser.parse_args(argv)

    pairs_path = Path(args.pairs_jsonl)
    # Gate 1 (review item 3): strict input isolation before anything else.
    try:
        requests = load_pair_requests(pairs_path)
    except RunnerError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    input_sha256 = _sha256_file(pairs_path)

    # Gate 2 (review item 2 + 2026-09-30 addendum): run authorization from the
    # freshly fetched origin/master ACTIVE_TASK.md + live detached/clean git.
    try:
        authorization = authorize_run(args.repo_root, cards_used=1)
    except RunnerError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    cfg = RunnerConfig(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        system_message_config=args.system_message_config,
        device=args.device,
        torch_dtype=args.torch_dtype,
        require_revision_in_path=True,
        asset_manifest=args.asset_manifest,
        logits_chunk_rows=args.logits_chunk_rows,
        budget_file=args.budget_file or (str(args.out_jsonl) + ".budget.json"),
        keep_token_scores=args.keep_token_scores,
    )

    out_path = Path(args.out_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else None
    repo_root = Path(args.repo_root).resolve()
    budget_path = Path(cfg.budget_file).resolve()
    for label, path in (("output", out_path), ("summary", summary_path), ("budget ledger", budget_path)):
        if path is not None and path.resolve().is_relative_to(repo_root):
            print(f"FATAL: {label} must be outside Git checkout: {path}", file=sys.stderr)
            return 2
    approved_manifest = repo_root / "experiments/configs/cadmium_asset_sha256_20260930.json"
    if Path(args.asset_manifest).resolve() != approved_manifest.resolve():
        print(f"FATAL: asset manifest must be the reviewed Git file {approved_manifest}", file=sys.stderr)
        return 2
    if args.spot_check:
        selection_file = repo_root / "experiments/configs/cadmium_m1b_pilot_selection_20260930.json"
        try:
            selection = json.loads(selection_file.read_text(encoding="utf-8"))
            expected_pair_sha = selection["pair_jsonl_sha256"]
        except (OSError, ValueError, KeyError) as exc:
            print(f"FATAL: cannot read reviewed spot-check selection {selection_file}: {exc}", file=sys.stderr)
            return 2
        if input_sha256 != expected_pair_sha:
            print("FATAL: spot-check pair SHA differs from reviewed selection", file=sys.stderr)
            return 2
    if out_path.exists() or (summary_path is not None and summary_path.exists()):
        print("FATAL: output or summary already exists; use a new run_id", file=sys.stderr)
        return 2
    scorer: CadmiumNpuScorer | None = None
    try:
        scorer = CadmiumNpuScorer(cfg, authorization)
        if args.spot_check:
            if len(requests) != 1 or args.batch_size != 1:
                raise RunnerError("--spot-check requires exactly one pair and batch-size 1")
            comparison = scorer.spot_check_with_cpu_core(requests[0])
            if not comparison.get("compared"):
                raise RunnerError(f"spot-check pair not scoreable: {comparison.get('reason')}")
            _atomic_write_new(out_path, [json.dumps(comparison)])
            n_written = 1
        else:
            results = scorer.score(requests, batch_size=args.batch_size)
            _atomic_write_new(
                out_path,
                [json.dumps(_serialize_result(res, args.keep_token_scores)) for res in results],
            )
            n_written = len(results)
        summary = {
            "status": "ok",
            "spot_check": bool(args.spot_check),
            "pairs_input": len(requests),
            "pairs_written": n_written,
            "batch_size": args.batch_size,
            "input_pairs_jsonl": str(pairs_path.resolve()),
            "input_pairs_sha256": input_sha256,
            "out_jsonl": str(out_path.resolve()),
            **scorer.summary_record(),
        }
    except (RunnerError, ScoringError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        summary = {"status": "failed", "error": str(e)}
        if scorer is not None:
            summary.update(scorer.summary_record())
        if summary_path and not summary_path.exists():
            _atomic_write_new(summary_path, [json.dumps(summary, indent=2)])
        return 2

    if summary_path:
        _atomic_write_new(summary_path, [json.dumps(summary, indent=2)])
    print(json.dumps({"status": "ok", "pairs_written": summary["pairs_written"],
                      "ledger": summary["ledger"]}, indent=2))
    return 0


__all__ = [
    "BASE_MODEL_ID",
    "ADAPTER_MODEL_ID",
    "EXPECTED_BASE_REVISION",
    "EXPECTED_ADAPTER_REVISION",
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    "REASON_TARGET_OUT_OF_VOCAB",
    "RunnerError",
    "RunnerConfig",
    "PairRequest",
    "RunnerPairResult",
    "BudgetLedger",
    "RunAuthorization",
    "authorize_run",
    "parse_active_task_authorization",
    "preflight_assets",
    "load_tokenizer",
    "load_model",
    "load_pair_requests",
    "CadmiumNpuScorer",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
