"""P1 matrix-runner pure helpers: pair-id parsing, statuses, stop policy,
resume planning, cumulative budget ledger, and pre-flight authorization checks.

Kept torch-free and unit-testable; the heavy runner (run_p1_matrix.py) wires
these. Codex review 2026-09-25 (R1/R3/R4):

  R1  Staged invocations in one run dir must never re-score or hide cost:
      existing records are verified (unique pair ids, status ok, identical
      selection/git/model/data/dtype/env hashes) BEFORE weights load; new
      forwards and wallclock accumulate in budget_ledger.json across
      invocations; the authorized whole-run caps (416 forwards / 45 min / 1
      card-hour) bind across ALL invocations. Staged plan: 1 -> +9 -> +404,
      ending with exactly 414 unique raw records.
  R3  Authorization is enforced BEFORE loading weights and is read from the
      Codex-controlled `origin/master:experiments/ACTIVE_TASK.md` AFTER a git
      fetch — never from the detached HEAD's working tree: a commit cannot
      contain its own approved_sha, so the working-tree file can never prove
      authorization for the checked-out code (Codex review addendum). The
      origin remote URL is validated against the expected project remote;
      ACTIVE_TASK must be `pilot_authorized` with `approved_sha` == HEAD and
      `approved_selection_sha256` == the selection manifest's sha256; the
      model/data runtime copies must pass their SHA256SUMS. CLI cannot relax
      budgets or override model/data dirs.
  R4  out_dir must be outside the Git checkout; invocation ids are unique
      (timestamp + uuid) so environment files never overwrite; an unfinalized
      ledger entry (crash) refuses the next invocation — no silent resume.

P1 vs P0 stop policy: an overlong planned pair is an ANOMALY that stops the
run, because the CPU gate pre-verified every planned pair at <= max_seq_len.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

MAX_PAIRS = 416  # 256 conditional + <=160 empty (plan sec.2: total forwards <= 416 < 512)

EXIT_OK = 0
EXIT_ANOMALY = 2
EXIT_BUDGET = 3

STATUS_OK = "ok"
STATUS_BOUNDARY = "boundary_error"
STATUS_NONFINITE = "nonfinite"
STATUS_ERROR = "error"
STATUS_OVERLONG = "overlong_sequence"

STOP_STATUSES = frozenset({STATUS_BOUNDARY, STATUS_NONFINITE, STATUS_ERROR, STATUS_OVERLONG})

# Authorized whole-run caps (plan sec.4); config/CLI may only tighten these.
HARD_CAPS = {"max_forwards": 416, "max_wallclock_s": 2700.0, "max_card_hours": 1.0}

LEDGER_FILE = "budget_ledger.json"


def should_stop(status: str) -> bool:
    """True => record the failed pair, then abort the run with EXIT_ANOMALY."""
    return status in STOP_STATUSES


def parse_pair_ids(spec: str, n_pairs: int, max_pairs: int = MAX_PAIRS) -> list[int]:
    """Parse '0', '1-59', 'all', '0,3,5-7'. Deduplicates (order preserved) and
    enforces the task pair cap. Raises ValueError on out-of-range or >max_pairs."""
    if n_pairs > max_pairs:
        raise ValueError(f"pair plan has {n_pairs} pairs, exceeds hard cap {max_pairs}")
    if spec == "all":
        ids = list(range(n_pairs))
    else:
        ids = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                ids.extend(range(int(a), int(b) + 1))
            else:
                ids.append(int(part))
    seen: set[int] = set()
    deduped: list[int] = []
    for i in ids:
        if not (0 <= i < n_pairs):
            raise ValueError(f"pair index {i} out of range [0, {n_pairs})")
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    if len(deduped) > max_pairs:
        raise ValueError(f"{len(deduped)} unique pairs requested, exceeds cap {max_pairs}")
    return deduped


# --------------------------------------------------------------- budgets (R4)

def effective_budgets(cfg_budget: dict) -> dict:
    """Config may only TIGHTEN the authorized caps, never relax them (R4)."""
    return {
        "max_forwards": min(int(cfg_budget.get("max_forwards", HARD_CAPS["max_forwards"])), HARD_CAPS["max_forwards"]),
        "max_wallclock_s": min(float(cfg_budget.get("max_wallclock_s", HARD_CAPS["max_wallclock_s"])), HARD_CAPS["max_wallclock_s"]),
        "max_card_hours": min(float(cfg_budget.get("npu_card_hours", HARD_CAPS["max_card_hours"])), HARD_CAPS["max_card_hours"]),
    }


def new_invocation_id() -> str:
    """Unique per invocation (timestamp + uuid8): environment files never
    overwrite each other even for sub-second repeated calls (R4)."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]


def ensure_outside_checkout(out_dir: Path, repo_root: Path) -> None:
    """R4: refuse to write run outputs inside the Git checkout."""
    out_resolved = Path(out_dir).resolve()
    repo_resolved = Path(repo_root).resolve()
    if out_resolved == repo_resolved or repo_resolved in out_resolved.parents:
        raise ValueError(f"out_dir {out_resolved} is inside the Git checkout {repo_resolved}")


# ---------------------------------------------------- existing records (R1)

RECORD_BINDING_FIELDS = (
    "selection_sha256",
    "git_sha",
    "model_dir",
    "data_dir",
    "model_revision",
    "data_revision",
    "dtype",
    "env_spec_canonical_json_sha256",
    "env_spec_file_bytes_sha256",
)


def verify_existing_records(records: list[dict], expected: dict) -> list[str]:
    """R1: every pre-existing record in the run dir must be a UNIQUE, fully
    successful pair scored under the identical selection/git/model/data/dtype/
    env binding. `expected` maps RECORD_BINDING_FIELDS to current values.
    Returns a list of problems (empty = OK)."""
    problems: list[str] = []
    seen: set[str] = set()
    for r in records:
        pid = str(r.get("pair_id"))
        if pid in seen:
            problems.append(f"duplicate pair_id {pid!r} in existing records")
            continue
        seen.add(pid)
        if r.get("status") != STATUS_OK or not r.get("all_finite"):
            problems.append(f"{pid}: existing record not ok/finite (status={r.get('status')})")
        for f in RECORD_BINDING_FIELDS:
            want = expected.get(f)
            if f in expected and r.get(f) != want:
                problems.append(f"{pid}: field {f} mismatch: record={r.get(f)!r} current={want!r}")
    return problems


def plan_invocation(existing_records: list[dict], requested_ids: list[int], pairs: list[dict], resume: bool) -> tuple[list[int], list[str]]:
    """R1: decide which requested pair indices still need scoring.

    Returns (to_score_indices, skipped_completed_pair_ids). Skips are LOUD
    (returned for logging/manifest), never silent. Raises ValueError if the
    run dir has records but --resume was not given."""
    if existing_records and not resume:
        raise ValueError(
            f"output dir already contains {len(existing_records)} pair records; "
            "re-running would duplicate scores and break the cumulative budget. "
            "Pass --resume to continue the SAME run after verification, or stop."
        )
    completed = {r["pair_id"] for r in existing_records}
    to_score: list[int] = []
    skipped: list[str] = []
    for pid in requested_ids:
        pair_id = pairs[pid]["pair_id"]
        if pair_id in completed:
            skipped.append(pair_id)
        else:
            to_score.append(pid)
    return to_score, skipped


# ------------------------------------------------------------ budget ledger (R1/R4)

class BudgetLedger:
    """Cumulative whole-run accounting across invocations (R1/R4).

    The ledger file lives in the run dir. A pending (unfinalized) invocation
    entry means the previous process crashed mid-run: the next invocation is
    REFUSED (no silent resume; cost accounting would be uncertain)."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    @classmethod
    def load(cls, path: Path, selection_sha256: str, caps: dict) -> "BudgetLedger":
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("selection_sha256") != selection_sha256:
                raise ValueError("ledger selection_sha256 mismatch (mixed runs in one dir)")
            if data.get("caps") != caps:
                raise ValueError(f"ledger caps {data.get('caps')} != current {caps} (budgets must not change mid-run)")
            pending = [i.get("invocation_id") for i in data.get("invocations", []) if not i.get("finalized")]
            if pending:
                raise ValueError(
                    f"ledger has unfinalized invocation(s) {pending}: previous run crashed or was killed; "
                    "stop and report to Codex - no automatic resume with uncertain cost accounting"
                )
        else:
            data = {
                "version": 1,
                "selection_sha256": selection_sha256,
                "caps": dict(caps),
                "invocations": [],
                "cumulative_forwards": 0,
                "cumulative_wall_s": 0.0,
            }
        return cls(path, data)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def cumulative(self) -> tuple[int, float]:
        return int(self.data["cumulative_forwards"]), float(self.data["cumulative_wall_s"])

    def check_can_start(self, planned_new_forwards: int) -> list[str]:
        caps = self.data["caps"]
        cum_f, cum_w = self.cumulative()
        problems = []
        if cum_f + planned_new_forwards > caps["max_forwards"]:
            problems.append(
                f"forwards budget: {cum_f} done + {planned_new_forwards} planned > cap {caps['max_forwards']}"
            )
        if cum_w >= caps["max_wallclock_s"]:
            problems.append(f"wallclock budget already exhausted: {cum_w:.1f}s >= {caps['max_wallclock_s']}s")
        if cum_w / 3600.0 >= caps["max_card_hours"]:
            problems.append(f"card-hour budget already exhausted: {cum_w/3600:.3f} >= {caps['max_card_hours']}")
        return problems

    def budget_exceeded(self, invocation_elapsed_s: float) -> str | None:
        """Checked inside the scoring loop; returns the violated cap name or None."""
        caps = self.data["caps"]
        cum_f, cum_w = self.cumulative()
        wall = cum_w + invocation_elapsed_s
        if cum_f >= caps["max_forwards"]:
            return "max_forwards"
        if wall > caps["max_wallclock_s"]:
            return "max_wallclock_s"
        if wall / 3600.0 > caps["max_card_hours"]:
            return "max_card_hours"
        return None

    def begin(self, invocation_id: str) -> None:
        self.data["invocations"].append(
            {
                "invocation_id": invocation_id,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "finalized": False,
            }
        )
        self.save()

    def finalize(self, invocation_id: str, wall_s: float, forwards: int, exit_code: int, note: str = "") -> None:
        for inv in self.data["invocations"]:
            if inv["invocation_id"] == invocation_id and not inv.get("finalized"):
                inv.update(
                    finalized=True,
                    ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    wall_s=round(wall_s, 2),
                    forwards=forwards,
                    exit_code=exit_code,
                    note=note,
                )
                break
        else:
            raise ValueError(f"no pending ledger entry for {invocation_id}")
        self.data["cumulative_forwards"] += forwards
        self.data["cumulative_wall_s"] = round(self.data["cumulative_wall_s"] + wall_s, 2)
        self.save()


# ------------------------------------------------------- authorization (R3)

EXPECTED_ORIGIN_URL = "https://cnb.cool/lzacode/CADRetrieval"
EXPECTED_ORIGIN_URL_GITHUB_MIRROR = "https://github.com/anszliuziang/CADRetrieval"
#: Trusted-remote allowlist (2026-11-12 E2 migration): CNB is the historical
#: authorized remote (share2); the GitHub mirror is the share4 checkout origin.
#: Both are accepted by default; anything else is refused. Callers that pass an
#: explicit `expected_origin` still get exact-match semantics (tests, toys).
EXPECTED_ORIGIN_URLS = (EXPECTED_ORIGIN_URL, EXPECTED_ORIGIN_URL_GITHUB_MIRROR)
ACTIVE_TASK_BLOB = "experiments/ACTIVE_TASK.md"
AUTHORIZATION_REF = "origin/master"


def normalize_remote_url(url: str) -> str:
    u = url.strip().lower().rstrip("/")
    if u.endswith(".git"):
        u = u[: -len(".git")]
    return u


def origin_allowed(url: str, allowed_origins=EXPECTED_ORIGIN_URLS) -> bool:
    want = {normalize_remote_url(o) for o in allowed_origins}
    return normalize_remote_url(url) in want


def fetch_authorized_active_task(repo_root: Path, expected_origin: str | None = None, ref: str = AUTHORIZATION_REF) -> tuple[str | None, dict]:
    """Read the authorization from the Codex-controlled origin ref AFTER a
    fetch OF THAT REF, never from the detached HEAD's working tree (a commit
    cannot contain its own approved_sha). Validates the origin remote URL
    against the trusted allowlist (or an explicitly required origin).

    Review round 3 fix: a bare `git fetch origin` succeeds even when the
    remote branch was DELETED, silently leaving a stale remote-tracking ref.
    We therefore fetch the exact refspec `+refs/heads/<branch>:refs/remotes/
    origin/<branch>` and refuse unless that fetch succeeds AND the tracking
    ref resolves; the blob is then read from the explicitly captured commit
    SHA, not from the movable symbolic ref.

    Returns (text, meta): text is None on refusal; meta carries audit fields
    (origin url, fetched commit sha, blob sha256, fetch rc) and problems."""
    meta: dict = {"expected_origin": expected_origin, "ref": ref, "problems": []}
    repo_root = Path(repo_root)

    def git(*args: str, timeout: int = 120, text: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=text, timeout=timeout, env=env)

    try:
        url = git("remote", "get-url", "origin").stdout.strip()
    except Exception as e:  # noqa: BLE001
        meta["problems"].append(f"cannot read origin remote url: {e}")
        return None, meta
    meta["origin_url"] = url
    if expected_origin is not None:
        origin_ok = normalize_remote_url(url) == normalize_remote_url(expected_origin)
        expected_desc = expected_origin
    else:
        origin_ok = origin_allowed(url)
        expected_desc = "allowlist " + " | ".join(EXPECTED_ORIGIN_URLS)
    if not origin_ok:
        meta["problems"].append(f"origin remote {url!r} not authorized (expected {expected_desc})")
        return None, meta
    if not ref.startswith("origin/"):
        meta["problems"].append(f"authorization ref must name an origin tracking branch: {ref!r}")
        return None, meta
    branch = ref[len("origin/"):]
    try:
        r = git("fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", timeout=300)
        meta["fetch_returncode"] = r.returncode
        meta["fetch_refspec"] = f"+refs/heads/{branch}:refs/remotes/origin/{branch}"
        if r.returncode != 0:
            meta["problems"].append(
                f"fetching authorization branch failed (branch missing on remote or network): "
                f"{r.stderr.strip()[:300]}"
            )
            return None, meta
    except Exception as e:  # noqa: BLE001
        meta["problems"].append(f"git fetch error: {type(e).__name__}: {e}")
        return None, meta
    try:
        resolved = git("rev-parse", f"refs/remotes/origin/{branch}^{{commit}}").stdout.strip()
        if len(resolved) != 40 or any(c not in "0123456789abcdef" for c in resolved):
            raise ValueError(f"unresolved commit: {resolved!r}")
        meta["origin_ref_sha"] = resolved
        r = git("show", f"{resolved}:{ACTIVE_TASK_BLOB}", text=False)
        if r.returncode != 0:
            meta["problems"].append(f"cannot read {resolved}:{ACTIVE_TASK_BLOB}")
            return None, meta
        blob = r.stdout
        meta["active_task_source"] = f"origin/{branch}:{ACTIVE_TASK_BLOB} @ {resolved}"
        meta["active_task_blob_sha256"] = hashlib.sha256(blob).hexdigest()
        return blob.decode("utf-8"), meta
    except Exception as e:  # noqa: BLE001
        meta["problems"].append(f"error reading authorized ACTIVE_TASK: {type(e).__name__}: {e}")
        return None, meta


def head_is_detached(repo_root: Path) -> tuple[bool, str]:
    """True iff HEAD is detached (plan requires a clean detached approved SHA)."""
    repo_root = Path(repo_root)
    r = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo_root, capture_output=True, text=True
    )
    if r.returncode != 0:
        return True, ""
    return False, r.stdout.strip()


def active_task_field(text: str, key: str) -> str | None:
    """Extract `key`: value from ACTIVE_TASK.md. Value is read up to the next
    backtick, full-width parenthesis, semicolon or line end (Codex format:
    - `key`: `value`(annotation))."""
    m = re.search(rf"^\s*-?\s*`{re.escape(key)}`:\s*(.+?)\s*$", text, re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip()
    if val.startswith("`"):
        val = val[1:]
    val = re.split(r"[`<；（；]", val)[0].strip()
    return val or None


def check_authorization(active_task_text: str, git_sha: str, selection_sha256: str, task_id: str) -> list[str]:
    """R3: hard authorization gate on the (fetched origin/master) ACTIVE_TASK,
    evaluated BEFORE loading weights.

    Requires: state == pilot_authorized, approved_sha == current HEAD,
    approved_selection_sha256 == the manifest's sha256 (Codex pins the
    reviewed manifest when authorizing), task_id match."""
    problems: list[str] = []
    state = active_task_field(active_task_text, "state")
    approved_sha = active_task_field(active_task_text, "approved_sha")
    approved_sel = active_task_field(active_task_text, "approved_selection_sha256")
    at_task = active_task_field(active_task_text, "task_id")
    if at_task is not None and at_task != task_id:
        problems.append(f"ACTIVE_TASK task_id {at_task!r} != config {task_id!r}")
    if state != "pilot_authorized":
        problems.append(f"ACTIVE_TASK state is {state!r}, not 'pilot_authorized' - NPU run NOT authorized")
    if not approved_sha:
        problems.append("ACTIVE_TASK approved_sha missing")
    elif approved_sha != git_sha:
        problems.append(f"approved_sha {approved_sha} != current HEAD {git_sha}")
    if not approved_sel:
        problems.append(
            "ACTIVE_TASK approved_selection_sha256 missing - Codex must pin the reviewed manifest SHA when authorizing"
        )
    elif approved_sel != selection_sha256:
        problems.append(f"approved_selection_sha256 {approved_sel} != selection file sha256 {selection_sha256}")
    return problems


# ---------------------------------------------------------- asset hashes (R3)

def verify_sha256sums(dir_path: Path) -> tuple[int, list[str]]:
    """Python equivalent of `sha256sum -c SHA256SUMS` inside dir_path (R3).
    Verifies every listed file; returns (n_ok, failures)."""
    dir_path = Path(dir_path)
    sums = dir_path / "SHA256SUMS"
    if not sums.exists():
        return 0, [f"missing {sums}"]
    n_ok = 0
    failures: list[str] = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            hexpart, rel = line.split(None, 1)
        except ValueError:
            failures.append(f"unparseable SHA256SUMS line: {line!r}")
            continue
        rel = rel.strip().lstrip("*")
        f = dir_path / rel
        if not f.exists():
            failures.append(f"missing file {rel}")
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != hexpart.lower():
            failures.append(f"sha256 mismatch {rel}")
        else:
            n_ok += 1
    if n_ok == 0 and not failures:
        failures.append(f"empty SHA256SUMS in {dir_path}")
    return n_ok, failures
