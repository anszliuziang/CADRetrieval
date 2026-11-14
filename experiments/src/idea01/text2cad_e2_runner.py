"""E2 empty-text background scorer: dedicated single-card entry (plan
IDEA01_E2_EMPTY_TEXT_BACKGROUND_2026-11-12.md, v2; fresh-review round applied).

Four frozen background manifests (built by build_text2cad_e2_manifests.py,
fingerprints AND counts pinned in the committed
configs/text2cad_e2_background_expected_20261112.json):

  b0        texts = [""]               -> 1 x 8046 pairs (primary background)
  b1        texts = ["a CAD model."]   -> 1 x 8046 pairs (pre-registered sensitivity)
  smoke_b0  texts = [""]               -> 1 x 1 pair (contractual 1-pair smoke,
            scored into the SAME shared ledger BEFORE the b0 batch)
  replay    texts = 8 pinned G4 texts  -> 8 x 1024 pairs (cross-machine
            comparability gate against share2 G4 records BEFORE any
            differencing; tolerance |ds| <= max(1e-3, 1e-6*|s|) and n_valid
            equality, checked analysis-side)

Gates (identical input-validation functions for --mode dry-run and --mode
score, per review; score additionally authorizes and charges budget):
  * committed expected-config fingerprints AND counts for all four manifests;
  * B0/B1/smoke admitted ONLY via load_background_texts (the general
    load_texts contract stays untouched);
  * candidate store byte-identical to the original pair_manifest_v4 cad_store
    (b0/b1/smoke: v4 prefixes of 8,046/8,046/1; replay: prefix of 1,024 rows —
    prefixes preserve cad_index equality with G4);
  * tokenizer evidence for B0/B1 recorded ([CLS]/[SEP] expectation for B0);
  * score mode ONLY: ACTIVE_TASK fetched from the plan-pinned authorization
    ref on an allowlisted origin (CNB / GitHub mirror) must state
    pilot_authorized, e2_approved_sha == detached clean HEAD, positive
    e2_max_pairs / e2_max_card_seconds and a parseable e2_deadline_utc.

Budget accounting (review C1 fix): ONE canonical ledger per approved SHA
(OUTSIDE the checkout, shared by smoke/b0/b1/replay and all invocations) —
per-invocation per-background out-dirs can no longer reset the cumulative
cap; the ledger refuses a pending (crashed) invocation outright — recovery is
a reviewer decision, never an automatic resume; chunks live under
<out-dir>/<which>/ so b0/b1/replay never collide; --resume only skips
verified chunks and still goes through the shared ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idea01.p1_runner import (  # noqa: E402
    active_task_field,
    fetch_authorized_active_task,
    head_is_detached,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_STORE_FILE,
    PAIR_MANIFEST_FILE,
    TEXTS_FILE,
    load_background_texts,
    load_cad_store,
    load_pair_manifest,
    load_texts,
    sha256_file,
)

E2_CONFIG_REL = "experiments/configs/text2cad_e2_background_expected_20261112.json"
#: injectable clock for controllable-time regressions (production = real time)
NOW_FN = time.time
B0_TEXT = ""
B1_TEXT = "a CAD model."
#: expected background text per manifest (replay uses the general load_texts)
EXPECTED_TEXT = {"b0": B0_TEXT, "b1": B1_TEXT, "smoke_b0": B0_TEXT}
BATCH_SIZE = 16  # G4 protocol (honest tail)
REPLAY_TOLERANCE_ABS = 1e-3
REPLAY_TOLERANCE_REL = 1e-6
#: exact success notes written into the shared ledger on a successful finalize;
#: the prerequisite gates match these EXACTLY (execution order is auditable)
SUCCESS_NOTE = {
    "smoke_b0": "e2_smoke_b0_ok",
    "replay": "e2_replay_ok",
    "b0": "e2_b0_ok",
    "b1": "e2_b1_ok",
}
SMOKE_NOTE = SUCCESS_NOTE["smoke_b0"]
B0_OK_NOTE = SUCCESS_NOTE["b0"]


class E2Error(RuntimeError):
    """Hard refusal; exit non-zero, no forward."""


def load_e2_config(repo_root: Path) -> dict:
    path = Path(repo_root) / E2_CONFIG_REL
    if not path.is_file():
        raise E2Error(f"committed E2 expected-config missing: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "text2cad-e2-background-expected-v1":
        raise E2Error(f"E2 config schema mismatch: {cfg.get('schema')!r}")
    for key in ("authorization_ref", "backgrounds", "original_cad_store_sha256",
                "replay_texts", "cad_seq_zip_sha256", "g4_sha256sums_sha256",
                "g4_chunk_sha256", "replay_dense_to_original_cad_index"):
        if key not in cfg:
            raise E2Error(f"E2 expected-config missing key {key!r}")
    for which, pins in cfg["backgrounds"].items():
        for k in ("n_unique_text", "n_gallery"):
            if not isinstance(pins.get(k), int) or pins[k] <= 0:
                raise E2Error(f"E2 config backgrounds[{which}] missing positive {k}")
    return cfg


def manifest_counts(cfg: dict, which: str) -> tuple[int, int]:
    pins = cfg["backgrounds"][which]
    return int(pins["n_unique_text"]), int(pins["n_gallery"])


def git_head(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(repo_root), capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def git_dirty(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=Path(repo_root), capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _manifest_fingerprint_ok(cfg: dict, which: str, manifest_dir: Path) -> dict:
    pins = cfg["backgrounds"][which]
    manifest = load_pair_manifest(
        manifest_dir,
        expected_sha256={
            TEXTS_FILE: pins["texts_sha256"],
            CAD_STORE_FILE: pins["cad_store_sha256"],
            "cad_length_index.jsonl": pins["cad_length_index_sha256"],
            PAIR_MANIFEST_FILE: pins["pair_manifest_sha256"],
        },
    )
    counts = manifest["counts"]
    want_t, want_c = manifest_counts(cfg, which)
    if counts.get("n_unique_text") != want_t or counts.get("n_gallery") != want_c:
        raise E2Error(
            f"{which}: counts {counts.get('n_unique_text')}x{counts.get('n_gallery')} != {want_t}x{want_c}"
        )
    return manifest


def validate_e2_manifests(cfg: dict, dirs: dict[str, Path], original_manifest_dir: Path) -> dict:
    """Shared validation for dry-run AND score (identical checks, per review)."""
    report: dict[str, Any] = {"backgrounds": {}}
    original_store_sha = sha256_file(Path(original_manifest_dir) / CAD_STORE_FILE)
    if original_store_sha != cfg["original_cad_store_sha256"]:
        raise E2Error(
            f"original pair_manifest_v4 cad_store sha {original_store_sha} != pinned "
            f"{cfg['original_cad_store_sha256']}"
        )
    for which, expected in EXPECTED_TEXT.items():
        if which not in dirs or dirs[which] is None:
            raise E2Error(f"missing manifest dir for {which!r}")
        d = Path(dirs[which])
        if not d.is_dir():
            raise E2Error(f"{which}: manifest dir does not exist: {d}")
        _manifest_fingerprint_ok(cfg, which, d)
        text = load_background_texts(d, expected)
        store_sha = sha256_file(d / CAD_STORE_FILE)
        allowed = {cfg["original_cad_store_sha256"], cfg["backgrounds"][which]["cad_store_sha256"]}
        if store_sha not in allowed:
            raise E2Error(f"{which}: cad_store sha {store_sha} is neither the v4 original nor its pinned prefix")
        members = load_cad_store(d)
        want_t, want_c = manifest_counts(cfg, which)
        if len(members) != want_c:
            raise E2Error(f"{which}: cad store has {len(members)} members, expected {want_c}")
        report["backgrounds"][which] = {"text": text, "n_texts": want_t, "n_cads": want_c}
    d = Path(dirs["replay"])
    _manifest_fingerprint_ok(cfg, "replay", d)
    texts = load_texts(d)
    pinned = cfg["replay_texts"]
    if len(texts) != len(pinned):
        raise E2Error(f"replay texts {len(texts)} != pinned {len(pinned)}")
    for i, (got, want) in enumerate(zip(texts, pinned)):
        if got != want["text"]:
            raise E2Error(f"replay text {i} differs from pinned original text {want['original_text_index']}")
    members = load_cad_store(d)
    _, want_c = manifest_counts(cfg, "replay")
    if len(members) != want_c:
        raise E2Error(f"replay cad store has {len(members)} members, expected {want_c} (prefix)")
    report["backgrounds"]["replay"] = {"n_texts": len(texts), "n_cads": want_c}
    return report


def tokenizer_evidence(bert_dir: Path) -> dict:
    """Slow BERT tokenizer evidence for B0/B1 (dry-run + score share this)."""
    from transformers import BertTokenizer  # tokenizer-only; no torch needed

    tok = BertTokenizer.from_pretrained(str(bert_dir), use_fast=False)
    ev: dict[str, Any] = {}
    for name, text in EXPECTED_TEXT.items():
        ids = tok(text, add_special_tokens=True)["input_ids"]
        if not ids or len(ids) > 512:
            raise E2Error(f"{name}: background encoding empty or >512 tokens: {ids!r}")
        ev[name] = {"text": text, "input_ids": ids, "n_tokens": len(ids)}
    ev["b0_expectation"] = "CLS/SEP-only (ids [101, 102] for bert-large-uncased vocab); recorded, not hardcoded as a gate"
    return ev


def _int_cap(value: str | None, field: str) -> int:
    try:
        v = int(str(value or "").replace(",", "").strip())
    except ValueError:
        raise E2Error(f"{field} missing/non-integer: {value!r}") from None
    if v <= 0:
        raise E2Error(f"{field} must be positive: {value!r}")
    return v


def parse_e2_authorization(text: str, approved_sha: str) -> dict:
    state = active_task_field(text, "state")
    sha = active_task_field(text, "e2_approved_sha")
    problems: list[str] = []
    if state != "pilot_authorized":
        problems.append(f"state is {state!r}, not pilot_authorized")
    if not sha or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        problems.append(f"e2_approved_sha missing/malformed: {sha!r}")
    elif sha != approved_sha:
        problems.append(f"e2_approved_sha {sha} != HEAD {approved_sha}")
    caps: dict[str, Any] = {}
    try:
        caps["max_pairs"] = _int_cap(active_task_field(text, "e2_max_pairs"), "e2_max_pairs")
    except E2Error as e:
        problems.append(str(e))
    max_card_seconds = active_task_field(text, "e2_max_card_seconds")
    try:
        caps["max_card_seconds"] = float(str(max_card_seconds or "").replace(",", "").strip())
        if not math.isfinite(caps["max_card_seconds"]) or caps["max_card_seconds"] <= 0:
            raise ValueError
    except ValueError:
        problems.append(f"e2_max_card_seconds missing/non-finite/non-positive: {max_card_seconds!r}")
    deadline = active_task_field(text, "e2_deadline_utc")
    try:
        dt = datetime.strptime(deadline or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        caps["deadline_epoch"] = dt.timestamp()
    except ValueError:
        problems.append(f"e2_deadline_utc missing/unparseable (want YYYY-MM-DDTHH:MM:SSZ): {deadline!r}")
    if problems:
        raise E2Error("E2 authorization refused: " + "; ".join(problems))
    return {"approved_sha": sha, **caps}


def authorize_e2_run(repo_root: Path, cfg: dict) -> dict:
    ref = cfg["authorization_ref"]
    text, meta = fetch_authorized_active_task(Path(repo_root), ref=ref)  # allowlisted origin
    if text is None:
        raise E2Error(f"authorization fetch refused: {meta.get('problems')}")
    head = git_head(repo_root)
    auth = parse_e2_authorization(text, head)
    auth["authorization_meta"] = meta
    auth["authorization_ref"] = ref
    return auth


GRACE_S = 30.0  # termination grace, COUNTED INTO the caps (KILL lands by the deadline)


def _supervise_loop(parent_pid: int, ledger_path: str, wall_deadline: float,
                    max_card_seconds: float, cards_used: int = 1,
                    now_fn=time.time, sleep_fn=time.sleep, ready_event=None):
    """Independent supervision PROCESS (G4-style, round 7): signals are
    scheduled on ABSOLUTE times computed from the FIRST ledger snapshot - no
    second sample is needed. With N cards, card-seconds accrue at N per wall
    second, so the card-second deadline is deterministic:

      card_deadline = first_snapshot_time + (cap - spent_at_snapshot) / N
      hard          = min(wall_deadline, card_deadline)
      TERM at hard - GRACE_S
      KILL at hard                     (grace counted INTO the caps; the child
                                        is dead BY the deadline)

    The wall arm is independent of ledger reads (a read failure never skips
    it). Once TERM is sent the supervisor ONLY waits for the unsent KILL at
    its absolute time - no further ledger polling. If the ledger later shows
    the cap NOT reached by TERM time that is irrelevant: the conservative
    first-snapshot arithmetic stands. now_fn/sleep_fn are injectable for
    controllable-clock regressions. A stuck child that IGNORES SIGTERM is
    killed by SIGKILL at `hard` (the POSIX regression asserts the exit signal
    and the timing).
    """
    import signal

    def hard_kill_at(reason: str, at: float) -> None:
        print(f"E2 supervision: SIGTERM to {parent_pid} at {at:.3f} ({reason})",
              file=sys.stderr, flush=True)
        try:
            os.kill(parent_pid, signal.SIGTERM)
        except OSError:
            pass
        while now_fn() < at - 1e-9:
            sleep_fn(max(0.0, min(1.0, at - now_fn())))
        print(f"E2 supervision: SIGKILL to {parent_pid} at {at:.3f}", file=sys.stderr,
              flush=True)
        try:
            os.kill(parent_pid, signal.SIGKILL)
        except OSError:
            pass
        os._exit(3)

    # FIRST snapshot fixes the absolute card deadline immediately (round 7,
    # item 1): cards_used card-seconds accrue per wall second, so the remaining
    # (cap - spent) card-seconds are exactly (cap - spent)/N wall seconds away.
    first = None
    while first is None:
        now = now_fn()
        readable = False
        try:
            data = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
            readable = True
            started = None
            for inv in reversed(data.get("invocations", [])):
                if not inv.get("finalized"):
                    started = float(inv["started_epoch"])
                    break
            if started is not None:
                spent = float(data.get("card_seconds_spent", 0.0)) + max(
                    0.0, now - started) * cards_used
                first = (now, spent)
        except Exception:
            pass  # ledger not written yet; retry on the next tick
        if first is None:
            if now >= wall_deadline:
                if ready_event is not None:
                    ready_event.set()  # report the wall-only fallback schedule
                hard_kill_at("wall deadline while ledger unreadable", now)
            # readable-but-unregistered means the scorer is mid-registration:
            # retry FAST so the absolute deadline reflects the real start
            # (round 8, item 1); only a truly unreadable ledger backs off 5s
            sleep_fn(0.1 if readable else min(5.0, max(0.05, wall_deadline - now)))
            continue

    snap_t, snap_spent = first
    card_deadline = snap_t + max(0.0, max_card_seconds - snap_spent) / cards_used
    hard = min(wall_deadline, card_deadline)
    term_at = hard - GRACE_S
    # readiness is signalled ONLY after the absolute schedule (term_at/hard)
    # is computed (review round 9, item 2)
    if ready_event is not None:
        ready_event.set()
    now = now_fn()
    if now >= hard:
        hard_kill_at("hard deadline already passed at supervision start", hard)

    term_sent = False
    while True:
        now = now_fn()
        if not term_sent and now >= term_at:
            print(f"E2 supervision: SIGTERM to {parent_pid} at {term_at:.3f} "
                  f"(hard {hard:.3f})", file=sys.stderr, flush=True)
            try:
                os.kill(parent_pid, signal.SIGTERM)
            except OSError:
                pass
            term_sent = True
        if term_sent:
            # ONLY wait for the unsent KILL at its absolute time; no ledger
            # polling after TERM (round 7, item 1)
            while now_fn() < hard - 1e-9:
                sleep_fn(max(0.0, min(1.0, hard - now_fn())))
            print(f"E2 supervision: SIGKILL to {parent_pid} at {hard:.3f}",
                  file=sys.stderr, flush=True)
            try:
                os.kill(parent_pid, signal.SIGKILL)
            except OSError:
                pass
            os._exit(3)
        nxt = min(term_at, wall_deadline)
        sleep_fn(max(0.05, min(5.0, nxt - now_fn())))


def start_deadline_supervision(auth: dict, ledger_path: Path, ready_event=None):
    """Spawn the independent supervisor process (POSIX only; elsewhere the
    in-loop checks and the operator's outer timeout remain, recorded honestly).
    `ready_event` (multiprocessing.Event) is set once the supervisor has taken
    its first snapshot / computed the absolute deadline. Returns the process
    handle or None."""
    import multiprocessing as mp

    if os.name != "posix":
        if ready_event is not None:
            ready_event.set()
        return None
    proc = mp.Process(
        target=_supervise_loop,
        args=(os.getpid(), str(ledger_path), float(auth["deadline_epoch"]),
              float(auth["max_card_seconds"]), 1),
        kwargs={"ready_event": ready_event},
        daemon=True,
    )
    proc.start()
    return proc


def amend_ledger_verdict_on_disk(ledger_path: Path, ledger, invocation_id: str,
                                  exit_code: int, note: str) -> None:
    """Settlement correction holding the LEDGER'S OWN LOCK for the whole
    read-modify-write (review round 9, item 1): a concurrent stage's
    charge/begin can only run before or after the amendment, never between
    its read and write, so no lost updates are possible. Inside the lock the
    freshest on-disk state is re-read, ONLY the matching invocation's verdict
    is patched, and the reviewed `_persist_locked()` publishes atomically
    (tmp + fsync + os.replace). The in-memory ledger.data is refreshed from
    the corrected state so the gates read the truth. Uses the existing
    `_locked()`/`_persist_locked()` of the reviewed ledger class - the class
    file itself is NOT modified."""
    with ledger._locked():
        ledger.data = ledger._load_or_init()
        hit = None
        for inv in reversed(ledger.data.get("invocations", [])):
            if inv.get("invocation_id") == invocation_id:
                hit = inv
                break
        if hit is None:
            raise E2Error(f"amend: invocation {invocation_id!r} not found in ledger")
        hit["exit_code"] = int(exit_code)
        hit["note"] = str(note)
        # interleave defence (round 9): re-read the file INSIDE the lock right
        # before persisting and merge in any invocation a non-locking writer
        # landed between our read and this write; our patched verdict wins for
        # the amended invocation, everything else from the file is preserved
        disk_now = json.loads(Path(ledger_path).read_text(encoding="utf-8")) \
            if Path(ledger_path).exists() else {}
        known = {i.get("invocation_id") for i in ledger.data.get("invocations", [])}
        for inv in reversed(disk_now.get("invocations", [])):
            if inv.get("invocation_id") not in known:
                ledger.data.setdefault("invocations", []).append(inv)
        disk_pairs = int(disk_now.get("pairs_charged", 0))
        if disk_pairs > int(ledger.data.get("pairs_charged", 0)):
            ledger.data["pairs_charged"] = disk_pairs
        ledger._persist_locked()


def check_deadline(auth: dict, now: Optional[float] = None) -> None:
    if (now if now is not None else time.time()) > auth["deadline_epoch"]:
        raise E2Error(
            f"e2_deadline_utc passed ({auth['deadline_epoch']}); no further forwards allowed"
        )


def _ensure_outside_checkout(out_dir: Path, repo_root: Path) -> None:
    """Refuse to write run outputs inside the Git checkout (torch-free local copy)."""
    out_resolved = Path(out_dir).resolve()
    repo_resolved = Path(repo_root).resolve()
    if out_resolved == repo_resolved or repo_resolved in out_resolved.parents:
        raise E2Error(f"out_dir {out_resolved} is inside the Git checkout {repo_resolved}")


def verify_background_done(run_root: Path, ledger, which: str, success_note: str, n_pairs: int, cap: float) -> dict:
    """Execution-order gate for a COMPLETED background: the shared ledger must
    contain a finalized, successful invocation with the exact success note and
    the real per-invocation pairs_charged field, and the background's chunk
    file must exist with the expected scored-pair count."""
    run_root = Path(run_root)
    chunk_files = sorted((run_root / which).glob("chunk_*.jsonl")) if (run_root / which).is_dir() else []
    if not chunk_files:
        raise E2Error(f"gate {which}: no chunk found under <run-root>/{which}")
    pair_rows = []
    for cf in chunk_files:
        rows = [json.loads(l) for l in cf.read_text(encoding="utf-8").splitlines() if l.strip()]
        pair_rows.extend(r for r in rows if r.get("record_type") != "chunk_header")
    if len(pair_rows) != n_pairs or any(r.get("status") != "scored" for r in pair_rows):
        raise E2Error(f"gate {which}: chunks must contain exactly {n_pairs} scored pair(s) "
                      f"across all files, found {len(pair_rows)}")
    invs = ledger.data.get("invocations", [])
    ok = [i for i in invs if i.get("finalized") and i.get("exit_code") == 0
          and str(i.get("note", "")) == success_note
          and int(i.get("pairs_charged", 0)) >= n_pairs]
    if not ok:
        raise E2Error(
            f"gate {which}: shared ledger has no finalized successful invocation with note "
            f"{success_note!r} and pairs_charged >= {n_pairs}"
        )
    # evidence check (review round 7, item 2): the FINAL SETTLEMENT record is
    # the authoritative verdict - it must exist, declare success, and bill
    # within the cap; the environment + summary records must merely exist
    missing = []
    settle_bad = []
    for i in ok[:1]:
        env_f = run_root / f"environment_{i['invocation_id']}.json"
        sum_f = run_root / f"summary_{i['invocation_id']}.json"
        fin_f = run_root / f"final_settlement_{i['invocation_id']}.json"
        if not env_f.is_file():
            missing.append(env_f.name)
        if not sum_f.is_file():
            missing.append(sum_f.name)
        if not fin_f.is_file():
            missing.append(fin_f.name)
            continue
        try:
            fin = json.loads(fin_f.read_text(encoding="utf-8"))
            if fin.get("settle_ok") is not True or fin.get("breach") is not False:
                settle_bad.append(f"{fin_f.name}: verdict/breach")
            elif fin.get("exit_code") != 0:
                settle_bad.append(f"{fin_f.name}: exit_code")
            elif float(fin.get("ledger_card_seconds", -1)) > cap + 1e-6:
                settle_bad.append(f"{fin_f.name}: card_seconds over cap")
            elif (fin.get("wall_crossed") is True
                    or float(fin.get("ended_epoch", 0)) > float(fin.get("wall_deadline_epoch", 0)) + 1e-6):
                settle_bad.append(f"{fin_f.name}: wall deadline crossed")
        except Exception:  # noqa: BLE001
            settle_bad.append(fin_f.name)
    if missing:
        raise E2Error(
            f"gate {which}: success note present but evidence records missing: {missing}"
        )
    if settle_bad:
        raise E2Error(
            f"gate {which}: final settlement does not authorize success: {settle_bad}"
        )
    return {"invocations": [i["invocation_id"] for i in ok], "chunk": chunk_files[0].name,
            "evidence_verified": True}


def verify_smoke_done(run_root: Path, ledger, cap: float) -> dict:
    """The contractual 1-pair smoke must be completed (exact note + real
    pairs_charged field + one scored pair on disk)."""
    return verify_background_done(run_root, ledger, "smoke_b0", SMOKE_NOTE, 1, cap)


def verify_replay_gate(run_root: Path, cfg: dict, auth: dict, ledger) -> dict:
    """b0/b1 may only start after the cross-machine REPLAY tolerance gate PASSED:
    replay_gate_record.json must exist in the run root, be bound to the same
    approved SHA, pass=true, pin ALL expected replay chunks (and their G4
    source chunks), cover every replay pair, and its pinned replay chunk
    hashes must match the actual chunk files on disk."""
    run_root = Path(run_root)
    want_pairs = int(cfg["backgrounds"]["replay"]["n_unique_text"]) * int(cfg["backgrounds"]["replay"]["n_gallery"])
    # the REPLAY stage itself must have SETTLED SUCCESSFULLY in the shared
    # ledger with its evidence records present (round 6, item 3: a FAILED
    # replay - e.g. evidence_write_failed - can never feed a batch)
    gates_ev = verify_background_done(run_root, ledger, "replay", SUCCESS_NOTE["replay"], want_pairs,
                                      float(auth["max_card_seconds"]))
    rec_path = run_root / "replay_gate_record.json"
    if not rec_path.is_file():
        raise E2Error(
            "replay gate: replay_gate_record.json missing — REPLAY must be scored and "
            "compared against the share2 G4 records (tolerance |ds| <= max(1e-3, 1e-6*|s|), "
            "n_valid equality) BEFORE any batch background scoring"
        )
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec.get("schema") != "text2cad-e2-replay-gate-v1":
        raise E2Error("replay gate: record schema mismatch")
    if rec.get("approved_sha") != auth["approved_sha"]:
        raise E2Error(f"replay gate: record bound to {rec.get('approved_sha')}, not this approval")
    if rec.get("pass") is not True:
        raise E2Error(f"replay gate: recorded verdict is pass={rec.get('pass')}")
    want_names = {f"chunk_s00_t{k:05d}_{k + 1:05d}.jsonl" for k in range(int(cfg["backgrounds"]["replay"]["n_unique_text"]))}
    pins = rec.get("replay_chunk_sha256") or {}
    if set(pins) != want_names:
        raise E2Error(
            f"replay gate: record pins {len(pins)} chunk(s), expected exactly the "
            f"{len(want_names)} replay chunks (empty/partial pin sets are refused)"
        )
    if not rec.get("g4_chunk_sha256"):
        raise E2Error("replay gate: record does not pin the certified G4 source chunk hashes")
    if rec.get("g4_sha256sums_sha256") != cfg["g4_sha256sums_sha256"]:
        raise E2Error("replay gate: record was not compared against the pinned certified G4 SHA256SUMS")
    if int(rec.get("n_pairs_compared", 0)) != want_pairs:
        raise E2Error(f"replay gate: compared {rec.get('n_pairs_compared')} != {want_pairs} replay pairs")
    # recorded G4 source hashes must match the CERTIFIED pins item by item
    want_g4_names = set()
    for meta in cfg["replay_texts"]:
        orig = int(meta["original_text_index"])
        card = 0 if orig < 1822 else 1
        want_g4_names.add(f"card{card}/chunk_s{card:02d}_t{orig:05d}_{orig + 1:05d}.jsonl")
    rec_g4 = rec.get("g4_chunk_sha256") or {}
    if set(rec_g4) != want_g4_names:
        raise E2Error(
            f"replay gate: record pins {len(rec_g4)} G4 chunk hash(es), expected exactly "
            f"the {len(want_g4_names)} certified sources"
        )
    for name in want_g4_names:
        if rec_g4[name] != cfg["g4_chunk_sha256"][name]:
            raise E2Error(f"replay gate: G4 source {name} hash does not match the certified value")
    # the candidate mapping must be the frozen one, item by item
    if list(rec.get("dense_to_original_cad_index") or []) != list(cfg["replay_dense_to_original_cad_index"]):
        raise E2Error("replay gate: candidate mapping differs from the frozen dense->original index")
    replay_dir = run_root / "replay"
    for fname, pin in sorted(pins.items()):
        f = replay_dir / fname
        if not f.is_file() or sha256_file(f) != pin:
            raise E2Error(f"replay gate: chunk {fname} missing or hash drift vs gate record")
    return {"compared": rec["n_pairs_compared"], "max_abs_delta": rec["max_abs_delta"]}


def require_prerequisite_gates(run_root: Path, which: str, cfg: dict, auth: dict, ledger) -> dict:
    """Execution order enforced here: smoke_b0 first, then REPLAY + its
    tolerance-gate record, then b0 (its own success record required by b1),
    and only then b1."""
    gates: dict[str, Any] = {}
    cap = float(auth["max_card_seconds"])
    if which != "smoke_b0":
        gates["smoke"] = verify_smoke_done(run_root, ledger, cap)
    if which in ("b0", "b1"):
        gates["replay_gate"] = verify_replay_gate(run_root, cfg, auth, ledger)
    if which == "b1":
        gates["b0_done"] = verify_background_done(run_root, ledger, "b0", B0_OK_NOTE, int(cfg["backgrounds"]["b0"]["n_unique_text"]) * int(cfg["backgrounds"]["b0"]["n_gallery"]), cap)
    return gates


class DeadlineLedgerProxy:
    """Delegates to the shared ledger but enforces the wall-clock deadline
    inside the scorer's per-batch heartbeat (review round 3, item 3)."""

    def __init__(self, ledger, auth: dict):
        self._ledger = ledger
        self._auth = auth

    def heartbeat(self, invocation_id: str) -> None:
        check_deadline(self._auth)
        self._ledger.heartbeat(invocation_id)

    def __getattr__(self, name):  # delegate everything else
        return getattr(self._ledger, name)


def dry_run_e2(cfg: dict, dirs: dict[str, Path], original_manifest_dir: Path, bert_dir: Path,
               cad_seq_zip: Path, out_dir: Path, repo_root: Path) -> dict:
    """Forward-free CPU preflight: refusal guards first, then the SAME input
    validation as score, then tokenizer evidence."""
    out_dir = Path(out_dir).resolve()
    _ensure_outside_checkout(out_dir, repo_root)
    out_path = out_dir / "dry_run.json"
    if out_path.exists():
        raise E2Error(f"refusing to overwrite {out_path}")
    report = validate_e2_manifests(cfg, dirs, Path(original_manifest_dir))
    zip_sha = sha256_file(Path(cad_seq_zip))
    if zip_sha != cfg["cad_seq_zip_sha256"]:
        raise E2Error(f"cad_seq.zip sha {zip_sha} != pinned {cfg['cad_seq_zip_sha256']}")
    report["cad_seq_zip_sha256"] = zip_sha
    report["tokenizer_evidence"] = tokenizer_evidence(Path(bert_dir))
    report["status"] = "dry_run_no_forward"
    report["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def score_e2(repo_root: Path, cfg: dict, which: str, dirs: dict[str, Path],
             checkpoint: Path, bert_dir: Path, cad_seq_zip: Path,
             out_dir: Path, device: str, resume: bool = False) -> int:
    """Single-card scoring of one background manifest; reuses the reviewed G4
    machinery with the E2 shared canonical ledger (review C1 fix)."""
    # lazy import: torch-bearing module only on the score path
    sys.path.insert(0, str(Path(repo_root) / "experiments" / "src"))
    from idea01.text2cad_frozen_runner import (  # noqa: E402
        CHUNK_SCHEMA_VERSION,
        PINNED_CHECKPOINT_SHA256,
        PRODUCTION_BUDGET_ROOT,
        SCORE_NAME,
        SCORE_SPEC_VERSION,
        SharedBudgetLedger,
        FrozenMatrixScorer,
        RunnerConfig,
        atomic_publish,
        check_revision_paths,
        chunk_file_name,
        environment_record,
        load_gated_model_and_store,
        new_invocation_id,
        plan_chunks,
    )
    from idea01.text2cad_pair_index import PairSpace  # noqa: E402

    repo_root = Path(repo_root)
    known = dict(EXPECTED_TEXT) | {"replay": None}
    if which not in known:
        raise E2Error(f"--which must be one of {sorted(known)}, got {which!r}")
    if git_dirty(repo_root):
        raise E2Error(f"git worktree not clean:\n{git_dirty(repo_root)}")
    detached, branch = head_is_detached(repo_root)
    if not detached:
        raise E2Error(f"HEAD is not detached (on branch {branch!r}); plan requires a detached approved SHA")
    run_root = Path(out_dir).resolve()
    _ensure_outside_checkout(run_root, repo_root)

    # 1. input validation — the SAME functions as dry-run
    validation = validate_e2_manifests(cfg, dirs, dirs["_original"])

    # 2. authorization (fetched ref, allowlisted origin, stale-tracking-ref safe)
    auth = authorize_e2_run(repo_root, cfg)

    # 3. shared canonical ledger: OUTSIDE checkout, keyed by the approved SHA —
    #    smoke/b0/b1/replay and every retry all charge the SAME cumulative cap.
    ledger_path = Path(PRODUCTION_BUDGET_ROOT) / f"text2cad_e2_{auth['approved_sha']}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)  # the real ledger takes its lock file here
    ledger = SharedBudgetLedger(
        ledger_path, approved_sha=auth["approved_sha"],
        max_pairs=auth["max_pairs"], max_card_seconds=auth["max_card_seconds"],
    )
    pending = [i["invocation_id"] for i in ledger.data.get("invocations", []) if not i.get("finalized")]
    if pending:
        raise E2Error(
            f"shared E2 ledger {ledger_path} has pending invocation(s) {pending}: previous run "
            "crashed or was killed; stop and report — no automatic resume with uncertain cost"
        )

    # 4. prerequisite execution-order gates (smoke -> replay-gate -> batches)
    gates = require_prerequisite_gates(run_root, which, cfg, auth, ledger)

    manifest_dir = Path(dirs[which])
    pins = cfg["backgrounds"][which]
    n_t, n_c = manifest_counts(cfg, which)
    pair_space = PairSpace(n_texts=n_t, n_cads=n_c)
    if which == "replay":
        texts = load_texts(manifest_dir)
    else:
        texts = [load_background_texts(manifest_dir, EXPECTED_TEXT[which])]
    members = load_cad_store(manifest_dir)

    cfg_run = RunnerConfig(
        checkpoint_path=str(checkpoint), bert_dir=str(bert_dir), cad_seq_zip=str(cad_seq_zip),
        pair_manifest_dir=str(manifest_dir), device=device, torch_dtype="float32",
        batch_size=BATCH_SIZE,
        expected_keys_manifest=str(repo_root / "experiments" / "configs" / "text2cad_checkpoint_expected_keys_20260930.json"),
        trim_mode="none", text_cache=False, shard_id=0, n_shards=1, texts_per_chunk=1,
    )
    check_revision_paths(cfg_run)

    todo, completed = plan_chunks(run_root / which, pair_space, cfg_run, {}, resume=resume)
    planned = sum((t1 - t0) * n_c for t0, t1, _p in todo)
    if planned > ledger.remaining_pairs():
        raise E2Error(f"planned {planned} pairs exceed remaining shared E2 budget {ledger.remaining_pairs()}")
    check_deadline(auth, now=NOW_FN())

    binding = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "pair_manifest_sha256": pins["pair_manifest_sha256"],
        "git_sha": auth["approved_sha"],
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "device": device,
        "torch_dtype": "float32",
        "trim_mode": "none",
        "text_cache": False,
        "score_name": SCORE_NAME,
        "spec_version": SCORE_SPEC_VERSION,
    }
    invocation_id = new_invocation_id()
    guarded_ledger = DeadlineLedgerProxy(ledger, auth)
    # 1. register FIRST so the supervisor's first ledger snapshot sees the live
    #    invocation and computes the absolute deadline from the REAL start
    stale_problems = ledger.begin_invocation(invocation_id, shard_id=0, cards_used=1)
    if stale_problems:
        ledger.finalize_invocation(invocation_id, exit_code=2, note="ledger_refused_start")
        raise E2Error(f"shared ledger refused start: {stale_problems}")
    started_epoch = next(i["started_epoch"] for i in ledger.data["invocations"]
                         if i["invocation_id"] == invocation_id)

    # 2. THEN start the independent supervision process (G4-style: a hung
    #    load/forward is killed BY the deadline; a hard kill leaves the ledger
    #    pending - recovery is a reviewer decision)
    ready_event = None
    supervisor = None
    if os.name == "posix":
        import multiprocessing as _mp
        ready_event = _mp.Event()
    supervisor = start_deadline_supervision(auth, ledger_path, ready_event=ready_event)
    # 3. confirm the supervisor is ready (snapshot taken / absolute deadline
    #    computed) BEFORE the model load
    if supervisor is not None and ready_event is not None:
        if not ready_event.wait(timeout=30.0):
            raise E2Error("deadline supervisor did not become ready within 30s")

    # 4. post-readiness re-check of BOTH caps BEFORE the model load (round 9,
    #    item 2): the schedule is now fixed, so this instant is authoritative
    now_r = NOW_FN()
    projected_r = float(ledger.data.get("card_seconds_spent", 0.0)) + max(
        0.0, now_r - float(started_epoch)) * 1
    if projected_r > auth["max_card_seconds"] - 0.05 or now_r > auth["deadline_epoch"]:
        fail_note = ("card_seconds_exceeded" if projected_r > auth["max_card_seconds"] - 0.05
                      else "deadline_exceeded")
        failure = (f"post-readiness re-check: projected card-seconds "
                    f"{projected_r:.2f}/{auth['max_card_seconds']}, "
                    f"wall now {now_r:.3f} vs deadline {auth['deadline_epoch']:.3f}")
        ledger.finalize_invocation(invocation_id, exit_code=2, note=fail_note)
        final_settlement = {
            "schema": "text2cad-e2-final-settlement-v1",
            "invocation_id": invocation_id, "e2_background": which,
            "approved_sha": auth["approved_sha"], "exit_code": 2,
            "note": fail_note, "settle_ok": False, "breach": True,
            "card_crossed": projected_r > auth["max_card_seconds"] - 0.05,
            "wall_crossed": now_r > auth["deadline_epoch"],
            "ledger_card_seconds": float(ledger.data.get("card_seconds_spent", 0.0)),
            "max_card_seconds": auth["max_card_seconds"],
            "wall_deadline_epoch": auth["deadline_epoch"],
            "ended_epoch": float(ledger.data["invocations"][-1].get("ended_epoch", 0.0)),
            "failure": failure, "phase": "pre_load",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_publish(run_root / f"final_settlement_{invocation_id}.json",
                       [json.dumps(final_settlement, indent=2, sort_keys=True, ensure_ascii=False)])
        print(f"E2 FATAL: {failure}", file=sys.stderr)
        return 2

    exit_code, note, failure, published = 0, "ok", None, 0
    load_report = preload_report = None
    scorer_stats: dict = {}
    spent: Any = None
    try:
        model, load_report, store, preload_report = load_gated_model_and_store(cfg_run, members)
        check_deadline(auth, now=NOW_FN())  # a load straddling the deadline refuses BEFORE scoring
        scorer = FrozenMatrixScorer(cfg_run, model, texts, store, pair_space, guarded_ledger, invocation_id, binding)
        which_dir = run_root / which
        which_dir.mkdir(parents=True, exist_ok=True)
        for t0, t1, _p in todo:
            check_deadline(auth, now=NOW_FN())
            chunk_records = scorer.score_chunk(t0, t1)
            atomic_publish(which_dir / chunk_file_name(0, t0, t1),
                           [json.dumps(r, ensure_ascii=False) for r in chunk_records])
            published += 1
        scorer_stats = dict(scorer.stats)
    except Exception as exc:  # noqa: BLE001
        exit_code, note, failure = 2, "aborted", f"{type(exc).__name__}: {exc}"

    # ---- ONE settle decision, then evidence, then the ledger (round 5 #2) ---
    now = NOW_FN()
    projected_card_seconds = float(ledger.data.get("card_seconds_spent", 0.0)) + max(
        0.0, now - float(started_epoch))
    spent = projected_card_seconds
    if exit_code == 0 and projected_card_seconds > auth["max_card_seconds"] - 0.05:
        exit_code, note, failure = 2, "card_seconds_exceeded", (
            f"projected cumulative card-seconds {projected_card_seconds:.2f} reach cap "
            f"{auth['max_card_seconds']}; settling as a failure")
    if exit_code == 0 and now > auth["deadline_epoch"]:
        exit_code, note, failure = 2, "deadline_exceeded", "wall-clock deadline passed before settle"

    if exit_code == 0:
        # evidence MUST be written and intact BEFORE the ledger records success;
        # an evidence-write failure settles this invocation as a failure
        try:
            env_record = environment_record(cfg_run, extra={
                "e2_background": which,
                "authorization_ref": auth["authorization_ref"],
                "authorization_meta": auth["authorization_meta"],
                "approved_sha": auth["approved_sha"],
                "validation_report": validation,
                "prerequisite_gates": gates,
                "cad_seq_zip_sha256": sha256_file(Path(cad_seq_zip)),
                "planned_pairs": planned,
                "completed_before": len(completed),
                "published_chunks": published,
                "resume": resume,
                "settlement": {"card_seconds_spent": spent,
                                "max_card_seconds": auth["max_card_seconds"], "failure": None},
                "command_flags": {"device": device, "batch_size": BATCH_SIZE, "which": which},
            })
            atomic_publish(run_root / f"environment_{invocation_id}.json",
                           [json.dumps(env_record, indent=2, sort_keys=True, ensure_ascii=False)])
            summary = {
                "invocation_id": invocation_id, "e2_background": which,
                "approved_sha": auth["approved_sha"], "device": device,
                "published_chunks": published, "planned_pairs": planned,
                "completed_before": len(completed), "scorer_stats": scorer_stats,
                "load_report": load_report, "preload_report": preload_report,
                "ledger": ledger.to_dict(), "failure": None, "settle_ok": True,
            }
            atomic_publish(run_root / f"summary_{invocation_id}.json",
                           [json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)])
        except Exception as exc:  # noqa: BLE001
            exit_code, note, failure = 2, "evidence_write_failed", (
                f"{type(exc).__name__}: {exc}")

    # ---- settle #2 at the SAME clock, AFTER the evidence writes (round 6 #2):
    # time spent writing evidence is honestly accounted; on failure the success
    # note is never written and the authoritative final settlement record
    # (written after finalize) carries settle_ok=false
    now2 = NOW_FN()
    if exit_code == 0:
        projected2 = float(ledger.data.get("card_seconds_spent", 0.0)) + max(
            0.0, now2 - float(started_epoch))
        spent = projected2
        if projected2 > auth["max_card_seconds"] - 0.05:
            exit_code, note, failure = 2, "card_seconds_exceeded", (
                f"projected cumulative card-seconds {projected2:.2f} reach cap "
                f"{auth['max_card_seconds']} after the evidence writes; settling as a failure")
        elif now2 > auth["deadline_epoch"]:
            exit_code, note, failure = 2, "deadline_exceeded", (
                "wall-clock deadline passed during the evidence writes")

    if exit_code == 0:
        note = SUCCESS_NOTE[which]
    # finalize LAST, with the verdict from settle #2; the ledger then bills
    # with its OWN clock - the AUTHORITATIVE settlement reads that billing AND
    # the ledger's own ended_epoch against BOTH caps (review round 8, item 2:
    # one settlement judges card-seconds AND wall clock)
    ledger.finalize_invocation(invocation_id, exit_code=exit_code, note=note)

    inv_final = next(i for i in ledger.data["invocations"]
                     if i["invocation_id"] == invocation_id)
    authoritative = float(ledger.data.get("card_seconds_spent", 0.0))
    ended_epoch = float(inv_final.get("ended_epoch", 0.0))
    card_crossed = authoritative > auth["max_card_seconds"] + 1e-6 or note == "card_seconds_exceeded"
    wall_crossed = ended_epoch > float(auth["deadline_epoch"]) + 1e-6 or note == "deadline_exceeded"
    breach = exit_code == 0 and (card_crossed or wall_crossed)
    if breach:
        note_name = "card_seconds_exceeded" if card_crossed else "deadline_exceeded"
        failure = (f"authoritative settlement crossed the "
                    f"{'card-second' if card_crossed else 'wall-clock'} cap "
                    f"(card={authoritative:.2f}/{auth['max_card_seconds']}, "
                    f"ended={ended_epoch:.3f} vs deadline {auth['deadline_epoch']:.3f}); "
                    f"verdict corrected")
        amend_ledger_verdict_on_disk(ledger_path, ledger, invocation_id, 2, note_name)
        exit_code, note = 2, note_name

    # independent FINAL SETTLEMENT record: authoritative post-finalize billing
    # and ended_epoch plus the verdict the process actually exits with;
    # downstream stages consume THIS record (the provisional success summary is
    # irrelevant to the gates). If this record cannot be published, the ledger
    # verdict is corrected to a failure and a distinct-name retry is attempted.
    final_settlement = {
        "schema": "text2cad-e2-final-settlement-v1",
        "invocation_id": invocation_id,
        "e2_background": which,
        "approved_sha": auth["approved_sha"],
        "exit_code": exit_code,
        "note": note,
        "settle_ok": exit_code == 0,
        "breach": breach,
        "card_crossed": card_crossed,
        "wall_crossed": wall_crossed,
        "ledger_card_seconds": authoritative,
        "max_card_seconds": auth["max_card_seconds"],
        "wall_deadline_epoch": auth["deadline_epoch"],
        "ended_epoch": ended_epoch,
        "failure": failure,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        atomic_publish(run_root / f"final_settlement_{invocation_id}.json",
                       [json.dumps(final_settlement, indent=2, sort_keys=True, ensure_ascii=False)])
    except Exception as exc:  # noqa: BLE001
        # the ledger verdict MUST reflect that the settlement record is missing
        amend_ledger_verdict_on_disk(ledger_path, ledger, invocation_id, 2, "final_settlement_write_failed")
        exit_code, note = 2, "final_settlement_write_failed"
        final_settlement.update({"exit_code": 2, "note": note, "settle_ok": False,
                                  "failure": f"{type(exc).__name__}: {exc}"})
        try:
            atomic_publish(run_root / f"final_settlement_retry_{invocation_id}.json",
                           [json.dumps(final_settlement, indent=2, sort_keys=True, ensure_ascii=False)])
        except Exception:  # noqa: BLE001 - the corrected LEDGER is the evidence of record
            pass

    if exit_code != 0:
        print(f"E2 FATAL: {failure}", file=sys.stderr)
        return exit_code
    print(json.dumps({"status": "ok", "invocation_id": invocation_id, "background": which,
                      "pairs": planned, "ledger_pairs_total": ledger.pairs_charged()}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["dry-run", "score"], default="dry-run")
    ap.add_argument("--checkpoint")
    ap.add_argument("--bert-dir", required=True)
    ap.add_argument("--cad-seq-zip")
    ap.add_argument("--b0-manifest-dir", required=True)
    ap.add_argument("--b1-manifest-dir", required=True)
    ap.add_argument("--smoke-manifest-dir", required=True)
    ap.add_argument("--replay-manifest-dir", required=True)
    ap.add_argument("--original-manifest-dir", required=True,
                    help="original pair_manifest_v4 dir (cad_store equality reference)")
    ap.add_argument("--out-dir", required=True, help="run root; chunks under <out-dir>/<which>/")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--which", choices=["b0", "b1", "smoke_b0", "replay"], default=None,
                    help="score mode: which background manifest to score")
    ap.add_argument("--resume", action="store_true",
                    help="skip verified chunks; the shared ledger must have no pending invocation")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)
    try:
        repo_root = Path(args.repo_root).resolve()
        cfg = load_e2_config(repo_root)
        dirs = {"b0": Path(args.b0_manifest_dir), "b1": Path(args.b1_manifest_dir),
                "smoke_b0": Path(args.smoke_manifest_dir), "replay": Path(args.replay_manifest_dir),
                "_original": Path(args.original_manifest_dir)}
        if args.mode == "dry-run":
            if not args.cad_seq_zip:
                raise E2Error("--cad-seq-zip required for dry-run")
            report = dry_run_e2(cfg, dirs, Path(args.original_manifest_dir), Path(args.bert_dir),
                                Path(args.cad_seq_zip), Path(args.out_dir), repo_root)
            print(json.dumps({"status": report["status"], "tokenizer": report["tokenizer_evidence"]}))
            return 0
        for req in (args.checkpoint, args.cad_seq_zip, args.which):
            if not req:
                raise E2Error("score mode requires --checkpoint, --cad-seq-zip and --which")
        return score_e2(repo_root, cfg, args.which, dirs, Path(args.checkpoint), Path(args.bert_dir),
                        Path(args.cad_seq_zip), Path(args.out_dir), args.device, resume=bool(args.resume))
    except Exception as exc:  # noqa: BLE001 — every refusal exits non-zero with the reason
        print(f"E2 FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


def compare_replay_to_g4(replay_run_dir: Path, g4_raw_dir: Path, cfg: dict,
                         approved_sha: str, out_path: Path) -> dict:
    """Cross-machine tolerance gate (review round 4, item 2/3): compare every
    REPLAY pair scored on share4 against the certified share2 G4 chunk records.

    Mapping: replay text k corresponds to original unique-text index
    cfg['replay_texts'][k]['original_text_index']; the G4 chunk lives at
    <g4_raw_dir>/card{0 if orig<1822 else 1}/chunk_s{card:02d}_t{orig:05d}_{orig+1:05d}.jsonl.
    Replay candidates are a dense-reindexed SELECTED set of original G4
    candidates (full 16-batches plus the true G4 tail batch 8,032-8,045);
    cfg['replay_dense_to_original_cad_index'] maps each dense chunk cad_index
    to the original G4 cad_index and is pinned in the gate record.

    Per pair: |dscore_total| <= max(TOLERANCE_ABS, TOLERANCE_REL*|g4|) AND
    n_valid_total equality AND status equality. Completeness is enforced:
    replay rows per text must be exactly the n_cads dense indices, unique and
    complete (duplicates/missing are violations), and each text's G4
    counterpart must be unique/complete too. The record pins the sha256 of ALL
    replay chunks AND their G4 source chunks (bound to the certified files).
    torch-free; run where the G4 matrix exists.
    """
    replay_dir = Path(replay_run_dir)
    g4_raw = Path(g4_raw_dir)
    # certified-source enforcement (review round 5, item 3): the G4 SHA256SUMS
    # file itself must match the pinned hash, and EVERY compared chunk must be
    # listed there with a matching sha - otherwise refuse before comparing
    sums_path = g4_raw / "SHA256SUMS"
    sums_sha = sha256_file(sums_path)
    if sums_sha != cfg["g4_sha256sums_sha256"]:
        raise E2Error(
            f"G4 SHA256SUMS sha {sums_sha} != pinned certified {cfg['g4_sha256sums_sha256']}"
        )
    certified = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            a, b = parts
            rel = b.lstrip("./") if len(a) == 64 else a.lstrip("./")
            h = a if len(a) == 64 else b
            if len(h) == 64:
                certified[rel] = h
    rows_out: list[dict] = []
    max_abs = 0.0
    n_compared = 0
    n_valid_mismatch = 0
    n_violations = 0
    chunk_pins: dict[str, str] = {}
    g4_chunk_pins: dict[str, str] = {}
    n_cads = int(cfg["backgrounds"]["replay"]["n_gallery"])
    dense_to_original = [int(x) for x in (cfg.get("replay_dense_to_original_cad_index") or range(n_cads))]
    if len(dense_to_original) != n_cads:
        raise E2Error(f"replay mapping has {len(dense_to_original)} entries, expected {n_cads}")
    expected_chunk_names = [f"chunk_s00_t{k:05d}_{k + 1:05d}.jsonl" for k in range(len(cfg["replay_texts"]))]
    for k, meta in enumerate(cfg["replay_texts"]):
        orig = int(meta["original_text_index"])
        card = 0 if orig < 1822 else 1
        g4_name = f"chunk_s{card:02d}_t{orig:05d}_{orig + 1:05d}.jsonl"
        g4_path = g4_raw / f"card{card}" / g4_name
        g4_file_sha = sha256_file(g4_path)
        if certified.get(f"card{card}/{g4_name}") != g4_file_sha:
            raise E2Error(
                f"G4 source chunk {g4_name} sha {g4_file_sha} is NOT in the certified "
                f"SHA256SUMS (or drifts from it) - refusing the comparison"
            )
        replay_name = expected_chunk_names[k]
        replay_path = replay_dir / replay_name
        chunk_pins[replay_name] = sha256_file(replay_path)
        # card-prefixed key so it matches cfg["g4_chunk_sha256"] (round 6)
        g4_chunk_pins[f"card{card}/{g4_name}"] = g4_file_sha
        g4_by_cad: dict[int, dict] = {}
        with open(g4_path, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                r = json.loads(line)
                cad = int(r["cad_index"])
                if cad in dense_to_original:
                    if cad in g4_by_cad:
                        raise E2Error(f"G4 chunk {g4_name}: duplicate candidate {cad} (data admission error)")
                    g4_by_cad[cad] = r
        if len(g4_by_cad) != n_cads:
            raise E2Error(f"G4 chunk {g4_name}: {len(g4_by_cad)} replay-range candidates, expected {n_cads}")
        new_rows = [json.loads(l) for l in replay_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        header_ok = new_rows and new_rows[0].get("record_type") == "chunk_header"
        body = new_rows[1:] if header_ok else new_rows
        seen: set[int] = set()
        for r in body:
            dense = int(r["cad_index"])
            if dense in seen:
                n_violations += 1
                rows_out.append({"text_index": k, "dense_cad_index": dense, "error": "duplicate_candidate"})
                continue
            seen.add(dense)
            cad = dense_to_original[dense]
            g = g4_by_cad.get(cad)
            if g is None:
                n_violations += 1
                rows_out.append({"text_index": k, "dense_cad_index": dense, "cad_index": cad,
                                  "error": "no_g4_counterpart"})
                continue
            ds = abs(float(r["score_total"]) - float(g["score_total"]))
            tol = max(REPLAY_TOLERANCE_ABS, REPLAY_TOLERANCE_REL * abs(float(g["score_total"])))
            max_abs = max(max_abs, ds)
            if int(r["n_valid_total"]) != int(g["n_valid_total"]):
                n_valid_mismatch += 1
            ok = (ds <= tol
                  and int(r["n_valid_total"]) == int(g["n_valid_total"])
                  and r.get("status") == g.get("status") == "scored")
            if not ok:
                n_violations += 1
                rows_out.append({"text_index": k, "dense_cad_index": dense, "cad_index": cad,
                                  "delta": ds, "tolerance": tol, "error": "tolerance_or_valid_mismatch"})
            n_compared += 1
        missing = set(range(n_cads)) - seen
        if missing:
            n_violations += len(missing)
            rows_out.append({"text_index": k, "error": "missing_candidates", "n_missing": len(missing)})
    passed = (n_violations == 0
              and n_compared == len(cfg["replay_texts"]) * n_cads
              and set(chunk_pins) == set(expected_chunk_names)
              and bool(chunk_pins) and bool(g4_chunk_pins))
    record = {
        "schema": "text2cad-e2-replay-gate-v1",
        "approved_sha": approved_sha,
        "pass": passed,
        "n_pairs_expected": len(cfg["replay_texts"]) * n_cads,
        "n_pairs_compared": n_compared,
        "max_abs_delta": max_abs,
        "tolerance_abs": REPLAY_TOLERANCE_ABS,
        "tolerance_rel": REPLAY_TOLERANCE_REL,
        "n_valid_mismatch": n_valid_mismatch,
        "n_violations": n_violations,
        "replay_chunk_sha256": chunk_pins,
        "g4_chunk_sha256": g4_chunk_pins,
        "g4_sha256sums_sha256": sums_sha,
        "dense_to_original_cad_index": dense_to_original,
        "violations": rows_out[:50],
        "g4_source_dir": str(g4_raw),
        "comparator": "idea01.text2cad_e2_runner.compare_replay_to_g4",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record
