#!/usr/bin/env python
"""P1 scoring matrix runner (gate 2 ONLY — requires the Codex-controlled
origin/master ACTIVE_TASK to be pilot_authorized with approved_sha == HEAD and
approved_selection_sha256 == manifest sha256, re-fetched at every invocation).

Codex review 2026-09-25 fixes:
  R1  Resume/cumulative budget: before weights load, existing pairs.jsonl
      records are verified (unique ids, ok/finite, identical selection/git/
      model/data/dtype/env binding). --resume continues the SAME run loudly
      skipping verified-completed pairs; without it, existing records refuse
      the run. budget_ledger.json accumulates forwards and wallclock across
      invocations; the whole-run caps (416 forwards / 45 min / 1 card-hour,
      config may only tighten) bind across ALL invocations. An unfinalized
      ledger entry (crash) refuses the next invocation. Staged plan:
      --pair-ids "0"  ->  --pair-ids "0-9" --resume  ->  --pair-ids "all" --resume
      yielding exactly 414 unique raw records.
  R3  Pre-flight BEFORE weights: authorization is read from
      origin/master:experiments/ACTIVE_TASK.md after `git fetch` (NEVER from
      the detached HEAD's working tree — a commit cannot contain its own
      approved_sha); the origin remote URL is validated; model+data
      SHA256SUMS verified in Python. --model-dir/--data-dir/--max-wallclock-s
      overrides REMOVED.
  R4  out_dir must be outside the Git checkout; environment files are unique
      (timestamp+uuid); ANY exception after startup leaves a failure record
      and exits non-zero.

Isolation contract: the scoring path receives ONLY the query text and the
candidate program text. UIDs / row indices / sources are bookkeeping only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.envspec import env_spec_hashes  # noqa: E402
from idea01.prompts import build_prefix  # noqa: E402
from idea01.p1_runner import (  # noqa: E402
    EXIT_ANOMALY,
    EXIT_BUDGET,
    EXIT_OK,
    LEDGER_FILE,
    MAX_PAIRS,
    STATUS_BOUNDARY,
    STATUS_ERROR,
    STATUS_NONFINITE,
    STATUS_OK,
    STATUS_OVERLONG,
    BudgetLedger,
    check_authorization,
    effective_budgets,
    ensure_outside_checkout,
    fetch_authorized_active_task,
    new_invocation_id,
    parse_pair_ids,
    plan_invocation,
    should_stop,
    verify_existing_records,
    verify_sha256sums,
)
from idea01.scoring import ScoringError, build_scored_sequence, gather_response_logprobs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SPEC_REL = "experiments/configs/idea01_yunnan_share2_env.json"
ENV_SPEC_PATH = REPO_ROOT / ENV_SPEC_REL


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def git_dirty() -> str:
    try:
        return subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_hashes() -> dict:
    if ENV_SPEC_PATH.exists():
        return env_spec_hashes(ENV_SPEC_PATH, REPO_ROOT, ENV_SPEC_REL)
    return {
        "canonical_json_sha256": None,
        "canonical_json_algorithm": "sha256(utf8(json.dumps(obj, sort_keys=True, separators=(',',':'))))",
        "file_bytes_sha256": None,
        "file_bytes_algorithm": "sha256(git blob bytes, LF endings)",
    }


def write_environment(out_dir: Path, cfg: dict, argv: list[str], device: str, invocation_id: str, pair_ids: str, selection_sha256: str, authorization: dict | None = None) -> Path:
    env = {
        "invocation_id": invocation_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "device": device,
        "dtype": cfg["dtype"],
        "git_sha": git_sha(),
        "env_spec_file": str(ENV_SPEC_PATH),
        "env_spec": spec_hashes(),
        "model_revision": cfg["model_revision"],
        "model_dir": cfg["model_dir"],
        "data_revision": cfg["data_revision"],
        "data_dir": cfg["data_dir"],
        "selection_sha256": selection_sha256,
        "authorization": authorization or {},
        "requested_pair_ids": pair_ids,
        "command": " ".join(argv),
        "env_var_names": sorted(
            k for k in ("PYTHONDONTWRITEBYTECODE", "HF_HOME", "XDG_CACHE_HOME", "TMPDIR", "OMP_NUM_THREADS") if k in os.environ
        ),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        import transformers

        env["transformers"] = transformers.__version__
        try:
            import torch_npu

            env["torch_npu"] = torch_npu.__version__
        except Exception as e:  # noqa: BLE001
            env["torch_npu"] = f"unavailable: {e}"
        env["npu_available"] = bool(torch.npu.is_available()) if hasattr(torch, "npu") else False
    except Exception as e:  # noqa: BLE001
        env["torch"] = f"unavailable: {e}"
    try:
        env["npu_smi"] = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        env["npu_smi"] = f"unavailable: {e}"
    path = out_dir / f"environment_{invocation_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--selection", required=True, help="p1_selection.json (sha256 must equal ACTIVE_TASK approved_selection_sha256)")
    ap.add_argument("--out-dir", required=True, help="MUST be outside the Git checkout")
    ap.add_argument("--pair-ids", default="all", help="e.g. '0', '0-9', 'all'; deduped, hard cap 416")
    ap.add_argument("--resume", action="store_true", help="continue the SAME run: verify existing records, loudly skip completed pairs")
    ap.add_argument("--device", default=None, help="override config device, e.g. npu:0 (budgets are NOT overridable)")
    args = ap.parse_args()

    t_start = time.time()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    budgets = effective_budgets(cfg.get("budget", {}))
    device = args.device or cfg["device"]

    # R4: output location guard comes first — nothing may be written inside the checkout
    out_dir = Path(args.out_dir)
    try:
        ensure_outside_checkout(out_dir, REPO_ROOT)
    except ValueError as e:
        print(f"[p1-matrix] {e}", flush=True)
        return EXIT_ANOMALY
    (out_dir / "tokens").mkdir(parents=True, exist_ok=True)

    invocation_id = new_invocation_id()

    def fail_early(problems: list[str], kind: str) -> int:
        rec = {
            "invocation_id": invocation_id,
            "kind": kind,
            "problems": problems,
            "git_sha": git_sha(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(out_dir / f"{kind}_{invocation_id}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        for p in problems:
            print(f"[p1-matrix] {kind}: {p}", flush=True)
        print(f"[p1-matrix] refusing to run; NO weights were loaded ({kind} evidence written)", flush=True)
        return EXIT_ANOMALY

    dirty = git_dirty()
    if dirty:
        return fail_early([f"git worktree not clean:\n{dirty}"], "preflight_failure")

    selection_path = Path(args.selection)
    sel_sha = file_sha256(selection_path)
    # R3 (review addendum): authorization comes from the fetched origin/master,
    # never from the detached HEAD's working tree; validated + audited here.
    at_text, at_meta = fetch_authorized_active_task(REPO_ROOT)
    write_environment(out_dir, cfg, sys.argv, device, invocation_id, args.pair_ids, sel_sha, at_meta)

    try:
        return _run(args, cfg, budgets, device, out_dir, invocation_id, sel_sha, selection_path, t_start, fail_early, at_text, at_meta)
    except Exception as e:  # noqa: BLE001  (R4: any failure leaves evidence + non-zero)
        rec = {
            "invocation_id": invocation_id,
            "kind": "run_failure",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(out_dir / f"run_failure_{invocation_id}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"[p1-matrix] run FAILURE (evidence written): {type(e).__name__}: {e}", flush=True)
        return EXIT_ANOMALY


def _run(args, cfg, budgets, device, out_dir, invocation_id, sel_sha, selection_path, t_start, fail_early, at_text, at_meta) -> int:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pairs = selection["pairs"]
    problems: list[str] = []
    if selection.get("gate_summary", {}).get("n_gate_fail", 1) != 0:
        problems.append("selection manifest reports CPU gate failures")
    if len(pairs) > MAX_PAIRS:
        problems.append(f"pair plan has {len(pairs)} pairs, exceeds hard cap {MAX_PAIRS}")

    # R3: authorization gate (fetched origin/master ACTIVE_TASK must name this
    # exact HEAD + manifest SHA); fetch/remote problems are also fatal
    problems += at_meta.get("problems", [])
    if at_text is not None:
        problems += check_authorization(at_text, git_sha(), sel_sha, cfg.get("task_id", ""))

    # R3: model/data runtime copies must pass their SHA256SUMS before weights load
    for label, d in (("model", Path(cfg["model_dir"])), ("data", Path(cfg["data_dir"]))):
        n_ok, fails = verify_sha256sums(d)
        print(f"[p1-matrix] sha256sums[{label}]: {n_ok} files OK, {len(fails)} failures in {d}", flush=True)
        problems += [f"{label}: {f}" for f in fails]

    # R1: existing records must be unique/ok/identically bound; resume is explicit
    pairs_path = out_dir / "pairs.jsonl"
    existing: list[dict] = []
    if pairs_path.exists():
        existing = [json.loads(ln) for ln in pairs_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    hashes = spec_hashes()
    expected_binding = {
        "selection_sha256": sel_sha,
        "git_sha": git_sha(),
        "model_dir": cfg["model_dir"],
        "data_dir": cfg["data_dir"],
        "model_revision": cfg["model_revision"],
        "data_revision": cfg["data_revision"],
        "dtype": cfg["dtype"],
        "env_spec_canonical_json_sha256": hashes["canonical_json_sha256"],
        "env_spec_file_bytes_sha256": hashes["file_bytes_sha256"],
    }
    problems += verify_existing_records(existing, expected_binding)

    try:
        pair_ids = parse_pair_ids(args.pair_ids, len(pairs), MAX_PAIRS)
        to_score, skipped = plan_invocation(existing, pair_ids, pairs, args.resume)
    except ValueError as e:
        problems.append(str(e))
        to_score, skipped = [], []

    try:
        ledger = BudgetLedger.load(out_dir / LEDGER_FILE, sel_sha, budgets)
        problems += ledger.check_can_start(len(to_score))
    except ValueError as e:
        problems.append(str(e))
        ledger = None

    if problems:
        return fail_early(problems, "preflight_failure")
    assert ledger is not None

    for pid in skipped:
        print(f"[p1-matrix] resume: skipping verified-completed pair {pid}", flush=True)
    print(
        f"[p1-matrix] preflight OK: {len(to_score)} new forwards planned "
        f"({len(skipped)} completed skipped); cumulative budgets {budgets}",
        flush=True,
    )

    ledger.begin(invocation_id)
    n_done = 0
    n_forwards = 0
    exit_code = EXIT_OK
    note = ""
    try:
        val_rows: list[dict] = []
        with open(Path(cfg["data_dir"]) / "data" / "data_val.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                val_rows.append(json.loads(line))

        import torch
        import torch_npu  # noqa: F401  (registers NPU backend)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["model_dir"], use_fast=False)
        eos_id = tok.eos_token_id
        print(f"[p1-matrix] tokenizer eos_id={eos_id} class={type(tok).__name__}", flush=True)

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], torch_dtype=dtype)
        model.eval()
        model.to(device)
        print(f"[p1-matrix] model loaded in {time.time()-t0:.1f}s dtype={cfg['dtype']} device={device}", flush=True)
        if hasattr(torch, "npu"):
            torch.npu.reset_peak_memory_stats()

        def encode(text: str) -> list[int]:
            return tok(text, add_special_tokens=False)["input_ids"]

        base_rec = {
            "invocation_id": invocation_id,
            "model_revision": cfg["model_revision"],
            "model_dir": cfg["model_dir"],
            "data_revision": cfg["data_revision"],
            "data_dir": cfg["data_dir"],
            "git_sha": git_sha(),
            "device": device,
            "dtype": cfg["dtype"],
            "env_spec_canonical_json_sha256": hashes["canonical_json_sha256"],
            "env_spec_file_bytes_sha256": hashes["file_bytes_sha256"],
            "selection_sha256": sel_sha,
        }

        def emit(pf, rec: dict) -> None:
            pf.write(json.dumps(rec) + "\n")
            pf.flush()

        with open(pairs_path, "a", encoding="utf-8") as pf:
            for pid in to_score:
                violated = ledger.budget_exceeded(time.time() - t_start)
                if violated:
                    note = f"budget cap {violated} reached after {n_done} new pairs (cumulative accounting)"
                    print(f"[p1-matrix] {note}; stopping", flush=True)
                    exit_code = EXIT_BUDGET
                    break
                p = pairs[pid]
                query = "" if p["kind"] == "empty" else val_rows[p["query_row_index"]]["input"]
                program = val_rows[p["candidate_row_index"]]["output"]

                rec = {
                    **base_rec,
                    "pair_id": p["pair_id"],
                    "pair_index": pid,
                    "kind": p["kind"],
                    "candidate_source": p["candidate_source"],
                    "offset": p["offset"],
                    "query_uid": p["query_uid"],
                    "candidate_uid": p["candidate_uid"],
                    "candidate_row_index": p["candidate_row_index"],
                    "query_row_index": p["query_row_index"],
                }
                t_pair = time.time()
                try:
                    prefix_ids = encode(build_prefix(query))
                    full_ids = encode(build_prefix(query) + program)
                    input_ids, target_start = build_scored_sequence(prefix_ids, full_ids, eos_id)
                    rec["input_ids"] = input_ids  # full model input token ids (contract)
                    rec["n_prefix_tokens"] = len(prefix_ids)
                    rec["n_input_tokens"] = len(input_ids)
                    rec["n_target_tokens"] = len(input_ids) - target_start
                    if len(input_ids) > cfg["max_seq_len"]:
                        # CPU gate pre-verified <= max_seq_len: oversize here means drift -> stop (plan sec.4)
                        rec["status"] = STATUS_OVERLONG
                        rec["failure_reason"] = f"input length {len(input_ids)} > max_seq_len {cfg['max_seq_len']} (CPU gate passed: env drift?)"
                        emit(pf, rec)
                        print(f"[p1-matrix] {p['pair_id']}: overlong_sequence len={len(input_ids)}; stopping", flush=True)
                        exit_code = EXIT_ANOMALY
                        note = "overlong planned pair (environment drift?)"
                        break

                    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = model(input_ids=ids_t).logits
                    n_forwards += 1
                    logits_np = logits[0].float().cpu().numpy()
                    del logits, ids_t
                    res = gather_response_logprobs(logits_np, input_ids, target_start)

                    rec["sum_logprob"] = res.sum_logprob
                    rec["mean_logprob"] = res.mean_logprob
                    rec["eos_logprob"] = res.tokens[-1].logprob if res.tokens else None
                    rec["all_finite"] = res.all_finite
                    rec["status"] = STATUS_OK if res.all_finite else STATUS_NONFINITE
                    rec["elapsed_s"] = round(time.time() - t_pair, 3)
                    if hasattr(torch, "npu"):
                        rec["peak_npu_alloc_bytes"] = int(torch.npu.max_memory_allocated())
                        torch.npu.reset_peak_memory_stats()

                    tok_path = out_dir / "tokens" / f"{p['pair_id']}.jsonl"
                    with open(tok_path, "w", encoding="utf-8") as tf:
                        for ts in res.tokens:
                            tf.write(
                                json.dumps(
                                    {
                                        "position": ts.position,
                                        "pred_row": ts.pred_row,
                                        "token_id": ts.token_id,
                                        "token_text": tok.decode([ts.token_id]),
                                        "logprob": ts.logprob,
                                    }
                                )
                                + "\n"
                            )
                    print(
                        f"[p1-matrix] {p['pair_id']} {p['kind']}: sum={res.sum_logprob:.3f} "
                        f"n={res.n_target_tokens} finite={res.all_finite} {rec['elapsed_s']}s",
                        flush=True,
                    )
                except ScoringError as e:
                    rec["status"] = STATUS_BOUNDARY
                    rec["failure_reason"] = str(e)
                    print(f"[p1-matrix] {p['pair_id']}: boundary_error: {e}", flush=True)
                except Exception as e:  # noqa: BLE001
                    rec["status"] = STATUS_ERROR
                    rec["failure_reason"] = f"{type(e).__name__}: {e}"
                    print(f"[p1-matrix] {p['pair_id']}: ERROR {e}", flush=True)

                emit(pf, rec)
                if rec["status"] == STATUS_OK:
                    n_done += 1
                if should_stop(rec["status"]):
                    note = f"anomaly status={rec['status']} on {p['pair_id']}"
                    print(f"[p1-matrix] {note}; stopping per contract", flush=True)
                    exit_code = EXIT_ANOMALY
                    break
    except Exception:
        exit_code = EXIT_ANOMALY
        note = "exception mid-run (see run_failure record)"
        raise
    finally:
        ledger.finalize(invocation_id, time.time() - t_start, n_forwards, exit_code, note)
        cum_f, cum_w = ledger.cumulative()
        print(
            f"[p1-matrix] ledger: this invocation {n_forwards} forwards / {time.time()-t_start:.1f}s; "
            f"cumulative {cum_f} forwards / {cum_w:.1f}s ({cum_w/3600:.4f} card-hours)",
            flush=True,
        )

    if exit_code == EXIT_OK:
        print(f"[p1-matrix] done: {n_done} new scored ({len(skipped)} skipped completed)", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
