#!/usr/bin/env python
"""P1 scoring matrix runner (gate 2 ONLY — requires Codex pilot_authorized + approved SHA).

Teacher-forced conditional scoring of the fixed P1 pair plan (<=416 forwards):
for each pair, score = FP32 log-softmax sum of response+EOS token log-probs,
reusing the P0-reviewed scorer (idea01.scoring) unchanged.

Isolation contract: the scoring path receives ONLY the query text and the
candidate program text. UIDs / row indices / sources are bookkeeping in the
output records, never scorer input.

Inputs:
  --config      P1 task config (model/data dirs, dtype, device, budgets)
  --selection   p1_selection.json produced by run_p1_selection.py (CPU gate);
                its sha256 is recorded into every record and the environment

Writes to --out-dir (outside the Git checkout):
  environment_<UTC timestamp>.json   one per invocation (never overwritten)
  pairs.jsonl                        one record per scored/failed pair incl.
                                     FULL input ids, model/data dirs, both
                                     env-spec hashes, selection manifest sha256,
                                     sum/mean/eos_logprob, device, dtype, timing
  tokens/<pair_id>.jsonl             per-token position/pred_row/id/text/logprob

Stop policy (plan sec.4: hash failure, boundary misalignment, non-finite score,
OOM, overlong planned pair, pair budget exceeded, dirty remote Git => record
and stop, exit non-zero; never report a success-only subset):
  - boundary_error / error / nonfinite / overlong_sequence: record, STOP, EXIT_ANOMALY.
  - wallclock budget reached (default 2700 s): stop, EXIT_BUDGET.
  - pair plan larger than 416, or --pair-ids exceeding 416 unique: refuse to run.
  - git worktree dirty: refuse to run.
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.envspec import env_spec_hashes  # noqa: E402
from idea01.prompts import build_prefix  # noqa: E402
from idea01.p1_runner import (  # noqa: E402
    DEFAULT_MAX_WALLCLOCK_S,
    EXIT_ANOMALY,
    EXIT_BUDGET,
    EXIT_OK,
    MAX_PAIRS,
    STATUS_BOUNDARY,
    STATUS_ERROR,
    STATUS_NONFINITE,
    STATUS_OK,
    STATUS_OVERLONG,
    parse_pair_ids,
    should_stop,
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
    """Empty string iff the checkout is clean (plan requires a clean remote Git)."""
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


def write_environment(out_dir: Path, cfg: dict, argv: list[str], device: str, invocation_id: str, pair_ids: list[int], selection_sha256: str) -> Path:
    env = {
        "invocation_id": invocation_id,
        "timestamp_utc": invocation_id,
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
    ap.add_argument("--selection", required=True, help="p1_selection.json from run_p1_selection.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pair-ids", default="all", help="e.g. '0', '0-9', 'all'; deduped, hard cap 416")
    ap.add_argument("--device", default=None, help="override config device, e.g. npu:0")
    ap.add_argument("--max-wallclock-s", type=float, default=DEFAULT_MAX_WALLCLOCK_S)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    (out_dir / "tokens").mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model_dir:
        cfg["model_dir"] = args.model_dir
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    device = args.device or cfg["device"]

    dirty = git_dirty()
    if dirty:
        print(f"[p1-matrix] git worktree not clean; refuse to run per plan:\n{dirty}", flush=True)
        return EXIT_ANOMALY

    selection_path = Path(args.selection)
    sel_sha = file_sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pairs = selection["pairs"]
    if selection.get("gate_summary", {}).get("n_gate_fail", 1) != 0:
        print("[p1-matrix] selection manifest reports CPU gate failures; refuse to run", flush=True)
        return EXIT_ANOMALY
    try:
        pair_ids = parse_pair_ids(args.pair_ids, len(pairs), MAX_PAIRS)
    except ValueError as e:
        print(f"[p1-matrix] invalid --pair-ids: {e}", flush=True)
        return EXIT_ANOMALY

    invocation_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_environment(out_dir, cfg, sys.argv, device, invocation_id, pair_ids, sel_sha)

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

    hashes = spec_hashes()
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

    pairs_path = out_dir / "pairs.jsonl"
    n_done = 0
    n_forwards = 0

    def emit(pf, rec: dict) -> None:
        pf.write(json.dumps(rec) + "\n")
        pf.flush()

    with open(pairs_path, "a", encoding="utf-8") as pf:
        for pid in pair_ids:
            if time.time() - t_start > args.max_wallclock_s:
                print(f"[p1-matrix] wallclock budget reached after {n_done} pairs; stopping", flush=True)
                return EXIT_BUDGET
            if n_forwards >= MAX_PAIRS:
                print(f"[p1-matrix] forward budget {MAX_PAIRS} reached; stopping", flush=True)
                return EXIT_BUDGET
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
                    return EXIT_ANOMALY

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
                print(f"[p1-matrix] anomaly status={rec['status']} on {p['pair_id']}; stopping per contract", flush=True)
                return EXIT_ANOMALY

    wall = time.time() - t_start
    print(
        f"[p1-matrix] done: {n_done} scored, {n_forwards} forwards in {wall:.1f}s "
        f"({wall/3600:.4f} card-hours on 1 card)",
        flush=True,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
