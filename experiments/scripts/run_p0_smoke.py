#!/usr/bin/env python
"""P0 smoke scoring: teacher-forced conditional scores on up to 60 pairs.

Isolation contract: the scoring path receives ONLY the query text and the
candidate program text. UIDs / qrels / CSV captions are used solely for
bookkeeping in the output records, never as scorer input.

Writes to --out-dir:
  environment_<UTC timestamp>.json   one per invocation (never overwritten);
                                     linked from each pair record via invocation_id
  pairs.jsonl                        one summary record per scored/failed pair,
                                     including FULL model input token ids,
                                     model/data dirs and both env-spec hashes
  tokens/<pair_id>.jsonl             per-token position/id/logprob/decoded evidence

Stop policy (Codex review 2026-09-25, item 2):
  - boundary_error / error / nonfinite: record the failed pair, STOP the run,
    exit EXIT_ANOMALY (non-zero).
  - wallclock budget reached: stop, exit EXIT_BUDGET (non-zero).
  - unsupported_full_sequence (>max_seq_len): recorded and counted, run continues.
  - --pair-ids are deduplicated and capped at MAX_PAIRS (60) per task contract.
"""

from __future__ import annotations

import argparse
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
from idea01.scoring import ScoringError, build_scored_sequence, gather_response_logprobs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SPEC_REL = "experiments/configs/idea01_yunnan_share2_env.json"
ENV_SPEC_PATH = REPO_ROOT / ENV_SPEC_REL

MAX_PAIRS = 60
EXIT_OK = 0
EXIT_ANOMALY = 2
EXIT_BUDGET = 3

STATUS_OK = "ok"
STATUS_UNSUPPORTED = "unsupported_full_sequence"
STATUS_BOUNDARY = "boundary_error"
STATUS_NONFINITE = "nonfinite"
STATUS_ERROR = "error"
STOP_STATUSES = frozenset({STATUS_BOUNDARY, STATUS_NONFINITE, STATUS_ERROR})


def should_stop(status: str) -> bool:
    """Pure stop policy: True => record then abort the run with EXIT_ANOMALY."""
    return status in STOP_STATUSES


def parse_pair_ids(spec: str, n_pairs: int, max_pairs: int = MAX_PAIRS) -> list[int]:
    """Parse '0', '1-59', 'all', '0,3,5-7'. Deduplicates (order preserved) and
    enforces the task pair cap. Raises ValueError on out-of-range or >max_pairs."""
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


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def spec_hashes() -> dict:
    if ENV_SPEC_PATH.exists():
        return env_spec_hashes(ENV_SPEC_PATH, REPO_ROOT, ENV_SPEC_REL)
    return {
        "canonical_json_sha256": None,
        "canonical_json_algorithm": "sha256(utf8(json.dumps(obj, sort_keys=True, separators=(',',':'))))",
        "file_bytes_sha256": None,
        "file_bytes_algorithm": "sha256(git blob bytes, LF endings)",
    }


def write_environment(out_dir: Path, cfg: dict, argv: list[str], device: str, invocation_id: str, pair_ids: list[int]) -> Path:
    """Per-invocation environment record; filename carries the invocation id so
    repeated calls never overwrite each other (Codex review 2026-09-25, item 1)."""
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
    ap.add_argument("--selection", required=True, help="selection.json from run_coverage.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pair-ids", default="all", help="e.g. '0', '1-59', 'all'; deduped, capped at 60")
    ap.add_argument("--device", default=None, help="override config device, e.g. npu:0")
    ap.add_argument("--max-wallclock-s", type=float, default=3300.0)
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

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    pairs = selection["pairs"]
    cands = {c["cand_index"]: c for c in selection["candidates"]}
    try:
        pair_ids = parse_pair_ids(args.pair_ids, len(pairs), MAX_PAIRS)
    except ValueError as e:
        print(f"[smoke] invalid --pair-ids: {e}", flush=True)
        return EXIT_ANOMALY

    invocation_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_environment(out_dir, cfg, sys.argv, device, invocation_id, pair_ids)

    val_rows: list[dict] = []
    with open(Path(cfg["data_dir"]) / "data" / "data_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_rows.append(json.loads(line))

    import torch
    import torch_npu  # noqa: F401  (registers NPU backend)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_dir"], use_fast=False)
    eos_id = tok.eos_token_id
    print(f"[smoke] tokenizer eos_id={eos_id} class={type(tok).__name__}", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], torch_dtype=dtype)
    model.eval()
    model.to(device)
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s dtype={cfg['dtype']} device={device}", flush=True)
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
    }

    pairs_path = out_dir / "pairs.jsonl"
    n_done = 0
    n_unsupported = 0

    def emit(pf, rec: dict) -> None:
        pf.write(json.dumps(rec) + "\n")
        pf.flush()

    with open(pairs_path, "a", encoding="utf-8") as pf:
        for pid in pair_ids:
            if time.time() - t_start > args.max_wallclock_s:
                print(f"[smoke] wallclock budget reached after {n_done} pairs; stopping", flush=True)
                return EXIT_BUDGET
            p = pairs[pid]
            cand = cands[p["cand_index"]]
            out_row = val_rows[cand["row_index"]]
            if p["condition"] == "correct":
                query = out_row["input"]
            elif p["condition"] == "mismatched":
                query = val_rows[cands[cand["mismatch_cand_index"]]["row_index"]]["input"]
            else:
                query = ""
            program = out_row["output"]

            rec = {
                **base_rec,
                "pair_id": p["pair_id"],
                "pair_index": pid,
                "condition": p["condition"],
                "query_uid": p["query_uid"],
                "candidate_uid": p["candidate_uid"],
                "candidate_row_index": cand["row_index"],
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
                    # coverage marker, not an anomaly: record and continue (review item 2)
                    rec["status"] = STATUS_UNSUPPORTED
                    emit(pf, rec)
                    n_unsupported += 1
                    print(f"[smoke] {p['pair_id']}: unsupported_full_sequence len={len(input_ids)}", flush=True)
                    continue

                ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(input_ids=ids_t).logits
                logits_np = logits[0].float().cpu().numpy()
                del logits, ids_t
                res = gather_response_logprobs(logits_np, input_ids, target_start)

                rec["sum_logprob"] = res.sum_logprob
                rec["mean_logprob"] = res.mean_logprob
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
                    f"[smoke] {p['pair_id']} {p['condition']}: sum={res.sum_logprob:.3f} "
                    f"n={res.n_target_tokens} finite={res.all_finite} {rec['elapsed_s']}s",
                    flush=True,
                )
            except ScoringError as e:
                rec["status"] = STATUS_BOUNDARY
                rec["failure_reason"] = str(e)
                print(f"[smoke] {p['pair_id']}: boundary_error: {e}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["status"] = STATUS_ERROR
                rec["failure_reason"] = f"{type(e).__name__}: {e}"
                print(f"[smoke] {p['pair_id']}: ERROR {e}", flush=True)

            emit(pf, rec)
            if rec["status"] == STATUS_OK:
                n_done += 1
            if should_stop(rec["status"]):
                print(f"[smoke] anomaly status={rec['status']} on {p['pair_id']}; stopping per contract", flush=True)
                return EXIT_ANOMALY

    print(
        f"[smoke] done: {n_done} scored, {n_unsupported} unsupported (recorded, continued) "
        f"in {time.time()-t_start:.1f}s",
        flush=True,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
