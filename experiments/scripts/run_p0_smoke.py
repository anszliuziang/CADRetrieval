#!/usr/bin/env python
"""P0 smoke scoring: teacher-forced conditional scores on up to 60 pairs.

Isolation contract: the scoring path receives ONLY the query text and the
candidate program text. UIDs / qrels / CSV captions are used solely for
bookkeeping in the output records, never as scorer input.

Writes to --out-dir:
  environment.json          host/versions/SHA/env-spec-hash/command
  pairs.jsonl               one summary record per scored (or failed) pair
  tokens/<pair_id>.jsonl    per-token position/id/logprob/decoded evidence
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

from idea01.prompts import build_prefix  # noqa: E402
from idea01.scoring import ScoringError, build_scored_sequence, gather_response_logprobs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SPEC_PATH = REPO_ROOT / "experiments" / "configs" / "idea01_yunnan_share2_env.json"


def parse_pair_ids(spec: str, n_pairs: int) -> list[int]:
    if spec == "all":
        return list(range(n_pairs))
    ids: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    for i in ids:
        if not (0 <= i < n_pairs):
            raise ValueError(f"pair index {i} out of range [0, {n_pairs})")
    return ids


def sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def write_environment(out_dir: Path, cfg: dict, argv: list[str], device: str) -> None:
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "device": device,
        "dtype": cfg["dtype"],
        "git_sha": git_sha(),
        "env_spec_file": str(ENV_SPEC_PATH),
        "env_spec_sha256_prefix": sha256_prefix(ENV_SPEC_PATH) if ENV_SPEC_PATH.exists() else None,
        "model_revision": cfg["model_revision"],
        "model_dir": cfg["model_dir"],
        "data_revision": cfg["data_revision"],
        "data_dir": cfg["data_dir"],
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
    with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--selection", required=True, help="selection.json from run_coverage.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pair-ids", default="all", help="e.g. '0', '1-59', 'all'")
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

    write_environment(out_dir, cfg, sys.argv, device)

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    pairs = selection["pairs"]
    cands = {c["cand_index"]: c for c in selection["candidates"]}
    pair_ids = parse_pair_ids(args.pair_ids, len(pairs))

    # validation rows for query/program text (scorer sees only text)
    import json as _json

    val_rows: list[dict] = []
    with open(Path(cfg["data_dir"]) / "data" / "data_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_rows.append(_json.loads(line))

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

    pairs_path = out_dir / "pairs.jsonl"
    n_done = 0
    with open(pairs_path, "a", encoding="utf-8") as pf:
        for pid in pair_ids:
            if time.time() - t_start > args.max_wallclock_s:
                print("[smoke] wallclock budget reached; stopping", flush=True)
                break
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
                "pair_id": p["pair_id"],
                "pair_index": pid,
                "condition": p["condition"],
                "query_uid": p["query_uid"],
                "candidate_uid": p["candidate_uid"],
                "candidate_row_index": cand["row_index"],
                "model_revision": cfg["model_revision"],
                "data_revision": cfg["data_revision"],
                "git_sha": git_sha(),
                "device": device,
                "dtype": cfg["dtype"],
            }
            t_pair = time.time()
            try:
                prefix_ids = encode(build_prefix(query))
                full_ids = encode(build_prefix(query) + program)
                input_ids, target_start = build_scored_sequence(prefix_ids, full_ids, eos_id)
                rec["n_prefix_tokens"] = len(prefix_ids)
                rec["n_input_tokens"] = len(input_ids)
                rec["n_target_tokens"] = len(input_ids) - target_start
                if len(input_ids) > cfg["max_seq_len"]:
                    rec["status"] = "unsupported_full_sequence"
                    pf.write(json.dumps(rec) + "\n")
                    pf.flush()
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
                rec["status"] = "ok" if res.all_finite else "nonfinite"
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
                rec["status"] = "boundary_error"
                rec["failure_reason"] = str(e)
                print(f"[smoke] {p['pair_id']}: boundary_error: {e}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["status"] = "error"
                rec["failure_reason"] = f"{type(e).__name__}: {e}"
                print(f"[smoke] {p['pair_id']}: ERROR {e}", flush=True)
            pf.write(json.dumps(rec) + "\n")
            pf.flush()
            n_done += 1
            if rec["status"] == "nonfinite":
                print("[smoke] non-finite scores detected; stopping per contract", flush=True)
                break

    print(f"[smoke] done: {n_done} pairs processed in {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
