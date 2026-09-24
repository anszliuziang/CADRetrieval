#!/usr/bin/env python
"""Build review_supplement/ for run idea01_p0_20260924_r01 (Codex review item 1).

CPU + tokenizer only. No model load, no NPU, no re-scoring. Reads the ORIGINAL
run dir (pairs.jsonl, tokens/, coverage/selection.json) read-only, rebuilds each
pair's full model input token ids from pinned data + checkpoint slow tokenizer,
cross-checks every target token id against the original evidence, and writes:

  <out-dir>/pairs_supplement.jsonl   per-pair: full input_ids, target ids,
                                     paths/revisions/hashes, original git_sha,
                                     cross-check result
  <out-dir>/DERIVATION.md            derivation rules (how inputs are rebuilt)
  <out-dir>/verification_summary.json  aggregate cross-check outcome

Exit code 0 iff all pairs cross-check clean.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.envspec import env_spec_hashes  # noqa: E402
from idea01.supplement import crosscheck_pair, query_for_condition, reconstruct_input_ids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SPEC_REL = "experiments/configs/idea01_yunnan_share2_env.json"

DERIVATION = """# review_supplement 派生规则（Codex review 2026-09-25, item 1）

目的：原始 P0 运行只保存了目标 token ID；本目录以只读 CPU/tokenizer 方式重建
60 对的**完整模型输入 token ID**，并与原始逐 token 证据逐 token 交叉校验。
原始 `smoke/pairs.jsonl` 与 `smoke/tokens/` 未做任何修改；无模型前向、无重评分。

## 派生规则

1. query 文本：按 `coverage/selection.json` 的 pair 记录重建——correct 用候选行
   `input`，mismatched 用 `mismatch_cand_index` 所指候选行的 `input`，empty 用空串。
   数据源：pinned `data/data_val.jsonl`（revision 见记录）。
2. 前缀：`### Instruction:\\n{query}\\n\\n### Response:\\n`（与原始运行相同模板）。
3. tokenize：checkpoint 自带 tokenizer，`use_fast=False`，`add_special_tokens=False`；
   `input_ids = tok(prefix + output) + [eos_id]`（eos_id=151645，`<|im_end|>`）。
4. 边界断言：`tok(prefix)` 必须是 `tok(prefix+output)` 的严格前缀，否则该对记
   `boundary_error`（与评分器同一断言函数 `scoring.build_scored_sequence`）。
5. 交叉校验（`idea01.supplement.crosscheck_pair`）：重建的 n_prefix/n_input/
   n_target 与原始 `pairs.jsonl` 一致；原始 tokens 文件每个 token_id 等于重建
   `input_ids` 同位置值；position 恰为 `target_start..L-1`，`pred_row=position-1`；
   末目标 = EOS。不重算 logprob（需要模型前向，超出本补证范围）。

## 记录口径

- `original_git_sha`：逐对取自原始 `pairs.jsonl`（本次运行为 4e19259f…），保留
  原始执行 provenance；`derived_by_git_sha` 为运行本脚本时 checkout 的 SHA。
- env spec 双哈希：`canonical_json_sha256`（sort_keys=True、紧凑分隔符的 canonical
  JSON 字节 SHA256，准备期记录前缀 10c18c17）与 `file_bytes_sha256`（Git blob
  原始字节 SHA256，LF 行尾，运行记录前缀 b6cdd884）。两者同源同内容，仅算法口径不同。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="original run dir (read-only usage)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="checkpoint dir (tokenizer only)")
    ap.add_argument("--out-dir", required=True, help="e.g. <run-dir>/review_supplement")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in open(run_dir / "smoke" / "pairs.jsonl", encoding="utf-8")]
    selection = json.loads((run_dir / "coverage" / "selection.json").read_text(encoding="utf-8"))
    cands = {c["cand_index"]: c for c in selection["candidates"]}
    sel_pairs = {p["pair_id"]: p for p in selection["pairs"]}

    val_rows: list[dict] = []
    with open(Path(args.data_dir) / "data" / "data_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_rows.append(json.loads(line))

    from transformers import AutoTokenizer  # tokenizer only; no model load

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    eos_id = tok.eos_token_id
    print(f"[supplement] tokenizer class={type(tok).__name__} eos_id={eos_id}", flush=True)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    derived_by_sha = "unavailable"
    try:
        import subprocess

        derived_by_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    spec = env_spec_hashes(REPO_ROOT / ENV_SPEC_REL, REPO_ROOT, ENV_SPEC_REL)

    n_ok = 0
    n_fail = 0
    failures: list[dict] = []
    with open(out_dir / "pairs_supplement.jsonl", "w", encoding="utf-8") as out:
        for rec in pairs:
            pid = rec["pair_id"]
            sp = sel_pairs[pid]
            cand = cands[sp["cand_index"]]
            own = val_rows[cand["row_index"]]
            mismatch_input = None
            if sp["condition"] == "mismatched":
                mismatch_input = val_rows[cands[cand["mismatch_cand_index"]]["row_index"]]["input"]
            query = query_for_condition(sp["condition"], own["input"], mismatch_input)
            program = own["output"]

            token_recs = [
                json.loads(l) for l in open(run_dir / "smoke" / "tokens" / f"{pid}.jsonl", encoding="utf-8")
            ]
            sup = {
                "pair_id": pid,
                "condition": rec["condition"],
                "query_uid": rec["query_uid"],
                "candidate_uid": rec["candidate_uid"],
                "candidate_row_index": rec["candidate_row_index"],
                "original_git_sha": rec.get("git_sha"),
                "derived_by_git_sha": derived_by_sha,
                "model_revision": rec["model_revision"],
                "data_revision": rec["data_revision"],
                "model_dir": args.model_dir,
                "data_dir": args.data_dir,
                "env_spec_canonical_json_sha256": spec["canonical_json_sha256"],
                "env_spec_file_bytes_sha256": spec["file_bytes_sha256"],
                "env_spec_hash_algorithms": {
                    "canonical": spec["canonical_json_algorithm"],
                    "file": spec["file_bytes_algorithm"],
                },
                "eos_id": eos_id,
            }
            try:
                input_ids, target_start, _ = reconstruct_input_ids(query, program, encode, eos_id)
                errors = crosscheck_pair(input_ids, target_start, eos_id, rec, token_recs)
                sup["input_ids"] = input_ids
                sup["target_ids"] = input_ids[target_start:]
                sup["n_prefix_tokens"] = target_start
                sup["n_input_tokens"] = len(input_ids)
                sup["n_target_tokens"] = len(input_ids) - target_start
                sup["crosscheck_errors"] = errors
                sup["status"] = "ok" if not errors else "crosscheck_failed"
            except Exception as e:  # noqa: BLE001
                sup["status"] = "reconstruction_error"
                sup["crosscheck_errors"] = [f"{type(e).__name__}: {e}"]
            if sup["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
                failures.append({"pair_id": pid, "errors": sup["crosscheck_errors"]})
            out.write(json.dumps(sup) + "\n")
            if sup["status"] != "ok":
                print(f"[supplement] {pid}: {sup['status']}: {sup['crosscheck_errors']}", flush=True)

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Codex review 2026-09-25 item 1: rebuild full model input token ids for the 60 P0 pairs (CPU/tokenizer only, no re-scoring); cross-check against original per-token evidence",
        "n_pairs": len(pairs),
        "n_crosscheck_ok": n_ok,
        "n_failed": n_fail,
        "failures": failures,
        "derived_by_git_sha": derived_by_sha,
        "original_run_git_sha": sorted({r.get("git_sha") for r in pairs}),
        "env_spec": spec,
        "eos_id": eos_id,
    }
    with open(out_dir / "verification_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (out_dir / "DERIVATION.md").write_text(DERIVATION, encoding="utf-8")
    print(f"[supplement] {n_ok}/{len(pairs)} cross-check ok, {n_fail} failed -> {out_dir}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
