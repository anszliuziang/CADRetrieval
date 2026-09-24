#!/usr/bin/env python
"""P1 CPU-only selection pre-check: eligible pool, fixed 32x8 selection, pair tokenization gate.

No model weights are loaded and no forward pass is run; the checkpoint dir is
used for the slow tokenizer only (task gate 1: 0 GPU hours, 0 scored pairs).

Independently recomputes everything from the pinned raw data (JSONL + CSV +
ZIP + tokenizer), then optionally cross-checks against the P0 artifacts
(--p0-uid-manifest / --p0-coverage). Writes to --out-dir (must be outside the
Git checkout on the server):

  p1_selection.json            full selection manifest: eligibility counts,
                               pool-vs-full length stats, seed/offsets, all
                               queries/candidates with sources/similarities/
                               token lengths, pair plan, gate summary, rules
  p1_eligible_pool.json        eligible pool uid list (uid ascending) + counts
  p1_pair_tokenization.jsonl   per-pair CPU tokenization gate evidence

Stop policy (plan sec.2/sec.4): any eligibility cross-check mismatch, pool
< 32, unfillable 8-candidate set, pair tokenization boundary error or
len(input_ids) > max_seq_len => record, exit non-zero (EXIT_ANOMALY). Do not
relax rules silently.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.coverage import measure_row  # noqa: E402
from idea01.data_audit import build_intermediate_index, read_jsonl  # noqa: E402
from idea01.prompts import build_prefix  # noqa: E402
from idea01.p1_selection import (  # noqa: E402
    CANDIDATE_SOURCES,
    MAX_FORWARDS,
    MAX_PREFIX_TOKENS,
    MAX_RESPONSE_TOKENS,
    METHOD_EXCLUDE,
    METHOD_RE,
    N_LENGTH_NEIGHBORS,
    N_METHOD_NEIGHBORS,
    N_OFFSETS,
    N_QUERIES,
    SEED,
    RowRecord,
    SelectionError,
    build_pair_plan,
    compute_eligibility,
    extract_method_set,
    offset_coverage_check,
    select_p1,
)
from idea01.scoring import ScoringError, build_scored_sequence  # noqa: E402

EXIT_OK = 0
EXIT_ANOMALY = 2

VAL_FILE = "data/data_val.jsonl"
CSV_FILE = "text2cad_v1.1.csv"
ZIP_FILE = "CadQuery.zip"

REFERENCE_ELIGIBLE = 4949  # Codex read-only pre-count from the P0 coverage table (plan sec.2)
REFERENCE_N_ROWS = 5513


def length_stats(rows: list[RowRecord]) -> dict:
    if not rows:
        return {"n": 0}
    pf = sorted(r.n_prefix_tokens for r in rows)
    rs = sorted(r.n_response_tokens for r in rows)

    def q(vals, p):
        return vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]

    return {
        "n": len(rows),
        "prefix_mean": sum(pf) / len(pf),
        "prefix_p50": q(pf, 0.5),
        "response_mean": sum(rs) / len(rs),
        "response_p50": q(rs, 0.5),
        "full_mean": sum(pf[i] + rs[i] for i in range(len(pf))) / len(pf),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="checkpoint dir; slow tokenizer only, weights untouched")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None, help="P1 task config json (revisions recorded into the manifest)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-prefix-tokens", type=int, default=MAX_PREFIX_TOKENS)
    ap.add_argument("--max-response-tokens", type=int, default=MAX_RESPONSE_TOKENS)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--reference-eligible", type=int, default=REFERENCE_ELIGIBLE)
    ap.add_argument("--p0-uid-manifest", default=None, help="optional P0 uid_manifest_val.jsonl for cross-check")
    ap.add_argument("--p0-coverage", default=None, help="optional P0 coverage_val.jsonl for cross-check")
    args = ap.parse_args()

    t_start = time.time()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}

    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        print(f"[p1-select] FAIL: {msg}", flush=True)

    print(f"[p1-select] python={platform.python_version()} platform={platform.platform()}", flush=True)

    # ---- 1. raw data: rows, uid mapping, ZIP identity (independent recompute) ----
    rows = read_jsonl(data_dir / VAL_FILE)
    n_rows = len(rows)
    print(f"[p1-select] val rows={n_rows}", flush=True)

    index, n_csv = build_intermediate_index(data_dir / CSV_FILE)
    print(f"[p1-select] csv rows={n_csv} unique intermediate keys={len(index)}", flush=True)

    row_uids: list[str | None] = []
    program_identical: list[bool] = []
    with zipfile.ZipFile(data_dir / ZIP_FILE) as zf:
        names = set(zf.namelist())
        for r in rows:
            hits = index.get(r["input"], [])
            uid = hits[0] if len(hits) == 1 else None
            row_uids.append(uid)
            if uid is None:
                program_identical.append(False)
                continue
            member = f"CQ/{uid}.py"
            if member not in names:
                program_identical.append(False)
                continue
            program_identical.append(zf.read(member).decode("utf-8") == r["output"])
    print(f"[p1-select] unique uid rows={sum(u is not None for u in row_uids)} zip-identical={sum(program_identical)}", flush=True)

    # ---- 2. per-row token measurements (slow tokenizer, CPU) ----
    from transformers import AutoTokenizer  # deferred: heavy import; tokenizer only, no weights

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    eos_id = tok.eos_token_id
    print(f"[p1-select] tokenizer class={type(tok).__name__} eos_id={eos_id}", flush=True)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    records: list[RowRecord] = []
    t1 = time.time()
    for i, r in enumerate(rows):
        cov = measure_row(i, r["input"], r["output"], encode, eos_id, args.max_seq_len, build_prefix)
        records.append(
            RowRecord(
                row_index=i,
                uid=row_uids[i],
                program_identical=program_identical[i],
                n_prefix_tokens=cov.n_prefix_tokens,
                n_response_tokens=cov.n_response_tokens,
                boundary_ok=cov.boundary_ok,
                eos_in_output=cov.eos_in_output,
                method_set=extract_method_set(r["output"]),
            )
        )
        if (i + 1) % 1000 == 0:
            print(f"[p1-select] measured {i+1}/{n_rows} rows, {time.time()-t1:.1f}s", flush=True)

    # ---- 3. eligibility ----
    flags, counts = compute_eligibility(records, args.max_prefix_tokens, args.max_response_tokens)
    counts["reference_eligible"] = args.reference_eligible
    counts["reference_n_rows"] = REFERENCE_N_ROWS
    counts["matches_reference"] = counts["n_eligible"] == args.reference_eligible and n_rows == REFERENCE_N_ROWS
    print(f"[p1-select] eligibility: {json.dumps(counts)}", flush=True)
    if not counts["matches_reference"]:
        fail(
            f"eligible pool {counts['n_eligible']}/{n_rows} != reference "
            f"{args.reference_eligible}/{REFERENCE_N_ROWS} (plan sec.2 reference)"
        )

    # ---- 4. optional cross-check vs P0 artifacts ----
    crosscheck: dict = {}
    if args.p0_uid_manifest:
        p0_uids = [json.loads(ln)["uid"] for ln in Path(args.p0_uid_manifest).read_text(encoding="utf-8").splitlines() if ln.strip()]
        mism = [i for i, (a, b) in enumerate(zip(row_uids, p0_uids)) if a != b]
        crosscheck["p0_uid_manifest"] = {"path": args.p0_uid_manifest, "n": len(p0_uids), "n_mismatch": len(mism), "mismatch_rows": mism[:20]}
        if len(p0_uids) != n_rows or mism:
            fail(f"P0 uid manifest cross-check mismatch: {len(mism)} rows (len {len(p0_uids)} vs {n_rows})")
    if args.p0_coverage:
        p0_cov = [json.loads(ln) for ln in Path(args.p0_coverage).read_text(encoding="utf-8").splitlines() if ln.strip()]
        mism = []
        for i, c in enumerate(p0_cov):
            r = records[c["row_index"]]
            if (
                c["n_prefix_tokens"] != r.n_prefix_tokens
                or c["n_response_tokens"] != r.n_response_tokens
                or c["boundary_ok"] != r.boundary_ok
                or c["eos_in_output"] != r.eos_in_output
            ):
                mism.append(c["row_index"])
        crosscheck["p0_coverage"] = {"path": args.p0_coverage, "n": len(p0_cov), "n_mismatch": len(mism), "mismatch_rows": mism[:20]}
        if len(p0_cov) != n_rows or mism:
            fail(f"P0 coverage cross-check mismatch: {len(mism)} rows (len {len(p0_cov)} vs {n_rows})")
    print(f"[p1-select] crosscheck: {json.dumps(crosscheck)}", flush=True)

    # ---- 5. fixed selection ----
    selection = select_p1(records, flags, seed=args.seed)
    cov = offset_coverage_check(selection)
    print(f"[p1-select] selection: {len(selection.queries)} queries, offsets={selection.offsets}, offset_coverage_ok={cov['ok']}", flush=True)
    if not cov["ok"]:
        fail(f"offset coverage violation: {cov['violations']}")

    pairs = build_pair_plan(selection)
    n_cond = sum(1 for p in pairs if p["kind"] == "conditional")
    n_empty = sum(1 for p in pairs if p["kind"] == "empty")
    print(f"[p1-select] pair plan: {n_cond} conditional + {n_empty} empty = {len(pairs)} (cap {MAX_FORWARDS})", flush=True)
    if len(pairs) > MAX_FORWARDS:
        fail(f"pair plan {len(pairs)} exceeds forward budget {MAX_FORWARDS}")

    # ---- 6. CPU pair tokenization gate: every planned pair, assert boundary/EOS/len ----
    gate_rows: list[dict] = []
    n_gate_fail = 0
    max_input_len = 0
    for p in pairs:
        query = "" if p["kind"] == "empty" else rows[p["query_row_index"]]["input"]
        program = rows[p["candidate_row_index"]]["output"]
        g = {
            "pair_id": p["pair_id"],
            "kind": p["kind"],
            "query_uid": p["query_uid"],
            "candidate_uid": p["candidate_uid"],
        }
        try:
            prefix_ids = encode(build_prefix(query))
            full_ids = encode(build_prefix(query) + program)
            input_ids, target_start = build_scored_sequence(prefix_ids, full_ids, eos_id)
            g.update(
                n_prefix_tokens=len(prefix_ids),
                n_input_tokens=len(input_ids),
                n_target_tokens=len(input_ids) - target_start,
                ends_with_eos=bool(input_ids[-1] == eos_id),
                gate_ok=len(input_ids) <= args.max_seq_len,
            )
            if not g["gate_ok"]:
                g["failure"] = f"input length {len(input_ids)} > max_seq_len {args.max_seq_len}"
        except ScoringError as e:
            g.update(gate_ok=False, failure=f"boundary_error: {e}")
        gate_rows.append(g)
        if g.get("gate_ok"):
            max_input_len = max(max_input_len, g["n_input_tokens"])
        else:
            n_gate_fail += 1
            fail(f"pair {p['pair_id']}: {g.get('failure')}")
    with open(out_dir / "p1_pair_tokenization.jsonl", "w", encoding="utf-8") as f:
        for g in gate_rows:
            f.write(json.dumps(g) + "\n")
    gate_summary = {
        "n_pairs": len(pairs),
        "n_gate_ok": len(pairs) - n_gate_fail,
        "n_gate_fail": n_gate_fail,
        "max_seq_len": args.max_seq_len,
        "max_input_len_observed": max_input_len,
        "all_ends_with_eos": all(g.get("ends_with_eos") for g in gate_rows if g.get("gate_ok")),
    }
    print(f"[p1-select] tokenization gate: {json.dumps(gate_summary)}", flush=True)

    # ---- 7. manifest ----
    eligible_records = [r for r, ok in zip(records, flags) if ok]
    pool_doc = {
        "n_rows": n_rows,
        "n_eligible": counts["n_eligible"],
        "eligible_fraction": counts["n_eligible"] / n_rows,
        "eligibility_counts": counts,
        "pool_uids_uid_ascending": selection.pool_uids,
        "length_stats_pool": length_stats(eligible_records),
        "length_stats_all_rows": length_stats(records),
    }
    with open(out_dir / "p1_eligible_pool.json", "w", encoding="utf-8") as f:
        json.dump(pool_doc, f, indent=2)

    manifest = {
        "task_id": cfg.get("task_id", "idea01_p1_conditional_sensitivity_20260925"),
        "plan": "experiments/plans/IDEA01_P1_CONDITIONAL_SENSITIVITY_2026-09-25.md",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "data_dir": str(data_dir),
        "model_dir": str(args.model_dir),
        "data_revision": cfg.get("data_revision"),
        "model_revision": cfg.get("model_revision"),
        "rules": {
            "eligibility": (
                "val rows with: unique CSV uid (JSONL input == CSV intermediate verbatim); "
                "ZIP member CQ/{uid}.py byte-identical to JSONL output; prefix token boundary OK; "
                "no EOS inside tokenized output; n_prefix_tokens <= 250; n_response_tokens <= 700 (EOS included)"
            ),
            "query_draw": f"random.Random({args.seed}).sample(pool_uids_uid_ascending, 32); sampled order kept",
            "offsets": "same rng instance then draws 3 distinct offsets from range(1, 32); same offsets for all queries; candidate (i+offset) mod 32",
            "method_neighbors": (
                "pool minus the 32 selected uids; rank by Jaccard(method_set) desc, "
                "|response token length diff| asc, uid asc; take 2"
            ),
            "length_neighbors": (
                "remaining pool (pool minus 32 selected minus this query's 2 method picks); "
                "rank by |response token length diff| asc, uid asc; take 2; 8 candidates must be distinct else stop"
            ),
            "method_name_extraction": f"regex {METHOD_RE.pattern!r} over the raw program; excluded names: {sorted(METHOD_EXCLUDE)}",
            "jaccard_empty_union": "defined 0.0 when both method sets are empty",
            "pair_tokenization_gate": (
                f"every planned pair tokenized on CPU with the slow tokenizer; "
                f"assert prefix boundary (strict), EOS appended at id level, len(input_ids) <= {args.max_seq_len}"
            ),
            "scores_never_used": "selection is a pure function of data + seed; model scores were never consulted",
        },
        "seed": args.seed,
        "max_prefix_tokens": args.max_prefix_tokens,
        "max_response_tokens": args.max_response_tokens,
        "n_queries": N_QUERIES,
        "n_offsets": N_OFFSETS,
        "n_method_neighbors": N_METHOD_NEIGHBORS,
        "n_length_neighbors": N_LENGTH_NEIGHBORS,
        "candidate_sources": list(CANDIDATE_SOURCES),
        "offsets": selection.offsets,
        "eligibility_counts": counts,
        "crosscheck_p0": crosscheck,
        "offset_coverage": cov,
        "n_conditional_pairs": n_cond,
        "n_empty_pairs": n_empty,
        "n_pairs_total": len(pairs),
        "max_forwards_cap": MAX_FORWARDS,
        "gate_summary": gate_summary,
        "distinct_candidate_uids": selection.distinct_candidate_uids,
        "queries": [dataclasses.asdict(q) for q in selection.queries],
        "pairs": pairs,
        "failures": failures,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    with open(out_dir / "p1_selection.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[p1-select] wrote {out_dir / 'p1_selection.json'} (+ p1_eligible_pool.json, p1_pair_tokenization.jsonl)", flush=True)

    if failures:
        print(f"[p1-select] {len(failures)} failure(s); exiting non-zero per stop policy", flush=True)
        return EXIT_ANOMALY
    print(f"[p1-select] OK in {time.time()-t_start:.1f}s", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelectionError as e:
        print(f"[p1-select] SelectionError: {e}; stop per plan", flush=True)
        sys.exit(EXIT_ANOMALY)
