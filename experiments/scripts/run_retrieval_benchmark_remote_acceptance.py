"""Evaluate the settled first Text2CAD benchmark batch on its scoring host."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

from certify_retrieval_benchmark_first_batch import (
    AUTHORIZATION_SHA,
    CONDITION_COUNTS,
    GALLERY_SIZE,
    IMPLEMENTATION_SHA,
    RECOVERY_ROWS,
    certify_matrix,
    certify_workers,
    load_pinned_approval,
    require_fields,
    settled_ledger,
)
from eval_retrieval_benchmark_fill import (
    FrozenCondition, TEXT2CAD_CHECKPOINT_SHA, read_rank_results, text2cad_rows,
)
from finish_retrieval_benchmark_first_batch import audit_generator
from prepare_retrieval_benchmark_fill import load_json, sha256, verify_sha256, write_json
from run_retrieval_benchmark_text2cad import condition_binding, publish_file_once


PLANNED_PAIRS = sum(count * GALLERY_SIZE for count, _ in CONDITION_COUNTS.values())


def preflight(run: Path, output: Path) -> dict:
    approval = load_pinned_approval()
    require_fields(approval, {
        "mode": "score", "status": "pilot_authorized", "run_id": run.name,
        "output_directory": "full", "approved_sha": IMPLEMENTATION_SHA,
        "conditions": list(CONDITION_COUNTS), "max_pairs": PLANNED_PAIRS + RECOVERY_ROWS * GALLERY_SIZE,
    }, "immutable scoring authorization")
    if load_json(run / "full/run_authorization.json") != approval:
        raise ValueError("Score directory does not match the immutable approval")
    verify_sha256(output / "frozen/freeze.json", approval["freeze_sha256"])
    verify_sha256(run / "frozen/freeze.json", approval["freeze_sha256"])
    for arm, (text_count, query_count) in CONDITION_COUNTS.items():
        frozen = FrozenCondition.read(output / "frozen", arm)
        if (len(frozen.gallery), len(frozen.texts), len(frozen.queries)) != (GALLERY_SIZE, text_count, query_count):
            raise ValueError(f"Unexpected frozen coverage for {arm}")
        verify_sha256(run / "frozen" / f"texts_{arm}.jsonl", frozen.text_sha)
        verify_sha256(run / "full/gallery.jsonl", frozen.gallery_sha)
    return approval


def verify_completion(run: Path, approval: dict, arm: str, cumulative_pairs: int) -> dict:
    complete = load_json(run / "full" / f"{arm}_complete.json")
    require_fields(complete, {
        "condition": arm, "gallery": GALLERY_SIZE, "complete_rows": CONDITION_COUNTS[arm][0],
        "binding": condition_binding(approval, arm), "pairs_charged_total": cumulative_pairs,
    }, f"{arm} completed score matrix")
    if not 0 < complete["cumulative_card_seconds"] <= approval["max_card_seconds"]:
        raise ValueError(f"Invalid recorded card time for {arm}")
    return complete


def evaluate(run: Path, output: Path, arm: str, approval: dict) -> dict:
    # A completed receipt is rechecked without repeating evaluation or independent ranking.
    proof_path = output / f"rank_audit_{arm}.json"
    completion = run / "full" / f"{arm}_complete.json"
    frozen = FrozenCondition.read(output / "frozen", arm)
    folder = run / "full/scores" / arm
    binding = condition_binding(approval, arm)
    if not proof_path.exists():
        if (output / "evaluation" / arm).exists():
            raise ValueError(f"Preserve incomplete {arm} evaluation in {output}")
        text2cad_rows(frozen, run / "full", output / "evaluation")
        audit_generator(output, run / "full", arm, proof_path)
    proof = load_json(proof_path)
    if proof.get("independent_checks_passed") is not True or proof["condition"] != arm:
        raise ValueError(f"Independent rank audit failed for {arm}")
    expected_count = CONDITION_COUNTS[arm][1]
    if set(proof["methods"]) != {"sum", "s48"}:
        raise ValueError(f"Missing expected scoring methods for {arm}")
    score_pins = {p.relative_to(run).as_posix(): sha256(p) for p in folder.glob("t[0-9]*.npy")}
    row_digest = certify_matrix(run, score_pins, frozen, binding)
    source = {
        "path": str(folder), "implementation_sha": IMPLEMENTATION_SHA,
        "authorization_sha256": sha256(run / "full/run_authorization.json"),
        "completion_sha256": sha256(completion), "probe_result_sha256": approval["probe_result_sha256"],
        "checkpoint_sha256": TEXT2CAD_CHECKPOINT_SHA, "ordered_score_row_digest": row_digest,
        "token_metadata_sha256": sha256(folder / "text_tokens.json"),
    }
    for method, witness in proof["methods"].items():
        path = output / "evaluation" / arm / method
        rank_path, metrics_path = path / "query_ranks.jsonl", path / "metrics.json"
        require_fields(witness, {
            "n_queries": expected_count, "rank_sha256": sha256(rank_path), "metric_sha256": sha256(metrics_path),
        }, f"{arm}/{method} independent rank witness")
        metrics = load_json(metrics_path)
        read_rank_results(path, frozen)
        require_fields(metrics, {
            "condition": arm, "method": method,
            "gallery": GALLERY_SIZE, "queries": expected_count, "failed_queries": 0, "truncated_queries": 0,
            "frozen_manifest_sha256": approval["freeze_sha256"],
            "hits": witness["hits"], "median_rank": witness["median_rank"],
        }, f"audited {arm}/{method} metrics")
        expected_source = source if method == "sum" else {**source,
            "background_sha256": frozen.manifest["files"]["background.npy"]}
        require_fields(metrics["source"], expected_source, f"{arm}/{method} score provenance")
    return {
        "rank_audit_sha256": sha256(proof_path), **proof,
        "gallery": GALLERY_SIZE, "queries": expected_count, "score_rows": len(frozen.texts),
        "ordered_score_row_digest": row_digest, "token_metadata_sha256": source["token_metadata_sha256"],
    }


def accept(run: Path, output: Path, approval: dict) -> dict:
    ledger_path = run / "full/ledger.json"
    ledger_sha = sha256(ledger_path)
    ledger, invocations = settled_ledger(run, approval, PLANNED_PAIRS)
    used = set()
    results = {}
    cumulative_pairs, total_card_seconds = 0, 0.0
    for arm, (text_count, _) in CONDITION_COUNTS.items():
        cumulative_pairs += text_count * GALLERY_SIZE
        complete = verify_completion(run, approval, arm, cumulative_pairs)
        ids, paths = certify_workers(run / "full", arm, complete, complete["binding"], approval, invocations)
        if used & ids:
            raise ValueError("A scoring invocation appears in two conditions")
        used |= ids
        total_card_seconds += sum(invocations[id_]["card_seconds"] for id_ in ids)
        if not math.isclose(complete["cumulative_card_seconds"], total_card_seconds,
                            abs_tol=0.011 * len(used), rel_tol=0):
            raise ValueError(f"Worker card time differs from {arm} completion")
        results[arm] = {
            **evaluate(run, output, arm, approval),
            "completion_sha256": sha256(run / "full" / f"{arm}_complete.json"),
            "worker_files": {path.name: sha256(path) for path in paths},
        }
    if used != invocations.keys() or sha256(ledger_path) != ledger_sha:
        raise ValueError("Final ledger coverage or stability check failed")
    require_fields(complete, {"cumulative_card_seconds": ledger["card_seconds_spent"]}, "final card time")
    return {
        "passed": True, "run_id": run.name, "authorization_sha": AUTHORIZATION_SHA,
        "scorer_sha": IMPLEMENTATION_SHA, "freeze_sha256": approval["freeze_sha256"],
        "pairs_charged": PLANNED_PAIRS, "card_seconds_spent": ledger["card_seconds_spent"],
        "ledger_sha256": ledger_sha,
        "controller_exit_sha256": sha256(run / "full/controller_exit.json"),
        "actual_exit_sha256": sha256(run / "control/score/controller.actual_exit"),
        "conditions": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("stage", choices=("preflight", "ct_deepcad", "final"))
    args = parser.parse_args()
    run, output = args.run.resolve(strict=True), args.output.resolve(strict=True)
    if output.parent != run or not output.name.startswith("remote_acceptance_"):
        raise ValueError("Use an isolated acceptance output under the scoring run")
    approval = preflight(run, output)
    if args.stage == "preflight":
        print("Frozen scoring authorization, gallery, and both query axes verified")
    elif args.stage == "ct_deepcad":
        verify_completion(run, approval, args.stage, CONDITION_COUNTS[args.stage][0] * GALLERY_SIZE)
        proof = evaluate(run, output, args.stage, approval)
        print(f"CT–DeepCAD rank audit passed for {proof['methods']['sum']['n_queries']} queries")
    else:
        destination = output / "first_batch_accepted.json"
        if destination.exists():
            raise FileExistsError(destination)
        certificate = accept(run, output, approval)
        with tempfile.TemporaryDirectory(dir=output) as scratch:
            pending = Path(scratch) / destination.name
            write_json(pending, certificate)
            publish_file_once(pending, destination)
        print(f"First batch accepted for {certificate['pairs_charged']} candidate pairs")


if __name__ == "__main__":
    main()
