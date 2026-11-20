"""Bind the completed CT–DeepCAD and L2 transfers, scores and rank proofs."""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
from pathlib import Path

from collect_retrieval_benchmark_fill import verify_received
from eval_retrieval_benchmark_fill import FrozenCondition, TEXT2CAD_CHECKPOINT_SHA, read_rank_results
from prepare_retrieval_benchmark_fill import load_json, sha256, text_digest, verify_sha256
from run_retrieval_benchmark_text2cad import condition_binding


IMPLEMENTATION_SHA = "6cd791312eefa5e575b0e56a39553fa9f92fbd3b"
AUTHORIZATION_SHA = "fc71f871a8da2aa78bfcff3714ff3b5afa9d546d"
AUTH_PATH = "experiments/configs/idea01_retrieval_benchmark_fill_run_authorization.json"
GALLERY_SIZE = 8032
# Values are distinct text rows and registered query UIDs, respectively.
CONDITION_COUNTS = {"ct_deepcad": (2029, 7513), "l2": (7604, 7614)}
RECOVERY_ROWS = 4


def load_pinned_approval() -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads(subprocess.check_output(["git", "-C", str(root), "show", f"{AUTHORIZATION_SHA}:{AUTH_PATH}"]))


def require_fields(record: dict, expected: dict, label: str) -> None:
    if any(key not in record or record[key] != value for key, value in expected.items()):
        raise ValueError(f"Inconsistent {label}")


def successful_step(logs: Path, phase: str) -> tuple[Path, dict]:
    path = logs / f"{phase}.receipt.json"
    record = load_json(path)
    require_fields(record, {"phase": phase, "actual_exit": 0}, f"successful finisher step {phase}")
    if type(record["actual_exit"]) is not int:
        raise ValueError(f"Invalid finisher exit code for {phase}")
    return path, record


def selected_transfer(run: Path, logs: Path, arm: str) -> tuple[Path, dict, dict]:
    # The evaluator receipt names the successful pull it consumed, even after a retry.
    evaluation_path, evaluation = successful_step(logs, f"eval_text2cad_{arm}")
    argv = evaluation["argv"]
    if (len(argv) < 8 or Path(argv[-8]).name != "eval_retrieval_benchmark_fill.py"
            or argv[-7] != "--runtime" or Path(argv[-6]).resolve() != run
            or argv[-5:-1] != ["text2cad", "--condition", arm, "--input"]):
        raise ValueError(f"Unexpected {arm} evaluation receipt command")
    full = Path(argv[-1]).resolve()
    raw, pull = full.parent, full.parent.parent
    if (full.name != "full" or raw.name != "raw" or pull.parent != run / "received"
            or not re.fullmatch(rf"text2cad_{arm}_first_batch(?:_[a-z0-9_]+)?", pull.name)):
        raise ValueError(f"Invalid {arm} evaluation input transfer path")
    receipt_path = pull / "pull_receipt.json"
    receipt = load_json(receipt_path)
    require_fields(receipt, {"actual_ssh_exit": 0, "kind": "text2cad", "conditions": [arm], "methods": None},
                   f"successful {arm} transfer receipt")
    if type(receipt["actual_ssh_exit"]) is not int:
        raise ValueError(f"Invalid {arm} transfer exit code")
    checked = verify_received(raw)
    require_fields(receipt, checked, f"verified {arm} transfer receipt")
    pins = {name: digest for digest, name in (line.split("  ", 1) for line in
            (raw / "TRANSFER_SHA256SUMS").read_text(encoding="utf-8").splitlines())}
    return raw, pins, {
        "relative_transfer_directory": pull.relative_to(run).as_posix(),
        "transfer_manifest_sha256": checked["manifest_sha256"], "transfer_payloads": checked["payloads"],
        "transfer_bytes": checked["bytes"],
        "transfer_receipt_sha256": sha256(receipt_path), "evaluation_receipt_sha256": sha256(evaluation_path),
    }


def require_score_command(argv: list[str], role: str, extra: dict[str, str]) -> None:
    if (len(argv) < 5 or argv[1:3] != ["-B", "-u"]
            or Path(argv[3]).name != "run_retrieval_benchmark_text2cad.py" or argv[4] != role):
        raise ValueError(f"Unexpected {role} scoring command")
    for flag, value in {"--auth-sha": AUTHORIZATION_SHA, "--mode": "score", **extra}.items():
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1:][:1] != [value]:
            raise ValueError(f"First batch {role} launch binding differs at {flag}")


def settled_ledger(l2: Path, approval: dict, planned: int) -> tuple[dict, dict[str, dict]]:
    controller = l2 / "control/score"
    if (controller / "controller.actual_exit").read_text(encoding="ascii").strip() != "0":
        raise ValueError("First batch controller did not exit successfully")
    actual = load_json(l2 / "full/controller_exit.json")
    require_fields(actual, {"exit_code": 0, "conditions": list(CONDITION_COUNTS),
                            "implementation_sha": IMPLEMENTATION_SHA, "authorization_sha": AUTHORIZATION_SHA},
                   "successful first batch controller receipt")
    if type(actual["exit_code"]) is not int:
        raise ValueError("Invalid first batch controller exit code")
    argv = shlex.split((controller / "launch_command.txt").read_text(encoding="utf-8"))
    require_score_command(argv, "controller", {})
    ledger = load_json(l2 / "full/ledger.json")
    require_fields(ledger, {"approved_sha": IMPLEMENTATION_SHA, "max_pairs": approval["max_pairs"],
                            "max_card_seconds": approval["max_card_seconds"]}, "final first batch ledger binding")
    if type(ledger["pairs_charged"]) is not int or ledger["pairs_charged"] != planned:
        raise ValueError("First batch charged pairs differ from the exact matrix; recovery requires manual review")
    if not 0 < ledger["card_seconds_spent"] <= approval["max_card_seconds"]:
        raise ValueError("First batch final ledger card time is invalid")
    invocations = {}
    for entry in ledger["invocations"]:
        invocation = entry["invocation_id"]
        if (invocation in invocations or entry["finalized"] is not True
                or type(entry["exit_code"]) is not int or entry["exit_code"] != 0
                or not 0 < entry["started_epoch"] <= entry["ended_epoch"] <= approval["deadline_epoch"]
                or not math.isclose(entry["card_seconds"], entry["ended_epoch"] - entry["started_epoch"],
                                    rel_tol=0, abs_tol=0.011)
                or type(entry.get("pairs_charged")) is not int or entry["pairs_charged"] <= 0):
            raise ValueError("First batch ledger has an unsettled or invalid worker invocation")
        invocations[invocation] = entry
    if (sum(entry["pairs_charged"] for entry in invocations.values()) != planned
            or not math.isclose(sum(entry["card_seconds"] for entry in invocations.values()),
                                ledger["card_seconds_spent"], rel_tol=0, abs_tol=0.011 * len(invocations))):
        raise ValueError("First batch final ledger totals do not match the settled invocations")
    return ledger, invocations


def one_worker_file(full: Path, pattern: str) -> Path:
    matches = list(full.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one successful worker evidence file matching {pattern}")
    return matches[0]


def certify_workers(full: Path, arm: str, complete: dict, binding: str,
                    approval: dict, invocations: dict[str, dict]) -> tuple[set[str], list[Path]]:
    devices = complete["devices"]
    if (not 1 <= len(devices) <= approval["max_devices"] or len(set(devices)) != len(devices)
            or any(type(device) is not int or device < 0 for device in devices)):
        raise ValueError(f"Invalid {arm} worker device set")
    exit_path = one_worker_file(full, f"{arm}_exit_*.json")
    exits = load_json(exit_path)
    require_fields(exits, {"devices": devices, "exits": [0] * len(devices)}, f"successful {arm} worker exits")
    if any(type(status) is not int for status in exits["exits"]):
        raise ValueError(f"Invalid {arm} worker exit code")
    used, evidence = set(), [exit_path]
    for shard, device in enumerate(devices):
        worker = one_worker_file(full, f"{arm}_worker{shard}_*.json")
        record = load_json(worker)
        written = len(range(shard, CONDITION_COUNTS[arm][0], len(devices)))
        require_fields(record, {"condition": arm, "device": device, "binding": binding,
                                "rows_written": written, "verified_prior_rows": 0}, f"complete {arm} worker shard")
        invocation = record["invocation"]
        if invocation in used or invocation not in invocations or worker.name != f"{arm}_worker{shard}_{invocation}.json":
            raise ValueError(f"Invalid {arm} worker invocation identity")
        launch = one_worker_file(full, f"{arm}_shard{shard}_*.launch.json")
        launch_record = load_json(launch)
        require_fields(launch_record, {"device": device}, f"{arm} worker launch")
        require_score_command(launch_record["argv"], "worker", {"--condition": arm, "--device": str(device),
                              "--shard": str(shard), "--shards": str(len(devices))})
        require_fields(invocations[invocation], {"note": f"benchmark_{arm}", "shard_id": device, "cards_used": 1,
                      "pairs_charged": written * GALLERY_SIZE, "pid": launch_record["pid"]},
                       f"settled {arm} worker ledger entry")
        evidence.extend((worker, launch))
        used.add(invocation)
    if set(full.glob(f"{arm}_worker*.json")) | set(full.glob(f"{arm}_shard*.launch.json")) != set(evidence[1:]):
        raise ValueError(f"Unexpected extra {arm} worker records")
    return used, evidence


def certify_matrix(raw: Path, pins: dict, frozen: FrozenCondition, binding: str) -> str:
    folder = raw / "full/scores" / frozen.name
    digests = []
    files = {folder / "text_tokens.json"}
    for row in frozen.texts:
        index = row["text_index"]
        score = folder / f"t{index:05d}.npy"
        receipt = score.with_suffix(".json")
        digest = pins.get(score.relative_to(raw).as_posix())
        if digest is None:
            raise ValueError(f"Missing verified score row for {frozen.name}/{index}")
        require_fields(load_json(receipt), {"text_index": index, "text_sha256": row["text_sha256"],
                                           "binding": binding, "scores_sha256": digest}, f"raw {frozen.name} score receipt")
        files.update((receipt, score))
        digests.append(digest)
    if set(folder.iterdir()) != files:
        raise ValueError(f"Unexpected raw matrix row coverage for {frozen.name}")
    return text_digest("\n".join(digests))


def certify_ranks(run: Path, logs: Path, raw: Path, frozen: FrozenCondition, approval: dict,
                  row_digest: str, publication: dict) -> dict:
    arm, full = frozen.name, raw / "full"
    witness_path = logs / f"audit_{arm}.json"
    witness = load_json(witness_path)
    if witness.get("independent_checks_passed") is not True or witness.get("condition") != arm:
        raise ValueError(f"Independent rank witness has not passed for {arm}")
    if set(witness["methods"]) != {"sum", "s48"}:
        raise ValueError(f"Incomplete independent rank witness for {arm}")
    expected_q = CONDITION_COUNTS[arm][1]
    methods = {}
    for method in ("sum", "s48"):
        result = run / "evaluation" / arm / method
        metrics_path, rank_path = result / "metrics.json", result / "query_ranks.jsonl"
        metrics = load_json(metrics_path)
        require_fields(metrics, {"method": method, "condition": arm, "gallery": GALLERY_SIZE,
                                 "queries": expected_q, "frozen_manifest_sha256": approval["freeze_sha256"],
                                 "failed_queries": 0, "truncated_queries": 0}, f"published {arm}/{method} metric")
        ranks = read_rank_results(result, frozen)
        if len(ranks) != expected_q:
            raise ValueError(f"Wrong registered rank denominator for {arm}/{method}")
        expected_source = {"implementation_sha": IMPLEMENTATION_SHA,
                           "authorization_sha256": sha256(full / "run_authorization.json"),
                           "completion_sha256": sha256(full / f"{arm}_complete.json"),
                           "probe_result_sha256": approval["probe_result_sha256"],
                           "checkpoint_sha256": TEXT2CAD_CHECKPOINT_SHA, "ordered_score_row_digest": row_digest,
                           "token_metadata_sha256": sha256(full / "scores" / arm / "text_tokens.json")}
        if method == "s48":
            expected_source["background_sha256"] = frozen.manifest["files"]["background.npy"]
        require_fields(metrics["source"], expected_source, f"accepted raw matrix source for {arm}/{method}")
        if Path(metrics["source"]["path"]).resolve() != full / "scores" / arm:
            raise ValueError(f"Metric came from an unselected transfer for {arm}/{method}")
        pins = {"metric_sha256": sha256(metrics_path), "rank_sha256": sha256(rank_path),
                "n_queries": expected_q, "hits": metrics["hits"], "median_rank": metrics["median_rank"]}
        require_fields(witness["methods"][method], pins, f"independent rank witness for {arm}/{method}")
        require_fields(publication[arm][method], {"metrics_sha256": pins["metric_sha256"],
                         "query_ranks_sha256": pins["rank_sha256"]}, f"final publication for {arm}/{method}")
        methods[method] = pins
    return {"witness_sha256": sha256(witness_path), "methods": methods}


def certify(run: Path, logs: Path) -> dict:
    """Return a read-only certificate for the exact settled first batch."""
    run, logs = run.resolve(), logs.resolve()
    if logs.parent != run / "control" or not re.fullmatch(r"finish_first_batch(?:_[a-z0-9_]+)?", logs.name):
        raise ValueError("Expected the first finisher's own control directory")
    current_path = logs / "current.json"
    state = load_json(current_path)
    require_fields(state, {"phase": "first_batch", "state": "completed", "external_all_passed": True,
                           "generator_all_passed": True}, "completed first finisher state")
    preview = logs / "preview_l2"
    final_steps = {phase: sha256(successful_step(logs, phase)[0])
                   for phase in ("summary_l2", "prepare_publication_l2", "latex_l2")}
    preview_path = preview / "preview_receipt.json"
    rendered = load_json(preview_path)
    require_fields(rendered, {"pages": 3, "pdf_sha256": sha256(preview / "preview.pdf")}, "final publication preview")
    publication_path = preview / "staged/RETRIEVAL_TABLE_EVIDENCE.json"
    publication = load_json(publication_path)["benchmark_fill"]

    approval = load_pinned_approval()
    planned = sum(texts * GALLERY_SIZE for texts, _ in CONDITION_COUNTS.values())
    reserve = RECOVERY_ROWS * GALLERY_SIZE
    require_fields(approval, {"approved_sha": IMPLEMENTATION_SHA, "conditions": list(CONDITION_COUNTS),
                    "run_id": run.name, "output_directory": "full", "mode": "score", "status": "pilot_authorized",
                    "max_pairs": planned + reserve}, "immutable first-batch approval")
    verify_sha256(run / "frozen/freeze.json", approval["freeze_sha256"])
    if (Path(publication["run_directory"]).resolve() != run
            or publication["freeze_sha256"] != approval["freeze_sha256"]):
        raise ValueError("Final publication points to a different frozen first batch")

    transfers = {arm: selected_transfer(run, logs, arm) for arm in CONDITION_COUNTS}
    final_raw, final_pins, _ = transfers["l2"]
    ledger, invocations = settled_ledger(final_raw, approval, planned)
    used, condition_proofs = set(), {}
    cumulative_pairs, cumulative_cards = 0, 0.0
    for arm, (expected_rows, expected_q) in CONDITION_COUNTS.items():
        raw, pins, transfer_proof = transfers[arm]
        full = raw / "full"
        if load_json(full / "run_authorization.json") != approval:
            raise ValueError(f"Received {arm} authorization differs from the immutable approval")
        frozen = FrozenCondition.read(run / "frozen", arm)
        if (len(frozen.gallery), len(frozen.texts), len(frozen.queries)) != (GALLERY_SIZE, expected_rows, expected_q):
            raise ValueError(f"Unexpected registered first-batch denominator for {arm}")
        verify_sha256(full / "gallery.jsonl", frozen.gallery_sha)
        binding = condition_binding(approval, arm)
        completion = full / f"{arm}_complete.json"
        complete = load_json(completion)
        shard_invocations, worker_files = certify_workers(full, arm, complete, binding, approval, invocations)
        if used & shard_invocations:
            raise ValueError("A worker invocation was reused across first-batch conditions")
        used |= shard_invocations
        cumulative_pairs += expected_rows * GALLERY_SIZE
        cumulative_cards += sum(invocations[inv]["card_seconds"] for inv in shard_invocations)
        require_fields(complete, {"condition": arm, "gallery": GALLERY_SIZE, "complete_rows": expected_rows,
                                  "binding": binding, "pairs_charged_total": cumulative_pairs}, f"{arm} matrix completion")
        # The ledger rounds individual records and its aggregate independently to centiseconds.
        if not math.isclose(complete["cumulative_card_seconds"], cumulative_cards, rel_tol=0, abs_tol=0.011 * len(used)):
            raise ValueError(f"{arm} condition card time differs from settled workers")
        if arm == "l2" and complete["cumulative_card_seconds"] != ledger["card_seconds_spent"]:
            raise ValueError("L2 completion and the final first-batch ledger disagree on card time")
        matrix_digest = certify_matrix(raw, pins, frozen, binding)
        ranks = certify_ranks(run, logs, raw, frozen, approval, matrix_digest, publication["conditions"])
        condition_proofs[arm] = {**transfer_proof, **ranks, "complete_rows": expected_rows,
                                 "gallery": GALLERY_SIZE, "queries": expected_q,
                                 "completion_sha256": sha256(completion), "ordered_score_row_digest": matrix_digest,
                                 "worker_pins": {path.name: sha256(path) for path in worker_files}}
    if used != invocations.keys() or cumulative_pairs != ledger["pairs_charged"]:
        raise ValueError("The final first-batch ledger contains unbound worker history")
    return {
        "schema_version": 1, "passed": True, "run_id": run.name, "implementation_sha": IMPLEMENTATION_SHA,
        "authorization_sha": AUTHORIZATION_SHA, "freeze_sha256": approval["freeze_sha256"],
        "planned_pairs": planned, "pairs_charged": ledger["pairs_charged"], "approved_recovery_reserve": reserve,
        "ledger_sha256": final_pins["full/ledger.json"], "card_seconds_spent": ledger["card_seconds_spent"],
        "controller_exit_sha256": final_pins["full/controller_exit.json"],
        "controller_actual_exit_sha256": final_pins["control/score/controller.actual_exit"],
        "controller_launch_sha256": final_pins["control/score/launch_command.txt"],
        "finish_state_sha256": sha256(current_path), "final_finisher_step_pins": final_steps,
        "publication_sha256": sha256(publication_path), "preview_receipt_sha256": sha256(preview_path),
        "conditions": condition_proofs,
    }
