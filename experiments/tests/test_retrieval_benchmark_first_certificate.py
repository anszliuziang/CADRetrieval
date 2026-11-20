"""Certify synthetic settled copies using the actual collector and evaluator schemas."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import certify_retrieval_benchmark_first_batch as cert
from collect_retrieval_benchmark_fill import verify_received
from eval_retrieval_benchmark_fill import FrozenCondition, text2cad_rows
from finish_retrieval_benchmark_first_batch import audit_generator
from prepare_retrieval_benchmark_fill import CONDITIONS, freeze_condition, load_json, sha256, write_json, write_jsonl
from run_retrieval_benchmark_text2cad import condition_binding, publish_score_row, score_row_path


def seal_transfer(pull, arm):
    raw = pull / "raw"
    pins = raw / "TRANSFER_SHA256SUMS"
    lines = [f"{sha256(path)}  {path.relative_to(raw).as_posix()}\n"
             for path in sorted(raw.rglob("*")) if path.is_file() and path != pins]
    pins.write_text("".join(lines), encoding="utf-8", newline="\n")
    write_json(pull / "pull_receipt.json", {"actual_ssh_exit": 0, "kind": "text2cad",
                                          "conditions": [arm], "methods": None, **verify_received(raw)})


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    run = tmp_path / "idea01_retrieval_benchmark_fill_20261120"
    logs = run / "control/finish_first_batch"
    logs.mkdir(parents=True)
    frozen = run / "frozen"
    frozen.mkdir()
    gallery = [{"uid": f"u{index}", "cad_index": index} for index in (0, 2, 5)]
    write_jsonl(frozen / "gallery.jsonl", gallery)
    np.save(frozen / "background.npy", np.array([-2.0, -1.0, 0.0], dtype="<f8"))
    np.save(frozen / "gallery_fields.npy", np.array([10, 14, 18], dtype="<i4"))
    conditions, axes = {}, {}
    for arm in CONDITIONS:
        by_uid = {"u0": "one", "u2": "one" if arm == "ct_deepcad" else "two", "u5": "three"}
        texts, queries = freeze_condition(gallery, by_uid)
        axes[arm] = texts
        unique = len(texts)
        conditions[arm] = {"queries": 3, "unique_texts": unique, "singleton_queries": 3 if unique == 3 else 1,
                           "largest_group": 1 if unique == 3 else 2}
        write_jsonl(frozen / f"texts_{arm}.jsonl", texts)
        write_jsonl(frozen / f"queries_{arm}.jsonl", queries)
    write_json(frozen / "freeze.json", {"gallery_count": 3, "domain": "DeepCAD", "conditions": conditions,
                                       "files": {path.name: sha256(path) for path in frozen.iterdir()}})
    approved = {"mode": "score", "status": "pilot_authorized", "run_id": run.name, "output_directory": "full",
                "approved_sha": cert.IMPLEMENTATION_SHA, "conditions": ["ct_deepcad", "l2"],
                "freeze_sha256": sha256(frozen / "freeze.json"), "probe_result_sha256": "1" * 64,
                "max_pairs": 27, "max_card_seconds": 60, "max_devices": 4, "deadline_epoch": 200}
    monkeypatch.setattr(cert, "load_pinned_approval", lambda: approved)
    monkeypatch.setattr(cert, "GALLERY_SIZE", 3)
    monkeypatch.setattr(cert, "CONDITION_COUNTS", {"ct_deepcad": (2, 3), "l2": (3, 3)})
    write_json(logs / "current.json", {"phase": "first_batch", "state": "completed",
                                      "external_all_passed": True, "generator_all_passed": True})

    pulls, invocations, cumulative_pairs = {}, [], 0
    for index, arm in enumerate(("ct_deepcad", "l2")):
        # The evaluator may have consumed a preserved, successful retry rather than the first directory.
        name = f"text2cad_{arm}_first_batch" + ("_retry1" if index == 0 else "")
        pull = run / "received" / name
        pulls[arm] = pull
        full = pull / "raw/full"
        score_dir = full / "scores" / arm
        score_dir.mkdir(parents=True)
        (full / "gallery.jsonl").write_bytes((frozen / "gallery.jsonl").read_bytes())
        write_json(full / "run_authorization.json", approved)
        write_json(full / "capacity_projection.json", {"gallery": 3})
        rows = axes[arm]
        binding = condition_binding(approved, arm)
        write_json(score_dir / "text_tokens.json", [{"text_index": row["text_index"],
                   "text_sha256": row["text_sha256"], "token_count": 3, "effective_token_count": 3,
                   "truncated": False, "ids_sha256": "2" * 64} for row in rows])
        for row in rows:
            publish_score_row(score_row_path(full, arm, row["text_index"]), row, binding,
                              np.array([-3.0, -2.0, -1.0], dtype="<f8"))
        cumulative_pairs += len(rows) * 3
        device, invocation_id = index, f"20261120T10000{index}Z-{index:08x}"
        write_json(full / f"{arm}_complete.json", {"condition": arm, "gallery": 3, "complete_rows": len(rows),
                    "devices": [device], "pairs_charged_total": cumulative_pairs,
                    "cumulative_card_seconds": 10.0 * (index + 1), "binding": binding})
        write_json(full / f"{arm}_exit_{index + 1}.json", {"devices": [device], "exits": [0]})
        write_json(full / f"{arm}_worker0_{invocation_id}.json", {"invocation": invocation_id, "device": device,
                   "condition": arm, "rows_written": len(rows), "verified_prior_rows": 0, "binding": binding})
        invocation = {"invocation_id": invocation_id, "shard_id": device, "cards_used": 1, "finalized": True,
                      "started_epoch": 20 + index * 20, "ended_epoch": 30 + index * 20, "card_seconds": 10.0,
                      "exit_code": 0, "note": f"benchmark_{arm}", "pairs_charged": len(rows) * 3, "pid": 10 + index}
        invocations.append(invocation)
        worker = full / f"{arm}_shard0_{index + 1}"
        write_json(worker.with_suffix(".launch.json"), {"device": device, "pid": 10 + index,
                   "started_epoch": invocation["started_epoch"], "argv": ["python", "-B", "-u",
                   "run_retrieval_benchmark_text2cad.py", "worker", "--mode", "score", "--auth-sha",
                   cert.AUTHORIZATION_SHA, "--condition", arm, "--device", str(device), "--shard", "0", "--shards", "1"]})
        worker.with_suffix(".stdout.log").touch()
        worker.with_suffix(".stderr.log").touch()
        if arm == "l2":
            controller = pull / "raw/control/score"
            controller.mkdir(parents=True)
            (controller / "controller.actual_exit").write_text("0\n", encoding="ascii")
            (controller / "launch_command.txt").write_text(
                "python -B -u run_retrieval_benchmark_text2cad.py controller "
                f"--mode score --auth-sha {cert.AUTHORIZATION_SHA}\n", encoding="ascii")
            write_json(full / "controller_exit.json", {"exit_code": 0, "conditions": ["ct_deepcad", "l2"],
                       "implementation_sha": cert.IMPLEMENTATION_SHA, "authorization_sha": cert.AUTHORIZATION_SHA})
            write_json(full / "ledger.json", {"approved_sha": cert.IMPLEMENTATION_SHA, "pairs_charged": 15,
                       "card_seconds_spent": 20.0, "max_pairs": 27, "max_card_seconds": 60,
                       "invocations": invocations})
        seal_transfer(pull, arm)
        phase = f"eval_text2cad_{arm}"
        write_json(logs / f"{phase}.receipt.json", {"actual_exit": 0, "phase": phase,
                   "argv": ["python", "-B", "eval_retrieval_benchmark_fill.py", "--runtime", str(run), "text2cad",
                            "--condition", arm, "--input", str(full)]})
        text2cad_rows(FrozenCondition.read(frozen, arm), full, run / "evaluation")
        audit_generator(run, full, arm, logs / f"audit_{arm}.json")
    preview = logs / "preview_l2"
    (preview / "staged").mkdir(parents=True)
    (preview / "preview.pdf").write_bytes(b"synthetic preview")
    write_json(preview / "preview_receipt.json", {"pages": 3, "pdf_sha256": sha256(preview / "preview.pdf")})
    published = {arm: {method: {**load_json(run / "evaluation" / arm / method / "metrics.json"),
                 "metrics_sha256": sha256(run / "evaluation" / arm / method / "metrics.json")}
                 for method in ("sum", "s48")} for arm in pulls}
    write_json(preview / "staged/RETRIEVAL_TABLE_EVIDENCE.json", {"benchmark_fill": {
        "run_directory": str(run), "freeze_sha256": approved["freeze_sha256"], "conditions": published,
    }})
    for phase in ("summary_l2", "prepare_publication_l2", "latex_l2"):
        write_json(logs / f"{phase}.receipt.json", {"phase": phase, "actual_exit": 0})
    return run, logs, pulls


def test_certificate_binds_the_consumed_retry_pull_and_raw_final_ledger(completed_run):
    run, logs, pulls = completed_run
    result = cert.certify(run, logs)
    assert result["passed"] is True
    assert result["ledger_sha256"] == sha256(pulls["l2"] / "raw/full/ledger.json")
    assert result["pairs_charged"] == 15 and result["approved_recovery_reserve"] == 12
    assert result["conditions"]["ct_deepcad"]["complete_rows"] == 2
    assert result["conditions"]["l2"]["complete_rows"] == 3
    assert result["conditions"]["ct_deepcad"]["transfer_manifest_sha256"] == sha256(
        pulls["ct_deepcad"] / "raw/TRANSFER_SHA256SUMS")


@pytest.mark.parametrize("change, error", [
    ("workflow", "completed"), ("controller", "controller"), ("pending", "ledger"), ("reserve", "charged"),
    ("transfer", "checksum"), ("rebound_transfer", "raw matrix source"), ("worker", "worker"),
    ("worker_launch", "worker ledger"), ("witness", "witness"), ("publication", "publication"),
])
def test_inconsistent_first_batch_does_not_get_a_certificate(completed_run, change, error):
    run, logs, pulls = completed_run
    full = pulls["l2"] / "raw/full"
    if change == "workflow":
        write_json(logs / "current.json", {"state": "partial_external", "phase": "first_batch"})
    elif change == "transfer":
        path = full / "scores/l2/t00000.npy"
        path.write_bytes(path.read_bytes() + b"mutated")
    elif change == "rebound_transfer":
        path = full / "scores/l2/t00000.npy"
        receipt = load_json(path.with_suffix(".json"))
        np.save(path, np.array([-4.0, -2.0, -1.0], dtype="<f8"))
        write_json(path.with_suffix(".json"), {**receipt, "scores_sha256": sha256(path)})
        seal_transfer(pulls["l2"], "l2")
    elif change == "witness":
        path = logs / "audit_l2.json"
        audit = load_json(path)
        audit["methods"]["s48"]["rank_sha256"] = "0" * 64
        write_json(path, audit)
    elif change == "publication":
        path = logs / "preview_l2/staged/RETRIEVAL_TABLE_EVIDENCE.json"
        receipt = load_json(path)
        receipt["benchmark_fill"]["conditions"]["l2"]["sum"]["metrics_sha256"] = "0" * 64
        write_json(path, receipt)
    else:
        if change == "controller":
            path = full / "controller_exit.json"
            record = load_json(path)
            record["authorization_sha"] = "0" * 40
        elif change == "worker":
            path = next(full.glob("l2_exit_*.json"))
            record = load_json(path)
            record["exits"] = [2]
        elif change == "worker_launch":
            path = next(full.glob("l2_shard*.launch.json"))
            record = load_json(path)
            record["pid"] = 456
        else:
            path = full / "ledger.json"
            record = load_json(path)
            if change == "pending":
                record["invocations"][-1]["finalized"] = False
            else:
                record["pairs_charged"] += 3
        write_json(path, record)
        seal_transfer(pulls["l2"], "l2")
    with pytest.raises(ValueError, match=error):
        cert.certify(run, logs)
