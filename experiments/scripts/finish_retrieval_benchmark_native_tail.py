"""Audit the completed L1/L3 CPU receipt and publish after the first batch."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np

from collect_retrieval_benchmark_fill import CORE_EXTERNAL, sha256, verify_received
from finish_retrieval_benchmark_first_batch import publish, python, read_rows, step, write_status


ARMS = ("l1", "l3")
KS = (1, 5, 10, 20)
GALLERY_ARTIFACTS = ("gallery.npy", "gallery_meta.json", "gallery_status.jsonl", "l0_replay.json")


def audit(scratch: Path, source: Path, arm: str, method: str) -> dict:
    """Sort every all-gallery row directly from the text and shape vectors."""
    cad_rows = read_rows(scratch / "frozen/gallery.jsonl")
    cad_indices = np.array([row["cad_index"] for row in cad_rows])
    cad_position = {int(index): i for i, index in enumerate(cad_indices)}
    queries = read_rows(scratch / "frozen" / f"queries_{arm}.jsonl")
    groups = defaultdict(list)
    for query in queries:
        groups[query["text_index"]].append(query)
    shape = np.load(source / "gallery.npy", mmap_mode="r", allow_pickle=False)
    text = np.load(source / f"texts_{arm}.npy", mmap_mode="r", allow_pickle=False)
    shape_flags = read_rows(source / "gallery_status.jsonl")
    text_flags = read_rows(source / f"texts_{arm}_status.jsonl")
    if len(queries) != 7614 or len(cad_rows) != 8032 or list(cad_position) != sorted(cad_position):
        raise ValueError("Independent L1/L3 rank audit found an incorrect gallery or query axis")
    if list(zip(cad_indices, (r["uid"] for r in cad_rows))) != [
        (r["cad_index"], r["uid"]) for r in shape_flags
    ] or sorted(groups) != list(range(len(text))) or len(text_flags) != len(text):
        raise ValueError("Independent rank audit found incomplete embedding axes")
    valid_shape = np.array([r["valid"] for r in shape_flags])
    expected = {}
    for start in range(0, len(text), 64):
        for i, score in enumerate(text[start : start + 64] @ shape.T, start):
            order = np.lexsort((cad_indices, np.where(valid_shape, -score, np.inf)))
            inverse = np.empty(len(order), dtype=np.int32)
            inverse[order] = np.arange(1, len(order) + 1)
            for query in groups[i]:
                pos = cad_position[query["cad_index"]]
                rank = int(inverse[pos]) if text_flags[i]["valid"] and valid_shape[pos] else len(order) + 1
                expected[query["uid"]] = (rank, text_flags[i]["truncated"])
    result = scratch / "evaluation" / arm / method
    rank_path = result / "query_ranks.jsonl"
    metric_path = result / "metrics.json"
    saved = read_rows(rank_path)
    metric = json.loads(metric_path.read_bytes())
    observed = {row["uid"]: (row["rank"], row["truncated"]) for row in saved}
    ranks = np.array([rank for rank, _ in expected.values()])
    hits = [int(np.count_nonzero(ranks <= k)) for k in KS]
    if (len(expected) != 7614 or len(saved) != 7614 or expected != observed
            or hits != metric["hits"] or np.median(ranks) != metric["median_rank"]):
        raise ValueError(f"Independent rank or metric mismatch for {arm}/{method}")
    return {"queries": len(ranks), "hits": hits, "median_rank": metric["median_rank"],
            "metric_sha256": sha256(metric_path), "rank_sha256": sha256(rank_path)}


def finish(run: Path, reports: Path, preview: Path, control: Path) -> None:
    first = run / "control/finish_first_batch"
    collection = run / "received/external_native_tail"
    receipt_path = collection / "pull_receipt.json"
    collector = json.loads((run / "control/external_native_tail_collect_launcher.json").read_bytes())
    write_status(control / "current.json", phase="waiting_for_external_l1_l3", state="waiting")
    while not receipt_path.is_file() and time.time() < collector["launch_epoch"] + 7500:
        time.sleep(15)
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt["actual_ssh_exit"] != 0 or receipt["kind"] != "external"
            or receipt["external_phase"] != "native_tail" or receipt["conditions"] != list(ARMS)
            or receipt["methods"] != list(CORE_EXTERNAL)):
        raise ValueError("The receipt does not certify the complete L1/L3 external collection")
    raw = collection / "raw"
    if any(receipt[key] != value for key, value in verify_received(raw).items()):
        raise ValueError("L1/L3 transport coverage changed")

    scratch = control / "staged"
    scratch.mkdir()
    shutil.copytree(run / "frozen", scratch / "frozen")
    audited = {}
    for method in CORE_EXTERNAL:
        source = raw / "external" / method
        previous_base = "clip_first_batch" if method == "clip" else "openshape_mmcad_first_batch"
        previous = run / "received" / previous_base / "raw/external" / method
        if any(sha256(source / name) != sha256(previous / name) for name in GALLERY_ARTIFACTS):
            raise ValueError(f"The L1/L3 encoder changed the original gallery evidence: {method}")
        for arm in ARMS:
            name = f"{arm}_{method}"
            step(control, f"evaluate_{name}", python(
                "eval_retrieval_benchmark_fill.py", "--runtime", scratch, "external",
                "--input", source, "--condition", arm, "--method", method,
            ))
            audited[name] = audit(scratch, source, arm, method)
    write_status(control / "external_ready.json", independent_checks_passed=True, audits=audited)
    write_status(control / "current.json", phase="waiting_for_first_batch_publication", state="waiting")
    deadline = time.time() + 33 * 3600
    while time.time() < deadline:
        state = json.loads((first / "current.json").read_bytes())
        if state["state"] == "completed":
            break
        if state["state"] in {"failed", "partial_external"}:
            raise RuntimeError("The first batch has not completed publication")
        time.sleep(45)
    else:
        raise TimeoutError("The first batch has not completed publication")
    for arm in ARMS:
        target = run / "evaluation" / arm
        target.mkdir(exist_ok=True)
        for method in CORE_EXTERNAL:
            if (target / method).exists():
                raise FileExistsError(target / method)
            result = scratch / "evaluation" / arm / method
            witness = audited[f"{arm}_{method}"]
            if (sha256(result / "metrics.json") != witness["metric_sha256"]
                    or sha256(result / "query_ranks.jsonl") != witness["rank_sha256"]):
                raise ValueError(f"An audited L1/L3 result changed before publication: {arm}/{method}")
    for arm in ARMS:
        for method in CORE_EXTERNAL:
            (scratch / "evaluation" / arm / method).rename(run / "evaluation" / arm / method)
    if not publish(run, reports, preview, control, "external_native_tail"):
        raise RuntimeError("The first-batch publication witness is incomplete")
    write_status(control / "current.json", phase="external_native_tail", state="completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "reports", "preview"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    control = args.runtime / "control/finish_external_native_tail"
    control.mkdir()
    try:
        finish(args.runtime, args.reports, args.preview, control)
    except (OSError, RuntimeError, ValueError) as error:
        write_status(control / "current.json", state="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
