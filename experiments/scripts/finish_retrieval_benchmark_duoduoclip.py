"""Evaluate and independently verify the five completed DuoduoCLIP conditions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time

from collect_retrieval_benchmark_duoduoclip import ARMS
from collect_retrieval_benchmark_fill import sha256, verify_received
from finish_retrieval_benchmark_first_batch import publish, python, read_rows, step, write_status


METHOD = "duoduoclip"
K_LEVELS = (1, 5, 10, 20)


def audit(runtime: Path, source: Path, arm: str) -> dict:
    import numpy as np
    from eval_retrieval_benchmark_fill import FrozenCondition

    frozen = FrozenCondition.read(runtime / "frozen", arm)
    cad_axis = np.array([row["cad_index"] for row in frozen.gallery])
    shapes = np.load(source / "gallery.npy", mmap_mode="r", allow_pickle=False)
    texts = np.load(source / f"texts_{arm}.npy", mmap_mode="r", allow_pickle=False)
    shape_flags = np.array([r["valid"] for r in read_rows(source / "gallery_status.jsonl")])
    text_status = read_rows(source / f"texts_{arm}_status.jsonl")
    result = runtime / "evaluation" / arm / METHOD
    ranks_file, metrics_file = result / "query_ranks.jsonl", result / "metrics.json"
    saved = read_rows(ranks_file)
    observed = {row["uid"]: (row["rank"], row["truncated"]) for row in saved}
    checked = {}
    for offset in range(0, len(texts), 64):
        for index, scores in enumerate(texts[offset:offset + 64] @ shapes.T, offset):
            order = np.lexsort((cad_axis, np.where(shape_flags, -scores, np.inf)))
            inverse = np.empty(len(order), dtype=np.int32)
            inverse[order] = np.arange(1, len(order) + 1)
            for query in frozen.queries_by_text[index]:
                column = frozen.positions[query["cad_index"]]
                rank = (int(inverse[column]) if text_status[index]["valid"] and shape_flags[column]
                        else len(order) + 1)
                checked[query["uid"]] = (rank, text_status[index]["truncated"])
    metrics = json.loads(metrics_file.read_bytes())
    ranks = np.array([rank for rank, _ in checked.values()])
    hits = [int(np.count_nonzero(ranks <= k)) for k in K_LEVELS]
    if (len(saved) != len(frozen.queries) or len(checked) != len(saved) or observed != checked
            or hits != metrics["hits"] or np.median(ranks) != metrics["median_rank"]):
        raise ValueError(f"Independent DuoduoCLIP ranking differs for {arm}")
    return {"queries": len(ranks), "hits": hits, "median_rank": metrics["median_rank"],
            "metric_sha256": sha256(metrics_file), "rank_sha256": sha256(ranks_file)}


def finish(run: Path, reports: Path, preview: Path, control: Path, source: Path) -> None:
    receipt_file = source / "pull_receipt.json"
    write_status(control / "current.json", phase="duoduoclip_artifact", state="waiting")
    deadline = time.time() + 18 * 3600
    while not receipt_file.is_file() and time.time() < deadline:
        if (source / "failure.json").exists():
            raise RuntimeError("The DuoduoCLIP archive requires recovery")
        time.sleep(45)
    receipt = json.loads(receipt_file.read_bytes())
    raw = source / "raw"
    if (receipt["actual_ssh_exit"] != 0 or receipt["kind"] != METHOD or receipt["conditions"] != list(ARMS)
            or any(receipt[key] != value for key, value in verify_received(raw).items())):
        raise ValueError("No complete certified DuoduoCLIP archive")
    embeddings = raw / "external" / METHOD
    staged = control / "staged"
    staged.mkdir()
    shutil.copytree(run / "frozen", staged / "frozen")
    witness = {}
    for arm in ARMS:
        step(control, f"eval_{arm}", python("eval_retrieval_benchmark_fill.py", "--runtime", staged,
                                           "external", "--condition", arm, "--method", METHOD,
                                           "--input", embeddings))
        witness[arm] = audit(staged, embeddings, arm)
    write_status(control / "duoduoclip_independent_audit.json", all_passed=True,
                 method=METHOD, conditions=witness)
    write_status(control / "current.json", phase="preceding_table_publications", state="waiting")
    tail_state = run / "control/finish_external_native_tail/current.json"
    deadline = time.time() + 33 * 3600
    while time.time() < deadline:
        state = json.loads(tail_state.read_bytes())
        if state["state"] == "completed":
            break
        if state["state"] == "failed":
            raise RuntimeError("The preceding table publication needs recovery")
        time.sleep(45)
    else:
        raise TimeoutError("The preceding tables have not settled")
    for arm in ARMS:
        result = staged / "evaluation" / arm / METHOD
        target = run / "evaluation" / arm / METHOD
        if (target.exists() or sha256(result / "metrics.json") != witness[arm]["metric_sha256"]
                or sha256(result / "query_ranks.jsonl") != witness[arm]["rank_sha256"]):
            raise ValueError(f"DuoduoCLIP results or destination changed before publication: {arm}")
    for arm in ARMS:
        (staged / "evaluation" / arm / METHOD).rename(run / "evaluation" / arm / METHOD)
    if not publish(run, reports, preview, control, METHOD):
        raise ValueError("Missing earlier independent rank evidence")
    write_status(control / "current.json", phase=METHOD, state="completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "reports", "preview"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--attempt", default="first")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.attempt):
        parser.error("Attempt must be a simple lower-case identifier")
    suffix = "" if args.attempt == "first" else "_" + args.attempt
    control = args.runtime / "control" / ("finish_duoduoclip_full" + suffix)
    control.mkdir()
    try:
        finish(args.runtime, args.reports, args.preview, control,
               args.collection or args.runtime / "received/duoduoclip_full")
    except (OSError, RuntimeError, ValueError) as error:
        write_status(control / "current.json", state="failed", error=str(error))
        raise


if __name__ == "__main__":
    for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "4"
    main()
