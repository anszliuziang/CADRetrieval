"""Collect, verify and publish the already-launched CT–DeepCAD and L2 conditions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from collect_retrieval_benchmark_fill import verify_received


HERE = Path(__file__).resolve().parent
ARMS = ("ct_deepcad", "l2")
TABLE_FILES = ("RETRIEVAL_COMPARISON_TABLES.md", "RETRIEVAL_COMPARISON_TABLES.tex",
               "RETRIEVAL_TABLE_EVIDENCE.json")


def write_status(path: Path, **data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"utc_epoch": time.time(), **data}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def step(logs: Path, name: str, argv: list, *, check=True) -> int:
    argv = [str(arg) for arg in argv]
    start = time.time()
    write_status(logs / "current.json", phase=name, state="running", argv=argv)
    with (logs / f"{name}.stdout.log").open("xb") as out, (logs / f"{name}.stderr.log").open("xb") as err:
        proc = subprocess.run(argv, stdout=out, stderr=err, check=False)
    write_status(logs / f"{name}.receipt.json", phase=name, argv=argv,
                 started_epoch=start, actual_exit=proc.returncode)
    if proc.returncode and check:
        raise RuntimeError(f"{name} exited {proc.returncode}")
    return proc.returncode


def python(script: str, *args) -> list:
    return [sys.executable, "-B", str(HERE / script), *args]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_generator(runtime: Path, raw: Path, arm: str, output: Path) -> None:
    """Reconstruct all ranks by sorting rows directly from the original arrays."""
    import numpy as np

    frozen = runtime / "frozen"
    cad_axis = np.array([row["cad_index"] for row in read_rows(frozen / "gallery.jsonl")])
    positions = {int(cad): i for i, cad in enumerate(cad_axis)}
    background = np.load(frozen / "background.npy", mmap_mode="r", allow_pickle=False)
    queries_by_text = defaultdict(list)
    for query in read_rows(frozen / f"queries_{arm}.jsonl"):
        queries_by_text[query["text_index"]].append(query)
    metrics = {}
    for method in ("sum", "s48"):
        evaluation = runtime / "evaluation" / arm / method
        results = read_rows(evaluation / "query_ranks.jsonl")
        saved = {row["uid"]: row["rank"] for row in results}
        recomputed = {}
        for text_index, queries in queries_by_text.items():
            row = np.load(raw / "scores" / arm / f"t{text_index:05d}.npy", allow_pickle=False)
            if method == "s48":
                row = row - background
            order = np.lexsort((cad_axis, -row))
            inverse = np.empty(len(order), dtype=np.int32)
            inverse[order] = np.arange(1, len(order) + 1)
            for query in queries:
                recomputed[query["uid"]] = int(inverse[positions[query["cad_index"]]])
        if len(results) != len(recomputed) or saved != recomputed:
            raise ValueError(f"independent rank comparison failed: {arm}/{method}")
        info = json.loads((evaluation / "metrics.json").read_bytes())
        ranks = np.array(list(recomputed.values()))
        hits = [int(np.count_nonzero(ranks <= k)) for k in (1, 5, 10, 20)]
        if hits != info["hits"] or np.median(ranks) != info["median_rank"]:
            raise ValueError(f"independent metrics comparison failed: {arm}/{method}")
        metrics[method] = {"n_queries": len(ranks), "hits": hits, "median_rank": info["median_rank"],
                           "rank_sha256": digest(evaluation / "query_ranks.jsonl"),
                           "metric_sha256": digest(evaluation / "metrics.json")}
    write_status(output, condition=arm, independent_checks_passed=True, methods=metrics)


def unaudited_rows(runtime: Path) -> list[str]:
    control = runtime / "control"
    witnesses = list(control.glob("finish_first_batch*/audit_*.json"))
    witnesses += list(control.glob("finish_first_batch*/*_independent_audit.json"))
    witnesses += list(control.glob("independent_*_rank_audit.json"))
    pins = defaultdict(set)
    for path in witnesses:
        report = json.loads(path.read_bytes())
        if report.get("independent_checks_passed") is True:
            rows = {(report["condition"], m): row for m, row in report["methods"].items()}
        elif report.get("all_passed") is True:
            rows = {(arm, report["method"]): row for arm, row in report["conditions"].items()}
        else:
            continue
        for key, row in rows.items():
            pins[key].add((row["metric_sha256"], row["rank_sha256"]))
    absent = []
    for arm in ARMS:
        for method in ("clip", "openshape", "mmcad_v2", "sum", "s48"):
            root = runtime / "evaluation" / arm / method
            if (root / "metrics.json").is_file():
                actual = (digest(root / "metrics.json"), digest(root / "query_ranks.jsonl"))
                if actual not in pins[(arm, method)]:
                    absent.append(f"{arm}/{method}")
    return absent


def publish(runtime: Path, reports: Path, preview: Path, logs: Path, label: str) -> bool:
    pending = unaudited_rows(runtime)
    if pending:
        write_status(logs / f"publication_held_{label}.json", missing_independent_rows=pending)
        return False
    evaluator = "eval_retrieval_benchmark_fill.py"
    step(logs, f"summary_{label}", python(evaluator, "--runtime", runtime, "summarize"))
    build = logs / f"preview_{label}"
    build.mkdir()
    original = {name: (reports / name).read_bytes() for name in TABLE_FILES}
    staged_tables = build / "staged"
    step(logs, f"prepare_publication_{label}", python("publish_retrieval_benchmark_fill.py",
                                                     "--runtime", runtime, "--source", reports,
                                                     "--output", staged_tables))
    source = (staged_tables / "RETRIEVAL_COMPARISON_TABLES.tex").read_text(encoding="utf-8")
    if source.count(r"\end{table*}") != 3:
        raise ValueError("LaTeX preview requires exactly three tables")
    (build / "tables.tex").write_text(source.replace(r"\end{table*}", "\\end{table*}\n\\clearpage"),
                                     encoding="utf-8")
    entry = build / "preview.tex"
    entry.write_text(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}
\usepackage{booktabs}
\begin{document}
\input{tables.tex}
\end{document}
""", encoding="ascii")
    step(logs, f"latex_{label}", ["tectonic", "--only-cached", "--keep-logs", "--outdir", build, entry])
    pdf = build / "preview.pdf"
    page_info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    if not any(line.split() == ["Pages:", "3"] for line in page_info.splitlines()):
        raise ValueError("expected one PDF preview page per table")
    if any((reports / name).read_bytes() != data for name, data in original.items()):
        raise ValueError("report source was edited while the verified preview was building")
    for name, data in original.items():
        (build / ("previous_" + name)).write_bytes(data)
    if preview.exists():
        shutil.copyfile(preview, build / "previous_preview.pdf")
    for source_file, destination in [(staged_tables / name, reports / name) for name in TABLE_FILES] + [(pdf, preview)]:
        staged = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(source_file, staged)
        staged.replace(destination)
    write_status(build / "preview_receipt.json", pages=3, pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--last-ssh-start", type=float, required=True)
    parser.add_argument("--attempt", default="first")
    parser.add_argument("--external-collection", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.attempt):
        parser.error("attempt must be a simple lower-case identifier")
    run = args.runtime
    suffix = "" if args.attempt == "first" else "_" + args.attempt
    logs = run / "control" / ("finish_first_batch" + suffix)
    logs.mkdir()
    try:
        wait_and_finish(run, args, logs)
    except (OSError, RuntimeError, ValueError) as error:
        write_status(logs / "current.json", state="failed", error=str(error))
        raise


def wait_and_finish(run: Path, args, logs: Path) -> None:
    evaluation = "eval_retrieval_benchmark_fill.py"
    external = args.external_collection or run / "received/openshape_mmcad_first_batch"
    external_receipt = external / "pull_receipt.json"
    write_status(logs / "current.json", state="waiting", phase="external_transfer")
    external_deadline = time.time() + 3600
    while not external_receipt.exists() and time.time() < external_deadline:
        time.sleep(20)
    external_ok = complete_pull(external, "external", list(ARMS), ["openshape", "mmcad_v2"])
    if external_ok:
        for method in ("openshape", "mmcad_v2"):
            source = external / "raw/external" / method
            for arm in ARMS:
                step(logs, f"eval_{method}_{arm}", python(evaluation, "--runtime", run, "external",
                                                        "--condition", arm, "--method", method, "--input", source))
            step(logs, f"audit_{method}", [sys.executable, "-B", run / "control/audit_external_ranks.py",
                                         run, method, source, logs / f"{method}_independent_audit.json"])
        released = publish(run, args.reports, args.preview, logs, "external")
        write_status(logs / "external_ready.json", all_passed=True, arms=list(ARMS), tables_updated=released)
    else:
        write_status(logs / "external_incomplete.json", all_passed=False, receipt=str(external_receipt))

    last_ssh = args.last_ssh_start
    for arm in ARMS:
        original = run / "received" / f"text2cad_{arm}_first_batch"
        prior = [original, *sorted(original.parent.glob(original.name + "_*"))]
        location = next((p for p in prior if complete_pull(p, "text2cad", [arm], None)), None)
        if location is None:
            location = original
            if original.exists():
                if args.attempt == "first":
                    raise ValueError(f"existing incomplete collection must be preserved: {location}")
                location = original.with_name(original.name + "_" + args.attempt)
            while time.time() < last_ssh + 65:
                time.sleep(min(30, last_ssh + 65 - time.time()))
            last_ssh = time.time()
            step(logs, f"collect_text2cad_{arm}", python("collect_retrieval_benchmark_fill.py", "text2cad",
                                                       "--condition", arm, "--wait-seconds", 112000,
                                                       "--output", location))
            if not complete_pull(location, "text2cad", [arm], None):
                raise ValueError(f"collection receipt did not certify {arm}")
        raw = location / "raw/full"
        step(logs, f"eval_text2cad_{arm}", python(evaluation, "--runtime", run, "text2cad",
                                                 "--condition", arm, "--input", raw))
        audit_generator(run, raw, arm, logs / f"audit_{arm}.json")
        released = publish(run, args.reports, args.preview, logs, arm)
        if arm == "l2" and not released:
            raise ValueError("first batch is scored but an independent audit required for publication is pending")
    write_status(logs / "current.json", state="completed" if external_ok else "partial_external",
                 phase="first_batch", external_all_passed=external_ok, generator_all_passed=True)


def complete_pull(directory: Path, kind: str, arms: list[str], methods) -> bool:
    path = directory / "pull_receipt.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_bytes())
    if receipt["actual_ssh_exit"] != 0:
        return False
    if receipt["kind"] != kind or receipt["conditions"] != arms or receipt["methods"] != methods:
        raise ValueError(f"collection receipt declares a different stage: {directory}")
    actual = verify_received(directory / "raw")
    if any(receipt[key] != value for key, value in actual.items()):
        raise ValueError(f"collection receipt changed: {directory}")
    return True


if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "4"
    main()
