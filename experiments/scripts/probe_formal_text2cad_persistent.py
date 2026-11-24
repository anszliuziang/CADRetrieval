#!/usr/bin/env python3
"""Compare natural and fixed Text2CAD layouts across successive validation queries."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from unittest.mock import patch

from experiments.scripts.probe_910a_text2cad_november_path import compare
from experiments.scripts.probe_910a_text2cad_shared_text import RECEIPT, model_loaded, shared_embedding
from experiments.scripts.probe_formal_text2cad_length_buckets import (
    BANDS, CAPS, CASES, file_hash, read_validation, score_batch, select_text, verify, write_json, write_rows,
)
from experiments.src.idea01.in_domain_pilot_native import check_text2cad_flags
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore


BATCH_SIZE = 16
TEXT_MULTIPLE = 32
LAYOUTS = {"natural": (False, False), "text_pad": (True, False),
           "tail_fill": (False, True), "fixed": (True, True)}


def gallery_batches(store, gallery):
    bins = defaultdict(list)
    for index, row in enumerate(gallery):
        cad = store.get(index)
        check_text2cad_flags(cad)
        length = cad["n_real_tokens"]
        verify(2 <= length <= CAPS[-1], f"Invalid native CAD length for {row['uid']}")
        bins[CAPS[bisect_left(CAPS, length)]].append(index)
    return [(width, indices[start:start + BATCH_SIZE]) for width, indices in sorted(bins.items())
            for start in range(0, len(indices), BATCH_SIZE)]


def prepare_text(model, query, fixed, device):
    import torch

    text = query["text"]
    embedder = model.base_text_embedder
    original = embedder.tokenizer
    length = query["full_input_tokens"]
    check = original([text], return_tensors="pt", padding=True, truncation=False)
    verify(check["input_ids"].shape[1] == length, "Declared and actual full text widths disagree")
    start = perf_counter()
    if fixed:
        verify(original.padding_side == "right", "Fixed layout requires right padding")

        def padded(*args, **kwargs):
            return original(*args, **kwargs, pad_to_multiple_of=TEXT_MULTIPLE)

        transformed = padded([text], return_tensors="pt", padding=True, truncation=False)
        verify(all(torch.equal(transformed[k][:, :length], check[k]) for k in check)
               and not bool(transformed["attention_mask"][:, length:].any()),
               "Fixed text layout changed an active token or its mask")
        with patch.object(embedder, "tokenizer", padded):
            prepared = shared_embedding(model, text)
    else:
        prepared = shared_embedding(model, text)
    torch.npu.synchronize(device)
    width = prepared[1].shape[1]
    expected_width = (length + TEXT_MULTIPLE - 1) // TEXT_MULTIPLE * TEXT_MULTIPLE if fixed else length
    verify(width == prepared[0].shape[1] == expected_width <= embedder.max_seq_len
           and prepared[0].dtype == torch.float32 and prepared[1].dtype == torch.bool
           and not bool(prepared[1][:, :length].any()) and bool(prepared[1][:, length:].all()),
           "Adaptive text output or padding mask differs from the declared layout")
    return prepared, width, perf_counter() - start


def run_query(model, store, batches, gallery, query, device, layout, seen, event_log):
    pad_text, fill_tail = LAYOUTS[layout]
    prepared, text_width, prep_seconds = prepare_text(model, query, pad_text, device)
    rows = {}
    first_seconds = steady_seconds = 0.0
    first_pairs = steady_pairs = dummies = 0
    for cad_width, indices in batches:
        dummy_count = (BATCH_SIZE - len(indices)) if fill_tail else 0
        physical = [*indices, *([indices[-1]] * dummy_count)]
        key = (text_width, cad_width, len(physical))
        first = key not in seen
        scores, timing = score_batch(model, store, physical, prepared, device, cad_width)
        verify(len(scores) == len(physical), "Wrong physical batch width")
        seen.add(key)
        for index, score in zip(indices, scores[:len(indices)], strict=True):
            verify(index not in rows and score["n_real_tokens"] == store.get(index)["n_real_tokens"],
                   f"CAD index order or effective length changed at {index}")
            rows[index] = {"cad_index": index, "uid": gallery[index]["uid"], **score}
        event = {"query_uid": query["query_uid"], "layout": layout,
                 "shape": key, "first_shape_in_process": first, "real_pairs": len(indices),
                 "dummy_lanes": dummy_count, **timing}
        event_log.write(json.dumps(event, allow_nan=False) + "\n")
        event_log.flush()
        if first:
            first_seconds += timing["wall_seconds"]
            first_pairs += len(indices)
        else:
            steady_seconds += timing["wall_seconds"]
            steady_pairs += len(indices)
        dummies += dummy_count
    verify(rows.keys() == set(range(len(gallery))), "Score writeback left a gap in the validation gallery")
    runtime = {"text_width": text_width, "text_prep_seconds": prep_seconds,
               "first_shape_pairs": first_pairs, "first_shape_seconds": first_seconds,
               "warm_pairs": steady_pairs, "warm_seconds": steady_seconds, "dummy_lanes": dummies,
               "warm_pairs_per_second": steady_pairs / steady_seconds if steady_seconds else None}
    return [rows[index] for index in range(len(gallery))], runtime


def tie_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["score_total"]].append(row["cad_index"])
    return {tuple(group) for group in groups.values() if len(group) > 1}


def compare_results(reference, fixed, axis):
    result = compare(reference, fixed, axis)
    exact_ties = tie_groups(reference) == tie_groups(fixed)
    return {**result, "same_exact_ties": exact_ties, "passed": result["passed"] and exact_ties}


def main():
    import torch
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "checkpoint", "bert", "cad-zip", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--band", choices=tuple(BANDS), action="append")
    parser.add_argument("--layout", choices=(*LAYOUTS, "both", "all"), default="both")
    args = parser.parse_args()
    bands = args.band or list(BANDS)
    if len(set(bands)) != len(bands):
        parser.error("Bands must be distinct")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Use a clean deployed Git revision")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        source, targets, gallery, manifest_files, zip_sha = read_validation(args, RECEIPT)
        start = perf_counter()
        torch.set_num_threads(2)
        torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        model, provenance = model_loaded(args, device)
        gallery_axis = [row["cad_index"] for row in gallery]
        store = ZipCandidateStore(args.cad_zip, [row["cad_member"] for row in gallery])
        verify(store.preload()["n_loaded"] == len(gallery), "Validation gallery preload incomplete")
        batches = gallery_batches(store, gallery)
        queries = [select_text(targets, args.case, band, model.base_text_embedder) for band in bands]
        verify(len({query["text"] for query in queries}) == len(queries), "Select distinct complete validation texts")
        if args.layout == "all":
            arms = tuple(LAYOUTS)
        else:
            arms = ("natural", "fixed") if args.layout == "both" else (args.layout,)
        comparing = len(arms) > 1
        initial_seconds = perf_counter() - start
        seen = set()
        results = []
        print(json.dumps({"source": source, "level": args.case, "gallery": len(gallery),
                          "n_queries": len(queries), "initial_seconds": initial_seconds}, allow_nan=False), flush=True)
        with (args.out / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as log:
            for index, query in enumerate(queries):
                output = args.out / f"query_{index:02d}"
                output.mkdir()
                rows_by_arm = {}
                timings = {}
                hashes = {}
                for arm in arms:
                    rows, timings[arm] = run_query(model, store, batches, gallery, query, device, arm, seen, log)
                    rows_by_arm[arm] = rows
                    scores = output / f"{arm}_scores.jsonl"
                    write_rows(scores, rows)
                    hashes[arm] = file_hash(scores)
                result = {"query_uid": query["query_uid"], "band": query["band"], "text_sha256": query["text_sha256"],
                          "query_tokens": query["full_input_tokens"], "timings": timings, "raw_score_file_sha256": hashes}
                if comparing:
                    result["comparisons"] = {arm: compare_results(rows_by_arm["natural"], rows_by_arm[arm], gallery_axis)
                                             for arm in arms[1:]}
                write_json(output / "receipt.json", result)
                results.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
        passed = comparing and all(check["passed"] for result in results for check in result["comparisons"].values())
        identity = [[row["cad_index"], row["uid"], row["cad_member"]] for row in gallery]
        write_json(args.out / "summary.json", {
            "scope": "published-weight validation layout qualification", "source": source, "case": args.case,
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "manifest_files": manifest_files, "cad_zip_sha256": zip_sha, "published_model_assets": provenance,
            "gallery_size": len(gallery), "gallery_axis_sha256": sha256(json.dumps(identity).encode()).hexdigest(),
            "layout": args.layout, "shape_count": len(seen), "initial_seconds": initial_seconds,
            "peak_device_memory_bytes": torch.npu.max_memory_allocated(device), "queries": results,
            "arm_order": arms, "shape_history_shared_between_arms": comparing,
            "background_correction_compared": False, "layout_parity_passed": passed,
            "elapsed_seconds_after_input_verification": perf_counter() - start,
        })
        return 0 if not comparing or passed else 2
    except Exception as exc:
        write_json(args.out / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
