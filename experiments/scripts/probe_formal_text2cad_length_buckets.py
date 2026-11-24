#!/usr/bin/env python3
"""Compare cached full-width and bucketed Text2CAD on one frozen validation query."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter


CASES = ("L0", "L1", "L2", "L3", "CT")
BANDS = {"short": (1, 4), "median": (1, 2), "long": (7, 8)}
CAPS = (32, 64, 96, 128, 160, 192, 224, 256, 272)
GALLERY_SIZE = 2048
SCORES = ("score_total", "score_x", "score_y")
COUNTS = ("n_valid_total", "n_valid_x", "n_valid_y", "valid_seq_len", "n_real_tokens")
ATOL = 0.001
RTOL = 1e-6


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def read_validation(args, receipt_sha):
    from experiments.src.idea01.in_domain_pilot import INPUT_HASHES

    source = "ct-deepcad" if args.case == "CT" else "t2cad-v1.0"
    receipt = (args.manifests / "manifest_receipt.json").read_bytes()
    verify(sha256(receipt).hexdigest() == receipt_sha, "Validation manifest receipt changed")
    pins = json.loads(receipt)["file_sha256"]
    files = {f"{source}/{name}": pins[f"{source}/{name}"]
             for name in ("validation_targets.jsonl", "validation_gallery.jsonl")}
    for name, expected in files.items():
        verify(file_hash(args.manifests / name) == expected, f"Frozen validation input changed: {name}")
    targets, gallery = (read_rows(args.manifests / name) for name in files)
    verify(len(targets) == len({row["uid"] for row in targets}) == 512, "Incomplete validation target axis")
    verify(len(gallery) == len({row["uid"] for row in gallery}) == len({row["cad_member"] for row in gallery})
           == GALLERY_SIZE and [row["cad_index"] for row in gallery] == list(range(GALLERY_SIZE)),
           "Expected the complete ordered 2,048-candidate validation gallery")
    official_zip_sha = file_hash(args.cad_zip)
    verify(official_zip_sha == INPUT_HASHES["official_zip"], "Official CAD zip changed")
    return source, targets, gallery, files, official_zip_sha


def select_text(targets, case, band, embedder):
    rows = []
    overlength = 0
    for target in targets:
        text = target["texts"][case]
        tokens = embedder.tokenizer(text, truncation=False, verbose=False)
        width = len(tokens["input_ids"])
        if width > embedder.max_seq_len:
            overlength += 1
        else:
            rows.append((width, target["uid"], text, dict(tokens)))
    verify(bool(rows), f"Every frozen {case} validation text exceeds the complete input window")
    rows.sort(key=lambda row: (row[0], row[1]))
    numerator, denominator = BANDS[band]
    index = (len(rows) - 1) * numerator // denominator
    width, uid, text, tokens = rows[index]
    return {
        "case": case, "band": band, "eligible_sorted_index": index, "query_uid": uid, "text": text,
        "text_sha256": sha256(text.encode()).hexdigest(), "full_input_tokens": width,
        "effective_tokenizer_sha256": sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest(),
        "excluded_over_window": overlength, "eligible_text_rows": len(rows), "window": embedder.max_seq_len,
    }


def score_batch(model, store, indices, prepared, device, n_rows):
    import torch

    from experiments.src.idea01.text2cad_frozen_runner import assemble_batch
    from experiments.src.idea01.text2cad_port.model import (
        build_teacher_forcing_inputs, prepare_cross_attention_mask_batch,
    )
    from experiments.src.idea01.text2cad_vector_scorer import two_head_factorized_score

    start = perf_counter()
    with torch.inference_mode():
        # Assemble with the audited CPU path, then transfer the same tensors and
        # canonical mask submatrix sliced to the registered bin width.
        host_vec, host_mask, _ = assemble_batch(store, indices, torch.device("cpu"))
        labels = host_vec["cad_vec"][:, :n_rows].contiguous()
        padding = host_mask["key_padding_mask"][:, :n_rows].contiguous()
        vec = {name: value[:, :n_rows].contiguous().to(device) for name, value in host_vec.items()}
        masks = {"key_padding_mask": padding.to(device),
                 "attn_mask": host_mask["attn_mask"][:n_rows - 1, :n_rows - 1].contiguous().to(device)}
        torch.npu.synchronize(device)
        transfer_done = perf_counter()
        v, m, _, _ = build_teacher_forcing_inputs(vec, masks)
        repeated_text = prepared[0].expand(len(indices), -1, -1).contiguous()
        kpm = prepared[1].expand(len(indices), -1).contiguous()
        ca = {"attn_mask": prepare_cross_attention_mask_batch(kpm, cad_seq_len=n_rows - 1), "key_padding_mask": kpm}
        pred, _ = model.cad_decoder(v, repeated_text, m, ca, False)
        verify(pred.dtype == torch.float32, f"CAD logits changed precision: {pred.dtype}")
        torch.npu.synchronize(device)
        decoder_done = perf_counter()
        logits = pred.cpu()
        torch.npu.synchronize(device)
        copy_done = perf_counter()
        result = two_head_factorized_score(logits, labels, padding)
        done = perf_counter()
    return result, {
        "assemble_and_transfer_seconds": transfer_done - start, "decoder_seconds": decoder_done - transfer_done,
        "logits_to_cpu_seconds": copy_done - decoder_done, "cpu_score_seconds": done - copy_done, "wall_seconds": done - start,
    }


def run_arm(output, arm, model, store, axis, prepared, device, batch_size):
    rows_by_width = defaultdict(list)
    for index, cad in enumerate(axis):
        width = CAPS[-1] if arm == "reference" else CAPS[bisect_left(CAPS, cad["n_real_tokens"])]
        rows_by_width[width].append(index)
    received = {}
    bins = []
    seen_shapes = set()
    with (output / f"{arm}_batches.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for width, positions in sorted(rows_by_width.items()):
            totals = Counter()
            cold = Counter()
            batches = 0
            for offset in range(0, len(positions), batch_size):
                indices = positions[offset:offset + batch_size]
                first = (width, len(indices)) not in seen_shapes
                seen_shapes.add((width, len(indices)))
                event = {"arm": arm, "input_cad_rows": width, "original_cad_indices": indices,
                         "pairs": len(indices), "phase": "first_shape" if first else "steady"}
                try:
                    scores, timings = score_batch(model, store, indices, prepared, device, width)
                    verify(len(scores) == len(indices), "Scorer returned the wrong batch count")
                    for index, score in zip(indices, scores, strict=True):
                        verify(index not in received and score["n_real_tokens"] == axis[index]["n_real_tokens"],
                               f"Effective row count or writeback axis changed at CAD index {index}")
                        received[index] = {"cad_index": index, "uid": axis[index]["uid"], **score}
                except Exception as exc:
                    log.write(json.dumps({**event, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                    log.flush()
                    raise
                log.write(json.dumps({**event, **timings}, allow_nan=False) + "\n")
                log.flush()
                batches += 1
                totals.update(timings)
                if first:
                    cold.update(timings)
            entry = {"input_cad_rows": width, "batches": batches, "pairs": len(positions),
                     "min_real_cad_rows": min(axis[index]["n_real_tokens"] for index in positions),
                     "max_real_cad_rows": max(axis[index]["n_real_tokens"] for index in positions),
                     "timing": dict(totals), "first_shape_timing": dict(cold),
                     "pairs_per_s": len(positions) / totals["wall_seconds"]}
            bins.append(entry)
            print(json.dumps({"stage": "cad_bucket", "arm": arm, **entry}, allow_nan=False), flush=True)
    verify(received.keys() == set(range(GALLERY_SIZE)), f"{arm} did not fill the frozen gallery axis")
    write_rows(output / f"{arm}_scores.jsonl", (received[index] for index in range(GALLERY_SIZE)))
    return bins


def compare(output, reference_dir, bucketed_dir):
    identity = (reference_dir / "inputs.json").read_bytes()
    verify(identity == (bucketed_dir / "inputs.json").read_bytes(), "Arms did not use identical frozen inputs")
    inputs = json.loads(identity)
    verify([row["cad_index"] for row in inputs["cad_axis"]] == list(range(GALLERY_SIZE)),
           "Comparison needs the full original candidate axis")
    models, arrays = [], []
    for arm, directory in (("reference", reference_dir), ("bucketed", bucketed_dir)):
        source = directory / f"{arm}_scores.jsonl"
        run = json.loads((directory / f"{arm}_run.json").read_bytes())
        verify(run["arm"] == arm and run["inputs_sha256"] == sha256(identity).hexdigest()
               and run["scores_sha256"] == file_hash(source)
               and run["batches_sha256"] == file_hash(directory / f"{arm}_batches.jsonl"),
               f"{arm} result receipt changed")
        rows = read_rows(source)
        verify([(row["cad_index"], row["uid"]) for row in rows]
               == [(row["cad_index"], row["uid"]) for row in inputs["cad_axis"]], f"{arm} score ordering changed")
        models.append(run)
        arrays.append(rows)
    reference, bucketed = arrays
    checks = []
    failures = Counter({name: 0 for name in SCORES})
    maxima = {name: 0.0 for name in SCORES}
    counts_changed = 0
    for left, right in zip(reference, bucketed, strict=True):
        cells = {}
        for name in SCORES:
            a, b = left[name], right[name]
            finite = math.isfinite(a) and math.isfinite(b)
            delta = abs(a - b) if finite else None
            tolerance = max(ATOL, RTOL * abs(a)) if math.isfinite(a) else None
            passed = finite and delta <= tolerance
            failures[name] += not passed
            if finite:
                maxima[name] = max(maxima[name], delta)
            cells[name] = {"finite": finite, "delta": delta, "tolerance": tolerance, "pass": passed}
        counts = {name: [left[name], right[name]] for name in COUNTS}
        counts_match = all(a == b for a, b in counts.values())
        counts_changed += not counts_match
        checks.append({"cad_index": left["cad_index"], "uid": left["uid"], "criteria": cells,
                       "counts": counts, "counts_match": counts_match})

    write_rows(output / "cell_comparison.jsonl", checks)
    totals = [[row["score_total"] for row in rows] for rows in arrays]
    verify(all(math.isfinite(value) for scores in totals for value in scores), "Cannot rank nonfinite totals")
    old, new = [sorted(range(GALLERY_SIZE), key=lambda index: (-values[index], index)) for values in totals]
    axes = inputs["cad_axis"]

    def tied_pairs(values):
        return sum(n * (n - 1) // 2 for n in Counter(values).values())

    def tie_groups(values):
        groups = defaultdict(list)
        for index, value in enumerate(values):
            groups[value].append(index)
        return {tuple(group) for group in groups.values() if len(group) > 1}

    reference_ties, bucketed_ties = map(tied_pairs, totals)
    shared_ties = tied_pairs(zip(*totals, strict=True))
    reference_groups, bucketed_groups = map(tie_groups, totals)
    write_rows(output / "rank_order.jsonl", (
        {"rank": rank, "reference_cad_index": i, "bucketed_cad_index": j,
         "reference_uid": axes[i]["uid"], "bucketed_uid": axes[j]["uid"]}
        for rank, (i, j) in enumerate(zip(old, new, strict=True), 1)
    ))
    passed = not any(failures.values()) and not counts_changed and reference_ties == shared_ties == bucketed_ties and old == new
    comparison = {
        "scope": "one complete query on the registered validation gallery", "inputs_sha256": sha256(identity).hexdigest(),
        "query": inputs["query"], "candidates": GALLERY_SIZE, "absolute_tolerance_floor": ATOL,
        "relative_tolerance": RTOL, "score_failures": dict(failures), "max_absolute_score_delta": maxima,
        "counts_changed_candidates": counts_changed, "reference_tied_candidate_pairs": reference_ties,
        "bucketed_tied_candidate_pairs": bucketed_ties, "tie_pairs_broken": reference_ties - shared_ties,
        "tie_pairs_created": bucketed_ties - shared_ties, "whole_order_equal": old == new,
        "reference_only_tie_groups_cad_indices": sorted(reference_groups - bucketed_groups),
        "bucketed_only_tie_groups_cad_indices": sorted(bucketed_groups - reference_groups),
        "different_rank_positions": sum(a != b for a, b in zip(old, new, strict=True)),
        "top20_order_equal": old[:20] == new[:20], "top20_same_members": set(old[:20]) == set(new[:20]),
        "lost_top20_uids": [axes[i]["uid"] for i in old[:20] if i not in new[:20]],
        "gained_top20_uids": [axes[i]["uid"] for i in new[:20] if i not in old[:20]],
        "boundary_reference_gap": totals[0][old[19]] - totals[0][old[20]],
        "boundary_bucketed_gap": totals[1][new[19]] - totals[1][new[20]],
        "runtime_by_arm": {record["arm"]: record["runtime"] for record in models},
        "passed": passed,
    }
    write_json(output / "comparison.json", comparison)
    print(json.dumps({key: comparison[key] for key in ("passed", "whole_order_equal", "top20_same_members",
                                                        "score_failures", "counts_changed_candidates")}), flush=True)
    return 0 if passed else 2


def run(args):
    import torch
    import torch_npu

    from experiments.scripts.probe_910a_text2cad_shared_text import shared_embedding
    from experiments.scripts.probe_formal_text2cad_cost import RECEIPT, model_loaded
    from experiments.src.idea01.in_domain_pilot_native import (
        _tensor_candidate_hashes, check_text2cad_flags, digest_tensor,
    )
    from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore
    from experiments.src.idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256, PINNED_CHECKPOINT_SHA256

    started = perf_counter()
    source, targets, gallery, manifest_files, zip_sha = read_validation(args, RECEIPT)
    manifest_done = perf_counter()
    torch.set_num_threads(args.cpu_threads)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device(args.device)
    torch.npu.set_device(device)
    model, provenance = model_loaded(args, device)
    torch.npu.synchronize(device)
    model_done = perf_counter()
    query = select_text(targets, args.case, args.band, model.base_text_embedder)
    selected = perf_counter()
    store = ZipCandidateStore(args.cad_zip, [row["cad_member"] for row in gallery])
    preload = store.preload()
    verify(preload["n_loaded"] == len(gallery), f"Full validation gallery failed preload: {preload}")
    axis = []
    for index, record in enumerate(gallery):
        candidate = store.get(index)
        check_text2cad_flags(candidate)
        count = candidate["n_real_tokens"]
        verify(2 <= count <= CAPS[-1], f"Invalid native length for {record['uid']}")
        axis.append({
            **{key: record[key] for key in ("cad_index", "uid", "cad_member", "member_sha256")},
            "effective_fields": _tensor_candidate_hashes(candidate, store.canonical_attn_mask),
            "n_real_tokens": count,
        })
    cad_done = perf_counter()
    shared = shared_embedding(model, query["text"])
    torch.npu.synchronize(device)
    text_done = perf_counter()
    verify(shared[0].dtype == torch.float32 and shared[0].shape[1] == shared[1].shape[1] == query["full_input_tokens"]
           and shared[1].dtype == torch.bool and not bool(shared[1].any()), "Shared adaptive input changed")
    identity = {
        "scope": "frozen validation only", "source": source, "manifest_receipt_sha256": RECEIPT,
        "validation_axes_sha256": manifest_files, "official_cad_zip_sha256": zip_sha,
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256, "bert_files_sha256": PINNED_BERT_FILE_SHA256,
        "query": query, "cad_axis": axis, "cad_bucket_caps": CAPS, "batch_size": args.batch_size,
        "cached_adaptive_text_sha256": digest_tensor(shared[0]), "cached_padding_mask_sha256": digest_tensor(shared[1]),
        "cpu_threads": args.cpu_threads, "runtime_precision": "FP32, must_keep_origin_dtype",
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": file_hash(Path(__file__)),
    }
    input_path = args.out / "inputs.json"
    write_json(input_path, identity)
    input_sha = file_hash(input_path)
    print(json.dumps({"stage": "prepared", "source": source, "query_uid": query["query_uid"],
                      "text_tokens": query["full_input_tokens"], "candidates": len(axis),
                      "shared_text_preparation_seconds": text_done - cad_done}), flush=True)
    arms = ("reference", "bucketed") if args.arm == "both" else (args.arm,)
    for arm in arms:
        torch.npu.reset_peak_memory_stats(device)
        arm_started = perf_counter()
        bins = run_arm(args.out, arm, model, store, axis, shared, device, args.batch_size)
        arm_finished = perf_counter()
        runtime = {
            "validation_manifest_and_archive_check_seconds": manifest_done - started,
            "cold_model_load_seconds": model_done - manifest_done, "tokenized_query_selection_seconds": selected - model_done,
            "gallery_preload_and_identity_seconds": cad_done - selected, "shared_text_preparation_seconds": text_done - cad_done,
            "score_arm_wall_seconds": arm_finished - arm_started,
            "first_batch_shapes_seconds": sum(row["first_shape_timing"]["wall_seconds"] for row in bins),
            "total_decoder_seconds": sum(row["timing"]["decoder_seconds"] for row in bins),
            "total_cpu_score_seconds": sum(row["timing"]["cpu_score_seconds"] for row in bins),
            "peak_device_memory_bytes": torch.npu.max_memory_allocated(device),
            "device": str(device), "device_name": torch.npu.get_device_name(device),
            "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__), "buckets": bins,
        }
        write_json(args.out / f"{arm}_run.json", {
            "arm": arm, "inputs_sha256": input_sha, "scores_sha256": file_hash(args.out / f"{arm}_scores.jsonl"),
            "batches_sha256": file_hash(args.out / f"{arm}_batches.jsonl"), "model_assets": provenance, "runtime": runtime,
        })
    if args.arm == "both":
        return compare(args.out, args.out, args.out)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifests", "checkpoint", "bert", "cad-zip"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", choices=CASES, default="L0")
    parser.add_argument("--band", choices=tuple(BANDS), default="median")
    parser.add_argument("--arm", choices=("both", "reference", "bucketed"), default="both")
    parser.add_argument("--compare-runs", type=Path, nargs=2, metavar=("REFERENCE", "BUCKETED"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.compare_runs is None and any(getattr(args, name) is None for name in ("manifests", "checkpoint", "bert", "cad_zip")):
        parser.error("Scoring requires --manifests, --checkpoint, --bert and --cad-zip")
    if args.device.partition(":")[0] != "npu" or min(args.batch_size, args.cpu_threads) < 1:
        parser.error("Select an NPU and positive batch and CPU-thread counts")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        return compare(args.out, *args.compare_runs) if args.compare_runs is not None else run(args)
    except Exception as exc:
        write_json(args.out / "failure.json", {"type": type(exc).__name__, "detail": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
