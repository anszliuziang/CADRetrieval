#!/usr/bin/env python3
"""Time the pinned November BERT-cache path and the post-adaptive cache on identical validation inputs."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch
import torch_npu

from experiments.scripts.probe_910a_text2cad_shared_text import (
    RECEIPT, model_loaded, raw_reference, score_with_shared_text, shared_embedding, verify,
)
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
from experiments.src.idea01.text2cad_frozen_runner import (
    ZipCandidateStore, assemble_batch, forward_and_score_batch,
)
from experiments.src.idea01.text2cad_port.model import teacher_forced_logits
from experiments.src.idea01.text2cad_vector_scorer import two_head_factorized_score

NOVEMBER = "6cd791312eefa5e575b0e56a39553fa9f92fbd3b"
CORE = (
    "experiments/src/idea01/text2cad_frozen_runner.py",
    "experiments/src/idea01/text2cad_port/model.py",
    "experiments/src/idea01/text2cad_vector_scorer.py",
)
WRAPPER = "experiments/scripts/run_retrieval_benchmark_text2cad.py"
WRAPPER_SHA = "f2008257343b628826f6b5c1e558e066fdca8834823e613adcc526d4750f202a"
SCORES = ("score_x", "score_y", "score_total", "mean_x", "mean_y", "mean_total")
COUNTS = ("n_valid_x", "n_valid_y", "n_valid_total", "n_real_tokens")


def historical_source():
    def blob(path):
        return subprocess.check_output(["git", "show", f"{NOVEMBER}:{path}"])

    core = {path: pilot.sha256_bytes(blob(path)) for path in CORE}
    verify(all(pilot.sha256_file(Path(path)) == digest for path, digest in core.items()),
           "The original November scorer core and the clean current checkout differ")
    wrapper = blob(WRAPPER)
    verify(pilot.sha256_bytes(wrapper) == WRAPPER_SHA, "Unexpected November BERT-cache source")
    factory, = (node for node in ast.parse(wrapper).body
                if isinstance(node, ast.FunctionDef) and node.name == "install_same_batch_cache")
    scope = {}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), WRAPPER, "exec"), scope)
    return scope[factory.name], core


class Timings:
    def __init__(self, device):
        self.device = device
        self.seconds = {}

    def call(self, name, fn, *args, **kwargs):
        torch.npu.synchronize(self.device)
        started = time.monotonic()
        result = fn(*args, **kwargs)
        torch.npu.synchronize(self.device)
        self.seconds[name] = self.seconds.get(name, 0.0) + time.monotonic() - started
        return result

    def watch(self, stack, obj, method, name):
        original = getattr(obj, method)

        def logged(*args, **kwargs):
            return self.call(name, original, *args, **kwargs)

        stack.enter_context(patch.object(obj, method, logged))


@torch.inference_mode()
def profile_november(model, store, indices, texts, device):
    """Instrument the audited runner operations and check the result against its actual entry."""
    timing = Timings(device)
    vec, mask, width = timing.call("input_assembly", assemble_batch, store, indices, device, "none")
    with ExitStack() as stack:
        timing.watch(stack, model.base_text_embedder, "get_embedding", "bert_cache_lookup")
        timing.watch(stack, model.adaptive_layer, "forward", "adaptive")
        timing.watch(stack, model.cad_decoder, "forward", "cad_decoder")
        pred, _, _ = timing.call("teacher_forced_model", teacher_forced_logits,
                                 model, texts, vec, mask, cad_seq_len=width)
    verify(pred.dtype == torch.float32, "November model logits are not FP32")
    inputs = timing.call("copy_to_cpu", lambda: (pred.to(torch.float32).cpu(), vec["cad_vec"].cpu(),
                                                mask["key_padding_mask"].cpu()))
    rows = timing.call("cpu_scoring", two_head_factorized_score, *inputs)
    return rows, timing.seconds


def compare(reference, rows, axis):
    deltas = dict.fromkeys(SCORES, 0.0)
    passed = len(reference) == len(rows) == len(axis)
    for expected, actual in zip(reference, rows, strict=True):
        for key in COUNTS:
            passed &= actual[key] == expected[key]
        for key in SCORES:
            x, y = actual[key], expected[key]
            if x is None or y is None:
                passed &= x is y
            else:
                delta = abs(x - y)
                deltas[key] = max(deltas[key], delta)
                passed &= math.isfinite(x) and math.isfinite(y) and delta <= max(0.001, 1e-6 * abs(y))

    def order(result):
        return sorted(range(len(axis)), key=lambda i: (-result[i]["score_total"], axis[i]))

    order_equal = order(reference) == order(rows)
    return {"passed": bool(passed and order_equal), "max_abs_delta": deltas, "same_full_order": order_equal}


def probe(args, model, device, old_cache, tensors, known):
    ref, reference_sha = raw_reference(args.reference, args.level, "full_steady1")
    paths = (f"{ref['source']}/validation_targets.jsonl", f"{ref['source']}/validation_gallery.jsonl")
    verify(all(pilot.sha256_file(args.manifests / path) == known[path] for path in paths),
           "Frozen validation identity changed")
    targets, gallery = [{r["uid"]: r for r in pilot.read_jsonl(args.manifests / path)} for path in paths]
    text = targets[ref["validation_query_uid"]]["texts"][args.level]
    cad = [gallery[uid] for uid in ref["uids"]]
    members = [row["cad_member"] for row in cad]
    verify(len(set(members)) == len(members) == 16, "The original full batch must contain 16 distinct members")
    axis = [row["cad_index"] for row in cad]
    indices = list(range(len(axis)))
    store = ZipCandidateStore(args.cad_zip, members)
    verify(store.preload()["n_loaded"] == len(axis), "Could not preload all fixed validation CADs")
    backend = Text2CadReadout(model, store, dict(zip(members, indices, strict=True)), device)
    for candidate, scored in zip(cad, ref["scores"], strict=True):
        current = backend.prepare(candidate, text)
        verify(all(current[key] == scored[key] for key in ("input_sha256", "target_sha256", "query_tokens")),
               "Current input differs from the recorded native full batch")

    texts = [text] * len(axis)
    timings = Timings(device)
    original, cached = old_cache(model)
    model.base_text_embedder = cached
    old_preparation = Timings(device)
    old_preparation.call("bert_full_batch_once", cached.get_embedding, texts)
    old_rows = timings.call("november_cold", forward_and_score_batch, model, store, texts, indices, device)
    reps = []
    for attempt in range(2):
        rows = timings.call(f"november_steady_{attempt + 1}", forward_and_score_batch,
                            model, store, texts, indices, device)
        reps.append(compare(old_rows, rows, axis))
    old_profiled, old_phases = profile_november(model, store, indices, texts, device)
    model.base_text_embedder = original

    new_preparation = Timings(device)
    with ExitStack() as stack:
        new_preparation.watch(stack, model.base_text_embedder, "get_embedding", "bert_once")
        new_preparation.watch(stack, model.adaptive_layer, "forward", "adaptive_once")
        prepared = new_preparation.call("text_total", shared_embedding, model, text)
    rows_by_variant = {"previous_uncached_path": ref["scores"], "november_phased": old_profiled}
    for name in ("new_first", "new_steady_1", "new_steady_2"):
        rows_by_variant[name], used = timings.call(name, score_with_shared_text,
                                                   model, store, indices, prepared, device, "none")
        verify(used == 271, "The cache comparison must retain the full CAD width")
    checks = {name: compare(old_rows, rows, axis) for name, rows in rows_by_variant.items()}
    verify(cached.hits == 4, "Unexpected BERT cache hits for the historical full-batch path")
    return {
        "scope": "phase timing on one intact validation batch",
        "level": args.level, "text_tokens": ref["query_tokens"], "query_uid": ref["validation_query_uid"],
        "uids": ref["uids"], "cad_indices": axis, "cad_input_rows": 271,
        "reference_row_sha256": reference_sha, "checkpoint": tensors,
        "november_source": NOVEMBER, "november_wrapper_sha256": WRAPPER_SHA,
        "november_text_preparation_seconds": old_preparation.seconds,
        "november_profiled_seconds": old_phases, "shared_text_preparation_seconds": new_preparation.seconds,
        "synchronized_batch_seconds": timings.seconds, "cache_hits": cached.hits,
        "comparisons": checks, "repetitions": reps,
        "all_passed": all(comparison["passed"] for comparison in (*checks.values(), *reps)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "checkpoint", "bert", "cad-zip", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--level", choices=("L0", "L1", "L2", "L3", "CT"), default="L2")
    args = parser.parse_args()
    old_cache, core = historical_source()
    manifest = args.manifests / "manifest_receipt.json"
    verify(pilot.sha256_file(manifest) == RECEIPT, "Frozen validation receipt changed")
    verify(pilot.sha256_file(args.cad_zip) == pilot.INPUT_HASHES["official_zip"], "Native CAD zip changed")
    args.out.mkdir()
    torch.set_num_threads(2)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model, tensors = model_loaded(args, device)
    result = probe(args, model, device, old_cache, tensors, json.loads(manifest.read_bytes())["file_sha256"])
    result.update({"november_core_sha256": core, "checkout": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "precision": "FP32/must_keep_origin_dtype"})
    pilot.write_json(args.out / "report.json", result)
    print(json.dumps({key: result[key] for key in ("level", "all_passed", "synchronized_batch_seconds",
                                                 "november_profiled_seconds")}), flush=True)
    return 0 if result["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
