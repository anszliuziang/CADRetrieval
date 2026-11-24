#!/usr/bin/env python3
"""Time native Text2CAD development input shapes and inspect their score equivalence."""
from __future__ import annotations

import argparse
import math
import subprocess
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
import torch_npu

from experiments.scripts.probe_910a_text2cad_training import loss_for
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore, assemble_batch
from experiments.src.idea01.text2cad_port.loader import load_frozen_text2cad
from experiments.src.idea01.text2cad_port.model import teacher_forced_logits

CASES = {
    "L0": ("t2cad-v1.0", "L0", "short"),
    "L1": ("t2cad-v1.0", "L1", "long"),
    "L2": ("t2cad-v1.0", "L2", "long"),
    "L3": ("t2cad-v1.0", "L3", "long"),
    "CT": ("ct-deepcad", "CT", "long"),
}
RECEIPT = "3c297920d0905b3dd5f14aa1b5ac163245a17b958cc3c2e1169ee65d5a8d055d"


def by_length(rows, level, width, n=16):
    candidates = [(width(r["texts"][level]), r["uid"], r) for r in rows]
    shapes = sorted(((w, u, r) for w, u, r in candidates if w <= 512), key=lambda x: (x[0], x[1]))
    if len(shapes) < n:
        raise ValueError("Too few complete training/validation prompts")
    return shapes, len(candidates) - len(shapes)


def model_loaded(args, device):
    p = Path("experiments/configs/text2cad_checkpoint_expected_keys_20260930.json")
    return load_frozen_text2cad(args.checkpoint, args.bert, device, expected_keys_manifest=p)


def timed_call(backend, pairs):
    torch.npu.synchronize(backend.device)
    tic = time.monotonic()
    rows, _ = backend.score(pairs)
    torch.npu.synchronize(backend.device)
    return rows, time.monotonic() - tic


def score_case(backend, ids, text, source, level):
    q = [backend.prepare(row, text) for row in ids]
    layout = [("full_cold", q), ("full_steady1", q), ("full_steady2", q),
              ("tail_cold", q[:14]), ("tail_steady", q[:14])]
    answers = []
    for mode, batch in layout:
        rows, elapsed = timed_call(backend, batch)
        if any(not math.isfinite(r["s0"]) or r["logit_dtype"] != "torch.float32" for r in rows):
            raise ValueError("Nonfinite native score or changed dtype")
        answers.append(rows)
        yield {"source": source, "level": level, "kind": mode, "seconds": elapsed,
               "pairs": len(rows), "pairs_per_s": len(rows) / elapsed,
               "uids": [r["uid"] for r in ids[:len(rows)]], "scores": rows}
    # Exact identities with true one-pair forwards for three fixed positions.
    reference_indices = range(0, 14, 6)
    reference = {i: timed_call(backend, [q[i]])[0][0] for i in reference_indices}
    checks = []
    for variant, rows in zip((mode for mode, _ in layout), answers, strict=True):
        for i, baseline in reference.items():
            actual = rows[i]
            delta = abs(actual["s0"] - baseline["s0"])
            tol = max(0.001, 1e-6 * abs(baseline["s0"]))
            matched = all(actual[k] == baseline[k] for k in
                          ("effective_units", "target_sha256", "input_sha256", "query_tokens"))
            checks.append({"layout": variant, "index": i, "uid": ids[i]["uid"],
                           "reference_s0": baseline["s0"], "actual_s0": actual["s0"],
                           "delta": delta, "tolerance": tol,
                           "matching_inputs": matched, "passed": matched and delta <= tol})
    # Layout changes near the boundary are explicitly counted even when within tolerance.
    flags = []
    for title, alter in (("full_repeat", answers[2]), ("full_vs_tail", answers[4])):
        canonical = [x["s0"] for x in answers[1][:len(alter)]]
        alternative = [x["s0"] for x in alter]
        order0 = sorted(range(len(alter)), key=lambda i: (-canonical[i], ids[i]["cad_index"]))
        order1 = sorted(range(len(alter)), key=lambda i: (-alternative[i], ids[i]["cad_index"]))
        cap = min(10, len(alter))
        flags.append({"layout": title, "top10_same_set": set(order0[:cap]) == set(order1[:cap]),
                      "whole_order_equal": order0 == order1,
                      "max_abs_delta": max(abs(a - b) for a, b in zip(canonical, alternative, strict=True))})
    yield {"source": source, "level": level, "kind": "layout_checks", "comparison": checks,
           "rank_checks": flags, "passed": all(item["passed"] for item in checks) and all(
                item["whole_order_equal"] for item in flags)}


def timing_train(model, store, indices, examples, device):
    model.train()
    model.base_text_embedder.eval()
    params = []
    for name, param in model.named_parameters():
        trainable = not name.startswith("base_text_embedder.")
        param.requires_grad_(trainable)
        if trainable:
            params.append(param)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    lengths = [item[0] for item in examples]
    pairs = [item[2] for item in examples]
    level = examples[0][3]
    text = [r["texts"][level] for r in pairs]
    vec, mask, cad_len = assemble_batch(store, [indices[row["cad_member"]] for row in pairs], device)
    for step in range(3):
        torch.npu.synchronize(device)
        tic = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logits, labels, valid_mask = teacher_forced_logits(model, text, vec, mask, cad_seq_len=cad_len)
        loss, counts = loss_for(logits, labels, valid_mask)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(params, 0.9).detach().cpu())
        optimizer.step()
        torch.npu.synchronize(device)
        elapsed = time.monotonic() - tic
        final = float(loss.detach().cpu())
        if not math.isfinite(final) or not math.isfinite(grad) or grad <= 0:
            raise ValueError("Text2CAD optimizer witness failed")
        if any(p.grad is not None for p in model.base_text_embedder.parameters()):
            raise ValueError("Text2CAD BERT unexpectedly received training gradients")
        yield {"step": step + 1, "phase": "cold" if step == 0 else "steady", "level": level,
               "pairs": len(pairs), "uids": [r["uid"] for r in pairs], "text_tokens": lengths,
               "effective_fields": counts, "seconds": elapsed, "pairs_per_s": len(pairs) / elapsed,
               "loss": final, "grad_norm": grad}


def main():
    import json

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifests", "checkpoint", "bert", "cad-zip", "out"):
        p.add_argument(f"--{key}", required=True, type=Path)
    p.add_argument("--case", choices=tuple(CASES), action="append",
                   help="Select validation/development conditions for independent timing jobs")
    p.add_argument("--stage", choices=("score", "train", "both"), default="both")
    a = p.parse_args()
    selected = a.case or ["L0", "L3", "CT"]
    if len(selected) != len(set(selected)):
        p.error("Each --case must be distinct")
    a.out.mkdir()
    receipt = a.manifests / "manifest_receipt.json"
    if pilot.sha256_file(receipt) != RECEIPT:
        raise ValueError("Input must come from certified development manifests")
    pins = json.loads(receipt.read_bytes())["file_sha256"]
    if pilot.sha256_file(a.cad_zip) != pilot.INPUT_HASHES["official_zip"]:
        raise ValueError("Unregistered CAD archive")
    def load(source, file):
        path = a.manifests / source / file
        if pilot.sha256_file(path) != pins[f"{source}/{file}"]:
            raise ValueError("Frozen input bytes differ")
        return pilot.read_jsonl(path)

    torch.set_num_threads(2)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(17)
    torch.npu.manual_seed(17)
    model, assets = model_loaded(a, device)
    tokenizer = model.base_text_embedder.tokenizer

    @lru_cache(maxsize=65536)
    def width(text):
        return len(tokenizer(text, truncation=False)["input_ids"])

    scored = []
    train = []
    for name in selected:
        source, level, band = CASES[name]
        if a.stage in ("score", "both"):
            gallery = sorted(load(source, "validation_gallery.jsonl"), key=lambda r: (r["n_fields"], r["cad_index"]))
            ids = [gallery[(len(gallery) - 1) * (2*i+1) // 32] for i in range(16)]
            targets, excluded = by_length(load(source, "validation_targets.jsonl"), level, width)
            j = len(targets)//4 if band == "short" else 7*len(targets)//8
            q_len, uid, r = targets[j]
            scored.append((source, level, ids, r["texts"][level], uid, q_len, excluded))
        if a.stage in ("train", "both"):
            eligible, excluded_train = by_length(load(source, "pilot_train.jsonl"), level, width)
            k = min(len(eligible) - 16, len(eligible)//4 if band == "short" else 7*len(eligible)//8)
            batch = [(*r, level) for r in eligible[k:k+16]]
            train.append((source, batch, excluded_train))
    members = sorted({r["cad_member"] for _, _, ids, *_ in scored for r in ids}
                     | {r[2]["cad_member"] for _, batch, _ in train for r in batch})
    store = ZipCandidateStore(a.cad_zip, members)
    if store.preload()["n_loaded"] != len(members):
        raise ValueError("Intended native candidates not loaded")
    indices = {member: i for i, member in enumerate(members)}
    backend = Text2CadReadout(model, store, indices, device)
    all_scores, all_training = [], []
    torch.npu.reset_peak_memory_stats(device)
    with (a.out / "timings.jsonl").open("x", encoding="utf-8") as stream:
        def write(stage, row):
            entry = {"stage": stage, **row}
            line = json.dumps(entry, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(json.dumps({k: v for k, v in entry.items() if k not in
                             ("scores", "uids", "comparison", "rank_checks")}), flush=True)
        for source, level, ids, text, uid, q_len, excluded in scored:
            for row in score_case(backend, ids, text, source, level):
                row.update(validation_query_uid=uid, query_tokens=q_len, overlength_in_validation=excluded)
                all_scores.append(row)
                write("score", row)
        for source, batch, excluded in train:
            for row in timing_train(model, store, indices, batch, device):
                row.update(source=source, overlength_in_pilot_train=excluded)
                all_training.append(row)
                write("train_probe", row)
    outcomes = Counter(row["passed"] for row in all_scores if row["kind"] == "layout_checks")
    pilot.write_json(a.out / "summary.json", {
        "scope": "validation scoring and train-only optimizer throughput, released weights as timing proxy",
        "requested_cases": selected, "requested_stage": a.stage,
        "optimizer_initialization": "new AdamW for each batch; the proxy model retains earlier probe updates",
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "full_formal_training_checkpoint_created": False, "layout_passes": outcomes[True],
        "layout_failures": outcomes[False], "assets": assets, "training_steps": len(all_training),
        "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(device),
    })
    return 0 if not outcomes[False] else 2


if __name__ == "__main__":
    raise SystemExit(main())
