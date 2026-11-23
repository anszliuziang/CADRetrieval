#!/usr/bin/env python3
"""Bounded Text2CAD optimizer witness using registered train-only examples."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F
import torch_npu

from experiments.scripts.run_910a_text2cad_dev_recognition import PINS, write_json, write_row
from experiments.src.idea01.in_domain_pilot import SOURCE_LEVELS, cad_member, official_splits
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore, assemble_batch
from experiments.src.idea01.text2cad_port.loader import load_frozen_text2cad
from experiments.src.idea01.text2cad_port.model import teacher_forced_logits

TRAIN_SHA = {
    "t2cad-v1.0": "a0a4637510e5223684fb6ce878826014ca62762c36c9a00e612fd29a19616a27",
    "ct-deepcad": "da4ee6f0d6c4578c4176766e2321f22ff18ebff395e38ec8e10e6350f7eb80f6",
}
SPLIT_SHA = "4ea6db9027b4cebee8f37903b6601b73b810adacd808a4874dbe847c7c508e7b"


def select_train_examples(metadata, manifests, official_train, embedder):
    inputs = {}
    evidence = {}
    for source, expected in TRAIN_SHA.items():
        paths = manifests / source / "pilot_train.jsonl"
        raw = paths.read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Training manifest pin mismatch: {source}")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != 4096 or len({row["uid"] for row in rows}) != 4096:
            raise ValueError(f"Pilot-only train axis must contain 4096 unique UIDs: {source}")
        if any(row["uid"] not in official_train or row["cad_member"] != cad_member(row["uid"]) for row in rows):
            raise ValueError(f"{source} contains a UID outside official training or a mismatched candidate")
        selected, skips = [], 0
        for row in rows:
            captions = [row["texts"][level] for level in SOURCE_LEVELS[source]]
            ids = embedder.tokenizer(captions, truncation=False)["input_ids"]
            if any(len(item) > embedder.max_seq_len for item in ids):
                skips += 1
                continue
            selected.extend({"uid": row["uid"], "member": row["cad_member"], "level": level,
                             "text": text, "tokens": len(tokens)}
                            for level, text, tokens in zip(SOURCE_LEVELS[source], captions, ids, strict=True))
            if len(selected) == 4:
                break
        if len(selected) != 4:
            raise ValueError(f"Unable to select a full training witness batch: {source}")
        inputs[source] = selected
        evidence[source] = {"manifest_sha256": expected, "skipped_exceeding_context": skips,
                            "batch": [{k: v for k, v in row.items() if k != "text"} for row in selected]}
    metadata["training_inputs"] = evidence
    return inputs


def loss_for(prediction, labels, mask):
    valid = ~mask
    widths = [int(valid[..., head].sum().item()) for head in (0, 1)]
    if min(widths) == 0 or prediction.dtype != torch.float32:
        raise ValueError("Witness requires FP32 logits and both valid heads")
    losses = []
    for head, width in enumerate(widths):
        raw = F.cross_entropy(prediction[:, :, head].transpose(1, 2), labels[:, :, head].long(),
                              reduction="none", label_smoothing=0.1)
        losses.append((raw * valid[:, :, head]).sum() / width)
    return (losses[0] + losses[1]) / 2, widths


def run_source(source, model, named, samples, store, locations, device, output):
    parameters = [param for _, param in named]
    tracked_name, tracked = named[-1]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001)
    tensors, masks, seq = assemble_batch(store, [locations[row["member"]] for row in samples], device)
    texts = [row["text"] for row in samples]
    torch.manual_seed(17)
    torch.npu.manual_seed(17)
    model.train()
    encoder = model.base_text_embedder
    torch.npu.reset_peak_memory_stats(device)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for step in range(1, 4):
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            pred, labels, mask = teacher_forced_logits(model, texts, tensors, masks, cad_seq_len=seq)
            loss, widths = loss_for(pred, labels, mask)
            smoothed_loss = float(loss.detach().cpu())
            if not math.isfinite(smoothed_loss):
                raise ValueError("Training loss is non-finite")
            torch.npu.synchronize(device)
            forward = time.perf_counter()
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.9).detach().cpu())
            if not math.isfinite(norm) or norm <= 0 or any(p.grad is not None for p in encoder.parameters()):
                raise ValueError("Invalid gradient norm or base BERT unexpectedly got gradients")
            torch.npu.synchronize(device)
            backward = time.perf_counter()
            old = tracked.detach().cpu().clone()
            optimizer.step()
            torch.npu.synchronize(device)
            state = int(optimizer.state[tracked]["step"])
            change = float((tracked.detach().cpu() - old).abs().max())
            if not math.isfinite(change) or change <= 0 or state != step:
                raise ValueError("AdamW did not update its tracked parameter at the intended step")
            write_row(stream, {"source": source, "step": step, "optimizer_step": state, "loss": smoothed_loss,
                               "max_parameter_update": change, "parameter_witness": tracked_name,
                               "gradient_norm_before_clip": norm, "valid_head_fields": widths, "pairs": len(texts),
                               "forward_seconds": forward - start, "backward_seconds": backward - forward,
                               "optimizer_and_witness_seconds": time.perf_counter() - backward,
                               "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
                               "peak_reserved_bytes": torch.npu.max_memory_reserved(device)})
            print(f"{source} step {step}: loss={smoothed_loss:.6f}, change={change:.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--train-manifests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Use the clean deployed implementation")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    pins = json.loads(PINS.read_bytes())
    r = pins["dataset_revision"]
    dataset = args.assets / "datasets/Text2CAD-v1.0" / r
    split_file = dataset / "text2cad_v1.0/train_test_val.json"
    if sha256(split_file.read_bytes()).hexdigest() != SPLIT_SHA:
        raise ValueError("Official training split file differs")
    split = official_splits(split_file)
    bert = args.assets / "models/bert-large-uncased" / pins["bert_revision"]
    model, loaded = load_frozen_text2cad(
        args.assets / "models/Text2CAD-v1.0" / r / "Text2CAD_1.0.pth", bert, device,
        expected_keys_manifest=Path("experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"),
    )
    meta = {"code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "load": loaded, "scope": "train-only runtime witness, discarded weights", "batch": 4,
            "adamw_lr": 0.0001, "clip_norm": 0.9, "seed": 17, "preset": "FP32 preserve"}
    examples = select_train_examples(meta, args.train_manifests, split["train"], model.base_text_embedder)
    members = list(dict.fromkeys(row["member"] for source in examples.values() for row in source))
    store = ZipCandidateStore(dataset / "cad_seq.zip", members)
    if store.preload()["n_loaded"] != len(members):
        raise ValueError("One or more train-only CAD vectors failed the native store gate")
    locations = dict(zip(members, range(len(members)), strict=True))
    names = [(name, value.requires_grad_(True)) for name, value in model.named_parameters()
             if not name.startswith("base_text_embedder.")]
    if not names or any(p.requires_grad for p in model.base_text_embedder.parameters()):
        raise ValueError("The base text encoder must stay frozen")
    initial = [value.detach().cpu().clone() for _, value in names]
    write_json(args.out / "runtime.json", meta)
    for source in ("t2cad-v1.0", "ct-deepcad"):
        with torch.no_grad():
            for (_, param), fixed in zip(names, initial, strict=True):
                param.copy_(fixed.to(device))
        run_source(source, model, names, examples[source], store, locations, device, args.out / (source + ".jsonl"))
    write_json(args.out / "summary.json", {"passed": True, "updates": 6, "fitted_weights_saved": False})


if __name__ == "__main__":
    main()
