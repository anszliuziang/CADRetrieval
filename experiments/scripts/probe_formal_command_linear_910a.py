#!/usr/bin/env python3
"""Compare the FP32 command tower against native CPU convolution, using the first CT training batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.scripts.run_in_domain_pilot_retrievers import (
    OUTPUT_DIM, TEMPERATURE, RetrievalTowers, TextInputs, file_sha256, initialize_devices, load_towers,
)
from experiments.scripts.probe_formal_text2cad_cost import RECEIPT
from experiments.src.idea01.in_domain_pilot_retrievers import (
    CommandTower, direct_registration, registered_positive_mask, symmetric_positive_set_loss,
)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def read_rows(file):
    with file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def source_batch(args):
    manifest = args.manifests / "manifest_receipt.json"
    require(file_sha256(manifest) == RECEIPT, "Expected frozen training manifest")
    name = "ct-deepcad/train.jsonl"
    require(file_sha256(args.manifests / name) == json.loads(manifest.read_bytes())["file_sha256"][name],
            "Expected immutable CT training text")
    train = read_rows(args.manifests / name)
    require(len(train) == 148046 and len({row["uid"] for row in train}) == len(train), "Wrong training axis")
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(17))[:32].tolist()
    batch = [train[index] for index in indices]
    texts = [row["texts"]["CT"] for row in batch]
    uids = [row["uid"] for row in batch]
    relations = direct_registration((row["texts"]["CT"], row["uid"]) for row in train)
    positives = registered_positive_mask(texts, uids, relations)

    root = args.cad_features
    summary = json.loads((root / "summary.json").read_bytes())
    for artifact in ("ct-deepcad_train_indices.jsonl", "cad_features.npy"):
        require(file_sha256(root / artifact) == summary["files_sha256"][artifact], "Shared training input changed")
    axis = read_rows(root / "ct-deepcad_train_indices.jsonl")
    require([row["uid"] for row in axis] == [row["uid"] for row in train], "Different training UID order")
    array = np.load(root / "cad_features.npy", mmap_mode="r", allow_pickle=False)
    cad = torch.from_numpy(array[[axis[index]["feature_index"] for index in indices]])
    require(cad.shape == (32, 272, 4) and cad.dtype == torch.int32, "Unexpected production CAD batch shape")
    return texts, uids, cad, positives


def objective(towers, text, cad, positives):
    sentence_vectors, cad_vectors = towers(text, cad)
    scores = sentence_vectors @ cad_vectors.T / TEMPERATURE
    return symmetric_positive_set_loss(scores, positives), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "cad-features", "bert", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    texts, uids, cad_cpu, positives_cpu = source_batch(args)
    group = initialize_devices("npu:0", keep_origin_dtype=True)
    torch.manual_seed(17)
    group.backend.manual_seed_all(17)
    towers, backbone, tokenizer, _, model = load_towers(SimpleNamespace(
        method="dual-tower", bert_dir=args.bert, precision="fp32", attention="eager", command_convolution="linear",
    ), group, None)
    inputs = TextInputs(tokenizer, list(dict.fromkeys(texts)), "", 512, group.device,
                        pad_multiple=64, native_truncation=True)
    frozen, positions = inputs.frozen(backbone, list(dict.fromkeys(texts)), 32)
    text = frozen[[positions[item] for item in texts]]
    cad, positives = cad_cpu.to(group.device), positives_cpu.to(group.device)

    # The reference uses exactly the same parameters and frozen text vectors. Disable dropout for this comparison.
    native = RetrievalTowers(nn.Sequential(nn.LayerNorm(text.shape[-1]), nn.Linear(text.shape[-1], OUTPUT_DIM)),
                             CommandTower()).eval()
    native.load_state_dict({key: value.cpu() for key, value in towers.state_dict().items()}, strict=True)
    towers.eval()
    ref_loss, ref_scores = objective(native, text.cpu(), cad_cpu, positives_cpu)
    loss, scores = objective(towers, text, cad, positives)
    torch.testing.assert_close(scores.detach().cpu(), ref_scores.detach(), rtol=5e-4, atol=2e-4)
    torch.testing.assert_close(loss.detach().cpu(), ref_loss.detach(), rtol=5e-4, atol=2e-4)
    ref_loss.backward()
    loss.backward()
    grads = {}
    for name, parameter in towers.named_parameters():
        actual, expected = parameter.grad.cpu(), native.get_parameter(name).grad
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-5, msg=name)
        grads[name] = float((actual - expected).abs().max())

    towers.train()
    optimizer = torch.optim.AdamW(towers.parameters(), lr=3e-4, weight_decay=0.01, foreach=False)
    initial = towers.cad.blocks[0][0].weight.detach().clone()
    updates = []
    for attempt in range(3):
        optimizer.zero_grad(set_to_none=True)
        group.backend.synchronize(group.device)
        start = time.monotonic()
        loss, _ = objective(towers, text, cad, positives)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(towers.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        group.backend.synchronize(group.device)
        updates.append({"step": attempt + 1, "seconds": time.monotonic() - start,
                        "loss": float(loss.detach()), "preclip_grad_norm": float(grad)})
        require(np.isfinite(updates[-1]["loss"]), "Optimizer produced nonfinite loss")
        print(json.dumps(updates[-1]), flush=True)
    require(not torch.equal(towers.cad.blocks[0][0].weight, initial), "Command convolution parameters were not updated")
    result = {"scope": "training batch only; no checkpoint or validation scores",
              "source": "ct-deepcad", "uids": uids, "bert": model,
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "max_abs_score_delta": float((scores.detach().cpu() - ref_scores.detach()).abs().max()),
              "cpu_native_loss": float(ref_loss.detach()), "max_abs_gradient_delta": max(grads.values()),
              "gradient_deltas": grads, "updates": updates,
              "steady_update_seconds": statistics.median(row["seconds"] for row in updates[1:]), "passed": True}
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
