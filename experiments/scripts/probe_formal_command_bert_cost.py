#!/usr/bin/env python3
"""Measure the frozen BERT stage for full-source command training on real training text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import time

import torch

from experiments.scripts.run_in_domain_pilot_retrievers import (
    TextInputs, initialize_devices, manifest_axes, write_json,
)
from experiments.src.idea01.in_domain_pilot import COHORTS, SOURCE_LEVELS
from experiments.src.idea01.text2cad_port.loader import verify_bert_asset_shas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-root", "bert-dir", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--source", choices=tuple(SOURCE_LEVELS), default="t2cad-v1.0")
    a = p.parse_args()
    a.train_split = "full"
    a.expected_train_uids = COHORTS[a.source]["train"][0]
    a.expected_targets = 512
    a.expected_gallery = 2048
    training, _, _, pins = manifest_axes(a)
    texts = list(dict.fromkeys(row["texts"][level] for row in training for level in SOURCE_LEVELS[a.source]))
    a.output.mkdir()
    verify_bert_asset_shas(a.bert_dir)
    group = initialize_devices(a.device, keep_origin_dtype=True)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.bert_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    inputs = TextInputs(tokenizer, texts, "", 512, group.device, pad_multiple=64, native_truncation=True)
    widths = defaultdict(list)
    for text in texts:
        ids = inputs.ids[text]
        if ids[0] != tokenizer.cls_token_id or ids[-1] != tokenizer.sep_token_id:
            raise ValueError("The formal training input lost a BERT boundary token")
        widths[64 * ((len(ids) + 63) // 64)].append(text)
    counts = {width: len(entries) for width, entries in sorted(widths.items())}
    batches = Counter()
    lengths = sorted(len(inputs.ids[text]) for text in texts)
    for i in range(0, len(lengths), 32):
        batches[64 * ((max(lengths[i:i + 32]) + 63) // 64)] += 1
    axis = {"source": a.source, "training_uids": len(training), "unique_training_texts": len(texts),
            "contexts_per_padded_width": counts, "batches_per_padded_width": batches,
            "native_window_truncated_texts": len(inputs.overlength), "manifest_sha256": pins}
    write_json(a.output / "axis.json", axis)
    print(json.dumps(axis), flush=True)
    model = AutoModel.from_pretrained(a.bert_dir, trust_remote_code=False, local_files_only=True,
                                     torch_dtype=torch.float32, attn_implementation="eager").eval().to(group.device)
    available = [width for width, count in counts.items() if count >= 32]
    if not available:
        raise ValueError("Need at least one full batch of training-only texts")
    chosen = sorted({available[0], available[len(available) // 2], available[-1]})
    rows = []
    with (a.output / "timings.jsonl").open("x", encoding="utf-8") as stream:
        for width in chosen:
            sample = widths[width][:32]
            batch = inputs.batch(sample)
            if batch["input_ids"].shape != (32, width):
                raise ValueError("Frozen BERT witness differs from the intended training layout")
            active = batch["attention_mask"].sum(dim=1).cpu().tolist()
            if active != [len(inputs.ids[text]) for text in sample]:
                raise ValueError("Frozen BERT witness changed the complete effective inputs")
            with torch.inference_mode():
                for step in range(3):
                    group.backend.synchronize(group.device)
                    start = time.monotonic()
                    hidden = model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                    group.backend.synchronize(group.device)
                    elapsed = time.monotonic() - start
                    if pooled.shape != (32, model.config.hidden_size) or not torch.isfinite(pooled).all():
                        raise ValueError("Frozen BERT witness produced invalid pooled features")
                    result = {"padded_width": width, "phase": "cold" if step == 0 else "steady",
                              "seconds": elapsed, "input_sha256": sha256(batch["input_ids"].cpu().numpy().tobytes()).hexdigest()}
                    rows.append(result)
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps(result), flush=True)
    bands = {width: median(r["seconds"] for r in rows if r["padded_width"] == width and r["phase"] == "steady")
             for width in chosen}
    write_json(a.output / "summary.json", {
        **axis, "steady_batch_seconds": bands,
        "seconds_if_every_batch_were_as_fast_as_fastest_measured": min(bands.values()) * sum(batches.values()),
        "scope": "training-side frozen BERT only; illustrative throughput, no checkpoint or validation metrics",
    })


if __name__ == "__main__":
    main()
