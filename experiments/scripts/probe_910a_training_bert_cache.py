"""Measure a small FP32 frozen-token cache on real T training microbatches."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from experiments.scripts import run_formal_text2cad_training as native


def encode(embedder, packed, device):
    inputs = {key: value.to(device) for key, value in packed.items()}
    with torch.inference_mode():
        states = embedder.model(**inputs)[0]
    torch.npu.synchronize(device)
    native.require(states.dtype == torch.float32, "BERT output precision changed")
    return states


def tokens_for_row(packed, row):
    mask = packed["attention_mask"][row]
    length = int(mask.sum())
    native.require(0 < length <= 512 and bool(mask[:length].all()) and not bool(mask[length:].any()),
                   "Expected nonempty right-padded tokens")
    return {key: tuple(value[row, :length].tolist()) for key, value in packed.items()}


def make_cache(embedder, texts, destination, device):
    tokenizer = embedder.tokenizer
    ordered = sorted(set(texts), key=lambda text: (min(embedder.length(text), 512), text))
    groups = [ordered[start:start + 8] for start in range(0, len(ordered), 8)]
    encoded = [native.encode_training_texts(tokenizer, batch, 64) for batch in groups]
    total_tokens = sum(int(group["attention_mask"].sum()) for group in encoded)
    hidden_size = embedder.model.config.hidden_size
    array = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                     shape=(total_tokens, hidden_size))
    index = {}
    offset = 0
    for texts_in_group, packed in zip(groups, encoded, strict=True):
        features = encode(embedder, packed, device).cpu()
        for row, text in enumerate(texts_in_group):
            tokens = tokens_for_row(packed, row)
            length = len(tokens["input_ids"])
            array[offset:offset + length] = features[row, :length].numpy()
            index[text] = (tokens, offset, length)
            offset += length
    native.require(offset == total_tokens, "Incomplete cache")
    array.flush()
    del array
    return index


def read_cache(array, index, texts, mask, device):
    host = np.zeros((*mask.shape, array.shape[1]), dtype=np.float32)
    for row, text in enumerate(texts):
        _, offset, length = index[text]
        host[row, :length] = array[offset:offset + length]
    states = torch.from_numpy(host).to(device)
    padding = (~mask).to(device)
    torch.npu.synchronize(device)
    return states, padding


def elapsed(operation):
    start = perf_counter()
    value = operation()
    return value, perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "bert", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir()
    source = "t2cad-v1.0"
    train, _, _ = native.axes(SimpleNamespace(manifests=args.manifests, source=source, train_only=True))
    levels = native.LEVELS[source]
    order = native.epoch_order(len(train) * len(levels), 1)
    batch = order[2 * native.BATCH:3 * native.BATCH].tolist()
    texts_by_rank = [[train[i // len(levels)].texts[i % len(levels)] for i in batch[rank::2]] for rank in (0, 1)]
    native.require(len(batch) == 16, "Expected the third registered global B16 batch")
    device, _, world = native.runtime(SimpleNamespace(device="npu"))
    native.require(world == 1, "Run the cache probe on one NPU")
    native.verify_bert_asset_shas(args.bert)
    embedder = native.TrainingEmbedder.from_local(args.bert, 512, device)
    native.verify_bert_model_load(embedder, args.bert)
    native.require(embedder.tokenizer.truncation_side == embedder.tokenizer.padding_side == "right",
                   "Wrong tokenizer sides")
    native.require(not embedder.model.training and all(not p.requires_grad for p in embedder.parameters()),
                   "BERT must be frozen in evaluation mode")

    path = args.output / "effective_token_features.npy"
    all_texts = [text for texts in texts_by_rank for text in texts]
    print(json.dumps({"stage": "frozen_bert_loaded", "pairs": len(all_texts)}), flush=True)
    index, build_seconds = elapsed(lambda: make_cache(embedder, all_texts, path, device))
    cache = np.load(path, mmap_mode="r", allow_pickle=False)
    result = []
    for ids, texts in zip((batch[::2], batch[1::2]), texts_by_rank, strict=True):
        packed = native.encode_training_texts(embedder.tokenizer, texts, 64)
        shape = tuple(packed["input_ids"].shape)
        mask = packed["attention_mask"].bool()
        for row, text in enumerate(texts):
            native.require(tokens_for_row(packed, row) == index[text][0], "Cache changed an effective token")
        def direct():
            return encode(embedder, packed, device)

        def replay():
            return read_cache(cache, index, texts, mask, device)

        expected, cold_forward_seconds = elapsed(direct)
        restored, padding = replay()
        native.require(torch.equal(padding.cpu(), ~mask), "Cache changed the padding mask")
        reference, candidate = expected.cpu()[mask], restored.cpu()[mask]
        difference = float((reference - candidate).abs().max())
        print(json.dumps({"stage": "compared_effective_features", "width": shape[1],
                          "tokens": int(mask.sum()), "max_abs_error": difference}), flush=True)
        torch.testing.assert_close(reference, candidate, atol=5e-5, rtol=1e-5)
        native.require(bool(torch.isfinite(candidate).all()), "Nonfinite cache feature")
        samples = {"direct": [], "cache": []}
        for repeat in range(3):
            ops = [("direct", direct), ("cache", replay)]
            if repeat % 2:
                ops.reverse()
            for kind, operation in ops:
                _, seconds = elapsed(operation)
                samples[kind].append(seconds)
        result.append(dict(pair_ids=ids, width=shape[1], tokens=int(mask.sum()),
                           cold_forward_seconds=cold_forward_seconds,
                           feature_max_abs_error=difference,
                           bitwise_equal=bool(torch.equal(reference, candidate)),
                           warmed_seconds=samples,
                           direct_median_seconds=median(samples["direct"]),
                           cache_median_seconds=median(samples["cache"])))
    encoded_index = [{"text_sha256": sha256(text.encode()).hexdigest(), "offset": value[1], "length": value[2]}
                     for text, value in index.items()]
    (args.output / "index.json").write_text(json.dumps(encoded_index, indent=2) + "\n", encoding="utf-8")
    with path.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
    receipt = dict(status="ok", source=source, batch_number=3, pairs=16,
                   device=torch.npu.get_device_name(device), freeze_verified=True, dtype=str(cache.dtype),
                   distinct_texts=len(index), stored_shape=list(cache.shape), cache_bytes=path.stat().st_size,
                   cache_sha256=digest, build_seconds=build_seconds,
                   tolerance=dict(atol=5e-5, rtol=1e-5), microbatches=result,
                   replay_timing="warm mmap, reconstruction and device transfer", update_executed=False)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
