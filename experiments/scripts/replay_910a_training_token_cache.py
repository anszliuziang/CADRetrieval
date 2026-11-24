"""Bounded train-mode A/B replay from the immutable T step-64 checkpoint."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import monotonic
from types import SimpleNamespace

import numpy as np
import torch

from experiments.scripts import probe_910a_training_bert_cache as cache_probe
from experiments.scripts import run_formal_text2cad_training as native
from experiments.scripts import run_formal_text2cad_verify_restore as verifier
from experiments.scripts.compare_formal_text2cad_updates import equal_tree


SOURCE = "t2cad-v1.0"
START_UPDATE, STOP_UPDATE = 64, 128
VALIDATION_ROWS = 16
CHECKPOINT_SHA = "34cf770210d359005310ad7aea3eaa20622345261aca5adbf171e5b1dff6d056"
SOURCE_TRAINER_SHA = "896f7160804adca5c7fed474d1f37904b7789a1b"


def text_digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def token_digest(tokens):
    return sha256(json.dumps(tokens, sort_keys=True).encode("ascii")).hexdigest()


def build(args):
    args.out.mkdir()
    train, _, _ = native.axes(SimpleNamespace(manifests=args.manifests, source=SOURCE, train_only=True))
    levels = native.LEVELS[SOURCE]
    order = native.epoch_order(len(train) * len(levels), 1)
    interval = order[START_UPDATE * native.BATCH:STOP_UPDATE * native.BATCH].tolist()
    # Include the real eight-pair epoch tail for a separate numerical witness.
    tail = order[(math.ceil(len(order) / native.BATCH) - 1) * native.BATCH:].tolist()
    texts = [train[pair // len(levels)].texts[pair % len(levels)] for pair in interval + tail]
    native.require(len(interval) == (STOP_UPDATE - START_UPDATE) * native.BATCH and len(tail) == 8,
                   "The registered training interval or tail changed")
    device, _, world = native.runtime(SimpleNamespace(device="npu"))
    native.require(world == 1, "Build the small replay cache on one NPU")
    native.verify_bert_asset_shas(args.bert)
    embedder = native.TrainingEmbedder.from_local(args.bert, 512, device)
    native.verify_bert_model_load(embedder, args.bert)
    native.require(embedder.tokenizer.truncation_side == embedder.tokenizer.padding_side == "right"
                   and not embedder.model.training and all(not p.requires_grad for p in embedder.parameters()),
                   "The replay cache needs the pinned right-padded eval-mode BERT")
    feature_path = args.out / "effective_token_features.npy"
    print(json.dumps(dict(stage="encoding", pairs=len(interval), tail_pairs=len(tail),
                          distinct_texts=len(set(texts)))), flush=True)
    started = monotonic()
    index = cache_probe.make_cache(embedder, texts, feature_path, device)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    for start in range(0, len(features), 8192):
        native.require(np.isfinite(features[start:start + 8192]).all(), "Nonfinite BERT cache")
    index_path = args.out / "index.json"
    native.write_json(index_path, {text_digest(text): dict(token_sha256=token_digest(tokens),
                                                           offset=offset, length=length)
                                  for text, (tokens, offset, length) in index.items()})
    native.write_json(args.out / "receipt.json", dict(
        complete=True, source=SOURCE, epoch=1, start_update=START_UPDATE, stop_update=STOP_UPDATE,
        batch_pairs=interval, tail_pairs=tail, distinct_texts=len(index), dtype=str(features.dtype),
        feature_shape=list(features.shape), pad_multiple=64, bert_sha256=native.PINNED_BERT_FILE_SHA256,
        feature_sha256=native.sha256_file(feature_path), index_sha256=native.sha256_file(index_path),
        build_seconds=monotonic() - started,
    ))


def cache_reader(directory):
    receipt = json.loads((directory / "receipt.json").read_bytes())
    expected = dict(complete=True, source=SOURCE, epoch=1, start_update=START_UPDATE,
                    stop_update=STOP_UPDATE, pad_multiple=64, bert_sha256=native.PINNED_BERT_FILE_SHA256)
    native.require(all(receipt.get(k) == v for k, v in expected.items()), "Unexpected token cache scope")
    feature_path = directory / "effective_token_features.npy"
    index_path = directory / "index.json"
    native.verified(feature_path, receipt["feature_sha256"])
    native.verified(index_path, receipt["index_sha256"])
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    index = json.loads(index_path.read_bytes())
    native.require(features.dtype == np.float32 and list(features.shape) == receipt["feature_shape"]
                   and features.ndim == 2 and features.shape[1] == 1024 and len(index) == receipt["distinct_texts"],
                   "Invalid cache shape or index cardinality")
    offset = 0
    for row in sorted(index.values(), key=lambda row: row["offset"]):
        native.require(type(row["offset"]) is int and row["offset"] == offset
                       and type(row["length"]) is int and 0 < row["length"] <= 512,
                       "Invalid cache index span")
        offset += row["length"]
    native.require(offset == len(features), "Cache index does not cover all stored token rows")
    return receipt, features, index


def make_embedder(features, index):
    class CachedTrainingEmbedder(native.TrainingEmbedder):
        def get_embedding(self, texts):
            if not self.truncate:
                return super().get_embedding(texts)
            if self.profile:
                native.synchronize(self.device)
                started = monotonic()
            packed = native.encode_training_texts(self.tokenizer, texts, self.pad_multiple)
            active = packed["attention_mask"].bool()
            offsets = {}
            for i, text in enumerate(texts):
                row = index[text_digest(text)]
                tokens = cache_probe.tokens_for_row(packed, i)
                native.require(len(tokens["input_ids"]) == row["length"]
                               and token_digest(tokens) == row["token_sha256"],
                               "Cached effective token inputs differ from this batch")
                offsets[text] = (None, row["offset"], row["length"])
            result = cache_probe.read_cache(features, offsets, texts, active, self.device)
            if self.profile:
                self.bert_seconds = monotonic() - started
                self.text_width = active.shape[1]
            return result

    return CachedTrainingEmbedder


def compare_recipes(incoming, current):
    expected = dict(incoming, script_sha256=current["script_sha256"], code_sha=current["code_sha"],
                    smoke_validation_count=VALIDATION_ROWS, training_text_pad_multiple=64)
    native.require(incoming["code_sha"] == SOURCE_TRAINER_SHA and incoming["smoke_validation_count"] == 0
                   and "training_text_pad_multiple" not in incoming and expected == current,
                   "The replay differs from the original training contract")


def replay(args):
    pointer = json.loads((args.out / "checkpoint.json").read_bytes())
    native.require((pointer["updates"], pointer["epoch"], pointer["next_batch"], pointer["file"]) ==
                   (START_UPDATE, 1, START_UPDATE, "progress-b.pt"), "Replay requires the step-64 checkpoint")
    source_checkpoint = native.verified(args.out / pointer["file"], CHECKPOINT_SHA)
    original_recipe = torch.load(source_checkpoint, map_location="cpu", weights_only=False)["recipe"]
    if args.cache:
        receipt, features, index = cache_reader(args.cache)
        native.TrainingEmbedder = make_embedder(features, index)
    else:
        receipt = None

    original_restore = native.restore
    original_run = native.run
    original_update = native.update
    steps = 0

    def restore_original(args, recipe, model, optimizer, device, rank):
        compare_recipes(original_recipe, recipe)
        state = original_restore(args, original_recipe, model, optimizer, device, rank)
        native.write_json(args.out / f"replay_recipes_rank_{rank}.json", dict(
            checkpoint_file_sha256=CHECKPOINT_SHA, incoming=original_recipe, diagnostic=recipe,
        ))
        return state

    def run_with_padding(trainer_args):
        trainer_args.train_text_pad_multiple = 64
        original_run(trainer_args)

    def observe_update(model, wrapped, optimizer, store, indices, train, levels, ids, rank, world, device, audit, profile):
        nonlocal steps
        first = steps == 0
        native.require(world == 2 and not audit and model.training and len(ids) == native.BATCH,
                       "Replay must exercise actual two-rank B16 training with dropout")
        if receipt:
            start = steps * native.BATCH
            native.require(ids == receipt["batch_pairs"][start:start + native.BATCH], "Cache replay axis changed")
        result, probes = original_update(model, wrapped, optimizer, store, indices, train, levels, ids, rank, world,
                                        device, audit=first, profile=first or steps % 16 == 15)
        result["pair_axis_sha256"] = token_digest(ids)
        if rank == 0 and first:
            native.atomic_save(args.out / "first_training_update.pt", dict(result, pair_ids=ids, parameters=probes))
        steps += 1
        return result, probes

    native.restore = restore_original
    native.run = run_with_padding
    native.update = observe_update
    argv = ["--restore-file-sha256", CHECKPOINT_SHA, "--source", SOURCE, "--device", "npu",
            "--manifests", str(args.manifests), "--bert", str(args.bert), "--cad-zip", str(args.cad_zip),
            "--out", str(args.out), "--resume", "--stop-after-updates", str(STOP_UPDATE),
            "--smoke-validation-count", str(VALIDATION_ROWS)]
    sys.argv[1:] = argv
    started = monotonic()
    verifier.main()
    # torchrun's RANK is available after the trainer tears down its process group.
    if os.environ.get("RANK") == "0":
        native.require(steps == STOP_UPDATE - START_UPDATE and (args.out / "first_training_update.pt").is_file()
                       and (args.out / "smoke-validation-128.json").is_file(), "Incomplete replay interval")
        native.write_json(args.out / "replay_receipt.json", dict(
            complete=True, checkpoint_file_sha256=CHECKPOINT_SHA, first_update=START_UPDATE + 1,
            last_update=STOP_UPDATE, steps=steps, world=2, global_batch=native.BATCH, pad_multiple=64,
            cache_sha256=receipt["feature_sha256"] if receipt else None, elapsed_seconds=monotonic() - started,
            validation_rows_per_level=VALIDATION_ROWS, training_mode=True,
        ))


def check_runs(args):
    roots = [args.online, args.cached]
    receipts = [json.loads((root / "replay_receipt.json").read_bytes()) for root in roots]
    native.require(all(r["complete"] and r["checkpoint_file_sha256"] == CHECKPOINT_SHA
                       and r["steps"] == STOP_UPDATE - START_UPDATE for r in receipts)
                   and receipts[0]["cache_sha256"] is None and receipts[1]["cache_sha256"],
                   "Missing matched replay receipts")
    first = [torch.load(root / "first_training_update.pt", map_location="cpu", weights_only=False) for root in roots]
    equal_tree(first[0]["pair_ids"], first[1]["pair_ids"], "Training step-65 pair IDs")
    equal_tree(first[0]["by_rank"], first[1]["by_rank"], "Training step-65 rank shards")
    gradients = {}
    for name, ref in first[0]["parameters"].items():
        other = first[1]["parameters"][name]
        equal_tree(ref["before"], other["before"], f"{name} start state")
        torch.testing.assert_close(ref["gradient"], other["gradient"], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(ref["after"] - ref["before"], other["after"] - other["before"], rtol=1e-4, atol=2e-6)
        gradients[name] = float((ref["gradient"] - other["gradient"]).abs().max())
    timing = []
    series = []
    for root in roots:
        for rank in range(2):
            record = json.loads((root / f"replay_recipes_rank_{rank}.json").read_bytes())
            witness = json.loads((root / f"exact_restore_step_64_rank_{rank}.json").read_bytes())
            native.require(record["checkpoint_file_sha256"] == CHECKPOINT_SHA
                           and witness["passed"] and witness["restore_file_sha256"] == CHECKPOINT_SHA
                           and witness["source_git_sha"] == record["diagnostic"]["code_sha"]
                           and witness["source_code_sha256"] == record["diagnostic"]["script_sha256"],
                           "Unverified replay restore source")
            compare_recipes(record["incoming"], record["diagnostic"])
        with (root / "steps.jsonl").open("rb") as stream:
            selected = [json.loads(line) for line in stream][START_UPDATE:]
        native.require([r["update"] for r in selected] == list(range(START_UPDATE + 1, STOP_UPDATE + 1)),
                       "Incomplete training-step log")
        durations = [r["seconds"] for r in selected]
        timing.append(dict(all_update_mean_s=mean(durations), last_32_mean_s=mean(durations[-32:]),
                           last_32_median_s=median(durations[-32:]), training_update_sum_s=sum(durations)))
        series.append(selected)
    max_train_loss_difference = 0.0
    for left, right in zip(*series, strict=True):
        for key in ("epoch", "batch", "update", "exposure", "pairs", "fields", "by_level", "truncated", "lr",
                    "pair_axis_sha256"):
            equal_tree(left[key], right[key], f"Step {left['update']} {key}")
        for key in ("smoothed_sum", "nll_sum"):
            for i, (a, b) in enumerate(zip(left[key], right[key], strict=True)):
                fields = left["fields"][i]
                native.require(math.isclose(a / fields, b / fields, rel_tol=1e-4, abs_tol=1e-5),
                               f"Step {left['update']} {key} head {i} changed")
        max_train_loss_difference = max(max_train_loss_difference, abs(left["loss"] - right["loss"]))
    a, b = [json.loads((root / "smoke-validation-128.json").read_bytes()) for root in roots]
    if len(a["rows"]) != VALIDATION_ROWS * len(native.LEVELS[SOURCE]):
        raise ValueError("Unexpected number of validation rows")
    score_keys = ("nll", "x_nll", "y_nll")
    for left, right in zip(a["rows"], b["rows"], strict=True):
        equal_tree({k: v for k, v in left.items() if k not in score_keys},
                   {k: v for k, v in right.items() if k not in score_keys},
                   "Validation axis and effective field counts")
        if left["status"] == "ok":
            for key, count_key in zip(score_keys, ("fields", "x_fields", "y_fields"), strict=True):
                fields = max(1, left[count_key])
                native.require(math.isclose(left[key] / fields, right[key] / fields, rel_tol=1e-4, abs_tol=1e-5),
                               f"Validation {key} changed")
    native.write_json(args.out, dict(passed=True, start_update=START_UPDATE, end_update=STOP_UPDATE,
                                    gradient_max_abs_error=gradients, timing=dict(zip(("online", "cached"), timing)),
                                    receipts=receipts, max_train_loss_difference=max_train_loss_difference,
                                    validation_rows=len(a["rows"])))


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("build", "replay"):
        entry = sub.add_parser(mode)
        for path in ("manifests", "bert", "out"):
            entry.add_argument("--" + path, type=Path, required=True)
        if mode == "replay":
            entry.add_argument("--cad-zip", type=Path, required=True)
            entry.add_argument("--cache", type=Path)
    compare = sub.add_parser("check")
    for path in ("online", "cached", "out"):
        compare.add_argument("--" + path, type=Path, required=True)
    args = parser.parse_args()
    {"build": build, "replay": replay, "check": check_runs}[args.mode](args)


if __name__ == "__main__":
    main()
