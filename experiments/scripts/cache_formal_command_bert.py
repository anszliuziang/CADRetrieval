#!/usr/bin/env python3
"""Prepare, shard and merge the formal T command dual tower's train-only BERT cache."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import chain
import json
import os
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.run_in_domain_pilot_retrievers import (
    BERT_REVISION, FORMAL_MANIFEST_RECEIPT_SHA, LEVELS, TextInputs, initialize_devices, load_towers,
)
from experiments.src.idea01.formal_command_bert_cache import (
    AXIS_SCHEMA, BATCH_COUNT, BATCH_SIZE, CACHE_SCHEMA, COST_SHA, INDEX, MATRIX, POLICY, SET_SHA,
    SOURCE, TAIL, TEXT_COUNT, TRAIN_SHA, WIDTH, encoding_runtime, file_sha256, read_index, tokenizer_identity,
)
from experiments.src.idea01.in_domain_pilot import COHORTS
from experiments.src.idea01.in_domain_pilot_retrievers import MAX_CAD_SEQUENCE_LENGTH
from experiments.src.idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256, verify_bert_asset_shas


WORKSET = Path(__file__).resolve().parents[1] / "configs" / "idea01_eight_910a_worksets_20261124.json"
TOKENS, OFFSETS, ORDER = "token_ids.u16", "token_offsets.npy", "length_order.npy"
MODEL = {"repo": "google-bert/bert-large-uncased", "revision": BERT_REVISION, "files": PINNED_BERT_FILE_SHA256}
AXIS_ARTIFACTS = (INDEX, TOKENS, OFFSETS, ORDER)


def checked_json(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned input differs from its registered SHA-256: {path}")
    return json.loads(data)


def workset(path):
    plan = checked_json(path, SET_SHA)
    work = plan["t_dual_tower_train_bert_work"]
    if (plan["evidence_sha256"]["bert_cost"] != COST_SHA or tuple(work[key] for key in (
        "source", "split", "train_manifest_sha256", "unique_texts", "batch_size", "num_batches",
        "real_texts_in_last_batch",
    )) != (SOURCE, "train", TRAIN_SHA, TEXT_COUNT, BATCH_SIZE, BATCH_COUNT, TAIL)):
        raise ValueError("Unexpected formal command BERT workset")
    return work


def load_tokenizer(bert_dir):
    from transformers import AutoTokenizer

    # These are the ModelSource paths used by the direct runner's local tokenizer.
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"):
        if file_sha256(bert_dir / name) != PINNED_BERT_FILE_SHA256[name]:
            raise ValueError(f"Pinned BERT tokenizer input changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, trust_remote_code=False, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.truncation_side != "right" or len(tokenizer) > 65536:
        raise ValueError("Expected native right-truncated BERT and uint16 token IDs")
    return tokenizer


def load_backbone(path, group):
    from transformers import AutoModel

    verify_bert_asset_shas(path)
    backbone = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                         torch_dtype=torch.float32, attn_implementation="eager")
    backbone = backbone.to(group.device).eval().requires_grad_(False)
    if backbone.config.hidden_size != WIDTH or {param.dtype for param in backbone.parameters()} != {torch.float32}:
        raise ValueError("Loaded BERT is incompatible with the registered FP32 pooling inputs")
    return backbone


def training_texts(root):
    receipt = checked_json(root / "manifest_receipt.json", FORMAL_MANIFEST_RECEIPT_SHA)
    name = f"{SOURCE}/train.jsonl"
    manifest = root / name
    if receipt["file_sha256"][name] != TRAIN_SHA or file_sha256(manifest) != TRAIN_SHA:
        raise ValueError("Q05 must use the frozen formal T training manifest")
    seen_text, seen_uid = {}, set()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            uid = row["uid"]
            if uid in seen_uid or set(row["texts"]) != set(LEVELS[SOURCE]):
                raise ValueError(f"Repeated UID or incomplete training record: {uid}")
            seen_uid.add(uid)
            for level in LEVELS[SOURCE]:
                text = row["texts"][level]
                if not isinstance(text, str) or not text or text != text.strip():
                    raise ValueError(f"Invalid text at {uid}/{level}")
                seen_text.setdefault(text, None)
    if (len(seen_uid), len(seen_text)) != (COHORTS[SOURCE]["train"][0], TEXT_COUNT):
        raise ValueError("Training UID or exact-text count differs from the registered Q05 axis")
    return list(seen_text)


def write_json(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def padded_width(length):
    return 64 * ((int(length) + 63) // 64)


def prepare(args):
    started = perf_counter()
    work = workset(args.workset)
    cost = checked_json(args.cost_summary, COST_SHA)
    if cost["batches_per_padded_width"] != work["batches_per_padded_width"]:
        raise ValueError("The pinned T BERT cost summary and workset disagree")
    texts = training_texts(args.manifest_root)
    tokenizer = load_tokenizer(args.bert_dir)
    tokenizer_source = tokenizer_identity(tokenizer)
    lengths = np.empty(TEXT_COUNT, dtype="<u2")
    overlength_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as stage_dir:
        stage = Path(stage_dir)
        with (stage / INDEX).open("w", encoding="utf-8", newline="\n") as index, (stage / TOKENS).open("wb") as tokens:
            for start in range(0, TEXT_COUNT, 2048):
                group = texts[start:start + 2048]
                inputs = TextInputs(tokenizer, group, "", 512, "cpu", pad_multiple=64, native_truncation=True)
                rows = [inputs.ids[text] for text in group]
                for ordinal, (text, row, audit) in enumerate(zip(group, rows, inputs.rows, strict=True), start):
                    if not 2 <= len(row) <= 512 or row[0] != tokenizer.cls_token_id or row[-1] != tokenizer.sep_token_id:
                        raise ValueError(f"Native BERT boundary or context differs at text {ordinal}")
                    lengths[ordinal] = len(row)
                    index.write(json.dumps({"text": text, **audit}, ensure_ascii=False, allow_nan=False) + "\n")
                np.fromiter(chain.from_iterable(rows), dtype="<u2", count=sum(map(len, rows))).tofile(tokens)
                overlength_count += len(inputs.overlength)
                if not start % (2048 * 32):
                    print(json.dumps({"phase": "tokenize", "texts": start + len(group), "total_texts": TEXT_COUNT}),
                          flush=True)
        order = np.argsort(lengths, kind="stable").astype("<u4")
        width_counts = Counter(str(padded_width(length)) for length in lengths)
        batch_counts = Counter(str(padded_width(lengths[order[end - 1]]))
                               for end in chain(range(BATCH_SIZE, TEXT_COUNT, BATCH_SIZE), (TEXT_COUNT,)))
        expected = (cost["contexts_per_padded_width"], work["batches_per_padded_width"],
                    cost["native_window_truncated_texts"])
        if (width_counts, batch_counts, overlength_count) != expected:
            raise ValueError(f"Reconstructed input histogram differs from the pinned T cost summary: "
                             f"actual={(dict(width_counts), dict(batch_counts), overlength_count)}, expected={expected}")
        offsets = np.empty(TEXT_COUNT + 1, dtype="<u8")
        offsets[0] = 0
        np.cumsum(lengths, dtype="<u8", out=offsets[1:])
        np.save(stage / OFFSETS, offsets, allow_pickle=False)
        np.save(stage / ORDER, order, allow_pickle=False)
        report = {"schema": AXIS_SCHEMA, "policy": POLICY, "model": MODEL, "tokenizer": tokenizer_source,
                  "input_histogram": {"contexts_per_padded_width": width_counts,
                                      "batches_per_padded_width": batch_counts,
                                      "native_window_truncated_texts": overlength_count},
                  "files_sha256": {name: file_sha256(stage / name) for name in AXIS_ARTIFACTS}}
        write_json(stage / "receipt.json", report)
        stage.rename(args.output)
    print(json.dumps({"phase": "prepared", "axis_sha256": file_sha256(args.output / "receipt.json"),
                      "wall_seconds": perf_counter() - started, **report}), flush=True)


class BatchAxis:
    def __init__(self, path):
        receipt_data = (path / "receipt.json").read_bytes()
        self.digest = sha256(receipt_data).hexdigest()
        self.receipt = json.loads(receipt_data)
        if (self.receipt["schema"] != AXIS_SCHEMA or self.receipt["policy"] != POLICY
                or self.receipt["model"] != MODEL):
            raise ValueError("Prepared BERT axis is incompatible with the formal T workset")
        for name in AXIS_ARTIFACTS:
            if file_sha256(path / name) != self.receipt["files_sha256"][name]:
                raise ValueError(f"Prepared BERT axis file differs from its receipt: {name}")
        self.offsets = np.load(path / OFFSETS, allow_pickle=False)
        self.order = np.load(path / ORDER, allow_pickle=False)
        self.tokens = np.memmap(path / TOKENS, dtype="<u2", mode="r")
        if (self.offsets.dtype != np.dtype("<u8") or self.offsets.shape != (TEXT_COUNT + 1,)
                or self.order.dtype != np.dtype("<u4") or self.order.shape != (TEXT_COUNT,)
                or self.offsets[0] != 0 or int(self.offsets[-1]) != self.tokens.size):
            raise ValueError("Invalid prepared BERT offset/order axes")
        lengths = np.diff(self.offsets)
        if (np.any((lengths < 2) | (lengths > 512))
                or not np.array_equal(self.order, np.argsort(lengths, kind="stable"))):
            raise ValueError("Prepared BERT axis does not use dense stable effective-length ordering")

    def batch(self, index):
        if not 0 <= index < BATCH_COUNT:
            raise ValueError(f"Out-of-range global batch number {index}")
        start = index * BATCH_SIZE
        ids = self.order[start:start + BATCH_SIZE]
        rows = [self.tokens[self.offsets[i]:self.offsets[i + 1]] for i in ids]
        width = padded_width(len(rows[-1]))
        fingerprint = sha256(ids.tobytes())
        for row in rows:
            fingerprint.update(len(row).to_bytes(2, "little"))
            fingerprint.update(row.tobytes())
        return ids, rows, {"axis_sha256": self.digest, "global_batch": index, "padded_width": width,
                           "batch_input_sha256": fingerprint.hexdigest()}


def fragment_path(root, index):
    return root / f"batch-{index:05d}.npz"


def read_fragment(path, ids, expected, runtime=None):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"text_ids", "pooled", "receipt"}:
            raise ValueError(f"Unexpected fragment schema: {path}")
        receipt = json.loads(archive["receipt"].item())
        values = archive["pooled"]
        text_ids = archive["text_ids"]
    source_runtime = receipt["encoding_runtime"]
    if (any(receipt.get(key) != value for key, value in expected.items())
            or set(source_runtime) != {"device_kind", "device_model", "versions"}
            or source_runtime["device_kind"] not in {"npu", "cuda"}
            or (runtime is not None and source_runtime != runtime)
            or text_ids.dtype != np.dtype("<u4") or not np.array_equal(text_ids, ids)
            or values.dtype != np.dtype("float32") or values.shape != (len(ids), WIDTH)
            or not np.isfinite(values).all() or sha256(values.tobytes()).hexdigest() != receipt["pooled_sha256"]):
        raise ValueError(f"BERT fragment fingerprints, vector shape, or contents diverge: {path}")
    return values, receipt


def write_fragment(path, ids, values, expected, runtime):
    receipt = {**expected, "encoding_runtime": runtime, "pooled_sha256": sha256(values.tobytes()).hexdigest()}
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", suffix=".part", delete=False) as stream:
        pending = Path(stream.name)
        try:
            np.savez(stream, text_ids=ids, pooled=values, receipt=json.dumps(receipt, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            # A simultaneous retry can publish only one file. Never replace a previously committed batch.
            os.link(pending, path)
        except FileExistsError:
            read_fragment(path, ids, expected, runtime)
        finally:
            pending.unlink(missing_ok=True)


def accelerator(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run one process independently per selected worker partition")
    return initialize_devices(args.device, keep_origin_dtype=True)


def shard(args):
    started = perf_counter()
    work = workset(args.workset)
    partitions = work["partitions_by_worker_count"][str(args.worker_count)]
    if not 0 <= args.partition_index < len(partitions):
        raise ValueError("Partition index is zero-based and must be less than worker count")
    partition = partitions[args.partition_index]
    first, stop = partition["first_batch"], partition["stop_batch_exclusive"]
    start = first if args.batch_start is None else args.batch_start
    end = stop if args.batch_stop is None else args.batch_stop
    if not first <= start < end <= stop:
        raise ValueError(f"Requested [{start},{end}) is outside partition [{first},{stop})")
    axis = BatchAxis(args.axis)
    tokenizer = load_tokenizer(args.bert_dir)
    if axis.receipt["tokenizer"] != tokenizer_identity(tokenizer):
        raise ValueError("Encoder tokenizer runtime differs from the prepared training axis")
    args.output.mkdir(parents=True, exist_ok=True)
    group = accelerator(args)
    group.backend.reset_peak_memory_stats(group.device)
    runtime = encoding_runtime(group)
    encoded, resumed = 0, 0
    load_seconds = 0.0
    encode_seconds, write_seconds = Counter(), 0.0
    backbone = None
    for batch in range(start, end):
        ids, rows, expected = axis.batch(batch)
        path = fragment_path(args.output, batch)
        if expected["padded_width"] not in partition["padded_widths"]:
            raise ValueError(f"Global batch {batch} width is outside its pinned workset interval")
        if path.is_file():
            read_fragment(path, ids, expected, runtime)
            resumed += 1
        else:
            if backbone is None:
                tick = perf_counter()
                backbone = load_backbone(args.bert_dir, group)
                group.backend.synchronize(group.device)
                load_seconds = perf_counter() - tick
            tick = perf_counter()
            inputs = tokenizer.pad([{"input_ids": row.tolist()} for row in rows], return_tensors="pt",
                                   padding=True, pad_to_multiple_of=64).to(group.device)
            if tuple(inputs["input_ids"].shape) != (len(ids), expected["padded_width"]):
                raise ValueError(f"Unexpected on-device BERT layout at global batch {batch}")
            with torch.no_grad():
                hidden = backbone(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                values = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).cpu().numpy()
            elapsed = perf_counter() - tick
            encode_seconds[str(expected["padded_width"])] += elapsed
            if values.shape != (len(ids), WIDTH) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid pooled BERT output at global batch {batch}")
            tick = perf_counter()
            write_fragment(path, ids, values, expected, runtime)
            write_seconds += perf_counter() - tick
            encoded += 1
            del hidden, mask
        if batch in (start, end - 1) or (batch - start + 1) % 256 == 0:
            print(json.dumps({"phase": "batch", **expected, "real_texts": len(ids), "encoded": encoded,
                              "resumed": resumed, "wall_seconds": perf_counter() - started}), flush=True)
    report = {"phase": "shard_complete", "axis_sha256": axis.digest, "encoding_runtime": runtime,
              "worker_count": args.worker_count, "partition_index": args.partition_index,
              "partition_first_batch": first, "partition_stop_batch_exclusive": stop,
              "first_batch": start, "stop_batch_exclusive": end, "encoded": encoded, "resumed": resumed,
              "real_texts": min(end * BATCH_SIZE, TEXT_COUNT) - start * BATCH_SIZE, "load_seconds": load_seconds,
              "encode_seconds_by_width": dict(encode_seconds), "write_seconds": write_seconds,
              "peak_accelerator_bytes": int(group.backend.max_memory_allocated(group.device)),
              "wall_seconds": perf_counter() - started}
    footer = args.output / f"worker-{args.worker_count:02d}-{args.partition_index:02d}-{start:05d}-{end:05d}.json"
    with NamedTemporaryFile(dir=args.output, prefix=f".{footer.name}-", suffix=".part", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.close()
            write_json(pending, report)
            pending.replace(footer)
        finally:
            pending.unlink(missing_ok=True)
    print(json.dumps({**report, "footer": str(footer)}), flush=True)


def merge(args):
    started = perf_counter()
    axis = BatchAxis(args.axis)
    expected = {fragment_path(args.fragments, index) for index in range(BATCH_COUNT)}
    available = set(args.fragments.glob("*.npz"))
    if expected != available:
        missing = [path.name for path in sorted(expected - available)]
        extra = [path.name for path in sorted(available - expected)]
        raise ValueError(f"Expected dense global fragment set; missing {len(missing)} {missing[:8]}, "
                         f"unexpected {len(extra)} {extra[:8]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as stage_dir:
        stage = Path(stage_dir)
        matrix = np.lib.format.open_memmap(stage / MATRIX, "w+", dtype=np.float32, shape=(TEXT_COUNT, WIDTH))
        fragments_digest = sha256()
        runtime = None
        checked_rows = np.zeros(TEXT_COUNT, dtype=bool)
        widths = Counter()
        for batch in range(BATCH_COUNT):
            ids, _rows, fingerprint = axis.batch(batch)
            path = fragment_path(args.fragments, batch)
            values, receipt = read_fragment(path, ids, fingerprint, runtime)
            if runtime is None:
                runtime = receipt["encoding_runtime"]
            if checked_rows[ids].any():
                raise ValueError(f"Repeated exact text IDs across fragment {batch}")
            matrix[ids] = values
            checked_rows[ids] = True
            widths[str(fingerprint["padded_width"])] += 1
            fragments_digest.update(bytes.fromhex(file_sha256(path)))
        if not checked_rows.all() or widths != axis.receipt["input_histogram"]["batches_per_padded_width"]:
            raise ValueError("Merged training rows or original B32 width histogram are incomplete")
        matrix.flush()
        del matrix
        copyfile(args.axis / INDEX, stage / INDEX)
        report = {"schema": CACHE_SCHEMA, "policy": POLICY, "axis_sha256": axis.digest, "model": MODEL,
                  "tokenizer": axis.receipt["tokenizer"], "encoding_runtime": runtime, "input_histogram":
                      axis.receipt["input_histogram"], "ordered_fragment_files_sha256": fragments_digest.hexdigest(),
                  "files_sha256": {name: file_sha256(stage / name) for name in (MATRIX, INDEX)},
                  "matrix_shape": [TEXT_COUNT, WIDTH], "matrix_dtype": "float32"}
        if report["files_sha256"][INDEX] != axis.receipt["files_sha256"][INDEX]:
            raise ValueError("Input text mapping differs from its prepared axis receipt")
        write_json(stage / "receipt.json", report)
        stage.rename(args.output)
    print(json.dumps({"phase": "merged", "receipt_sha256": file_sha256(args.output / "receipt.json"),
                      "wall_seconds": perf_counter() - started, **report}), flush=True)


def replay_cad(root):
    """Read 32 registered real T training CAD inputs for the optional cosine witness."""
    summary = json.loads((root / "summary.json").read_bytes())
    name = f"{SOURCE}_train_indices.jsonl"
    checksums = summary["files_sha256"]
    if summary["representation"] != "command-dual-tower":
        raise ValueError("Replay needs the shared command dual-tower CAD feature store")
    for entry in (name, "coverage.jsonl"):
        if file_sha256(root / entry) != checksums[entry]:
            raise ValueError(f"Replay CAD inputs differ from their receipt: {entry}")
    selected = {}
    with (root / name).open(encoding="utf-8") as stream:
        for _ in range(BATCH_SIZE):
            row = json.loads(next(stream))
            selected[row["feature_index"]] = row["uid"]
    store = np.load(root / "cad_features.npy", mmap_mode="r", allow_pickle=False)
    if (store.dtype != np.int32 or store.shape != (summary["axis_count"], MAX_CAD_SEQUENCE_LENGTH, 4)
            or len(selected) != BATCH_SIZE or min(selected) < 0 or max(selected) >= len(store)):
        raise ValueError("Replay CAD input tensor has the wrong shape or dtype")
    last = max(selected)
    checked = 0
    with (root / "coverage.jsonl").open(encoding="utf-8") as stream:
        for idx, line in enumerate(stream):
            if idx > last:
                break
            if idx in selected:
                row = json.loads(line)
                if (row["uid"] != selected[idx] or row["status"] != "ok"
                        or row["feature_sha256"] != sha256(store[idx].tobytes()).hexdigest()):
                    raise ValueError(f"Invalid replay CAD coverage for training UID {selected[idx]}")
                checked += 1
    if checked != len(selected):
        raise ValueError("Incomplete replay CAD coverage for the selected training rows")
    return torch.from_numpy(store[list(selected)]), list(selected.values())


def replay(args):
    """Compare stored pools to the unchanged native reference on the exact original batch."""
    axis = BatchAxis(args.axis)
    batches = sorted(set(args.batch))
    examples = {batch: axis.batch(batch) for batch in batches}
    wanted = {int(idx) for ids, _, _ in examples.values() for idx in ids}
    last = max(wanted)
    examples_text = {}
    for idx, (text, _) in enumerate(read_index(args.axis / INDEX)):
        if idx in wanted:
            examples_text[idx] = text
        if idx == last:
            break
    if set(examples_text) != wanted:
        raise ValueError("Incomplete exact training text index for replay")
    group = accelerator(args)
    runtime = encoding_runtime(group)
    torch.manual_seed(args.seed)
    group.backend.manual_seed_all(args.seed)
    config = SimpleNamespace(method="dual-tower", bert_dir=args.bert_dir, precision="fp32", attention="eager",
                             command_convolution="linear")
    towers, backbone, tokenizer, _, _ = load_towers(config, group, None)
    towers.eval()
    if tokenizer_identity(tokenizer) != axis.receipt["tokenizer"]:
        raise ValueError("Replay tokenizer differs from the prepared training axis")
    cad_values, cad_uids = None, None
    if args.shared_feature_store is not None:
        cad_inputs, cad_uids = replay_cad(args.shared_feature_store)
        with torch.no_grad():
            cad_values = towers.encode_cad(cad_inputs.to(group.device))
    reports = []
    for batch, (ids, token_rows, expected) in examples.items():
        texts = [examples_text[int(idx)] for idx in ids]
        direct = TextInputs(tokenizer, texts, "", 512, group.device, pad_multiple=64, native_truncation=True)
        for text, row in zip(texts, token_rows, strict=True):
            if direct.ids[text] != row.tolist():
                raise ValueError(f"Direct token IDs differ at global BERT batch {batch}")
        cached, receipt = read_fragment(fragment_path(args.fragments, batch), ids, expected, runtime)
        with torch.no_grad():
            reference, _ = direct.frozen(backbone, texts, BATCH_SIZE)
            values = torch.from_numpy(cached).to(group.device)
            comparisons = {"pooled": (values, reference),
                           "projected": (towers.encode_text(values), towers.encode_text(reference))}
            if cad_values is not None:
                actual, baseline = comparisons["projected"]
                comparisons["retrieval_cosine"] = (actual @ cad_values.T, baseline @ cad_values.T)
        metrics = {}
        for field, (actual, expected_value) in comparisons.items():
            actual, expected_value = actual.cpu(), expected_value.cpu()
            torch.testing.assert_close(actual, expected_value, rtol=1e-5, atol=1e-6)
            metrics[field] = {"shape": list(actual.shape), "bitwise_equal": torch.equal(actual, expected_value),
                              "max_abs_error": float((actual - expected_value).abs().max())}
        reports.append({**receipt, "real_texts": len(ids), "comparisons": metrics})
    report = {"phase": "replay_passed", "axis_sha256": axis.digest, "rtol": 1e-5, "atol": 1e-6,
              "projection_seed": args.seed, "training_cad_uids": cad_uids, "batches": reports}
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Compile only the pinned T training texts into the fixed B32 axis")
    encode = commands.add_parser("shard", help="Pool one partition or a bounded interval with native BERT")
    combine = commands.add_parser("merge", help="Verify every committed batch and publish one dense matrix")
    compare = commands.add_parser("replay", help="Compare selected committed batches with direct native pooling")
    for name in ("manifest-root", "bert-dir", "cost-summary", "output"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    for name in ("axis", "bert-dir", "output"):
        encode.add_argument(f"--{name}", type=Path, required=True)
    for name in ("axis", "fragments", "output"):
        combine.add_argument(f"--{name}", type=Path, required=True)
    for name in ("axis", "fragments", "bert-dir"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--shared-feature-store", type=Path, help="Include native cosine scores on 32 T training CADs")
    compare.add_argument("--batch", action="append", required=True, type=int, help="Global batch ID, repeatable")
    compare.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    for command in (prep, encode):
        command.add_argument("--workset", type=Path, default=WORKSET)
    for command in (encode, compare):
        command.add_argument("--device", default="npu:0")
    encode.add_argument("--worker-count", required=True, type=int, choices=(1, 2, 4, 8))
    encode.add_argument("--partition-index", required=True, type=int)
    encode.add_argument("--batch-start", type=int, help="Optional global inclusive bound inside the partition")
    encode.add_argument("--batch-stop", type=int, help="Optional global exclusive bound inside the partition")
    prep.set_defaults(run=prepare)
    encode.set_defaults(run=shard)
    combine.set_defaults(run=merge)
    compare.set_defaults(run=replay)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
