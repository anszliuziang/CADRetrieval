#!/usr/bin/env python3
"""Train a neural retriever on frozen pilot or formal train and validation axes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.src.idea01.in_domain_pilot_retrievers import (
    MAX_CAD_SEQUENCE_LENGTH,
    OUTPUT_DIM,
    SPECTRA_COMMIT,
    SPECTRA_DESCRIPTOR_DIM,
    SPECTRA_SOURCE_SHA256,
    SPECTRA_TREE,
    CommandTower,
    direct_registration,
    known_positive_counts,
    load_spectra_modules,
    prepare_cad_features,
    registered_positive_mask,
    symmetric_positive_set_loss,
    verify_spectra_source,
)

LEVELS = {"t2cad-v1.0": ("L0", "L1", "L2", "L3"), "ct-deepcad": ("CT",)}
BERT_REVISION = "6da4b6a26a1877e173fca3225479512db81a5e5b"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_FILES = {
    "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
MANIFESTS = ("pilot_train.jsonl", "validation_targets.jsonl", "validation_gallery.jsonl")
TEMPERATURE = 0.05
SEED = 17
FORMAL_MANIFEST_RECEIPT_SHA = "3c297920d0905b3dd5f14aa1b5ac163245a17b958cc3c2e1169ee65d5a8d055d"
FORMAL_SPECTRA_STORE_SHA = "35b4c86ac6bd33cc9ff1359ebd7a7449f2e93f6475005792040f6b2ab6d77f0b"
FORMAL_TEST_SCOPE = {"test_queries_opened": False, "test_qrels_opened": False, "test_scoring_executed": False}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def manifest_axes(args):
    names = ("train.jsonl", *MANIFESTS[1:]) if args.train_split == "full" else MANIFESTS
    paths = [args.manifest_root / args.source / filename for filename in names]
    files = {path.name: file_sha256(path) for path in paths}
    if args.train_split == "full":
        receipt = args.manifest_root / "manifest_receipt.json"
        data = receipt.read_bytes()
        if sha256(data).hexdigest() != FORMAL_MANIFEST_RECEIPT_SHA:
            raise ValueError("Formal training needs the certified DeepCAD manifest receipt")
        pins = json.loads(data)["file_sha256"]
        if any(pins[f"{args.source}/{name}"] != checksum for name, checksum in files.items()):
            raise ValueError("A frozen formal train or validation manifest changed")
    train, targets, gallery = (read_rows(path) for path in paths)
    levels = LEVELS[args.source]
    axes = zip(paths, (train, targets, gallery),
               (args.expected_train_uids, args.expected_targets, args.expected_gallery), strict=True)
    for path, rows, count in axes:
        if len(rows) != count or len({row["uid"] for row in rows}) != count:
            raise ValueError(f"Expected {count} unique UIDs in {path}")
    for row in train + targets:
        if set(row["texts"]) != set(levels) or any(
            not isinstance(text, str) or not text or text != text.strip() for text in row["texts"].values()
        ):
            raise ValueError(f"Incomplete or invalid native texts for {row['uid']}")
    train_uids = {row["uid"] for row in train}
    gallery_uids = {row["uid"] for row in gallery}
    if train_uids & gallery_uids or {row["uid"] for row in targets} - gallery_uids:
        raise ValueError("Train and validation must be disjoint, and all targets must be in the gallery")
    if len({row["cad_index"] for row in gallery}) != len(gallery):
        raise ValueError("Validation gallery cad_index must be unique")
    queries = [
        {"uid": row["uid"], "level": level, "text": row["texts"][level]}
        for level in levels for row in targets
    ]
    return train, queries, gallery, files


def cad_cache(args, rows, files, spectra):
    identity = {
        "source": args.source,
        "method": args.method,
        "manifest_files": files,
        "official_zip": str(args.official_zip.resolve()),
        "official_zip_bytes": args.official_zip.stat().st_size,
        "spectrabrep_source_sha256": SPECTRA_SOURCE_SHA256 if spectra is not None else None,
    }
    cached = args.prepared_cad
    if cached is not None:
        prepared = json.loads((cached / "cad_cache.json").read_text(encoding="utf-8"))
        if prepared["identity"] != identity:
            raise ValueError("Prepared features were built for different CAD inputs")
        for name, expected in prepared["files"].items():
            if file_sha256(cached / name) != expected:
                raise ValueError(f"Prepared CAD artifact differs from its receipt: {name}")
        features = np.load(cached / "cad_features.npy", allow_pickle=False)
        coverage = read_rows(cached / "cad_coverage.jsonl")
        if list((row["uid"], row["cad_member"]) for row in coverage) != [
            (row["uid"], row["cad_member"]) for row in rows
        ]:
            raise ValueError("Prepared feature axis differs from the current manifest")
        write_rows(args.output / "cad_coverage.jsonl", coverage)
        write_json(args.output / "cad_cache_input.json", {"path": str(cached), **prepared})
        return features, coverage
    features, coverage = prepare_cad_features(
        rows, args.official_zip, workers=args.workers, spectra=spectra, report_progress=True,
    )
    np.save(args.output / "cad_features.npy", features, allow_pickle=False)
    write_rows(args.output / "cad_coverage.jsonl", coverage)
    write_json(args.output / "cad_cache.json", {
        "identity": identity,
        "files": {name: file_sha256(args.output / name) for name in ("cad_features.npy", "cad_coverage.jsonl")},
        "coverage": dict(Counter(row["status"] for row in coverage)),
    })
    return features, coverage


def shared_cad_cache(args, train, gallery):
    """Verify the combined input-only store and materialize only the requested two roles."""
    from experiments.src.idea01.in_domain_pilot import INPUT_HASHES

    root = args.shared_feature_store
    summary_data = (root / "summary.json").read_bytes()
    summary = json.loads(summary_data)
    roles = ((args.source + "_train", train), (args.source + "_validation_gallery", gallery))
    files = ("cad_features.npy", "feature_axis.jsonl", "coverage.jsonl",
             *(f"{role}_indices.jsonl" for role, _ in roles))
    pins = summary["files_sha256"]
    spectra = args.method == "spectrabrep"
    if spectra and sha256(summary_data).hexdigest() != FORMAL_SPECTRA_STORE_SHA:
        raise ValueError("SpectraBRep requires the certified DeepCAD descriptor summary")
    if (summary["representation"] != ("spectrabrep" if spectra else "command-dual-tower")
            or summary["cad_zip_sha256"] != INPUT_HASHES["official_zip"]
            or file_sha256(args.official_zip) != INPUT_HASHES["official_zip"]):
        raise ValueError("The shared store does not match the requested representation and certified CAD archive")
    for name in files:
        if name not in pins or file_sha256(root / name) != pins[name]:
            raise ValueError(f"Shared feature artifact differs from its receipt: {name}")
    axis = read_rows(root / "feature_axis.jsonl")
    coverage = read_rows(root / "coverage.jsonl")
    store = np.load(root / "cad_features.npy", mmap_mode="r", allow_pickle=False)
    dimensions = (SPECTRA_DESCRIPTOR_DIM,) if spectra else (MAX_CAD_SEQUENCE_LENGTH, 4)
    expected = (summary["axis_count"], *dimensions)
    if (store.shape != expected or list(store.shape) != summary["feature_shape"]
            or store.dtype != np.dtype("float32" if spectra else "int32") or str(store.dtype) != summary["feature_dtype"]
            or len(axis) != len(coverage) or len(axis) != expected[0]):
        raise ValueError("The shared feature tensor and registered CAD axes differ")
    uids = [row["uid"] for row in axis]
    if uids != sorted(set(uids)) or [row["uid"] for row in coverage] != uids:
        raise ValueError("Unexpected UID order or duplicates in the shared feature store")
    indices, subset = [], []
    for role, manifest in roles:
        role_rows = read_rows(root / f"{role}_indices.jsonl")
        if [row["uid"] for row in role_rows] != [row["uid"] for row in manifest]:
            raise ValueError(f"Shared feature role does not match the ordered {role} manifest")
        counts = Counter()
        for located, expected_row in zip(role_rows, manifest, strict=True):
            index = located["feature_index"]
            identity = {key: expected_row[key] for key in ("uid", "cad_member", "member_sha256")}
            if type(index) is not int or not 0 <= index < len(axis) or axis[index] != identity:
                raise ValueError(f"Incorrect shared feature identity for {role}/{located['uid']}")
            entry = coverage[index]
            if (entry["cad_member"] != identity["cad_member"]
                    or (entry["status"] == "ok" and (
                        entry.get("member_sha256") != identity["member_sha256"]
                        or sha256(store[index].tobytes()).hexdigest() != entry.get("feature_sha256")
                    ))):
                raise ValueError(f"Incorrect shared feature coverage for {role}/{located['uid']}")
            indices.append(index)
            subset.append(entry)
            counts[entry["status"]] += 1
        if counts != summary["role_coverage"][role]:
            raise ValueError(f"Shared feature coverage counts differ for {role}")
    selected = np.asarray(indices, dtype=np.int64)
    np.save(args.output / "cad_feature_indices.npy", selected, allow_pickle=False)
    write_rows(args.output / "cad_coverage.jsonl", subset)
    write_json(args.output / "cad_cache_input.json", {
        "path": str(root.resolve()), "summary_sha256": sha256(summary_data).hexdigest(),
        "representation": summary["representation"], "cad_zip_sha256": summary["cad_zip_sha256"],
        "files_sha256": {name: pins[name] for name in files},
        "roles": {role: len(rows) for role, rows in roles},
    })
    return store[selected], subset


@dataclass(frozen=True)
class DeviceGroup:
    device: torch.device
    backend: object
    rank: int
    world_size: int

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()


def initialize_devices(label: str, *, keep_origin_dtype: bool = False) -> DeviceGroup:
    kind = label.partition(":")[0]
    if kind == "npu":
        import torch_npu  # noqa: F401  Registers the Ascend device and HCCL backend.

        backend = torch.npu
        if keep_origin_dtype:
            backend.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    else:
        backend = torch.cuda
    if not backend.is_available():
        raise RuntimeError(f"Requested accelerator is unavailable: {label}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and label != kind:
        raise ValueError("Under torchrun use --device npu or --device cuda, with local rank selecting the device")
    device = torch.device(label if ":" in label else f"{kind}:{os.environ.get('LOCAL_RANK', '0')}")
    backend.set_device(device)
    if world_size > 1:
        dist.init_process_group("hccl" if kind == "npu" else "nccl")
    rank = dist.get_rank() if world_size > 1 else 0
    return DeviceGroup(device, backend, rank, world_size)


class GatherEmbeddings(torch.autograd.Function):
    """Equal-sized rank slices, with summed incoming gradients before DDP averaging."""

    @staticmethod
    def forward(ctx, values):
        chunks = [torch.empty_like(values) for _ in range(dist.get_world_size())]
        dist.all_gather(chunks, values.contiguous())
        return torch.cat(chunks, dim=0)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient)
        return gradient.chunk(dist.get_world_size(), dim=0)[dist.get_rank()].contiguous()


class RetrievalTowers(nn.Module):
    def __init__(self, text: nn.Module, cad: nn.Module):
        super().__init__()
        self.text = text
        self.cad = cad

    def encode_text(self, inputs):
        if isinstance(inputs, torch.Tensor):
            value = self.text(inputs)
        else:
            value = self.text(inputs["input_ids"], inputs["attention_mask"])
        return F.normalize(value.float(), dim=-1)

    def encode_cad(self, inputs):
        return F.normalize(self.cad(inputs).float(), dim=-1)

    def forward(self, text_inputs, cad_inputs):
        return self.encode_text(text_inputs), self.encode_cad(cad_inputs)


def load_towers(args, group, spectra):
    from transformers import AutoModel, AutoTokenizer

    spectra_method = args.method == "spectrabrep"
    qwen = spectra_method or args.bert_dir is None
    path = args.qwen_dir if qwen else args.bert_dir
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    if qwen:
        for name, expected in QWEN_FILES.items():
            if file_sha256(path / name) != expected:
                raise ValueError(f"Qwen asset differs from pinned {QWEN_REVISION}: {path / name}")
        model_id, revision, file_hashes = "Qwen/Qwen3-Embedding-0.6B", QWEN_REVISION, QWEN_FILES
    else:
        from experiments.src.idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256, verify_bert_asset_shas

        verify_bert_asset_shas(path)
        model_id, revision, file_hashes = "google-bert/bert-large-uncased", BERT_REVISION, PINNED_BERT_FILE_SHA256

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    tokenizer.padding_side = "left" if spectra_method else "right"
    model_source = {"repo": model_id, "revision": revision, "path": str(path.resolve()), "files": file_hashes}
    if spectra_method:
        upstream = load_spectra_modules(spectra, ("config", "model", "runtime"), attention=args.attention)
        text = upstream["model"].TextEncoder(str(path), lora_rank=16, lora_alpha=32, dtype=dtype)
        illegal = [name for name, value in text.backbone.named_parameters()
                   if value.requires_grad and "lora_" not in name]
        if illegal:
            raise RuntimeError(f"SpectraBRep backbone trainable parameters outside LoRA: {illegal[:5]}")
        backbone = None
        cad = upstream["model"].CADTower()
        model_source["actual_backbone_dtypes"] = sorted({
            str(value.dtype) for value in text.backbone.parameters() if not value.requires_grad
        })
    else:
        upstream = None
        backbone = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
            attn_implementation=args.attention,
        ).to(group.device).eval().requires_grad_(False)
        width = backbone.config.hidden_size
        text = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, OUTPUT_DIM))
        cad = CommandTower(convolution=args.command_convolution)
        model_source["actual_backbone_dtypes"] = sorted({str(value.dtype) for value in backbone.parameters()})
    if model_source["actual_backbone_dtypes"] != [str(dtype)]:
        raise RuntimeError(f"Loaded backbone dtype differs from the requested {dtype}: {model_source}")
    return RetrievalTowers(text, cad).to(group.device), backbone, tokenizer, upstream, model_source


class TextInputs:
    def __init__(self, tokenizer, texts, prefix, length, device, *, pad_multiple=None, native_truncation=False,
                 chunk_size=None):
        self.tokenizer = tokenizer
        self.device = device
        self.length = length
        self.pad_multiple = pad_multiple
        self.rows = []
        self.ids = {}
        self.overlength = set()
        size = chunk_size or max(1, len(texts))
        for start in range(0, len(texts), size):
            self._add_texts(tokenizer, texts[start:start + size], prefix, length, native_truncation)

    def _add_texts(self, tokenizer, texts, prefix, length, native_truncation):
        ids = tokenizer(
            [prefix + text for text in texts], padding=False, truncation=False,
            return_attention_mask=False, verbose=False,
        )["input_ids"]
        truncated = {}
        if native_truncation:
            long_texts = [text for text, row in zip(texts, ids, strict=True) if len(row) > length]
            if long_texts:
                clipped = tokenizer([prefix + text for text in long_texts], padding=False, truncation=True,
                                    max_length=length, return_attention_mask=False, verbose=False)["input_ids"]
                truncated = dict(zip(long_texts, clipped, strict=True))
        for text, row in zip(texts, ids, strict=True):
            if native_truncation:
                used = truncated.get(text, row)
            else:
                used = row[-length:] if tokenizer.truncation_side == "left" else row[:length]
            self.ids[text] = used
            if len(row) > length:
                if not 0 < len(used) <= length:
                    raise ValueError("Tokenizer returned IDs outside the declared context")
                self.overlength.add(text)
            self.rows.append({
                "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "input_token_count": len(row),
                "effective_token_count": len(used),
                "truncated": len(used) < len(row),
            })

    def batch(self, texts):
        return self.tokenizer.pad(
            [{"input_ids": self.ids[text]} for text in texts], return_tensors="pt", padding=True,
            pad_to_multiple_of=self.pad_multiple,
        ).to(self.device)

    def frozen(self, backbone, texts, batch_size, *, report_progress=False):
        """Masked means from a fixed eval-mode model, indexed by exact text."""
        indexes = {text: index for index, text in enumerate(texts)}
        length_order = sorted(texts, key=lambda text: len(self.ids[text]))
        length_index = {text: index for index, text in enumerate(length_order)}
        values = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                inputs = self.batch(length_order[start:start + batch_size])
                hidden = backbone(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                values.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
                if report_progress and (start == 0 or len(values) % 256 == 0):
                    print(f"Frozen BERT inputs encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(values)[[length_index[text] for text in texts]], indexes


def exposure_rows(train, levels, counters):
    for row in train:
        counts = {level: counters[row["uid"], level] for level in levels}
        yield {"uid": row["uid"], "pair_exposures": sum(counts.values()), "level_exposures": counts}


def training_batches(per_rank, world_size, pair_count, spectra):
    batch_size = per_rank * world_size
    batches = [(start, min(start + batch_size, pair_count)) for start in range(0, pair_count, batch_size)]
    if (len(batches) < 2 or any(end - start < 2 or (end - start) % world_size for start, end in batches)
            or (spectra and len(batches) == 10)):
        raise ValueError(f"Unsupported batch layout for {pair_count} pairs, {world_size} ranks and batch {per_rank}. "
                         "Each of at least two updates needs equal nonempty rank slices and two global pairs. "
                         "SpectraBRep also needs a nondegenerate OneCycle schedule.")
    return batches


def training_order(pair_count, seed):
    return torch.randperm(pair_count, generator=torch.Generator().manual_seed(seed)).tolist()


def spectra_timing_prefix(batches, updates):
    if updates is None:
        return batches
    if not 2 <= updates < len(batches):
        raise ValueError("SpectraBRep timing needs a strict prefix of at least two full-run updates")
    return batches[:updates]


def valid_formal_spectra_layout(batch_size, world_size, timing_updates):
    if world_size == 1:
        return batch_size in (None, 16)
    return timing_updates is not None and world_size == 4 and batch_size == 4


def fit(args, group, towers, backbone, text_inputs, cad, pairs, upstream, cached_frozen=None):
    all_texts = list(dict.fromkeys(row["text"] for row in pairs))
    registration = direct_registration((row["text"], row["uid"]) for row in pairs)
    frozen = None
    if cached_frozen is not None:
        frozen = torch.empty(cached_frozen.shape, dtype=torch.float32, device=group.device)
        for start in range(0, len(cached_frozen), 8192):
            rows = cached_frozen[start:start + 8192]
            frozen[start:start + len(rows)].copy_(torch.from_numpy(rows.copy()))
        frozen_index = {text: index for index, text in enumerate(all_texts)}
    elif backbone is not None:
        frozen, frozen_index = text_inputs.frozen(backbone, all_texts, args.eval_batch_size,
                                                   report_progress=group.primary and args.train_split == "full")
    if group.world_size > 1:
        runner = DistributedDataParallel(towers, device_ids=[group.device.index])
    else:
        runner = towers
    lora = [parameter for name, parameter in towers.named_parameters()
            if parameter.requires_grad and ".backbone." in name]
    heads = [parameter for name, parameter in towers.named_parameters()
             if parameter.requires_grad and ".backbone." not in name]
    groups = [{"params": heads, "lr": 3e-4}]
    if lora:
        groups.insert(0, {"params": lora, "lr": 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)
    batches = training_batches(args.batch_size, group.world_size, len(pairs), upstream is not None)
    scheduler = None
    if upstream is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, [item["lr"] for item in groups], total_steps=len(batches), pct_start=0.1,
        )
    order = training_order(len(pairs), args.seed)
    witness = args.spectra_update_witness
    timing_updates = getattr(args, "spectra_timing_updates", None)
    active_batches = batches[:1] if witness else spectra_timing_prefix(batches, timing_updates)
    if witness:
        from experiments.src.idea01.formal_spectra_witness import parameter_probes, update_evidence

        probes = parameter_probes(towers)
    runner.train()
    stats = Counter()
    exposure = Counter()
    sum_loss = sum_pair_loss = 0.0
    stream = (args.output / "training_steps.jsonl").open("w", encoding="utf-8", newline="\n") if group.primary else None
    try:
        train_start = perf_counter()
        for step, (start, stop) in enumerate(active_batches, 1):
            if timing_updates is not None:
                step_start = perf_counter()
            global_rows = [pairs[index] for index in order[start:stop]]
            texts = [row["text"] for row in global_rows]
            uids = [row["uid"] for row in global_rows]
            mask = registered_positive_mask(texts, uids, registration)
            counts = known_positive_counts(mask, texts, uids)
            local_size = len(global_rows) // group.world_size
            index = group.rank * local_size
            local = global_rows[index:index + local_size]
            local_texts = texts[index:index + local_size]
            if frozen is None:
                model_inputs = text_inputs.batch(local_texts)
            else:
                model_inputs = frozen[[frozen_index[text] for text in local_texts]]
            local_cad = cad[[row["cad_row"] for row in local]]
            text_embeddings, cad_embeddings = runner(model_inputs, local_cad)
            if group.world_size > 1:
                gather = upstream["runtime"].gather_with_gradient if upstream is not None else GatherEmbeddings.apply
                text_embeddings = gather(text_embeddings)
                cad_embeddings = gather(cad_embeddings)
            if upstream is None:
                logits = (text_embeddings @ cad_embeddings.T) / TEMPERATURE
                objective = symmetric_positive_set_loss(logits, mask.to(group.device))
            else:
                objective = upstream["runtime"].symmetric_info_nce(text_embeddings, cad_embeddings, TEMPERATURE)
            loss = float(objective.detach())
            if not math.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at update {step}")
            objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(lora + heads, 1.0, error_if_nonfinite=True, foreach=False)
            lr = [item["lr"] for item in optimizer.param_groups]
            optimizer.step()
            if witness:
                changes = update_evidence(probes)
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            exposure.update((row["uid"], row["level"]) for row in global_rows)
            stats.update(counts)
            sum_loss += loss
            sum_pair_loss += loss * len(global_rows)
            if stream is not None:
                row = {
                    "update": step, "global_pairs": len(global_rows), "pairs_per_rank": local_size,
                    "train_pair_indices": order[start:stop], "loss": loss, "learning_rates": lr,
                    "preclip_grad_norm": float(grad_norm), **counts,
                }
                if timing_updates is not None:
                    group.backend.synchronize(group.device)
                    row.update(seconds=perf_counter() - step_start, text_width=model_inputs["input_ids"].shape[1])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if step == 1 or step % 20 == 0 or step == len(batches):
                    print(f"{args.source}/{args.method} update {step}/{len(batches)} loss={loss:.5f}", flush=True)
        group.backend.synchronize(group.device)
        train_seconds = perf_counter() - train_start
    finally:
        if stream is not None:
            stream.close()
    if witness:
        return exposure, {
            "actual_updates": 1, "actual_training_pairs": len(global_rows),
            "full_epoch_planned_updates": len(batches), "global_pair_indices": order[start:stop],
            "training_loss": loss, "preclip_grad_norm": float(grad_norm), "learning_rates": lr,
            "learning_rates_after_scheduler_step": scheduler.get_last_lr(), "scheduler_last_epoch": scheduler.last_epoch,
            "optimizer_loop_seconds": train_seconds, "parameter_changes": changes,
            "known_off_diagonal_positives_treated_as_negatives_each_direction": counts["known_off_diagonal_positives"],
        }
    if timing_updates is not None:
        return exposure, {
            "actual_updates": len(active_batches), "actual_training_pairs": sum(exposure.values()),
            "full_epoch_planned_updates": len(batches), "optimizer_loop_seconds": train_seconds,
            "scheduler_last_epoch": scheduler.last_epoch,
        }
    if any(value != 1 for value in exposure.values()) or len(exposure) != len(pairs):
        raise RuntimeError("The epoch must expose each registered UID and level exactly once")
    if upstream is not None:
        stats["known_negative_treatment_text_to_cad"] = stats["known_off_diagonal_positives"]
        stats["known_negative_treatment_cad_to_text"] = stats["known_off_diagonal_positives"]
    return exposure, {
        "actual_updates": len(batches), "actual_training_pairs": sum(exposure.values()),
        "actual_training_uids": len({uid for uid, _ in exposure}),
        "actual_distinct_training_texts": len(all_texts),
        "training_texts_registered_to_multiple_uids": sum(len(uids) > 1 for uids in registration.values()),
        "mean_step_loss": sum_loss / len(batches),
        "mean_pair_weighted_loss": sum_pair_loss / len(pairs),
        "optimizer_loop_seconds": train_seconds,
        "known_positive_audit": dict(stats),
    }


def score(args, group, towers, backbone, text_inputs, cad, queries, covered):
    towers.eval()
    eligible = [index for index, row in enumerate(queries)
                if not args.require_full_validation_text or row["text"] not in text_inputs.overlength]
    texts = [queries[index]["text"] for index in eligible]
    unique = list(dict.fromkeys(texts))
    text_locations = {text: index for index, text in enumerate(unique)}
    valid_queries = np.zeros(len(queries), dtype=bool)
    scores = np.full((len(queries), len(covered)), np.nan, dtype=np.float32)
    with torch.no_grad():
        if unique:
            frozen = backbone is not None
            if frozen:
                text_values, _ = text_inputs.frozen(backbone, unique, args.eval_batch_size)
            text_embeddings = []
            for start in range(0, len(unique), args.eval_batch_size):
                stop = start + args.eval_batch_size
                value = text_values[start:stop] if frozen else text_inputs.batch(unique[start:stop])
                text_embeddings.append(towers.encode_text(value))
            query_values = torch.cat(text_embeddings)[[text_locations[text] for text in texts]]
            valid_queries[eligible] = torch.isfinite(query_values).all(dim=1).cpu().numpy()
        cad_values = torch.full((len(covered), OUTPUT_DIM), torch.nan, device=group.device)
        valid = [index for index, value in enumerate(covered) if value]
        for start in range(0, len(valid), args.eval_batch_size):
            selected = valid[start:start + args.eval_batch_size]
            cad_values[selected] = towers.encode_cad(cad[selected])
        valid_candidates = torch.isfinite(cad_values).all(dim=1).cpu().numpy()
        if unique:
            scores[eligible, :] = (query_values @ cad_values.T).cpu().numpy()
    scores[~valid_queries, :] = np.nan
    scores[:, ~valid_candidates] = np.nan
    return scores, valid_queries, valid_candidates


def runtime_versions():
    versions = {}
    for package in ("torch", "torch-npu", "transformers", "peft", "numpy", "scipy", "safetensors", "tokenizers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version.split()[0], "packages": versions}


def run(args, group, axes, features, coverage, spectra, started):
    from experiments.src.idea01.in_domain_pilot import evaluate_rankings

    train, queries, gallery, files = axes
    formal = args.train_split == "full"
    train_size = len(train)
    levels = LEVELS[args.source]
    pairs = [
        {"uid": row["uid"], "level": level, "text": row["texts"][level], "cad_row": index}
        for index, row in enumerate(train) for level in levels
    ]
    train_ok = sum(row["status"] == "ok" for row in coverage[:train_size])
    val_covered = [row["status"] == "ok" for row in coverage[train_size:]]
    if train_ok != train_size:
        raise ValueError(f"Only {train_ok}/{train_size} training CAD encoded; see cad_coverage.jsonl. "
                         "Resolve the common training manifest before optimizer updates.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    group.backend.manual_seed_all(args.seed)
    group.backend.reset_peak_memory_stats(group.device)
    phase_start = perf_counter()
    towers, backbone, tokenizer, upstream, model_source = load_towers(args, group, spectra)
    witness = args.spectra_update_witness
    timing_updates = getattr(args, "spectra_timing_updates", None)
    if witness or timing_updates is not None:
        prefix_pairs = args.batch_size * group.world_size * (timing_updates or 1)
        prefix_order = training_order(len(pairs), args.seed)[:prefix_pairs]
        all_texts = list(dict.fromkeys(pairs[i]["text"] for i in prefix_order))
    else:
        all_texts = list(dict.fromkeys(row["text"] for row in pairs + queries))
    prefix = upstream["config"].INSTRUCTION if upstream is not None else ""
    window = 384 if upstream is not None else 512
    cached_frozen = cache_source = None
    if getattr(args, "training_bert_cache", None) is None:
        text_inputs = TextInputs(tokenizer, all_texts, prefix, window, group.device,
                                 pad_multiple=64 if formal else None, native_truncation=formal,
                                 chunk_size=1024 if formal and upstream is not None else None)
    else:
        from experiments.src.idea01.formal_command_bert_cache import open_training_cache

        training_texts = list(dict.fromkeys(row["text"] for row in pairs))
        cached_frozen, training_rows, cache_source = open_training_cache(
            args.training_bert_cache, training_texts, files["train.jsonl"], model_source, tokenizer, group,
        )
        validation_texts = list(dict.fromkeys(row["text"] for row in queries))
        text_inputs = TextInputs(tokenizer, validation_texts, prefix, window, group.device,
                                 pad_multiple=64, native_truncation=True)
        seen = set(training_texts)
        text_inputs.rows = training_rows + [row for text, row in zip(validation_texts, text_inputs.rows, strict=True)
                                            if text not in seen]
        text_inputs.overlength.update(text for text, row in zip(training_texts, training_rows, strict=True)
                                      if row["truncated"])
    cad = torch.from_numpy(features)
    mean = std = None
    if upstream is not None:
        mean = cad[:train_size].mean(0)
        std = cad[:train_size].std(0).clamp(min=1e-4)
        cad = (cad - mean) / std
    cad = cad.to(group.device)
    group.backend.synchronize(group.device)
    load_seconds = perf_counter() - phase_start
    trainable = sum(value.numel() for value in towers.parameters() if value.requires_grad)
    total = sum(value.numel() for value in towers.parameters())
    if backbone is not None:
        total += sum(value.numel() for value in backbone.parameters())
    config = {
        "source": args.source, "method": args.method, "seed": args.seed, "epochs": 1,
        "levels": levels, "manifest_sha256": files, "official_zip": str(args.official_zip.resolve()),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        "command": sys.argv,
        "model": str(args.qwen_dir if upstream is not None or args.bert_dir is None else args.bert_dir),
        "initialization": model_source, "backbone_precision": args.precision, "attention": args.attention,
        "tuning": "lora" if upstream is not None else "frozen",
        "command_convolution": args.command_convolution if upstream is None else None,
        "head_dtype": "float32", "max_text_tokens": window, "instruction": prefix,
        "tokenizer": {"class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side,
                      "truncation_side": tokenizer.truncation_side},
        "trainable_parameters": trainable, "total_parameters": total,
        "per_device_batch": args.batch_size, "global_pair_batch": args.batch_size * group.world_size,
        "per_device_evaluation_batch": args.eval_batch_size, "world_size": group.world_size,
        "primary_device": str(group.device), "device_model": group.backend.get_device_name(group.device),
        "visible_devices": {name: os.environ[name] for name in ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                            if name in os.environ},
        "temperature": TEMPERATURE, "optimizer": "AdamW", "weight_decay": 0.01, "grad_clip": 1.0,
        "head_lr": 3e-4,
        "objective": ("symmetric diagonal InfoNCE" if upstream is not None
                      else "symmetric negative log posterior mass of directly registered positives"),
        "registration": "exact raw text to UID edges from the frozen pilot training manifest",
        "gradient_multiplier_before_clipping": 1,
        "train_uid_count": train_size, "train_pair_count": len(pairs),
        "validation_query_count": len(queries), "validation_gallery_count": len(gallery),
        "checkpoint_selection": "only end-of-epoch checkpoint in this one-epoch pilot",
        "runtime": runtime_versions(),
    }
    if formal:
        config.update({
            "run_scope": "spectra_update_witness" if witness else "formal_full_train_one_epoch",
            "train_split": "full", "train_manifest": f"{args.source}/train.jsonl",
            "manifest_receipt_sha256": FORMAL_MANIFEST_RECEIPT_SHA,
            "cad_feature_selection": ("cad_cache_input.json"
                                      if args.shared_feature_store or args.prepared_cad else "cad_cache.json"),
            "shared_cad_catalog_includes_unscored_test_candidates": args.shared_feature_store is not None,
            "registration": "exact raw text to UID edges from the full frozen source training manifest",
            "training_text_policy": "original text with native-window truncation recorded per exposure",
            "validation_text_policy": "complete native input including special tokens, with overlength queries failed",
            "pad_to_multiple_of": 64,
            "npu_precision_mode": "must_keep_origin_dtype" if group.device.type == "npu" else None,
            "checkpoint_selection": "sole candidate at end of epoch 1 evaluated on frozen full-text validation R@20",
            "candidate_epochs": [1], **FORMAL_TEST_SCOPE,
        })
    if upstream is not None:
        config.update({
            "upstream": {"commit": SPECTRA_COMMIT, "tree": SPECTRA_TREE, "source_sha256": SPECTRA_SOURCE_SHA256},
            "cad_representation": "upstream 280-dimensional descriptors, training UID mean/sample std",
            "text_pooling": "last non-padding token",
            "lora_rank": 16, "lora_alpha": 32, "backbone_lr": 1e-4,
            "scheduler": "OneCycleLR", "warmup_fraction": 0.1,
            "release_differences": [
                f"Frozen in-domain {'full source' if formal else 'pilot'} manifests, one epoch, seed {args.seed}, recorded device count and batch size",
                "Explicit accelerator rather than CUDA-only launcher",
                "Pretrained model loader uses torch_dtype, local assets, no remote code, and explicit attention",
                "Global average gradient with no additional world-size multiplier",
                "Fixed validation gallery replaces random evaluation pools",
            ],
        })
    else:
        config.update({
            "cad_representation": "canonical real ordered x/y/flag/index rows including both START tokens",
            "cad_tower": {"width": 128, "blocks": 3, "convolution_kernel": 5, "output_dim": OUTPUT_DIM},
            "text_pooling": "masked mean of frozen last hidden states",
            "frozen_text_encoding": "unique exact strings with length-bucketed batches within each split",
            "scheduler": "constant",
        })
    if cache_source is not None:
        config["training_bert_cache"] = cache_source
    if witness:
        config.update(checkpoint_selection="temporary one-update checkpoint for restore verification",
                      candidate_epochs=[], epochs=0, validation_scoring_executed=False)
    if timing_updates is not None:
        config.update(run_scope="spectra_timing_probe", checkpoint_selection="no candidate checkpoint",
                      candidate_epochs=[], epochs=0, validation_scoring_executed=False)
    if group.primary:
        write_json(args.output / "config.json", config)
        write_rows(args.output / "text_inputs.jsonl", text_inputs.rows)
        if not witness and timing_updates is None:
            write_rows(args.output / "prediction_queries.jsonl", queries)
            write_rows(args.output / "prediction_gallery.jsonl", gallery)
    group.barrier()
    if group.world_size > 1:
        torch.manual_seed(args.seed + group.rank)
        group.backend.manual_seed_all(args.seed + group.rank)
    phase_start = perf_counter()
    exposure, training = fit(args, group, towers, backbone, text_inputs, cad[:train_size], pairs, upstream, cached_frozen)
    fit_seconds = perf_counter() - phase_start
    if witness:
        from experiments.src.idea01.formal_spectra_witness import checkpoint_replay

        batch_rows = [pairs[i] for i in training["global_pair_indices"]]
        tokens = text_inputs.batch([row["text"] for row in batch_rows])
        replay = checkpoint_replay(args, towers, upstream, tokenizer, mean, std, config, tokens,
                                   cad[[row["cad_row"] for row in batch_rows]])
        write_json(args.output / "spectra_update_witness.json", {
            "status": "passed", "run_scope": "spectra_update_witness", "source": args.source,
            "method": args.method, "seed": args.seed, "training": training, "restore": replay,
            "full_train_pairs": len(pairs), "normalization_train_uids": train_size,
            "model_revision": model_source["revision"], "descriptor_store_sha256": FORMAL_SPECTRA_STORE_SHA,
            "manifest_sha256": files, "training_input_shape": list(tokens["input_ids"].shape),
            "primary_device": str(group.device), "device_model": config["device_model"],
            "primary_peak_memory_bytes": int(group.backend.max_memory_allocated(group.device)),
            "validation_scoring_executed": False, **FORMAL_TEST_SCOPE,
        })
        return
    if timing_updates is not None:
        write_json(args.output / "spectra_timing_probe.json", {
            "status": "ok", "run_scope": "spectra_timing_probe", "source": args.source,
            "training": training, "full_train_pairs": len(pairs),
            "primary_peak_memory_bytes": int(group.backend.max_memory_allocated(group.device)),
            "validation_scoring_executed": False, **FORMAL_TEST_SCOPE,
        })
        return
    peak = int(group.backend.max_memory_allocated(group.device))
    peak_by_rank = [peak]
    if group.world_size > 1:
        peak_value = torch.tensor([peak], dtype=torch.int64, device=group.device)
        peak_values = [torch.empty_like(peak_value) for _ in range(group.world_size)]
        dist.all_gather(peak_values, peak_value)
        peak_by_rank = [int(item.item()) for item in peak_values]
    if group.primary:
        write_rows(args.output / "training_exposure.jsonl", exposure_rows(train, levels, exposure))
        write_json(args.output / "training_summary.json", training)
        phase_start = perf_counter()
        if upstream is not None:
            upstream["runtime"].save_checkpoint(args.output, towers.cad, towers.text, tokenizer, mean, std, config)
        else:
            path = args.output / "retrieval.pt.part"
            torch.save({
                "config": config,
                "cad_tower": {key: value.detach().cpu() for key, value in towers.cad.state_dict().items()},
                "text_head": {key: value.detach().cpu() for key, value in towers.text.state_dict().items()},
            }, path)
            path.replace(args.output / "retrieval.pt")
            tokenizer.save_pretrained(args.output / "tokenizer")
        checkpoint_seconds = perf_counter() - phase_start
        phase_start = perf_counter()
        scores, valid_q, valid_c = score(args, group, towers, backbone, text_inputs, cad[train_size:], queries, val_covered)
        np.save(args.output / "prediction_scores.npy", scores, allow_pickle=False)
        ranking_rows, metrics = evaluate_rankings(scores, queries, gallery)
        write_rows(args.output / "ranks.jsonl", ranking_rows)
        write_json(args.output / "metrics.json", metrics)
        write_rows(args.output / "prediction_failures.jsonl", (
            [{"axis": "query", "row": index, **queries[index], **(
                {"reason": "overlength_full_text" if queries[index]["text"] in text_inputs.overlength
                 else "nonfinite_query_embedding"} if args.require_full_validation_text else {}
            )} for index in np.flatnonzero(~valid_q).tolist()]
            + [{"axis": "gallery", "row": index, "uid": gallery[index]["uid"],
                "cad_status": coverage[train_size + index]["status"]} for index in np.flatnonzero(~valid_c).tolist()]
        ))
        evaluation_seconds = perf_counter() - phase_start
        validation_overlength = sum(row["text"] in text_inputs.overlength for row in queries)
        if formal:
            write_json(args.output / "selection.json", {
                "run_scope": "formal_full_train_one_epoch", "source": args.source, "method": args.method,
                "seed": args.seed, "candidate_epochs": [1], "selected_epoch": 1,
                "selected_checkpoint": "retrieval.pt",
                "checkpoint_sha256": file_sha256(args.output / "retrieval.pt"),
                **({"adapter_files_sha256": {
                    path.relative_to(args.output).as_posix(): file_sha256(path)
                    for path in sorted((args.output / "adapter").iterdir()) if path.is_file()
                }} if upstream is not None else {}),
                "validation_metrics": "metrics.json", "rank_rows": "ranks.jsonl",
                "validation_R@20_equal_level_macro": sum(metrics[level]["R@20"] for level in levels) / len(levels),
                "validation_overlength_failures": validation_overlength, **FORMAL_TEST_SCOPE,
            })
        arts = sorted(path for path in args.output.rglob("*") if path.is_file())
        input_by_text = dict(zip(all_texts, text_inputs.rows, strict=True))
        result = {
            "status": "completed", "source": args.source, "method": args.method,
            "training": training, "training_cad_coverage": {"valid": train_ok, "total": train_size},
            "train_truncated_pairs": sum(input_by_text[row["text"]]["truncated"] for row in pairs),
            "validation_truncated_queries": 0 if args.require_full_validation_text else validation_overlength,
            "validation_gallery_cad_status": dict(Counter(row["status"] for row in coverage[train_size:])),
            "validation_queries_valid": int(valid_q.sum()), "validation_queries_total": len(queries),
            "validation_gallery_valid": int(valid_c.sum()), "validation_gallery_total": len(gallery),
            "validation_finite_scores": int(np.isfinite(scores).sum()),
            "runtime": {
                "bundle_wall_seconds": perf_counter() - started,
                "model_load_tokenization_and_transfer_seconds": load_seconds,
                "fit_including_frozen_text_encoding_seconds": fit_seconds,
                "checkpoint_seconds": checkpoint_seconds, "validation_seconds": evaluation_seconds,
                "optimizer_loop_accelerator_hours": training["optimizer_loop_seconds"] * group.world_size / 3600,
                "primary_peak_memory_bytes": int(group.backend.max_memory_allocated(group.device)),
                "training_peak_memory_bytes_by_rank": peak_by_rank,
            },
            "files": {path.relative_to(args.output).as_posix(): file_sha256(path) for path in arts},
        }
        if args.require_full_validation_text:
            result["validation_overlength_failures"] = validation_overlength
        if formal:
            result.update({
                "run_scope": "formal_full_train_one_epoch", "seed": args.seed, "selected_epoch": 1,
                "checkpoint_selection_receipt": "selection.json", **FORMAL_TEST_SCOPE,
            })
            result["runtime"]["fit_accelerator_hours"] = fit_seconds * group.world_size / 3600
        write_json(args.output / "run.json", result)
        print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)
    group.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--source", choices=tuple(LEVELS), required=True)
    parser.add_argument("--method", choices=("dual-tower", "spectrabrep"), required=True)
    parser.add_argument("--official-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bert-dir", type=Path)
    parser.add_argument("--training-bert-cache", type=Path, help="Merged read-only Q05 cache for full T training")
    parser.add_argument("--qwen-dir", type=Path)
    parser.add_argument("--upstream", type=Path, help="Clean SpectraBRep release tree for the pinned commit")
    parser.add_argument("--device", default="npu", help="Training accelerator, for example npu:0 or cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--command-convolution", choices=("native", "linear"), default="native",
                        help="FP32-equivalent convolution implementation for the command tower")
    parser.add_argument("--batch-size", type=int, help="Per-device pairs per update, 32 for dual-tower or 16 for SpectraBRep")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepared-cad", type=Path)
    parser.add_argument("--shared-feature-store", type=Path, help="Certified DeepCAD feature store for formal train/val")
    parser.add_argument("--prepare-cad-only", action="store_true")
    parser.add_argument("--train-split", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--spectra-update-witness", action="store_true",
                        help="Run the exact first formal SpectraBRep update and replay a temporary checkpoint")
    parser.add_argument("--spectra-timing-updates", type=int,
                        help="Time a strict prefix of actual formal SpectraBRep updates")
    parser.add_argument("--require-full-validation-text", action="store_true",
                        help="Keep overlength queries as failed rows without encoding them")
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=SEED)
    parser.add_argument("--expected-train-uids", type=int,
                        help="Defaults to 4096 for pilot or the certified source count for full")
    parser.add_argument("--expected-targets", type=int, default=512)
    parser.add_argument("--expected-gallery", type=int, default=2048)
    args = parser.parse_args()
    formal = args.train_split == "full"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.spectra_update_witness and (not formal or args.method != "spectrabrep" or args.prepare_cad_only):
        parser.error("--spectra-update-witness requires formal SpectraBRep training")
    if args.spectra_timing_updates is not None and (
        not 2 <= args.spectra_timing_updates <= 32 or not formal or args.source != "t2cad-v1.0"
        or args.method != "spectrabrep" or args.prepare_cad_only or args.spectra_update_witness
    ):
        parser.error("--spectra-timing-updates requires a 2–32 update full-source T SpectraBRep timing prefix")
    if formal:
        from experiments.src.idea01.in_domain_pilot import COHORTS

        expected_train = COHORTS[args.source]["train"][0]
        if args.precision != "fp32" or not args.require_full_validation_text:
            parser.error("Full training requires FP32 and --require-full-validation-text")
        if args.method == "dual-tower" and not args.prepare_cad_only and args.bert_dir is None:
            parser.error("Full command training requires the certified BERT backbone")
        if args.method == "spectrabrep" and (
            args.shared_feature_store is None or args.attention != "eager"
            or args.device.partition(":")[0] != "npu"
            or not valid_formal_spectra_layout(args.batch_size, world_size, args.spectra_timing_updates)
            or args.bert_dir is not None or args.training_bert_cache is not None
        ):
            parser.error("Full SpectraBRep uses single-NPU B16, eager attention and --shared-feature-store. "
                         "A T timing prefix may use four NPUs at four pairs each.")
        if args.expected_train_uids not in (None, expected_train):
            parser.error("Full training uses the certified source UID count")
        args.expected_train_uids = expected_train
    elif args.expected_train_uids is None:
        args.expected_train_uids = 4096
    if args.shared_feature_store is not None and (not formal or args.prepared_cad is not None):
        parser.error("--shared-feature-store is for formal full training and replaces --prepared-cad")
    if args.batch_size is None:
        args.batch_size = 16 if args.method == "spectrabrep" else 32
    if args.training_bert_cache is not None and (
        not formal or args.source != "t2cad-v1.0" or args.prepare_cad_only or args.attention != "eager"
        or args.eval_batch_size != 32 or args.batch_size * int(os.environ.get("WORLD_SIZE", "1")) != 32
    ):
        parser.error("--training-bert-cache requires formal T training, eager attention, and B32 train and evaluation")
    if min(args.batch_size, args.eval_batch_size, args.workers,
           args.expected_train_uids, args.expected_targets, args.expected_gallery) < 1:
        parser.error("Batch sizes, workers, and manifest sizes must be positive")
    if args.device.partition(":")[0] not in {"npu", "cuda"}:
        parser.error("Choose npu[:INDEX] or cuda[:INDEX]")
    if args.method == "spectrabrep" and args.upstream is None:
        parser.error("SpectraBRep needs --upstream")
    if args.method != "dual-tower" and args.command_convolution != "native":
        parser.error("--command-convolution only applies to the command dual tower")
    if not args.prepare_cad_only and (args.qwen_dir is None and (args.method == "spectrabrep" or args.bert_dir is None)):
        parser.error("Specify --bert-dir for the dual tower or --qwen-dir for the pinned Qwen base")
    if args.prepare_cad_only and (args.prepared_cad is not None or int(os.environ.get("WORLD_SIZE", "1")) != 1):
        parser.error("Prepare CAD features once without torchrun or an existing prepared cache")

    started = perf_counter()
    axes = manifest_axes(args)
    train, _, gallery, files = axes
    spectra = verify_spectra_source(args.upstream) if args.method == "spectrabrep" else None
    if formal:
        planned_batches = training_batches(args.batch_size, world_size, len(train) * len(LEVELS[args.source]),
                                           spectra is not None)
    group = None
    if world_size > 1:
        group = initialize_devices(args.device, keep_origin_dtype=formal)
    primary = group is None or group.primary
    if primary:
        args.output.mkdir(parents=True, exist_ok=False)
    if group is not None:
        group.barrier()
    try:
        if primary:
            if args.shared_feature_store is not None:
                features, coverage = shared_cad_cache(args, train, gallery)
            else:
                if formal:
                    from experiments.src.idea01.in_domain_pilot import INPUT_HASHES

                    if file_sha256(args.official_zip) != INPUT_HASHES["official_zip"]:
                        raise ValueError("The formal CAD archive differs from the registered release")
                features, coverage = cad_cache(args, train + gallery, files, spectra)
        else:
            group.barrier()
            if args.shared_feature_store is not None:
                selected = np.load(args.output / "cad_feature_indices.npy", allow_pickle=False)
                store = np.load(args.shared_feature_store / "cad_features.npy", mmap_mode="r", allow_pickle=False)
                features = store[selected]
                coverage = read_rows(args.output / "cad_coverage.jsonl")
            else:
                cache = args.prepared_cad or args.output
                features = np.load(cache / "cad_features.npy", allow_pickle=False)
                coverage = read_rows(cache / "cad_coverage.jsonl")
        if args.prepare_cad_only:
            report = {
                "seconds": perf_counter() - started, "runtime": runtime_versions(), "workers": args.workers,
                "training_status": dict(Counter(row["status"] for row in coverage[:len(train)])),
                "validation_status": dict(Counter(row["status"] for row in coverage[len(train):])),
            }
            if formal:
                report.update(run_scope="formal_full_train_one_epoch", seed=args.seed, train_pair_count=(
                    len(train) * len(LEVELS[args.source])), planned_updates=len(planned_batches),
                    final_global_batch=planned_batches[-1][1] - planned_batches[-1][0],
                    shared_cad_catalog_includes_unscored_test_candidates=args.shared_feature_store is not None,
                    manifest_sha256=files, **FORMAL_TEST_SCOPE)
            write_json(args.output / "prepare_runtime.json", report)
            if formal and report["training_status"] != {"ok": len(train)}:
                raise ValueError("At least one registered training CAD is invalid. See prepare_runtime.json.")
            return
        if group is None:
            group = initialize_devices(args.device, keep_origin_dtype=formal)
        if primary:
            group.barrier()
        run(args, group, axes, features, coverage, spectra, started)
    except Exception as exc:
        if primary:
            write_json(args.output / "failure.json", {"type": type(exc).__name__, "detail": str(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
