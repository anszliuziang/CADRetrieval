#!/usr/bin/env python3
"""Train and evaluate one neural retriever on a frozen in-domain pilot manifest."""

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
    OUTPUT_DIM,
    SPECTRA_COMMIT,
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
    paths = [args.manifest_root / args.source / filename for filename in MANIFESTS]
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
    files = {path.name: file_sha256(path) for path in paths}
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


def initialize_devices(label: str) -> DeviceGroup:
    kind = label.partition(":")[0]
    if kind == "npu":
        import torch_npu  # noqa: F401  Registers the Ascend device and HCCL backend.

        backend = torch.npu
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
        cad = CommandTower()
        model_source["actual_backbone_dtypes"] = sorted({str(value.dtype) for value in backbone.parameters()})
    if model_source["actual_backbone_dtypes"] != [str(dtype)]:
        raise RuntimeError(f"Loaded backbone dtype differs from the requested {dtype}: {model_source}")
    return RetrievalTowers(text, cad).to(group.device), backbone, tokenizer, upstream, model_source


class TextInputs:
    def __init__(self, tokenizer, texts, prefix, length, device):
        self.tokenizer = tokenizer
        self.device = device
        self.length = length
        ids = tokenizer(
            [prefix + text for text in texts], padding=False, truncation=False,
            return_attention_mask=False, verbose=False,
        )["input_ids"]
        self.rows = []
        self.ids = {}
        for text, row in zip(texts, ids, strict=True):
            used = row[-length:] if tokenizer.truncation_side == "left" else row[:length]
            self.ids[text] = used
            self.rows.append({
                "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "input_token_count": len(row),
                "effective_token_count": len(used),
                "truncated": len(used) < len(row),
            })

    def batch(self, texts):
        return self.tokenizer.pad(
            [{"input_ids": self.ids[text]} for text in texts], return_tensors="pt", padding=True,
        ).to(self.device)

    def frozen(self, backbone, texts, batch_size):
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
        return torch.cat(values)[[length_index[text] for text in texts]], indexes


def exposure_rows(train, levels, counters):
    for row in train:
        counts = {level: counters[row["uid"], level] for level in levels}
        yield {"uid": row["uid"], "pair_exposures": sum(counts.values()), "level_exposures": counts}


def fit(args, group, towers, backbone, text_inputs, cad, pairs, upstream):
    all_texts = list(dict.fromkeys(row["text"] for row in pairs))
    registration = direct_registration((row["text"], row["uid"]) for row in pairs)
    frozen = None
    if backbone is not None:
        frozen, frozen_index = text_inputs.frozen(backbone, all_texts, args.eval_batch_size)
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
    batch_size = args.batch_size * group.world_size
    batches = [(start, min(start + batch_size, len(pairs))) for start in range(0, len(pairs), batch_size)]
    if (len(batches) < 2 or any(end - start < 2 or (end - start) % group.world_size for start, end in batches)
            or (upstream is not None and len(batches) == 10)):
        raise ValueError("Choose a batch size giving at least two updates, equal nonempty rank slices, "
                         "at least two pairs per global batch, and a nondegenerate SpectraBRep OneCycle schedule")
    scheduler = None
    if upstream is not None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, [item["lr"] for item in groups], total_steps=len(batches), pct_start=0.1,
        )
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(SEED)).tolist()
    runner.train()
    stats = Counter()
    exposure = Counter()
    sum_loss = sum_pair_loss = 0.0
    stream = (args.output / "training_steps.jsonl").open("w", encoding="utf-8", newline="\n") if group.primary else None
    try:
        train_start = perf_counter()
        for step, (start, stop) in enumerate(batches, 1):
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
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            exposure.update((row["uid"], row["level"]) for row in global_rows)
            stats.update(counts)
            sum_loss += loss
            sum_pair_loss += loss * len(global_rows)
            if stream is not None:
                stream.write(json.dumps({
                    "update": step, "global_pairs": len(global_rows), "pairs_per_rank": local_size,
                    "train_pair_indices": order[start:stop], "loss": loss, "learning_rates": lr,
                    "preclip_grad_norm": float(grad_norm), **counts,
                }, allow_nan=False) + "\n")
                stream.flush()
                if step == 1 or step % 20 == 0 or step == len(batches):
                    print(f"{args.source}/{args.method} update {step}/{len(batches)} loss={loss:.5f}", flush=True)
        group.backend.synchronize(group.device)
        train_seconds = perf_counter() - train_start
    finally:
        if stream is not None:
            stream.close()
    if any(value != 1 for value in exposure.values()) or len(exposure) != len(pairs):
        raise RuntimeError("The pilot must expose each registered UID and level exactly once")
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
    unique = list(dict.fromkeys(row["text"] for row in queries))
    text_locations = {text: index for index, text in enumerate(unique)}
    with torch.no_grad():
        frozen = backbone is not None
        if frozen:
            text_values, _ = text_inputs.frozen(backbone, unique, args.eval_batch_size)
        text_embeddings = []
        for start in range(0, len(unique), args.eval_batch_size):
            stop = start + args.eval_batch_size
            value = text_values[start:stop] if frozen else text_inputs.batch(unique[start:stop])
            text_embeddings.append(towers.encode_text(value))
        query_values = torch.cat(text_embeddings)[[text_locations[row["text"]] for row in queries]]
        cad_values = torch.full((len(covered), OUTPUT_DIM), torch.nan, device=group.device)
        valid = [index for index, value in enumerate(covered) if value]
        for start in range(0, len(valid), args.eval_batch_size):
            selected = valid[start:start + args.eval_batch_size]
            cad_values[selected] = towers.encode_cad(cad[selected])
        valid_queries = torch.isfinite(query_values).all(dim=1).cpu().numpy()
        valid_candidates = torch.isfinite(cad_values).all(dim=1).cpu().numpy()
        scores = (query_values @ cad_values.T).cpu().numpy()
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    group.backend.manual_seed_all(SEED)
    group.backend.reset_peak_memory_stats(group.device)
    phase_start = perf_counter()
    towers, backbone, tokenizer, upstream, model_source = load_towers(args, group, spectra)
    all_texts = list(dict.fromkeys(row["text"] for row in pairs + queries))
    prefix = upstream["config"].INSTRUCTION if upstream is not None else ""
    window = 384 if upstream is not None else 512
    text_inputs = TextInputs(tokenizer, all_texts, prefix, window, group.device)
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
        "source": args.source, "method": args.method, "seed": SEED, "epochs": 1,
        "levels": levels, "manifest_sha256": files, "official_zip": str(args.official_zip.resolve()),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        "command": sys.argv,
        "model": str(args.qwen_dir if upstream is not None or args.bert_dir is None else args.bert_dir),
        "initialization": model_source, "backbone_precision": args.precision, "attention": args.attention,
        "tuning": "lora" if upstream is not None else "frozen",
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
    if upstream is not None:
        config.update({
            "upstream": {"commit": SPECTRA_COMMIT, "tree": SPECTRA_TREE, "source_sha256": SPECTRA_SOURCE_SHA256},
            "cad_representation": "upstream 280-dimensional descriptors, training UID mean/sample std",
            "text_pooling": "last non-padding token",
            "lora_rank": 16, "lora_alpha": 32, "backbone_lr": 1e-4,
            "scheduler": "OneCycleLR", "warmup_fraction": 0.1,
            "release_differences": [
                "Frozen in-domain pilot manifests, one epoch, seed 17, recorded device count and batch size",
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
    if group.primary:
        write_json(args.output / "config.json", config)
        write_rows(args.output / "text_inputs.jsonl", text_inputs.rows)
        write_rows(args.output / "prediction_queries.jsonl", queries)
        write_rows(args.output / "prediction_gallery.jsonl", gallery)
    group.barrier()
    if group.world_size > 1:
        torch.manual_seed(SEED + group.rank)
        group.backend.manual_seed_all(SEED + group.rank)
    phase_start = perf_counter()
    exposure, training = fit(args, group, towers, backbone, text_inputs, cad[:train_size], pairs, upstream)
    fit_seconds = perf_counter() - phase_start
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
            [{"axis": "query", "row": index, **queries[index]} for index in np.flatnonzero(~valid_q).tolist()]
            + [{"axis": "gallery", "row": index, "uid": gallery[index]["uid"],
                "cad_status": coverage[train_size + index]["status"]} for index in np.flatnonzero(~valid_c).tolist()]
        ))
        evaluation_seconds = perf_counter() - phase_start
        arts = sorted(path for path in args.output.rglob("*") if path.is_file())
        input_by_text = dict(zip(all_texts, text_inputs.rows, strict=True))
        result = {
            "status": "completed", "source": args.source, "method": args.method,
            "training": training, "training_cad_coverage": {"valid": train_ok, "total": train_size},
            "train_truncated_pairs": sum(input_by_text[row["text"]]["truncated"] for row in pairs),
            "validation_truncated_queries": sum(input_by_text[row["text"]]["truncated"] for row in queries),
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
    parser.add_argument("--qwen-dir", type=Path)
    parser.add_argument("--upstream", type=Path, help="Clean SpectraBRep release tree for the pinned commit")
    parser.add_argument("--device", default="npu", help="Training accelerator, for example npu:0 or cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--batch-size", type=int, help="Per-device pairs per update, 32 for dual-tower or 16 for SpectraBRep")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepared-cad", type=Path)
    parser.add_argument("--prepare-cad-only", action="store_true")
    parser.add_argument("--expected-train-uids", type=int, default=4096)
    parser.add_argument("--expected-targets", type=int, default=512)
    parser.add_argument("--expected-gallery", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.method == "spectrabrep" else 32
    if min(args.batch_size, args.eval_batch_size, args.workers,
           args.expected_train_uids, args.expected_targets, args.expected_gallery) < 1:
        parser.error("Batch sizes, workers, and manifest sizes must be positive")
    if args.device.partition(":")[0] not in {"npu", "cuda"}:
        parser.error("Choose npu[:INDEX] or cuda[:INDEX]")
    if args.method == "spectrabrep" and args.upstream is None:
        parser.error("SpectraBRep needs --upstream")
    if not args.prepare_cad_only and (args.qwen_dir is None and (args.method == "spectrabrep" or args.bert_dir is None)):
        parser.error("Specify --bert-dir for the dual tower or --qwen-dir for the pinned Qwen base")
    if args.prepare_cad_only and (args.prepared_cad is not None or int(os.environ.get("WORLD_SIZE", "1")) != 1):
        parser.error("Prepare CAD features once without torchrun or an existing prepared cache")

    started = perf_counter()
    axes = manifest_axes(args)
    train, _, gallery, files = axes
    spectra = verify_spectra_source(args.upstream) if args.method == "spectrabrep" else None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    group = None
    if world_size > 1:
        group = initialize_devices(args.device)
    primary = group is None or group.primary
    if primary:
        args.output.mkdir(parents=True, exist_ok=False)
    if group is not None:
        group.barrier()
    try:
        if primary:
            features, coverage = cad_cache(args, train + gallery, files, spectra)
        else:
            group.barrier()
            cache = args.prepared_cad or args.output
            features = np.load(cache / "cad_features.npy", allow_pickle=False)
            coverage = read_rows(cache / "cad_coverage.jsonl")
        if args.prepare_cad_only:
            write_json(args.output / "prepare_runtime.json", {
                "seconds": perf_counter() - started, "runtime": runtime_versions(), "workers": args.workers,
                "training_status": dict(Counter(row["status"] for row in coverage[:len(train)])),
                "validation_status": dict(Counter(row["status"] for row in coverage[len(train):])),
            })
            return
        if group is None:
            group = initialize_devices(args.device)
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
