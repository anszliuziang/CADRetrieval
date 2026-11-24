"""Train fresh Text2CAD weights on the frozen T, CT or Fusion training axis.

Run as python -m experiments.scripts.run_formal_text2cad_training, or use
torchrun --standalone --nproc-per-node 2 -m with the same module and arguments.
--train-only --stop-after-updates N opens only training inputs and writes a
resumable smoke checkpoint. N is the absolute update number after resuming.
--audit-batch first|tail makes a separate dropout-free one-update witness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import time
from zipfile import ZipFile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from experiments.scripts.prepare_formal_fusion360 import SOURCE as FUSION
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore, assemble_batch
from experiments.src.idea01.text2cad_port.loader import (
    PINNED_BERT_FILE_SHA256, build_text2cad, sha256_file, verify_bert_asset_shas, verify_bert_model_load,
)
from experiments.src.idea01.text2cad_port.model import teacher_forced_logits
from experiments.src.idea01.text2cad_port.text_embed import TextEmbedder
from experiments.src.idea01.text2cad_vector_scorer import two_head_factorized_score

LEVELS = {**pilot.SOURCE_LEVELS, FUSION: ("annotation",)}
TRAIN_UIDS = {"t2cad-v1.0": 151454, "ct-deepcad": 148046, FUSION: 2467}
DEEPCAD_RECEIPT = "3c297920d0905b3dd5f14aa1b5ac163245a17b958cc3c2e1169ee65d5a8d055d"
FUSION_RECEIPT = "0ad0092cb9d9710fa11af260bdf3dd6111015a76d37d465c9a280b9a50557127"
FUSION_CONVERSION = "e41f9478f05bbe5c8aa824e0b757baa6ce427d18c0c31602341e5bc76c01230c"
BATCH = 16
SEED = 17
CONTROL_GROUP = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verified(path, sha):
    require(sha256_file(path) == sha, f"Frozen input differs: {path}")
    return path


@dataclass(frozen=True, slots=True)
class Row:
    uid: str
    member: str | None
    texts: tuple
    geometry_verified: bool | None = None


def axes(args):
    root = args.manifests
    levels = LEVELS[args.source]
    pins = {}

    def rows(name):
        path = verified(root / name, pins[name])
        with path.open(encoding="utf-8") as stream:
            yield from map(json.loads, stream)

    if args.source == FUSION:
        manifest = json.loads(verified(root / "eligibility_manifest.json", FUSION_RECEIPT).read_bytes())
        conversion = json.loads(verified(args.cad_zip.parent / "conversion_manifest.json",
                                         FUSION_CONVERSION).read_bytes())
        require(manifest["conversion_manifest_sha256"] == FUSION_CONVERSION, "Fusion receipt mismatch")
        archive_pin = conversion["file_sha256"]["text2cad_candidates.zip"]
        pins = manifest["file_sha256"]
        train = []
        for pair, cad in zip(rows("train_pairs.jsonl"), rows("shared_training_candidate.jsonl"), strict=True):
            require(pair["uid"] == cad["uid"] and pair["status"] == "ok", "Fusion training axes differ")
            train.append(Row(pair["uid"], cad["cad_member"], (pair["text"],), True))
        targets = []
        if not args.train_only:
            gallery = {r["uid"]: r for r in rows("validation_gallery.jsonl")}
            path = verified(args.fusion_validation_qrels, manifest["validation_qrels_sha256"])
            for q, rel in zip(rows("validation_queries.jsonl"), pilot.read_jsonl(path), strict=True):
                require(q["query_id"] == rel["query_id"], "Fusion validation axis mismatch")
                cad = gallery[rel["uid"]]
                targets.append(Row(cad["uid"], cad["cad_member"], (q["text"],), cad["geometry_verified"]))
    else:
        manifest = json.loads(verified(root / "manifest_receipt.json", DEEPCAD_RECEIPT).read_bytes())
        pins = manifest["file_sha256"]
        archive_pin = pilot.INPUT_HASHES["official_zip"]

        def load(split):
            return [Row(r["uid"], r["cad_member"], tuple(r["texts"][level] for level in levels))
                    for r in rows(f"{args.source}/{split}.jsonl")]

        train = load("train")
        targets = [] if args.train_only else load("validation_targets")

    train_uids = {r.uid for r in train}
    require(len(train) == len(train_uids) == TRAIN_UIDS[args.source], "Incomplete training UID axis")
    require(all(r.member and all(isinstance(t, str) and t.strip() for t in r.texts) for r in train),
            "Incomplete training pair")
    if not args.train_only:
        require(len(targets) == len({r.uid for r in targets}) == 512 and
                train_uids.isdisjoint(r.uid for r in targets), "Validation panel cardinality or leakage")
    return train, targets, archive_pin


class LazyCandidates(ZipCandidateStore):
    """Use the pinned archive reader on demand without loading 151k candidates per rank."""

    def __init__(self, path, members, sha):
        super().__init__(path, members, expected_sha256=sha)
        self.archive = ZipFile(path)

    @lru_cache(maxsize=1024)
    def get(self, index):
        return self._load_one(self.members[index], self.archive)


def encode_training_texts(tokenizer, texts, pad_multiple):
    options = dict(return_tensors="pt", padding=True, truncation=True, max_length=512)
    if pad_multiple is not None:
        options["pad_to_multiple_of"] = pad_multiple
    return tokenizer(texts, **options)


class TrainingEmbedder(TextEmbedder):
    truncate = False
    profile = False
    pad_multiple = None

    @lru_cache(maxsize=65536)
    def length(self, text):
        return len(self.tokenizer(text, truncation=False, verbose=False)["input_ids"])

    def get_embedding(self, texts):
        if not self.truncate:
            return super().get_embedding(texts)
        if self.profile:
            synchronize(self.device)
            started = time.monotonic()
        packed = encode_training_texts(self.tokenizer, texts, self.pad_multiple)
        inputs = packed.to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)[0], inputs["attention_mask"] == 0
        if self.profile:
            synchronize(self.device)
            self.bert_seconds = time.monotonic() - started
            self.text_width = int(inputs["input_ids"].shape[1])
        return result


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type != "cpu":
        getattr(torch, device.type).manual_seed(seed)


def rng(device):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
    if device.type != "cpu":
        state["device"] = getattr(torch, device.type).get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(state["device"], device)


def gather(value, world):
    if world == 1:
        return [value]
    items = [None] * world
    dist.all_gather_object(items, value, group=CONTROL_GROUP)
    return items


def cpu_weights(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.get_trainable_state_dict().items()}


def weight_sha(weights):
    h = hashlib.sha256()
    for k, v in sorted(weights.items()):
        h.update(json.dumps([k, list(v.shape), str(v.dtype)]).encode() + b"\n")
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def save(args, recipe, model, optimizer, state, device, rank, world, epoch=None):
    states = gather(rng(device), world)
    if rank == 0:
        start = time.monotonic()
        weights = cpu_weights(model)
        sha = weight_sha(weights)
        pointer = args.out / "checkpoint.json"
        previous = json.loads(pointer.read_bytes())["file"] if pointer.exists() else None
        progress = "progress-b.pt" if previous == "progress-a.pt" else "progress-a.pt"
        path = args.out / (progress if epoch is None else f"epoch-{epoch:02}.pt")
        atomic_save(path, dict(recipe=recipe, weights=weights, weight_sha256=sha, optimizer=optimizer.state_dict(),
                               rng_by_rank=states, state=state))
        write_json(pointer, dict(file=path.name, weight_sha256=sha, **state, save_seconds=time.monotonic()-start))
    if world > 1:
        dist.barrier()


def restore(args, recipe, model, optimizer, device, rank):
    pointer = json.loads((args.out / "checkpoint.json").read_bytes())
    checkpoint = torch.load(args.out / pointer["file"], map_location="cpu", weights_only=False)
    require(recipe == checkpoint["recipe"], "Resume recipe, inputs, initial state, code or rank layout changed")
    weights = checkpoint["weights"]
    require(weight_sha(weights) == checkpoint["weight_sha256"] == pointer["weight_sha256"] and
            checkpoint["state"]["updates"] == pointer["updates"], "Committed checkpoint mismatch")
    require(set(weights) == set(model.get_trainable_state_dict()), "Resume trainable parameter names changed")
    for name in ("adaptive_layer", "cad_decoder"):
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng_by_rank"][rank], device)
    return checkpoint["state"]


def repair_log(out, committed_updates):
    path = out / "steps.jsonl"
    temp = out / "steps.recover.tmp"
    with path.open("rb") as src, temp.open("wb") as dest:
        for expected in range(1, committed_updates + 1):
            line = src.readline()
            require(line.endswith(b"\n") and json.loads(line)["update"] == expected, "Committed step log is incomplete")
            dest.write(line)
    os.replace(temp, path)


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def epoch_order(size, epoch):
    generator = torch.Generator(device="cpu").manual_seed(SEED + epoch - 1)
    return torch.randperm(size, generator=generator, device="cpu")


def stats_for(levels):
    return dict(exposure=0, by_level={k: 0 for k in levels}, truncated={k: 0 for k in levels},
                fields=[0, 0], smoothed_sum=[0., 0.], nll_sum=[0., 0.])


def head_losses(logits, targets, mask):
    require(logits.dtype == torch.float32, "FP32 logits required")
    counts, smoothed, nll = [], [], []
    for h in range(2):
        pred = logits[:, :, h].transpose(1, 2)
        labels = targets[:, :, h].long()
        valid = ~mask[:, :, h]
        counts.append(valid.sum())
        ce = F.cross_entropy(pred, labels, reduction="none", label_smoothing=0.1)
        smoothed.append((ce * valid).sum())
        with torch.no_grad():
            nll.append((F.cross_entropy(pred, labels, reduction="none") * valid).sum())
    return torch.stack(counts), torch.stack(smoothed), torch.stack(nll)


def update(model, wrapped, optimizer, store, indices, train, levels, ids, rank, world, device, audit, profile):
    begin = time.monotonic()
    model.base_text_embedder.profile = profile
    local_ids = ids[rank::world]
    require(bool(local_ids), "Empty rank in global batch")
    rows, level_ids = zip(*(divmod(i, len(levels)) for i in local_ids))
    texts = [train[r].texts[l] for r, l in zip(rows, level_ids)]
    widths = [model.base_text_embedder.length(t) for t in texts]
    vec, masks, seq = assemble_batch(store, [indices[train[r].member] for r in rows], device)
    if profile:
        synchronize(device)
        forward = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    logits, targets, mask = teacher_forced_logits(wrapped, texts, vec, masks, cad_seq_len=seq)
    counts, smooth, nll = head_losses(logits, targets, mask)
    truncated = [sum(width > 512 and l == j for l, width in zip(level_ids, widths)) for j in range(len(levels))]
    totals = torch.cat((counts.float(), smooth.detach(), nll, smooth.new_tensor(truncated)))
    if profile:
        synchronize(device)
        reduce = time.monotonic()
    if world > 1:
        dist.all_reduce(totals)
    values = totals.cpu().tolist()
    require(min(values[:2]) > 0 and all(math.isfinite(v) for v in values), "Invalid global head totals")
    # DDP averages the per-rank gradients; each true field contributes 1/(2 * global_head_count).
    objective = (smooth / totals[:2]).sum() * (world / 2)
    if profile:
        synchronize(device)
        backward = time.monotonic()
    objective.backward()
    if profile:
        synchronize(device)
        clip = time.monotonic()
    params = [p for p in model.parameters() if p.requires_grad]
    probes = {}
    if audit:
        for name in ("adaptive_layer", "cad_decoder.seq_output_x", "cad_decoder.seq_output_y"):
            key, param = next((k, p) for k, p in model.named_parameters() if k.startswith(name) and
                              p.grad is not None and p.ndim == 2)
            probes[key] = dict(gradient=param.grad.detach().cpu().clone(), before=param.detach().cpu().clone())
    norm = float(torch.nn.utils.clip_grad_norm_(params, 0.9).detach().cpu())
    require(math.isfinite(norm) and norm > 0, "Non-finite or zero global gradient norm")
    optimizer.step()
    synchronize(device)
    finished = time.monotonic()
    if audit:
        named = dict(model.named_parameters())
        for k, item in probes.items():
            item["after"] = named[k].detach().cpu().clone()
    result = dict(pairs=len(ids), fields=list(map(int, values[:2])), smoothed_sum=values[2:4], nll_sum=values[4:6],
                  truncated=dict(zip(levels, map(int, values[6:]))),
                  by_level=dict(Counter(levels[i % len(levels)] for i in ids)),
                  loss=sum(values[2+h] / values[h] for h in range(2)) / 2,
                  grad_norm=norm, seconds=finished - begin)
    if profile:
        timing = dict(rank=rank, pairs=len(local_ids), prepare_seconds=forward-begin,
                      bert_seconds=model.base_text_embedder.bert_seconds,
                      text_width=model.base_text_embedder.text_width,
                      forward_and_loss_seconds=reduce-forward, head_count_reduce_seconds=backward-reduce,
                      backward_including_ddp_seconds=clip-backward, clip_update_seconds=finished-clip)
        if device.type == "npu":
            timing.update(peak_allocated_bytes=torch.npu.max_memory_allocated(device),
                          peak_reserved_bytes=torch.npu.max_memory_reserved(device))
        result["timing_by_rank"] = gather(timing, world)
    if audit:
        result["by_rank"] = gather(dict(pair_ids=local_ids, fields=counts.tolist(), text_lengths=widths), world)
    return result, probes


def validation_row(index, level, target, text, embedder):
    width = embedder.length(text) if isinstance(text, str) and text.strip() else 0
    if not width:
        status = "missing_text"
    elif width > 512:
        status = "text_too_long"
    else:
        status = "ok" if target.member else "missing_cad_input"
    return dict(index=index, level=level, uid=target.uid, member=target.member, tokens=width,
                geometry_verified=target.geometry_verified, status=status)


def validation_support(store, indices, targets, levels, embedder):
    support = []
    for column, level in enumerate(levels):
        for index, target in enumerate(targets):
            row = validation_row(index, level, target, target.texts[column], embedder)
            if row["status"] == "ok":
                mask = store.get(indices[target.member])["key_padding_mask"][1:]
                x, y = (~mask).sum(0).tolist()
                row.update(fields=x+y, x_fields=x, y_fields=y)
            support.append(row)
    return sorted(support, key=lambda row: (row["level"], row["index"]))


@torch.no_grad()
def validate(model, store, indices, targets, levels, rank, world, batch_size, device, support, *, smoke=False):
    model.eval()
    embedder = model.base_text_embedder
    embedder.truncate = False
    local = []
    pending = []

    def score():
        rows, texts = zip(*pending)
        vec, masks, seq = assemble_batch(store, [indices[r["member"]] for r in rows], device)
        logits, _, _ = teacher_forced_logits(model, list(texts), vec, masks, cad_seq_len=seq)
        require(logits.dtype == torch.float32, "Validation dtype changed")
        scores = two_head_factorized_score(logits.cpu(), vec["cad_vec"].cpu(), masks["key_padding_mask"].cpu())
        for row, s in zip(rows, scores, strict=True):
            local.append(dict(**row, nll=-s["score_total"], fields=s["n_valid_total"],
                              x_nll=-s["score_x"], y_nll=-s["score_y"],
                              x_fields=s["n_valid_x"], y_fields=s["n_valid_y"]))
        pending.clear()

    for j, lev in enumerate(levels):
        for i in range(rank, len(targets), world):
            target = targets[i]
            text = target.texts[j]
            row = validation_row(i, lev, target, text, embedder)
            if row["status"] != "ok":
                local.append(row)
            else:
                pending.append((row, text))
                if len(pending) == batch_size:
                    score()
        if pending:
            score()
    results = sorted((r for part in gather(local, world) for r in part), key=lambda r: (r["level"], r["index"]))
    require(len(results) == len(support) and
            all(all(actual[key] == value for key, value in expected.items())
                for actual, expected in zip(results, support, strict=True)), "Frozen pre-training validation support changed")
    summary = {}
    for lev in levels:
        rows = [r for r in results if r["level"] == lev]
        good = [r for r in rows if r["status"] == "ok"]
        require([r["index"] for r in rows] == list(range(len(targets))) and (smoke or good), "Missing validation axis")
        nll = math.fsum(r["nll"] for r in good)
        fields = sum(r["fields"] for r in good)
        require(math.isfinite(nll) and (smoke or fields > 0), "Invalid native positive score")
        summary[lev] = dict(denominator=len(rows), valid=len(good), statuses=dict(Counter(r["status"] for r in rows)),
                            fields=fields, total_nll=nll, nll=nll/fields if fields else None,
                            valid_uid_sha256=pilot.sha256_lines(r["uid"] for r in good))
    metric = (math.fsum(s["nll"] for s in summary.values()) / len(levels)
              if all(s["fields"] for s in summary.values()) else None)
    return dict(nll=metric, levels=summary), results


def add_stats(stats, step):
    stats["exposure"] += step["pairs"]
    for key in ("by_level", "truncated"):
        for level, n in step[key].items():
            stats[key][level] += n
    for key in ("fields", "smoothed_sum", "nll_sum"):
        for h in range(2):
            stats[key][h] += step[key][h]


def publish_curve(out, state):
    if not state["history"]:
        return
    write_json(out / "curve.json", state["history"])
    if state["completed"]:
        write_json(out / "selected.json", dict(**state["best"], exposure=state["exposure"],
                   improved_at_budget=state["best"]["epoch"] == state["epoch"] - 1,
                   candidates=[r["epoch"] for r in state["history"] if "validation" in r]))


def runtime(args):
    global CONTROL_GROUP
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    require(world in (1, 2), "Registered layouts have one or two ranks")
    if args.device == "npu":
        import torch_npu
        device = torch.device("npu", local)
        torch.npu.set_device(device)
        torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    else:
        device = torch.device("cpu")
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("hccl" if args.device == "npu" else "gloo", timeout=timedelta(hours=2))
        if args.device == "npu" and dist.is_gloo_available():
            CONTROL_GROUP = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    return device, rank, world


def run(args):
    device, rank, world = runtime(args)
    levels = LEVELS[args.source]
    if rank == 0:
        if args.resume:
            require((args.out / "checkpoint.json").is_file(), "Missing committed checkpoint")
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            require(not any(args.out.iterdir()), "Fresh run requires empty output directory")
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    train, targets, archive_pin = axes(args)
    size = len(train) * len(levels)
    batches = math.ceil(size/BATCH)
    if args.train_only and not args.audit_batch:
        require(args.stop_after_updates < batches, "Train-only smoke must stop before the first epoch ends")
    if args.smoke_validation_count:
        require(args.stop_after_updates < batches, "Validation smoke must stop before the first epoch ends")
    members = sorted({r.member for r in train + targets if r.member})
    indices = {s: i for i, s in enumerate(members)}
    store = LazyCandidates(args.cad_zip, members, archive_pin)
    verify_bert_asset_shas(args.bert)
    embedder = TrainingEmbedder.from_local(args.bert, 512, device)
    embedder.pad_multiple = getattr(args, "train_text_pad_multiple", None)
    verify_bert_model_load(embedder, args.bert)
    require(embedder.tokenizer.truncation_side == embedder.tokenizer.padding_side == "right", "Wrong tokenizer sides")
    seed_all(SEED, device)
    model = build_text2cad(embedder, device).to(device)
    require(all(not p.requires_grad for p in embedder.parameters()), "Unfrozen BERT")
    initial = weight_sha(cpu_weights(model))
    require(len(set(gather(initial, world))) == 1, "Fresh non-BERT state differs by rank")
    wrapped = model
    if world > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None if device.type == "cpu" else [device.index],
            broadcast_buffers=False, find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
                                  betas=(0.9, 0.999), weight_decay=0.01, foreach=False)
    milestone = 16 if args.source == FUSION else 8
    candidates = [i for i in (1, 2, 4, 8, 16) if i <= milestone]
    recipe = dict(source=args.source, seed=SEED, levels=levels, world=world, device=device.type,
                  device_model=(torch.npu.get_device_name(device) if device.type == "npu" else "cpu"),
                  torch_version=str(torch.__version__), initial_trainable_sha256=initial,
                  bert_sha256=PINNED_BERT_FILE_SHA256, cad_zip_sha256=archive_pin, global_batch=BATCH,
                  formal_receipt=FUSION_RECEIPT if args.source == FUSION else DEEPCAD_RECEIPT,
                  validation_batch_per_rank=args.validation_batch, train_pairs=size, updates_per_epoch=batches,
                  base_candidates=candidates, conditional_extension=2*milestone, selection="minimum native validation NLL; earliest tie",
                  optimizer="AdamW", initial_lr=1e-4, betas=(0.9,0.999), weight_decay=0.01,
                  epoch_lr_multiplier=0.999, global_clip_norm=0.9, label_smoothing=0.1,
                  training_window="native right truncation 512", validation_window="reject full text over 512",
                  sampler="CPU torch.randperm(seed=17 + epoch - 1), row-major (UID, level), rank-strided batch",
                  train_only=args.train_only, audit_batch=args.audit_batch, smoke_validation_count=args.smoke_validation_count,
                  script_sha256=sha256_file(Path(__file__)),
                  code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    if embedder.pad_multiple is not None:
        recipe["training_text_pad_multiple"] = embedder.pad_multiple
    layout = gather(dict(rank=rank, device=str(device), hostname=socket.gethostname(), pid=os.getpid(),
                         visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                         visible_ascend_devices=os.environ.get("ASCEND_VISIBLE_DEVICES")), world)
    if rank == 0:
        write_json(args.out / ("resumed-layout.json" if args.resume else "run.json"),
                   dict(recipe=recipe, layout=layout, startup_seconds=time.monotonic()-started))
    seed_all(SEED + rank, device)
    state = dict(epoch=1, next_batch=0, updates=0, exposure=0, best=None, history=[], stop_epoch=milestone,
                 stats=stats_for(levels), completed=False)
    support = validation_support(store, indices, targets, levels, embedder) if targets else []
    if rank == 0 and support:
        path = args.out / "validation-support.json"
        if args.resume:
            require(json.loads(path.read_bytes()) == support, "Resume validation support changed")
        else:
            write_json(path, support)
    if args.resume:
        state = restore(args, recipe, model, optimizer, device, rank)
        if rank == 0:
            repair_log(args.out, state["updates"])
            publish_curve(args.out, state)
    if args.audit_batch == "tail":
        state["next_batch"] = batches - 1
    elif args.audit_batch == "third":
        state["next_batch"] = 2
    while not state["completed"] and (args.stop_after_updates is None or state["updates"] < args.stop_after_updates):
        epoch = state["epoch"]
        order = epoch_order(size, epoch)
        model.train(not bool(args.audit_batch))
        embedder.eval()
        embedder.truncate = True
        for b in range(state["next_batch"], batches):
            ids = order[b*BATCH:(b+1)*BATCH].tolist()
            step, probes = update(model, wrapped, optimizer, store, indices, train, levels,
                                  ids, rank, world, device, bool(args.audit_batch), state["updates"] < 3)
            state["updates"] += 1
            state["next_batch"] = b + 1
            state["exposure"] += len(ids)
            add_stats(state["stats"], step)
            record = dict(epoch=epoch, batch=b, update=state["updates"], exposure=state["exposure"],
                          lr=optimizer.param_groups[0]["lr"], **step)
            if rank == 0:
                # A resumed process can replay steps newer than the last committed checkpoint.
                # The commit cursor is authoritative; repair the uncommitted log suffix on resume.
                with (args.out / "steps.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                if state["updates"] <= 3 or state["updates"] % 25 == 0:
                    print(json.dumps(record), flush=True)
            if args.audit_batch:
                if rank == 0:
                    atomic_save(args.out / "update-witness.pt", dict(pair_ids=ids, **record, parameters=probes))
                save(args, recipe, model, optimizer, state, device, rank, world)
                return
            if b + 1 < batches:
                stop = state["updates"] == args.stop_after_updates
                if stop or state["updates"] % args.save_every == 0:
                    save(args, recipe, model, optimizer, state, device, rank, world)
                if stop:
                    break
        if state["next_batch"] < batches:
            break
        require(state["stats"]["exposure"] == size and all(n == len(train) for n in state["stats"]["by_level"].values()),
                "Epoch exposure does not cover the registered pairs exactly once")
        row = dict(epoch=epoch, global_update=state["updates"], lr=optimizer.param_groups[0]["lr"], train=state["stats"])
        counts = row["train"]["fields"]
        row["train"]["nll"] = sum(row["train"]["nll_sum"])/sum(counts)
        row["train"]["smoothed_loss"] = sum(v/c for v,c in zip(row["train"]["smoothed_sum"], counts))/2
        if epoch in candidates or epoch == milestone * 2:
            decision, details = validate(model, store, indices, targets, levels, rank, world,
                                         args.validation_batch, device, support)
            row["validation"] = decision
            prior = state["best"]
            if prior is not None:
                first = state["history"][0]["validation"]["levels"]
                require(all((v["fields"], v["valid_uid_sha256"]) == (first[k]["fields"], first[k]["valid_uid_sha256"])
                            for k, v in decision["levels"].items()), "Validation support changed")
            if prior is None or decision["nll"] < prior["nll"]:
                state["best"] = dict(epoch=epoch, nll=decision["nll"], file=f"epoch-{epoch:02}.pt")
                if epoch == milestone:
                    state["stop_epoch"] = 2 * milestone
            if rank == 0:
                write_json(args.out / f"validation-{epoch:02}.json", dict(summary=decision, rows=details))
        state["history"].append(row)
        for group in optimizer.param_groups:
            group["lr"] *= 0.999
        state.update(epoch=epoch+1, next_batch=0, stats=stats_for(levels), completed=epoch == state["stop_epoch"])
        save(args, recipe, model, optimizer, state, device, rank, world, epoch)
        if rank == 0:
            publish_curve(args.out, state)
    if args.smoke_validation_count:
        limits = [row for row in support if row["index"] < args.smoke_validation_count]
        summary, details = validate(model, store, indices, targets[:args.smoke_validation_count], levels,
                                    rank, world, args.validation_batch, device, limits, smoke=True)
        if rank == 0:
            write_json(args.out / f"smoke-validation-{state['updates']}.json",
                       dict(scope="development smoke, ineligible for selection", summary=summary, rows=details))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=LEVELS, required=True)
    for name in ("manifests", "cad-zip", "bert", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--fusion-validation-qrels", type=Path)
    p.add_argument("--device", choices=("npu", "cpu"), default="npu")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--train-text-pad-multiple", type=int, choices=(32, 64),
                   help="Compare right padding in a fresh bounded T training probe")
    p.add_argument("--stop-after-updates", type=int)
    p.add_argument("--audit-batch", choices=("first", "third", "tail"),
                   help="One dropout-free real batch for comparing training layouts")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--validation-batch", type=int, default=8)
    p.add_argument("--smoke-validation-count", type=int, default=0,
                   help="After the bounded update, score the first N registered validation rows per level")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.train_text_pad_multiple is not None:
        require(args.source == "t2cad-v1.0" and args.train_only and not args.resume,
                "Text width changes are limited to fresh train-only T probes")
    if args.audit_batch:
        require(args.train_only and not args.resume and args.stop_after_updates in (None, 1),
                "Audit requires --train-only, a fresh run, and exactly one update")
        args.stop_after_updates = 1
    require(not args.train_only or args.stop_after_updates is not None, "Train-only smoke requires an update limit")
    require(args.stop_after_updates is None or args.stop_after_updates > 0, "Update limit must be positive")
    require(args.save_every > 0 and args.validation_batch > 0, "Positive checkpoint interval and validation batch required")
    require(0 <= args.smoke_validation_count <= 512 and (not args.smoke_validation_count or
            not args.train_only and args.stop_after_updates), "Validation smoke needs bounded training with validation enabled")
    require(args.source != FUSION or args.train_only or args.fusion_validation_qrels is not None,
            "Fusion validation requires --fusion-validation-qrels")
    try:
        run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
