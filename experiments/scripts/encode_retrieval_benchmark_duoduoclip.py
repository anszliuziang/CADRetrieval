"""Audit, smoke and encode frozen DeepCAD axes with official DuoduoCLIP on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import os
import platform
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from encode_retrieval_benchmark_external import (
    axis_files, load_gallery, load_texts, log, normalize_rows, output_lock, packages, write_axis,
)
from eval_retrieval_benchmark_fill import array_artifact, read_axis_status, verify_unit_vectors
from prepare_retrieval_benchmark_fill import CONDITIONS, load_json, sha256, verify_sha256, write_json


SPEC = Path(__file__).resolve().parents[1] / "configs" / "retrieval_benchmark_duoduoclip.json"
FRAME_COUNT = 8
FRAME_SIZE = 224
VIEW_PERMUTATION_ATOL = 2e-5
TOP_LEVEL_METADATA = {".cache", ".gitattributes", "README.md", "LICENSE", "assets"}


def audit_snapshot(snapshot, spec):
    """Check every model and demo input before importing any publisher code."""
    pins = spec["files"] | spec["demo_images"]
    unpinned = {path.name for path in snapshot.iterdir()} - spec["files"].keys() - TOP_LEVEL_METADATA
    if unpinned:
        raise ValueError(f"unpinned snapshot entries: {sorted(unpinned)}")
    weights = snapshot / "model.safetensors"
    if weights.stat().st_size != spec["weights_bytes"]:
        raise ValueError("snapshot does not contain the complete publisher safetensors file")
    for name, expected in pins.items():
        verify_sha256(snapshot / name, expected)
    return {"repo": spec["repo"], "revision": spec["revision"], "files": pins}


def pack_views(image_processor, groups, frame_count):
    if not groups or any(len(group) != frame_count for group in groups):
        raise ValueError(f"every shape must contain exactly {frame_count} ordered views")
    pixels = image_processor(images=[image for group in groups for image in group], return_tensors="np")["pixel_values"]
    expected = (len(groups) * frame_count, 3, FRAME_SIZE, FRAME_SIZE)
    if pixels.shape != expected or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError(f"expected finite float32 processor pixels with shape {expected}, got {pixels.shape}")
    return np.ascontiguousarray(pixels).reshape(len(groups), frame_count, 3, FRAME_SIZE, FRAME_SIZE)


def tokenized_prefix(tokenizer, texts, window):
    full = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True, verbose=False)["input_ids"]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=window, add_special_tokens=True,
                        return_attention_mask=True, return_tensors="np")
    ids, masks = encoded["input_ids"], encoded["attention_mask"]
    if (len(full) != len(texts) or ids.ndim != 2 or ids.shape != masks.shape or ids.shape[0] != len(texts)
            or ids.shape[1] > window):
        raise ValueError("tokenizer batch differs from the registered text axis or window")
    counts = []
    for original, row, mask in zip(full, ids, masks, strict=True):
        count = len(original)
        kept = min(count, window)
        expected = original if count <= window else original[:window - 1] + [original[-1]]
        if (count < 2 or original[-1] != tokenizer.eos_token_id or not np.all(mask[:kept] == 1)
                or not np.all(mask[kept:] == 0) or row[:kept].tolist() != expected):
            raise ValueError("tokenizer did not keep the registered prefix and EOS")
        counts.append(count)
    return ids, masks, counts


def require_projection(values, count, dim):
    if values.shape != (count, dim) or values.dtype != np.float32:
        raise ValueError(f"expected {count} float32 projections of dimension {dim}, got {values.shape}/{values.dtype}")
    return values


class ModelEncoder:
    def __init__(self, model, image_processor, tokenizer, torch, dim, window):
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.torch = torch
        self.dim = dim
        self.window = window

    def images(self, groups, frame_count):
        pixels = self.torch.from_numpy(pack_views(self.image_processor, groups, frame_count))
        with self.torch.inference_mode():
            features = self.model.get_image_features(pixel_values=pixels).detach().cpu().numpy()
        return require_projection(features, len(groups), self.dim)

    def texts(self, texts):
        ids, masks, counts = tokenized_prefix(self.tokenizer, texts, self.window)
        with self.torch.inference_mode():
            features = self.model.get_text_features(input_ids=self.torch.from_numpy(ids),
                                                    attention_mask=self.torch.from_numpy(masks)).detach().cpu().numpy()
        return require_projection(features, len(texts), self.dim), counts


@contextmanager
def load_encoder(snapshot, spec, threads):
    audit = audit_snapshot(snapshot, spec["model"])
    model_spec = spec["model"]
    installed = importlib.metadata.version("transformers")
    if installed != model_spec["transformers"]:
        raise RuntimeError(f"DuoduoCLIP requires Transformers {model_spec['transformers']}, found {installed}")
    if "transformers" in sys.modules or "huggingface_hub" in sys.modules:
        raise RuntimeError("the audited loader must start in a fresh process")
    with tempfile.TemporaryDirectory(prefix="duoduoclip_modules_") as private_modules:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_MODULES_CACHE=private_modules,
                          TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        import torch
        from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizerFast

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("highest")
        model, loading = AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True, use_safetensors=True,
            dtype=torch.float32, attn_implementation=model_spec["attention"], output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"publisher model did not load strictly: {loading}")
        for instance, name in ((model, "modeling_duoduoclip.py"), (model.config, "configuration_duoduoclip.py")):
            source = Path(sys.modules[type(instance).__module__].__file__).resolve()
            if not source.is_relative_to(Path(private_modules).resolve()):
                raise ValueError(f"publisher class loaded outside the isolated code cache: {type(instance).__name__}")
            verify_sha256(source, model_spec["files"][name])
        model.eval().requires_grad_(False)
        window = spec["text"]["max_tokens_including_special_tokens"]
        dim = model_spec["projection_dim"]
        if (model.config.model_type != "duoduoclip" or model.config.projection_dim != dim
                or model.config.layers_threshold != 6 or model.config.vision_config.image_size != FRAME_SIZE
                or model.config.text_config.max_position_embeddings != window
                or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError("loaded model does not match the frozen DuoduoCLIP-B/32 CPU FP32 architecture")
        image_processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
        tokenizer = CLIPTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        if tokenizer.model_max_length != window or tokenizer.eos_token_id != model.config.text_config.eos_token_id:
            raise ValueError("publisher tokenizer disagrees with the frozen text tower's window or EOS")
        tokenizer.truncation_side = spec["text"]["truncation_side"]
        tokenizer.padding_side = "right"
        versions = packages("transformers", "tokenizers", "safetensors", "Pillow", "numpy")
        yield ModelEncoder(model, image_processor, tokenizer, torch, dim, window), {
            "model_snapshot": audit, "threads": threads, "device": "cpu", "forward_dtype": "float32",
            "platform": platform.platform(), "python": sys.version.split()[0],
            "image_processor": type(image_processor).__name__, "tokenizer": type(tokenizer).__name__,
            "versions": versions | {"torch": str(torch.__version__)},
        }


def unit_vectors(values, context):
    values = values.astype(np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"invalid projection in {context}")
    return values / norms


def official_demo(encoder, snapshot, spec):
    views = []
    for relative in spec["demo_images"]:
        with Image.open(snapshot / relative) as image:
            views.append(image.convert("RGB"))
    raw_image = encoder.images([views], len(views))
    raw_text, lengths = encoder.texts(spec["demo_texts"])
    image = unit_vectors(raw_image, "official demo image")
    texts = unit_vectors(raw_text, "official demo text")
    reversed_image = unit_vectors(encoder.images([views[::-1]], len(views)), "reversed demo image")
    delta = float(np.max(np.abs(image - reversed_image)))
    if delta > VIEW_PERMUTATION_ATOL or max(lengths) > encoder.window:
        raise ValueError(f"official demo differs on reversed views or exceeds its text window, delta={delta:.6g}")
    return {"image_projection_shape": list(raw_image.shape), "text_projection_shape": list(raw_text.shape),
            "pixel_shape": [1, len(views), 3, FRAME_SIZE, FRAME_SIZE],
            "image_norm": float(np.linalg.norm(image)), "text_norms": np.linalg.norm(texts, axis=1).tolist(),
            "text_token_counts": lengths, "reversed_views_max_abs_delta": delta,
            "cosines": dict(zip(spec["demo_texts"], (image @ texts.T)[0].tolist(), strict=True))}


def load_render_axis(render_root, gallery, binding, spec_sha):
    summary_path = render_root / "render_summary.json"
    summary = load_json(summary_path)
    if (summary.get("full_gallery") is not True or summary["rows"] != len(gallery)
            or summary["gallery_sha256"] != binding["gallery_sha256"]
            or summary["source_steps_sha256"] != binding["steps_manifest_sha256"]
            or summary["render_spec_sha256"] != spec_sha):
        raise ValueError("render run does not bind the full frozen gallery, STEP sources and render specification")
    status_path = render_root / "gallery_render_status.jsonl"
    verify_sha256(status_path, summary["status_sha256"])
    records = read_axis_status(status_path, gallery, is_text=False)
    source = hashlib.sha256()
    view_names = [f"{i:02}.png" for i in range(FRAME_COUNT)]
    for row, record in zip(gallery, records, strict=True):
        digest = record["source_sha256"]
        if (record["uid"] != row["uid"] or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or record["producer"] != summary["producer"] or record["render_spec_sha256"] != spec_sha):
            raise ValueError(f"render status does not bind the registered source UID: {row['uid']}")
        if record["valid"] and [view["filename"] for view in record["views"]] != view_names:
            raise ValueError(f"render record omits or reorders a fixed view: {row['uid']}")
        source.update(f"{row['uid']}\t{digest}\n".encode("utf-8"))
    if (source.hexdigest() != binding["steps_manifest_sha256"]
            or sum(record["valid"] for record in records) != summary["valid"]):
        raise ValueError("render status content disagrees with its registered STEP manifest or coverage")
    return records, {"render_summary_sha256": sha256(summary_path), "render_status_sha256": summary["status_sha256"],
                     "render_spec_sha256": spec_sha, "source_steps_sha256": summary["source_steps_sha256"]}


def open_views(record, render_root, image_size):
    folder = render_root / "views" / record["uid"].replace("/", "__")
    images = []
    for view in record["views"]:
        path = folder / view["filename"]
        data = path.read_bytes()
        if len(data) != view["bytes"] or hashlib.sha256(data).hexdigest() != view["sha256"]:
            raise ValueError(f"render PNG does not match its committed bytes: {path}")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (image_size, image_size):
                raise ValueError(f"render PNG has the wrong format, mode or size: {path}")
            images.append(image.copy())
    return images


def gallery_smoke(records, read_views, images_forward):
    sample = [record for record in records if record["valid"]][:2]
    if len(sample) != 2:
        raise ValueError("eight-view smoke requires two admitted render records")
    groups = [read_views(record) for record in sample]
    joint = unit_vectors(images_forward(groups, FRAME_COUNT), "batched smoke")
    individual = unit_vectors(np.concatenate([images_forward([group], FRAME_COUNT) for group in groups]), "single smoke")
    reversed_views = unit_vectors(images_forward([group[::-1] for group in groups], FRAME_COUNT), "reversed smoke")
    max_batch_delta = float(np.max(np.abs(joint - individual)))
    max_view_delta = float(np.max(np.abs(joint - reversed_views)))
    if max(max_batch_delta, max_view_delta) > VIEW_PERMUTATION_ATOL:
        raise ValueError(f"eight-view inference changed across view order or batch placement: {max_batch_delta}, {max_view_delta}")
    return {"uids": [record["uid"] for record in sample], "pixel_shape": [2, FRAME_COUNT, 3, FRAME_SIZE, FRAME_SIZE],
            "batched_max_abs_delta": max_batch_delta, "reversed_views_max_abs_delta": max_view_delta}


def gallery_embeddings(gallery, records, read_views, images_forward, *, batch_size, dim):
    raw = np.zeros((len(gallery), dim), dtype=np.float32)
    details = []
    for lo in range(0, len(gallery), batch_size):
        positions, groups = [], []
        for index, record in enumerate(records[lo:lo + batch_size], start=lo):
            info = {"source_sha256": record["source_sha256"]}
            if record["valid"]:
                positions.append(index)
                groups.append(read_views(record))
                info["views_encoded"] = FRAME_COUNT
            else:
                info.update(reason=f"render failed: {record['reason']}", views_encoded=0)
            details.append(info)
        if groups:
            raw[positions] = require_projection(images_forward(groups, FRAME_COUNT), len(groups), dim)
        if (lo // batch_size) % 100 == 0 or len(details) == len(gallery):
            log("duoduoclip_gallery_progress", processed=len(details), total=len(gallery))
    return normalize_rows(raw, gallery, details, epsilon=0.0)


def text_embeddings(rows, text_forward, *, batch_size, dim, window):
    raw = np.empty((len(rows), dim), dtype=np.float32)
    details = []
    for lo in range(0, len(rows), batch_size):
        texts = [row["text"] for row in rows[lo:lo + batch_size]]
        features, counts = text_forward(texts)
        if len(counts) != len(texts) or any(type(count) is not int or count < 2 for count in counts):
            raise ValueError("text token counts do not match the text batch")
        raw[lo:lo + len(texts)] = require_projection(features, len(texts), dim)
        details.extend({"tokens_before_truncation": count, "tokens_encoded": min(count, window),
                        "truncated": count > window} for count in counts)
        if (lo // batch_size) % 20 == 0 or len(details) == len(rows):
            log("duoduoclip_text_progress", processed=len(details), total=len(rows))
    return normalize_rows(raw, rows, details, epsilon=0.0)


def write_receipt(path, value):
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"existing receipt differs, use a new output: {path}")
        return
    with tempfile.TemporaryDirectory(dir=path.parent, prefix="duoduoclip_receipt_") as scratch:
        staged = Path(scratch) / path.name
        write_json(staged, value)
        staged.replace(path)


def publish_axis(output, stem, rows, meta, dim, build, *, is_text=False):
    vector_path, status_path, meta_path = axis_files(output, stem)
    if meta_path.exists():
        saved = load_json(meta_path)
        if any(saved.get(key) != value for key, value in meta.items()):
            raise ValueError(f"existing encoding has different provenance: {meta_path}")
        verify_sha256(status_path, saved["status_sha256"])
        statuses = read_axis_status(status_path, rows, is_text=is_text)
        vectors, _ = array_artifact(vector_path, saved, ("vectors_sha256",))
        log("duoduoclip_axis_reused", axis=stem, rows=len(rows))
    else:
        if vector_path.exists() or status_path.exists():
            raise FileExistsError(f"partial encoding exists without its metadata: {meta_path}")
        vectors, statuses = build()
    if vectors.shape != (len(rows), dim):
        raise ValueError(f"unexpected embedding axis shape: {stem}")
    valid = np.array([row["valid"] for row in statuses], dtype=bool)
    verify_unit_vectors(vectors, valid, vector_path)
    if not meta_path.exists():
        write_axis(output, stem, vectors, statuses, meta)
        log("duoduoclip_axis_complete", axis=stem, rows=len(rows), valid=int(valid.sum()))


def scoring_inputs(frozen, binding, conditions):
    manifest_path = frozen / "freeze.json"
    verify_sha256(manifest_path, binding["freeze_sha256"])
    manifest = load_json(manifest_path)
    gallery, gallery_sha = load_gallery(frozen / "gallery.jsonl")
    if (manifest["domain"] != "DeepCAD" or gallery_sha != binding["gallery_sha256"]
            or gallery_sha != manifest["files"]["gallery.jsonl"]
            or len(gallery) != binding["gallery_count"] or len(gallery) != manifest["gallery_count"]):
        raise ValueError("the DeepCAD axis differs from the frozen DuoduoCLIP input")
    text_axes = {}
    for condition in dict.fromkeys(conditions):
        name = f"texts_{condition}.jsonl"
        rows, digest = load_texts(frozen / name)
        if digest != manifest["files"][name] or len(rows) != manifest["conditions"][condition]["unique_texts"]:
            raise ValueError(f"unexpected frozen text axis: {condition}")
        text_axes[condition] = (rows, digest)
    return gallery, gallery_sha, text_axes


def encode(args, spec):
    binding = spec["deepcad"]
    gallery, gallery_sha, text_axes = scoring_inputs(args.frozen, binding, args.condition or binding["conditions"])
    records, render_meta = load_render_axis(args.renders, gallery, binding, sha256(SPEC))
    read_views = partial(open_views, render_root=args.renders, image_size=spec["render"]["size_px"])
    log("duoduoclip_input_axes_verified", gallery=len(gallery), gallery_sha256=gallery_sha, **render_meta)
    with output_lock(args.output), load_encoder(args.snapshot, spec, args.threads) as (encoder, model_meta):
        demo = official_demo(encoder, args.snapshot, spec["model"])
        smoke = gallery_smoke(records, read_views, encoder.images)
        demo_path = args.output / "duoduoclip_demo.json"
        write_receipt(demo_path, {"model": model_meta, "official_demo": demo, "eight_view_smoke": smoke})
        log("duoduoclip_smoke_passed", official_demo=demo, eight_view_smoke=smoke)
        meta = {"method": spec["method"], "weights": model_meta["model_snapshot"],
                "model_revision": spec["model"]["revision"], "training_source": spec["training_source"],
                "checkpoint_sha256": spec["model"]["files"]["model.safetensors"],
                "preprocessing": spec["model"]["preprocessing"], "text_prompt": spec["text"]["template"],
                "freeze_sha256": binding["freeze_sha256"], "demo_sha256": sha256(demo_path),
                "encoder_sha256": sha256(Path(__file__)), "runtime": model_meta}
        publish_axis(args.output, "gallery", gallery, meta | render_meta | {
            "axis_sha256": gallery_sha, "image_batch": args.image_batch, "view_count": FRAME_COUNT,
        }, encoder.dim, partial(gallery_embeddings, gallery, records, read_views, encoder.images,
                               batch_size=args.image_batch, dim=encoder.dim))
        for condition, (rows, digest) in text_axes.items():
            name = f"texts_{condition}"
            log("duoduoclip_text_condition", condition=condition, rows=len(rows), text_axis_sha256=digest)
            publish_axis(args.output, name, rows, meta | {"axis_sha256": digest, "condition": condition,
                         "text_batch": args.text_batch, "context_window": encoder.window}, encoder.dim,
                         partial(text_embeddings, rows, encoder.texts, batch_size=args.text_batch,
                                 dim=encoder.dim, window=encoder.window), is_text=True)


def bounded_count(maximum):
    def convert(value):
        count = int(value)
        if not 1 <= count <= maximum:
            raise argparse.ArgumentTypeError(f"choose an integer from 1 to {maximum}")
        return count
    return convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Local publisher snapshot with all pinned files")
    parser.add_argument("--threads", type=bounded_count(32), default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Verify model and demo files without importing Torch")
    demo = sub.add_parser("demo", help="Run the publisher's text-to-shape demo")
    demo.add_argument("--output", type=Path, required=True, help="JSON receipt for the demo")
    full = sub.add_parser("encode", help="Audit, smoke and encode the full frozen DeepCAD gallery")
    full.add_argument("--frozen", type=Path, required=True)
    full.add_argument("--renders", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True, help="Directory for shared evaluator artifacts")
    full.add_argument("--condition", choices=CONDITIONS, action="append", help="Defaults to all frozen text axes")
    full.add_argument("--image-batch", type=bounded_count(8), default=2, help="Shapes per batch, with eight views each")
    full.add_argument("--text-batch", type=bounded_count(128), default=32)
    args = parser.parse_args()
    args.snapshot = args.snapshot.resolve(strict=True)
    spec = load_json(SPEC)
    started = time.perf_counter()
    if args.command == "audit":
        log("duoduoclip_snapshot_verified", **audit_snapshot(args.snapshot, spec["model"]))
    elif args.command == "demo":
        with load_encoder(args.snapshot, spec, args.threads) as (encoder, model_meta):
            receipt = {"model": model_meta, "official_demo": official_demo(encoder, args.snapshot, spec["model"])}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_receipt(args.output, receipt)
            log("duoduoclip_demo_passed", **receipt["official_demo"])
    else:
        encode(args, spec)
    log("duoduoclip_complete", command=args.command, seconds=time.perf_counter() - started)


if __name__ == "__main__":
    main()
