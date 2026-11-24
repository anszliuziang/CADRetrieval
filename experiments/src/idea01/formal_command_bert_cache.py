"""Read the certified training-only pooled BERT cache for the T command dual tower."""

from hashlib import sha256
from importlib import metadata
import json

import numpy as np


SOURCE = "t2cad-v1.0"
BATCH_SIZE, TEXT_COUNT, BATCH_COUNT, TAIL, WIDTH = 32, 446_461, 13_952, 29, 1024
SET_SHA = "526c8c73cf05a901f41d6b53c949d547c92a773efa2483307a0b8d9c3db8139b"
COST_SHA = "05d18584bf715bcd3ac13e1a35c710935de5da38f09d3fe686d4074fd7dfeac7"
TRAIN_SHA = "dc68c8b83d973c4b11a80edb8e4431094420ce504c2f41e75a41824f9d940af5"
POLICY = {
    "source": SOURCE, "split": "train", "train_manifest_sha256": TRAIN_SHA,
    "workset_sha256": SET_SHA, "cost_sha256": COST_SHA, "attention": "eager",
    "backbone_dtype": "torch.float32", "pool_dtype": "float32", "hidden_width": WIDTH,
    "pooling": "masked mean of FP32 last hidden states", "batch_size": BATCH_SIZE,
    "batch_count": BATCH_COUNT, "text_count": TEXT_COUNT, "final_batch_size": TAIL,
    "window": 512, "padding_side": "right", "truncation_side": "right", "pad_multiple": 64,
    "batch_order": "stable effective token length after first-seen manifest/L0-L3 enumeration",
}
AXIS_SCHEMA = "idea01-t-command-bert-axis-v1"
CACHE_SCHEMA = "idea01-t-command-bert-cache-v1"
INDEX = "text_index.jsonl"
MATRIX = "pooled.npy"


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(*names):
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def tokenizer_identity(tokenizer):
    backend = tokenizer.backend_tokenizer.to_str()
    return {"class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "backend_sha256": sha256(backend.encode("utf-8")).hexdigest(),
            "versions": package_versions("transformers", "tokenizers")}


def encoding_runtime(group):
    return {"device_kind": group.device.type, "device_model": group.backend.get_device_name(group.device),
            "versions": package_versions("torch", "torch-npu", "transformers", "tokenizers")}


def read_index(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            text = row.pop("text")
            yield text, row


def open_training_cache(path, texts, manifest_sha, model_source, tokenizer, group):
    """Validate cache provenance and bind row numbers to the runner's exact training strings."""
    raw = (path / "receipt.json").read_bytes()
    receipt = json.loads(raw)
    if (receipt["schema"] != CACHE_SCHEMA or receipt["policy"] != POLICY
            or manifest_sha != TRAIN_SHA or len(texts) != TEXT_COUNT
            or receipt["model"] != {key: model_source[key] for key in ("repo", "revision", "files")}
            or receipt["tokenizer"] != tokenizer_identity(tokenizer)
            or receipt["encoding_runtime"] != encoding_runtime(group)):
        raise ValueError("Pooled BERT cache is incompatible with the formal T training input or runtime")
    for name in (INDEX, MATRIX):
        if file_sha256(path / name) != receipt["files_sha256"][name]:
            raise ValueError(f"Pooled BERT cache file differs from its merged receipt: {name}")
    values = np.load(path / MATRIX, mmap_mode="r", allow_pickle=False)
    if values.shape != (TEXT_COUNT, WIDTH) or values.dtype != np.dtype("float32"):
        raise ValueError("Invalid pooled BERT cache shape or dtype")
    text_rows = []
    for index, (actual, (cached, row)) in enumerate(zip(texts, read_index(path / INDEX), strict=True)):
        if actual != cached:
            raise ValueError(f"Pooled BERT cache diverges at exact training text {index}")
        text_rows.append(row)
    return values, text_rows, {"path": str(path.resolve()), "receipt_sha256": sha256(raw).hexdigest(),
                               **receipt}
