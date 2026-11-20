"""Encode frozen DeepCAD retrieval axes with the three historical G5 baselines."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


HISTORICAL_SIDECARS = {
    "clip": "c0711c197fcb999776d304274da64e9ea77a2d53054ceda933ab9c1cf09275f9",
    "openshape": "1c8ebc4616d652d3ad5c9e7755f3b33ebefdecf8bb6b86e8c39310561744640e",
    "mmcad_v2": "c08d47a997b92188ab3437facb8b4f86e6bffeb956ee42b5f779b3592af2ca51",
}
CLIP_WEIGHTS = "99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d"
CLIP_PROCESSOR_FILES = {
    "config.json": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
    "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    "tokenizer.json": "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    "tokenizer_config.json": "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
}
CLIP_RENDER_SPEC = "938b800c50f3113042f0cc602e19a55ceea5039ac4fd3b780643bd6cb5f10079"
MM_G5_CHECKSUMS = "0228d6d26bf9c537cd633d07b73f6732eb12339d32cff23e196bd21c67fb7e80"
DIMENSIONS = {"clip": 512, "openshape": 1280, "mmcad_v2": 768}
DEFAULT_BATCH = {"clip": 64, "openshape": 32, "mmcad_v2": 32}
DEFAULT_THREADS = {"clip": 8, "openshape": 16, "mmcad_v2": 8}
CONDITIONS = ("ct_deepcad", "l2", "l1", "l3")
REPLAY_TOLERANCE = 5e-6  # CPU float32 encoders, float64 normalization and dot product.


def file_sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"SHA-256 differs from the registered input: {path}")


def read_rows(path):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8-sig").split("\n") if line.strip()]
    if not rows:
        raise ValueError(f"empty axis: {path}")
    return rows, hashlib.sha256(payload).hexdigest()


def load_gallery(path):
    rows, digest = read_rows(path)
    indices, uids = [], []
    for row in rows:
        if set(row) != {"cad_index", "uid", "member"}:
            raise ValueError(f"unexpected gallery axis fields: {path}")
        cad_index, uid, member = row["cad_index"], row["uid"], row["member"]
        if type(cad_index) is not int or cad_index < 0 or not isinstance(member, str) or not member:
            raise ValueError(f"invalid gallery row: {row}")
        if not isinstance(uid, str) or not re.fullmatch(r"[0-9]{4}/[0-9]{8}", uid):
            raise ValueError(f"invalid DeepCAD UID: {uid}")
        indices.append(cad_index)
        uids.append(uid)
    if indices != sorted(set(indices)) or len(set(uids)) != len(uids):
        raise ValueError("gallery must have unique UIDs and strictly ordered historical cad_index values")
    return rows, digest


def load_texts(path, *, historical=False):
    rows, digest = read_rows(path)
    seen = set()
    required = {"text_index", "text", "text_sha256"}
    allowed = required | {"n_chars"} if historical else required
    for i, row in enumerate(rows):
        if not required <= set(row) <= allowed:
            raise ValueError(f"unexpected text axis fields: {path}")
        text, text_sha = row["text"], row["text_sha256"]
        if (type(row["text_index"]) is not int or row["text_index"] != i
                or not isinstance(text, str) or not text or text != text.strip()
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != text_sha or text_sha in seen):
            raise ValueError(f"invalid or duplicate text axis row {i}: {path}")
        seen.add(text_sha)
    return rows, digest


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def log(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=True, allow_nan=False), flush=True)


def prepare_replay(gallery_path, l0_path, historical_gallery, historical_texts, matrix, sidecar, out,
                   *, expected_sidecar_sha256):
    """Bind sampled historical L0 scores to exact original texts and the complete gallery."""
    gallery, gallery_sha = load_gallery(gallery_path)
    l0, l0_sha = load_texts(l0_path)
    old_gallery, old_gallery_sha = read_rows(historical_gallery)
    old_texts, old_texts_sha = load_texts(historical_texts, historical=True)
    check_sha(sidecar, expected_sidecar_sha256)
    history = json.loads(sidecar.read_bytes())
    indices = [row["cad_index"] for row in gallery]
    if (indices[-1] >= len(old_gallery)
            or any(old_gallery[row["cad_index"]]["cad_id"] != row["uid"] for row in gallery)):
        raise ValueError("historical candidate UID and frozen cad_index disagree")
    if (history["schema"] != "text2cad-method-scores-sidecar-v2" or history["direction"] != "higher"
            or history["dtype"] != "float64" or history["text_indices"] != list(range(len(old_texts)))
            or history["cad_indices"] != indices):
        raise ValueError("historical score format or candidate/text axis differs from the frozen input")
    shape = (len(old_texts), len(gallery))
    if matrix.stat().st_size != history["matrix_bytes"] or history["matrix_bytes"] != 8 * np.prod(shape):
        raise ValueError("historical score byte count disagrees with its axes")
    check_sha(matrix, history["matrix_sha256"])
    frozen = {row["text_sha256"]: row for row in l0}
    shared = [row for row in old_texts if row["text_sha256"] in frozen]
    if len(shared) != len(l0) or any(row["text"] != frozen[row["text_sha256"]]["text"] for row in shared):
        raise ValueError("frozen L0 text axis does not map to the original G5 texts")
    ordered = sorted(shared, key=lambda row: (len(row["text"]), row["text_index"]))
    selected = sorted((ordered[i] for i in np.linspace(0, len(ordered) - 1, min(16, len(ordered)), dtype=int)),
                      key=lambda row: row["text_index"])
    rows = [{"historical_text_index": row["text_index"],
             "frozen_text_index": frozen[row["text_sha256"]]["text_index"],
             "text": row["text"], "text_sha256": row["text_sha256"]} for row in selected]
    source = np.memmap(matrix, mode="r", dtype="<f8", shape=shape)
    scores = np.asarray(source[[row["historical_text_index"] for row in rows]], dtype="<f8")
    if not np.isfinite(scores).all():
        raise ValueError("historical reference contains nonfinite scores")
    smoke = set(np.linspace(0, len(gallery) - 1, min(32, len(gallery)), dtype=int).tolist())
    for row in scores[:4]:
        order = np.lexsort((indices, -row))
        smoke.update(order[18:22].tolist())
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "scores.npy", scores, allow_pickle=False)
    save_json(out / "replay.json", {
        "gallery_sha256": gallery_sha, "l0_axis_sha256": l0_sha, "historical_gallery_sha256": old_gallery_sha,
        "historical_texts_sha256": old_texts_sha, "historical_sidecar_sha256": expected_sidecar_sha256,
        "historical_scores_sha256": history["matrix_sha256"], "scores_sha256": file_sha256(out / "scores.npy"),
        "texts": rows, "smoke_gallery_positions": sorted(smoke), "score_atol": REPLAY_TOLERANCE,
    })
    log("replay_reference_prepared", query_rows=len(rows), gallery_rows=len(gallery), smoke_gallery=len(smoke),
        replay_sha256=file_sha256(out / "replay.json"))


def load_replay(directory, method, gallery_sha, l0, l0_sha, gallery_size):
    path = directory / "replay.json"
    meta = json.loads(path.read_bytes())
    scores_path = directory / "scores.npy"
    check_sha(scores_path, meta["scores_sha256"])
    scores = np.load(scores_path, allow_pickle=False)
    positions = meta["smoke_gallery_positions"]
    if (meta["gallery_sha256"] != gallery_sha or meta["l0_axis_sha256"] != l0_sha
            or meta["historical_sidecar_sha256"] != HISTORICAL_SIDECARS[method]
            or meta["score_atol"] != REPLAY_TOLERANCE
            or len(meta["texts"]) != min(16, len(l0)) or len({row["frozen_text_index"] for row in meta["texts"]}) != len(meta["texts"])
            or scores.shape != (len(meta["texts"]), gallery_size) or scores.dtype != np.dtype("<f8")
            or not np.isfinite(scores).all() or not positions or positions != sorted(set(positions))
            or positions[0] < 0 or positions[-1] >= gallery_size):
        raise ValueError("replay reference does not match the frozen axes or G5 provenance")
    rows = [l0[row["frozen_text_index"]] for row in meta["texts"]]
    if any({key: row[key] for key in ("text", "text_sha256")}
           != {key: source[key] for key in ("text", "text_sha256")}
           for row, source in zip(rows, meta["texts"], strict=True)):
        raise ValueError("replay reference text differs from frozen L0")
    return rows, scores, positions, file_sha256(path)


def compare_replay(expected, observed, cad_indices, *, atol):
    if expected.shape != observed.shape or expected.ndim != 2 or expected.shape[1] != len(cad_indices):
        raise ValueError("cosine replay has incompatible comparison axes")
    max_error = float(np.max(np.abs(expected - observed)))
    rank_changes = near_swaps = distant_swaps = top20_order_changes = top20_membership_changes = 0
    for old, new in zip(expected, observed, strict=True):
        previous = np.lexsort((cad_indices, -old))
        current = np.lexsort((cad_indices, -new))
        rank = np.empty_like(current)
        rank[current] = np.arange(len(current))
        swaps = rank[previous[:-1]] > rank[previous[1:]]
        near = old[previous[:-1]] - old[previous[1:]] <= 2 * atol
        rank_changes += int(np.count_nonzero(previous != current))
        near_swaps += int(np.count_nonzero(swaps & near))
        distant_swaps += int(np.count_nonzero(swaps & ~near))
        top20_order_changes += int(not np.array_equal(previous[:20], current[:20]))
        top20_membership_changes += int(set(previous[:20]) != set(current[:20]))
    return {"passed": max_error <= atol and distant_swaps == 0,
            "score_atol": atol, "max_absolute_cosine_error": max_error,
            "query_rows": expected.shape[0], "gallery_rows": expected.shape[1], "rank_changes": rank_changes,
            "historical_near_tie_adjacent_swaps": near_swaps, "historical_distant_adjacent_swaps": distant_swaps,
            "queries_with_top20_order_changes": top20_order_changes,
            "queries_with_top20_membership_changes": top20_membership_changes}


def normalize_rows(raw, axis, details, epsilon):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != len(axis) or len(axis) != len(details):
        raise ValueError("encoder output rows disagree with the input axis")
    norms = np.linalg.norm(raw, axis=1)
    vectors = np.zeros_like(raw, dtype="<f8", order="C")
    status = []
    for i, (row, info) in enumerate(zip(axis, details, strict=True)):
        keys = ("uid", "cad_index") if "uid" in row else ("text_index", "text_sha256")
        record = {key: row[key] for key in keys} | info
        reason = info.get("reason")
        if reason is None and (not np.isfinite(norms[i]) or norms[i] <= 0):
            reason = "embedding has zero or nonfinite norm"
        if reason is None:
            vectors[i] = raw[i] / (norms[i] + epsilon)
        else:
            record["reason"] = reason
        status.append(record | {"valid": reason is None})
    return vectors, status


def axis_files(out, stem):
    return out / f"{stem}.npy", out / f"{stem}_status.jsonl", out / f"{stem}_meta.json"


def write_axis(out, stem, vectors, status, meta):
    vector_path, status_path, meta_path = axis_files(out, stem)
    if any(p.exists() for p in (vector_path, status_path, meta_path)):
        raise FileExistsError(f"encoder output already exists for {stem}: {out}")
    with tempfile.TemporaryDirectory(dir=out) as temp:
        staged_vectors, staged_status, staged_meta = axis_files(Path(temp), stem)
        np.save(staged_vectors, vectors, allow_pickle=False)
        with staged_status.open("w", encoding="utf-8", newline="\n") as stream:
            for row in status:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        save_json(staged_meta, meta | {
            "dtype": "float64", "shape": list(vectors.shape), "failed": sum(not row["valid"] for row in status),
            "truncated": sum(row.get("truncated", False) for row in status),
            "vectors_sha256": file_sha256(staged_vectors), "status_sha256": file_sha256(staged_status),
        })
        # The metadata is the completion record and is only visible after both payloads.
        for staged, final in zip((staged_vectors, staged_status, staged_meta),
                                 (vector_path, status_path, meta_path), strict=True):
            staged.replace(final)


def read_axis(out, stem, axis_sha, method, weights, nrows):
    vector_path, status_path, meta_path = axis_files(out, stem)
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_bytes())
    if (meta["axis_sha256"] != axis_sha or meta["method"] != method or meta["weights"] != weights
            or meta["shape"] != [nrows, DIMENSIONS[method]]):
        raise ValueError(f"existing artifact has different axes or model weights: {meta_path}")
    check_sha(vector_path, meta["vectors_sha256"])
    check_sha(status_path, meta["status_sha256"])
    status, _ = read_rows(status_path)
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    if len(status) != nrows or list(vectors.shape) != meta["shape"] or vectors.dtype != np.dtype("<f8"):
        raise ValueError(f"existing artifact has an invalid payload: {meta_path}")
    return vectors, status


@contextmanager
def output_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".encoder.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(f"{os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink()


def packages(*names):
    return {name: importlib.metadata.version(name) for name in names}


def torch_features(result, dim):
    if hasattr(result, "pooler_output"):
        result = result.pooler_output
    raw = result.detach().cpu().numpy()
    if raw.ndim != 2 or raw.shape[1] != dim:
        raise ValueError(f"encoder returned shape {raw.shape} for dimension {dim}")
    return raw


def hf_text_encoder(tokenizer, forward, torch, window, dim, *, template="{q}"):
    tokenizer.truncation_side = "right"

    def encode(texts):
        prompts = [template.format(q=text) for text in texts]
        lengths = [len(ids) for ids in tokenizer(prompts, padding=False, truncation=False,
                                                verbose=False)["input_ids"]]
        tokens = tokenizer(prompts, padding=True, truncation=True, max_length=window,
                           return_tensors="pt").to("cpu")
        retained = tokens["attention_mask"].sum(dim=1).tolist()
        if retained != [min(length, window) for length in lengths]:
            raise ValueError("tokenizer did not preserve the registered text window")
        with torch.no_grad():
            raw = torch_features(forward(tokens), dim)
        return raw, lengths

    return encode


def load_clip(a, torch):
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

    if a.clip_weights.name != "model.safetensors":
        raise ValueError("the frozen CLIP weight file must be named model.safetensors")
    check_sha(a.clip_weights, CLIP_WEIGHTS)
    for name, digest in CLIP_PROCESSOR_FILES.items():
        check_sha(a.clip_processor / name, digest)
    config = CLIPConfig.from_pretrained(a.clip_processor, local_files_only=True)
    if config.text_config.max_position_embeddings != 77 or config.projection_dim != DIMENSIONS["clip"]:
        raise ValueError("CLIP config differs from the G5 window or output dimension")
    model = CLIPModel.from_pretrained(a.clip_weights.parent, config=config, use_safetensors=True,
                                      local_files_only=True).eval().to(device="cpu", dtype=torch.float32)
    processor = CLIPImageProcessor.from_pretrained(a.clip_processor, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(a.clip_processor, local_files_only=True)
    encode_text = hf_text_encoder(tokenizer, lambda tokens: model.get_text_features(**tokens),
                                  torch, 77, DIMENSIONS["clip"])

    def encode_image(images):
        with torch.no_grad():
            out = model.get_image_features(**processor(images=images, return_tensors="pt"))
        return torch_features(out, DIMENSIONS["clip"])

    weights = {"model_safetensors_sha256": CLIP_WEIGHTS, "weights_revision": "c237dc49a33fc61debc9276459120b7eac67e7ef",
               "processor_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
               "processor_files": CLIP_PROCESSOR_FILES, "render_spec_sha256": CLIP_RENDER_SPEC}
    return encode_text, encode_image, weights, packages("transformers", "tokenizers", "Pillow")


def load_openshape(a, torch):
    import open_clip

    weight_hash = file_sha256(a.bigg_checkpoint)
    blob_name = a.bigg_checkpoint.resolve().name
    if re.fullmatch(r"[0-9a-f]{64}", blob_name) and blob_name != weight_hash:
        raise ValueError("OpenShape bigG checkpoint does not match its cached blob digest")
    model_name = "ViT-bigG-14"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=str(a.bigg_checkpoint))
    model = model.eval().to("cpu")
    tokenizer = open_clip.get_tokenizer(model_name)
    if int(tokenizer.context_length) != 77:
        raise ValueError("OpenShape bigG tokenizer window differs from G5")

    def encode_text(texts):
        # OpenCLIP's SimpleTokenizer preserves the prefix and replaces the last
        # retained token with EOT. Count the untruncated tokens including SOT/EOT.
        lengths = [len(tokenizer.encode(text)) + 2 for text in texts]
        with torch.no_grad():
            raw = torch_features(model.encode_text(tokenizer(texts).to("cpu")), DIMENSIONS["openshape"])
        return raw, lengths

    weights = {"bigg_checkpoint_sha256": weight_hash, "bigg_model": model_name,
               "pointbert_g5_checkpoint_sha256": "34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9",
               "gallery_preprocessing": "G5 adapted PointBERT with gray RGB, XYZ-to-XZY and deterministic FPS shim"}
    return encode_text, None, weights, packages("open_clip_torch")


def load_mmcad(a, torch):
    from run_mmcad_e1_20261120 import (CHECKPOINT_SHA, INFERENCE_LF_SHA, STATS_SHA, TEXT_PROMPT, TEXT_WINDOW,
                                      TOKENIZER_SHA, load_model)

    _, mm, model, strict = load_model(a)
    stats = mm.load_brep_stats(a.stats)
    dim = DIMENSIONS["mmcad_v2"]

    def forward(tokens):
        return mm.truncate(model.text_encoder(tokens["input_ids"], tokens["attention_mask"]), dim)

    def encode_brep(raw):
        with torch.no_grad():
            out = mm.embed_brep(model, mm.prepare_brep(raw, *stats), dim=dim, device="cpu")
        return torch_features(out, dim)[0]

    encode_text = hf_text_encoder(model.text_encoder.tokenizer, forward, torch, TEXT_WINDOW, dim,
                                  template=TEXT_PROMPT)
    weights = {"checkpoint_sha256": CHECKPOINT_SHA, "normalization_stats_sha256": STATS_SHA,
               "inference_lf_sha256": INFERENCE_LF_SHA, "tokenizer_sha256": TOKENIZER_SHA,
               "text_prompt": TEXT_PROMPT, "strict_load": strict, "fixed_topology": True}
    return encode_text, encode_brep, weights, packages("sentence-transformers", "transformers", "tokenizers")


def mmcad_cache_pins(path):
    check_sha(path, MM_G5_CHECKSUMS)
    files = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        if filename.startswith("embeds_cache/"):
            files[filename.split("/", 1)[1]] = digest
    return files


def read_gallery_asset(row, root, suffix, pins=None):
    filename = row["uid"].replace("/", "__") + suffix
    path = root / filename
    detail = {"source_file": filename}
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return None, detail | {"reason": f"{type(exc).__name__} reading {filename}"}
    digest = hashlib.sha256(payload).hexdigest()
    if pins is not None and digest != pins.get(filename):
        raise ValueError(f"historical MM-CAD gallery cache SHA-256 differs: {filename}")
    return payload, detail | {"source_sha256": digest}


def encode_gallery(a, rows, geometry_forward, *, report=True):
    dim = DIMENSIONS[a.method]
    epsilon = 0.0 if a.method == "clip" else 1e-12
    vectors, details = np.zeros((len(rows), dim), dtype=np.float64), []
    if a.renders:
        from PIL import Image

        for start in range(0, len(rows), a.batch):
            images, positions = [], []
            for i, row in enumerate(rows[start:start + a.batch], start):
                payload, info = read_gallery_asset(row, a.renders, ".png")
                if payload is not None:
                    try:
                        with Image.open(io.BytesIO(payload)) as img:
                            images.append(img.convert("RGB"))
                        positions.append(i)
                    except (OSError, ValueError) as exc:
                        info["reason"] = f"{type(exc).__name__} decoding {info['source_file']}"
                details.append(info)
            if images:
                raw = geometry_forward(images)
                if raw.shape != (len(images), dim):
                    raise ValueError(f"image batch has shape {raw.shape}, expected {(len(images), dim)}")
                vectors[positions] = raw
            if report and (start // a.batch) % 10 == 0:
                log("gallery_progress", method=a.method, done=min(start + a.batch, len(rows)), total=len(rows))
        source = {"kind": "historical G5 single-view RGB render", "path": str(a.renders)}
    else:
        cache = a.vector_cache
        pins = mmcad_cache_pins(a.cache_checksums) if a.cache_checksums else None
        suffix = ("_pc_e1280.npy" if a.method == "openshape" else "_e768.npy") if cache else ".npz"
        root = cache or a.brep_cache
        for i, row in enumerate(rows):
            payload, info = read_gallery_asset(row, root, suffix, pins)
            if payload is not None:
                try:
                    if cache:
                        raw = np.load(io.BytesIO(payload), allow_pickle=False)
                    else:
                        with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
                            raw_features = dict(npz)
                        edges = raw_features["edge_to_faces"]
                        if (int(raw_features["adjacency_error_count"]) != 0
                                or not ((edges[:, 0] >= 0) & (edges[:, 1] >= 0)).any()):
                            raise ValueError("v2 input does not have valid adjacency")
                        raw = geometry_forward(raw_features)
                    if raw.shape != (dim,):
                        raise ValueError(f"embedding has shape {raw.shape}, expected ({dim},)")
                    vectors[i] = raw
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    info["reason"] = f"{type(exc).__name__}: {exc}"
            details.append(info)
            if report and ((i + 1) % 400 == 0 or i + 1 == len(rows)):
                log("gallery_progress", method=a.method, done=i + 1, total=len(rows))
        source = {"kind": "historical G5 UID-keyed vectors" if cache else "historical G5 v2 B-Rep inputs",
                  "path": str(root)}
        if pins:
            source["historical_checksum_manifest_sha256"] = MM_G5_CHECKSUMS
    vectors, status = normalize_rows(vectors, rows, details, epsilon)
    return vectors, status, source


def encode_text_axis(a, rows, text_forward, *, report=True):
    dim = DIMENSIONS[a.method]
    epsilon = 0.0 if a.method == "clip" else 1e-12
    window = 512 if a.method == "mmcad_v2" else 77
    vectors, details = np.empty((len(rows), dim), dtype=np.float64), []
    for start in range(0, len(rows), a.batch):
        texts = [row["text"] for row in rows[start:start + a.batch]]
        values, lengths = text_forward(texts)
        if values.shape != (len(texts), dim) or len(lengths) != len(texts):
            raise ValueError("text tokenizer/encoder rows disagreed with the input batch")
        vectors[start:start + len(texts)] = values
        details.extend({"tokens_before_truncation": count, "tokens_encoded": min(count, window),
                        "truncated": count > window} for count in lengths)
        if report and ((start // a.batch) % 10 == 0 or start + len(texts) == len(rows)):
            log("text_progress", method=a.method, done=start + len(texts), total=len(rows))
    return normalize_rows(vectors, rows, details, epsilon)


def replay_report(scores, gallery, status, text, text_status, *, indices):
    valid = np.array([row["valid"] for row in status])
    if not all(row["valid"] for row in text_status) or np.count_nonzero(valid) < min(16, len(valid)):
        raise ValueError("insufficient valid embeddings for the fixed historical L0 replay")
    observed = text @ gallery[valid].T
    return compare_replay(scores[:, valid], observed, np.asarray(indices)[valid], atol=REPLAY_TOLERANCE) | {
        "failed_gallery_rows": int(np.count_nonzero(~valid)),
        "truncated_replay_texts": sum(row["truncated"] for row in text_status),
    }


def write_replay_receipt(path, data):
    if path.exists():
        previous = json.loads(path.read_bytes())
        if previous == data:
            return
        # Verified follow-up conditions can reuse the same gallery. Keep its original
        # passing receipt and log the current replay, including roundoff differences.
        binding = ("method", "replay_reference_sha256", "gallery_sha256")
        if previous["passed"] and data["passed"] and all(previous.get(k) == data.get(k) for k in binding):
            return
        raise ValueError(f"current L0 replay differs from existing receipt: {path}")
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        save_json(staged, data)
        staged.replace(path)


def encode(a):
    started = time.monotonic()
    gallery, gallery_sha = load_gallery(a.frozen / "gallery.jsonl")
    l0, l0_sha = load_texts(a.frozen / "texts_l0.jsonl")
    frozen = json.loads((a.frozen / "freeze.json").read_bytes())
    axes = {"gallery.jsonl": gallery_sha, "texts_l0.jsonl": l0_sha}
    conditions = {}
    for condition in a.condition:
        name = f"texts_{condition}.jsonl"
        rows, axes[name] = load_texts(a.frozen / name)
        if len(rows) != frozen["conditions"][condition]["unique_texts"]:
            raise ValueError(f"text axis count disagrees with the frozen manifest: {condition}")
        conditions[condition] = (rows, axes[name])
    if (len(gallery) != frozen["gallery_count"] or len(l0) != frozen["conditions"]["l0"]["unique_texts"]
            or any(frozen["files"].get(name) != digest for name, digest in axes.items())):
        raise ValueError("encoder input assets do not match freeze.json")
    replay_rows, history, smoke_positions, replay_sha = load_replay(
        a.replay, a.method, gallery_sha, l0, l0_sha, len(gallery))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.set_num_threads(a.threads)
    loaders = {"clip": load_clip, "openshape": load_openshape, "mmcad_v2": load_mmcad}
    with output_lock(a.output):
        text_forward, geometry_forward, weights, versions = loaders[a.method](a, torch)
        provenance = {"method": a.method, "weights": weights, "device": "cpu", "forward_dtype": "float32",
                      "threads": a.threads, "batch": a.batch, "python": platform.python_version(),
                      "versions": versions | {"torch": str(torch.__version__), "numpy": np.__version__},
                      "encoder_sha256": file_sha256(Path(__file__))}
        log("model_ready", method=a.method, seconds=round(time.monotonic() - started, 2),
            weights=weights, versions=provenance["versions"])
        replay_text, replay_text_status = encode_text_axis(a, replay_rows, text_forward, report=False)
        cached = read_axis(a.output, "gallery", gallery_sha, a.method, weights, len(gallery))
        if cached is None:
            smoke_vectors, smoke_status, _ = encode_gallery(a, [gallery[i] for i in smoke_positions],
                                                             geometry_forward, report=False)
        else:
            smoke_vectors, smoke_status = cached[0][smoke_positions], [cached[1][i] for i in smoke_positions]
        smoke = replay_report(history[:, smoke_positions], smoke_vectors, smoke_status,
                              replay_text, replay_text_status,
                              indices=[gallery[i]["cad_index"] for i in smoke_positions])
        smoke = smoke | {"method": a.method, "replay_reference_sha256": replay_sha}
        log("l0_smoke_replay", **smoke)
        if a.smoke_only or not smoke["passed"]:
            write_replay_receipt(a.output / "l0_smoke_replay.json", smoke)
        if not smoke["passed"]:
            raise ValueError("historical L0 cosine smoke replay failed")
        if a.smoke_only:
            return

        if cached is None:
            vectors, status, source = encode_gallery(a, gallery, geometry_forward)
        else:
            vectors, status = cached
        if any((row["cad_index"], row["uid"]) != (record["cad_index"], record["uid"])
               for row, record in zip(gallery, status, strict=True)):
            raise ValueError("candidate status does not follow the frozen cad_index and UID axes")
        full = replay_report(history, vectors, status, replay_text, replay_text_status,
                             indices=[row["cad_index"] for row in gallery])
        full = full | {"method": a.method, "replay_reference_sha256": replay_sha, "gallery_sha256": gallery_sha}
        log("l0_full_gallery_replay", **full)
        case = "l0_replay" if full["passed"] else "l0_failed_replay"
        write_replay_receipt(a.output / f"{case}.json", full)
        if not full["passed"]:
            raise ValueError("historical L0 cosine replay failed on the full ordered gallery")
        replay_receipt_sha = file_sha256(a.output / "l0_replay.json")
        provenance |= {"l0_replay_sha256": replay_receipt_sha}
        if cached is None:
            write_axis(a.output, "gallery", vectors, status, provenance | {
                "axis_sha256": gallery_sha, "gallery_source": source,
            })
        del vectors, status, cached
        for condition, (rows, digest) in conditions.items():
            stem = f"texts_{condition}"
            if read_axis(a.output, stem, digest, a.method, weights, len(rows)) is not None:
                log("condition_reused", method=a.method, condition=condition, axis_sha256=digest)
                continue
            log("condition_started", method=a.method, condition=condition, texts=len(rows))
            vectors, status = encode_text_axis(a, rows, text_forward)
            write_axis(a.output, stem, vectors, status, provenance | {
                "axis_sha256": digest, "condition": condition, "context_window": 512 if a.method == "mmcad_v2" else 77,
                "truncation": "prefix, retaining required special tokens",
            })
            log("condition_complete", method=a.method, condition=condition, texts=len(rows),
                failed=sum(not row["valid"] for row in status), truncated=sum(row["truncated"] for row in status))
            del vectors, status
        log("encoding_complete", method=a.method, seconds=round(time.monotonic() - started, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-replay", help="extract pinned historical L0 score rows without model forward")
    worker = sub.add_parser("encode", help="run CPU encoders and validate against the L0 replay")
    for command in (prep, worker):
        command.add_argument("--method", choices=tuple(DIMENSIONS), required=True)
        command.add_argument("--frozen", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for flag in ("historical-gallery", "historical-texts", "historical-matrix", "historical-sidecar"):
        prep.add_argument("--" + flag, type=Path, required=True)
    for flag in ("replay", "renders", "clip-weights", "clip-processor", "vector-cache", "bigg-checkpoint",
                 "cache-checksums", "brep-cache", "upstream", "checkpoint", "stats", "text-model"):
        worker.add_argument("--" + flag, type=Path, required=flag == "replay")
    worker.add_argument("--condition", choices=CONDITIONS, action="append",
                        help="repeat to select conditions in order (default: ct_deepcad, l2)")
    worker.add_argument("--threads", type=int)
    worker.add_argument("--batch", type=int)
    worker.add_argument("--smoke-only", action="store_true")
    a = parser.parse_args()
    if a.command == "prepare-replay":
        prepare_replay(a.frozen / "gallery.jsonl", a.frozen / "texts_l0.jsonl", a.historical_gallery,
                       a.historical_texts, a.historical_matrix, a.historical_sidecar, a.output,
                       expected_sidecar_sha256=HISTORICAL_SIDECARS[a.method])
        return
    required = {"clip": ("renders", "clip_weights", "clip_processor"),
                "openshape": ("vector_cache", "bigg_checkpoint"),
                "mmcad_v2": ("upstream", "checkpoint", "stats", "text_model")}[a.method]
    if a.method == "mmcad_v2":
        if bool(a.vector_cache) == bool(a.brep_cache):
            parser.error("MM-CAD requires exactly one of --vector-cache or --brep-cache")
        if a.vector_cache:
            required += ("cache_checksums",)
    if any(getattr(a, flag) is None for flag in required):
        parser.error("required method paths: " + ", ".join("--" + flag.replace("_", "-") for flag in required))
    path_flags = {"renders", "clip_weights", "clip_processor", "vector_cache", "bigg_checkpoint",
                  "cache_checksums", "brep_cache", "upstream", "checkpoint", "stats", "text_model"}
    if a.method == "mmcad_v2" and a.brep_cache:
        required += ("brep_cache",)
    if any(getattr(a, flag) is not None for flag in path_flags.difference(required)):
        parser.error("only use path flags for the selected method and gallery source")
    a.threads = DEFAULT_THREADS[a.method] if a.threads is None else a.threads
    a.batch = DEFAULT_BATCH[a.method] if a.batch is None else a.batch
    a.condition = [] if a.smoke_only else a.condition or list(CONDITIONS[:2])
    if a.threads < 1 or a.batch < 1:
        parser.error("--threads and --batch must be positive")
    if len(set(a.condition)) != len(a.condition):
        parser.error("each condition may be requested once")
    encode(a)


if __name__ == "__main__":
    main()
