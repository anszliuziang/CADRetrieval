#!/usr/bin/env python3
"""Build raw SpectraBRep descriptors for the official train and validation splits.

Each fold contains rows.jsonl for the complete attempted UID axis, vectors.npy
for successful rows indexed by vector_index, and two JSONL UID lists for the
native and native-plus-CadTranslator cohorts. A pilot takes the first N official
UIDs per fold and records that limit in the output manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType
from zipfile import BadZipFile, ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_e1_inventory import EXPECTED_SOURCE_SPLIT, check_split, official_member
from prepare_retrieval_benchmark_fill import ASSETS, EMPTY_ANNOTATIONS, LEVEL_FIELDS, sha256, verify_sha256, write_json
from prepare_retrieval_benchmark_source_supervision import ASSET_NAMES, FOLDS


AUTHOR_COMMIT = "5254a331d69f6eab08a66369c184f2effb117623"
AUTHOR_GIT_TREE = "fa7cf0a4c13f63bba05374fe5652e8b40fd6c79e"
# SHA-256 of path + NUL + file SHA-256 + LF for each of the 27 verified Git blobs, sorted by path.
AUTHOR_SOURCE_SHA256 = "fd82c2552373d179cf981cdcda8e3dd8aa81892088ebdd82654935b08187e9a5"
AUTHOR_MODULES = ("config", "geometry")
FROZEN_SUPERVISION_SHA256 = {
    "train.jsonl": "97db5ef6ca0a41022459cc1c8e7f471725f668a478f6dc168eb2614930a13377",
    "validation.jsonl": "500151a5bec4444f8d8adf18d7e9ad3e3b03f2d436579a59f23855d4cbffc9fd",
}
DESCRIPTOR_DIM = 280
MAX_WORKERS = 8

_ARCHIVE = None
_DESCRIBE_SEQUENCE = None
_DESCRIPTOR_VECTOR = None


def verify_author_source(root: Path) -> tuple[bytes, bytes]:
    if not root.is_dir():
        raise ValueError(f"SpectraBRep author source directory is missing: {root}")
    tree = hashlib.sha256()
    contents = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"SpectraBRep author source contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_size > 25_000 or len(contents) == 27:
            raise ValueError(f"Unexpected SpectraBRep author source entry: {path}")
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        contents[name] = data
        tree.update(f"{name}\0{hashlib.sha256(data).hexdigest()}\n".encode("utf-8"))
    if tree.hexdigest() != AUTHOR_SOURCE_SHA256:
        raise ValueError(f"SpectraBRep author source differs from {AUTHOR_COMMIT}: {root}")
    return tuple(contents[f"src/cadret/{name}.py"] for name in AUTHOR_MODULES)


def load_registration(path: Path, fold: str, allowed: set[str], stats: dict) -> dict:
    registered = {}
    actual_hash = hashlib.sha256()
    native_by_level = Counter(dict.fromkeys(LEVEL_FIELDS, 0))
    matched_by_level = native_by_level.copy()
    matched = complete = 0
    with path.open("rb") as source:
        for raw in source:
            actual_hash.update(raw)
            row = json.loads(raw)
            if set(row) != {"uid", "member", "native_texts", "ct_deepcad_text"}:
                raise ValueError(f"Unexpected source supervision fields in {fold}")
            uid = row["uid"]
            if uid not in allowed or uid in registered or row["member"] != official_member(uid):
                raise ValueError(f"Source supervision UID or ZIP member is invalid for {fold}: {uid}")
            native_texts = row["native_texts"]
            short = row["ct_deepcad_text"]
            if (not isinstance(native_texts, dict) or not native_texts
                    or native_texts.keys() - LEVEL_FIELDS.keys()
                    or any(not isinstance(text, str) or text != text.strip()
                           or text.casefold() in EMPTY_ANNOTATIONS for text in native_texts.values())
                    or (short is not None and (not isinstance(short, str) or not short or short != short.strip()))):
                raise ValueError(f"Source supervision text fields are invalid for {fold}: {uid}")
            levels = tuple(level for level in LEVEL_FIELDS if level in native_texts)
            has_short = short is not None
            registered[uid] = (levels, has_short)
            native_by_level.update(levels)
            complete += len(levels) == len(LEVEL_FIELDS)
            if has_short:
                matched += 1
                matched_by_level.update(levels)
    if actual_hash.hexdigest() != FROZEN_SUPERVISION_SHA256[path.name]:
        raise ValueError(f"Source supervision bytes differ from the frozen file: {path}")
    audited = {
        "official_uids": len(allowed),
        "native_uids": len(registered),
        "native_four_level_uids": complete,
        "common_native_and_ct_uids": matched,
        "native_pairs_per_level": dict(native_by_level),
        "common_native_pairs_per_level": dict(matched_by_level),
    }
    if any(stats.get(key) != value for key, value in audited.items()):
        raise ValueError(f"Source supervision counts disagree with the frozen rows for {fold}")
    if not matched <= stats["source_ct_caption_uids"] <= len(allowed):
        raise ValueError(f"Source supervision short-caption count is invalid for {fold}")
    return registered


def initialize_worker(program_zip: Path, author_modules: tuple[bytes, bytes]) -> None:
    global _ARCHIVE, _DESCRIBE_SEQUENCE, _DESCRIPTOR_VECTOR
    package_name = "_spectrabrep_5254a331"
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    # Execute exactly the authenticated bytes, without import-path or bytecode-cache lookup.
    for name, data in zip(AUTHOR_MODULES, author_modules, strict=True):
        module = ModuleType(f"{package_name}.{name}")
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        exec(compile(data, f"SpectraBRep/{AUTHOR_COMMIT}/src/cadret/{name}.py", "exec"), module.__dict__)
    geometry = sys.modules[f"{package_name}.geometry"]
    _DESCRIBE_SEQUENCE = geometry.describe_sequence
    _DESCRIPTOR_VECTOR = geometry.descriptor_vector
    _ARCHIVE = ZipFile(program_zip, "r")


def describe_program(uid: str):
    import torch

    try:
        data = _ARCHIVE.read(official_member(uid))
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except (OSError, BadZipFile, KeyError, EOFError, ValueError, RuntimeError, pickle.UnpicklingError) as exc:
        return "unreadable_program", f"{type(exc).__name__}: {exc}", None
    try:
        cad = payload["vec"]["cad_vec"].numpy()
        index = payload["vec"]["index_vec"].numpy()
    except (KeyError, AttributeError, TypeError, RuntimeError) as exc:
        return "invalid_program", f"{type(exc).__name__}: {exc}", None
    if cad.shape != (272, 2) or index.shape != (272,) or cad.dtype.kind not in "iu" or index.dtype.kind not in "iu":
        return "invalid_program", "Expected integer cad_vec[272,2] and index_vec[272]", None
    try:
        vector = _DESCRIPTOR_VECTOR(_DESCRIBE_SEQUENCE(cad, index))
    except (TypeError, ValueError, ArithmeticError, IndexError, KeyError, RuntimeError) as exc:
        return "descriptor_error", f"{type(exc).__name__}: {exc}", None
    if not isinstance(vector, np.ndarray) or vector.shape != (DESCRIPTOR_DIM,) or vector.dtype != np.float32:
        return "invalid_descriptor", f"Expected float32[{DESCRIPTOR_DIM}]", None
    if not np.isfinite(vector).all():
        return "nonfinite_descriptor", None, None
    if not np.any(vector):
        return "empty_geometry", None, None
    return "ok", None, vector


def ordered_descriptors(uids: list[str], member_counts: Counter, pool, workers: int):
    # Executor.map eagerly submits its input on older Python releases.
    window = workers * 8
    for start in range(0, len(uids), window):
        block = uids[start:start + window]
        present = [uid for uid in block if member_counts[official_member(uid)] == 1]
        accepted = iter(map(describe_program, present) if pool is None else pool.map(describe_program, present))
        for uid in block:
            count = member_counts[official_member(uid)]
            if count == 1:
                yield next(accepted)
            elif count == 0:
                yield "missing_program", None, None
            else:
                yield "ambiguous_program", f"{count} matching ZIP members", None


def write_row(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def build_fold(folder: Path, all_uids: list[str], uids: list[str], registered: dict,
               member_counts: Counter, pool, workers: int) -> dict:
    folder.mkdir()
    counts = Counter()
    native_by_level = Counter(dict.fromkeys(LEVEL_FIELDS, 0))
    matched_by_level = native_by_level.copy()
    native_count = matched_count = 0
    with ExitStack() as stack:
        rows, native, matched = (
            stack.enter_context((folder / name).open("x", encoding="utf-8", newline="\n"))
            for name in ("rows.jsonl", "native_eligible.jsonl", "matched_eligible.jsonl")
        )
        raw = stack.enter_context(tempfile.TemporaryFile(dir=folder))
        results = ordered_descriptors(uids, member_counts, pool, workers)
        for official_index, (uid, (status, detail, vector)) in enumerate(zip(uids, results, strict=True)):
            levels, has_short = registered.get(uid, ((), False))
            vector_index = None
            if status == "ok":
                vector_index = counts["ok"]
                raw.write(vector.tobytes())
                if levels:
                    write_row(native, uid)
                    native_count += 1
                    native_by_level.update(levels)
                    if has_short:
                        write_row(matched, uid)
                        matched_count += 1
                        matched_by_level.update(levels)
            counts[status] += 1
            write_row(rows, {
                "uid": uid, "member": official_member(uid), "official_index": official_index,
                "status": status, "detail": detail[:500] if detail else None, "vector_index": vector_index,
                "native_levels": levels, "has_ct_short_caption": has_short,
            })
        if counts["ok"]:
            raw.flush()
            vectors = np.memmap(raw, mode="r", dtype=np.float32, shape=(counts["ok"], DESCRIPTOR_DIM))
            np.save(folder / "vectors.npy", vectors, allow_pickle=False)
            del vectors
        else:
            np.save(folder / "vectors.npy", np.empty((0, DESCRIPTOR_DIM), dtype=np.float32), allow_pickle=False)
    return {
        "official_uids": len(all_uids), "attempted_uids": len(uids), "vectors": counts["ok"],
        "status_counts": dict(counts), "native_eligible_uids": native_count,
        "matched_eligible_uids": matched_count, "native_pairs_per_level": dict(native_by_level),
        "matched_native_pairs_per_level": dict(matched_by_level),
    }


def build_cache(upstream: Path, program_zip: Path, split_path: Path, supervision: Path, output: Path,
                *, workers: int = 4, pilot_per_fold: int | None = None) -> dict:
    if not 1 <= workers <= MAX_WORKERS or (pilot_per_fold is not None and pilot_per_fold < 1):
        raise ValueError(f"Expected 1–{MAX_WORKERS} workers and a positive pilot limit")
    if output.exists():
        raise FileExistsError(f"Descriptor cache already exists: {output}")
    author_modules = verify_author_source(upstream)
    dataset_pins = {name: ASSETS[name][2] for name in ASSET_NAMES}
    verify_sha256(program_zip, dataset_pins["text2cad_program_zip"])
    verify_sha256(split_path, dataset_pins["text2cad_split_json"])
    splits = json.loads(split_path.read_bytes())
    official_uids = check_split(splits, EXPECTED_SOURCE_SPLIT)
    supervision_manifest_path = supervision / "manifest.json"
    manifest_bytes = supervision_manifest_path.read_bytes()
    registration = json.loads(manifest_bytes)
    if (registration.get("schema_version") != 1 or registration.get("files") != FROZEN_SUPERVISION_SHA256
            or registration.get("official_test_uids_excluded") != len(official_uids["test"])
            or set(registration["splits"]) != set(FOLDS)
            or {name: values["sha256"] for name, values in registration["sources"].items()} != dataset_pins):
        raise ValueError(f"Source supervision manifest differs from the frozen inputs: {supervision_manifest_path}")
    registered = {
        fold: load_registration(supervision / f"{fold}.jsonl", fold, official_uids[fold], registration["splits"][fold])
        for fold in FOLDS
    }
    selected = {fold: splits[fold][:pilot_per_fold] for fold in FOLDS}
    members = {official_member(uid) for uids in selected.values() for uid in uids}
    with ZipFile(program_zip, "r") as archive:
        member_counts = Counter(item.filename for item in archive.infolist() if item.filename in members)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        with ExitStack() as stack:
            if workers == 1:
                initialize_worker(program_zip, author_modules)
                stack.callback(_ARCHIVE.close)
                pool = None
            else:
                pool = stack.enter_context(ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker, initargs=(program_zip, author_modules),
                ))
            folds = {
                fold: build_fold(staged / fold, splits[fold], selected[fold], registered[fold], member_counts, pool, workers)
                for fold in FOLDS
            }
        result = {
            "schema_version": 1, "mode": "full" if pilot_per_fold is None else "pilot",
            "pilot_per_fold": pilot_per_fold, "workers": workers, "descriptor_dim": DESCRIPTOR_DIM,
            "dtype": "float32", "order": "official split JSON order", "training_input": "raw descriptors",
            "author": {"commit": AUTHOR_COMMIT, "git_tree": AUTHOR_GIT_TREE, "source_sha256": AUTHOR_SOURCE_SHA256},
            "sources": dataset_pins,
            "source_supervision_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_supervision_files": FROZEN_SUPERVISION_SHA256,
            "runtime": {"python": sys.version.split()[0], **{
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")
            }},
            "folds": folds,
            "files": {path.relative_to(staged).as_posix(): sha256(path)
                      for path in sorted(staged.glob("*/*"))},
        }
        write_json(staged / "manifest.json", result)
        staged.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="Verified author source snapshot")
    parser.add_argument("--official-zip", required=True, type=Path, help="Official Text2CAD cad_seq.zip")
    parser.add_argument("--official-split", required=True, type=Path, help="Official train_test_val.json")
    parser.add_argument("--source-supervision", required=True, type=Path, help="Frozen train/validation registration")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the descriptor cache")
    parser.add_argument("--workers", type=int, default=4, help=f"CPU workers, at most {MAX_WORKERS}")
    parser.add_argument("--pilot-per-fold", type=int, help="Limit both splits to their first N official UIDs")
    args = parser.parse_args()
    result = build_cache(args.upstream.resolve(), args.official_zip.resolve(), args.official_split.resolve(),
                         args.source_supervision.resolve(), args.output.resolve(),
                         workers=args.workers, pilot_per_fold=args.pilot_per_fold)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
