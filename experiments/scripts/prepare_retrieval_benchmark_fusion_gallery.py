"""Prepare the CadTranslator–Fusion gallery using the released HDF5 geometry.

Extract on the h5py runtime, then index source tensors and replay on local CPU.
Target pilots and partial source indexes produce evidence without a final cohort.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from time import monotonic
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

from prepare_retrieval_benchmark_fusion import (
    CADTRANSLATOR_ZIP_SHA, DEEP_CAD_SHA, EXTRUDE, decode_h5_vector, file_digest, make_minimal_program,
    read_h5, read_jsonl, replay_h5, text2cad_frame, vector_digest,
)
import numpy as np

from idea01 import text2cad_e1_geometry as geometry
from idea01.text2cad_e1_inventory import EXPECTED_SOURCE_SPLIT, atomic_json, check_split, official_member
from prepare_retrieval_benchmark_fill import (
    ASSETS, BACKGROUND_ORDER_SHA256, freeze_condition, load_json, parse_cadtranslator_texts,
    read_csv, text_digest, verify_sha256,
)
from prepare_text2cad_e1_geometry import save_brep


DOMAIN = "Fusion360"
CONDITION = "ct_fusion"
INPUTS = "fusion_inputs.jsonl"
INPUT_PIN = "fusion_input_pin.json"
SOURCE_PIN = "validation_index_pin.json"
VALIDATION_INDEX = "validation_vec_hashes.jsonl"
REPLAY_PIN = "fusion_replay_pin.json"
E1_INDEX_DIGESTS = {
    "train": "f7f86162d22e9caa0b158828252f4c16d38563c0209c95386d91fa5d049d530d",
    "test": "8dc4418fac7971fc04e20b4af47c429cac220ac72e36499c2aa4096db8f68e71",
}
FUSION_UID = re.compile(r"[0-9]{4}/[0-9]{8}")
REJECTED_OUTCOMES = {"syntax", "h5_replay", "float_conversion", "float_geometry",
                     "text2cad_encoding", "quantized_replay", "quantized_geometry", "source_duplicate"}


@contextmanager
def single_writer(directory: Path):
    """An OS lock is released even if the CPU worker is interrupted."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".fusion_preparation.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_same_or_write(path: Path, data: dict):
    if path.exists():
        if load_json(path) != data:
            raise ValueError(f"Existing run pins differ at {path}; use a new output directory")
    else:
        atomic_json(path, data)


def digest_of(asset_name):
    return ASSETS[asset_name][2]


def extract_fusion(archive_path, inventory_path, output, pilot_uids=()):
    verify_sha256(archive_path, CADTRANSLATOR_ZIP_SHA)
    verify_sha256(inventory_path, digest_of("inventory"))
    chosen = [row for row in read_csv(inventory_path)
              if row["domain"] == DOMAIN and row["recommended_candidate_before_conversion"] == "1"]
    by_uid = {row["cad_uid"]: row for row in chosen}
    if len(chosen) != 6_757 or len(by_uid) != len(chosen):
        raise ValueError("Unexpected preregistered Fusion UID coverage")
    if len(set(pilot_uids)) != len(pilot_uids) or set(pilot_uids) - by_uid.keys():
        raise ValueError("Pilot UIDs must be distinct members of the preregistered gallery")
    selection = sorted(pilot_uids or by_uid)
    pins = {"scope": "pilot" if pilot_uids else "full", "selected_uids": selection,
            "archive_sha256": CADTRANSLATOR_ZIP_SHA, "inventory_sha256": digest_of("inventory")}
    with single_writer(output):
        if (output / INPUT_PIN).exists():
            previous = load_json(output / INPUT_PIN)
            if any(previous[k] != v for k, v in pins.items()):
                raise ValueError("Fusion extraction pins changed")
            verify_sha256(output / INPUTS, previous["inputs_sha256"])
            return previous
        with ZipFile(archive_path) as archive:
            with archive.open("dataset/fusion_360_txt/text.txt") as raw:
                captions = parse_cadtranslator_texts(io.TextIOWrapper(raw, encoding="utf-8-sig"), set(by_uid))
            if len(captions) != 6_502:
                raise ValueError("Unexpected preregistered Fusion caption coverage")
            for uid, source in by_uid.items():
                caption = captions.get(uid)
                if ((caption is not None) != (source["recommended_query_before_conversion"] == "1")
                        or (text_digest(caption) if caption else "") != source["cadtranslator_query_stripped_sha256"]):
                    raise ValueError(f"Fusion caption differs from the inventory for {uid}")
            members = Counter(info.filename for info in archive.infolist()
                              if info.filename.startswith("dataset/fusion_360_txt/"))

            def extract_rows():
                for uid in selection:
                    if FUSION_UID.fullmatch(uid) is None:
                        raise ValueError(f"Invalid Fusion UID {uid!r}")
                    source = by_uid[uid]
                    member = source["candidate_program_member"]
                    if member != f"dataset/fusion_360_txt/cad_vec/{uid}.h5" or members[member] != 1:
                        raise ValueError(f"Unexpected HDF5 archive member for {uid}")
                    vec = read_h5(archive, member)
                    if vec.dtype.kind != "i" or not np.array_equal(vec, vec.astype("<i4")):
                        raise ValueError(f"Lossy HDF5 integer conversion for {uid}")
                    digest = vector_digest(vec)
                    if digest != source["h5_17int32_program_sha256"]:
                        raise ValueError(f"HDF5 vector differs from the inventory for {uid}")
                    yield {"domain": DOMAIN, "uid": uid, "h5_member": member, "h5_sha256": digest,
                           "caption": captions.get(uid), "vec": vec.tolist()}

            atomic_json(output / INPUTS, extract_rows(), lines=True)
        pins["inputs_sha256"] = file_digest(output / INPUTS)
        pins["n_queries"] = sum(uid in captions for uid in selection)
        atomic_json(output / INPUT_PIN, pins)
        return pins


def official_splits(path):
    verify_sha256(path, digest_of("text2cad_split_json"))
    return check_split(load_json(path), EXPECTED_SOURCE_SPLIT)


def load_source_tensors(archive, member):
    import torch

    source = torch.load(io.BytesIO(archive.read(member)), weights_only=True, map_location="cpu")
    tensors = {
        "cad_vec": source["vec"]["cad_vec"].numpy(),
        "flag_vec": source["vec"]["flag_vec"].numpy(),
        "index_vec": source["vec"]["index_vec"].numpy(),
        "key_padding_mask": source["mask_cad_dict"]["key_padding_mask"].numpy(),
    }
    for key in ("cad_vec", "flag_vec", "index_vec"):
        if tensors[key].dtype.kind not in "iu":
            raise ValueError(f"Noninteger official {key} in {member}")
    if tensors["key_padding_mask"].dtype.kind != "b":
        raise ValueError(f"Nonboolean official key padding mask in {member}")
    return tensors


def native_digest(cad):
    if cad.shape != (272, 2) or cad.dtype.kind not in "iu" or not np.array_equal(cad, cad.astype("<i4")):
        raise ValueError("Invalid official CAD vector for exact deduplication")
    return vector_digest(cad)


def prepare_validation_index(archive_path, split_path, output, pilot_uids=()):
    verify_sha256(archive_path, digest_of("text2cad_program_zip"))
    validation = official_splits(split_path)["validation"]
    if len(set(pilot_uids)) != len(pilot_uids) or set(pilot_uids) - validation:
        raise ValueError("Validation pilot UIDs must be distinct official validation UIDs")
    selected = sorted(pilot_uids or validation)
    pins = {
        "validation_uids": selected,
        "official_archive_sha256": digest_of("text2cad_program_zip"),
        "official_split_sha256": digest_of("text2cad_split_json"),
    }
    with single_writer(output):
        if (output / SOURCE_PIN).exists():
            record = load_json(output / SOURCE_PIN)
            if any(record[k] != v for k, v in pins.items()):
                raise ValueError("Validation index scope or source changed")
            verify_sha256(output / VALIDATION_INDEX, record["index_sha256"])
            return record
        names = {official_member(uid): uid for uid in selected}
        with ZipFile(archive_path) as archive:
            observed = Counter(info.filename for info in archive.infolist() if info.filename in names)
            if len(observed) != len(names) or any(n != 1 for n in observed.values()):
                raise ValueError("An official validation ZIP member is missing or duplicated")

            def records():
                for member, uid in names.items():
                    vector = load_source_tensors(archive, member)["cad_vec"]
                    yield {"uid": uid, "sha256": native_digest(vector)}

            atomic_json(output / VALIDATION_INDEX, records(), lines=True)
        pins["index_sha256"] = file_digest(output / VALIDATION_INDEX)
        atomic_json(output / SOURCE_PIN, pins)
        return pins


def source_catalog(e1_prep, validation_dir, split_path):
    splits = official_splits(split_path)
    validation = load_json(validation_dir / SOURCE_PIN)
    if (validation["official_archive_sha256"] != digest_of("text2cad_program_zip")
            or validation["official_split_sha256"] != digest_of("text2cad_split_json")):
        raise ValueError("The validation index lost its official archive or split binding")
    expected_validation = set(validation["validation_uids"])
    if (len(expected_validation) != len(validation["validation_uids"])
            or not expected_validation or not expected_validation <= splits["validation"]):
        raise ValueError("Invalid validation-index coverage")
    sources = {split: (e1_prep / f"{split}_vec_hashes.jsonl", digest, splits[split])
               for split, digest in E1_INDEX_DIGESTS.items()}
    sources["validation"] = validation_dir / VALIDATION_INDEX, validation["index_sha256"], expected_validation
    catalog = defaultdict(list)
    coverage = {}
    for split, (path, digest, expected) in sources.items():
        verify_sha256(path, digest)
        rows = read_jsonl(path)
        if len(rows) != len(expected) or {row["uid"] for row in rows} != expected:
            raise ValueError(f"Incomplete source fingerprint coverage for {split}")
        for row in rows:
            if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
                raise ValueError(f"Invalid source fingerprint for {row['uid']}")
            catalog[row["sha256"]].append((split, row["uid"]))
        coverage[split] = {"count": len(rows), "sha256": digest}
    return catalog, coverage


def source_duplicate_matches(arrays, catalog, load_uid):
    """Use cad_vec only for lookup and compare all four tensor values for each hit."""
    digest = vector_digest(arrays["cad_vec"])
    matches = []
    for split, uid in sorted(catalog.get(digest, ())):
        source = load_uid(uid)
        if vector_digest(source["cad_vec"]) != digest:
            raise ValueError(f"Official CAD vector differs from its audited index for {uid}")
        if all(np.array_equal(arrays[key], source[key]) for key in geometry.ARRAY_KEYS):
            matches.append({"source_split": split, "source_uid": uid})
    return matches


def object_directory(uid):
    if FUSION_UID.fullmatch(uid) is None:
        raise ValueError(f"Invalid Fusion UID {uid!r}")
    return Path("objects") / uid.replace("/", "_")


def artifact(root, member):
    path = root / member
    return {"member": Path(member).as_posix(), "sha256": file_digest(path)}


def verify_object(output, record, source, replay_sha):
    fields = ("domain", "uid", "caption", "h5_member", "h5_sha256")
    if any(record[key] != source[key] for key in fields) or record["replay_pin_sha256"] != replay_sha:
        raise ValueError(f"Stale object evidence for {source['uid']}")
    outcome = record["outcome"]
    if outcome == "accepted":
        required = {"arrays": "program.npz", "reference_brep": "h5_reference.brep",
                    "converted_brep": "text2cad_replay.brep"}
        if set(record["artifacts"]) != set(required):
            raise ValueError(f"Incomplete object artifacts for {source['uid']}")
        base = object_directory(source["uid"])
        for name, filename in required.items():
            result = record["artifacts"][name]
            expected = base / filename
            if result["member"] != expected.as_posix():
                raise ValueError(f"Wrong member for {source['uid']}: {name}")
            verify_sha256(output / expected, result["sha256"])
    elif outcome not in REJECTED_OUTCOMES:
        raise ValueError(f"Invalid terminal object outcome for {source['uid']}")


def convert_object(row, output, replay_sha, catalog, load_uid):
    uid = row["uid"]
    vector = np.asarray(row["vec"], dtype=np.int64)
    if vector_digest(vector) != row["h5_sha256"]:
        raise ValueError(f"Corrupt extracted HDF5 vector for {uid}")
    result = {key: row[key] for key in ("domain", "uid", "caption", "h5_member", "h5_sha256")}
    result["replay_pin_sha256"] = replay_sha
    result["n_h5_rows"] = len(vector)
    result["extent_modes"] = vector[vector[:, 0] == EXTRUDE, 16].tolist()
    result["operations"] = vector[vector[:, 0] == EXTRUDE, 15].tolist()
    stage = "syntax"
    try:
        features = decode_h5_vector(vector)
        stage = "h5_replay"
        reference, low, scale = text2cad_frame(replay_h5(features))
        result["h5_reference_frame"] = {"bbox_min": low.tolist(), "scale": scale}
        stage = "float_conversion"
        program = make_minimal_program(features, low, scale)
        floating = geometry.minimal_sequence(program)
        float_shape = geometry.strict_solid(floating)
        probes = geometry.cut_probes(floating)
        stage = "float_geometry"
        result["float_to_h5"] = geometry.compare_shapes(float_shape, reference, uid, probes)
        if not result["float_to_h5"]["pass"]:
            return {**result, "outcome": stage, "reason": "E1 geometry limits"}
        stage = "text2cad_encoding"
        arrays = geometry.encode_minimal(program)
        diagnostics = geometry.validate_arrays(arrays)
        result["candidate_fields"] = diagnostics.n_valid_total
        result["n_real_tokens"] = diagnostics.n_real_tokens
        result.update(geometry.fingerprints(arrays))
        stage = "quantized_replay"
        converted = geometry.strict_solid(geometry.decode(arrays["cad_vec"]))
        stage = "quantized_geometry"
        result["quantized_to_h5"] = geometry.compare_shapes(converted, reference, uid, probes)
        if not result["quantized_to_h5"]["pass"]:
            return {**result, "outcome": stage, "reason": "E1 geometry limits"}
    except Exception as exc:
        return {**result, "outcome": stage, "reason": str(exc), "exception": type(exc).__name__}

    duplicates = source_duplicate_matches(arrays, catalog, load_uid)
    result["source_duplicates"] = duplicates
    if duplicates:
        return {**result, "outcome": "source_duplicate", "reason": "four official tensors match exactly"}
    base = object_directory(uid)
    program_path = output / base / "program.npz"
    program_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = program_path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(program_path)
    reference_path = save_brep(output, base / "h5_reference.brep", reference)
    converted_path = save_brep(output, base / "text2cad_replay.brep", converted)
    result["artifacts"] = {
        "arrays": artifact(output, base / "program.npz"),
        "reference_brep": artifact(output, reference_path),
        "converted_brep": artifact(output, converted_path),
    }
    return {**result, "outcome": "accepted"}


def load_inputs(output):
    pins = load_json(output / INPUT_PIN)
    if (pins["archive_sha256"] != CADTRANSLATOR_ZIP_SHA or pins["inventory_sha256"] != digest_of("inventory")
            or pins["scope"] not in {"full", "pilot"}):
        raise ValueError("Unexpected Fusion input pins")
    verify_sha256(output / INPUTS, pins["inputs_sha256"])
    rows = read_jsonl(output / INPUTS)
    uids = [row["uid"] for row in rows]
    if (uids != pins["selected_uids"] or uids != sorted(set(uids))
            or any(row["domain"] != DOMAIN for row in rows)
            or sum(row["caption"] is not None for row in rows) != pins["n_queries"]):
        raise ValueError("Fusion extraction and input pins disagree")
    if pins["scope"] == "full" and (len(rows), pins["n_queries"]) != (6_757, 6_502):
        raise ValueError("A full Fusion run requires the entire registered inventory")
    return pins, rows


def replay_fusion(output, source_archive, split_path, e1_prep, validation_dir):
    verify_sha256(source_archive, digest_of("text2cad_program_zip"))
    with single_writer(output):
        inputs, rows = load_inputs(output)
        catalog, coverage = source_catalog(e1_prep, validation_dir, split_path)
        complete = {key: item["count"] for key, item in coverage.items()} == EXPECTED_SOURCE_SPLIT
        if inputs["scope"] == "full" and not complete:
            raise ValueError("A full Fusion run requires all three official source splits")
        upstream = Path(os.environ.get("TEXT2CAD_SOURCE", str(geometry.DEFAULT_SOURCE)))
        actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual != geometry.UPSTREAM_SHA:
            raise ValueError("Expected the pinned Text2CAD source")
        subprocess.run(["git", "-C", str(upstream), "diff", "--quiet", actual, "--", "CadSeqProc"], check=True)
        geometry.sequence_class()
        geometry.causal_mask()
        import OCC
        import scipy
        import torch
        from vtkmodules.vtkCommonCore import vtkVersion

        pins = {
            "scope": inputs["scope"], "input_pin_sha256": file_digest(output / INPUT_PIN),
            "complete_source_index": complete, "source_coverage": coverage,
            "source_archive_sha256": digest_of("text2cad_program_zip"),
            "source_split_sha256": digest_of("text2cad_split_json"),
            "h5_replay_revision": DEEP_CAD_SHA, "text2cad_revision": actual,
            "runtime": {"python": sys.version, "occt": OCC.VERSION, "vtk": vtkVersion.GetVTKVersion(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
            "implementation": {
                name: file_digest(path) for name, path in (
                    ("gallery", Path(__file__)),
                    ("codec", Path(__file__).with_name("prepare_retrieval_benchmark_fusion.py")),
                    ("geometry", Path(geometry.__file__)),
                )
            },
        }
        check_same_or_write(output / REPLAY_PIN, pins)
        replay_sha = file_digest(output / REPLAY_PIN)
        records = []
        with ZipFile(source_archive) as archive:
            for index, row in enumerate(rows, 1):
                path = output / object_directory(row["uid"]) / "evidence.json"
                if path.exists():
                    record = load_json(path)
                else:
                    started = monotonic()
                    record = convert_object(row, output, replay_sha, catalog,
                                            lambda uid: load_source_tensors(archive, official_member(uid)))
                    record["elapsed_seconds"] = monotonic() - started
                    atomic_json(path, record)
                verify_object(output, record, row, replay_sha)
                records.append(record)
                if index % 100 == 0:
                    print(f"Replayed {index}/{len(rows)} Fusion candidates", flush=True)
        atomic_json(output / "fusion_object_evidence.jsonl", records, lines=True)
        modes = Counter(f"{'+'.join(map(str, sorted(set(r['extent_modes']))))}/{r['outcome']}" for r in records)
        receipt = {
            "scope": inputs["scope"], "replay_pin_sha256": replay_sha,
            "objects_sha256": file_digest(output / "fusion_object_evidence.jsonl"),
            "n_candidates": len(records), "complete_source_index": complete,
            "outcomes": Counter(row["outcome"] for row in records), "outcomes_by_mode_set": modes,
            "replay_object_seconds": sum(row["elapsed_seconds"] for row in records),
        }
        atomic_json(output / "fusion_replay_receipt.json", receipt)
        return receipt


def make_axes(records):
    accepted = sorted((row for row in records if row["outcome"] == "accepted"), key=lambda row: row["uid"])
    gallery = [
        {"uid": row["uid"], "cad_index": index, "member": row["artifacts"]["arrays"]["member"],
         "h5_member": row["h5_member"]}
        for index, row in enumerate(accepted)
    ]
    if len({row["uid"] for row in gallery}) != len(gallery):
        raise ValueError("Duplicate accepted Fusion UID")
    texts, queries = freeze_condition(gallery, {row["uid"]: row["caption"] for row in accepted if row["caption"]})
    qrels = [{"uid": row["uid"], "candidate_uid": row["uid"], "relevance": 1} for row in queries]
    fields = np.asarray([row["candidate_fields"] for row in accepted], dtype="<i4")
    if np.any(fields <= 0):
        raise ValueError("Invalid Fusion scorer field count")
    return gallery, texts, queries, qrels, fields


def freeze_fusion(output):
    """Publish an atomic final axis after complete, source-deduplicated replay."""
    with single_writer(output):
        inputs, rows = load_inputs(output)
        if inputs["scope"] != "full":
            raise ValueError("Pilot runs do not produce a final Fusion cohort")
        replay_pins = load_json(output / REPLAY_PIN)
        receipt = load_json(output / "fusion_replay_receipt.json")
        replay_sha = file_digest(output / REPLAY_PIN)
        if (replay_pins["scope"] != "full" or not replay_pins["complete_source_index"]
                or {key: item["count"] for key, item in replay_pins["source_coverage"].items()} != EXPECTED_SOURCE_SPLIT
                or replay_pins["input_pin_sha256"] != file_digest(output / INPUT_PIN)
                or receipt["replay_pin_sha256"] != replay_sha or receipt["n_candidates"] != len(rows)):
            raise ValueError("The replay does not cover the full pinned Fusion and source inventories")
        ledger = output / "fusion_object_evidence.jsonl"
        verify_sha256(ledger, receipt["objects_sha256"])
        records = read_jsonl(ledger)
        if [row["uid"] for row in records] != [row["uid"] for row in rows]:
            raise ValueError("Incomplete or reordered conversion ledger")
        for record, row in zip(records, rows):
            verify_object(output, record, row, replay_sha)
        gallery, texts, queries, qrels, fields = make_axes(records)
        if not gallery or not queries:
            raise ValueError("An empty gallery or query axis cannot be evaluated")
        groups = Counter(row["text_index"] for row in queries)
        destination = output / "frozen"
        manifest = {
            "schema_version": 1, "domain": DOMAIN, "gallery_count": len(gallery),
            "gallery_order": "ascending Fusion UID",
            "candidate_member_root": "preparation_run_directory", "candidate_program_source": "converted_h5",
            "background_config": "logmix_l1", "background_source_text_count": 48,
            "background_ordered_digest": BACKGROUND_ORDER_SHA256,
            "conditions": {CONDITION: {
                "queries": len(queries), "unique_texts": len(texts), "largest_group": max(groups.values()),
                "singleton_queries": sum(n == 1 for n in groups.values()),
            }},
            "source_replay": receipt,
        }
        with tempfile.TemporaryDirectory(prefix=".fusion-freeze-", dir=output) as staging:
            path = Path(staging) / "payload"
            path.mkdir()
            for name, axis in (
                ("gallery.jsonl", gallery), (f"texts_{CONDITION}.jsonl", texts),
                (f"queries_{CONDITION}.jsonl", queries), ("qrels.jsonl", qrels),
            ):
                atomic_json(path / name, axis, lines=True)
            np.save(path / "gallery_fields.npy", fields, allow_pickle=False)
            manifest["files"] = {file.name: file_digest(file) for file in sorted(path.iterdir())}
            if destination.exists():
                if load_json(destination / "freeze.json") != manifest:
                    raise ValueError("An existing final Fusion cohort has different pins")
                for name, digest in manifest["files"].items():
                    verify_sha256(destination / name, digest)
            else:
                atomic_json(path / "freeze.json", manifest)
                path.rename(destination)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    extract = commands.add_parser("extract")
    validation = commands.add_parser("validation-index")
    replay = commands.add_parser("replay")
    freeze = commands.add_parser("freeze")
    for command in (extract, validation, replay, freeze):
        command.add_argument("--output", type=Path, required=True)
    for command in (extract, validation):
        command.add_argument("--pilot-uid", action="append", default=[])
        command.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--inventory", type=Path, required=True)
    for command in (validation, replay):
        command.add_argument("--official-split", type=Path, required=True)
    replay.add_argument("--source-archive", type=Path, required=True)
    replay.add_argument("--e1-prep", type=Path, required=True)
    replay.add_argument("--validation-index", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "extract":
        report = extract_fusion(args.archive, args.inventory, args.output, args.pilot_uid)
        report = {**report, "selected_uids": len(report["selected_uids"])}
    elif args.stage == "validation-index":
        report = prepare_validation_index(args.archive, args.official_split, args.output, args.pilot_uid)
        report = {**report, "validation_uids": len(report["validation_uids"])}
    elif args.stage == "replay":
        report = replay_fusion(args.output, args.source_archive, args.official_split, args.e1_prep,
                              args.validation_index)
    else:
        report = freeze_fusion(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
