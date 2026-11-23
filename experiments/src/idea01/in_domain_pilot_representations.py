"""Audit released CAD program representations on the frozen in-domain pilot axes."""

from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import io
from itertools import combinations
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import numpy as np

from .in_domain_pilot import (
    COHORTS, GALLERY_LENGTH, INPUT_HASHES, ORIGINAL_UID, PANEL_LENGTH, SOURCE_LEVELS,
    SPLIT_COUNTS, TRAIN_LENGTH, cad_member, hash_order, minimal_member, primitive_counts,
    read_jsonl, select_panel, sha256_bytes, sha256_file, sha256_lines, write_json, write_jsonl,
)


MINIMAL_ZIP_SHA256 = "7e9a29ead357fe284c29d1b1c7a411e1b8c681bdd48e2aba0542a376955fb879"
MINIMAL_PART_FIELDS = ("coordinate_system", "sketch", "extrusion")
MINIMAL_BOOLEAN_OPERATIONS = {"NewBodyFeatureOperation", "JoinFeatureOperation",
                              "CutFeatureOperation", "IntersectFeatureOperation"}
MINIMAL_CURVE_FIELDS = {"line": ("Start Point", "End Point"),
                        "arc": ("Start Point", "Mid Point", "End Point"),
                        "circle": ("Center", "Radius")}
PILOT_AXES = ("pilot_train", "validation_targets", "validation_gallery")
SPLIT_FILES = {"train": "train", "validation": "validation", "test": "test_queries"}
FORMAT_STAGES = ("text2cad_pth", "minimal_json", "ct_hdf5", "minimal_to_text2cad8")
SHA256 = re.compile(r"[0-9a-f]{64}")
# DeepCAD cadlib/macro.py. This structure test does not run the CAD kernel.
CT_ARGS = ((0, 1), (0, 1, 2, 3), (0, 1, 4), (), (), tuple(range(5, 16)))
CT_COMMANDS = ("line", "arc", "circle", "eos", "loop_start", "extrusions")
T_QUANTIZATION_BBOX_TOLERANCE = 2.0 / 256
T_QUANTIZATION_RELATIVE_SYMDIFF_TOLERANCE = 0.05


class FormatProblem(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def failure(status, exc):
    message = str(exc).splitlines()
    return {"status": status, "error_type": type(exc).__name__, "detail": message[0][:320] if message else ""}


def member_inventory(archive):
    return Counter(info.filename for info in archive.infolist())


def inspect_member(archive, inventory, member, parse):
    result = {"member": member, "archive_occurrences": inventory[member]}
    if inventory[member] != 1:
        return {**result, "status": "missing_member" if not inventory[member] else "duplicate_member"}, None
    try:
        data = archive.read(member)
    except Exception as exc:
        return {**result, **failure("archive_read_failed", exc)}, None
    result.update(member_sha256=sha256_bytes(data), member_bytes=len(data),
                  zip_crc32=f"{archive.getinfo(member).CRC:08x}")
    try:
        diagnostics, payload = parse(data)
    except FormatProblem as exc:
        return {**result, **failure(exc.status, exc)}, None
    except Exception as exc:
        return {**result, **failure("parse_failed", exc)}, None
    return {**result, **diagnostics, "status": "ok"}, payload


def program_hash(program):
    return sha256_bytes(np.asarray(program, dtype="<i4", order="C").tobytes())


def official_pth(data):
    import torch

    from .text2cad_vector_scorer import ScorerInputError, validate_cad_vec

    record = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    native = record["vec"]
    cad, flags, indices = (native[key] for key in ("cad_vec", "flag_vec", "index_vec"))
    mask = record["mask_cad_dict"]["key_padding_mask"]
    if tuple(cad.shape) != (272, 2):
        status = "overlength_program" if cad.ndim == 2 and cad.shape[0] > 272 else "unexpected_shape"
        raise FormatProblem(status, f"expected padded (272, 2), observed {tuple(cad.shape)}")
    try:
        diagnostics = validate_cad_vec(cad, mask, flags, indices)
    except ScorerInputError as exc:
        if int(cad[-1, 0]) > 1 and exc.reason == "bad_terminal_token":
            raise FormatProblem("suspected_truncation", str(exc)) from exc
        raise FormatProblem("native_validation_failed", str(exc)) from exc
    program = cad.numpy()
    real = program[:diagnostics.n_real_tokens]
    counts = {name: int(np.count_nonzero(real[:, 0] == end))
              for name, end in (("sketches", 2), ("faces", 3), ("loops", 4), ("curves", 5), ("extrusions", 6))}
    counts.update(zip(("line", "arc", "circle", "other_curve"), primitive_counts(real[:, 0]), strict=True))
    return {
        "shape": list(program.shape), "dtype": str(program.dtype), "quantized_sha256": program_hash(program),
        "flag_vec_sha256": program_hash(flags.numpy()), "index_vec_sha256": program_hash(indices.numpy()),
        "key_padding_mask_sha256": sha256_bytes(mask.numpy().astype(np.uint8, copy=False).tobytes()),
        "n_real_rows": len(real), "n_fields": int(np.count_nonzero(program[1:])),
        "at_row_limit": len(real) == 272, "structure": counts,
    }, (program, flags.numpy(), indices.numpy())


def _json_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise FormatProblem("duplicate_json_key", "duplicate field in released Minimal JSON")
    return result


def _nonfinite_json(value):
    raise FormatProblem("nonfinite_json", f"non-finite JSON token {value}")


def _objects(value, label):
    if not isinstance(value, dict) or not value or not all(isinstance(item, dict) for item in value.values()):
        raise FormatProblem("unexpected_schema", f"{label} must contain ordered nonempty objects")
    return value


def _numeric(value, point_size=None):
    if point_size is not None:
        return isinstance(value, list) and len(value) == point_size and all(_numeric(item) for item in value)
    return type(value) in (int, float) and math.isfinite(value)


def minimal_json(data):
    model = json.loads(data, object_pairs_hook=_json_object, parse_constant=_nonfinite_json)
    if not isinstance(model, dict):
        raise FormatProblem("unexpected_schema", "Minimal JSON root must be an object")
    parts = _objects(model.get("parts"), "parts")
    counts = dict.fromkeys(("extrusions", "sketches", "faces", "loops", "curves", "line", "arc", "circle", "other_curve"), 0)
    counts["extrusions"] = counts["sketches"] = len(parts)
    part_geometry = {}
    for name, part in parts.items():
        coords, sketch, extrusion = (part.get(field) for field in MINIMAL_PART_FIELDS)
        if not all(isinstance(section, dict) for section in (coords, sketch, extrusion)):
            raise FormatProblem("unexpected_schema", f"missing coordinate system, sketch or extrusion in {name}")
        if not all(_numeric(coords.get(field), 3) for field in ("Euler Angles", "Translation Vector")):
            raise FormatProblem("unexpected_schema", f"invalid coordinate system in {name}")
        scalars = ("extrude_depth_towards_normal", "extrude_depth_opposite_normal", "sketch_scale")
        if not all(_numeric(extrusion.get(field)) for field in scalars):
            raise FormatProblem("unexpected_schema", f"invalid extrusion scalars in {name}")
        if extrusion.get("operation") not in MINIMAL_BOOLEAN_OPERATIONS:
            raise FormatProblem("unsupported_operation", f"unsupported extrusion operation in {name}")
        if extrusion["sketch_scale"] <= 0:
            raise FormatProblem("invalid_sketch_scale", f"nonpositive sketch scale in {name}")
        for face in _objects(sketch, "sketch faces").values():
            counts["faces"] += 1
            for loop in _objects(face, "face loops").values():
                counts["loops"] += 1
                for key, curve in _objects(loop, "loop curves").items():
                    curve_type = key.split("_", 1)[0]
                    if curve_type not in MINIMAL_CURVE_FIELDS:
                        raise FormatProblem("unsupported_curve", f"unsupported curve type {curve_type}")
                    if not all(_numeric(curve.get(field), None if field == "Radius" else 2)
                               for field in MINIMAL_CURVE_FIELDS[curve_type]):
                        raise FormatProblem("unexpected_schema", f"invalid {curve_type} scalars or points in {name}")
                    if curve_type == "circle" and curve["Radius"] <= 0:
                        raise FormatProblem("invalid_circle_radius", f"nonpositive circle radius in {name}")
                    counts[curve_type] += 1
                    counts["curves"] += 1
        part_geometry[name] = {field: part[field] for field in MINIMAL_PART_FIELDS}
    ordered_geometry = json.dumps(part_geometry, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return {"structure": counts, "ordered_geometry_sha256": sha256_bytes(ordered_geometry.encode("utf-8"))}, model


def ct_hdf5(data):
    import h5py

    with h5py.File(io.BytesIO(data), "r") as stream:
        keys = sorted(stream.keys())
        link = stream.get("vec", getlink=True)
        if not isinstance(link, h5py.HardLink) or not isinstance(stream["vec"], h5py.Dataset):
            raise FormatProblem("unexpected_schema", f"expected local HDF5 dataset 'vec', found {keys}")
        dataset = stream["vec"]
        if dataset.is_virtual or dataset.external or len(dataset.shape) != 2 or dataset.shape[1] != 17:
            raise FormatProblem("unexpected_shape", f"expected an embedded (N, 17) dataset, found {dataset.shape}")
        if not np.issubdtype(dataset.dtype, np.integer) or dataset.shape[0] < 3:
            raise FormatProblem("unexpected_dtype_or_length", f"vec has dtype {dataset.dtype}, shape {dataset.shape}")
        array = dataset[...]
    commands = array[:, 0]
    if np.any((commands < 0) | (commands >= len(CT_COMMANDS))):
        raise FormatProblem("unknown_command", "DeepCAD command index outside 0..5")
    eos = np.flatnonzero(commands == 3)
    if not len(eos):
        raise FormatProblem("suspected_truncation", "DeepCAD sequence has no EOS row")
    end = int(eos[0])
    if commands[0] != 4 or end < 3 or commands[end - 1] != 5 or not np.all(commands[end:] == 3):
        raise FormatProblem("invalid_command_order", "expected SOL at start, EXT directly before terminal EOS, and EOS tail")
    for command, active in enumerate(CT_ARGS):
        parameters = array[commands == command, 1:]
        inactive = sorted(set(range(16)) - set(active))
        if np.any(parameters[:, inactive] != -1) or np.any((parameters[:, active] < 0) | (parameters[:, active] > 255)):
            raise FormatProblem("invalid_arguments", f"DeepCAD {CT_COMMANDS[command]} has invalid active/pad arguments")
    extrusions = array[commands == 5]
    if np.any(extrusions[:, -2] > 3) or np.any(extrusions[:, -1] > 2):
        raise FormatProblem("invalid_arguments", "DeepCAD extrusion Boolean or extent type outside vocabulary")
    counts = {command: int(np.count_nonzero(commands[:end + 1] == index))
              for index, command in enumerate(CT_COMMANDS)}
    return {"hdf5_root_keys": keys, "dataset": "vec", "shape": list(array.shape), "dtype": str(array.dtype),
            "array_sha256": program_hash(array), "effective_program_sha256": program_hash(array[:end + 1]),
            "n_effective_rows": end + 1, "n_eos_padding_rows": len(array) - end - 1,
            "above_upstream_60_row_limit": end + 1 > 60, "structure": counts}, array


def _frozen_rows(path, receipt, relative):
    manifest = path / relative
    if sha256_file(manifest) != receipt["file_sha256"][relative]:
        raise ValueError(f"frozen manifest fingerprint mismatch for {relative}")
    rows = read_jsonl(manifest)
    seen = set()
    for row in rows:
        uid = row["uid"]
        if (not ORIGINAL_UID.fullmatch(uid) or uid in seen or row["cad_member"] != cad_member(uid)
                or not SHA256.fullmatch(row["quantized_sha256"]) or not SHA256.fullmatch(row["member_sha256"])
                or type(row["n_fields"]) is not int or row["n_fields"] <= 0):
            raise ValueError(f"invalid UID, member or fingerprint in {relative}")
        seen.add(uid)
    return {row["uid"]: row for row in rows}


def _counter_stats(counts):
    return {"rows": sum(counts.values()), "unique": len(counts),
            "duplicate_groups": sum(count > 1 for count in counts.values()),
            "rows_in_duplicate_groups": sum(count for count in counts.values() if count > 1),
            "largest_group": max(counts.values(), default=0)}


def _split_keys(rows, levels):
    texts = {level: Counter(sha256_bytes(row["texts"][level].encode("utf-8")) for row in rows.values())
             for level in levels}
    all_texts = sum(texts.values(), Counter())
    return {"uid": Counter(rows.keys()), "quantized_program": Counter(row["quantized_sha256"] for row in rows.values()),
            "text_any_level": all_texts, **{f"text_{key}": counts for key, counts in texts.items()}}


def _overlap(left, right):
    shared = left.keys() & right.keys()
    return {"left_rows": left.total(), "right_rows": right.total(), "shared_groups": len(shared),
            "left_rows_in_shared_groups": sum(left[key] for key in shared),
            "right_rows_in_shared_groups": sum(right[key] for key in shared)}


def _compare_splits(left, right, levels):
    comparison = {key: _overlap(left[key], right[key]) for key in left}
    comparison["right_level_against_left_any_text_level"] = {
        level: _overlap(left["text_any_level"], right[f"text_{level}"]) for level in levels}
    return comparison


def manifest_inventory(manifests):
    """Read pinned manifests and use only counts from test records."""
    receipt = json.loads((manifests / "manifest_receipt.json").read_text(encoding="utf-8"))
    test_gallery = _frozen_rows(manifests, receipt, "test_gallery.jsonl")
    if (len(test_gallery) != SPLIT_COUNTS["test"] or sha256_lines(test_gallery) != receipt["test_cad_axis_sha256"]
            or [row["cad_index"] for row in test_gallery.values()] != list(range(len(test_gallery)))):
        raise ValueError("test gallery axis differs from the frozen receipt")
    test_gallery_keys = _split_keys(test_gallery, ())
    sources = {}
    references = defaultdict(list)
    provenance = {}
    for source, levels in SOURCE_LEVELS.items():
        corpora = {split: _frozen_rows(manifests, receipt, f"{source}/{name}.jsonl")
                   for split, name in SPLIT_FILES.items()}
        axes = {name: _frozen_rows(manifests, receipt, f"{source}/{name}.jsonl") for name in PILOT_AXES}
        for split, rows in corpora.items():
            if (len(rows), sha256_lines(sorted(rows))) != COHORTS[source][split]:
                raise ValueError(f"{source} {split} cohort differs from the registered contract")
            if any(set(row["texts"]) != set(levels) for row in rows.values()):
                raise ValueError(f"{source} {split} caption levels differ from the registered contract")
        targets, gallery = select_panel(corpora["validation"], source)
        train = hash_order(corpora["train"], "cadretrieval-pilot-train-v1", source)[:TRAIN_LENGTH]
        for name, expected, split in zip(PILOT_AXES, (train, targets, gallery),
                                         ("train", "validation", "validation"), strict=True):
            if list(axes[name]) != expected:
                raise ValueError(f"{source} {name} differs from the frozen selection")
            for index, (uid, row) in enumerate(axes[name].items()):
                canonical = corpora[split][uid]
                expected_row = ({**canonical, "texts": {}, "cad_index": index}
                                if name == "validation_gallery" else canonical)
                if row != expected_row:
                    raise ValueError(f"{source} {name}/{uid} differs from the source split manifest")
                references[uid].append((source, name, row))
        provenance[source] = {name: {"uids": len(rows), "ordered_uid_sha256": sha256_lines(rows)}
                              for name, rows in axes.items()}
        expected_lengths = dict(zip(PILOT_AXES, (TRAIN_LENGTH, PANEL_LENGTH, GALLERY_LENGTH), strict=True))
        if {name: len(rows) for name, rows in axes.items()} != expected_lengths:
            raise ValueError(f"{source} unexpected pilot cohort sizes")
        keys = {split: _split_keys(rows, levels) for split, rows in corpora.items()}
        source_keys = {"full_train": keys["train"], "pilot_train": _split_keys(axes["pilot_train"], levels),
                       "full_validation": keys["validation"],
                       "validation_targets": _split_keys(axes["validation_targets"], levels)}
        sources[source] = {
            "full_cohort_duplicates": {split: {key: _counter_stats(counts) for key, counts in values.items()}
                                       for split, values in keys.items()},
            "full_cohort_cross_split": {f"{left}_to_{right}": _compare_splits(keys[left], keys[right], levels)
                                        for left, right in combinations(keys, 2)},
            "train_or_validation_to_test_queries": _compare_splits(
                {key: counts + keys["validation"][key] for key, counts in keys["train"].items()}, keys["test"], levels),
            "pilot_train_to_validation_targets": _compare_splits(
                source_keys["pilot_train"], source_keys["validation_targets"], levels),
            "pilot_train_to_test_queries": _compare_splits(source_keys["pilot_train"], keys["test"], levels),
            "test_gallery_overlap": {name: {key: _overlap(values[key], test_gallery_keys[key])
                                            for key in ("uid", "quantized_program")}
                                     for name, values in source_keys.items()},
        }
    return references, provenance, {
        "text_fingerprint": "SHA-256 of the exact UTF-8 manifest caption, with only release ingestion's edge trim",
        "program_fingerprint": "SHA-256 of the common official padded Text2CAD cad_vec as little-endian int32",
        "test_output": "aggregate counts from frozen test manifests only, with zero test rankings or model scores",
        "test_gallery_program_groups": _counter_stats(test_gallery_keys["quantized_program"]),
        "sources": sources,
    }, receipt


def release_assets(paths, receipt):
    assets = {}
    for name, expected in (("official_zip", INPUT_HASHES["official_zip"]), ("minimal_zip", MINIMAL_ZIP_SHA256),
                           ("ct_zip", INPUT_HASHES["ct_zip"])):
        path = paths[name]
        actual = sha256_file(path)
        if actual != expected or actual != receipt["assets"][name]["sha256"]:
            raise ValueError(f"{name} does not match the frozen release and manifest receipt")
        assets[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    return assets


def activate_deepcad(root):
    from unittest.mock import patch

    import matplotlib

    root = root.resolve()
    git = ["git", "-C", str(root)]
    remote = subprocess.check_output([*git, "remote", "get-url", "origin"], text=True).strip()
    if re.fullmatch(r"(?:https://github\.com/|git@github\.com:)(?:rundiwu|ChrisWu1997)/DeepCAD(?:\.git)?", remote) is None:
        raise ValueError("the DeepCAD checkout does not identify the official source")
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output([*git, "status", "--porcelain", "--", "cadlib"], text=True)
    if changes:
        raise ValueError("the DeepCAD converter has local modifications")
    files = {path.relative_to(root).as_posix(): sha256_file(path)
             for path in sorted((root / "cadlib").glob("*.py"))}
    sys.path.insert(0, str(root))
    # The release selects TkAgg at import even when only its solid converter is used.
    set_backend = matplotlib.use

    def headless_backend(name, **kwargs):
        return set_backend("Agg" if name.casefold() == "tkagg" else name, **kwargs)

    with patch.object(matplotlib, "use", headless_backend):
        from cadlib import visualize

    if Path(visualize.__file__).resolve().parent != root / "cadlib":
        raise ValueError("the imported DeepCAD converter came from another checkout")
    return visualize.vec2CADsolid, {"revision": revision, "origin": remote, "files_sha256": files,
                                   "plotting_backend": "Agg for the release's TkAgg import"}


def load_converter(root, activate):
    if root is None:
        return None, {"status": "unavailable", "detail": "no official converter checkout supplied"}
    try:
        converter, provenance = activate(root)
    except Exception as exc:
        return None, failure("unavailable", exc)
    return converter, {"status": "ready", "checkout": str(root), **provenance}


def minimal_to_text2cad_8bit(sequence_type, model, native_arrays):
    """Invert the author export's sketch-only rescaling, then run its native 8-bit encoder."""
    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT

    from .in_domain_pilot_geometry import minimal_replay_sequence
    from .text2cad_vector_scorer import validate_cad_vec

    if N_BIT != 8 or MAX_CAD_SEQUENCE_LENGTH != 272:
        raise ValueError("the verified converter does not use the published program constants")
    sequence = minimal_replay_sequence(sequence_type, model)
    for sketch, extrusion in zip(sequence.sketch_seq, sequence.extrude_seq, strict=True):
        # The author exports globally normalized origin/orientation/extrusion fields. Its
        # Minimal JSON sketch was expanded by sketch_scale / (2**N_BIT - 1).
        sketch.transform(0.0, (2**N_BIT - 1) / extrusion.metadata["sketch_size"])
        # The official numericalizer calls scalar .clip. The JSON reader yields Python scalars.
        for key in ("extent_one", "extent_two", "sketch_size"):
            extrusion.metadata[key] = np.float64(extrusion.metadata[key])
    encoded = sequence.numericalize(bit=N_BIT).to_vec(padding=False)
    if encoded is None:
        raise ValueError("the official encoder returned no program")
    converted = encoded.cad_vec.numpy()
    if len(converted) > MAX_CAD_SEQUENCE_LENGTH:
        return {"status": "overlength_program", "n_real_rows": len(converted),
                "untruncated_program_sha256": program_hash(converted)}
    cad, flags, indices = native_arrays
    n_fields = int(np.count_nonzero(converted[1:]))
    real_native = cad[cad[:, 0] != 0]
    result = {"n_real_rows": len(converted), "n_fields": n_fields,
              "native_n_real_rows": len(real_native), "native_n_fields": int(np.count_nonzero(real_native[1:])),
              "padded_quantized_sha256": program_hash(np.pad(converted, ((0, len(cad) - len(converted)), (0, 0))))}
    validate_cad_vec(encoded.cad_vec, encoded.cad_vec == 0, encoded.flag_vec, encoded.index_vec)
    if len(converted) != len(real_native):
        return {**result, "status": "row_count_mismatch"}
    values_match = np.array_equal(converted, real_native)
    fields_equal = ((real_native != 0) == (converted != 0)).all()
    flags_match = np.array_equal(encoded.flag_vec.numpy(), flags[:len(real_native)])
    indices_match = np.array_equal(encoded.index_vec.numpy(), indices[:len(real_native)])
    if values_match and flags_match and indices_match:
        status = "exact_agreement"
    else:
        status = "auxiliary_mismatch" if values_match else "native_program_mismatch"
    return {**result, "status": status, "valid_field_mask_equal": bool(fields_equal),
            "flag_vec_equal": flags_match, "index_vec_equal": indices_match,
            "mismatched_fields": int(np.count_nonzero(real_native != converted)),
            "max_absolute_label_difference": int(np.abs(real_native - converted).max())}


def _shape_metrics(shape):
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from .in_domain_pilot_geometry import _bounds, _volume

    if shape is None or shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("OpenCASCADE returned an invalid shape")
    solids = TopologyExplorer(shape).number_of_solids()
    volume = _volume(shape)
    bounds = _bounds(shape)
    if not solids or volume <= 0 or not all(math.isfinite(value) for value in bounds):
        raise ValueError("OpenCASCADE returned no measurable positive-volume solid")
    return {"volume": volume, "bounds_xyz_min_max": bounds, "solids": solids}


def _strict_text2cad_replay(sequence, expected_extrusions):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

    if len(sequence.sketch_seq) != expected_extrusions or not expected_extrusions:
        raise ValueError("the decoder did not retain all registered extrusion steps")
    operations = (BRepAlgoAPI_Fuse, BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common)
    result = None
    for sketch, extrusion in zip(sequence.sketch_seq, sequence.extrude_seq, strict=True):
        metadata = extrusion.metadata
        tool = sketch.build_body({"extrude_values": [metadata["extent_one"], metadata["extent_two"]]})
        _shape_metrics(tool)
        if result is None:
            if metadata["boolean"] not in (0, 1):
                raise ValueError("the initial extrusion does not create material")
            result = tool
        else:
            operation = operations[metadata["boolean"]](result, tool)
            if not operation.IsDone():
                raise ValueError("OpenCASCADE did not finish the released extrusion Boolean")
            result = operation.Shape()
            _shape_metrics(result)
    return result


def _attempt_replay(make_shape):
    try:
        shape = make_shape()
        return {"status": "ok", **_shape_metrics(shape)}, shape
    except Exception as exc:
        return failure("replay_failed", exc), None


def _compare_replays(left, right, left_shape, right_shape):
    from .in_domain_pilot_geometry import _cut, _volume

    try:
        difference = _volume(_cut(left_shape, right_shape)) + _volume(_cut(right_shape, left_shape))
    except Exception as exc:
        return failure("boolean_comparison_failed", exc)
    larger_volume = max(left["volume"], right["volume"])
    relative_difference = difference / larger_volume
    bound_error = max(abs(a - b) for a, b in zip(left["bounds_xyz_min_max"], right["bounds_xyz_min_max"], strict=True))
    same_solid_count = left["solids"] == right["solids"]
    within_tolerance = (same_solid_count and bound_error <= T_QUANTIZATION_BBOX_TOLERANCE
                        and relative_difference <= T_QUANTIZATION_RELATIVE_SYMDIFF_TOLERANCE)
    return {"status": "within_diagnostic_tolerance" if within_tolerance else "outside_diagnostic_tolerance",
            "bbox_max_absolute_difference": bound_error, "symmetric_difference_volume": difference,
            "relative_symmetric_difference_volume": relative_difference,
            "absolute_volume_difference": abs(left["volume"] - right["volume"]),
            "same_solid_count": same_solid_count}


def replay_released_programs(sequence_type, deepcad_convert, row, native_arrays, minimal, ct_vector):
    from .in_domain_pilot_geometry import minimal_replay_sequence

    blocked = {"status": "unavailable", "detail": "official converter unavailable"}
    t_replay = minimal_replay = ct_replay = blocked
    t_shape = minimal_shape = None
    if sequence_type is not None:
        t_replay = {"status": "blocked_by_format"}
        minimal_replay = {"status": "blocked_by_format"}
        if native_arrays is not None:
            t_replay, t_shape = _attempt_replay(lambda: _strict_text2cad_replay(
                sequence_type.from_vec(native_arrays[0].copy(), bit=8, post_processing=False),
                row["text2cad_pth"]["structure"]["extrusions"],
            ))
        if minimal is not None:
            minimal_replay, minimal_shape = _attempt_replay(lambda: _strict_text2cad_replay(
                minimal_replay_sequence(sequence_type, minimal), row["minimal_json"]["structure"]["extrusions"],
            ))
    if deepcad_convert is not None:
        ct_replay = {"status": "blocked_by_format"}
        if ct_vector is not None:
            ct_replay, _ = _attempt_replay(lambda: deepcad_convert(ct_vector.copy()))
    comparison = {"status": "replay_unavailable"}
    if t_shape is not None and minimal_shape is not None:
        comparison = _compare_replays(t_replay, minimal_replay, t_shape, minimal_shape)
    return {"official_text2cad": t_replay, "released_minimal_json": minimal_replay, "released_ct_hdf5": ct_replay,
            "text2cad_pth_vs_minimal_json": comparison,
            "ct_cross_format_geometry": {"status": "unsupported_frame_alignment",
                                         "detail": "raw DeepCAD frame and Text2CAD frame are distinct"}}


def select_replay_uids(references, count):
    groups = defaultdict(list)
    for uid, refs in sorted(references.items()):
        target_sources = {source for source, axis, _ in refs if axis == "validation_targets"}
        for source, axis, _ in refs:
            if axis != "validation_gallery" or source not in target_sources:
                groups[(source, axis)].append(uid)
    axes = [iter(hash_order(uids, "cadretrieval-pilot-format-replay-v1", f"{source}|{axis}"))
            for (source, axis), uids in sorted(groups.items())]
    selected = set()
    while axes and len(selected) < count:
        remaining = []
        for group in axes:
            if len(selected) == count:
                break
            uid = next(group, None)
            if uid is not None:
                selected.add(uid)
                remaining.append(group)
        axes = remaining
    return selected


def _audit_uid(uid, refs, archives, inventories, t_converter, ct_converter, replay_uids):
    t_pth, native = inspect_member(archives[0], inventories[0], cad_member(uid), official_pth)
    minimal, model = inspect_member(archives[1], inventories[1], minimal_member(uid), minimal_json)
    ct_path = f"dataset/deepcad_txt/cad_vec_original/{uid}.h5"
    ct_h5, ct_vector = inspect_member(archives[2], inventories[2], ct_path, ct_hdf5)
    agreed = {field: all(t_pth.get(field) == row[field] for _, _, row in refs)
              for field in ("member_sha256", "quantized_sha256", "n_fields")}
    minimal_registration = all(row["minimal_member"] == (minimal_member(uid) if minimal["archive_occurrences"] == 1 else None)
                               for _, _, row in refs)
    ct_registration = all(row["ct_vector_member"] == (ct_path if ct_h5["archive_occurrences"] == 1 else None)
                           for source, _, row in refs if source == "ct-deepcad")
    structure = {}
    if native is not None and model is not None:
        structure["text2cad_pth_vs_minimal_json"] = {key: t_pth["structure"][key] == minimal["structure"][key]
                                                    for key in minimal["structure"]}
    if ct_vector is not None and model is not None:
        structure["ct_hdf5_vs_minimal_json"] = {
            key: ct_h5["structure"][key] == minimal["structure"][key] for key in ("extrusions", "line", "arc", "circle")}
    row = {"uid": uid, "source_cohorts": [[source, axis] for source, axis, _ in refs],
           "text2cad_pth": t_pth, "minimal_json": minimal, "ct_hdf5": ct_h5, "structure_comparisons": structure,
           "frozen_manifest_agreement": {**agreed, "minimal_member": minimal_registration,
                                         "ct_member": ct_registration if any(source == "ct-deepcad" for source, _, _ in refs) else None}}
    if native is None or model is None:
        rebuilt = {"status": "blocked_by_format"}
    elif t_converter is None:
        rebuilt = {"status": "unavailable"}
    else:
        try:
            rebuilt = minimal_to_text2cad_8bit(t_converter, model, native)
        except Exception as exc:
            rebuilt = failure("conversion_failed", exc)
    row["minimal_to_text2cad8"] = rebuilt
    if uid in replay_uids:
        row["replay"] = replay_released_programs(t_converter, ct_converter, row, native, model, ct_vector)
    return row


def _update_coverage(summary, row):
    summary["uids"] += 1
    for name in FORMAT_STAGES:
        summary["stages"][name][row[name]["status"]] += 1
    for field, matches in row["frozen_manifest_agreement"].items():
        status = "not_registered" if matches is None else ("match" if matches else "mismatch")
        summary["frozen_manifest_agreement"][field][status] += 1
    for comparison, counts in row["structure_comparisons"].items():
        summary["structure_comparisons"][comparison]["all_fields_match" if all(counts.values()) else "counts_differ"] += 1
    summary["text2cad_pth_at_row_limit"] += row["text2cad_pth"].get("at_row_limit", False)
    summary["ct_above_upstream_60_row_limit"] += row["ct_hdf5"].get("above_upstream_60_row_limit", False)
    for comparison, detail in row.get("replay", {}).items():
        summary["replay"][comparison][detail["status"]] += 1


def _coverage():
    return {"uids": 0, "stages": defaultdict(Counter), "frozen_manifest_agreement": defaultdict(Counter),
            "structure_comparisons": defaultdict(Counter), "text2cad_pth_at_row_limit": 0,
            "ct_above_upstream_60_row_limit": 0, "replay": defaultdict(Counter)}


def run_audit(manifests, official_zip, minimal_zip, ct_zip, output, *, text2cad_root=None, deepcad_root=None,
              replay_count=16):
    import h5py
    import torch

    from .in_domain_pilot_geometry import activate_text2cad

    if output.exists():
        raise FileExistsError(output)
    if replay_count < 0:
        raise ValueError("replay count must be nonnegative")
    torch.set_num_threads(1)
    references, axes, overlaps, receipt = manifest_inventory(manifests)
    if len(references) < TRAIN_LENGTH + GALLERY_LENGTH:
        raise ValueError("unique representation denominator is smaller than one source's disjoint pilot cohorts")
    print("Frozen train and validation axes and cross-split manifest counts verified", flush=True)
    paths = {"official_zip": official_zip, "minimal_zip": minimal_zip, "ct_zip": ct_zip}
    assets = release_assets(paths, receipt)
    print("Release archive SHA-256 fingerprints match", flush=True)
    replay_uids = select_replay_uids(references, replay_count)
    union = _coverage()
    by_axis = {source: {name: _coverage() for name in PILOT_AXES} for source in SOURCE_LEVELS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".representation-audit-") as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        with (staged / "converter.log").open("w", encoding="utf-8") as converter_log:
            with redirect_stdout(converter_log), redirect_stderr(converter_log):
                t_converter, t_provenance = load_converter(text2cad_root, activate_text2cad)
                ct_converter, ct_provenance = load_converter(deepcad_root, activate_deepcad)
            with ZipFile(official_zip) as official, ZipFile(minimal_zip) as minimal, ZipFile(ct_zip) as ct:
                archives = (official, minimal, ct)
                inventories = tuple(member_inventory(archive) for archive in archives)

                def results():
                    for number, (uid, refs) in enumerate(sorted(references.items()), start=1):
                        with redirect_stdout(converter_log), redirect_stderr(converter_log):
                            row = _audit_uid(uid, refs, archives, inventories, t_converter, ct_converter, replay_uids)
                        _update_coverage(union, row)
                        for source, name, _ in refs:
                            _update_coverage(by_axis[source][name], row)
                        if number % 1000 == 0 or number == len(references):
                            print(f"Audited {number}/{len(references)} unique train and validation programs", flush=True)
                        yield row

                write_jsonl(staged / "representation_audit.jsonl", results())
        write_json(staged / "cross_split_overlaps.json", overlaps)
        result = {
            "schema": "in-domain-pilot-representations-v1",
            "manifest_receipt_sha256": sha256_file(manifests / "manifest_receipt.json"),
            "assets": assets, "pilot_axes": axes,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": torch.__version__, "h5py": h5py.__version__, "libhdf5": h5py.version.hdf5_version},
            "converter_provenance": {"text2cad": t_provenance, "deepcad": ct_provenance},
            "coverage_unique_uids": union, "coverage_by_source_cohort": by_axis,
            "round_trip_scope": {
                "text2cad": "From release Minimal JSON with official reader, undo the local sketch scaling, "
                            "restore the circular cardinal points and radian Euler convention, then official 8-bit numericalize/to_vec. "
                            "Exact match requires every effective cad, flag and index value to agree.",
                "ct_deepcad": "Release member bytes, native HDF5 array and DeepCAD command/argument structure; "
                               "no cross-format spatial equivalence from shared UID or counts.",
            },
            "bounded_brep_scope": {
                "requested": replay_count, "sampled": len(replay_uids), "sampled_uids": sorted(replay_uids),
                "selection": "round robin deterministic salted UID hash within each source/axis, "
                             "with validation target UIDs omitted from that source's separate gallery draw",
                "normalized_coordinate_system": "the released normalized coordinate system of each converter; nonphysical units",
                "text2cad_diagnostic_absolute_bbox_tolerance": T_QUANTIZATION_BBOX_TOLERANCE,
                "text2cad_diagnostic_relative_symdiff_volume_tolerance": T_QUANTIZATION_RELATIVE_SYMDIFF_TOLERANCE,
                "text2cad_diagnostic_scope": "A finite B-Rep sample only. The thresholds are descriptive, "
                                            "with separate bounds, volume, and solid-count differences.",
                "ct_geometry_scope": "An optional native replay can only witness a valid individual solid because "
                                     "DeepCAD scales absolute raw coordinates while Text2CAD translates and scales by raw bounds; "
                                     "the original frame transform is unavailable.",
            },
            "test_scope": "frozen manifest fingerprint and overlap counts only",
            "output_sha256": {path.name: sha256_file(path) for path in staged.iterdir() if path.is_file()},
        }
        write_json(staged / "representation_audit_summary.json", result)
        staged.rename(output)
    return result
