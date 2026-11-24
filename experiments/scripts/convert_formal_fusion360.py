"""Prepare and independently replay Table 3's Fusion candidate representations."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from prepare_formal_fusion360 import (
    EXPECTED_SPLITS, OPERATIONS, SOURCE, SPLIT_SHA256, UID, ProgramError, clean_program, hash_order,
    json_bytes, number, read_jsonl, require_sha, sha256_bytes, sha256_file, write_jsonl,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARCHIVE_SHA256 = "485601de3d23e25a5d63a75588ca780fa7881d7a0bea1ab6c24dbcefa57ad5c9"
ARCHIVE_BYTES = 2103449157
AUTODESK_MEMBERS = "r1.0.1/reconstruction"
TEXT2CAD_REVISION = "669a81472ede2269837c22a4d00070eaac360c82"
HIERCAD_REVISION = "c10c4dce9fb3674932431905879ec597b65b9122"
NATIVE_CURVES = {
    "Line3D": (("start_point", "end_point"), ()),
    "Arc3D": (("start_point", "end_point", "center_point", "normal", "reference_vector"),
              ("radius", "start_angle", "end_angle")),
    "Circle3D": (("center_point", "normal"), ("radius",)),
}
ROUTES = ("autodesk_history", "cadmium_minimal")
MODELS = ("text2cad", "hiercad")


def open_catalog(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest["source"] != SOURCE or manifest["frozen_counts"] != EXPECTED_SPLITS:
        raise ValueError("Unexpected source catalog")
    for filename in ("source_inventory.jsonl", "source_programs.jsonl"):
        require_sha(root / filename, manifest["file_sha256"][filename])
    records = list(read_jsonl(root / "source_inventory.jsonl"))
    programs = {row["uid"]: row["program"] for row in read_jsonl(root / "source_programs.jsonl")}
    if (Counter(row["split"] for row in records) != EXPECTED_SPLITS
            or any(not UID.fullmatch(row["uid"]) for row in records)
            or len({row["uid"] for row in records}) != len(records)
            or set(programs) != {row["uid"] for row in records if row["minimal_status"] == "ok"}):
        raise ValueError("Catalog program or split coverage mismatch")
    for row in records:
        if row["uid"] in programs and sha256_bytes(json_bytes(programs[row["uid"]])) != row["minimal_ordered_sha256"]:
            raise ValueError("Catalog program fingerprint mismatch")
    return records, programs


def check_checkout(path, revision):
    location = path.resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(location), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"Unexpected author revision at {location}: {head}")
    changes = subprocess.check_output(["git", "-C", str(location), "status", "--porcelain",
                                       "--untracked-files=no"], text=True)
    if changes.strip():
        raise ValueError(f"Tracked changes in author checkout at {location}")
    return location


def source_point(value):
    return {axis: float(number(value[axis], axis)) for axis in "xyz"}


def source_box(raw):
    box = raw["properties"]["bounding_box"]
    return {key: source_point(box[key]) for key in ("min_point", "max_point")}


def used_profiles(raw, feature):
    for ref in feature["profiles"]:
        sketch = raw["entities"][ref["sketch"]]
        if sketch["type"] != "Sketch":
            raise ProgramError("non_sketch_profile_reference")
        yield sketch["transform"], sketch["profiles"][ref["profile"]]["loops"]


def used_features(raw):
    sequence = raw["sequence"]
    if any(item["type"] not in ("Sketch", "ExtrudeFeature") for item in sequence):
        raise ProgramError("unsupported_native_feature_type")
    features = [raw["entities"][item["entity"]] for item in sequence if item["type"] == "ExtrudeFeature"]
    if not features or any(feature["type"] != "ExtrudeFeature" for feature in features):
        raise ProgramError("invalid_or_empty_native_extrusion_history")
    return features


def source_diagnostics(raw, features):
    diagnostics = []
    for feature in features:
        profiles = []
        for transform, loops in used_profiles(raw, feature):
            profiles.append({"transform": {key: source_point(transform[key])
                                            for key in ("origin", "x_axis", "y_axis", "z_axis")},
                             "loops": [{"outer": loop["is_outer"],
                                        "curves": dict(Counter(curve["type"] for curve in loop["profile_curves"]))}
                                       for loop in loops]})
        row = {"operation": feature["operation"], "extent_type": feature["extent_type"],
               "start_extent_type": feature["start_extent"]["type"], "profiles": profiles}
        for key in ("extent_one", "extent_two"):
            if key in feature:
                extent = feature[key]
                row[key] = {"type": extent.get("type"), "distance_cm": extent.get("distance", {}).get("value"),
                            "taper_rad": extent.get("taper_angle", {}).get("value", 0),
                            "is_full_length": extent.get("is_full_length")}
        diagnostics.append(row)
    return diagnostics


def native_distances(feature):
    def distance(extent):
        if extent.get("type") not in {"DistanceExtentDefinition", "SymmetricExtentDefinition"}:
            raise ProgramError("unsupported_native_extent_definition")
        if number(extent.get("taper_angle", {}).get("value", 0), "taper_angle") != 0:
            raise ProgramError("unsupported_native_taper")
        return float(number(extent["distance"]["value"], "extrusion_distance"))

    mode = feature["extent_type"]
    one = feature["extent_one"]
    d1 = distance(one)
    if mode == "OneSideFeatureExtentType":
        lengths = max(d1, 0.0), max(-d1, 0.0)
    elif mode == "SymmetricFeatureExtentType":
        if d1 <= 0 or type(one.get("is_full_length")) is not bool:
            raise ProgramError("invalid_native_symmetric_extent")
        half_length = d1 / 2 if one["is_full_length"] else d1
        lengths = half_length, half_length
    elif mode == "TwoSidesFeatureExtentType":
        d2 = distance(feature["extent_two"])
        if min(d1, d2) < 0:
            raise ProgramError("unsupported_native_negative_two_sided_extent")
        lengths = d1, d2
    else:
        raise ProgramError("unsupported_native_extent_type")
    if sum(lengths) <= 0:
        raise ProgramError("empty_native_extrusion")
    start = feature["start_extent"]
    if start["type"] == "OffsetStartDefinition":
        return lengths, float(number(start["offset"]["value"], "offset_start"))
    if start["type"] != "ProfilePlaneStartDefinition":
        raise ProgramError("unsupported_native_start_extent")
    return lengths, 0.0


def native_curve(curve):
    kind = curve["type"]
    if kind not in NATIVE_CURVES:
        raise ProgramError("unsupported_native_profile_curve")
    points, scalars = NATIVE_CURVES[kind]
    parsed = {"type": kind, **{key: source_point(curve[key]) for key in points},
              **{key: float(number(curve[key], key)) for key in scalars}}
    if "radius" in parsed and parsed["radius"] <= 0:
        raise ProgramError("invalid_native_curve_radius")
    return parsed


def native_sequence(raw, features):
    """Validate native geometry, then project it into the pinned sketch/extrude grammar."""
    import numpy as np

    from idea01 import text2cad_e1_geometry as g

    transformed = {"properties": {"bounding_box": source_box(raw)}, "sequence": [], "entities": {}}
    corrected_depths = []
    for index, feature in enumerate(features, 1):
        operation = feature["operation"]
        if operation not in OPERATIONS:
            raise ProgramError("unsupported_native_boolean_operation")
        lengths, offset = native_distances(feature)
        corrected_depths.append(lengths)
        sketch_id, ext_id = f"sketch_{index}", f"extrusion_{index}"
        profiles = {}
        common_frame = None
        for profile_index, (transform, loops) in enumerate(used_profiles(raw, feature), 1):
            frame = {key: source_point(transform[key]) for key in ("origin", "x_axis", "y_axis", "z_axis")}
            basis = np.array([list(frame[key].values()) for key in ("x_axis", "y_axis", "z_axis")])
            if not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-6) or np.linalg.det(basis) <= 0:
                raise ProgramError("unsupported_native_sketch_transform")
            if common_frame is not None and frame != common_frame:
                raise ProgramError("unsupported_native_multi_frame_extrusion")
            common_frame = frame
            if not loops or sum(loop["is_outer"] is True for loop in loops) != 1:
                raise ProgramError("invalid_native_profile_loops")
            clean_loops = []
            for loop in loops:
                curves = [native_curve(curve) for curve in loop["profile_curves"]]
                if not curves or any(curve["type"] == "Circle3D" for curve in curves) != (len(curves) == 1):
                    raise ProgramError("unsupported_native_closed_loop")
                clean_loops.append({"is_outer": loop["is_outer"], "profile_curves": curves})
            profiles[f"profile_{profile_index}"] = {"loops": clean_loops}
        if not profiles:
            raise ProgramError("missing_native_profiles")
        for axis in "xyz":
            common_frame["origin"][axis] += offset * common_frame["z_axis"][axis]
        transformed["entities"][sketch_id] = {"type": "Sketch", "transform": common_frame, "profiles": profiles}
        transformed["entities"][ext_id] = {
            "type": "ExtrudeFeature", "operation": operation,
            "profiles": [{"sketch": sketch_id, "profile": key} for key in profiles],
            "start_extent": {"type": "ProfilePlaneStartDefinition"},
            "extent_type": "OneSideFeatureExtentType", "extent_one": {"distance": {"value": sum(lengths)}},
        }
        transformed["sequence"].append({"type": "ExtrudeFeature", "entity": ext_id})

    sequence = g.sequence_class().from_dict(transformed)
    if len(sequence.extrude_seq) != len(features) or len(sequence.sketch_seq) != len(features):
        raise ProgramError("native_feature_dropped")
    # The pinned parser ignores its second extent and halves every symmetric extent.
    # Correct both with the actual Autodesk extent values before normalization.
    for extrusion, lengths in zip(sequence.extrude_seq, corrected_depths, strict=True):
        extrusion.metadata.update(extent_one=lengths[0], extent_two=lengths[1])
    return sequence.normalize(bit=8)


def normalized_program(sequence):
    from idea01 import text2cad_e1_geometry as g

    floating = copy.deepcopy(sequence)
    parts = {}
    for index, (sketch, extrude) in enumerate(zip(floating.sketch_seq, floating.extrude_seq, strict=True), 1):
        sketch.denormalize(bbox_size=extrude.metadata["sketch_size"], translate=0, bit=8)
        parts[f"part_{index}"] = {"coordinate_system": sketch.coordsystem._json(),
                                  "sketch": sketch._json(), "extrusion": extrude._json()}
    # Convert NumPy scalars from the pinned author's exporters to JSON numbers.
    native_export = json.dumps({"parts": parts}, default=lambda value: value.item(), allow_nan=False)
    program = clean_program(json.loads(native_export))
    if len(g.minimal_sequence(program).extrude_seq) != len(sequence.extrude_seq):
        raise ProgramError("normalized_native_feature_dropped")
    return program


def failure(stage, exc):
    # Exception text can contain a source filename or a native identifier.
    if isinstance(exc, ProgramError):
        return {"status": str(exc), "stage": stage}
    if str(exc).startswith("sequence_length_over_max:"):
        return {"status": "sequence_length_over_max", "stage": stage, "rows": int(str(exc).split(":")[1])}
    return {"status": "failed", "stage": stage, "error_type": type(exc).__name__}


def comparison(sequence, reference, uid):
    from idea01 import text2cad_e1_geometry as g

    shape = g.strict_solid(sequence)
    if reference is None:
        return {"status": "independent_reference_unavailable", "candidate": g.measures(shape)}
    result = g.compare_shapes(shape, reference, uid, g.cut_probes(sequence))
    boxes = result["candidate"]["bbox"], result["reference"]["bbox"]
    result["bbox_max_abs_difference"] = max(abs(a - b) for a, b in zip(*boxes, strict=True))
    return {"status": "aligned" if result["pass"] else "geometry_mismatch", "comparison": result}


def evaluate_models(program, reference, uid, native=None):
    from idea01 import text2cad_e1_geometry as g
    from idea01.in_domain_pilot_native import hiercad_target

    result = {}
    outputs = {}
    try:
        arrays = (g.padded_arrays(copy.deepcopy(native).numericalize(bit=8))
                  if native is not None else g.encode_minimal(program))
        structure = g.validate_arrays(arrays)
        result["text2cad"] = {"input_status": "ok", "rows": structure.n_real_tokens,
                              "effective_fields": structure.n_valid_total, **g.fingerprints(arrays)}
        outputs["text2cad"] = arrays
    except Exception as exc:
        result["text2cad"] = {"input_status": "failed", "failure": failure("text2cad_encoding", exc)}
    else:
        try:
            result["text2cad"]["geometry"] = comparison(g.decode(arrays["cad_vec"]), reference, uid)
        except Exception as exc:
            result["text2cad"]["geometry"] = failure("text2cad_replay", exc)

    try:
        target = hiercad_target(program)
        result["hiercad"] = {"input_status": "ok", "response_sha256": sha256_bytes(target.response.encode("utf-8")),
                             "replay_program_sha256": sha256_bytes(json_bytes(target.replay_program))}
        outputs["hiercad"] = {"response": target.response, "program": target.replay_program}
    except Exception as exc:
        result["hiercad"] = {"input_status": "failed", "failure": failure("hiercad_render_parser_round_trip", exc)}
    else:
        try:
            replay = g.minimal_sequence(target.replay_program)
            result["hiercad"]["geometry"] = comparison(replay, reference, uid)
        except Exception as exc:
            result["hiercad"]["geometry"] = failure("hiercad_parsed_program_replay", exc)
    return result, outputs


def model_alignment(routes, route, model):
    return routes[route].get("models", {}).get(model, {}).get("geometry", {}).get("status") == "aligned"


def read_original(archive, uid):
    members = {kind: f"{AUTODESK_MEMBERS}/{uid}.{extension}" for kind, extension in (("native", "json"), ("step", "step"))}
    result = {kind: {"member": member} for kind, member in members.items()}
    payload = {}
    for kind, member in members.items():
        try:
            data = archive.read(member)  # ZIP also checks the member's CRC32.
        except KeyError:
            result[kind]["status"] = "missing_archive_member"
        except Exception as exc:
            result[kind].update(failure("archive_read", exc))
        else:
            payload[kind] = data
            result[kind].update(status="ok", sha256=sha256_bytes(data), bytes=len(data))
    return result, payload


def stage_candidate(item):
    from idea01 import text2cad_e1_geometry as g

    row, published = item
    uid = row["uid"]
    official, payload = read_original(_WORKER_ARCHIVE, uid)
    routes = {"autodesk_history": {"program_status": official["native"]["status"]},
              "cadmium_minimal": {"program_status": row["minimal_status"]}}
    route_data = {}
    ref = None
    raw = None
    if "native" in payload:
        try:
            raw = json.loads(payload["native"])
        except Exception as exc:
            official["native"].update(failure("native_json_parse", exc))
            routes["autodesk_history"]["program_status"] = "native_json_parse_failed"
    if raw is not None and "step" in payload:
        try:
            step_file = _WORKER_TMP / "reference.step"
            step_file.write_bytes(payload["step"])
            ref, frame = g.read_reference(step_file, {"properties": {"bounding_box": source_box(raw)}})
            official["step"].update(frame=frame, normalized_geometry=g.measures(ref), replay_status="ok")
        except Exception as exc:
            official["step"]["reference_failure"] = failure("step_reference", exc)
            ref = None
    if raw is not None:
        route = routes["autodesk_history"]
        try:
            features = used_features(raw)
            route["features"] = source_diagnostics(raw, features)
            native = native_sequence(raw, features)
            minimal = normalized_program(native)
        except Exception as exc:
            reason = failure("native_program_conversion", exc)
            route.update(program_status=reason["status"], failure=reason)
        else:
            route["normalized_program_sha256"] = sha256_bytes(json_bytes(minimal))
            route["models"], route_data["autodesk_history"] = evaluate_models(minimal, ref, uid, native)

    native_both = all(model_alignment(routes, "autodesk_history", model) for model in MODELS)
    if published is not None and not native_both:
        routes["cadmium_minimal"]["models"], route_data["cadmium_minimal"] = evaluate_models(published, ref, uid)
    elif published is not None:
        routes["cadmium_minimal"]["conversion_status"] = "native_jointly_aligned"

    joint = next((route for route in ROUTES if all(model_alignment(routes, route, model) for model in MODELS)), None)
    selected = {model: joint or next((route for route in ROUTES if model_alignment(routes, route, model)), None)
                for model in MODELS}
    evidence = {"uid": uid, "split": row["split"], "test_project_group": row["test_project_group"],
                "source_assets": official, "source_routes": routes, "selected_routes": selected,
                "joint_aligned_route": joint, "text_present": row["annotation_status"] == "ok"}
    chosen_data = {model: route_data[route][model] for model, route in selected.items() if route is not None}
    return evidence, chosen_data


def configure_worker(archive_path, text2cad, hiercad):
    global _WORKER_ARCHIVE, _WORKER_STAGE, _WORKER_TMP

    import torch

    from idea01 import text2cad_e1_geometry as g
    from idea01.in_domain_pilot_native import use_hiercad_repo

    torch.set_num_threads(1)
    os.environ["TEXT2CAD_SOURCE"] = str(text2cad)
    g.sequence_class()
    g.causal_mask()
    use_hiercad_repo(hiercad)
    # Detect missing geometry libraries before emitting misleading per-UID failures.
    import OCC  # noqa: F401
    from hiercad.src.utils.hierarchical_alignment import build_training_example  # noqa: F401
    from hiercad.src.utils.hierarchical_parse import parse_hierarchical_response  # noqa: F401
    from scipy.spatial import cKDTree  # noqa: F401

    _WORKER_ARCHIVE = ZipFile(archive_path)
    _WORKER_STAGE = tempfile.TemporaryDirectory(prefix="fusion-step-")
    _WORKER_TMP = Path(_WORKER_STAGE.name)


def pth_bytes(arrays):
    import torch

    from idea01 import text2cad_e1_geometry as g

    data = {"vec": {key: torch.from_numpy(arrays[key]) for key in ("cad_vec", "flag_vec", "index_vec")},
            "mask_cad_dict": {"key_padding_mask": torch.from_numpy(arrays["key_padding_mask"]),
                              "attn_mask": torch.from_numpy(g.causal_mask())}}
    output = io.BytesIO()
    torch.save(data, output)
    return output.getvalue()


def count_features(arrays):
    import numpy as np

    from idea01.in_domain_pilot import primitive_counts

    cad = arrays["cad_vec"]
    values = cad[1:][cad[1:] >= 11]
    return [*np.bincount(cad[1:, 0], minlength=267)[2:11].tolist(), *primitive_counts(cad[:, 0]),
            *np.bincount((values - 11) // 16, minlength=16).tolist()]


def write_outputs(results, out):
    from idea01.in_domain_pilot import FEATURE_NAMES

    coverage, axes, responses, features = [], [], [], []
    with ZipFile(out / "text2cad_candidates.zip", "x", compression=ZIP_DEFLATED, compresslevel=3) as z:
        for row, models in results:
            uid = row["uid"]
            member = None
            if "text2cad" in models:
                member = f"cad_seq/{uid}/seq/{uid}.pth"
                z.writestr(ZipInfo(member), pth_bytes(models["text2cad"]), compress_type=ZIP_DEFLATED, compresslevel=3)
                features.append({"uid": uid, "features": count_features(models["text2cad"])})
            if "hiercad" in models:
                responses.append({"uid": uid, **models["hiercad"]})
            axes.append({"uid": uid, "text2cad_member": member, "hiercad_input": "hiercad" in models})
            coverage.append(row)
            if len(coverage) % 100 == 0:
                print(f"Completed {len(coverage)} Fusion coverage records", flush=True)

    write_jsonl(out / "geometry_coverage.jsonl", coverage)
    write_jsonl(out / "candidate_inputs.jsonl", axes)
    write_jsonl(out / "hiercad_targets.jsonl", responses)
    write_jsonl(out / "command_count_features.jsonl", features)
    write_jsonl(out / "geometry_verified_train_intersection.jsonl", (
        {"uid": row["uid"], "route": row["joint_aligned_route"]}
        for row in coverage if row["split"] == "train" and row["joint_aligned_route"] and row["text_present"]))
    (out / "command_count_feature_names.json").write_bytes(json_bytes(FEATURE_NAMES) + b"\n")
    return coverage


def receipt(args, out, coverage):
    statistics = {}
    for split in EXPECTED_SPLITS:
        rows = [row for row in coverage if row["split"] == split]
        statistics[split] = {
            "attempted": len(rows), "frozen_denominator": EXPECTED_SPLITS[split],
            "text2cad_aligned": sum(row["selected_routes"]["text2cad"] is not None for row in rows),
            "hiercad_aligned": sum(row["selected_routes"]["hiercad"] is not None for row in rows),
            "jointly_aligned": sum(row["joint_aligned_route"] is not None for row in rows),
            "jointly_aligned_with_text": sum(bool(row["joint_aligned_route"]) and row["text_present"] for row in rows),
            "joint_input_routes": dict(Counter(row["joint_aligned_route"] or "neither" for row in rows)),
        }
    result = {"source": SOURCE, "scope": "pilot" if args.pilot_per_split else "full",
              "catalog_manifest_sha256": sha256_file(args.catalog / "manifest.json"),
              "autodesk_archive_sha256": ARCHIVE_SHA256, "text2cad_revision": TEXT2CAD_REVISION,
              "hiercad_revision": HIERCAD_REVISION, "statistics": statistics,
              "hiercad_replay": "author-rendered response and author-parsed program, strict OCCT replay",
              "text2cad_replay": "author 8-bit vector and strict OCCT replay",
              "comparison": "final Autodesk STEP, Text2CAD source-normalized frame, E1 geometry thresholds, "
                            "SciPy cKDTree and exact point-to-OCC-triangle distances",
              "file_sha256": {path.name: sha256_file(path) for path in sorted(out.iterdir())}}
    (out / "conversion_manifest.json").write_bytes(json_bytes(result) + b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--autodesk-zip", type=Path, required=True)
    parser.add_argument("--text2cad-repo", type=Path, required=True)
    parser.add_argument("--hiercad-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pilot-per-split", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1 or args.pilot_per_split < 0:
        parser.error("workers must be positive and pilot-per-split nonnegative")
    if args.autodesk_zip.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Autodesk archive size mismatch")
    require_sha(args.autodesk_zip, ARCHIVE_SHA256)
    with ZipFile(args.autodesk_zip) as archive:
        if sha256_bytes(archive.read("r1.0.1/train_test.json")) != SPLIT_SHA256:
            raise ValueError("Autodesk archive official split mismatch")
    text2cad = check_checkout(args.text2cad_repo, TEXT2CAD_REVISION)
    hiercad = check_checkout(args.hiercad_repo, HIERCAD_REVISION)
    records, programs = open_catalog(args.catalog)
    if args.pilot_per_split:
        keep = set()
        for split in EXPECTED_SPLITS:
            uids = [row["uid"] for row in records if row["split"] == split]
            keep.update(hash_order(uids, f"cadretrieval-fusion-conversion-probe-v1|{split}|")[:args.pilot_per_split])
        records = [row for row in records if row["uid"] in keep]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    init = (args.autodesk_zip, text2cad, hiercad)
    work = ((row, programs.get(row["uid"])) for row in records)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VTK_SMP_MAX_THREADS"):
        os.environ.setdefault(variable, "1")
    with tempfile.TemporaryDirectory(prefix=".fusion-conversion-", dir=args.output.parent) as stage:
        payload = Path(stage) / "output"
        payload.mkdir()
        if args.workers == 1:
            configure_worker(*init)
            coverage = write_outputs(map(stage_candidate, work), payload)
        else:
            with ProcessPoolExecutor(args.workers, initializer=configure_worker, initargs=init) as pool:
                coverage = write_outputs(pool.map(stage_candidate, work, chunksize=4), payload)
        if [row["uid"] for row in coverage] != [row["uid"] for row in records]:
            raise ValueError("Unexpected conversion coverage")
        result = receipt(args, payload, coverage)
        payload.rename(args.output)
    print(json.dumps(result["statistics"], indent=2))


if __name__ == "__main__":
    main()
