"""CPU source preflight for CadTranslator HDF5 to Text2CAD conversion.

The two subcommands allow HDF5 extraction and OCCT replay in separate runtimes.
The replay compares the published DeepCAD viewer convention and, for symmetric
extrusions, the alternative convention used by the Text2CAD raw JSON importer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VTK_SMP_MAX_THREADS"):
    os.environ[name] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEEP_CAD_SHA = "f7b350790a3bce0267a5345339fd09be0250c8cb"
CADTRANSLATOR_ZIP_SHA = "0c6c57a2266f6519bb43bc85c16d51450db38429237889faf4683043b1089170"
SOURCE_VALIDATION_SHA = "332ec7bd7e2c607480cc616daa1f60f65a4382a060436bd636405f61a1315987"
SOURCE_E1_GEOMETRY_SHA = "24c673de0fd812ff3d42e51492862ddee1d88f8d3eb9a74395ce377c665c4a59"

LINE, ARC, CIRCLE, EOS, SOL, EXTRUDE = range(6)
ACTIVE_COLUMNS = ((1, 2), (1, 2, 3, 4), (1, 2, 5), (), (), tuple(range(6, 17)))
FIELD_MASK = np.zeros((6, 17), dtype=bool)
for command, columns in enumerate(ACTIVE_COLUMNS):
    FIELD_MASK[command, list(columns)] = True
FIELD_MASK[:, 0] = True
OPERATIONS = ("NewBodyFeatureOperation", "JoinFeatureOperation",
              "CutFeatureOperation", "IntersectFeatureOperation")


@dataclass(frozen=True)
class Feature:
    loops: tuple[dict, ...]
    axes: np.ndarray
    origin: np.ndarray
    extent_one: float
    extent_two: float
    operation: int
    mode: int

    def extrusion_steps(self, symmetric_total=False):
        if self.mode == 0:
            return (self.extent_one,)
        if self.mode == 1:
            depth = self.extent_one / 2 if symmetric_total else self.extent_one
            return (depth, -depth)
        return (self.extent_one, -self.extent_two)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_digest(vec):
    return hashlib.sha256(vec.astype("<i4", copy=False).tobytes()).hexdigest()


def matrix_commands(vec):
    if vec.ndim != 2 or vec.shape[1] != 17 or not len(vec) or vec.dtype.kind != "i":
        raise ValueError("expected a nonempty signed integer matrix with 17 columns")
    commands = vec[:, 0]
    if np.any((commands < LINE) | (commands > EXTRUDE)):
        raise ValueError("unknown HDF5 command")
    fields = FIELD_MASK[commands]
    if np.any(vec[~fields] != -1):
        raise ValueError("non-padding value in an inactive command field")
    if np.any((vec[fields] < 0) | (vec[fields] > 255)):
        raise ValueError("active field outside the eight-bit range")
    ends = np.flatnonzero(commands == EOS)
    if not len(ends) or not np.all(commands[ends[0]:] == EOS):
        raise ValueError("expected terminal EOS with no later commands")
    return commands[:ends[0]]


def sketch_axes(angles):
    theta, phi, gamma = (angles / 128 - 1) * np.pi
    st, sp, sg = np.sin([theta, phi, gamma])
    ct, cp, cg = np.cos([theta, phi, gamma])
    normal = np.array([st * cp, st * sp, ct])
    reference_x = np.array([ct * cp, ct * sp, -st])
    reference_y = np.array([-sp, cp, 0])
    x_axis = cg * reference_x + sg * reference_y
    return np.stack((x_axis, np.cross(normal, x_axis), normal))


def decode_loop(rows, scale):
    if not len(rows):
        raise ValueError("empty SOL loop")
    endpoints = (rows[:, 1:3] - 128) * scale
    if len(rows) == 1 and rows[0, 0] == CIRCLE:
        radius = float(rows[0, 5] * scale)
        if radius <= 0:
            raise ValueError("zero circle radius")
        return {"circle_1": {"Center": endpoints[0].tolist(), "Radius": radius}}
    if len(rows) < 2 or np.any(rows[:, 0] > ARC):
        raise ValueError("a closed circle must occupy its own loop")
    loop = {}
    for i, row in enumerate(rows):
        start, end = endpoints[i - 1], endpoints[i]
        chord = end - start
        if not np.any(chord):
            raise ValueError("coincident curve endpoints")
        fields = {"Start Point": start.tolist(), "End Point": end.tolist()}
        if row[0] == ARC:
            if not 1 <= row[3] <= 255 or row[4] not in (0, 1):
                raise ValueError("invalid arc sweep or direction")
            left = np.array([-chord[1], chord[0]])
            mid = (start + end) / 2 + (1 - 2 * row[4]) * left * np.tan(row[3] * np.pi / 512) / 2
            fields["Mid Point"] = mid.tolist()
            name = "arc"
        else:
            name = "line"
        loop[f"{name}_{i + 1}"] = fields
    return loop


def decode_h5_vector(vec):
    commands = matrix_commands(vec)
    ends = np.flatnonzero(commands == EXTRUDE)
    if not len(ends) or ends[-1] != len(commands) - 1:
        raise ValueError("unterminated sketch or missing extrusion")
    features = []
    start = 0
    for end in ends:
        row = vec[end]
        starts = np.flatnonzero(commands[start:end] == SOL) + start
        if not len(starts) or starts[0] != start:
            raise ValueError("each extrusion must start with SOL")
        if row[12] == 0:
            raise ValueError("zero sketch scale")
        if row[15] > 3 or row[16] > 2:
            raise ValueError("invalid extrusion operation or extent type")
        # DeepCAD's 256-cell sketch grid centers the start at 128. Its native
        # scale records the size corresponding to 256/2*0.75 - 1 grid units.
        scale = row[12] / (128 * 95)
        loops = tuple(decode_loop(vec[a + 1:b], scale)
                      for a, b in zip(starts, (*starts[1:], end)))
        features.append(Feature(loops, sketch_axes(row[6:9]), row[9:12] / 128 - 1,
                                row[13] / 128 - 1, row[14] / 128 - 1,
                                int(row[15]), int(row[16])))
        start = end + 1
    return features


def profile_face(feature):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.ShapeFix import ShapeFix_Face
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Pln, gp_Pnt

    x_axis, _, normal = feature.axes
    plane = gp_Pln(gp_Ax3(gp_Pnt(*feature.origin), gp_Dir(*normal), gp_Dir(*x_axis)))

    def point(xy):
        return gp_Pnt(*(np.asarray(xy) @ feature.axes[:2] + feature.origin))

    wires = []
    for loop in feature.loops:
        wire = BRepBuilderAPI_MakeWire()
        for curve in loop.values():
            if "Radius" in curve:
                edge = BRepBuilderAPI_MakeEdge(gp_Circ(
                    gp_Ax2(point(curve["Center"]), gp_Dir(*normal)), curve["Radius"]))
            elif "Mid Point" in curve:
                arc = GC_MakeArcOfCircle(point(curve["Start Point"]), point(curve["Mid Point"]),
                                         point(curve["End Point"]))
                edge = BRepBuilderAPI_MakeEdge(arc.Value())
            else:
                edge = BRepBuilderAPI_MakeEdge(point(curve["Start Point"]), point(curve["End Point"]))
            wire.Add(edge.Edge())
        if not wire.IsDone():
            raise ValueError("HDF5 wire construction failed")
        wires.append(wire.Wire())
    face = BRepBuilderAPI_MakeFace(plane, wires[0], True)
    for hole in wires[1:]:
        face.Add(topods.Wire(hole.Reversed()))
    if not face.IsDone():
        raise ValueError("HDF5 face construction failed")
    orientation = ShapeFix_Face(face.Face())
    orientation.FixOrientation()
    return orientation.Face()


def replay_h5(features, symmetric_total=False):
    """Replay a decoded HDF5 program independently of Text2CAD's CAD builder."""
    from idea01.text2cad_e1_geometry import measures
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.gp import gp_Vec

    model = None
    boolean = (BRepAlgoAPI_Fuse, BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common)
    for feature in features:
        depths = feature.extrusion_steps(symmetric_total)
        if feature.mode == 2 and 0 in depths:
            # The published viewer constructs both prisms. OCCT refuses a zero vector.
            raise ValueError("HDF5 two-sided extrusion has a zero-depth side")
        face = profile_face(feature)
        if not any(depths):
            raise ValueError("zero signed extrusion depth")
        prisms = [BRepPrimAPI_MakePrism(face, gp_Vec(*(feature.axes[2] * depth))).Shape()
                  for depth in depths]
        feature_solid = prisms[0]
        if len(prisms) == 2:
            fusion = BRepAlgoAPI_Fuse(*prisms)
            if not fusion.IsDone():
                raise ValueError("HDF5 two-prism union failed")
            feature_solid = fusion.Shape()
        measures(feature_solid)
        if model is None:
            if feature.operation not in (0, 1):
                raise ValueError("first operation requires an existing solid")
            model = feature_solid
        else:
            operation = boolean[feature.operation](model, feature_solid)
            if not operation.IsDone():
                raise ValueError("HDF5 boolean failed")
            model = operation.Shape()
        measures(model)
    return model


def text2cad_frame(shape):
    from idea01.text2cad_e1_geometry import measures
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf

    low, high = np.asarray(measures(shape)["bbox"]).reshape(2, 3)
    scale = 0.75 / float(np.max(high - low))
    transform = gp_Trsf()
    affine = np.column_stack((np.eye(3) * scale, -low * scale))
    transform.SetValues(*affine.ravel())
    result = BRepBuilderAPI_Transform(shape, transform, True)
    return result.Shape(), low, scale


def make_minimal_program(features, origin, scale, symmetric_total=False):
    from scipy.spatial.transform import Rotation

    program = {"parts": {}}
    for i, feature in enumerate(features, 1):
        # Text2CAD uses a sketch box of recorded line endpoints, the three
        # recorded arc points, and circle cardinal extrema.
        profile_points = []
        for loop in feature.loops:
            for curve in loop.values():
                if "Radius" in curve:
                    center = np.asarray(curve["Center"])
                    radius = curve["Radius"]
                    profile_points.extend((center - radius, center + radius))
                else:
                    profile_points.extend(curve.values())
        low = np.min(profile_points, axis=0)
        size = float(np.max(np.max(profile_points, axis=0) - low) * scale)
        loops = {}
        for j, loop in enumerate(feature.loops, 1):
            loops[f"loop_{j}"] = {
                name: {key: value * scale if key == "Radius" else ((value - low) * scale).tolist()
                       for key, value in fields.items()}
                for name, fields in loop.items()
            }
        depths = (0, *feature.extrusion_steps(symmetric_total))
        plane_origin = (feature.origin + low @ feature.axes[:2] - origin) * scale
        angles = Rotation.from_matrix(feature.axes).as_euler("zyx", degrees=True)
        program["parts"][f"part_{i}"] = {
            "sketch": {"face_1": loops},
            "coordinate_system": {
                "Translation Vector": plane_origin.tolist(),
                "Euler Angles": angles.tolist(),
            },
            "extrusion": {
                "extrude_depth_towards_normal": max(depths) * scale,
                "extrude_depth_opposite_normal": -min(depths) * scale,
                "sketch_scale": size,
                "operation": OPERATIONS[feature.operation],
            },
        }
    return program


def read_h5(archive, member):
    import h5py

    with h5py.File(io.BytesIO(archive.read(member)), "r") as handle:
        if set(handle) != {"vec"}:
            raise ValueError(f"unexpected HDF5 datasets in {member}")
        return handle["vec"][:]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def extract_source(cadtranslator_zip, development, output):
    if file_digest(cadtranslator_zip) != CADTRANSLATOR_ZIP_SHA:
        raise ValueError("CadTranslator ZIP digest differs from the benchmark inventory")
    if file_digest(development) != SOURCE_VALIDATION_SHA:
        raise ValueError("source selection differs from the frozen E1 development list")
    records = []
    with ZipFile(cadtranslator_zip) as archive:
        for row in read_jsonl(development):
            if row["domain"] != "deepcad" or row["published_split"] != "validation":
                raise ValueError("source preflight requires only DeepCAD validation UIDs")
            member = f"dataset/deepcad_txt/cad_vec_original/{row['uid']}.h5"
            vec = read_h5(archive, member)
            records.append({
                "uid": row["uid"], "h5_member": member, "h5_dtype": str(vec.dtype),
                "h5_sha256": vector_digest(vec), "vec": vec.tolist(),
                "official_member": row["official_member"], "minimal": row["minimal"],
            })
    write_json(output, {"cadtranslator_zip_sha256": CADTRANSLATOR_ZIP_SHA,
                        "source_selection_sha256": SOURCE_VALIDATION_SHA, "records": records})


def check_variant(features, official_arrays, official_shape, uid, symmetric_total):
    from idea01 import text2cad_e1_geometry as geometry
    from prepare_text2cad_e1_geometry import array_diff

    result = {}
    stage = "h5_replay"
    try:
        raw = replay_h5(features, symmetric_total)
        reference, origin, scale = text2cad_frame(raw)
        result["h5_reference_frame"] = {"bbox_min": origin.tolist(), "scale": scale}
        stage = "native_geometry_comparison"
        result["h5_to_official"] = geometry.compare_shapes(reference, official_shape, uid)
        stage = "minimal_conversion"
        minimal = make_minimal_program(features, origin, scale, symmetric_total)
        float_seq = geometry.minimal_sequence(minimal)
        float_shape = geometry.strict_solid(float_seq)
        probes = geometry.cut_probes(float_seq)
        result["float_to_h5"] = geometry.compare_shapes(float_shape, reference, uid, probes)
        stage = "text2cad_quantization"
        arrays = geometry.encode_minimal(minimal)
        result.update(geometry.fingerprints(arrays))
        result["official_array_diff"] = array_diff(arrays, official_arrays)
        result["n_valid_tokens"] = int(np.count_nonzero(arrays["cad_vec"][:, 0]))
        stage = "quantized_replay"
        shape = geometry.strict_solid(geometry.decode(arrays["cad_vec"]))
        stage = "converted_geometry_comparison"
        result["quantized_to_h5"] = geometry.compare_shapes(shape, reference, uid, probes)
        result["quantized_to_official"] = geometry.compare_shapes(shape, official_shape, uid, probes)
    except Exception as error:
        result["failure"] = {"stage": stage, "exception": type(error).__name__, "reason": str(error)}
    return result


def check_source(source_vectors, e1_prep, output):
    from idea01 import text2cad_e1_geometry as geometry
    from prepare_text2cad_e1_geometry import official_arrays

    upstream = Path(os.environ.get("TEXT2CAD_SOURCE", str(geometry.DEFAULT_SOURCE)))
    sha = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if sha != geometry.UPSTREAM_SHA:
        raise ValueError("expected the pinned Text2CAD source revision")
    subprocess.run(["git", "-C", str(upstream), "diff", "--quiet", sha, "--", "CadSeqProc"], check=True)
    ledger = e1_prep / "geometry_development.jsonl"
    if file_digest(ledger) != SOURCE_E1_GEOMETRY_SHA:
        raise ValueError("expected the independently recorded E1 source geometry ledger")
    expected = {row["uid"]: row for row in read_jsonl(ledger)}
    inputs = json.loads(source_vectors.read_text(encoding="utf-8"))
    if (inputs["cadtranslator_zip_sha256"] != CADTRANSLATOR_ZIP_SHA
            or inputs["source_selection_sha256"] != SOURCE_VALIDATION_SHA):
        raise ValueError("source vectors are unbound")
    if sorted(record["uid"] for record in inputs["records"]) != sorted(expected):
        raise ValueError("source vector UIDs differ from the validation ledger")
    started = time.monotonic()
    geometry.sequence_class()
    records = []
    for index, row in enumerate(inputs["records"], 1):
        uid = row["uid"]
        path = e1_prep / "references" / "official" / f"{uid.replace('/', '_')}.pth.gz"
        vec = np.asarray(row["vec"], dtype=np.int64)
        if vector_digest(vec) != row["h5_sha256"]:
            raise ValueError(f"HDF5 vector digest differs for {uid}")
        result = {k: row[k] for k in ("uid", "h5_member", "h5_sha256", "official_member")}
        result["h5_extrusion_fields"] = vec[vec[:, 0] == EXTRUDE, 6:].tolist()
        result["text2cad_float_extrusions"] = [
            part["extrusion"] for part in row["minimal"]["parts"].values()
        ]
        records.append(result)
        stage = "source_h5_decode"
        try:
            features = decode_h5_vector(vec)
            result["extent_modes"] = [feature.mode for feature in features]
            result["operations"] = [feature.operation for feature in features]
            stage = "official_tensor"
            if row["official_member"] != expected[uid]["reference_member"]:
                raise ValueError("official member differs from the E1 ledger")
            official_sha = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
            if official_sha != expected[uid]["reference_file_sha256"]:
                raise ValueError("official tensor digest differs from the E1 ledger")
            arrays = official_arrays(path)
            result["official_tensor_sha256"] = official_sha
            result["official_cad_vec_sha256"] = geometry.fingerprints(arrays)["cad_vec_sha256"]
            result["official_extents_q8"] = arrays["cad_vec"][arrays["flag_vec"] == 1, 0].reshape(-1, 2).tolist()
            stage = "official_replay"
            shape = geometry.strict_solid(geometry.decode(arrays["cad_vec"]))
            stage = "variant_comparison"
            result["viewer_convention"] = check_variant(features, arrays, shape, uid, False)
            if 1 in result["extent_modes"]:
                result["symmetric_total_diagnostic"] = check_variant(features, arrays, shape, uid, True)
        except Exception as error:
            result["failure"] = {"stage": stage, "exception": type(error).__name__, "reason": str(error)}
        if index % 8 == 0:
            print(f"Checked {index}/{len(expected)} source UIDs", flush=True)
    summary = Counter()
    for row in records:
        if "failure" in row:
            summary[row["failure"]["stage"]] += 1
            continue
        variant = row["viewer_convention"]
        if "failure" in variant:
            summary[variant["failure"]["stage"]] += 1
        for key in ("h5_to_official", "float_to_h5", "quantized_to_h5", "quantized_to_official"):
            if key in variant:
                summary[f"{key}_{'pass' if variant[key]['pass'] else 'fail'}"] += 1
        if "official_array_diff" in variant:
            exact = all(diff["different"] == 0 for diff in variant["official_array_diff"].values())
            summary[f"exact_tensor_{'match' if exact else 'mismatch'}"] += 1
    receipt = {"stage": "source_validation_only", "source_vectors_sha256": file_digest(source_vectors),
               "e1_source_geometry_sha256": SOURCE_E1_GEOMETRY_SHA, "text2cad_git_sha": sha,
               "deepcad_viewer_git_sha": DEEP_CAD_SHA, "python": sys.version, "numpy": np.__version__,
               "n_source_uids": len(records), "elapsed_seconds": time.monotonic() - started,
               "summary": summary, "records": records}
    write_json(output, receipt)
    print(json.dumps({"n_source_uids": len(records), "summary": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-source")
    extract.add_argument("--archive", required=True, type=Path)
    extract.add_argument("--development", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    check = commands.add_parser("check-source")
    check.add_argument("--vectors", required=True, type=Path)
    check.add_argument("--e1-prep", required=True, type=Path)
    check.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "extract-source":
        extract_source(args.archive, args.development, args.out)
    else:
        check_source(args.vectors, args.e1_prep, args.out)


if __name__ == "__main__":
    main()
