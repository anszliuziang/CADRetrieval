"""Development-only §5 geometry and weight-free representation diagnostics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


TEXT2CAD_SHA = "669a81472ede2269837c22a4d00070eaac360c82"
FAMILIES = {
    "rectangular_notch_plate": ("rectangular plate with a rectangular edge notch", 2),
    "v_notch_flange": ("chamfered plate with a V-shaped edge notch", 1),
}
ATTRIBUTES = {"size_location": ("larger", "smaller"), "shape_location": ("round", "oblong")}
MODELS = ("text2cad", "hiercad", "command_dual_tower", "spectrabrep")
HISTORIES = ("A0", "A1", "B0", "B1")
HISTORY_LINKS = (("A0", "A1"), ("B0", "B1"))
ENTITY_LINKS = tuple(product(HISTORIES[:2], HISTORIES[2:]))
FIRST_CUT_NEAR = (True, False, False, True, True, False, False, True, False, True)
A_MARKED_NEAR = (True, True, False, False, False, True, True, False, False, True)


@dataclass(frozen=True)
class Hole:
    center: tuple[int, int]
    radius: int
    half_straight: int = 0


@dataclass(frozen=True)
class Design:
    family: str
    attribute: str
    index: int
    outline: tuple[tuple[int, int], ...]
    thickness: int
    quarter_turns: int
    near_center: tuple[int, int]
    far_center: tuple[int, int]
    sizes: tuple[int, int]
    slot_half_straight: int

    @property
    def pair_id(self) -> str:
        return f"{self.family}-{self.attribute}-{self.index:02d}"

    def world(self, point):
        x, y = point
        return ((x, y), (-y, x), (-x, -y), (y, -x))[self.quarter_turns]

    def holes(self, entity: str) -> tuple[Hole, Hole]:
        first, second = self.sizes
        if self.attribute == "size_location":
            forms = ((first, 0), (second, 0))
        else:
            forms = ((second, 0), (second, self.slot_half_straight))
        if (entity == "A") != A_MARKED_NEAR[self.index]:
            forms = forms[::-1]
        return tuple(Hole(center, *shape) for center, shape in zip((self.near_center, self.far_center), forms, strict=True))

    def expected_words(self, entity: str) -> tuple[str, str]:
        words = ATTRIBUTES[self.attribute]
        return words if (entity == "A") == A_MARKED_NEAR[self.index] else words[::-1]


def pilot_designs():
    """A fixed grid of ten dimensioned variants of each family/attribute."""
    for family_number, family in enumerate(FAMILIES):
        for attribute_number, attribute in enumerate(ATTRIBUTES):
            for index in range(10):
                width = 120 + 2 * index
                height = 73 + 2 * (index % 5)
                notch_depth = 16 + index % 4
                notch_half = 10 + index % 3
                left = (index + attribute_number + family_number) % 2 == 0
                near_x = round((0.27 if left else 0.73) * width)
                far_x = width - near_x
                notch_x = near_x + (index % 3 - 1) * 2
                right_mouth = (notch_x + notch_half, height)
                left_mouth = (notch_x - notch_half, height)
                if family == "rectangular_notch_plate":
                    outline = (
                        (0, 0), (width, 0), (width, height), right_mouth,
                        (right_mouth[0], height - notch_depth),
                        (left_mouth[0], height - notch_depth), left_mouth, (0, height),
                    )
                else:
                    bevel = 12 + index % 3
                    outline = (
                        (0, 0), (width - bevel, 0), (width, bevel),
                        (width, height - bevel), (width - bevel, height), right_mouth,
                        (notch_x, height - notch_depth), left_mouth, (0, height),
                    )
                yield Design(
                    family, attribute, index, outline, 10 + index % 4,
                    (index + 2 * family_number + attribute_number) % 4,
                    (near_x, round(0.43 * height) + index % 3),
                    (far_x, round(0.44 * height) - index % 3),
                    (10 + index % 3, 5 + index % 3), 9 + index % 2,
                )


def opposite_captions(design: Design) -> dict[str, dict[str, str]]:
    part = FAMILIES[design.family][0]
    captions = {}
    for entity in ("A", "B"):
        near, far = design.expected_words(entity)
        captions[entity] = {
            "near_first": f"A {part} has two through holes. The nearer hole is {near} and the farther hole is {far}.",
            "far_first": f"A {part} has two through holes. The farther hole is {far} and the nearer hole is {near}.",
            "near_first_rephrased": (
                f"On a {part}, the through hole closer to the notch is {near}. "
                f"The through hole farther from the notch is {far}."
            ),
            "far_first_rephrased": (
                f"On a {part}, the through hole farther from the notch is {far}. "
                f"The through hole closer to the notch is {near}."
            ),
        }
    for key, text in captions["A"].items():
        other = captions["B"][key]
        if text == other or Counter(re.findall(r"[a-z0-9]+", text.lower())) != Counter(re.findall(r"[a-z0-9]+", other.lower())):
            raise ValueError("opposite captions must differ while preserving word counts")
    return captions


def _xyz(point, z=0) -> dict:
    return dict(zip("xyz", (*point, z), strict=True))


def _line(start, end) -> dict:
    return {"type": "Line3D", "start_point": _xyz(start), "end_point": _xyz(end)}


def _hole_curves(design: Design, hole: Hole) -> list[dict]:
    x, y = hole.center
    radius = hole.radius
    half = hole.half_straight
    if not half:
        return [{
            "type": "Circle3D", "center_point": _xyz(design.world(hole.center)),
            "radius": radius, "normal": _xyz((0, 0), 1),
        }]
    left, right = x - half, x + half
    bl, br, tr, tl = (design.world(p) for p in ((left, y - radius), (right, y - radius),
                                                 (right, y + radius), (left, y + radius)))

    def semicircle(start, end, center, start_vector):
        return {
            "type": "Arc3D", "start_point": _xyz(start), "end_point": _xyz(end),
            "center_point": _xyz(design.world(center)), "radius": radius,
            "normal": _xyz((0, 0), 1), "start_angle": 0, "end_angle": math.pi,
            "reference_vector": _xyz(design.world(start_vector)),
        }

    return [_line(bl, br), semicircle(br, tr, (right, y), (0, -1)),
            _line(tr, tl), semicircle(tl, bl, (left, y), (0, 1))]


def deepcad_history(design: Design, history: str) -> tuple[dict, tuple[str, str]]:
    """Use a fixed stock and opposite orders of two independent through-cuts."""
    if history not in HISTORIES:
        raise ValueError(f"unknown history {history!r}")
    outline = tuple(design.world(p) for p in design.outline)
    stock_curves = [_line(a, b) for a, b in zip(outline, outline[1:] + outline[:1], strict=True)]
    cuts = list(zip(("near", "far"), design.holes(history[0]), strict=True))
    if FIRST_CUT_NEAR[design.index] != (history[1] == "0"):
        cuts.reverse()
    features = [(stock_curves, 0, design.thickness, "NewBodyFeatureOperation")]
    features.extend((_hole_curves(design, hole), -4, design.thickness + 8, "CutFeatureOperation") for _, hole in cuts)
    entities = {}
    sequence = []
    for index, (curves, z, depth, operation) in enumerate(features):
        sketch, extrusion = f"sketch_{index + 1}", f"extrude_{index + 1}"
        entities[sketch] = {
            "type": "Sketch",
            "transform": {"origin": _xyz((0, 0), z), "x_axis": _xyz((1, 0)),
                          "y_axis": _xyz((0, 1)), "z_axis": _xyz((0, 0), 1)},
            "profiles": {"profile_1": {"loops": [{"is_outer": True, "profile_curves": curves}]}},
        }
        entities[extrusion] = {
            "type": "ExtrudeFeature", "profiles": [{"sketch": sketch, "profile": "profile_1"}],
            "start_extent": {"type": "ProfilePlaneStartDefinition"},
            "extent_type": "OneSideFeatureExtentType", "extent_one": {"distance": {"value": depth}},
            "operation": operation,
        }
        sequence.extend(({"type": "Sketch", "entity": sketch}, {"type": "ExtrudeFeature", "entity": extrusion}))
    xy = np.asarray(outline)
    bounds = {"min_point": _xyz(tuple(map(int, xy.min(axis=0)))),
              "max_point": _xyz(tuple(map(int, xy.max(axis=0))), design.thickness)}
    return {"sequence": sequence, "entities": entities, "properties": {"bounding_box": bounds}}, tuple(k for k, _ in cuts)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def json_bytes(value) -> bytes:
    """Preserve insertion order, including the order of native parts and curves."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                      default=_json_default).encode("utf-8")


def fingerprint(value) -> str:
    return sha256(json_bytes(value)).hexdigest()


def jsonl_write(stream, row) -> None:
    stream.write(json_bytes(row) + b"\n")


def activate_text2cad(root: Path):
    """Load the actual CadSeqProc implementation, bound to upstream Git blobs."""
    blobs = {
        "cad_sequence.py": "bffe93c540513788c000b7318ea9395dbe97c560",
        "geometry/arc.py": "ab84ffe7f691f705c341ff40c80d90af46c63ff0",
        "geometry/circle.py": "48ecead6909e2d502cd36050a9173811ce7dbc0c",
        "geometry/curve.py": "35cb61179f7a86205e35d469ddb16d4a356e692a",
        "geometry/line.py": "b17119efc19da9d0dae75aff611c3447b3950b3a",
        "sequence/sketch/coord_system.py": "e794b91605f51f307baeb1c679196877249cea39",
        "sequence/sketch/face.py": "1c491644fdc7aadb5fd4e8e95c4bb065dc5d6c15",
        "sequence/sketch/loop.py": "def5e82b5af5581c478cad644d4ad392c0538979",
        "sequence/sketch/sketchsequence.py": "159cfb6ab0fffea8a0b6dcdd151bdc039e84b6a6",
        "sequence/transformation/extrude_sequence.py": "a0a4881a1cc14002ddc59ce80e738a8d1f4197c6",
        "utility/macro.py": "6645cba995907771313d3c2ebcc2b403867b2bcc",
        "utility/utils.py": "38da79b64d2ba5ea0b4bb7c3ff1564bcd01fc84e",
    }
    from hashlib import sha1

    root = root.resolve()
    for relative, expected in blobs.items():
        data = (root / "CadSeqProc" / relative).read_bytes().replace(b"\r\n", b"\n")
        actual = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise ValueError(f"CadSeqProc Git blob mismatch at {relative}")
    sys.path.insert(0, str(root))
    from CadSeqProc.cad_sequence import CADSequence

    return CADSequence, {"revision": TEXT2CAD_SHA, "git_blobs": blobs}


def activate_spectrabrep(root: Path):
    """Use the source tree verified by the prior author descriptor cache."""
    from .in_domain_pilot_retrievers import (
        SPECTRA_COMMIT, SPECTRA_SOURCE_SHA256, load_spectra_modules, verify_spectra_source,
    )

    source = verify_spectra_source(root.resolve())
    geometry = load_spectra_modules(source, ("config", "geometry"))["geometry"]
    eigsh = geometry.eigsh
    solver_errors = []

    def audit_eigsh(*args, **kwargs):
        try:
            return eigsh(*args, **kwargs)
        except Exception as exc:
            solver_errors.append(type(exc).__name__)
            raise

    geometry.eigsh = audit_eigsh
    return (geometry, solver_errors), {"revision": SPECTRA_COMMIT, "tree_sha256": SPECTRA_SOURCE_SHA256}


def normalized_minimal(sequence_type, raw: dict) -> dict:
    """Use the author export and retain only replayable geometry fields."""
    program = sequence_type.json_to_NormalizedCAD(raw, bit=8)._json()
    normalized = {"parts": {name: {key: part[key] for key in ("coordinate_system", "sketch", "extrusion")}
                            for name, part in program["parts"].items()}}
    return json.loads(json_bytes(normalized))


def minimal_replay_sequence(sequence_type, minimal: dict):
    """Restore the circle cardinal points used by the f75eed8 minimal reader."""
    seq = sequence_type.from_minimal_json(minimal)
    for sketch in seq.sketch_seq:
        sketch.coordsystem.metadata["euler_angles"] = np.deg2rad(sketch.coordsystem.metadata["euler_angles"])
        for curve in sketch.all_curves:
            if curve.curve_type == "circle":
                m = curve.metadata
                for index, direction in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0)), 1):
                    m[f"pt{index}"] = m["center"] + m["radius"] * np.asarray(direction)
    return seq


@lru_cache(maxsize=1)
def _text2cad_causal_mask(size):
    from CadSeqProc.utility.utils import generate_attention_mask

    causal = generate_attention_mask(size)
    return causal, sha256(causal.numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def text2cad_program(sequence_type, raw: dict):
    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT

    from .text2cad_port.model import build_teacher_forcing_inputs
    from .text2cad_vector_scorer import validate_cad_vec

    _, cad, flags, indices = sequence_type.json_to_vec(raw, bit=N_BIT, padding=False)
    if len(cad) > MAX_CAD_SEQUENCE_LENGTH:
        raise ValueError(f"cad_sequence_over_limit:{len(cad)}")
    # The author stores these padding values in json2vec.py and CADSequence.to_vec.
    pads = MAX_CAD_SEQUENCE_LENGTH - len(cad)
    if pads:
        import torch

        cad = torch.cat((cad, cad.new_zeros((pads, 2))))
        flags = torch.cat((flags, flags.new_full((pads,), 11)))
        indices = torch.cat((indices, indices.new_full((pads,), int(indices[-1]) + 1)))
    pad_mask = cad == 0
    diag = validate_cad_vec(cad, pad_mask, flags, indices)
    vectors = {"cad_vec": cad, "flag_vec": flags, "index_vec": indices}
    causal, causal_hash = _text2cad_causal_mask(MAX_CAD_SEQUENCE_LENGTH - 1)
    _, _, target, shifted_mask = build_teacher_forcing_inputs(
        {name: value[None, ...] for name, value in vectors.items()},
        {"key_padding_mask": pad_mask[None, ...], "attn_mask": causal},
    )
    target = target[0, :diag.n_real_tokens - 1]
    mask = shifted_mask[0, :diag.n_real_tokens - 1]
    effective = {"target_rows": target.tolist(), "valid_fields": (~mask).tolist()}
    return {**vectors, "key_padding_mask": pad_mask}, {
        "status": "ok", "effective_target": effective, "input_sha256": fingerprint(effective),
        "padded_program_sha256": fingerprint({key: array.tolist() for key, array in vectors.items()}),
        "fixed_causal_mask_sha256": causal_hash,
        "real_rows": diag.n_real_tokens, "valid_fields": int((~mask).sum()), "truncated": False,
        "program_rows": cad[:diag.n_real_tokens].tolist(),
        "flag_vec": flags[:diag.n_real_tokens].tolist(), "index_vec": indices[:diag.n_real_tokens].tolist(),
    }


def text2cad_tokenizer(root: Path):
    from transformers import BertTokenizer

    from .text2cad_port.loader import PINNED_BERT_FILE_SHA256

    files = {}
    for name in ("vocab.txt", "tokenizer_config.json", "tokenizer.json", "config.json"):
        path = root / name
        if name != "vocab.txt" and not path.exists():
            continue
        files[name] = sha256(path.read_bytes()).hexdigest()
        if files[name] != PINNED_BERT_FILE_SHA256[name]:
            raise ValueError(f"BERT {name} differs from the pinned Text2CAD tokenizer")
    tokenizer = BertTokenizer.from_pretrained(str(root), local_files_only=True, do_lower_case=True)
    return tokenizer, {"file_sha256": files, "context": 512, "name": "google-bert/bert-large-uncased"}


def text2cad_captions(tokenizer, captions):
    records = {entity: {} for entity in captions}
    for variant in captions["A"]:
        texts = [captions[entity][variant] for entity in captions]
        full = tokenizer(texts, truncation=False)
        actual = tokenizer(texts, max_length=512, truncation=True, padding=True)
        for index, entity in enumerate(captions):
            active = int(sum(actual["attention_mask"][index]))
            effective = {key: values[index][:active] for key, values in actual.items()}
            records[entity][variant] = {
                "status": "ok", "input_sha256": fingerprint(effective), "effective_input": effective,
                "untruncated_tokens": len(full["input_ids"][index]), "effective_tokens": active,
                "truncated": len(full["input_ids"][index]) > active,
            }
    return records


def dual_tower_program(arrays):
    from .in_domain_pilot_retrievers import effective_command_input

    effective = effective_command_input(arrays).tolist()
    return {"status": "ok", "input_sha256": fingerprint(effective), "effective_input": effective,
            "real_rows": len(effective), "truncated": False}


def spectra_program(implementation, arrays):
    geometry, solver_errors = implementation
    solver_errors.clear()
    cad, index = (arrays[key].numpy() for key in ("cad_vec", "index_vec"))
    vector = geometry.descriptor_vector(geometry.describe_sequence(cad, index))
    if vector.shape != (280,) or vector.dtype != np.float32 or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError("invalid or empty SpectraBRep descriptor")
    fallback_types = dict(Counter(solver_errors))
    solver_errors.clear()
    repeated = geometry.descriptor_vector(geometry.describe_sequence(cad, index))
    if not np.isfinite(repeated).all():
        raise ValueError("nonfinite repeated SpectraBRep descriptor")
    vector_hash = sha256(vector.astype("<f4", copy=False).tobytes()).hexdigest()
    repeat_hash = sha256(repeated.astype("<f4", copy=False).tobytes()).hexdigest()
    return {"status": "ok" if vector_hash == repeat_hash else "unstable_input",
            "input_sha256": vector_hash, "repeat_input_sha256": repeat_hash,
            "repeat_max_abs_difference": float(np.max(np.abs(repeated - vector))),
            "solver_fallbacks": fallback_types, "repeat_solver_fallbacks": dict(Counter(solver_errors)),
            "effective_input": vector.tolist(), "dimension": 280, "dtype": "float32", "truncated": False}


def _volume(shape) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    if shape.IsNull():
        return 0.0
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = float(props.Mass())
    if not math.isfinite(volume) or volume < -1e-10:
        raise ValueError("invalid solid volume")
    return max(0.0, volume)


def _solid_volume(shape) -> float:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Extend.TopologyUtils import TopologyExplorer

    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("invalid replayed B-Rep")
    volume = _volume(shape)
    if TopologyExplorer(shape).number_of_solids() != 1 or volume <= 0:
        raise ValueError("expected one positive-volume solid")
    return volume


def _bounds(shape) -> tuple[float, ...]:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    bounds = Bnd_Box()
    brepbndlib.AddOptimal(shape, bounds, False, False)
    return tuple(bounds.Get())


def _cut(left, right, *, fuzzy=None):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

    operation = BRepAlgoAPI_Cut(left, right)
    if fuzzy is not None:
        operation.SetFuzzyValue(fuzzy)
        operation.Build()
    if not operation.IsDone():
        raise ValueError("OpenCASCADE cut failed")
    return operation.Shape()


def strict_replay(sequence):
    """Replay all three author sketch bodies with checked OpenCASCADE booleans."""
    from CadSeqProc.utility.macro import PRECISION

    shape = None
    step_volumes = []
    for index, (sketch, extrude) in enumerate(zip(sequence.sketch_seq, sequence.extrude_seq, strict=True)):
        meta = extrude.metadata
        if meta["boolean"] != (0 if index == 0 else 2):
            raise ValueError("expected creation of stock and two cut extrusions")
        body = sketch.build_body({"extrude_values": [meta["extent_one"], meta["extent_two"]]})
        _solid_volume(body)
        shape = body if shape is None else _cut(shape, body, fuzzy=PRECISION)
        volume = _solid_volume(shape)
        if step_volumes and not volume < step_volumes[-1] * (1 - 1e-6):
            raise ValueError(f"cut {index} failed to remove positive material volume")
        step_volumes.append(volume)
    if len(step_volumes) != 3:
        raise ValueError(f"incomplete replay of {len(step_volumes)}/3 features")
    return shape, step_volumes


def _cross2(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _measure_notch(wire, reflex_count: int):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Line
    from OCC.Extend.TopologyUtils import WireExplorer

    explorer = WireExplorer(wire)
    edges = list(explorer.ordered_edges())
    if any(BRepAdaptor_Curve(edge).GetType() != GeomAbs_Line for edge in edges):
        raise ValueError("expected a straight-edge outer perimeter")
    points = [BRep_Tool.Pnt(vertex) for vertex in explorer.ordered_vertices()]
    xy = np.asarray([(point.X(), point.Y()) for point in points])
    if len(xy) != len(edges) or len(xy) < 5:
        raise ValueError("incomplete outer-wire traversal")
    twice_area = sum(_cross2(a, b) for a, b in zip(xy, np.roll(xy, -1, axis=0), strict=True))
    if abs(twice_area) <= 1e-10:
        raise ValueError("degenerate outer perimeter")
    winding = math.copysign(1, twice_area)
    reflex = [point for i, point in enumerate(xy)
              if winding * _cross2(point - xy[i - 1], xy[(i + 1) % len(xy)] - point) < -1e-8 * abs(twice_area)]
    if len(reflex) != reflex_count:
        raise ValueError(f"found {len(reflex)} notch vertices, expected {reflex_count}")
    return np.mean(reflex, axis=0), np.asarray(reflex).tolist()


def _measure_hole(wire) -> dict:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line
    from OCC.Core.GProp import GProp_GProps
    from OCC.Extend.TopologyUtils import TopologyExplorer

    curves = [BRepAdaptor_Curve(edge) for edge in TopologyExplorer(wire).edges()]
    circles = [curve for curve in curves if curve.GetType() == GeomAbs_Circle]
    line_count = sum(curve.GetType() == GeomAbs_Line for curve in curves)
    radii = [float(curve.Circle().Radius()) for curve in circles]
    sweeps = [float(abs(curve.LastParameter() - curve.FirstParameter())) for curve in circles]
    bounds = _bounds(wire)
    short, long = sorted((bounds[3] - bounds[0], bounds[4] - bounds[1]))
    if short <= 0:
        raise ValueError("invalid hole dimensions")
    if len(curves) == len(circles) == 1 and abs(sweeps[0] - 2 * math.pi) <= 1e-5 and long / short < 1.01:
        kind = "round"
    elif (len(curves) == 4 and len(circles) == line_count == 2 and long / short >= 1.5
          and all(abs(sweep - math.pi) < 0.08 for sweep in sweeps) and max(radii) < 1.05 * min(radii)):
        kind = "oblong"
    else:
        raise ValueError(f"unrecognized through-hole profile: {len(circles)} arcs, {line_count} lines, "
                         f"sweeps={sweeps}, radii={radii}, extents={(short, long)}")
    face_builder = BRepBuilderAPI_MakeFace(wire, True)
    if not face_builder.IsDone():
        raise ValueError("cannot measure the planar hole profile")
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face_builder.Face(), props)
    center = props.CentreOfMass()
    area = abs(float(props.Mass()))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("invalid hole area")
    return {"kind": kind, "center_xy": [center.X(), center.Y()], "area": area,
            "arc_radii": radii, "arc_sweeps_radians": sweeps, "line_count": line_count,
            "diameter_or_end_cap_diameters": [2 * radius for radius in radii],
            "short_length": short, "long_length": long}


def measure_binding(shape, design: Design, entity: str) -> dict:
    """Measure the actual solid, with no authored centers or cutter data as input."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_OUT
    from OCC.Core.gp import gp_Pnt
    from OCC.Extend.TopologyUtils import TopologyExplorer

    bounds = _bounds(shape)
    volume = _solid_volume(shape)
    z_bottom, z_top = bounds[2], bounds[5]
    horizontal = []
    for face in TopologyExplorer(shape).faces():
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Plane:
            plane = surface.Plane()
            if abs(plane.Axis().Direction().Z()) >= 1 - 1e-8:
                horizontal.append((plane.Location().Z(), face))
    openings = []
    for z in (z_bottom, z_top):
        plane_faces = [face for position, face in horizontal if abs(position - z) <= 1e-6]
        if len(plane_faces) != 1:
            raise ValueError(f"expected a single planar outer face at z={z}")
        face = plane_faces[0]
        outer = breptools.OuterWire(face)
        holes = [_measure_hole(wire) for wire in TopologyExplorer(face).wires() if not wire.IsSame(outer)]
        if len(holes) != 2:
            raise ValueError(f"expected two closed hole openings at z={z}, found {len(holes)}")
        openings.append((outer, sorted(holes, key=lambda item: item["center_xy"])))

    (_, bottom), (top_wire, top) = openings
    notch, reflex = _measure_notch(top_wire, FAMILIES[design.family][1])
    for hole, other in zip(top, bottom, strict=True):
        if (hole["kind"] != other["kind"] or not math.isclose(hole["area"], other["area"], rel_tol=1e-6)
                or math.dist(hole["center_xy"], other["center_xy"]) > 1e-6):
            raise ValueError("bottom and top hole sections disagree")
        x, y = hole["center_xy"]
        samples = [z_bottom + fraction * (z_top - z_bottom) for fraction in (0.25, 0.5, 0.75)]
        hole["open_center_samples"] = [float(z) for z in samples
                                      if BRepClass3d_SolidClassifier(shape, gp_Pnt(x, y, z), 1e-7).State() == TopAbs_OUT]
        if len(hole["open_center_samples"]) != len(samples):
            raise ValueError("hole center is blocked through the solid")
        hole["notch_distance"] = math.dist(hole["center_xy"], notch)

    near, far = sorted(top, key=lambda hole: hole["notch_distance"])
    if far["notch_distance"] <= near["notch_distance"] * 1.10:
        raise ValueError("hole center distances to the notch are insufficiently separated")
    if design.attribute == "size_location":
        if near["kind"] != "round" or far["kind"] != "round":
            raise ValueError("size binding requires two circular holes")
        near_radius, far_radius = near["arc_radii"][0], far["arc_radii"][0]
        if max(near_radius, far_radius) <= 1.20 * min(near_radius, far_radius):
            raise ValueError("hole diameters are insufficiently separated")
        measured = ("larger", "smaller") if near_radius > far_radius else ("smaller", "larger")
    else:
        measured = (near["kind"], far["kind"])
    return {"status": "ok" if measured == design.expected_words(entity) else "binding_mismatch",
            "volume": volume, "bounds": bounds, "notch_anchor_xy": notch.tolist(), "notch_reflex_xy": reflex,
            "near": near, "far": far, "bottom_openings": bottom, "measured_binding": measured,
            "expected_binding": design.expected_words(entity)}


def geometry_links(records: dict, shapes: dict) -> dict:
    """Compare actual bodies with bilateral Boolean differences, including all A/B crosses."""
    results = {}
    for label, links in (("histories", HISTORY_LINKS), ("opposite_entities", ENTITY_LINKS)):
        comparisons = []
        for left, right in links:
            item = {"histories": [left, right]}
            if any(records[name]["status"] != "ok" for name in (left, right)):
                item["status"] = "geometry_unverified"
                item["replay_statuses"] = [records[name]["status"] for name in (left, right)]
            else:
                try:
                    from OCC.Core.BRepCheck import BRepCheck_Analyzer

                    differences = [_cut(shapes[a], shapes[b]) for a, b in ((left, right), (right, left))]
                    if any(not result.IsNull() and not BRepCheck_Analyzer(result).IsValid() for result in differences):
                        raise ValueError("invalid Boolean difference B-Rep")
                    volumes = [_volume(result) for result in differences]
                    reference = max(records[name]["volume"] for name in (left, right))
                    ratio = sum(volumes) / reference
                    expected = ratio <= 2e-6 if label == "histories" else ratio >= 1e-3
                    item.update(status="verified" if expected else "geometry_mismatch",
                                difference_volumes=volumes, relative_symmetric_difference=ratio)
                except Exception as exc:
                    item.update(status="comparison_failed", error=f"{type(exc).__name__}: {exc}")
            comparisons.append(item)
        results[label] = comparisons
    results["status"] = "verified" if all(
        row["status"] == "verified" for label in ("histories", "opposite_entities") for row in results[label]
    ) else "geometry_unverified"
    return results


def input_link(left: dict, right: dict) -> dict:
    states = [left["status"], right["status"]]
    if states != ["ok", "ok"]:
        status = next((key for key in ("overlength", "unavailable") if key in states), "input_failed")
        return {"status": status, "input_statuses": states}
    if left.get("truncated") or right.get("truncated"):
        status = "truncated"
    else:
        status = "collision" if left["input_sha256"] == right["input_sha256"] else "distinct"
    info = {"status": status, "input_sha256": [left["input_sha256"], right["input_sha256"]]}
    if left.get("dtype") == "float32":
        difference = np.asarray(left["effective_input"]) - np.asarray(right["effective_input"])
        info["max_abs_difference"] = float(np.max(np.abs(difference)))
        info["exact_value_equality"] = bool(np.all(difference == 0))
    return info
