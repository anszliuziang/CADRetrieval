"""Pinned Text2CAD encoder and independent geometry checks from E1, cf959f3."""
from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

UPSTREAM_SHA = "669a81472ede2269837c22a4d00070eaac360c82"
SURFACE_POINTS = 16_384
LINEAR_DEFLECTION = 0.0002
ANGULAR_DEFLECTION = 0.12
ARRAY_KEYS = ("cad_vec", "flag_vec", "index_vec", "key_padding_mask")
DEFAULT_SOURCE = Path("D:/projects/IdeaSpark/CADRetrievalIdea1/.cache/Text2CAD-official")
if os.name == "nt" and (Path(sys.prefix) / "Library/bin").is_dir():
    _dll = os.add_dll_directory(str(Path(sys.prefix) / "Library/bin"))


@lru_cache(maxsize=1)
def sequence_class():
    sys.path.insert(0, os.environ.get("TEXT2CAD_SOURCE", str(DEFAULT_SOURCE)))
    from CadSeqProc.cad_sequence import CADSequence
    return CADSequence


def minimal_sequence(program):
    """Construct the export's normalized floating model, with its original operation order."""
    parts = {}
    for key, item in program["parts"].items():
        part = {k: item[k] for k in ("sketch", "extrusion", "coordinate_system")}
        if not part["sketch"]:
            raise ValueError("empty sketch")
        for face in part["sketch"].values():
            if not face:
                raise ValueError("empty face")
            for loop in face.values():
                kinds = [k.split("_")[0] for k in loop]
                if not kinds or any(k not in {"line", "circle", "arc"} for k in kinds):
                    raise ValueError("unsupported curve or empty loop")
                if ("circle" in kinds) != (len(kinds) == 1):
                    raise ValueError("single closed curve must be a circle")
        parts[key] = part
    if not parts:
        raise ValueError("no extrusion")
    seq = sequence_class().from_minimal_json({"parts": parts})
    if seq.bbox is not None or len(seq.extrude_seq) != len(parts):
        raise ValueError("upstream minimal reader contract changed")
    for sketch, ext in zip(seq.sketch_seq, seq.extrude_seq):
        system = sketch.coordsystem.metadata
        system["euler_angles"] = np.deg2rad(system["euler_angles"])
        for k in ("extent_one", "extent_two", "sketch_size"):
            ext.metadata[k] = np.float64(ext.metadata[k])
        if not (ext.metadata["sketch_size"] > 0 and np.isfinite(ext.metadata["sketch_size"])):
            raise ValueError("invalid sketch scale")
        for curve in sketch.all_curves:
            if not all(np.isfinite(x).all() for x in curve.metadata.values()):
                raise ValueError("non-finite curve")
            if curve.curve_type == "circle":
                m = curve.metadata
                if m["radius"] <= 0:
                    raise ValueError("non-positive radius")
                # Mirror Circle.from_dict. Only the upper (+Y) cardinal point is emitted.
                for i, direction in enumerate(((0,1), (0,-1), (1,0), (-1,0)), 1):
                    m[f"pt{i}"] = m["center"] + m["radius"] * np.asarray(direction)
    return seq


def validate_arrays(arrays):
    from .text2cad_vector_scorer import validate_cad_vec
    cad, flags, indices, mask = (arrays[k] for k in ARRAY_KEYS)
    if cad.shape != (272, 2) or any(a.dtype != np.int32 for a in (cad, flags, indices)):
        raise ValueError("expected padded little-endian int32 Text2CAD tensors")
    diag = validate_cad_vec(cad, mask, flags, indices)
    # Check all real flag/index values and structure; validate_cad_vec checks padding and range.
    cursor = 1
    eid = 0
    while cursor < diag.n_real_tokens - 1:
        start = cursor
        nfaces = 0
        while cursor < diag.n_real_tokens and cad[cursor, 0] != 2:
            nloops = 0
            while cursor < diag.n_real_tokens and cad[cursor, 0] != 3:
                sizes = []
                while cursor < diag.n_real_tokens and cad[cursor, 0] != 4:
                    count = 0
                    while cursor < diag.n_real_tokens and min(cad[cursor]) >= 11:
                        count += 1
                        cursor += 1
                    if count not in (1, 2) or cad[cursor].tolist() != [5,0]:
                        raise ValueError("invalid point/END_CURVE grammar")
                    sizes.append(count)
                    cursor += 1
                if not sizes or (len(sizes) == 1 and sizes[0] != 2) or cad[cursor].tolist() != [4,0]:
                    raise ValueError("invalid END_LOOP grammar")
                nloops += 1
                cursor += 1
            if not nloops or cad[cursor].tolist() != [3,0]:
                raise ValueError("invalid END_FACE grammar")
            nfaces += 1
            cursor += 1
        if not nfaces or cad[cursor].tolist() != [2,0] or np.any(flags[start:cursor+1]):
            raise ValueError("invalid END_SKETCH or sketch flags")
        cursor += 1
        ext = cad[cursor:cursor+11]
        if (len(ext) != 11 or np.any(ext[:,1]) or np.any(ext[:8,0] < 11)
                or not 7 <= ext[8,0] <= 10 or ext[9,0] <= 11 or ext[10,0] != 6
                or flags[cursor:cursor+11].tolist() != [1,1,2,3,4,5,6,7,8,9,10]):
            raise ValueError("invalid extrusion fields or flags")
        cursor += 11
        if not np.all(indices[start:cursor] == eid):
            raise ValueError("invalid extrusion indices")
        eid += 1
    if (not eid or cursor != diag.n_real_tokens-1 or indices[0] != 0
            or indices[cursor] != eid-1 or flags[0] or flags[cursor]):
        raise ValueError("invalid program boundary")
    return diag


def padded_arrays(quantized):
    seq = quantized.to_vec(padding=False)
    if seq is None:
        raise ValueError("upstream to_vec failed")
    n = len(seq.cad_vec)
    if n > 272:
        raise ValueError(f"sequence_length_over_max:{n}")
    cad = np.zeros((272,2), dtype=np.int32)
    flags = np.full(272, 11, dtype=np.int32)
    indices = np.full(272, len(seq.sketch_seq), dtype=np.int32)
    cad[:n], flags[:n], indices[:n] = seq.cad_vec.numpy(), seq.flag_vec.numpy(), seq.index_vec.numpy()
    arrays = dict(zip(ARRAY_KEYS, (cad, flags, indices, cad == 0)))
    validate_arrays(arrays)
    return arrays


def encode_minimal(program):
    seq = minimal_sequence(program)
    for sk, ext in zip(seq.sketch_seq, seq.extrude_seq):
        # Invert json_to_NormalizedCAD's denormalize(translate=0) literally. The
        # translated sketch origin is already in its exported coordinate system.
        sk.transform(translate=0, scale=255.0 / ext.metadata["sketch_size"])
    return padded_arrays(seq.numericalize(bit=8))


def decode(cad):
    seq = sequence_class().from_vec(cad.copy(), bit=8, post_processing=False)
    if len(seq.extrude_seq) != np.count_nonzero(cad[:,0] == 6):
        raise ValueError("incomplete decode")
    return seq


@lru_cache(maxsize=1)
def causal_mask():
    from CadSeqProc.utility.utils import generate_attention_mask
    return generate_attention_mask(271).cpu().numpy().astype("<f4", copy=False)


def fingerprints(arrays):
    h = hashlib.sha256()
    for k in ARRAY_KEYS:
        h.update(np.ascontiguousarray(arrays[k], dtype="u1" if k == "key_padding_mask" else "<i4").tobytes())
    h.update(causal_mask().tobytes())
    return {"cad_vec_sha256": hashlib.sha256(arrays["cad_vec"].astype("<i4", copy=False).tobytes()).hexdigest(),
            "model_input_sha256": h.hexdigest()}


def measures(shape):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("invalid B-rep")
    solids = TopologyExplorer(shape, ignore_orientation=True).number_of_solids()
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()
    if not solids or not np.isfinite(volume) or volume <= 0:
        raise ValueError("missing solid or invalid volume")
    box = Bnd_Box()
    brepbndlib.AddOptimal(shape, box, False, False)
    return {"n_solids": solids, "volume": volume, "bbox": list(box.Get())}


def strict_solid(seq):
    """Use each official sketch builder; reject failures instead of returning its prefix."""
    from CadSeqProc.utility.utils import perform_op
    current = None
    for sk, ext in zip(seq.sketch_seq, seq.extrude_seq):
        m = ext.metadata
        body = sk.build_body({"extrude_values": [m["extent_one"], m["extent_two"]]})
        measures(body)
        op = m["boolean"]
        if current is None:
            if op not in (0,1):
                raise ValueError("first feature cannot cut or intersect")
            current = body
        else:
            current = perform_op(current, body, ("fuse","fuse","cut","common")[op])
            measures(current)
    if current is None:
        raise ValueError("empty solid")
    return current


def reference_frame(raw):
    box = raw["properties"]["bounding_box"]
    lo, hi = (np.asarray([box[key][k] for k in "xyz"], dtype=float) for key in ("min_point","max_point"))
    width = np.max(hi-lo)
    if not np.isfinite([lo,hi]).all() or width <= 0 or np.any(hi < lo):
        raise ValueError("invalid original bounding box")
    scale = .75/width
    result = np.eye(4)
    result[:3,:3] *= scale/10  # OCCT STEP importer target is mm, native JSON is cm
    result[:3,3] = -lo*scale
    return result


def read_reference(path, raw):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TColStd import TColStd_SequenceOfAsciiString
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError("STEP ReadFile failed")
    reader.SetSystemLengthUnit(1.0)
    units, angle, solid_angle = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.FileUnits(units, angle, solid_angle)
    if reader.TransferRoots() != reader.NbRootsForTransfer():
        raise ValueError("incomplete STEP transfer")
    trsf = gp_Trsf()
    matrix = reference_frame(raw)
    trsf.SetValues(*matrix[:3].ravel())
    result = BRepBuilderAPI_Transform(reader.OneShape(), trsf, True).Shape()
    return result, {"step_length_units": [units.Value(i).ToCString() for i in range(1,units.Length()+1)],
                    "kernel_mm_per_unit": reader.SystemLengthUnit(), "raw_json_units": "cm",
                    "raw_bbox_cm": raw["properties"]["bounding_box"], "step_mm_to_normalized": matrix.tolist()}


def mesh_triangles(shape):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Extend.TopologyUtils import TopologyExplorer
    mesher = BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION, False, ANGULAR_DEFLECTION, False)
    if not mesher.IsDone():
        raise ValueError("meshing failed")
    mesh = []
    for face in TopologyExplorer(shape).faces():
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None or not tri.NbTriangles():
            raise ValueError("face without triangulation")
        transform = loc.Transformation()
        points = np.array([tri.Node(i).Transformed(transform).Coord() for i in range(1,tri.NbNodes()+1)])
        faces = np.array([tri.Triangle(i).Get() for i in range(1,tri.NbTriangles()+1)]) - 1
        if face.Orientation() == TopAbs_REVERSED:
            faces = faces[:,[0,2,1]]
        mesh.append(points[faces])
    return np.concatenate(mesh)


def sample_surface(tri, seed):
    u, v = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
    areas = np.linalg.norm(np.cross(u,v),axis=1)
    rng = np.random.default_rng(seed)
    which = rng.choice(len(tri), SURFACE_POINTS, p=areas/areas.sum())
    uv = rng.random((SURFACE_POINTS,2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1-uv[flip]
    return tri[which,0] + uv[:,:1]*u[which] + uv[:,1:]*v[which]


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _triangles_squared_distance(points, triangles):
    """Distance for matched point/triangle pairs, including degenerate triangles."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    normal2 = _dot(normal, normal)
    # These cross products give the two unnormalized barycentric coordinates
    # of the orthogonal projection while avoiding cancellation in |ab × ac|².
    along_b = _dot(np.cross(ap, ac), normal)
    along_c = _dot(np.cross(ab, ap), normal)
    interior = (normal2 > 0) & (along_b >= 0) & (along_c >= 0) & (along_b + along_c <= normal2)
    plane = np.square(_dot(ap, normal)) / np.where(normal2 > 0, normal2, 1)
    boundary = np.full(len(points), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        vector = end - start
        delta = points - start
        norm2 = _dot(vector, vector)
        fraction = np.clip(_dot(delta, vector) / np.where(norm2 > 0, norm2, 1), 0.0, 1.0)
        residual = delta - fraction[:, None] * vector
        np.minimum(boundary, _dot(residual, residual), out=boundary)
    return np.where(interior, plane, boundary)


def distances(points, triangles):
    """Exact unsigned nearest-triangle distances on the same OCC tessellation as E1.

    Start with the distance to one actual triangle as an upper bound. A triangle
    can improve it only if its centroid lies within that bound plus its radius.
    Radius bands keep the search selective even when tiny and large faces coexist.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if points.shape != (len(points), 3) or triangles.shape != (len(triangles), 3, 3) or not len(triangles):
        raise ValueError("invalid sampled surface or target tessellation")
    if not np.isfinite(points).all() or not np.isfinite(triangles).all():
        raise ValueError("nonfinite surface coordinates")
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    all_centers = cKDTree(centers)
    _, seed = all_centers.query(points)
    best = _triangles_squared_distance(points, triangles[seed])
    _, exponent = np.frexp(radii)
    # Outward FP slack only expands the search and never changes distance values.
    slack = 32 * np.finfo(np.float64).eps * max(1, np.abs(points).max(initial=0), np.abs(triangles).max())
    for band in np.unique(exponent):
        group = np.flatnonzero(exponent == band)
        tree = cKDTree(centers[group])
        max_radius = radii[group].max()
        for start in range(0, len(points), 256):
            finish = start + 256
            subset = points[start:finish]
            bounds = best[start:finish]
            nearby = tree.query_ball_point(subset, np.sqrt(bounds) + max_radius + slack, return_sorted=False)
            count = np.fromiter(map(len, nearby), dtype=np.int64, count=len(subset))
            if not count.sum():
                continue
            point_indices = np.repeat(np.arange(len(subset)), count)
            face_indices = np.fromiter((index for indices in nearby for index in indices), dtype=np.int64,
                                       count=len(point_indices))
            delta = _triangles_squared_distance(subset[point_indices], triangles[group[face_indices]])
            np.minimum.at(bounds, point_indices, delta)
    return np.sqrt(best)


def cut_probes(seq):
    """Interior locations along circular cut axes. Boundary-near probes are filtered later."""
    probes = []
    for sk, ext in zip(seq.sketch_seq, seq.extrude_seq):
        if ext.metadata["boolean"] != 2:
            continue
        for curve in sk.all_curves:
            if curve.curve_type == "circle":
                center = sk.coordsystem.rotate_vec(curve.metadata["center"])
                depth = np.linspace(-ext.metadata["extent_two"], ext.metadata["extent_one"], 17)
                probes.extend(center + depth[:,None]*sk.coordsystem.normal)
    return probes


def occupancy(shape, points):
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_IN
    from OCC.Core.gp import gp_Pnt
    from OCC.Extend.TopologyUtils import TopologyExplorer
    inside = np.zeros(len(points), dtype=bool)
    for solid in TopologyExplorer(shape, ignore_orientation=True).solids():
        classifier = BRepClass3d_SolidClassifier(solid)
        for i, p in enumerate(points):
            classifier.Perform(gp_Pnt(*p), 1e-7)
            inside[i] |= classifier.State() == TopAbs_IN
    return inside


def compare_shapes(candidate, reference, uid, probes=()):
    cm, rm = measures(candidate), measures(reference)
    diag = np.linalg.norm(np.subtract(rm["bbox"][3:], rm["bbox"][:3]))
    ct, rt = mesh_triangles(candidate), mesh_triangles(reference)
    seed = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little")
    pq = float(np.quantile(distances(sample_surface(ct,seed),rt),.99)/diag)
    qp = float(np.quantile(distances(sample_surface(rt,seed+1),ct),.99)/diag)
    volume_error = abs(cm["volume"]-rm["volume"])/rm["volume"]
    eligible = disagreements = 0
    if len(probes):
        points = np.asarray(probes)
        keep = np.minimum(distances(points,ct),distances(points,rt)) > .01*diag
        eligible = int(keep.sum())
        if eligible:
            disagreements = int(np.count_nonzero(occupancy(candidate,points[keep]) != occupancy(reference,points[keep])))
    finite_p99 = bool(np.isfinite([pq, qp]).all())
    return {"candidate":cm, "reference":rm, "reference_diagonal":float(diag),
            "candidate_to_ref_p99":pq if np.isfinite(pq) else None,
            "ref_to_candidate_p99":qp if np.isfinite(qp) else None, "rel_volume":volume_error,
            "interior_probes": eligible, "material_disagreements": disagreements,
            "pass":bool(finite_p99 and cm["n_solids"] == rm["n_solids"] and volume_error <= .05
                        and max(pq,qp) <= .01 and not disagreements)}


def program_features(program):
    single_circle = near_rectangle = multi_or_cut = False
    numeric = []
    for part in program["parts"].values():
        e = part["extrusion"]
        multi_or_cut |= e["operation"] == "CutFeatureOperation"
        numeric.append({"attribute":"depth", "value_space":"normalized", "unit":None,
                        "value":e["extrude_depth_towards_normal"] + e["extrude_depth_opposite_normal"]})
        for face in part["sketch"].values():
            multi_or_cut |= len(face) > 1
            for loop in face.values():
                if len(loop) == 1 and next(iter(loop)).startswith("circle_"):
                    single_circle = True
                    numeric.append({"attribute":"diameter", "value_space":"normalized", "unit":None,
                                    "value":2*next(iter(loop.values()))["Radius"]})
                if len(loop) == 4 and all(k.startswith("line_") for k in loop):
                    edges = np.array([np.subtract(v["End Point"], v["Start Point"]) for v in loop.values()])
                    lengths = np.linalg.norm(edges,axis=1)
                    if np.all(lengths > 0):
                        directions = edges/lengths[:,None]
                        near_rectangle |= bool(np.max(abs(np.sum(directions*np.roll(directions,1,axis=0),axis=1))) < .02)
    return {"has_single_circle":single_circle, "has_near_rectangle":near_rectangle,
            "has_multiloop_or_cut":multi_or_cut,"numeric_values":numeric,"n_extrusions":len(program["parts"])}
