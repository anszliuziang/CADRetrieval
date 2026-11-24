"""OpenCASCADE replay and final-solid measurements for formal binding motifs."""
from __future__ import annotations

from collections import Counter
import math

from . import in_domain_pilot_geometry as pilot


BOOLEAN_FUZZY = 1e-5
SAME_BODY_LIMIT = 2e-6
MIN_RELATIVE_DISTANCE_GAP = 0.015
MIN_DISTANCE_RATIO = 1.10
MIN_SIZE_RATIO = 1.20
MAX_EQUAL_SIZE_RATIO = 1.05


def replay_sequence(sequence):
    """Replay through OCC and require every cut or join to change the solid."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Extend.TopologyUtils import TopologyExplorer

    shape = None
    volumes = []
    for index, (sketch, extrusion) in enumerate(zip(sequence.sketch_seq, sequence.extrude_seq, strict=True)):
        meta = extrusion.metadata
        operation = meta["boolean"]
        if operation not in ((0,) if index == 0 else (1, 2)):
            raise ValueError(f"unexpected operation {operation} at feature {index}")
        frame = sketch.coordsystem.metadata
        for channel in ("origin", "x_axis", "y_axis", "z_axis"):
            frame[channel] = frame[channel].astype("float64", copy=False)
        tool = sketch.build_body({"extrude_values": [meta["extent_one"], meta["extent_two"]]})
        pilot._solid_volume(tool)
        if shape is None:
            shape = tool
        elif operation == 1:
            join = BRepAlgoAPI_Fuse(shape, tool)
            join.SetFuzzyValue(BOOLEAN_FUZZY)
            join.Build()
            if not join.IsDone():
                raise ValueError(f"join failed at feature {index}")
            shape = join.Shape()
        else:
            shape = pilot._cut(shape, tool, fuzzy=BOOLEAN_FUZZY)
        # Equal-domain fusion removes artificial internal seams at a fused deck or wall.
        clean = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        clean.Build()
        shape = clean.Shape()
        value = pilot._solid_volume(shape)
        if volumes:
            change = (value - volumes[-1]) * (1 if operation == 1 else -1)
            if change <= volumes[-1] * 1e-6:
                raise ValueError(f"feature {index} did not change material volume as declared")
        volumes.append(value)
    if len(volumes) < 3:
        raise ValueError("expected a stock plus the two through cuts")
    return next(TopologyExplorer(shape).solids()), volumes


def replay_raw(sequence_type, raw: dict):
    return replay_sequence(sequence_type.from_dict(raw))


def _center_area(face):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    center = props.CentreOfMass()
    area = float(props.Mass())
    if not math.isfinite(area) or area <= 0:
        raise ValueError("invalid planar face area")
    return (center.X(), center.Y(), center.Z()), area


def _wire_center_area(wire):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    face = BRepBuilderAPI_MakeFace(wire, True)
    if not face.IsDone():
        raise ValueError("cannot reconstruct the face of a closed wire")
    return _center_area(face.Face())


def _curve_types(wire):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Extend.TopologyUtils import TopologyExplorer

    return [BRepAdaptor_Curve(edge).GetType() for edge in TopologyExplorer(wire).edges()]


def _circle_radius(wire):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Circle
    from OCC.Extend.TopologyUtils import TopologyExplorer

    edges = list(TopologyExplorer(wire).edges())
    if len(edges) == 1:
        curve = BRepAdaptor_Curve(edges[0])
        if (curve.GetType() == GeomAbs_Circle
                and abs(abs(curve.LastParameter() - curve.FirstParameter()) - 2 * math.pi) < 1e-5):
            return float(curve.Circle().Radius())
    return None


def _planar_faces(shape):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Extend.TopologyUtils import TopologyExplorer

    records = []
    kinds = Counter()
    for face in TopologyExplorer(shape).faces():
        surface = BRepAdaptor_Surface(face)
        typ = surface.GetType()
        kinds[str(typ)] += 1
        if typ == GeomAbs_Plane:
            direction = surface.Plane().Axis().Direction()
            sign = -1 if face.Orientation() == TopAbs_REVERSED else 1
            normal = tuple(sign * component for component in (direction.X(), direction.Y(), direction.Z()))
            outer = breptools.OuterWire(face)
            inner = tuple(wire for wire in TopologyExplorer(face).wires() if not wire.IsSame(outer))
            center, area = _center_area(face)
            records.append({"center": center, "area": area, "normal": normal, "outer": outer, "inner": inner})
    return records, kinds


def _outside(shape, point, tolerance):
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_OUT
    from OCC.Core.gp import gp_Pnt

    return BRepClass3d_SolidClassifier(shape, gp_Pnt(*point), tolerance).State() == TopAbs_OUT


def _vertical_bores(shape, horizontal, *, tolerance):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.GeomAbs import GeomAbs_Circle
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    openings = []
    unclassified = []
    for face in horizontal:
        z = face["center"][2]
        for wire in face["inner"]:
            if GeomAbs_Circle not in _curve_types(wire):
                continue
            try:
                hole = pilot._measure_hole(wire)
            except ValueError as exc:
                unclassified.append({"z": z, "reason": str(exc)})
                continue
            openings.append({"z": z, "normal_z": face["normal"][2], **hole})
    bores = []
    lone_openings = []
    for top in (item for item in openings if item["normal_z"] > 0):
        bottom_matches = [
            other for other in openings if other["normal_z"] < 0 and other["z"] < top["z"] - tolerance
            and other["kind"] == top["kind"]
            and math.dist(top["center_xy"], other["center_xy"]) <= tolerance
            and math.isclose(top["area"], other["area"], rel_tol=1e-6)
        ]
        if not bottom_matches:
            lone_openings.append(top)
            continue
        if len(bottom_matches) != 1:
            raise ValueError("nonunique matching bottom section for a vertical bore")
        bottom = bottom_matches[0]
        z0, z1 = bottom["z"], top["z"]
        x, y = top["center_xy"]
        fractions = (0.05, 0.25, 0.50, 0.75, 0.95)
        zs = [z0 + (z1 - z0) * fraction for fraction in fractions]
        if not all(_outside(shape, (x, y, z), tolerance) for z in zs):
            raise ValueError("a bore centerline enters material between its openings")
        axis = gp_Ax2(gp_Pnt(x, y, z0), gp_Dir(0.0, 0.0, 1.0))
        probe = BRepPrimAPI_MakeCylinder(axis, 0.25 * min(top["arc_radii"]), z1 - z0).Shape()
        common = BRepAlgoAPI_Common(shape, probe)
        if not common.IsDone():
            raise ValueError("through-bore core intersection could not be checked")
        blocked = pilot._volume(common.Shape())
        if blocked > 1e-6 * pilot._volume(probe):
            raise ValueError("material crosses the measured through-bore core")
        bores.append({**top, "bottom": bottom, "open_center_sample_z": zs,
                      "blocked_probe_volume": blocked, "through_span": [z0, z1]})
    return bores, {"single_top_openings": lone_openings, "unclassified_curved_openings": unclassified}


def _unique(items, label):
    if len(items) != 1:
        raise ValueError(f"{label} is not unique: found {len(items)}")
    return items[0]


def _triangle_tip(face):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GeomAbs import GeomAbs_Line
    from OCC.Extend.TopologyUtils import WireExplorer

    if _curve_types(face["outer"]) != [GeomAbs_Line] * 3:
        raise ValueError("upper perimeter is not one three-edge triangle")
    points = [BRep_Tool.Pnt(vertex) for vertex in WireExplorer(face["outer"]).ordered_vertices()]
    if len(points) != 3:
        raise ValueError("incomplete triangle vertex traversal")
    xy = [(p.X(), p.Y()) for p in points]
    corners = []
    for tip, first, second in zip(xy, xy[1:] + xy[:1], xy[2:] + xy[:2], strict=True):
        u = [a - b for a, b in zip(first, tip, strict=True)]
        v = [a - b for a, b in zip(second, tip, strict=True)]
        cosine = sum(a * b for a, b in zip(u, v, strict=True)) / (math.hypot(*u) * math.hypot(*v))
        corners.append((math.acos(max(-1.0, min(1.0, cosine))), tip))
    corners.sort()
    if corners[1][0] < 1.20 * corners[0][0]:
        raise ValueError("triangle has no uniquely sharper tip")
    return corners[0][1], {"corner_angles_radians": [value for value, _ in corners]}


def _side_opening_anchor(shape, vertical, family, tolerance):
    from OCC.Core.GeomAbs import GeomAbs_Line

    if family == "side_window_portal":
        windows = [
            (_wire_center_area(wire)[0], face["normal"])
            for face in vertical for wire in face["inner"] if _curve_types(wire) == [GeomAbs_Line] * 4
        ]
        if len(windows) != 2:
            raise ValueError(f"expected two rectangular side-window openings, found {len(windows)}")
        (a, na), (b, nb) = windows
        axis = tuple(right - left for left, right in zip(a, b, strict=True))
        dot = lambda left, right: sum(x * y for x, y in zip(left, right, strict=True))
        center = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
        if (not math.isclose(dot(na, nb), -1.0, abs_tol=1e-8)
                or not math.isclose(abs(dot(na, axis)), math.hypot(*axis), abs_tol=tolerance)
                or not _outside(shape, center, tolerance)):
            raise ValueError("rectangular openings do not bound an open straight side window")
        return center[:2], {"side_opening_centers": [a, b]}

    findings = [
        (_wire_center_area(wire)[0], face["normal"], radius)
        for face in vertical for wire in face["inner"] if (radius := _circle_radius(wire)) is not None
    ]
    center, normal, radius = _unique(findings, "circular side-bore entrance")
    caps = []
    for face in vertical:
        r = _circle_radius(face["outer"])
        if r is None or face["inner"] or not math.isclose(radius, r, rel_tol=1e-6):
            continue
        delta = tuple(end - start for end, start in zip(face["center"], center, strict=True))
        depth = -sum(x * y for x, y in zip(delta, normal, strict=True))
        if depth > tolerance and math.isclose(depth, math.hypot(*delta), abs_tol=tolerance):
            caps.append(face["center"])
    cap = _unique(caps, "closed coaxial cap of the horizontal bore")
    return center[:2], {"side_bore_opening_xyz": center, "blind_cap_xyz": cap, "radius": radius}


def _anchor(shape, family, planes, horizontal, bores, bounds, tolerance):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Line
    from OCC.Extend.TopologyUtils import TopologyExplorer

    zlo, zhi = bounds[2], bounds[5]
    up = [face for face in horizontal if face["normal"][2] > 0]
    down = [face for face in horizontal if face["normal"][2] < 0]
    top = [face for face in up if abs(face["center"][2] - zhi) <= tolerance]

    if family == "end_eye_link":
        by_radius = sorted((hole for hole in bores if hole["kind"] == "round"),
                           key=lambda hole: hole["arc_radii"][0], reverse=True)
        if not by_radius or (len(by_radius) > 1 and by_radius[0]["arc_radii"][0] < 1.20 * by_radius[1]["arc_radii"][0]):
            raise ValueError("no unique large circular through eye")
        eye = by_radius[0]
        return tuple(eye["center_xy"]), {"auxiliary_through_eye": eye}, [hole for hole in bores if hole is not eye]
    if family in ("side_window_portal", "transverse_blind_lug"):
        vertical = [face for face in planes if abs(face["normal"][2]) <= 1e-8]
        center, evidence = _side_opening_anchor(shape, vertical, family, tolerance)
        return center, evidence, bores
    if family == "inclined_gusset_plate":
        edges = []
        for edge in TopologyExplorer(shape).edges():
            data = pilot._bounds(edge)
            if (abs(data[2] - zhi) <= tolerance and abs(data[5] - zhi) <= tolerance
                    and BRepAdaptor_Curve(edge).GetType() == GeomAbs_Line):
                edges.append(data)
        data = _unique(edges, "highest straight gusset edge")
        return ((data[0] + data[3]) / 2, (data[1] + data[4]) / 2), {"edge_bounds": data}, bores
    if family == "two_level_terrace":
        face = _unique(top, "upper deck")
        edges = []
        for edge in TopologyExplorer(face["outer"]).edges():
            data = pilot._bounds(edge)
            midpoint = ((data[0] + data[3]) / 2, (data[1] + data[4]) / 2)
            if all(bounds[i] + tolerance < midpoint[i] < bounds[i + 3] - tolerance for i in range(2)):
                edges.append(midpoint)
        midpoint = _unique(edges, "upper deck edge interior to the part in plan")
        return midpoint, {"upper_step_edge_xy": midpoint}, bores
    if family == "pointed_mounting_web":
        tip, evidence = _triangle_tip(_unique(top, "upper triangular web"))
        return tip, evidence, bores
    if family == "annular_socket_plate":
        face = _unique(top, "top collar face")
        inside = _unique(list(face["inner"]), "inside edge of the top annulus")
        outer_radius, inner_radius = _circle_radius(face["outer"]), _circle_radius(inside)
        outer_center, inner_center = (_wire_center_area(wire)[0] for wire in (face["outer"], inside))
        if (outer_radius is None or inner_radius is None or outer_radius <= inner_radius
                or math.dist(outer_center, inner_center) > tolerance):
            raise ValueError("raised collar lacks concentric circular rims")
        floors = []
        for candidate in up:
            radius = _circle_radius(candidate["outer"])
            if (radius is not None and not candidate["inner"] and candidate["center"][2] < zhi - tolerance
                    and math.isclose(radius, inner_radius, rel_tol=1e-6)
                    and math.dist(candidate["center"][:2], inner_center[:2]) <= tolerance):
                floors.append(candidate["center"])
        floor = _unique(floors, "blind circular socket floor")
        return inner_center[:2], {"rim_xyz": inner_center, "floor_xyz": floor,
                                  "radii": [inner_radius, outer_radius]}, bores
    if family == "dual_skid_plate":
        bottom = sorted((face for face in down if abs(face["center"][2] - zlo) <= tolerance),
                        key=lambda face: face["area"], reverse=True)
        if len(bottom) != 2 or bottom[0]["area"] < 1.20 * bottom[1]["area"]:
            raise ValueError("expected two bottom supports with one distinctly longer skid")
        face = bottom[0]
        return face["center"][:2], {"bottom_support_areas": [item["area"] for item in bottom]}, bores
    if family == "returned_corner_bracket":
        face = _unique(top, "top of the returned flange")
        vertex, reflex = pilot._measure_notch(face["outer"], 1)
        return vertex.tolist(), {"flange_inside_corner_xy": reflex[0]}, bores
    if family == "cantilever_lip_plate":
        face = _unique([item for item in down if item["center"][2] > zlo + tolerance], "cantilever underside")
    elif family == "blind_keyseat_block":
        face = _unique([item for item in up
                        if zlo + tolerance < item["center"][2] < zhi - tolerance], "rectangular keyseat floor")
        if face["inner"] or _curve_types(face["outer"]) != [GeomAbs_Line] * 4:
            raise ValueError("keyseat floor must be a quadrilateral with no opening")
    elif family == "dovetail_saddle":
        deck = min(item["center"][2] for item in up)
        face = _unique([item for item in up if deck + tolerance < item["center"][2] < zhi - tolerance],
                       "dovetail floor above the main deck")
    else:
        raise ValueError(f"Unknown geometry rule for {family}")
    return face["center"][:2], {"face_center_xyz": face["center"], "face_area": face["area"]}, bores


def measure(shape, family: str, attribute: str, kind: str, expected) -> dict:
    """Measure the finished B-Rep using the frozen family rule and no design parameters."""
    bounds = pilot._bounds(shape)
    volume = pilot._solid_volume(shape)
    scale = max(bounds[i + 3] - bounds[i] for i in range(3))
    tolerance = max(1e-7, 1e-7 * scale)
    planes, surfaces = _planar_faces(shape)
    horizontal = [face for face in planes if abs(face["normal"][2]) >= 1 - 1e-8]
    bores, details = _vertical_bores(shape, horizontal, tolerance=tolerance)
    anchor, anchor_details, targets = _anchor(shape, family, planes, horizontal, bores, bounds, tolerance)
    if len(targets) != 2:
        raise ValueError(f"expected two target through bores, found {len(targets)}")
    for hole in targets:
        hole["anchor_distance"] = math.dist(hole["center_xy"], anchor)
    near, far = sorted(targets, key=lambda hole: hole["anchor_distance"])
    distances = (near["anchor_distance"], far["anchor_distance"])
    if distances[1] - distances[0] <= MIN_RELATIVE_DISTANCE_GAP * scale or distances[1] <= MIN_DISTANCE_RATIO * distances[0]:
        raise ValueError(f"ambiguous anchor-to-hole distances: {distances}")
    if family == "two_level_terrace" and abs(near["z"] - far["z"]) <= tolerance:
        raise ValueError("terrace bores must emerge at distinct upper levels")
    if attribute == "size_location":
        if any(hole["kind"] != "round" for hole in targets):
            raise ValueError("diameter comparison requires two round holes")
        nr, fr = (hole["arc_radii"][0] for hole in (near, far))
        ratio = max(nr, fr) / min(nr, fr)
        if kind == "easy" and ratio <= MAX_EQUAL_SIZE_RATIO:
            answer = "same"
        else:
            if ratio <= MIN_SIZE_RATIO:
                raise ValueError(f"ambiguous diameter ratio: {ratio}")
            answer = "different" if kind == "easy" else (("larger", "smaller") if nr > fr else ("smaller", "larger"))
    elif attribute == "shape_location":
        forms = (near["kind"], far["kind"])
        answer = ("same" if forms[0] == forms[1] else "different") if kind == "easy" else forms
    else:
        raise ValueError(f"unknown attribute {attribute}")
    return {"status": "ok" if answer == expected else "binding_mismatch", "volume": volume, "bounds": bounds,
            "anchor_xy": anchor, "anchor_evidence": anchor_details, "near": near, "far": far,
            "measured_binding": answer, "expected_binding": expected, "surface_types": surfaces,
            "distance_gap": distances[1] - distances[0], "distance_tolerance": MIN_RELATIVE_DISTANCE_GAP * scale,
            **details}


def compare_same_body(left, right):
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    differences = [pilot._cut(a, b) for a, b in ((left, right), (right, left))]
    if any(not shape.IsNull() and not BRepCheck_Analyzer(shape).IsValid() for shape in differences):
        raise ValueError("invalid bilateral difference in equivalent-body check")
    delta = sum(pilot._volume(shape) for shape in differences)
    ratio = delta / max(pilot._solid_volume(left), pilot._solid_volume(right))
    return {"status": "verified" if ratio <= SAME_BODY_LIMIT else "geometry_mismatch",
            "relative_symmetric_difference": ratio}
