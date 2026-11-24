"""Regression checks for the Fusion metadata boundary and geometry distance."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "scripts"))
from prepare_formal_fusion360 import ProgramError, json_bytes, read_program  # noqa: E402
from convert_formal_fusion360 import native_distances, select_routes  # noqa: E402
from idea01.text2cad_e1_geometry import _triangles_squared_distance, distances  # noqa: E402


def extrusion(distance, *, kind="DistanceExtentDefinition", **fields):
    return {"type": kind, "distance": {"value": distance, "name": "unused"}, **fields}


@pytest.mark.parametrize("mode,one,two,expected", [
    ("OneSideFeatureExtentType", extrusion(-2), None, (0.0, 2.0)),
    ("SymmetricFeatureExtentType", extrusion(8, kind="SymmetricExtentDefinition", is_full_length=True), None, (4.0, 4.0)),
    ("SymmetricFeatureExtentType", extrusion(8, kind="SymmetricExtentDefinition", is_full_length=False), None, (8.0, 8.0)),
    ("TwoSidesFeatureExtentType", extrusion(1.2), extrusion(2.4), (1.2, 2.4)),
])
def test_native_distance_conventions(mode, one, two, expected):
    feature = {"extent_type": mode, "extent_one": one, "extent_two": two,
               "start_extent": {"type": "OffsetStartDefinition", "offset": {"value": -0.25}}}
    assert native_distances(feature) == (expected, -0.25)


def test_rejects_taper_and_unsupported_extent():
    feature = {"extent_type": "OneSideFeatureExtentType",
               "start_extent": {"type": "ProfilePlaneStartDefinition"}}
    with pytest.raises(ProgramError, match="unsupported_native_taper"):
        native_distances({**feature, "extent_one": extrusion(2, taper_angle={"value": 0.02})})
    with pytest.raises(ProgramError, match="unsupported_native_extent_definition"):
        native_distances({**feature, "extent_one": extrusion(2, kind="ThroughAllExtentDefinition")})


def test_candidate_input_coverage_is_independent_of_aligned_training_coverage():
    aligned = {"input_status": "ok", "geometry": {"status": "aligned"}}
    other = {"input_status": "ok", "geometry": {"status": "geometry_mismatch"}}
    failed = {"input_status": "failed"}
    routes = {"autodesk_history": {"models": {"text2cad": other, "hiercad": failed}},
              "cadmium_minimal": {"models": {"text2cad": failed, "hiercad": aligned}}}
    assert select_routes(routes) == ({"text2cad": None, "hiercad": "cadmium_minimal"},
                                    {"text2cad": "autodesk_history", "hiercad": "cadmium_minimal"}, None)
    routes["cadmium_minimal"]["models"]["text2cad"] = aligned
    assert select_routes(routes) == ({"text2cad": "cadmium_minimal", "hiercad": "cadmium_minimal"},
                                    {"text2cad": "cadmium_minimal", "hiercad": "cadmium_minimal"}, "cadmium_minimal")


def test_minimal_literal_carries_only_typed_geometry():
    circle = {"Center": [0.5, 1], "Radius": 0.15, "description": "part text"}
    part = {"coordinate_system": {"Euler Angles": [0, 0, 0], "Translation Vector": [0, 0, 0], "name": "query"},
            "sketch": {"face name (source)": {"loop name (source)": {"circle_17": circle}}},
            "extrusion": {"operation": "NewBodyFeatureOperation", "extrude_depth_towards_normal": .1,
                          "extrude_depth_opposite_normal": 0, "sketch_scale": 1, "file_name": "query"},
            "description": {"name": "part text"}}
    protected = {"parts": {"name of part": part}, "uid": "source UID", "annotation": "query"}
    cleaned = read_program(str(protected))
    assert list(cleaned["parts"]) == ["part_1"]
    assert cleaned["parts"]["part_1"]["sketch"] == {"face_1": {"loop_1": {"circle_1": {"Center": [.5, 1], "Radius": .15}}}}
    assert b"part text" not in json_bytes(cleaned)
    protected_variant = deepcopy(protected)
    protected_variant["parts"]["name of part"]["description"] = ["arbitrary text"]
    assert json_bytes(read_program(str(protected_variant))) == json_bytes(cleaned)
    circle["Radius"] = "0.15"
    with pytest.raises(ProgramError, match="invalid_Radius"):
        read_program(str(protected))


def test_distance_to_analytic_triangle_regions():
    pytest.importorskip("scipy")
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    points = np.array([[.2, .4, 2.], [.5, -.4, 0.], [1.1, -.2, 0.], [.7, .7, 0.]])
    expected = np.array([2, .4, np.hypot(.1, .2), np.hypot(.2, .2)])
    np.testing.assert_allclose(distances(points, triangle), expected, rtol=0, atol=1e-13)


def test_pruning_matches_all_pairs_with_heterogeneous_and_degenerate_triangles():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(219235)
    points = rng.normal(size=(273, 3))
    centers = rng.normal(size=(37, 1, 3))
    scale = np.geomspace(.003, 50, num=len(centers))[:, None, None]
    triangles = centers + rng.normal(size=(len(centers), 3, 3)) * scale
    triangles[0] = centers[0]
    triangles[1, 1] = triangles[1, 0]
    repeated_points = np.repeat(points, len(triangles), axis=0)
    all_faces = np.tile(triangles, (len(points), 1, 1))
    brute_force = _triangles_squared_distance(repeated_points, all_faces).reshape(len(points), -1).min(axis=1)
    np.testing.assert_allclose(distances(points, triangles), np.sqrt(brute_force), rtol=0, atol=1e-13)
