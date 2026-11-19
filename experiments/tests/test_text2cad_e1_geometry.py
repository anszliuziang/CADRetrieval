"""CPU regression witnesses for the inverse minimal export and independent geometry gate."""
import copy
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from idea01.text2cad_e1_geometry import (
    DEFAULT_SOURCE, encode_minimal, reference_frame, compare_shapes, decode, strict_solid, validate_arrays,
)

_available = (
    (Path(os.environ.get("TEXT2CAD_SOURCE", DEFAULT_SOURCE)) / "CadSeqProc/cad_sequence.py").is_file()
    and all(importlib.util.find_spec(m) for m in ("OCC", "loguru", "trimesh", "vtkmodules", "plyfile", "rich"))
)
geometry_runtime = unittest.skipUnless(_available, "requires pinned Text2CAD and the cadclip OCCT runtime")


def circle():
    return {"parts": {"part_1": {
        "coordinate_system": {"Euler Angles": [90., 0., 0.], "Translation Vector": [.25, .125, 0.]},
        "sketch": {"face_1": {"loop_1": {"circle_1": {"Center": [.25, .25], "Radius": .25}}}},
        "extrusion": {"extrude_depth_towards_normal": .25, "extrude_depth_opposite_normal": 0.,
                      "sketch_scale": .5, "operation": "NewBodyFeatureOperation"},
        "description": {"length": 12000}
    }}}


class GeometryTests(unittest.TestCase):
    def test_nonfinite_distance_in_either_direction_rejects_and_serializes(self):
        import json
        import idea01.text2cad_e1_geometry as geometry
        measures = {"n_solids":1, "volume":1., "bbox":[0,0,0,1,1,1]}
        triangles = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        for pair in ([0.,np.nan],[np.nan,0.]):
            with patch.object(geometry,"measures",return_value=measures), \
                    patch.object(geometry,"mesh_triangles",return_value=triangles), \
                    patch.object(geometry,"distances",side_effect=[np.array([x]) for x in pair]):
                result = compare_shapes(None,None,"uid")
                self.assertFalse(result["pass"])
                self.assertIn(None,[result["candidate_to_ref_p99"],result["ref_to_candidate_p99"]])
                json.dumps(result,allow_nan=False)

    @geometry_runtime
    def test_circle_orientation_local_grid_no_global_rescale_and_padding(self):
        arrays = encode_minimal(circle())
        cad = arrays["cad_vec"]
        self.assertEqual(cad[1:3].tolist(), [[139, 139], [139, 266]])
        ext = np.flatnonzero(cad[:, 0] == 2)[0] + 1
        self.assertEqual(cad[ext:ext+10, 0].tolist(), [171,139,171,155,139,203,139,139,7,75])
        self.assertTrue(np.array_equal(arrays["key_padding_mask"], cad == 0))
        self.assertEqual(arrays["key_padding_mask"].dtype, np.bool_)
        self.assertTrue(np.all(arrays["flag_vec"][cad[:, 0] == 0] == 11))
        self.assertTrue(np.all(arrays["index_vec"][cad[:, 0] == 0] == 1))
        modified = circle()
        modified["parts"]["part_1"]["description"] = "unrelated descriptions"
        np.testing.assert_array_equal(cad, encode_minimal(modified)["cad_vec"])

    def test_reference_transform_uses_raw_cm_bounds_for_step_mm(self):
        raw = {"properties": {"bounding_box": {"min_point": dict(x=-4,y=1,z=2),
                                              "max_point": dict(x=2,y=3,z=5)}}}
        matrix = reference_frame(raw)
        np.testing.assert_allclose(matrix @ [-40, 10, 20, 1], [0,0,0,1], atol=1e-12)
        np.testing.assert_allclose(matrix @ [20,30,50,1], [.75,.25,.375,1], atol=1e-12)
        self.assertGreater(np.linalg.det(matrix[:3,:3]), 0)

    @geometry_runtime
    def test_surface_gate_detects_changed_depth(self):
        solid = strict_solid(decode(encode_minimal(circle())["cad_vec"]))
        damaged = circle()
        damaged["parts"]["part_1"]["extrusion"]["extrude_depth_towards_normal"] = .125
        second = strict_solid(decode(encode_minimal(damaged)["cad_vec"]))
        self.assertTrue(compare_shapes(solid, solid, "control")["pass"])
        result = compare_shapes(second, solid, "control")
        self.assertFalse(result["pass"])
        self.assertAlmostEqual(result["rel_volume"], .5, places=5)

    @geometry_runtime
    def test_real_flags_and_boolean_fields_are_checked(self):
        a = encode_minimal(circle())
        broken = {k:v.copy() for k,v in a.items()}
        ext = np.flatnonzero(a["cad_vec"][:,0] == 2)[0]+1
        broken["flag_vec"][ext+2] = 7
        with self.assertRaisesRegex(ValueError, "flags"):
            validate_arrays(broken)
        broken = {k:v.copy() for k,v in a.items()}
        broken["cad_vec"][ext+8,0] = 11
        with self.assertRaisesRegex(ValueError, "extrusion"):
            validate_arrays(broken)

    @geometry_runtime
    def test_multiple_extrusion_indices_and_cut_operation(self):
        model = circle()
        part = copy.deepcopy(model["parts"]["part_1"])
        part["extrusion"]["operation"] = "CutFeatureOperation"
        model["parts"]["part_2"] = part
        a = encode_minimal(model)
        self.assertEqual(a["cad_vec"][a["flag_vec"]==8,0].tolist(),[7,9])
        self.assertEqual(a["index_vec"][a["flag_vec"]==8].tolist(),[0,1])
        self.assertTrue(np.all(a["index_vec"][a["cad_vec"][:,0]==0] == 2))


if __name__ == "__main__":
    unittest.main()
