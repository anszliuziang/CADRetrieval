"""Check the field and geometry assumptions of the Fusion conversion preflight."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
TEXT2CAD_SOURCE = Path(os.environ.get(
    "TEXT2CAD_SOURCE", "D:/projects/IdeaSpark/CADRetrievalIdea1/.cache/Text2CAD-official",
))


def rectangle(mode=1, extent_one=160, extent_two=128):
    vec = np.full((7, 17), -1, dtype=np.int32)
    vec[:, 0] = [4, 0, 0, 0, 0, 5, 3]
    vec[1:5, 1:3] = [[223, 128], [223, 176], [128, 176], [128, 128]]
    vec[5, 6:] = [128, 128, 128, 128, 128, 128, 96, extent_one, extent_two, 0, mode]
    return vec


class CadTranslatorVectorTests(unittest.TestCase):
    def test_signed_profile_coordinates_and_symmetric_depth(self):
        from prepare_retrieval_benchmark_fusion import decode_h5_vector

        feature, = decode_h5_vector(rectangle())
        np.testing.assert_array_equal(feature.axes, np.eye(3))
        np.testing.assert_array_equal(feature.origin, [0, 0, 0])
        self.assertEqual(feature.loops[0]["line_1"]["Start Point"], [0, 0])
        self.assertEqual(feature.loops[0]["line_1"]["End Point"], [0.75, 0])
        self.assertEqual(feature.extrusion_steps(), (0.25, -0.25))
        self.assertEqual(feature.extrusion_steps(symmetric_total=True), (0.125, -0.125))

    def test_two_signed_distances_can_point_to_the_same_side(self):
        from prepare_retrieval_benchmark_fusion import decode_h5_vector

        feature, = decode_h5_vector(rectangle(mode=2, extent_one=96, extent_two=144))
        self.assertEqual(feature.extrusion_steps(), (-0.25, -0.125))

    @unittest.skipUnless(importlib.util.find_spec("OCC"), "requires the OCCT runtime")
    def test_replay_rejects_zero_on_either_side_of_a_two_sided_extrusion(self):
        from prepare_retrieval_benchmark_fusion import decode_h5_vector, replay_h5

        for extent_one, extent_two in ((128, 160), (160, 128), (128, 128)):
            with self.subTest(extent_one=extent_one, extent_two=extent_two):
                features = decode_h5_vector(rectangle(mode=2, extent_one=extent_one, extent_two=extent_two))
                with self.assertRaisesRegex(ValueError, "HDF5 two-sided extrusion has a zero-depth side"):
                    replay_h5(features)

    def test_arc_uses_preceding_end_and_the_encoded_sweep_side(self):
        from prepare_retrieval_benchmark_fusion import decode_h5_vector

        vec = rectangle()
        vec[1, :5] = [1, 223, 128, 128, 1]
        feature, = decode_h5_vector(vec)
        arc = feature.loops[0]["arc_1"]
        np.testing.assert_allclose(arc["Start Point"], [0, 0])
        np.testing.assert_allclose(arc["Mid Point"], [0.375, -0.375])
        np.testing.assert_allclose(arc["End Point"], [0.75, 0])

    def test_rejects_discarded_commands_and_invalid_field_masks(self):
        from prepare_retrieval_benchmark_fusion import decode_h5_vector

        vec = rectangle()
        with self.assertRaisesRegex(ValueError, "EOS"):
            decode_h5_vector(np.concatenate((vec, vec[1:2])))
        vec[1, 6] = 3
        with self.assertRaisesRegex(ValueError, "inactive"):
            decode_h5_vector(vec)

    def test_zero_sketch_scale_is_not_recovered_by_quantization(self):
        from prepare_retrieval_benchmark_fusion import decode_h5_vector

        vec = rectangle()
        vec[5, 12] = 0
        with self.assertRaisesRegex(ValueError, "sketch scale"):
            decode_h5_vector(vec)

    @unittest.skipUnless(
        (TEXT2CAD_SOURCE / "CadSeqProc" / "cad_sequence.py").is_file()
        and all(importlib.util.find_spec(mod) for mod in
                ("OCC", "vtkmodules", "scipy", "torch", "loguru", "trimesh", "plyfile", "rich")),
        "requires the E1 CAD geometry runtime",
    )
    def test_independent_replay_and_text2cad_quantization_preserve_geometry(self):
        from prepare_retrieval_benchmark_fusion import (
            decode_h5_vector, make_minimal_program, replay_h5, text2cad_frame,
        )
        from idea01 import text2cad_e1_geometry as geometry

        features = decode_h5_vector(rectangle())
        raw = replay_h5(features)
        reference, origin, scale = text2cad_frame(raw)
        np.testing.assert_allclose(geometry.measures(reference)["bbox"],
                                   [0, 0, 0, 0.75, 36/95, 0.5], atol=1e-8)
        arrays = geometry.encode_minimal(make_minimal_program(features, origin, scale))
        candidate = geometry.strict_solid(geometry.decode(arrays["cad_vec"]))
        comparison = geometry.compare_shapes(candidate, reference, "synthetic-rect")
        self.assertTrue(comparison["pass"], comparison)
        self.assertLess(comparison["rel_volume"], 0.005)


class FusionGalleryTests(unittest.TestCase):
    def test_duplicate_requires_all_four_exact_tensors(self):
        from prepare_retrieval_benchmark_fusion import vector_digest
        from prepare_retrieval_benchmark_fusion_gallery import source_duplicate_matches

        tensors = {
            "cad_vec": np.array([[1, 0], [22, 10], [1, 0]], dtype=np.int32),
            "flag_vec": np.array([0, 0, 0], dtype=np.int32),
            "index_vec": np.array([0, 0, 0], dtype=np.int32),
            "key_padding_mask": np.array([[False, True], [False, False], [False, True]]),
        }
        digest = vector_digest(tensors["cad_vec"])
        index = {digest: [("train", "0000/00000001"), ("validation", "0000/00000002"),
                          ("test", "0000/00000003"), ("train", "0000/00000004")]}
        programs = {
            "0000/00000001": {**tensors, "flag_vec": np.array([0, 1, 0], dtype=np.int32)},
            "0000/00000002": tensors,
            "0000/00000003": {**tensors, "index_vec": np.array([0, 1, 0], dtype=np.int32)},
            "0000/00000004": {**tensors, "key_padding_mask": np.zeros((3, 2), dtype=bool)},
        }
        self.assertEqual(
            source_duplicate_matches(tensors, index, programs.__getitem__),
            [{"source_split": "validation", "source_uid": "0000/00000002"}],
        )

    def test_frozen_axis_retains_captionless_candidates_and_counts_prompt_reuse(self):
        from prepare_retrieval_benchmark_fusion_gallery import make_axes

        records = [
            {"uid": "0001/00010003", "outcome": "accepted", "h5_member": "c.h5",
             "caption": "part", "candidate_fields": 6, "artifacts": {"arrays": {"member": "c.npz"}}},
            {"uid": "0001/00010002", "outcome": "accepted", "h5_member": "b.h5",
             "caption": None, "candidate_fields": 4, "artifacts": {"arrays": {"member": "b.npz"}}},
            {"uid": "0001/00010001", "outcome": "accepted", "h5_member": "a.h5",
             "caption": "part", "candidate_fields": 5, "artifacts": {"arrays": {"member": "a.npz"}}},
            {"uid": "0001/00010000", "outcome": "source_duplicate", "caption": "excluded"},
        ]
        gallery, texts, queries, qrels, fields = make_axes(records)
        self.assertEqual([(r["uid"], r["cad_index"]) for r in gallery],
                         [("0001/00010001", 0), ("0001/00010002", 1), ("0001/00010003", 2)])
        self.assertEqual([(r["cad_index"], r["text_index"]) for r in queries], [(0, 0), (2, 0)])
        self.assertEqual([r["text"] for r in texts], ["part"])
        self.assertEqual([(r["uid"], r["candidate_uid"]) for r in qrels],
                         [("0001/00010001", "0001/00010001"), ("0001/00010003", "0001/00010003")])
        np.testing.assert_array_equal(fields, [5, 4, 6])


if __name__ == "__main__":
    unittest.main()
