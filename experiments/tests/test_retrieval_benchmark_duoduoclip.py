import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "scripts"))


class DuoduoCLIPRenderTests(unittest.TestCase):
    def renderer(self):
        name = "render_retrieval_benchmark_duoduoclip"
        self.assertIsNotNone(importlib.util.find_spec(name), "missing DuoduoCLIP renderer")
        return importlib.import_module(name)

    def test_nearest_fragment_wins_in_both_draw_orders(self):
        render = self.renderer()
        triangles = np.array([[[2, 2, 1], [10, 2, 1], [2, 10, 1]],
                              [[2, 2, -1], [10, 2, -1], [2, 10, -1]]], dtype=float)
        colors = np.array([[22, 33, 44], [55, 66, 77]], dtype=np.uint8)
        first = render.rasterize(triangles, colors, 12, [255, 255, 255])
        np.testing.assert_array_equal(first[4, 4], [22, 33, 44])
        np.testing.assert_array_equal(first[9, 9], [255, 255, 255])
        np.testing.assert_array_equal(first, render.rasterize(triangles[::-1], colors[::-1], 12, [255, 255, 255]))

    def test_cube_corners_keep_up_and_render_the_visible_faces(self):
        render = self.renderer()
        spec_path = EXPERIMENTS / "configs" / "retrieval_benchmark_duoduoclip.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))["render"]
        # Three unequal, offset faces distinguish all eight orientations.
        triangles = np.array([[[0, 0, 0], [.7, 0, 0], [0, .4, 0]],
                              [[0, 0, 0], [0, .4, 0], [0, 0, .3]],
                              [[0, 0, 0], [0, 0, .3], [.7, 0, 0]]])
        images = [render.render_view(triangles, direction, spec) for direction in spec["view_directions"]]
        self.assertEqual(len({image.tobytes() for image in images}), 8)
        for image in images:
            self.assertEqual(image.shape, (448, 448, 3))
            np.testing.assert_array_equal(image[0, 0], [255, 255, 255])
            self.assertTrue(np.any(image != 255))


if __name__ == "__main__":
    unittest.main()
