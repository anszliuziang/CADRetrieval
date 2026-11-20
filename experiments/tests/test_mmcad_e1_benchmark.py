import numpy as np
import pytest

from experiments.scripts.run_mmcad_e1_20261120 import matched_panel


def test_partial_cad_coverage_changes_every_methods_ranks():
    queries = ["uid-a", "uid-b", "uid-c"]
    full_gallery = queries.copy()
    available = ["uid-a", "uid-c"]
    mmcad = np.array([[0.1, 0.9], [0.5, 0.5], [0.1, 0.9]])
    original = {"Text2CAD S0": np.array([[0.7, 0.9, 0.4], [0.2, 0.6, 0.3], [0.1, 0.8, 0.9]])}

    keep, ranks = matched_panel(queries, full_gallery, available, mmcad, original, "deepcad")

    assert keep.tolist() == [0, 2]
    assert ranks["MM-CAD v2"].tolist() == [2, 1]
    assert ranks["Text2CAD S0"].tolist() == [1, 1]


def test_complete_text_mask_filters_queries_but_keeps_distractors():
    queries = ["uid-a", "uid-b"]
    gallery = [*queries, "distractor"]
    scores = np.array([[0.5, 0.4, 0.9], [0.2, 0.7, 0.1]])

    keep, ranks = matched_panel(queries, gallery, gallery, scores, {"S0": scores}, "fusion360",
                                complete=np.array([True, False]))

    assert keep.tolist() == [0]
    assert ranks["MM-CAD v2"].tolist() == [2]
    assert ranks["S0"].tolist() == [2]


@pytest.mark.parametrize("available", [["uid-b", "uid-a"], ["uid-a", "invalid"]])
def test_rejects_drifted_candidate_axis(available):
    queries = ["uid-a", "uid-b"]
    scores = np.ones((2, 2))
    with pytest.raises(ValueError, match="candidate axis"):
        matched_panel(queries, queries, available, scores, {"S0": scores}, "deepcad")
