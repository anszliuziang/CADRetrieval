import numpy as np
import pytest

from experiments.src.idea01.in_domain_pilot import (
    evaluate_rankings, select_background, select_panel, sha256_lines,
)
from experiments.src.idea01.in_domain_pilot_ridge import tokenize_unigrams


def test_ranks_use_frozen_index_and_keep_individual_qrels():
    gallery = [{"uid": "a", "cad_index": 5}, {"uid": "b", "cad_index": 2}, {"uid": "c", "cad_index": 9}]
    queries = [{"uid": uid, "level": "CT", "text": "the same short sentence"} for uid in "abc"]
    ranks, metrics = evaluate_rankings(
        np.array([[0.0, 0.0, np.nan], [0.0, 0.0, np.nan], [0.0, 1.0, np.nan]]), queries, gallery)
    assert [row["rank"] for row in ranks] == [2, 1, 4]
    assert metrics["CT"]["hits"]["1"] == 1
    assert metrics["CT"]["R@1"] == pytest.approx(100 / 3)
    assert metrics["CT"]["failures"] == 1
    assert metrics["CT"]["hits"]["20"] == 2


def test_panel_targets_are_included_with_stable_axis():
    validation = [f"0000/{i:08d}" for i in range(3000)]
    targets, gallery = select_panel(validation, "t2cad-v1.0")
    assert len(targets) == 512
    assert len(gallery) == 2048
    assert set(targets) <= set(gallery)
    assert gallery == sorted(gallery)
    assert (targets, gallery) == select_panel(validation[::-1], "t2cad-v1.0")
    assert targets != select_panel(validation, "ct-deepcad")[0]


def test_background_groups_text_before_hashing():
    rows = {"a": {"CT": "same"}, "b": {"CT": "same"}, "c": {"CT": "third"}}
    background = select_background(rows, "ct-deepcad")
    assert {row["text"]: row["exact_count"] for row in background} == {"same": 2, "third": 1}
    assert sha256_lines(row["text"] for row in background) == sha256_lines(
        row["text"] for row in select_background(dict(reversed(list(rows.items()))), "ct-deepcad"))


def test_number_strings_are_unigrams():
    assert tokenize_unigrams("one -0.25-mm hole at 1.40 and 2e-4") == [
        "one", "-0.25", "mm", "hole", "at", "1.40", "and", "2e-4",
    ]
