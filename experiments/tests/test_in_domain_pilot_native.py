import math

import pytest
import torch

from idea01.in_domain_pilot_native import (
    background_logmean,
    calibrated_scores,
    check_text2cad_flags,
    hiercad_timing_sample,
)


def test_native_text2cad_requires_the_training_time_part_index_and_flag():
    # Two short circle sketches followed by complete 11-row extrusions.
    program = [1]
    for _ in range(2):
        program += [11, 11, 5, 4, 3, 2] + [11] * 8 + [7, 11, 6]
    program += [1, 0, 0]
    flags = [0] + ([0] * 6 + [1, *range(1, 11)]) * 2 + [0, 11, 11]
    indices = [0] * 18 + [1] * 18 + [2, 2]
    sample = {
        "cad_vec": torch.tensor([[x, 0] for x in program]),
        "flag_vec": torch.tensor(flags),
        "index_vec": torch.tensor(indices),
        "n_real_tokens": len(program) - 2,
    }
    check_text2cad_flags(sample)
    sample["index_vec"][7] = 1
    with pytest.raises(ValueError, match="flags or indices"):
        check_text2cad_flags(sample)


def test_native_background_is_stable_and_fails_without_all_distinct_texts():
    rows = [
        {"status": "scored", "query_text_sha256": str(i), "candidate_i": 0,
         "s0": -10000.0 + (i == 5)}
        for i in range(48)
    ]
    expected = -10000.0 + math.log((47 + math.e) / 48)
    assert background_logmean(rows, 1) == [pytest.approx(expected)]
    rows[-1]["query_text_sha256"] = "0"
    assert background_logmean(rows, 1) == [None]


def test_native_calibration_keeps_fixed_cad_order_and_detects_target_token_drift():
    first = {
        "status": "scored", "query_i": 0, "query_text_sha256": "text-a", "candidate_i": 0,
        "target_sha256": "program", "target_token_sha256": "tokens", "s0": -10001.5,
        "effective_units": 252,
    }
    second = {
        **first, "query_i": 1, "query_text_sha256": "text-b", "s0": -10003.5,
        "target_token_sha256": "drift",
    }
    result = calibrated_scores({"pairs": [first, second]}, {"pairs": []}, [{"uid": "c"}], [-10010])
    assert [pair["s48"] for pair in result["pairs"]] == [8.5, 6.5]
    assert result["fixed_cad_invariance"]["passed"]
    assert result["candidate_target_invariance"][0]["passed"] is False



def test_native_timing_preserves_overlength_counts_and_uses_longest_supported():
    rows = [
        (7600, 0, {"uid": "short"}),
        (7900, 1, {"uid": "longest_legal"}),
        (8100, 2, {"uid": "partially_legal"}),
    ]
    queries = [
        {"query_i": 0, "uid": "q1", "prefix_tokens": 75},
        {"query_i": 1, "uid": "q2", "prefix_tokens": 220},
    ]
    selected, coverage = hiercad_timing_sample(rows, queries, 1, "long")
    assert selected == [{"uid": "longest_legal"}]
    assert coverage["expected_tokenizable_pairs"] == 6
    assert coverage["fits"] == 5
    assert coverage["overlength_pairs"] == [{
        "candidate_uid": "partially_legal",
        "pairs": [{"query_i": 1, "query_uid": "q2", "context_tokens": 8320}],
    }]
