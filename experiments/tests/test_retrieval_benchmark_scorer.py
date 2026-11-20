from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_retrieval_benchmark_text2cad import (
    assert_equivalent, load_score_row, project_capacity, publish_score_row, score_row_path,
)


def test_reference_checks_every_score_and_exact_field_count():
    reference = np.array([-903.0, -60.0])
    counts = [17, 23]
    assert_equivalent(reference + 1e-4, counts, reference, counts)
    with pytest.raises(ValueError, match="score mismatch"):
        assert_equivalent(reference + 0.01, counts, reference, counts)
    with pytest.raises(ValueError, match="field counts"):
        assert_equivalent(reference, [17, 24], reference, counts)
    with pytest.raises(ValueError, match="score mismatch"):
        assert_equivalent(np.array([float("nan"), -60.0]), counts, reference, counts)


def test_completed_row_binds_bytes_and_frozen_text(tmp_path):
    path = score_row_path(tmp_path, "ct_deepcad", 42)
    path.parent.mkdir(parents=True)
    row = {"text_index": 42, "text_sha256": "a" * 64}
    values = np.array([-11.0, -12.0], dtype="<f8")
    publish_score_row(path, row, "b" * 64, values)
    assert np.array_equal(load_score_row(path, row, "b" * 64, 2), values)
    with pytest.raises(ValueError, match="receipt"):
        load_score_row(path, row, "d" * 64, 2)
    with pytest.raises(ValueError, match="receipt"):
        load_score_row(path, {"text_index": 42, "text_sha256": "c" * 64}, "b" * 64, 2)
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="receipt"):
        load_score_row(path, row, "b" * 64, 2)


def test_second_conflicting_writer_cannot_replace_the_score_row(tmp_path):
    path = score_row_path(tmp_path, "ct_deepcad", 42)
    path.parent.mkdir(parents=True)
    row = {"text_index": 42, "text_sha256": "a" * 64}
    values = np.array([-11.0, -12.0], dtype="<f8")
    publish_score_row(path, row, "b" * 64, values)
    publish_score_row(path, row, "b" * 64, values)
    with pytest.raises(ValueError, match="conflicting"):
        publish_score_row(path, row, "b" * 64, values + 1)
    assert np.array_equal(load_score_row(path, row, "b" * 64, 2), values)
    assert not list(path.parent.glob("*.tmp"))


def test_capacity_falls_back_to_measured_uncached_cost():
    measurements = [
        {"arm": "ct_deepcad", "cached_batch_seconds": [0.1] * 8, "uncached_batch_seconds": [0.2] * 8}
        for _ in range(3)
    ]
    probe = {"asset_preparation_seconds": 43.0, "measurements": measurements, "cache_passed": False,
             "peak_memory_allocated_bytes": 1024, "peak_memory_reserved_bytes": 2048}
    result = project_capacity(probe, ["ct_deepcad"], 3)
    assert result["card_seconds"] == pytest.approx(2029 * (8032 / 16) * 0.2 + 3 * 43)
    assert result["wall_seconds"] == pytest.approx(result["card_seconds"] / 3)
