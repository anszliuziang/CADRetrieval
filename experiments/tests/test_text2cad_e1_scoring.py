import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_e1_scoring as scoring


def test_replay_requires_score_tolerance_and_exact_field_counts():
    reference = np.array([[-120.0, -2000.0]])
    counts = np.array([12, 32])
    observed = np.array([[-120.0009, -2000.0019]])
    receipt = scoring.compare_replay(observed, counts, reference, counts)
    assert receipt["passed"] and receipt["pairs"] == 2
    assert not scoring.compare_replay(observed, counts + 1, reference, counts)["passed"]
    observed[0, 1] = -2000.0021
    assert not scoring.compare_replay(observed, counts, reference, counts)["passed"]


def test_frozen_input_verification_rejects_drift(tmp_path):
    leaf = tmp_path / "queries_b.jsonl"
    leaf.write_bytes(b'{"uid":"x"}\n')
    manifest = {leaf.name: hashlib.sha256(leaf.read_bytes()).hexdigest()}
    assert scoring.input_path(tmp_path, manifest, leaf.name) == leaf.resolve()
    leaf.write_bytes(b'{"uid":"y"}\n')
    with pytest.raises(ValueError, match="digest"):
        scoring.input_path(tmp_path, manifest, leaf.name)


def test_shard_merge_rejects_missing_duplicate_or_mislabeled_rows(tmp_path):
    query = ["a", "b", "c"]
    columns = ["idea", "other"]
    expected = np.array([[3.0, 1.0], [5.0, 2.0], [-7.0, 6.0]])
    shard0, shard1 = tmp_path / "zero.npz", tmp_path / "one.npz"
    np.savez(shard0, rows=np.array([0, 2]), query_uids=np.array(["a", "c"]),
             cad_uids=np.array(columns), scores=expected[[0, 2]], counts=np.array([4, 9]))
    np.savez(shard1, rows=np.array([1]), query_uids=np.array(["b"]),
             cad_uids=np.array(columns), scores=expected[[1]], counts=np.array([4, 9]))
    matrix, counts = scoring.merge_shards([shard1, shard0], query, columns)
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(counts, [4, 9])
    with pytest.raises(ValueError, match="coverage"):
        scoring.merge_shards([shard0], query, columns)
    with pytest.raises(ValueError, match="duplicate"):
        scoring.merge_shards([shard0, shard0, shard1], query, columns)
    np.savez(shard1, rows=np.array([1]), query_uids=np.array(["a"]),
             cad_uids=np.array(columns), scores=expected[[1]], counts=np.array([4, 9]))
    with pytest.raises(ValueError, match="axis"):
        scoring.merge_shards([shard0, shard1], query, columns)


def test_real_share4_table_excludes_process_even_with_zero_core_and_remaps_ids():
    mapping = (
        " 2 0 0 4 Ascend910\n 2 1 1 5 Ascend910\n"
        " 3 0 2 6 Ascend910\n 3 1 3 7 Ascend910\n"
    )
    inventory = (
        "| 2 Ascend910 | OK | ... |\n"
        "| 0 4 | 0000:95:00.0 | 0 0 / 0 3206 / 65536 |\n"
        "| 2 Ascend910 | OK | ... |\n"
        "| 1 5 | 0000:97:00.0 | 0 0 / 0 2876 / 65536 |\n"
        "| 3 Ascend910 | OK | ... |\n"
        "| 0 6 | 0000:91:00.0 | 0 0 / 0 9138 / 65536 |\n"
        "| 3 Ascend910 | OK | ... |\n"
        "| 1 7 | 0000:93:00.0 | 59 0 / 0 20750/ 65536 |\n"
        "| NPU Chip | Process id | ... |\n"
        "| 2 0 | 999 | python3 | 125 |\n"
        "| 3 1 | 999 | python3 | 17929 |\n"
    )
    assert scoring.idle_from_smi(inventory, mapping) == [1]


def import_script(filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pending_rewrite_is_never_reported_as_adjudicated(tmp_path):
    freeze = import_script("prepare_text2cad_e1_freeze.py")
    path = tmp_path / "reviews.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="reviews still pending"):
        freeze.prepare_rewrites(path, ["case1"], {"case1": {"text_sha256": "hash", "text": "original"}})


def test_target_worker_needs_controller_and_successful_source_gate(tmp_path, monkeypatch):
    controller = import_script("run_text2cad_e1_score.py")
    out = tmp_path / "e1"
    (out / "target").mkdir(parents=True)
    (out / "source_evaluation").mkdir()
    auth = {"run_id": "e1", "freeze_sha256": "freeze"}
    args = SimpleNamespace(out=out, stage="target", device=1, shard=0, auth_sha="approval")
    argv = ["run.py", "worker"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(controller.os, "getppid", lambda: 42)
    scoring.write_json(out / "state.json", {"controller_pid": 42, "status": "scoring", "stage": "target"})
    scoring.write_json(out / "target/launch.json", {
        "controller_pid": 42, "argv": [["python", "-B", "-u", *argv]],
        "selected_logic_ids": [1], "auth_sha": "approval", "freeze_sha256": "freeze",
    })
    scoring.write_json(out / "replay_gate.json", {"passed": True})
    source = out / "source_evaluation/encoding_check.json"
    scoring.write_json(source, {"status": "fail"})
    with pytest.raises(ValueError, match="did not release C"):
        controller.require_worker_admission(args, auth)
    scoring.write_json(source, {"status": "inconclusive"})
    controller.require_worker_admission(args, auth)
