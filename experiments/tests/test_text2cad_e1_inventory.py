import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_e1_inventory as inventory


def test_minimal_hash_excludes_description_and_keeps_execution_order():
    part = {"coordinate_system": {}, "sketch": {"face_2": {}, "face_1": {}},
            "extrusion": {"operation": "NewBodyFeatureOperation"}, "description": "first"}
    parsed = inventory.parse_minimal(json.dumps({"parts": {"part_2": part, "part_1": part}}), "deepcad")
    assert list(parsed["parts"]) == ["part_2", "part_1"]
    assert list(parsed["parts"]["part_2"]["sketch"]) == ["face_2", "face_1"]
    other = dict(reversed(list(part.items())))
    other["description"] = "second"
    fusion = inventory.parse_minimal(repr({"parts": {"part_1": other, "part_2": other}}), "fusion360")
    assert inventory.program_key(parsed) == inventory.program_key(fusion)
    assert inventory.program_key(parsed) != inventory.program_key({"parts": {"part_1": part}})
    with pytest.raises(ValueError, match="missing"):
        inventory.parse_minimal('{"parts":{"part_1":{"sketch":{}}}}', "deepcad")


def test_exact_train_vector_scan_records_missing_official_uid(tmp_path):
    import torch

    uid = "0001/00010000"
    missing = "0002/00020000"
    archive = tmp_path / "cad_seq.zip"
    cad_vec = np.arange(544, dtype=np.int64).reshape(272, 2)
    serial = io.BytesIO()
    torch.save({"vec": {"cad_vec": torch.from_numpy(cad_vec)}}, serial)
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr(inventory.official_member(uid), serial.getvalue())
    records, receipt = inventory.scan_train_archive(archive, [uid, missing], tmp_path / "hashes.jsonl")
    assert records == [{"uid": uid, "sha256": hashlib.sha256(cad_vec.astype("<i4").tobytes()).hexdigest()}]
    assert receipt["expected_uid_count"] == 2
    assert receipt["hashed_uid_count"] == 1
    assert receipt["coverage_complete"] is False
    assert receipt["failed_uids"] == [{"uid": missing, "reason": "missing_member"}]
    assert (tmp_path / "hashes.jsonl").read_bytes().endswith(b"\n")


def test_background_duplicate_uid_matches_same_frozen_text():
    digest = inventory.sha256_text("a part")
    freeze = {"count": 1, "ordered_digest": inventory.sha256_text(digest), "text_field": "abstract",
              "rows": [{"index": 0, "validation_uids": ["a", "b"], "text_sha256": digest}]}
    assert inventory.recover_background(freeze, {"a": "a part", "b": "a part"}, {"a", "b"}) == [
        {"uid": "a", "text": "a part", "text_sha256": digest, "index": 0}
    ]
    with pytest.raises(ValueError, match="background"):
        inventory.recover_background(freeze, {"a": "a part", "b": "wrong"}, {"a", "b"})


def test_train_text_index_compares_strip_only_and_all_wording_levels(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("uid,abstract,beginner,intermediate,expert\n"
                    "train,  abstract , L0 ,E1! ,expert\n"
                    "test,L0,unused,unused,unused\n", encoding="utf-8")
    l0, matches, receipt = inventory.source_csv_indexes(path, {"train"}, {"test"}, {"E1!", "expert", "Expert"})
    assert l0 == {"test": "L0"}
    assert matches["L0"] == [{"source": "text2cad_v1.0", "field": "beginner", "uid": "train"}]
    assert matches["E1!"][0]["field"] == "intermediate"
    assert matches["expert"][0]["field"] == "expert"
    assert "Expert" not in matches
    assert receipt["found_uid_count"] == 1


def test_nearest_train_matches_order_ties_by_program_hash():
    import numpy as np

    query = np.asarray([[0.0, 0.0], [5.0, 5.0]])
    # The input training axis is already ordered by (SHA, UID).
    train = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [5.0, 5.0]])
    ranked = list(inventory.nearest_feature_indexes(query, train, k=2))
    assert ranked[0] == [(0, 1.0), (1, 1.0)]
    assert ranked[1] == [(3, 0.0), (0, 41.0)]


def test_coarse_features_distinguish_sketch_aspect_with_identical_counts():
    def part(width, height):
        return {"parts": {"p": {
            "coordinate_system": {"Translation Vector": [0, 0, 0]},
            "extrusion": {"operation": "NewBodyFeatureOperation", "sketch_scale": 4,
                          "extrude_depth_towards_normal": 1, "extrude_depth_opposite_normal": 0},
            "sketch": {"f": {"l": {"line_1": {"Start Point": [0, 0], "End Point": [width, height]}}}},
        }}}

    square = dict(zip(inventory.NEAR_FEATURES, inventory.coarse_program_features(part(4, 4)), strict=True))
    tall = dict(zip(inventory.NEAR_FEATURES, inventory.coarse_program_features(part(1, 4)), strict=True))
    assert square["log1p_primitive_count:line"] == tall["log1p_primitive_count:line"]
    assert square["mean_sketch_aspect"] == 1
    assert tall["mean_sketch_aspect"] == 0.25


def test_program_corroboration_keeps_small_deltas_and_coarse_collisions_suspected():
    def signature(x, operation="JoinFeatureOperation"):
        program = {"parts": {"part_1": {"coordinate_system": {"x": x}, "sketch": {},
                                        "extrusion": {"operation": operation}}}}
        return inventory.program_signature(inventory.program_key(program))

    query = signature(0.1)
    exact = inventory.corroborate_program_pair(query, signature(0.1), 30)
    small = inventory.corroborate_program_pair(query, signature(0.101), 10)
    collision = inventory.corroborate_program_pair(query, signature(0.4, "CutFeatureOperation"), 0)
    unrelated = inventory.corroborate_program_pair(query, signature(0.4), 1000)
    assert exact["status"] == "confirmed"
    assert small["status"] == "suspected" and small["matching_topology"]
    assert small["max_abs_field_delta"] == pytest.approx(0.001)
    assert collision["status"] == "suspected" and not collision["matching_topology"]
    assert unrelated["status"] == "no_corroboration"
