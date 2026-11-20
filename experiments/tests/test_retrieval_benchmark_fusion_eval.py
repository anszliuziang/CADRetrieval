"""Exercise the final Fusion reference run on a small frozen gallery."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_retrieval_benchmark_fusion import export_fusion_references
from prepare_retrieval_benchmark_fill import (
    freeze_condition, load_json, read_jsonl, sha256, write_json, write_jsonl,
)


def repin(directory, manifest):
    manifest["files"] = {
        path.name: sha256(path) for path in directory.iterdir() if path.name != "freeze.json"
    }
    write_json(directory / "freeze.json", manifest)
    return sha256(directory / "freeze.json")


@pytest.fixture
def fusion_freeze(tmp_path):
    directory = tmp_path / "frozen"
    directory.mkdir()
    uids = [f"0001/{10000 + index:08d}" for index in range(6)]
    gallery = [
        {"uid": uid, "cad_index": index, "member": f"objects/{uid.replace('/', '_')}/program.npz",
         "h5_member": f"dataset/fusion_360_txt/cad_vec/{uid}.h5"}
        for index, uid in enumerate(uids)
    ]
    texts, queries = freeze_condition(gallery, dict(zip(uids[:4], ("plate", "bracket", "plate", "ring"))))
    qrels = [{"uid": row["uid"], "candidate_uid": row["uid"], "relevance": 1} for row in queries]
    for name, rows in (("gallery.jsonl", gallery), ("texts_ct_fusion.jsonl", texts),
                       ("queries_ct_fusion.jsonl", queries), ("qrels.jsonl", qrels)):
        write_jsonl(directory / name, rows)
    np.save(directory / "gallery_fields.npy", np.array([8, 2, 2, 9, 6, 1], dtype="<i4"))
    manifest = {
        "schema_version": 1, "domain": "Fusion360", "gallery_count": 6,
        "gallery_order": "ascending Fusion UID", "candidate_member_root": "preparation_run_directory",
        "candidate_program_source": "converted_h5",
        "conditions": {"ct_fusion": {
            "queries": 4, "unique_texts": 3, "largest_group": 2, "singleton_queries": 2,
        }},
        "source_replay": {"scope": "full", "n_candidates": 6757, "complete_source_index": True,
                          "replay_pin_sha256": "a" * 64, "objects_sha256": "b" * 64,
                          "outcomes": {"accepted": 6, "quantized_geometry": 6751}},
    }
    return directory, repin(directory, manifest)


def test_references_keep_no_caption_distractors_and_tie_break_by_uid(fusion_freeze, tmp_path):
    frozen, pin = fusion_freeze
    output = tmp_path / "references"
    result = export_fusion_references(frozen, pin, output)
    assert result == load_json(output / "references.json")
    assert result["table"] == 3
    assert (result["gallery"], result["queries"], result["distinct_original_texts"]) == (6, 4, 3)
    assert result["random"]["expected_hits"] == pytest.approx([4/6, 20/6, 4, 4])
    assert result["random"]["population_median_rank"] == 3.5
    assert result["random"]["singleton"]["queries"] == 2
    assert result["random"]["singleton"]["expected_hits"] == pytest.approx([2/6, 10/6, 2, 2])
    assert result["shortest_first"]["hits"] == [0, 3, 4, 4]
    assert result["shortest_first"]["median_rank"] == 4
    assert result["shortest_first"]["singleton"]["hits"] == [0, 1, 2, 2]
    assert result["same_text_upper_bound"]["hits"] == [3, 4, 4, 4]
    assert result["same_text_upper_bound"]["singleton"]["hits"] == [2, 2, 2, 2]
    ranks = read_jsonl(output / "shortest_first_query_ranks.jsonl")
    assert [row["rank"] for row in ranks] == [5, 2, 3, 6]
    assert all(row["domain"] == "Fusion360" and not row["failed"] for row in ranks)
    assert result["shortest_first"]["query_ranks_sha256"] == sha256(output / "shortest_first_query_ranks.jsonl")


@pytest.mark.parametrize("change, error", [
    ("manifest_pin", "SHA256"),
    ("payload_sha", "SHA256"),
    ("extra", "file coverage"),
    ("pilot", "full Fusion"),
    ("eligible_count", "eligible"),
    ("gallery_order", "gallery"),
    ("text_hash", "text axis"),
    ("query_link", "query axis"),
    ("qrels_link", "qrels"),
    ("field_shape", "field count"),
    ("field_range", "field count"),
])
def test_invalid_freeze_fails_before_publishing(fusion_freeze, tmp_path, change, error):
    directory, pin = fusion_freeze
    manifest = load_json(directory / "freeze.json")
    if change == "manifest_pin":
        pin = "0" * 64
    elif change == "extra":
        (directory / "unregistered.jsonl").touch()
    elif change in {"eligible_count", "pilot"}:
        if change == "pilot":
            manifest["source_replay"]["scope"] = "pilot"
        else:
            manifest["source_replay"]["outcomes"] = {"accepted": 5, "quantized_geometry": 6752}
        pin = repin(directory, manifest)
    elif change in {"payload_sha", "field_shape"}:
        np.save(directory / "gallery_fields.npy", np.array([8, 2, 2, 9, 6], dtype="<i4"))
        if change == "field_shape":
            pin = repin(directory, manifest)
    elif change == "field_range":
        np.save(directory / "gallery_fields.npy", np.array([8, 2, 2, 9, 6, 543], dtype="<i4"))
        pin = repin(directory, manifest)
    else:
        name, field, value = {
            "gallery_order": ("gallery.jsonl", "cad_index", 1),
            "text_hash": ("texts_ct_fusion.jsonl", "text_sha256", "0" * 64),
            "query_link": ("queries_ct_fusion.jsonl", "cad_index", 2),
            "qrels_link": ("qrels.jsonl", "candidate_uid", "0001/00010001"),
        }[change]
        rows = read_jsonl(directory / name)
        rows[0][field] = value
        write_jsonl(directory / name, rows)
        pin = repin(directory, manifest)
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError, match=error):
        export_fusion_references(directory, pin, output)
    assert not output.exists()
