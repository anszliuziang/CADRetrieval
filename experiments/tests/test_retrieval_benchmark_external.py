import hashlib
import json
import sys

import numpy as np
import pytest

from experiments.scripts import encode_retrieval_benchmark_external as enc


def test_mmcad_cli_accepts_the_historical_vector_cache(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(enc, "encode", captured.append)
    argv = ["encode_retrieval_benchmark_external.py", "encode", "--method", "mmcad_v2"]
    for name in ("frozen", "replay", "output", "upstream", "checkpoint", "stats", "text-model",
                 "vector-cache", "cache-checksums"):
        argv += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", argv)
    enc.main()
    assert captured[0].vector_cache == tmp_path / "vector-cache"


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def text_row(index, text):
    return {"text_index": index, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def test_frozen_axes_preserve_sparse_candidate_indices_and_exclude_qrels(tmp_path):
    gallery_path = tmp_path / "gallery.jsonl"
    text_path = tmp_path / "texts_l2.jsonl"
    gallery = [{"cad_index": 5, "uid": "0000/00000001", "member": "one"},
               {"cad_index": 9, "uid": "0000/00000002", "member": "two"}]
    texts = [text_row(0, "one"), text_row(1, "two")]
    write_rows(gallery_path, gallery)
    write_rows(text_path, texts)
    assert enc.load_gallery(gallery_path)[0] == gallery
    assert enc.load_texts(text_path)[0] == texts
    write_rows(gallery_path, gallery[::-1])
    with pytest.raises(ValueError, match="ordered"):
        enc.load_gallery(gallery_path)
    texts[1]["target_uid"] = gallery[0]["uid"]
    write_rows(text_path, texts)
    with pytest.raises(ValueError, match="fields"):
        enc.load_texts(text_path)


def test_replay_links_original_text_hash_and_historical_candidate_id(tmp_path):
    gallery_path = tmp_path / "gallery.jsonl"
    text_path = tmp_path / "texts_l0.jsonl"
    hist_gallery = tmp_path / "history_gallery.jsonl"
    hist_text = tmp_path / "history_text.jsonl"
    matrix = tmp_path / "history.npy"
    sidecar = tmp_path / "history.json"
    uids = ["0000/00000001", "0000/00000002", "0000/00000003"]
    write_rows(gallery_path, [{"cad_index": i, "uid": uids[i], "member": str(i)} for i in (0, 2)])
    write_rows(text_path, [text_row(0, "long text"), text_row(1, "short")])
    write_rows(hist_gallery, [{"cad_id": uid} for uid in uids])
    write_rows(hist_text, [{**text_row(0, "short"), "n_chars": 5}, text_row(1, "long text")])
    vectors = np.array([[0.0, 0.2], [0.3, 0.4]], dtype="<f8")
    matrix.write_bytes(vectors.tobytes())
    sidecar.write_text(json.dumps({"schema": "text2cad-method-scores-sidecar-v2", "direction": "higher",
                                  "dtype": "float64", "text_indices": [0, 1], "cad_indices": [0, 2],
                                  "matrix_sha256": enc.file_sha256(matrix), "matrix_bytes": matrix.stat().st_size}))
    out = tmp_path / "replay"
    enc.prepare_replay(gallery_path, text_path, hist_gallery, hist_text, matrix, sidecar, out,
                       expected_sidecar_sha256=enc.file_sha256(sidecar))
    meta = json.loads((out / "replay.json").read_text())
    assert [row["historical_text_index"] for row in meta["texts"]] == [0, 1]
    np.testing.assert_array_equal(np.load(out / "scores.npy"), vectors)
    write_rows(hist_gallery, [{"cad_id": uid} for uid in uids[::-1]])
    with pytest.raises(ValueError, match="historical candidate"):
        enc.prepare_replay(gallery_path, text_path, hist_gallery, hist_text, matrix, sidecar,
                           tmp_path / "bad", expected_sidecar_sha256=enc.file_sha256(sidecar))


def test_replay_reports_near_tie_swaps_and_rejects_larger_cosine_error():
    cad_indices = np.array([2, 5, 10])
    expected = np.array([[0.8, 0.8, 0.3]])
    observed = np.array([[0.7999999, 0.8, 0.3]])
    close = enc.compare_replay(expected, observed, cad_indices, atol=1e-6)
    assert close["passed"]
    assert close["rank_changes"] == 2
    assert close["historical_near_tie_adjacent_swaps"] == 1
    observed[0, 2] = 0.6
    assert not enc.compare_replay(expected, observed, cad_indices, atol=1e-6)["passed"]


def test_failed_embeddings_leave_zero_rows_and_preserve_text_truncation(tmp_path):
    axis = [text_row(0, "long text"), text_row(1, "short")]
    raw = np.array([[3.0, 4.0], [float("nan"), 0.0]])
    vectors, status = enc.normalize_rows(raw, axis, [{"truncated": True}, {"truncated": False}], 0.0)
    assert vectors.dtype == np.float64
    np.testing.assert_array_equal(vectors, [[0.6, 0.8], [0.0, 0.0]])
    assert [(row["valid"], row["truncated"]) for row in status] == [(True, True), (False, False)]
    assert status[1]["reason"]
    enc.write_axis(tmp_path, "texts_l3", vectors, status, {"axis_sha256": "a" * 64})
    meta = json.loads((tmp_path / "texts_l3_meta.json").read_text())
    assert meta["axis_sha256"] == "a" * 64
    assert meta["vectors_sha256"] == enc.file_sha256(tmp_path / "texts_l3.npy")
    assert meta["status_sha256"] == enc.file_sha256(tmp_path / "texts_l3_status.jsonl")
