"""Check the portable Fusion likelihood inputs and the separation from qrels."""

import hashlib

import numpy as np
import pytest

from test_retrieval_benchmark_fusion_eval import fusion_freeze, repin

import prepare_retrieval_benchmark_fusion_score_inputs as binder
from prepare_retrieval_benchmark_fill import load_json, read_jsonl, sha256, text_digest, write_json, write_jsonl


def pin_ledger(root, frozen, ledger):
    path = root / "fusion_object_evidence.jsonl"
    write_jsonl(path, ledger)
    freeze = load_json(frozen / "freeze.json")
    freeze["source_replay"]["objects_sha256"] = sha256(path)
    write_json(root / "fusion_replay_receipt.json", freeze["source_replay"])
    return repin(frozen, freeze)


def program(n_valid):
    cad = np.zeros((272, 2), dtype="<i4")
    cad[0, 0] = 1
    cad[1:n_valid, 0] = 11
    cad[n_valid, 0] = 1
    flags = np.zeros(272, dtype="<i4")
    flags[n_valid + 1:] = 11
    indices = np.zeros(272, dtype="<i4")
    indices[n_valid + 1:] = 1
    return {"cad_vec": cad, "flag_vec": flags, "index_vec": indices, "key_padding_mask": cad == 0}


@pytest.fixture
def score_sources(tmp_path, fusion_freeze, monkeypatch):
    frozen, _ = fusion_freeze
    gallery = read_jsonl(frozen / "gallery.jsonl")
    texts = read_jsonl(frozen / "texts_ct_fusion.jsonl")
    queries = read_jsonl(frozen / "queries_ct_fusion.jsonl")
    captions = {row["uid"]: texts[row["text_index"]]["text"] for row in queries}
    background = [
        {"index": i, "uid": f"0002/{20000 + i:08}", "text": f"reference prompt {i}",
         "text_sha256": text_digest(f"reference prompt {i}")}
        for i in range(48)
    ]
    spec = {
        "selected_config": "logmix_l1", "count": 48, "split": "validation", "text_field": "abstract",
        "ordered_digest": text_digest("\n".join(row["text_sha256"] for row in background)),
        "rows": [{"index": row["index"], "validation_uids": [row["uid"]], "text_sha256": row["text_sha256"]}
                 for row in background],
    }
    spec_path = tmp_path / "background_spec.json"
    write_json(spec_path, spec)
    monkeypatch.setattr(binder, "BACKGROUND_SPEC", (spec_path, sha256(spec_path)))
    monkeypatch.setattr(binder, "BACKGROUND_ORDER_SHA256", spec["ordered_digest"])
    background_path = tmp_path / "background_source.jsonl"
    write_jsonl(background_path, background)
    replay_pin = tmp_path / "fusion_replay_pin.json"
    write_json(replay_pin, {"scope": "full", "complete_source_index": True})
    replay_sha = sha256(replay_pin)
    manifest = load_json(frozen / "freeze.json")
    manifest.update(background_config="logmix_l1", background_source_text_count=48,
                    background_ordered_digest=spec["ordered_digest"])
    manifest["source_replay"]["replay_pin_sha256"] = replay_sha
    repin(frozen, manifest)
    ledger = []
    fields = np.load(frozen / "gallery_fields.npy", allow_pickle=False)
    for row, n_valid in zip(gallery, fields, strict=True):
        file = tmp_path / row["member"]
        file.parent.mkdir(parents=True)
        arrays = program(int(n_valid))
        np.savez(file, **arrays)
        ledger.append({
            "uid": row["uid"], "caption": captions.get(row["uid"]), "domain": "Fusion360",
            "replay_pin_sha256": replay_sha, "outcome": "accepted", "h5_member": row["h5_member"],
            "source_duplicates": [], "float_to_h5": {"pass": True}, "quantized_to_h5": {"pass": True},
            "candidate_fields": int(n_valid), "cad_vec_sha256": hashlib.sha256(arrays["cad_vec"].tobytes()).hexdigest(),
            "artifacts": {"arrays": {"member": row["member"], "sha256": sha256(file)}},
        })
    for i in range(len(gallery), 6757):
        ledger.append({"uid": f"0001/{10000 + i:08}", "domain": "Fusion360",
                       "replay_pin_sha256": replay_sha, "outcome": "quantized_geometry"})
    return tmp_path, background_path, pin_ledger(tmp_path, frozen, ledger), ledger


def test_portable_score_inputs_hold_order_and_omit_positive_links(score_sources):
    root, background, pin, _ = score_sources
    output = root / "score_view"
    result = binder.export_fusion_score_inputs(root, pin, background, output)
    assert result == load_json(output / "freeze.json")
    assert result["score_shape"] == [51, 6]
    assert result["planned_pairs"] == 306
    gallery = read_jsonl(output / "gallery.jsonl")
    assert [row["cad_index"] for row in gallery] == list(range(6))
    assert [row["n_valid"] for row in gallery] == [8, 2, 2, 9, 6, 1]
    assert all(set(row) == {"uid", "cad_index", "vector_relpath", "n_valid"} for row in gallery)
    assert all((output / row["vector_relpath"]).read_bytes() == (root / row["vector_relpath"]).read_bytes()
               for row in gallery)
    assert sha256(out := output / "texts_ct_fusion.jsonl") == sha256(root / "frozen" / out.name)
    assert all(set(row) == {"index", "text", "text_sha256"} for row in read_jsonl(output / "background.jsonl"))
    role_rows = read_jsonl(output / "score_rows.jsonl")
    assert [row["row_index"] for row in role_rows] == list(range(51))
    assert [row["role"] for row in role_rows] == ["query"] * 3 + ["background"] * 48
    assert [row["axis_index"] for row in role_rows] == list(range(3)) + list(range(48))
    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert files == {"freeze.json", "gallery.jsonl", "texts_ct_fusion.jsonl", "background.jsonl", "score_rows.jsonl",
                     *(row["vector_relpath"] for row in gallery)}
    assert set(result["score_inputs_sha256"]) == files - {"freeze.json"}
    assert all(sha256(output / name) == digest for name, digest in result["score_inputs_sha256"].items())


@pytest.mark.parametrize("change, message", [
    ("payload_sha", "SHA256"),
    ("dtype", "flag_vec"),
    ("valid_fields", "valid field"),
    ("background", "background"),
])
def test_score_input_mismatch_does_not_publish(score_sources, change, message):
    root, background, pin, ledger = score_sources
    payload = ledger[0]["artifacts"]["arrays"]
    file = root / payload["member"]
    if change == "payload_sha":
        np.savez(file, **program(8))
        file.write_bytes(file.read_bytes() + b"tampered")
    elif change in {"dtype", "valid_fields"}:
        arrays = program(7 if change == "valid_fields" else 8)
        if change == "dtype":
            arrays["flag_vec"] = arrays["flag_vec"].astype(np.int64)
        np.savez(file, **arrays)
        payload["sha256"] = sha256(file)
        ledger[0]["cad_vec_sha256"] = hashlib.sha256(arrays["cad_vec"].tobytes()).hexdigest()
        pin = pin_ledger(root, root / "frozen", ledger)
    else:
        rows = read_jsonl(background)
        rows[0]["text"] = "unapproved text"
        write_jsonl(background, rows)
    output = root / "unpublished"
    with pytest.raises(ValueError, match=message):
        binder.export_fusion_score_inputs(root, pin, background, output)
    assert not output.exists()
