"""Bind completed Fusion replay programs to a portable likelihood-scoring view."""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from eval_retrieval_benchmark_fusion import CONDITION, DOMAIN, read_fusion_condition
from prepare_retrieval_benchmark_fill import (
    ASSETS, BACKGROUND_ORDER_SHA256, load_json, read_jsonl, sha256, text_digest,
    verify_sha256, write_json, write_jsonl,
)


_REPO = Path(__file__).resolve().parents[2]
BACKGROUND_SPEC = (_REPO / ASSETS["background_spec"][1], ASSETS["background_spec"][2])
N_BACKGROUND = 48
PROGRAM_ARRAYS = {
    "cad_vec": (np.dtype("<i4"), (272, 2)),
    "flag_vec": (np.dtype("<i4"), (272,)),
    "index_vec": (np.dtype("<i4"), (272,)),
    "key_padding_mask": (np.dtype(np.bool_), (272, 2)),
}
TEXT_AXIS = f"texts_{CONDITION}.jsonl"


def read_pinned_background(path: Path, manifest: dict) -> list[dict]:
    spec_path, spec_sha = BACKGROUND_SPEC
    verify_sha256(spec_path, spec_sha)
    spec = load_json(spec_path)
    if (spec["selected_config"] != "logmix_l1" or spec["count"] != N_BACKGROUND
            or spec["split"] != "validation" or spec["text_field"] != "abstract"
            or spec["ordered_digest"] != BACKGROUND_ORDER_SHA256
            or manifest.get("background_config") != spec["selected_config"]
            or manifest.get("background_source_text_count") != N_BACKGROUND
            or manifest.get("background_ordered_digest") != BACKGROUND_ORDER_SHA256):
        raise ValueError("Fusion background disagrees with the frozen 48-text protocol")
    registered = spec["rows"]
    rows = read_jsonl(path)
    if len(registered) != N_BACKGROUND or len(rows) != N_BACKGROUND:
        raise ValueError("Expected all 48 registered background rows")
    for index, (row, pin) in enumerate(zip(rows, registered, strict=True)):
        text = row.get("text")
        if (row.get("index") != index or pin["index"] != index or row.get("uid") not in pin["validation_uids"]
                or not isinstance(text, str) or not text or text != text.strip()
                or row.get("text_sha256") != pin["text_sha256"] or text_digest(text) != pin["text_sha256"]):
            raise ValueError(f"Frozen background text, order or source UID differs at index {index}")
    if text_digest("\n".join(row["text_sha256"] for row in rows)) != BACKGROUND_ORDER_SHA256:
        raise ValueError("Frozen background ordered digest differs")
    return [{key: row[key] for key in ("index", "text", "text_sha256")} for row in rows]


def admitted_program_ledger(root: Path, manifest: dict) -> dict[str, dict]:
    replay = manifest["source_replay"]
    verify_sha256(root / "fusion_replay_pin.json", replay["replay_pin_sha256"])
    if load_json(root / "fusion_replay_receipt.json") != replay:
        raise ValueError("Fusion replay receipt differs from the final freeze")
    ledger_path = root / "fusion_object_evidence.jsonl"
    verify_sha256(ledger_path, replay["objects_sha256"])
    records = read_jsonl(ledger_path)
    uids = [record["uid"] for record in records]
    if (len(records) != replay["n_candidates"] or uids != sorted(set(uids))
            or Counter(record["outcome"] for record in records) != replay["outcomes"]
            or any(record["domain"] != DOMAIN or record["replay_pin_sha256"] != replay["replay_pin_sha256"]
                   for record in records)):
        raise ValueError("Incomplete or inconsistent Fusion object replay ledger")
    return {row["uid"]: row for row in records if row["outcome"] == "accepted"}


def checked_program_bytes(path: Path, evidence: dict, expected_fields: int) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != evidence["artifacts"]["arrays"]["sha256"]:
        raise ValueError(f"Fusion program SHA256 differs for {evidence['uid']}")
    with np.load(io.BytesIO(data), allow_pickle=False) as program:
        if len(program.files) != len(PROGRAM_ARRAYS) or set(program.files) != PROGRAM_ARRAYS.keys():
            raise ValueError(f"Unexpected program arrays for {evidence['uid']}")
        arrays = {name: program[name] for name in PROGRAM_ARRAYS}
    for name, (dtype, shape) in PROGRAM_ARRAYS.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError(f"Invalid {name} dtype or shape for {evidence['uid']}")
    cad_vec = arrays["cad_vec"]
    if hashlib.sha256(cad_vec.tobytes(order="C")).hexdigest() != evidence["cad_vec_sha256"]:
        raise ValueError(f"Native cad_vec SHA256 differs for {evidence['uid']}")
    if not np.array_equal(arrays["key_padding_mask"], cad_vec == 0):
        raise ValueError(f"Padding mask differs from the native CAD sequence for {evidence['uid']}")
    if (type(evidence["candidate_fields"]) is not int or evidence["candidate_fields"] != expected_fields
            or np.count_nonzero(cad_vec[1:]) != expected_fields):
        raise ValueError(f"Fusion valid field count differs for {evidence['uid']}")
    return data


def export_fusion_score_inputs(root: Path, freeze_sha256: str, background_path: Path, output: Path) -> dict:
    root = root.resolve()
    frozen_path = root / "frozen"
    if any(output.resolve().is_relative_to(directory) for directory in (frozen_path, root / "objects")):
        raise ValueError("Fusion score inputs must stay outside the frozen and replay-object directories")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Score input directory already exists: {output}")
    frozen, fields = read_fusion_condition(frozen_path, freeze_sha256)
    background = read_pinned_background(background_path, frozen.manifest)
    admitted = admitted_program_ledger(root, frozen.manifest)
    if list(admitted) != [row["uid"] for row in frozen.gallery]:
        raise ValueError("Accepted Fusion replay UIDs differ from the dense gallery axis")
    if any(set(row) != {"text_index", "text", "text_sha256"} for row in frozen.texts):
        raise ValueError("Query scoring axis must contain only original indexed texts and their hashes")
    captions = {row["uid"]: frozen.texts[row["text_index"]]["text"] for row in frozen.queries}
    n_candidates, n_texts = len(frozen.gallery), len(frozen.texts)
    n_rows = n_texts + N_BACKGROUND
    role_rows = [
        {"row_index": offset + index, "role": role, "axis_index": index, "text_sha256": row["text_sha256"]}
        for role, offset, axis in (("query", 0, frozen.texts), ("background", n_texts, background))
        for index, row in enumerate(axis)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fusion-score-inputs-", dir=output.parent) as temp:
        stage = Path(temp) / "payload"
        stage.mkdir()
        columns = []
        program_pins = {}
        for row, n_valid in zip(frozen.gallery, fields, strict=True):
            uid, member = row["uid"], row["member"]
            evidence = admitted[uid]
            artifact = evidence["artifacts"]["arrays"]
            if (evidence.get("h5_member") != row["h5_member"] or evidence.get("caption") != captions.get(uid)
                    or artifact["member"] != member
                    or evidence.get("source_duplicates") != [] or evidence.get("float_to_h5", {}).get("pass") is not True
                    or evidence.get("quantized_to_h5", {}).get("pass") is not True):
                raise ValueError(f"Inconsistent accepted Fusion replay evidence for {uid}")
            source = root / member
            if not source.resolve().is_relative_to(root) or source.is_symlink():
                raise ValueError(f"Invalid converted program path for {uid}")
            data = checked_program_bytes(source, evidence, int(n_valid))
            destination = stage / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            program_pins[member] = artifact["sha256"]
            columns.append({"uid": uid, "cad_index": row["cad_index"], "vector_relpath": member,
                            "n_valid": int(n_valid)})
        shutil.copyfile(frozen_path / TEXT_AXIS, stage / TEXT_AXIS)
        verify_sha256(stage / TEXT_AXIS, frozen.text_sha)
        write_jsonl(stage / "gallery.jsonl", columns)
        write_jsonl(stage / "background.jsonl", background)
        write_jsonl(stage / "score_rows.jsonl", role_rows)
        staged_pins = {path.relative_to(stage).as_posix(): sha256(path)
                       for path in sorted(stage.rglob("*")) if path.is_file()}
        if any(staged_pins.get(member) != digest for member, digest in program_pins.items()):
            raise ValueError("Staged program SHA256 differs from the accepted Fusion replay")
        result = {
            "schema_version": 1, "domain": DOMAIN, "condition": CONDITION, "kind": "fusion_score_only_inputs",
            "gallery_size": n_candidates, "unique_query_texts": n_texts, "background_count": N_BACKGROUND,
            "score_shape": [n_rows, n_candidates], "planned_pairs": n_rows * n_candidates,
            "pairs_by_role": {"query": n_texts * n_candidates, "background": N_BACKGROUND * n_candidates},
            "role_axes": {"query": TEXT_AXIS, "background": "background.jsonl"},
            "field_counts_sha256": hashlib.sha256(fields.tobytes(order="C")).hexdigest(),
            "source_pins": {
                "fusion_freeze_sha256": freeze_sha256, "gallery_axis_sha256": frozen.gallery_sha,
                "text_axis_sha256": frozen.text_sha, "gallery_fields_sha256": frozen.manifest["files"]["gallery_fields.npy"],
                "replay_pin_sha256": frozen.manifest["source_replay"]["replay_pin_sha256"],
                "replay_objects_sha256": frozen.manifest["source_replay"]["objects_sha256"],
                "background_spec_sha256": BACKGROUND_SPEC[1], "background_source_sha256": sha256(background_path),
                "background_ordered_digest": BACKGROUND_ORDER_SHA256,
            },
            "score_inputs_sha256": staged_pins,
        }
        write_json(stage / "freeze.json", result)
        stage.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-run", required=True, type=Path, help="Completed Fusion CPU preparation run")
    parser.add_argument("--freeze-sha256", required=True, help="Verified SHA-256 of the final frozen/freeze.json")
    parser.add_argument("--background-rows", required=True, type=Path, help="Local E1 background.jsonl with pinned texts")
    parser.add_argument("--output", required=True, type=Path, help="New scoring input directory")
    args = parser.parse_args()
    result = export_fusion_score_inputs(args.fusion_run, args.freeze_sha256, args.background_rows, args.output)
    print(f"Bound {result['planned_pairs']} Fusion score pairs to {args.output / 'freeze.json'}")


if __name__ == "__main__":
    main()
