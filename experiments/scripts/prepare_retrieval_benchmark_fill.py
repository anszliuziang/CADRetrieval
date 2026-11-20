"""Freeze the common DeepCAD gallery and the five registered text conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np


LEVEL_FIELDS = {
    "l0": "abstract",
    "l1": "beginner",
    "l2": "intermediate",
    "l3": "expert",
}
CONDITIONS = (*LEVEL_FIELDS, "ct_deepcad")
EMPTY_ANNOTATIONS = {"", "nan", "n/a", "null", "<na>"}
COUNTS = {
    "l0": (7614, 3502, 2890, 354),
    "l1": (7614, 6300, 5978, 148),
    "l2": (7614, 7604, 7600, 7),
    "l3": (7614, 7614, 7614, 1),
    "ct_deepcad": (7513, 2029, 1431, 237),
}
INVENTORY_DIR = Path("reports/MMCAD_AND_CAD_UNDERSTANDING_2026-11-20")
G5_REPORT_DIR = Path("reports/G5_DEEP_ANALYSIS_2026-11-19")
OFFICIAL = Path(".cache/text2cad_official")
OFFICIAL_RELEASE = OFFICIAL / "text2cad_v1.0"
INVENTORY_NAME = "CADTRANSLATOR_RETRIEVAL_INVENTORY.csv"
ASSETS = {
    "inventory": ("repo", INVENTORY_DIR / INVENTORY_NAME,
                  "c201e4266895698c44b2f2f19e38bf15305bee306528b5d8c30566c1db3fdcd8"),
    "inventory_receipt": ("repo", INVENTORY_DIR / "CADTRANSLATOR_RETRIEVAL_INVENTORY.json",
                          "d9921c1ff70d8f096f31150f9411b4dbbbd1f0d80c5a75215446ce80479f544e"),
    "cadtranslator_dataset_zip": (
        "runtime", Path("benchmark_source_checks_20261120/cad_translator_dataset.zip"),
        "0c6c57a2266f6519bb43bc85c16d51450db38429237889faf4683043b1089170",
    ),
    "text2cad_prompt_csv": ("repo", OFFICIAL_RELEASE / "text2cad_v1.0.csv",
                            "32427ca7169f8b93411b64a025c387cdb0b55d9b9ce0e9cf9755ac8a3528f4a3"),
    "text2cad_split_json": ("repo", OFFICIAL_RELEASE / "train_test_val.json",
                            "4ea6db9027b4cebee8f37903b6601b73b810adacd808a4874dbe847c7c508e7b"),
    "text2cad_program_zip": ("repo", OFFICIAL / "cad_seq.zip",
                             "9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa"),
    "background_spec": (
        "repo", Path("experiments/configs/idea01_e1_fusion360_background_48_20261119.json"),
        "0adc647f0ca4282baf22a56962973a9e50cf065ec28fadc3af21f0b07520c4ea",
    ),
    "g5_full_gallery_jsonl": (
        "runtime", Path("text2cad_accuracy_night_20261118/g5_input_view/gallery.jsonl"),
        "09782bd6d316e7b0863c70f1df9cd6504e7dc41ce3bb9a433821b1c22ab85e30",
    ),
    "g5_common_axis": (
        "repo", Path("experiments/runs/idea01_openshape_g5_20261116/openshape_scores_sidecar.json"),
        "1c8ebc4616d652d3ad5c9e7755f3b33ebefdecf8bb6b86e8c39310561744640e",
    ),
    "g5_backgrounds": ("repo", G5_REPORT_DIR / "candidate_background_and_exposure.csv",
                       "427db34754666e9675f5b6914c2f837b0f77447c7e2e39652d37e16f7d39903e"),
    "g5_text_axis": ("runtime", Path("pair_manifest_v4/texts.jsonl"),
                     "c59b4aef8137b1e71c1581291a4c1d2bcbafe36f22d81d01431a5cc02dad7a21"),
}
CT_UID_PREFIX = re.compile(r"^/?([0-9]{4}/[0-9]{8})/")
BACKGROUND_ORDER_SHA256 = "44fa6f881cb46d9ab11dc7a1599495ba1a50b9d78551a978cc54f1e197aa8dfb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Unexpected SHA256 for {path}: {actual} (expected {expected})")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def parse_cadtranslator_texts(records, target_uids: set[str]) -> dict[str, str]:
    texts = {}
    for line_number, line in enumerate(records, 1):
        if not line.strip():
            continue
        prefix = CT_UID_PREFIX.match(line)
        if prefix is None:
            raise ValueError(f"CadTranslator text line {line_number} has an invalid UID prefix")
        uid = prefix[1]
        if uid in target_uids:
            text = line[prefix.end():].strip()
            if uid in texts:
                raise ValueError(f"Duplicate CadTranslator UID in one namespace: {uid}")
            texts[uid] = text
    return {uid: text for uid, text in texts.items() if text}


def native_texts(path: Path, gallery_uids: set[str]) -> dict[str, dict[str, str]]:
    levels = {condition: {} for condition in LEVEL_FIELDS}
    csv.field_size_limit(32 << 20)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            uid = row["uid"]
            if uid not in gallery_uids:
                continue
            texts = {level: row[field].strip() for level, field in LEVEL_FIELDS.items()}
            if any(text.casefold() in EMPTY_ANNOTATIONS for text in texts.values()):
                continue
            if uid in levels["l0"]:
                raise ValueError(f"Duplicate complete native Text2CAD annotation: {uid}")
            for level, text in texts.items():
                levels[level][uid] = text
    return levels


def freeze_condition(gallery: list[dict], text_by_uid: dict[str, str]):
    # Sorting by exact-text digest makes the scoring axis independent of positive CAD order.
    exact_texts = set(text_by_uid.values())
    by_digest = {text_digest(text): text for text in exact_texts}
    if len(by_digest) != len(exact_texts):
        raise ValueError("Different texts share a SHA256 digest")
    texts = [
        {"text_index": i, "text": by_digest[digest], "text_sha256": digest}
        for i, digest in enumerate(sorted(by_digest))
    ]
    index_by_text = {row["text"]: row["text_index"] for row in texts}
    queries = [
        {"uid": row["uid"], "cad_index": row["cad_index"],
         "text_index": index_by_text[text_by_uid[row["uid"]]]}
        for row in gallery if row["uid"] in text_by_uid
    ]
    if len(queries) != len(text_by_uid):
        raise ValueError("Query UIDs are missing in the gallery")
    return texts, queries


def frozen_gallery(paths: dict[str, Path], inventory: dict[str, dict]):
    full = read_jsonl(paths["g5_full_gallery_jsonl"])
    split = load_json(paths["text2cad_split_json"])
    full_uids = [row["cad_id"] for row in full]
    if len(full_uids) != 8046 or len(set(full_uids)) != 8046 or set(full_uids) != set(split["test"]):
        raise ValueError("Original G5 axis differs from the complete official test UID set")
    indices = load_json(paths["g5_common_axis"])["cad_indices"]
    if len(indices) != 8032 or indices != sorted(set(indices)):
        raise ValueError("Common gallery axis is incomplete or unordered")
    gallery = [
        {"cad_index": index, "uid": full[index]["cad_id"], "member": full[index]["cad_seq_member"]}
        for index in indices
    ]
    if {row["uid"] for row in gallery} != set(inventory):
        raise ValueError("Historical common-gallery UID set differs from the pinned inventory")
    if any(inventory[row["uid"]]["candidate_program_member"] != row["member"] for row in gallery):
        raise ValueError("Historical gallery program members differ from the pinned inventory")
    with ZipFile(paths["text2cad_program_zip"]) as release:
        members = Counter(entry.filename for entry in release.infolist())
        if any(members[row["member"]] != 1 for row in gallery):
            raise ValueError("Canonical program member missing or duplicated in the official ZIP")
    backgrounds = read_csv(paths["g5_backgrounds"])
    if [(int(row["cad_index"]), row["uid"]) for row in backgrounds] != list(enumerate(full_uids)):
        raise ValueError("Background CSV has a different historical CAD axis")
    background = np.array([float(backgrounds[i]["background_logmix"]) for i in indices], dtype="<f8")
    fields = np.array([int(backgrounds[i]["fields"]) for i in indices], dtype="<i4")
    if not np.all(np.isfinite(background)) or not np.all(fields > 0):
        raise ValueError("Invalid source background or field counts")
    spec = load_json(paths["background_spec"])
    actual_digest = text_digest("\n".join(row["text_sha256"] for row in spec["rows"]))
    if (spec["selected_config"] != "logmix_l1" or spec["count"] != 48
            or [row["index"] for row in spec["rows"]] != list(range(48))
            or spec["ordered_digest"] != actual_digest or actual_digest != BACKGROUND_ORDER_SHA256):
        raise ValueError("Background configuration differs from frozen S48")
    return gallery, background, fields


def verify_freeze(directory: Path) -> dict:
    manifest = load_json(directory / "freeze.json")
    names = {"gallery.jsonl", "background.npy", "gallery_fields.npy"} | {
        f"{kind}_{condition}.jsonl" for kind in ("texts", "queries") for condition in CONDITIONS
    }
    if set(manifest["files"]) != names:
        raise ValueError("Freeze manifest is incomplete")
    for name, digest in manifest["files"].items():
        verify_sha256(directory / name, digest)
    return manifest


def prepare(main_repo: Path, asset_runtime: Path, output: Path) -> dict:
    if output.exists():
        return verify_freeze(output)
    roots = {"repo": main_repo, "runtime": asset_runtime}
    paths = {key: roots[anchor] / path for key, (anchor, path, _) in ASSETS.items()}
    sources = {}
    for key, (_, _, digest) in ASSETS.items():
        verify_sha256(paths[key], digest)
        sources[key] = {"path": str(paths[key]), "sha256": digest}
    print("Pinned source files verified", flush=True)
    rows = [
        row for row in read_csv(paths["inventory"])
        if row["domain"] == "DeepCAD" and row["recommended_candidate_before_conversion"] == "1"
    ]
    inventory = {row["cad_uid"]: row for row in rows}
    if len(rows) != len(inventory) or len(inventory) != 8032:
        raise ValueError("Unexpected UID count in the DeepCAD inventory")
    gallery, background, fields = frozen_gallery(paths, inventory)
    with ZipFile(paths["cadtranslator_dataset_zip"]) as release:
        with release.open("dataset/deepcad_txt/text.txt") as source:
            translator = parse_cadtranslator_texts(
                io.TextIOWrapper(source, encoding="utf-8-sig"), set(inventory)
            )
    for uid, row in inventory.items():
        actual_digest = text_digest(translator[uid]) if uid in translator else ""
        if (actual_digest != row["cadtranslator_query_stripped_sha256"]
                or (uid in translator) != (row["recommended_query_before_conversion"] == "1")):
            raise ValueError(f"CadTranslator text differs from pinned inventory for {uid}")
    print("Gallery and CadTranslator query SHA values verified", flush=True)
    by_level = native_texts(paths["text2cad_prompt_csv"], set(inventory))
    expected_uids = {
        uid for uid, row in inventory.items() if row["all_four_native_text2cad_queries"] == "1"
    }
    if set(by_level["l0"]) != expected_uids or len(expected_uids & translator.keys()) != 7122:
        raise ValueError("Text2CAD native query set or CT/Text2CAD intersection differs from plan")
    historical_texts = {row["text"] for row in read_jsonl(paths["g5_text_axis"])}
    if len(historical_texts) != 3644 or set(by_level["l0"].values()) - historical_texts:
        raise ValueError("Native L0 texts do not match the historical G5 scoring text axis")
    by_level["ct_deepcad"] = translator
    l2_texts, ct_texts = set(by_level["l2"].values()), set(translator.values())
    if l2_texts & ct_texts or (l2_texts | ct_texts) & historical_texts:
        raise ValueError("Unexpected exact-text overlap with L2 or CT–DeepCAD")
    print("Official native query sets and overlap verified", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        # A private staging child lets the entire freeze become visible atomically.
        destination = Path(staging) / "payload"
        destination.mkdir()
        write_jsonl(destination / "gallery.jsonl", gallery)
        np.save(destination / "background.npy", background, allow_pickle=False)
        np.save(destination / "gallery_fields.npy", fields, allow_pickle=False)
        condition_stats = {}
        for condition, text_by_uid in by_level.items():
            texts, queries = freeze_condition(gallery, text_by_uid)
            counts = Counter(row["text_index"] for row in queries)
            stats = (len(queries), len(texts), sum(n == 1 for n in counts.values()), max(counts.values()))
            if stats != COUNTS[condition]:
                raise ValueError(f"Registered counts differ for {condition}: {stats}")
            write_jsonl(destination / f"texts_{condition}.jsonl", texts)
            write_jsonl(destination / f"queries_{condition}.jsonl", queries)
            condition_stats[condition] = dict(zip(
                ("queries", "unique_texts", "singleton_queries", "largest_group"), stats, strict=True,
            ))
        manifest = {
            "schema_version": 1,
            "domain": "DeepCAD",
            "gallery_count": len(gallery),
            "gallery_order": "historical cad_index ascending, with gaps preserved",
            "text_order": "SHA256 of the stripped UTF-8 original text ascending",
            "candidate_program_source": "text2cad_program_zip",
            "background_config": "logmix_l1",
            "background_source_text_count": 48,
            "background_ordered_digest": BACKGROUND_ORDER_SHA256,
            "paired_uid_count": len(expected_uids & translator.keys()),
            "conditions": condition_stats,
            "sources": sources,
            "files": {path.name: sha256(path) for path in sorted(destination.iterdir())},
        }
        write_json(destination / "freeze.json", manifest)
        verify_freeze(destination)
        destination.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--asset-runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(args.main_repo.resolve(), args.asset_runtime.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "gallery_count": manifest["gallery_count"],
                      "conditions": manifest["conditions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
