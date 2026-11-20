"""Freeze the source train/validation pairs for the supervised retrieval models."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_e1_inventory import EXPECTED_SOURCE_SPLIT, check_split, official_member
from prepare_retrieval_benchmark_fill import (
    ASSETS, EMPTY_ANNOTATIONS, LEVEL_FIELDS, load_json, parse_cadtranslator_texts,
    sha256, verify_sha256, write_json,
)


FOLDS = ("train", "validation")
ASSET_NAMES = ("text2cad_prompt_csv", "text2cad_split_json",
               "text2cad_program_zip", "cadtranslator_dataset_zip")
CT_TEXT_MEMBER = "dataset/deepcad_txt/text.txt"


def source_rows(records, source_fold: dict[str, str], captions: dict[str, str]):
    seen = set()
    for row in records:
        uid = row["uid"]
        fold = source_fold.get(uid)
        if fold is None:
            continue
        if uid in seen:
            raise ValueError(f"Duplicate native source annotation UID: {uid}")
        seen.add(uid)
        native = {level: row[field].strip() for level, field in LEVEL_FIELDS.items()}
        native = {level: text for level, text in native.items() if text.casefold() not in EMPTY_ANNOTATIONS}
        if native:
            yield fold, {"uid": uid, "member": official_member(uid), "native_texts": native,
                         "ct_deepcad_text": captions.get(uid)}


def prepare(main_repo: Path, asset_runtime: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Source supervision freeze already exists: {output}")
    roots = {"repo": main_repo, "runtime": asset_runtime}
    assets = {}
    for name in ASSET_NAMES:
        anchor, relpath, digest = ASSETS[name]
        path = roots[anchor] / relpath
        verify_sha256(path, digest)
        assets[name] = path
    print("Source supervision inputs verified", flush=True)

    splits = check_split(load_json(assets["text2cad_split_json"]), EXPECTED_SOURCE_SPLIT)
    source_fold = {uid: name for name in FOLDS for uid in splits[name]}
    members = {official_member(uid) for uid in source_fold}
    with ZipFile(assets["text2cad_program_zip"]) as release:
        seen_members = Counter(item.filename for item in release.infolist() if item.filename in members)
    if seen_members.keys() != members or any(count != 1 for count in seen_members.values()):
        raise ValueError("Official train/validation programs are missing or ambiguous")
    del members, seen_members

    with ZipFile(assets["cadtranslator_dataset_zip"]) as release:
        if release.namelist().count(CT_TEXT_MEMBER) != 1:
            raise ValueError("CadTranslator DeepCAD captions member is missing or ambiguous")
        with release.open(CT_TEXT_MEMBER) as source:
            captions = parse_cadtranslator_texts(
                io.TextIOWrapper(source, encoding="utf-8-sig"), set(source_fold),
            )

    stats = {name: {"official_uids": len(splits[name]),
                    "source_ct_caption_uids": sum(uid in captions for uid in splits[name]),
                    "native_uids": 0, "native_four_level_uids": 0,
                    "common_native_and_ct_uids": 0,
                    "native_pairs_per_level": dict.fromkeys(LEVEL_FIELDS, 0),
                    "common_native_pairs_per_level": dict.fromkeys(LEVEL_FIELDS, 0)}
             for name in FOLDS}
    output.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(32 << 20)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        with ExitStack() as stack:
            outputs = {name: stack.enter_context((staged / f"{name}.jsonl").open(
                "x", encoding="utf-8", newline="\n")) for name in FOLDS}
            source = stack.enter_context(assets["text2cad_prompt_csv"].open(
                encoding="utf-8-sig", newline=""))
            for fold, row in source_rows(csv.DictReader(source), source_fold, captions):
                count = stats[fold]
                count["native_uids"] += 1
                count["native_four_level_uids"] += len(row["native_texts"]) == len(LEVEL_FIELDS)
                has_both = row["ct_deepcad_text"] is not None
                count["common_native_and_ct_uids"] += has_both
                for level in row["native_texts"]:
                    count["native_pairs_per_level"][level] += 1
                    count["common_native_pairs_per_level"][level] += has_both
                outputs[fold].write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                               separators=(",", ":")) + "\n")
        if any(not count["common_native_and_ct_uids"] or 0 in count["native_pairs_per_level"].values()
               for count in stats.values()):
            raise ValueError("Source supervision has an empty registered fold or condition")
        manifest = {
            "schema_version": 1,
            "rows": "one UID per row in pinned native CSV order, every available original native level",
            "ct_comparison": "select rows with ct_deepcad_text present for both training descriptions",
            "splits": stats,
            "official_test_uids_excluded": len(splits["test"]),
            "sources": {name: {"path": str(path), "sha256": ASSETS[name][2]} for name, path in assets.items()},
            "files": {path.name: sha256(path) for path in sorted(staged.iterdir())},
        }
        write_json(staged / "manifest.json", manifest)
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--asset-runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.main_repo.resolve(), args.asset_runtime.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
