"""Compare exact published CADmium geometry-program keys across Fusion and DeepCAD."""

from __future__ import annotations

import argparse
import ast
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from prepare_formal_fusion360 import (
    EXPECTED_SPLITS, PARQUET_SHA256, SOURCE, json_bytes, read_jsonl, require_sha, sha256_bytes,
    sha256_file, write_jsonl,
)

DEEPCAD_REVISION = "ebde0c81e141b4ef7829be1e4b2ac7126e31d0c4"
DEEPCAD_UID = re.compile(r"[0-9]{4}/[0-9]{8}")
DEEPCAD = {
    "train": (118299, "22429763b823fb67bdf240a12b2932664647b96099b35a641a6379c085fc4d30"),
    "validation": (8925, "1608b54362402b48d8a4f0dbf687c42de2363510ec3efe3e59d7ad066b3fd110"),
    "test": (8046, "c0f09e44d4d2540c1add36690a4c09c5929671299d0248aa5d3e1b822e4dddac"),
}
GEOMETRY_FIELDS = ("coordinate_system", "sketch", "extrusion")


def key(raw, dataset):
    """Reimplement the E1 exact geometry-field key without reading paired annotations."""
    try:
        model = ast.literal_eval(raw) if dataset == "fusion" else json.loads(raw)
        parts = model.get("parts") if isinstance(model, dict) else None
        if not isinstance(parts, dict) or not parts:
            return None
        geometry = {}
        for identifier, part in parts.items():
            if not isinstance(part, dict) or any(not isinstance(part.get(field), dict) for field in GEOMETRY_FIELDS):
                return None
            geometry[identifier] = {field: part[field] for field in GEOMETRY_FIELDS}
        return json_bytes({"parts": geometry}, sort_keys=True)
    except (SyntaxError, TypeError, ValueError, RecursionError):
        return None


def program_rows(path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=["uid", "json_desc"], batch_size=512):
        yield from batch.to_pylist()


def fusion_keys(root, parquet):
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest["source"] != SOURCE or manifest["published_parquet_sha256"] != PARQUET_SHA256:
        raise ValueError("Unexpected Fusion catalog source")
    inventory_path = root / "source_inventory.jsonl"
    require_sha(inventory_path, manifest["file_sha256"][inventory_path.name])
    inventory_rows = list(read_jsonl(inventory_path))
    inventory = {row["uid"]: row for row in inventory_rows}
    if (len(inventory) != len(inventory_rows) or
            Counter(row["split"] for row in inventory_rows) != EXPECTED_SPLITS):
        raise ValueError("Unexpected Fusion catalog axis")
    by_key = defaultdict(list)
    unseen = set(inventory)
    missing = []
    require_sha(parquet, PARQUET_SHA256)
    for row in program_rows(parquet):
        uid = row["uid"]
        if uid not in unseen:
            raise ValueError("Unknown or duplicate Fusion publication UID")
        unseen.remove(uid)
        program = key(row["json_desc"], "fusion")
        if program is None:
            missing.append(uid)
        else:
            by_key[program].append(uid)
    if unseen:
        raise ValueError("Fusion publication and catalog differ")
    return inventory, by_key, sorted(missing)


def compare_deepcad(path, split, target_keys, fusion_by_uid):
    expected_count, expected_sha = DEEPCAD[split]
    require_sha(path, expected_sha)
    audited, unavailable, matches = set(), [], []
    for row in program_rows(path):
        uid = row["uid"]
        if not isinstance(uid, str) or not DEEPCAD_UID.fullmatch(uid) or uid in audited:
            raise ValueError(f"Invalid or duplicate published DeepCAD {split} UID")
        audited.add(uid)
        program = key(row["json_desc"], "deepcad")
        if program is None:
            unavailable.append(uid)
        elif program in target_keys:
            digest = sha256_bytes(program)
            matches.extend({"fusion_uid": candidate, "fusion_split": fusion_by_uid[candidate]["split"],
                            "deepcad_uid": uid, "deepcad_publication_split": split,
                            "exact_program_key_sha256": digest}
                           for candidate in target_keys[program])
    if len(audited) != expected_count:
        raise ValueError(f"Unexpected published DeepCAD {split} coverage")
    match_counts = {}
    for fusion_split in EXPECTED_SPLITS:
        rows = [row for row in matches if row["fusion_split"] == fusion_split]
        match_counts[fusion_split] = {"matching_fusion_uids": len({row["fusion_uid"] for row in rows}),
                                     "matching_deepcad_uids": len({row["deepcad_uid"] for row in rows}),
                                     "matching_uid_pairs": len(rows)}
    return matches, {"deepcad_parquet_sha256": expected_sha, "deepcad_uid_count": len(audited),
                     "deepcad_program_key_available": len(audited) - len(unavailable),
                     "deepcad_program_key_unavailable_uids": sorted(unavailable), "by_fusion_split": match_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--fusion-parquet", type=Path, required=True)
    for split in DEEPCAD:
        parser.add_argument(f"--deepcad-{split}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    supplied = {split: getattr(args, f"deepcad_{split}") for split in DEEPCAD if getattr(args, f"deepcad_{split}")}
    if not supplied:
        parser.error("pass at least one pinned DeepCAD publication parquet")

    fusion, target_keys, missing = fusion_keys(args.catalog, args.fusion_parquet)
    summary = {
        "study": "exact_published_program_geometry_fields", "source": SOURCE,
        "native_step_geometric_identity": "unverified",
        "comparison": "equal canonical UTF-8 JSON bytes, sorted keys, exact values, includes part/face/loop/curve keys",
        "geometry_fields": GEOMETRY_FIELDS, "deepcad_revision": DEEPCAD_REVISION,
        "fusion_catalog_sha256": sha256_file(args.catalog / "manifest.json"),
        "fusion_parquet_sha256": PARQUET_SHA256, "fusion_frozen_counts": EXPECTED_SPLITS,
        "fusion_program_key_unavailable_uids": missing, "deepcad_publication_splits": {},
    }
    collected = []
    for split, path in supplied.items():
        matches, facts = compare_deepcad(path, split, target_keys, fusion)
        collected.extend(matches)
        summary["deepcad_publication_splits"][split] = facts
        if split == "train":
            all_known = len({row["fusion_uid"] for row in matches})
            test_known = facts["by_fusion_split"]["test"]["matching_fusion_uids"]
            summary["historical_e1_train_key_reproduction"] = {
                "expected_all_fusion": 133, "observed_all_fusion": all_known,
                "expected_fusion_test": 29, "observed_fusion_test": test_known,
                "agrees": (all_known, test_known) == (133, 29),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fusion-overlap-", dir=args.output.parent) as staging:
        out = Path(staging) / "output"
        out.mkdir()
        matches_path = out / "exact_published_program_overlaps.jsonl"
        write_jsonl(matches_path, sorted(collected, key=lambda row: (
            row["fusion_uid"], row["deepcad_publication_split"], row["deepcad_uid"])))
        summary["overlap_rows_sha256"] = sha256_file(matches_path)
        (out / "overlap_manifest.json").write_bytes(json_bytes(summary) + b"\n")
        out.rename(args.output)
    print(json.dumps({"matches": len(collected), "output": str(args.output),
                      "historical_e1_train_key_reproduction": summary.get("historical_e1_train_key_reproduction")}))


if __name__ == "__main__":
    main()
