"""Prepare pinned Text2CAD E1 inputs on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "src"))

from idea01 import text2cad_e1_inventory as inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train-vectors", "test-vectors", "inventory", "train-shortlist", "train-corroboration"))
    parser.add_argument("--asset-root", type=Path, required=True, help="Original project root containing .cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, help="Pinned DeepCAD validation shard override")
    parser.add_argument("--vector-dir", type=Path, help="Verified official vector receipts for train-corroboration")
    args = parser.parse_args()

    cache = args.asset_root / ".cache"
    official = cache / "text2cad_official"
    package = official / "text2cad_v1.0"
    assets = inventory.read_json(EXPERIMENTS / "configs/text2cad_original_asset_sha256_20260930.json")
    freeze_path = EXPERIMENTS / "configs/idea01_e1_fusion360_background_48_20261119.json"
    freeze = inventory.read_json(freeze_path)
    if (freeze["dataset_revision"], freeze["count"], freeze["selected_config"], freeze["ordered_digest"]) != (
        assets["dataset_revision"], 48, "logmix_l1", "44fa6f881cb46d9ab11dc7a1599495ba1a50b9d78551a978cc54f1e197aa8dfb"
    ):
        raise ValueError("Background manifest disagrees with the E1 protocol")
    split_path = package / freeze["split_file"]
    zip_path = official / "cad_seq.zip"
    start = time.perf_counter()
    survey = cache / "crossdomain_survey_20261119"
    if args.stage in ("train-shortlist", "train-corroboration"):
        seed_path = args.output_dir / "inventory_seed.jsonl"
        manifest = inventory.read_json(args.output_dir / "inventory_inputs.json")
        train_path = survey / "deepcad-ebde0c.train.parquet"
        inputs = {
            "deepcad_train": inventory.check_sha(train_path, inventory.INVENTORY_SHA256["deepcad_train"]),
            "inventory_seed": inventory.check_sha(seed_path, manifest["outputs"]["inventory_seed.jsonl"]),
        }
        if args.stage == "train-corroboration":
            shortlist_path = args.output_dir / "near_train_shortlist.jsonl"
            shortlist_manifest = inventory.read_json(args.output_dir / "near_train_coverage.json")
            if shortlist_manifest["inputs"] != inputs:
                raise ValueError("Shortlist does not bind the current inventory inputs")
            inputs["shortlist"] = inventory.check_sha(shortlist_path, shortlist_manifest["output_sha256"])
            vector_dir = args.vector_dir or args.output_dir
            for split in ("train", "test"):
                vector_receipt = inventory.read_json(vector_dir / f"{split}_vec_coverage.json")
                if not vector_receipt["coverage_complete"] or vector_receipt["hashed_uid_count"] != inventory.EXPECTED_SOURCE_SPLIT[split]:
                    raise ValueError(f"Incomplete official vector coverage for {split}")
                source_inputs = vector_receipt["inputs"]
                if (source_inputs["official_split"]["sha256"] != freeze["split_sha256"] or
                    source_inputs["official_cad_zip"]["sha256"] != assets["cad_sequence"]["cad_seq.zip"]):
                    raise ValueError(f"Unpinned official vector scan sources for {split}")
                inputs[f"{split}_vectors"] = inventory.check_sha(vector_dir / f"{split}_vec_hashes.jsonl", vector_receipt["output_sha256"])
            receipt = inventory.corroborate_train_shortlist(
                train_path, seed_path, shortlist_path, vector_dir / "train_vec_hashes.jsonl", vector_dir / "test_vec_hashes.jsonl",
                args.output_dir / "near_train_corroboration.jsonl")
            receipt["inputs"] = inputs
            receipt["elapsed_seconds"] = time.perf_counter() - start
            inventory.atomic_json(args.output_dir / "near_train_corroboration_coverage.json", receipt)
            print(json.dumps({key: receipt[key] for key in ("pairs_compared", "query_status_by_domain", "elapsed_seconds")}))
            return 0
        receipt = inventory.shortlist_train_programs(train_path, seed_path, args.output_dir / "near_train_shortlist.jsonl")
        receipt["inputs"] = inputs
        receipt["elapsed_seconds"] = time.perf_counter() - start
        inventory.atomic_json(args.output_dir / "near_train_coverage.json", receipt)
        print(json.dumps({key: receipt[key] for key in ("train_feature_count", "query_feature_count",
                                                       "queries_by_domain", "elapsed_seconds")}))
        return 0

    asset_receipts = {
        "official_split": inventory.check_sha(split_path, freeze["split_sha256"]),
        "official_cad_zip": inventory.check_sha(zip_path, assets["cad_sequence"]["cad_seq.zip"]),
    }
    source_split_raw = inventory.read_json(split_path)
    source_split = inventory.check_split(source_split_raw, inventory.EXPECTED_SOURCE_SPLIT)
    if args.stage.endswith("-vectors"):
        split = args.stage.removesuffix("-vectors")
        _, receipt = inventory.scan_train_archive(zip_path, source_split_raw[split],
                                                  args.output_dir / f"{split}_vec_hashes.jsonl")
        receipt["inputs"] = asset_receipts
        receipt["split"] = split
        receipt["elapsed_seconds"] = time.perf_counter() - start
        inventory.atomic_json(args.output_dir / f"{split}_vec_coverage.json", receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key not in {"failed_uids", "inputs"}}))
        return 0 if receipt["coverage_complete"] else 2

    bert = cache / "bert_large_uncased"
    paths = {
        "deepcad_train": survey / "deepcad-ebde0c.train.parquet",
        "deepcad_test": survey / "deepcad-ebde0c.test.parquet",
        "deepcad_validation": args.validation_parquet or survey / "deepcad-ebde0c.validation.parquet",
        "fusion_published": survey / "fusion360-ea7d06f.validation.parquet",
        "fusion_split": survey / "fusion-original-train_test.json",
        "tokenizer": bert / "tokenizer.json",
        "tokenizer_config": bert / "tokenizer_config.json",
        "tokenizer_vocab": bert / "vocab.txt",
        "official_csv": package / freeze["csv_file"],
    }
    expectations = {
        **inventory.INVENTORY_SHA256,
        "official_csv": freeze["csv_sha256"],
        "tokenizer": assets["bert_large_uncased"]["tokenizer.json"],
        "tokenizer_config": assets["bert_large_uncased"]["tokenizer_config.json"],
        "tokenizer_vocab": assets["bert_large_uncased"]["vocab.txt"],
    }
    asset_receipts.update({name: inventory.check_sha(path, expectations[name]) for name, path in paths.items()})
    asset_receipts["background_manifest"] = inventory.check_sha(freeze_path)
    counts = inventory.build_inventory(paths, source_split, zip_path, freeze, args.output_dir)
    inventory.atomic_json(args.output_dir / "inventory_inputs.json", {
        "inputs": asset_receipts,
        "counts": counts,
        "outputs": {name: inventory.sha256_file(args.output_dir / name) for name in
                    ("inventory_seed.jsonl", "development.jsonl", "background.jsonl", "l0_train_match_index.jsonl")},
        "elapsed_seconds": time.perf_counter() - start,
    })
    print(json.dumps({key: counts[key] for key in ("deepcad", "fusion360", "development", "background")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
