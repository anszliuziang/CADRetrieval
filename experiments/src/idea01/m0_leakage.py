"""CPU-only UID split and training-overlap audit for the Text2CAD benchmark.

The output is an audit artifact, never a scoring input. Pass every source path
explicitly; no model forward, network access, or benchmark filtering occurs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .m0_manifest import load_split_json, sha256_file


SPLITS = ("train", "validation", "test")


def _uid_jsonl(path: Path) -> tuple[set[str], int, dict[str, int]]:
    uids: set[str] = set()
    statuses: Counter[str] = Counter()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            uid = row.get("uid")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"{path}: invalid UID on line {count + 1}")
            uids.add(uid)
            statuses[str(row.get("map_status", "missing"))] += 1
            count += 1
    return uids, count, dict(sorted(statuses.items()))


def _parquet_uids(path: Path) -> tuple[set[str], int]:
    import pyarrow.parquet as pq

    values = pq.read_table(path, columns=["uid"]).column("uid").to_pylist()
    if any(not isinstance(uid, str) or not uid for uid in values):
        raise ValueError(f"{path}: invalid UID")
    return set(values), len(values)


def _matrix(rows: dict[str, set[str]], columns: dict[str, set[str]]) -> dict:
    return {
        row_name: {
            column_name: len(row_uids & column_uids)
            for column_name, column_uids in columns.items()
        }
        for row_name, row_uids in rows.items()
    }


def build_audit(
    official_split: Path,
    cadmium: dict[str, Path],
    cadquery: dict[str, Path],
) -> dict:
    original = load_split_json(official_split)
    official = {name: set(original[name]) for name in SPLITS}
    cadmium_sets: dict[str, set[str]] = {}
    cadquery_sets: dict[str, set[str]] = {}
    sources = {"official_split": {"path": str(official_split), "sha256": sha256_file(official_split)}}
    cadmium_rows: dict[str, int] = {}
    cadquery_rows: dict[str, int] = {}
    cadquery_status: dict[str, dict[str, int]] = {}
    for name in SPLITS:
        cadmium_sets[name], cadmium_rows[name] = _parquet_uids(cadmium[name])
        cadquery_sets[name], cadquery_rows[name], cadquery_status[name] = _uid_jsonl(cadquery[name])
        sources[f"cadmium_{name}"] = {"path": str(cadmium[name]), "sha256": sha256_file(cadmium[name])}
        sources[f"cadquery_{name}"] = {"path": str(cadquery[name]), "sha256": sha256_file(cadquery[name])}
    official_test = official["test"]
    clean_train = official_test - cadquery_sets["train"]
    clean_train_val = clean_train - cadquery_sets["validation"]
    issues: list[str] = []
    for name in SPLITS:
        if len(original[name]) != len(official[name]):
            issues.append(f"official_{name}_duplicate_uid")
        if cadmium_rows[name] != len(cadmium_sets[name]):
            issues.append(f"cadmium_{name}_duplicate_uid")
        if cadquery_rows[name] != len(cadquery_sets[name]):
            issues.append(f"cadquery_{name}_duplicate_uid")
    if cadmium_sets["test"] != official["test"]:
        issues.append("cadmium_test_differs_from_official_test")
    if cadmium_sets["validation"] != official["validation"]:
        issues.append("cadmium_validation_differs_from_official_validation")
    if not cadmium_sets["train"] <= official["train"]:
        issues.append("cadmium_train_outside_official_train")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if official[left] & official[right]:
            issues.append(f"official_{left}_{right}_overlap")
        if cadmium_sets[left] & cadmium_sets[right]:
            issues.append(f"cadmium_{left}_{right}_overlap")
        if cadquery_sets[left] & cadquery_sets[right]:
            issues.append(f"cadquery_{left}_{right}_overlap")
    return {
        "schema_version": "m0-leakage-v1",
        "sources": sources,
        "official_sizes": {name: len(official[name]) for name in SPLITS},
        "cadmium_rows": cadmium_rows,
        "cadmium_unique_uids": {name: len(cadmium_sets[name]) for name in SPLITS},
        "cadmium_by_official": _matrix(cadmium_sets, official),
        "cadquery_rows": cadquery_rows,
        "cadquery_unique_uids": {name: len(cadquery_sets[name]) for name in SPLITS},
        "cadquery_map_status": cadquery_status,
        "cadquery_by_official": _matrix(cadquery_sets, official),
        "cadquery_internal_intersections": {
            "train__validation": len(cadquery_sets["train"] & cadquery_sets["validation"]),
            "train__test": len(cadquery_sets["train"] & cadquery_sets["test"]),
            "validation__test": len(cadquery_sets["validation"] & cadquery_sets["test"]),
        },
        "official_test_exposed_to_cadquery_train": sorted(official_test & cadquery_sets["train"]),
        "official_test_clean_of_cadquery_train": sorted(clean_train),
        "official_test_clean_of_cadquery_train_val": sorted(clean_train_val),
        "integrity_issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-split", type=Path, required=True)
    for name in SPLITS:
        parser.add_argument(f"--cadmium-{name}", type=Path, required=True)
        parser.add_argument(f"--cadquery-{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_audit(
        args.official_split,
        {name: getattr(args, f"cadmium_{name}") for name in SPLITS},
        {name: getattr(args, f"cadquery_{name}") for name in SPLITS},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "cadmium_test_uid": audit["cadmium_unique_uids"]["test"],
        "cadquery_train_on_official_test": audit["cadquery_by_official"]["train"]["test"],
        "clean_of_train": len(audit["official_test_clean_of_cadquery_train"]),
        "clean_of_train_val": len(audit["official_test_clean_of_cadquery_train_val"]),
        "integrity_issues": audit["integrity_issues"],
        "sha256": sha256_file(args.out),
    }, sort_keys=True))
    return 0 if not audit["integrity_issues"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
