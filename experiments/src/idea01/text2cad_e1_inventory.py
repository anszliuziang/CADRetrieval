"""Frozen Text2CAD E1 text, program, and official-vector inventory."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


PROGRAM_FIELDS = ("coordinate_system", "sketch", "extrusion")
SOURCE_TEXT_FIELDS = ("abstract", "beginner", "intermediate", "expert")
EXPECTED_SOURCE_SPLIT = {"train": 159_049, "validation": 8_925, "test": 8_046}
INVENTORY_SHA256 = {
    "deepcad_train": "22429763b823fb67bdf240a12b2932664647b96099b35a641a6379c085fc4d30",
    "deepcad_test": "c0f09e44d4d2540c1add36690a4c09c5929671299d0248aa5d3e1b822e4dddac",
    "deepcad_validation": "1608b54362402b48d8a4f0dbf687c42de2363510ec3efe3e59d7ad066b3fd110",
    "fusion_published": "a00dea0d893a8edd1fbd02329bc7db571e8f6ed6eae0956e27a852f0b3f5755f",
    "fusion_split": "2de40371e99fb32c09ca11ba76995ab40fe465aa09609b4de6c81de1bc553ef4",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_sha(path: Path, expected: str | None = None) -> dict:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, data, *, lines: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            if lines:
                for row in data:
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                    out.write("\n")
            else:
                json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
                out.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def official_member(uid: str) -> str:
    return f"cad_seq/{uid}/seq/{uid.rsplit('/', 1)[-1]}.pth"


def check_split(split: dict, counts: dict[str, int]) -> dict[str, set[str]]:
    if set(split) != set(counts):
        raise ValueError("Unexpected split names")
    result = {name: set(uids) for name, uids in split.items()}
    if any(len(uids) != counts[name] or len(uids) != len(result[name]) for name, uids in split.items()):
        raise ValueError("Split size or UID uniqueness mismatch")
    if len(set().union(*result.values())) != sum(counts.values()):
        raise ValueError("Official splits overlap")
    return result


def scan_train_archive(archive_path: Path, uids, output_path: Path) -> tuple[list[dict], dict]:
    import numpy as np
    import torch

    uids = list(uids)
    target = {official_member(uid): uid for uid in uids}
    if len(target) != len(uids):
        raise ValueError("Duplicate official train UIDs")
    records = []
    failures = {}
    with zipfile.ZipFile(archive_path) as archive:
        counts = Counter(info.filename for info in archive.infolist() if info.filename in target)
        for member, uid in target.items():
            if counts[member] != 1:
                failures[uid] = "missing_member" if counts[member] == 0 else "duplicate_member"
        # Central directory order follows physical order in the official archive.
        for info in archive.infolist():
            if counts.get(info.filename) != 1:
                continue
            uid = target[info.filename]
            try:
                payload = torch.load(io.BytesIO(archive.read(info)), map_location="cpu", weights_only=True)
                vector = payload["vec"]["cad_vec"].numpy()
                if vector.shape != (272, 2) or vector.dtype.kind not in "iu":
                    failures[uid] = "invalid_vector_shape_or_dtype"
                    continue
                encoded = np.ascontiguousarray(vector, dtype="<i4")
                if not np.array_equal(vector, encoded):
                    failures[uid] = "vector_int32_overflow"
                    continue
                records.append({"uid": uid, "sha256": hashlib.sha256(encoded.tobytes()).hexdigest()})
            except Exception as exc:
                failures[uid] = f"unreadable_vector:{type(exc).__name__}"
    if len(records) + len(failures) != len(uids):
        raise ValueError("Train scan lost UID accounting")
    atomic_json(output_path, records, lines=True)
    receipt = {
        "expected_uid_count": len(uids),
        "hashed_uid_count": len(records),
        "distinct_hash_count": len({record["sha256"] for record in records}),
        "coverage_complete": not failures,
        "failed_uids": [{"uid": uid, "reason": reason} for uid, reason in sorted(failures.items())],
        "encoding": "C-order little-endian int32, shape (272, 2), SHA-256",
        "output_sha256": sha256_file(output_path),
    }
    return records, receipt


def parse_minimal(raw: str, domain: str) -> dict:
    if domain == "deepcad":
        parsed = json.loads(raw)
    elif domain == "fusion360":
        parsed = ast.literal_eval(raw)
    else:
        raise ValueError(f"Unexpected domain {domain}")
    parts = parsed.get("parts") if isinstance(parsed, dict) else None
    if not isinstance(parts, dict) or not parts:
        raise ValueError("Minimal parts missing or empty")
    for part in parts.values():
        if not isinstance(part, dict) or any(not isinstance(part.get(field), dict) for field in PROGRAM_FIELDS):
            raise ValueError("Minimal geometry field missing or invalid")
    return parsed


def program_key(minimal: dict) -> str:
    parts = {name: {field: part[field] for field in PROGRAM_FIELDS} for name, part in minimal["parts"].items()}
    return json.dumps({"parts": parts}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def recover_background(freeze: dict, l0: dict[str, str], validation_uids: set[str]) -> list[dict]:
    rows = freeze["rows"]
    if freeze["text_field"] != "abstract" or len(rows) != freeze["count"]:
        raise ValueError("Unexpected background configuration")
    result = []
    for index, row in enumerate(rows):
        uids = row["validation_uids"]
        digest = row["text_sha256"]
        if row["index"] != index or not uids:
            raise ValueError(f"Invalid background index {index}")
        if any(uid not in validation_uids or not l0.get(uid) or sha256_text(l0[uid]) != digest for uid in uids):
            raise ValueError(f"Frozen background UID or text mismatch at {index}")
        result.append({"uid": uids[0], "text": l0[uids[0]], "text_sha256": digest, "index": index})
    if sha256_text("\n".join(row["text_sha256"] for row in result)) != freeze["ordered_digest"]:
        raise ValueError("Frozen background ordered hash mismatch")
    return result


def parquet_rows(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=["uid", "annotation", "json_desc"], batch_size=512):
        yield from batch.to_pylist()


def source_csv_indexes(path: Path, train_uids: set[str], l0_uids: set[str], expert_texts: set[str]):
    """Index matching text occurrences while streaming the 1 GB published CSV."""
    l0 = {}
    matches = defaultdict(list)
    train_seen = set()
    field_counts = Counter()

    def values():
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            uid_index = header.index("uid")
            field_indexes = [header.index(field) for field in SOURCE_TEXT_FIELDS]
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Invalid official CSV field count at line {reader.line_num}")
                yield row[uid_index], row, field_indexes

    previous_limit = csv.field_size_limit(16 * 1024 * 1024)
    try:
        for uid, row, indexes in values():
            if uid in l0_uids:
                if uid in l0:
                    raise ValueError(f"Duplicate relevant L0 UID {uid}")
                l0[uid] = row[indexes[0]].strip()
        queried_texts = expert_texts | (set(l0.values()) - {""})
        for uid, row, indexes in values():
            if uid not in train_uids:
                continue
            if uid in train_seen:
                raise ValueError(f"Duplicate official training text UID {uid}")
            train_seen.add(uid)
            for field, index in zip(SOURCE_TEXT_FIELDS, indexes, strict=True):
                text = row[index].strip()
                if text:
                    field_counts[field] += 1
                    if text in queried_texts:
                        matches[text].append({"source": "text2cad_v1.0", "field": field, "uid": uid})
    finally:
        csv.field_size_limit(previous_limit)
    return l0, matches, {
        "expected_uid_count": len(train_uids),
        "found_uid_count": len(train_seen),
        "missing_uid_count": len(train_uids - train_seen),
        "missing_uids": sorted(train_uids - train_seen),
        "nonempty_fields": {field: field_counts[field] for field in SOURCE_TEXT_FIELDS},
    }


def checked_minimal(raw: str, domain: str) -> tuple[dict | None, str | None]:
    try:
        minimal = parse_minimal(raw, domain)
        return minimal, program_key(minimal)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return None, None


def summarize_screen(rows: list[dict]) -> dict:
    test = [row for row in rows if row["published_split"] == "test"]
    short = [row for row in test if row["text"] and row["token_count"] <= 512]
    after_program = [row for row in short if not row["program_train_uids"]]
    eligible = [row for row in rows if row["first_exclusion"] is None]
    return {
        "published_rows": len(rows),
        "original_test_rows": len(test),
        "original_train_rows": sum(row["published_split"] == "train" for row in rows),
        "test_text_within_limit": len(short),
        "test_text_over_limit": sum(row["token_count"] > 512 for row in test),
        "test_program_matches_train": sum(bool(row["program_train_uids"]) for row in test),
        "within_limit_program_matches_train": sum(bool(row["program_train_uids"]) for row in short),
        "published_program_matches_train": sum(bool(row["program_train_uids"]) for row in rows),
        "after_length_and_program": len(after_program),
        "projects_after_length_and_program": len({r["project"] for r in after_program if r["project"]}),
        "projects_also_in_original_train": len({r["project"] for r in after_program if r["project_in_original_train"]}),
        "test_expert_text_matches_train": sum(bool(row["text_train_matches"]) for row in test),
        "eligible_before_vectors_geometry": len(eligible),
        "eligible_projects_before_vectors_geometry": len({r["project"] for r in eligible if r["project"]}),
        "all_reasons": dict(Counter(reason for row in rows for reason in row["all_reasons"])),
        "first_exclusions": dict(Counter(row["first_exclusion"] for row in rows if row["first_exclusion"])),
    }


def build_inventory(paths: dict[str, Path], source_split: dict, zip_path: Path,
                    freeze: dict, output: Path) -> dict:
    from tokenizers import Tokenizer

    fusion_split = check_split(read_json(paths["fusion_split"]), {"train": 6_900, "test": 1_725})
    fusion_train_projects = {uid.split("_", 1)[0] for uid in fusion_split["train"]}
    stages = (
        ("deepcad_test", "deepcad", 8_046),
        ("fusion_published", "fusion360", 8_307),
        ("deepcad_validation", "deepcad", 8_925),
    )
    published = {}
    for stage, domain, expected_size in stages:
        batch = list(parquet_rows(paths[stage]))
        uids = {row["uid"] for row in batch}
        if len(batch) != expected_size or len(uids) != expected_size:
            raise ValueError(f"Publication row or UID count mismatch: {stage}")
        if domain == "deepcad" and uids != source_split[stage.removeprefix("deepcad_")]:
            raise ValueError(f"DeepCAD official UID linkage mismatch: {stage}")
        if domain == "fusion360" and not uids <= fusion_split["train"] | fusion_split["test"]:
            raise ValueError("Unknown Fusion360 UIDs in published parquet")
        for row in batch:
            row["text"] = (row.pop("annotation") or "").strip()
            row["minimal"], row["program_key"] = checked_minimal(row.pop("json_desc"), domain)
        published[stage] = batch

    candidate_keys = {row["program_key"] for batch in published.values() for row in batch if row["program_key"]}
    candidate_texts = {row["text"] for batch in published.values() for row in batch if row["text"]}
    l0, text_overlaps, csv_coverage = source_csv_indexes(
        paths["official_csv"], source_split["train"], source_split["test"] | source_split["validation"], candidate_texts)
    queried_texts = candidate_texts | (set(l0.values()) - {""})
    background = recover_background(freeze, l0, source_split["validation"])

    program_overlaps = defaultdict(list)
    source_train_uids = set()
    source_program_ok = 0
    source_text_ok = 0
    source_parse_failures = []
    for row in parquet_rows(paths["deepcad_train"]):
        uid = row["uid"]
        if uid in source_train_uids or uid not in source_split["train"]:
            raise ValueError(f"Unexpected CADmium train UID {uid}")
        source_train_uids.add(uid)
        _, key = checked_minimal(row["json_desc"], "deepcad")
        text = (row["annotation"] or "").strip()
        if key:
            source_program_ok += 1
            if key in candidate_keys:
                program_overlaps[key].append(uid)
        else:
            source_parse_failures.append(uid)
        if text:
            source_text_ok += 1
            if text in queried_texts:
                text_overlaps[text].append({"source": "cadmium_ds", "field": "annotation", "uid": uid})
    if len(source_train_uids) != 118_299:
        raise ValueError("CADmium train coverage count mismatch")

    expected_members = {official_member(uid) for uid in source_split["test"] | source_split["validation"]}
    with zipfile.ZipFile(zip_path) as archive:
        visible_members = Counter(info.filename for info in archive.infolist() if info.filename in expected_members)
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def record(row, stage, domain, text_count, text_ids_sha, l0_count):
        uid = row["uid"]
        text, minimal, key = row["text"], row["minimal"], row["program_key"]
        split = stage.removeprefix("deepcad_") if domain == "deepcad" else (
            "test" if uid in fusion_split["test"] else "train")
        member = official_member(uid) if domain == "deepcad" else None
        member_valid = member is not None and visible_members.get(member) == 1
        project = uid.split("_", 1)[0] if domain == "fusion360" else None
        raw_l0 = (l0.get(uid) or None) if domain == "deepcad" else None
        exact_text = text_overlaps.get(text, [])
        exact_program = program_overlaps.get(key, [])
        reasons = []
        if split == "train":
            reasons.append("outside_official_test")
        if member is not None and not member_valid:
            reasons.append("missing_or_duplicate_official_member")
        if not text:
            reasons.append("missing_expert_text")
        if text_count > 512:
            reasons.append("expert_text_over_512_tokens")
        if exact_text:
            reasons.append("source_train_text_exact")
        if exact_program:
            reasons.append("source_train_program_exact")
        if minimal is None:
            reasons.append("minimal_unparseable")
        return {
            "domain": domain, "uid": uid, "project": project, "published_split": split,
            "project_in_original_train": project in fusion_train_projects,
            "text": text, "text_sha256": sha256_text(text) if text else None,
            "token_count": text_count, "token_ids_sha256": text_ids_sha,
            "l0_text": raw_l0, "l0_token_count": l0_count,
            "l0_text_sha256": sha256_text(raw_l0) if raw_l0 else None,
            "l0_text_train_match_count": len(text_overlaps.get(raw_l0, [])),
            "minimal": minimal, "program_sha256": sha256_text(key) if key is not None else None,
            "official_member": member if member_valid else None,
            "text_train_matches": sorted(exact_text, key=lambda item: (item["source"], item["field"], item["uid"])),
            "program_train_uids": sorted(exact_program),
            "all_reasons": reasons, "first_exclusion": reasons[0] if reasons else None,
        }

    for stage, domain, _ in stages:
        batch = published[stage]
        records = []
        for start in range(0, len(batch), 128):
            chunk = batch[start:start + 128]
            texts = tokenizer.encode_batch([row["text"] for row in chunk], add_special_tokens=True)
            levels = tokenizer.encode_batch([l0.get(row["uid"], "") for row in chunk], add_special_tokens=True)
            for row, encoded, abstract in zip(chunk, texts, levels, strict=True):
                ids_sha = sha256_text(json.dumps(encoded.ids, separators=(",", ":"))) if row["text"] else None
                records.append(record(row, stage, domain, len(encoded.ids) if row["text"] else 0,
                                      ids_sha, len(abstract.ids) if l0.get(row["uid"]) else None))
        published[stage] = records

    source_test = published["deepcad_test"]
    fusion = published["fusion_published"]
    source_val = published["deepcad_validation"]
    counts = {"deepcad": summarize_screen(source_test), "fusion360": summarize_screen(fusion)}
    expected_counts = {
        "deepcad": {"original_test_rows": 8_046, "test_text_within_limit": 7_606,
                    "within_limit_program_matches_train": 2_051, "after_length_and_program": 5_555},
        "fusion360": {"original_test_rows": 1_661, "original_train_rows": 6_646, "test_text_within_limit": 1_370,
                      "test_text_over_limit": 291, "test_program_matches_train": 29,
                      "within_limit_program_matches_train": 29, "published_program_matches_train": 133,
                      "after_length_and_program": 1_341, "projects_after_length_and_program": 943,
                      "projects_also_in_original_train": 596},
    }
    discrepancies = {f"{domain}:{key}": {"actual": counts[domain][key], "expected": expected}
                     for domain, keys in expected_counts.items() for key, expected in keys.items()
                     if counts[domain][key] != expected}
    if discrepancies:
        raise ValueError(f"Registered inventory checkpoints diverged: {discrepancies}")

    background_uids = {uid for item in freeze["rows"] for uid in item["validation_uids"]}
    development_rank = lambda row: (sha256_text(f"cad-domain-v1|development|deepcad|{row['uid']}"), row["uid"])
    development_pool = [row for row in source_val
                        if row["uid"] not in background_uids and row["official_member"] and row["minimal"]]
    development = sorted(development_pool, key=development_rank)[:64]
    if len(development) != 64:
        raise ValueError("Fewer than 64 usable development parts")
    counts["development"] = {
        "published_rows": len(source_val), "background_excluded_uids": len(background_uids),
        "pool_rows": len(development_pool), "selected_rows": len(development),
        "selection": "First 64 by SHA-256(cad-domain-v1|development|deepcad|UID), ties by UID",
        "part_count": dict(Counter(len(row["minimal"]["parts"]) for row in development)),
        "extrusion_operation_count": dict(Counter(
            part["extrusion"].get("operation", "missing")
            for row in development for part in row["minimal"]["parts"].values())),
    }
    counts["background"] = {"rows": len(background), "ordered_digest": freeze["ordered_digest"]}
    counts["training_coverage"] = {
        "text2cad_csv": csv_coverage,
        "cadmium_published_uid_count": len(source_train_uids),
        "cadmium_nonempty_text_count": source_text_ok,
        "cadmium_parsed_program_count": source_program_ok,
        "cadmium_program_parse_failed_uids": source_parse_failures,
    }
    l0_match_records = []
    for text in {row["l0_text"] for row in (*source_test, *development) if row["l0_text"]}:
        if text in text_overlaps:
            l0_match_records.append({
                "l0_text_sha256": sha256_text(text),
                "matches": sorted(text_overlaps[text], key=lambda item: (item["source"], item["field"], item["uid"])),
            })
    l0_match_records.sort(key=lambda row: row["l0_text_sha256"])
    counts["deepcad"]["l0_rows_with_observed_train_match"] = sum(
        row["l0_text_train_match_count"] > 0 for row in source_test)
    counts["deepcad"]["l0_rows_with_observed_train_match_before_vectors_geometry"] = sum(
        row["l0_text_train_match_count"] > 0 and row["first_exclusion"] is None for row in source_test)
    atomic_json(output / "inventory_seed.jsonl", [*source_test, *fusion], lines=True)
    atomic_json(output / "development.jsonl", development, lines=True)
    atomic_json(output / "background.jsonl", background, lines=True)
    atomic_json(output / "l0_train_match_index.jsonl", l0_match_records, lines=True)
    return counts


NEAR_OPERATIONS = ("NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation",
                   "IntersectFeatureOperation")
NEAR_PRIMITIVES = ("line", "arc", "circle")
NEAR_FEATURES = (
    "log1p_part_count", "log1p_face_count", "log1p_loop_count",
    *(f"log1p_operation_count:{operation}" for operation in (*NEAR_OPERATIONS, "other")),
    *(f"log1p_primitive_count:{primitive}" for primitive in (*NEAR_PRIMITIVES, "other")),
    "relative_sketch_scale_min", "relative_sketch_scale_mean",
    "relative_extrude_depth_mean", "relative_extrude_depth_max", "relative_extrude_depth_min",
    "relative_translation_span_x", "relative_translation_span_y", "relative_translation_span_z",
    "min_sketch_aspect", "mean_sketch_aspect", "relative_min_sketch_span", "relative_mean_sketch_span",
)


def coarse_program_features(program: dict) -> list[float]:
    """Compute coarse topology counts and relative sizes from the minimal program."""
    parts = program["parts"]
    operations = Counter()
    primitives = Counter()
    scales = []
    depths = []
    translations = []
    sketch_aspects = []
    sketch_spans = []
    faces = loops = 0
    for part in parts.values():
        extrusion = part["extrusion"]
        operation = extrusion["operation"]
        operations[operation if operation in NEAR_OPERATIONS else "other"] += 1
        scales.append(abs(float(extrusion["sketch_scale"])))
        depths.append(abs(float(extrusion["extrude_depth_towards_normal"])) +
                      abs(float(extrusion["extrude_depth_opposite_normal"])))
        translations.append([float(value) for value in part["coordinate_system"]["Translation Vector"]])
        points = []
        for face in part["sketch"].values():
            faces += 1
            for loop in face.values():
                loops += 1
                for curve, data in loop.items():
                    kind = curve.split("_", 1)[0]
                    primitives[kind if kind in NEAR_PRIMITIVES else "other"] += 1
                    if kind == "circle":
                        x, y = data["Center"]
                        radius = abs(float(data["Radius"]))
                        points.extend(((x - radius, y - radius), (x + radius, y + radius)))
                    else:
                        points.extend(data[field] for field in ("Start Point", "Mid Point", "End Point") if field in data)
        sketch_spread = [max(axis) - min(axis) for axis in zip(*points, strict=True)] if points else [0.0, 0.0]
        sketch_span = max(sketch_spread)
        sketch_spans.append(sketch_span)
        sketch_aspects.append(min(sketch_spread) / max(sketch_span, 1e-12))
    counts = [len(parts), faces, loops, *(operations[key] for key in (*NEAR_OPERATIONS, "other")),
              *(primitives[key] for key in (*NEAR_PRIMITIVES, "other"))]
    spreads = [max(axis) - min(axis) for axis in zip(*translations, strict=True)]
    size = max(*scales, *depths, *spreads, *sketch_spans, 1e-12)
    ratios = [min(scales), sum(scales) / len(scales), sum(depths) / len(depths), max(depths), min(depths), *spreads]
    features = [*(math.log1p(count) for count in counts), *(value / size for value in ratios),
                min(sketch_aspects), sum(sketch_aspects) / len(sketch_aspects),
                min(sketch_spans) / size, sum(sketch_spans) / len(sketch_spans) / size]
    if len(features) != len(NEAR_FEATURES) or any(not math.isfinite(value) for value in features):
        raise ValueError("Invalid geometry feature")
    return features


def nearest_feature_indexes(queries, train, k: int = 5):
    """Yield top-k squared distances, breaking exact ties by the supplied training order."""
    import numpy as np

    if not 1 <= k <= len(train):
        raise ValueError("Top-k must be within the training axis")
    train_norm = np.einsum("ij,ij->i", train, train)
    for start in range(0, len(queries), 32):
        batch = queries[start:start + 32]
        query_norm = np.einsum("ij,ij->i", batch, batch)
        distances = query_norm[:, None] + train_norm[None, :] - 2 * (batch @ train.T)
        for row in distances:
            kth = np.partition(row, k - 1)[k - 1]
            closer = np.flatnonzero(row < kth)
            ties = np.flatnonzero(row == kth)[:k - len(closer)]
            indexes = np.concatenate((closer, ties))
            ranked = indexes[np.lexsort((indexes, row[indexes]))]
            yield [(int(index), float(row[index])) for index in ranked]


def shortlist_train_programs(source_path: Path, seed_path: Path, output_path: Path) -> dict:
    import numpy as np

    train = []
    train_failures = []
    for row in parquet_rows(source_path):
        uid = row["uid"]
        minimal, key = checked_minimal(row["json_desc"], "deepcad")
        try:
            if key is None:
                raise ValueError("Minimal program parsing failed")
            train.append((sha256_text(key), uid, coarse_program_features(minimal)))
        except (TypeError, KeyError, AttributeError, ValueError):
            train_failures.append(uid)
    train.sort(key=lambda item: item[:2])

    query_keys, query_features, query_failures = [], [], []
    with seed_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["first_exclusion"] is not None:
                continue
            key = {field: row[field] for field in ("domain", "uid", "program_sha256")}
            try:
                features = coarse_program_features(row["minimal"])
            except (TypeError, KeyError, AttributeError, ValueError):
                query_failures.append(key)
                continue
            query_keys.append(key)
            query_features.append(features)

    train_raw = np.asarray([row[2] for row in train], dtype=np.float64)
    train_center = train_raw.mean(axis=0)
    train_scale = np.maximum(train_raw.std(axis=0), 0.25)
    train_features = np.rint(1000 * (train_raw - train_center) / train_scale)
    queries = np.rint(1000 * (np.asarray(query_features, dtype=np.float64) - train_center) / train_scale)
    if max(float(np.abs(train_features).max()), float(np.abs(queries).max())) > 1_000_000:
        raise ValueError("Standardized features exceed exact-integer distance range")
    nearest = nearest_feature_indexes(queries, train_features)
    rows = []
    for query, matches in zip(query_keys, nearest, strict=True):
        rows.append({**query, "neighbors": [
            {"uid": train[index][1], "program_sha256": train[index][0], "distance_sq": int(distance)}
            for index, distance in matches]})
    atomic_json(output_path, rows, lines=True)
    return {
        "train_uid_count": len(train) + len(train_failures), "train_feature_count": len(train),
        "train_failed_uids": train_failures,
        "query_uid_count": len(query_keys) + len(query_failures), "query_feature_count": len(query_keys),
        "queries_by_domain": dict(Counter(query["domain"] for query in query_keys)),
        "zero_coarse_distance_by_domain": dict(Counter(
            row["domain"] for row in rows if row["neighbors"][0]["distance_sq"] == 0)),
        "query_failures": query_failures, "k": 5,
        "features": list(NEAR_FEATURES), "source_mean": train_center.tolist(), "feature_scale": train_scale.tolist(),
        "search": "Exhaustive squared Euclidean distance between standardized features rounded to 1/1000; "
                  "distances in integer radius units squared, ties sorted by program SHA then UID",
        "train_axis_sha256": sha256_text("\n".join(f"{digest}|{uid}" for digest, uid, _ in train)),
        "output_sha256": sha256_file(output_path),
        "review_status": "awaiting program or geometry confirmation",
    }


NATIVE_NUMERIC_SUSPICION_STEP = 1 / 255


def program_signature(canonical_program: str) -> tuple[str, str, tuple[float, ...]]:
    numbers = []

    def replace_numeric(value):
        if type(value) in (int, float):
            numbers.append(float(value))
            return ("number",)
        if isinstance(value, dict):
            return ("dict", tuple((name, replace_numeric(item)) for name, item in value.items()))
        if isinstance(value, list):
            return ("list", tuple(replace_numeric(item) for item in value))
        return ("value", value)

    structure = replace_numeric(json.loads(canonical_program))
    topology = json.dumps(structure, separators=(",", ":"), ensure_ascii=False)
    return canonical_program, topology, tuple(numbers)


def corroborate_program_pair(query: tuple, train: tuple, coarse_distance_sq: int) -> dict:
    exact = query[0] == train[0]
    matching_topology = query[1] == train[1]
    delta = None
    if matching_topology:
        delta = max((abs(a - b) for a, b in zip(query[2], train[2], strict=True)), default=0.0)
    evidence = []
    if exact:
        evidence.append("geometry_only_program_exact")
    elif delta is not None and delta <= NATIVE_NUMERIC_SUSPICION_STEP:
        evidence.append("matching_topology_within_one_native_numeric_step")
    if coarse_distance_sq == 0:
        evidence.append("coarse_feature_collision")
    return {
        "status": "confirmed" if exact else ("suspected" if evidence else "no_corroboration"),
        "matching_topology": matching_topology, "max_abs_field_delta": delta, "evidence": evidence,
    }


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def corroborate_train_shortlist(source_path: Path, seed_path: Path, shortlist_path: Path,
                                train_vectors_path: Path, test_vectors_path: Path, output_path: Path) -> dict:
    shortlist = list(jsonl_rows(shortlist_path))
    train_needed = {item["uid"]: item["program_sha256"] for row in shortlist for item in row["neighbors"]}
    source_programs = {}
    for row in parquet_rows(source_path):
        uid = row["uid"]
        if uid in train_needed:
            _, key = checked_minimal(row["json_desc"], "deepcad")
            if key is None or sha256_text(key) != train_needed[uid] or uid in source_programs:
                raise ValueError(f"Source program does not match shortlist for {uid}")
            source_programs[uid] = program_signature(key)
    if set(source_programs) != set(train_needed):
        raise ValueError("Shortlisted training UID is missing from pinned source parquet")

    query_needed = {(row["domain"], row["uid"]): row["program_sha256"] for row in shortlist}
    if len(query_needed) != len(shortlist):
        raise ValueError("Duplicate query UID in shortlist")
    query_programs = {}
    for row in jsonl_rows(seed_path):
        identity = row["domain"], row["uid"]
        if identity in query_needed:
            key = program_key(row["minimal"])
            if row["first_exclusion"] is not None or sha256_text(key) != query_needed[identity] or identity in query_programs:
                raise ValueError(f"Query program does not match shortlist for {identity}")
            query_programs[identity] = program_signature(key)
    if set(query_programs) != set(query_needed):
        raise ValueError("Shortlisted query UID is missing from inventory seed")

    b_uids = {uid for domain, uid in query_programs if domain == "deepcad"}
    b_vector_hashes = {row["uid"]: row["sha256"] for row in jsonl_rows(test_vectors_path) if row["uid"] in b_uids}
    if set(b_vector_hashes) != b_uids:
        raise ValueError("Official test cad_vec coverage is incomplete for B")
    compared_hashes = set(b_vector_hashes.values())
    vector_exposures = defaultdict(list)
    for row in jsonl_rows(train_vectors_path):
        if row["sha256"] in compared_hashes:
            vector_exposures[row["sha256"]].append(row["uid"])

    output = []
    pair_status_counts = Counter()
    evidence_by_domain = defaultdict(Counter)
    status_by_domain = defaultdict(Counter)
    for candidate in shortlist:
        domain, uid = candidate["domain"], candidate["uid"]
        query_program = query_programs[domain, uid]
        b_vector_matches = sorted(vector_exposures.get(b_vector_hashes.get(uid), ())) if domain == "deepcad" else []
        neighbors = []
        evidence = set()
        statuses = set()
        for neighbor in candidate["neighbors"]:
            comparison = corroborate_program_pair(query_program, source_programs[neighbor["uid"]], neighbor["distance_sq"])
            evidence.update(comparison["evidence"])
            statuses.add(comparison["status"])
            pair_status_counts[comparison["status"]] += 1
            neighbors.append({**neighbor, **comparison})
        if b_vector_matches:
            evidence.add("official_cad_vec_exact")
            statuses.add("confirmed")
        status = next(status for status in ("confirmed", "suspected", "no_corroboration") if status in statuses)
        status_by_domain[domain][status] += 1
        evidence_by_domain[domain].update(evidence)
        output.append({
            "domain": domain, "uid": uid, "program_sha256": candidate["program_sha256"], "status": status,
            "nearest_uids": [neighbor["uid"] for neighbor in neighbors], "evidence": sorted(evidence),
            "official_cad_vec_sha256": b_vector_hashes.get(uid) if domain == "deepcad" else None,
            "official_cad_vec_train_uids": b_vector_matches, "neighbors": neighbors,
        })
    atomic_json(output_path, output, lines=True)
    return {
        "query_uid_count": len(output), "shortlisted_training_uid_count": len(source_programs),
        "pairs_compared": sum(pair_status_counts.values()), "pair_status_counts": dict(pair_status_counts),
        "query_status_by_domain": {domain: dict(counts) for domain, counts in status_by_domain.items()},
        "query_evidence_by_domain": {domain: dict(sorted(counts.items())) for domain, counts in evidence_by_domain.items()},
        "B_official_vector_coverage": len(b_vector_hashes), "C_official_vector_coverage": 0,
        "criteria": {
            "confirmed": "Literal equality of canonical geometry-only program strings or exact same published cad_vec SHA-256",
            "matching_topology": "Same part, face, loop and curve identifiers, list lengths, categorical values and numeric positions",
            "suspected_numeric": "Same program topology and maximum absolute delta in published numeric fields <= 1/255. "
                                 "One 8-bit grid step is the fixed comparison scale, including for native numeric fields in other units. "
                                 "This is a screen and has no geometric identity interpretation.",
            "suspected_coarse": "Zero distance under the frozen rounded coarse feature representation",
            "no_corroboration": "No evidence under these checks. Coverage is limited to five program neighbors and the "
                                "separate published official-vector equality check for B.",
            "geometric_identity": "No independent entity comparison in this stage",
        },
        "numeric_step": NATIVE_NUMERIC_SUSPICION_STEP, "output_sha256": sha256_file(output_path),
    }
