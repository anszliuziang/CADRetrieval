"""Freeze Table 3's released Fusion corpus before any conversion or fitting."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import tempfile


SOURCE = "cadmium-fusion-r1.0.1"
CADMIUM_REVISION = "ea7d06f44706ad49cf78f7740c570a8fb8b34f66"
PARQUET_SHA256 = "a00dea0d893a8edd1fbd02329bc7db571e8f6ed6eae0956e27a852f0b3f5755f"
SPLIT_SHA256 = "2de40371e99fb32c09ca11ba76995ab40fe465aa09609b4de6c81de1bc553ef4"
EXPECTED_SPLITS = {"train": 6011, "validation": 635, "test": 1661}
EXPECTED_OFFICIAL = {"train": 6900, "test": 1725}
EXPECTED_PROJECTS = {"train_seen": 1027, "validation_only": 127, "strict_unseen": 507}
UID = re.compile(r"[0-9]+_[0-9a-fA-F]{8}_[0-9]{4}")
CURVE_FIELDS = {
    "line": (("Start Point", 2), ("End Point", 2)),
    "arc": (("Start Point", 2), ("Mid Point", 2), ("End Point", 2)),
    "circle": (("Center", 2), ("Radius", 0)),
}
OPERATIONS = {"NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation",
              "IntersectFeatureOperation"}
EXTRUSION_FIELDS = ("extrude_depth_towards_normal", "extrude_depth_opposite_normal", "sketch_scale")
METADATA_KEYS = {"annotation", "caption", "description", "filename", "finalname", "finalshape", "name", "uid"}


class ProgramError(ValueError):
    """A geometry field cannot be accepted, with a stable reason for the ledger."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Source checksum mismatch for {path}")


def json_bytes(value, *, sort_keys=False) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys,
                      allow_nan=False).encode("utf-8")


def write_jsonl(out: Path, rows) -> None:
    with out.open("xb") as stream:
        for row in rows:
            stream.write(json_bytes(row) + b"\n")


def read_jsonl(path: Path):
    with path.open("rb") as stream:
        yield from map(json.loads, stream)


def metadata_key(key) -> bool:
    return isinstance(key, str) and re.sub(r"[ _-]", "", key.lower()) in METADATA_KEYS


def perturb_metadata(obj, *, remove):
    if isinstance(obj, dict):
        return {key: ("metadata regression witness" if metadata_key(key)
                      else perturb_metadata(value, remove=remove))
                for key, value in obj.items() if not (remove and metadata_key(key))}
    if isinstance(obj, (list, tuple)):
        return [perturb_metadata(value, remove=remove) for value in obj]
    return obj


def record(value, label):
    if not isinstance(value, dict):
        raise ProgramError(f"invalid_{label}")
    return value


def children(value, label):
    visible = {key: child for key, child in record(value, label).items() if not metadata_key(key)}
    if not visible or any(not isinstance(key, str) or not isinstance(child, dict)
                          for key, child in visible.items()):
        raise ProgramError(f"invalid_{label}")
    return visible


def number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ProgramError(f"invalid_{field}")
    return value


def fields(value, names):
    data = record(value, "geometry_record")
    result = {}
    for field, width in names:
        entry = data.get(field)
        if width:
            if not isinstance(entry, (list, tuple)) or len(entry) != width:
                raise ProgramError(f"invalid_{field}")
            result[field] = [number(item, field) for item in entry]
        else:
            result[field] = number(entry, field)
    return result


def clean_program(value):
    """Validate a Minimal program and retain only typed geometry in its original order."""
    parts = {}
    for part_index, part in enumerate(children(record(value, "program").get("parts"), "parts").values(), 1):
        coord = fields(part.get("coordinate_system"), (("Euler Angles", 3), ("Translation Vector", 3)))
        extrude = record(part.get("extrusion"), "extrusion")
        operation = extrude.get("operation")
        if not isinstance(operation, str) or operation not in OPERATIONS:
            raise ProgramError("unsupported_operation")
        extrusion = fields(extrude, ((name, 0) for name in EXTRUSION_FIELDS))
        if extrusion["sketch_scale"] <= 0:
            raise ProgramError("invalid_sketch_scale")
        if (extrusion["extrude_depth_towards_normal"] < 0 or extrusion["extrude_depth_opposite_normal"] < 0
                or not (extrusion["extrude_depth_towards_normal"] or extrusion["extrude_depth_opposite_normal"])):
            raise ProgramError("invalid_extrusion_depth")
        extrusion["operation"] = operation
        sketch = {}
        for face_index, face in enumerate(children(part.get("sketch"), "faces").values(), 1):
            loops = {}
            for loop_index, loop in enumerate(children(face, "loops").values(), 1):
                counts = Counter()
                curves = {}
                for label, curve in children(loop, "curves").items():
                    kind = label.split("_", 1)[0]
                    if kind not in CURVE_FIELDS:
                        raise ProgramError("unsupported_curve")
                    counts[kind] += 1
                    geometry = fields(curve, CURVE_FIELDS[kind])
                    if kind == "circle" and geometry["Radius"] <= 0:
                        raise ProgramError("invalid_circle_radius")
                    curves[f"{kind}_{counts[kind]}"] = geometry
                if bool(counts["circle"]) != (len(curves) == 1):
                    raise ProgramError("unsupported_closed_loop")
                loops[f"loop_{loop_index}"] = curves
            sketch[f"face_{face_index}"] = loops
        parts[f"part_{part_index}"] = {"coordinate_system": coord, "sketch": sketch, "extrusion": extrusion}
    return {"parts": parts}


def read_program(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ProgramError("missing_json_desc")
    try:
        tree = ast.parse(raw, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                if len(keys) != len(node.keys) or len(keys) != len(set(keys)):
                    raise ProgramError("invalid_or_duplicate_literal_key")
        value = ast.literal_eval(tree)
    except (SyntaxError, TypeError, RecursionError) as exc:
        raise ProgramError("malformed_literal") from exc
    except ValueError as exc:
        if isinstance(exc, ProgramError):
            raise
        raise ProgramError("malformed_literal") from exc
    program = clean_program(value)
    frozen = json_bytes(program)
    for remove in (False, True):
        if json_bytes(clean_program(perturb_metadata(value, remove=remove))) != frozen:
            raise AssertionError("Descriptive metadata changed candidate input geometry")
    return program


def structure(program):
    counts = Counter()
    for part in program["parts"].values():
        counts["extrusions"] += 1
        counts[part["extrusion"]["operation"]] += 1
        for face in part["sketch"].values():
            counts["faces"] += 1
            for loop in face.values():
                counts["loops"] += 1
                counts.update(key.split("_", 1)[0] for key in loop)
    return dict(counts)


def load_splits(path):
    observed = json.loads(path.read_bytes())
    if not isinstance(observed, dict) or observed.keys() != EXPECTED_OFFICIAL.keys():
        raise ValueError("Unexpected Autodesk split fields")
    splits = {}
    for split, expected in EXPECTED_OFFICIAL.items():
        values = observed[split]
        if len(values) != expected or not all(isinstance(uid, str) and UID.fullmatch(uid) for uid in values):
            raise ValueError(f"Invalid Autodesk {split} UID axis")
        splits[split] = set(values)
        if len(splits[split]) != expected:
            raise ValueError(f"Duplicate Autodesk {split} UID")
    if splits["train"] & splits["test"]:
        raise ValueError("Autodesk project-component lists overlap at the UID level")
    return splits


def read_catalog(path):
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    columns = ("uid", "annotation", "json_desc")
    if set(source.schema_arrow.names) != set(columns) or source.metadata.num_rows != sum(EXPECTED_SPLITS.values()):
        raise ValueError("Unexpected CADmium Fusion release schema or count")
    catalog = {}
    for batch in source.iter_batches(columns=list(columns), batch_size=512):
        for released in batch.to_pylist():
            uid, annotation, raw = (released[key] for key in columns)
            if not isinstance(uid, str) or UID.fullmatch(uid) is None or uid in catalog:
                raise ValueError("Invalid or duplicate CADmium Fusion UID")
            if annotation is not None and not isinstance(annotation, str):
                raise ValueError("CADmium annotation column must be text")
            metadata = {"uid": uid, "project_id": uid.split("_", 1)[0],
                        "annotation_status": "ok" if annotation and annotation.strip() else "missing",
                        "text_sha256": sha256_bytes(annotation.encode("utf-8")) if annotation is not None else None,
                        "json_desc_sha256": sha256_bytes(raw.encode("utf-8")) if isinstance(raw, str) else None}
            try:
                program = read_program(raw)
            except ProgramError as exc:
                metadata["minimal_status"] = str(exc)
                program = None
            else:
                metadata.update(minimal_status="ok", geometry_structure=structure(program),
                                minimal_ordered_sha256=sha256_bytes(json_bytes(program)),
                                minimal_exact_key_sha256=sha256_bytes(json_bytes(program, sort_keys=True)),
                                metadata_replacement_and_deletion_invariant=True)
            catalog[uid] = {"text": annotation, "program": program, "record": metadata}
    return catalog


def hash_order(keys, prefix):
    return sorted(keys, key=lambda key: (sha256_bytes((prefix + key).encode("utf-8")), key))


def partition(catalog, official):
    if set(catalog) - (official["train"] | official["test"]):
        raise ValueError("Published CADmium UID absent from Autodesk's official split")
    train_side = official["train"] & catalog.keys()
    projects = {catalog[uid]["record"]["project_id"] for uid in train_side}
    if len(train_side) != 6646 or len(projects) != 3075:
        raise ValueError("Unexpected Fusion official training-side publication coverage")
    validation_projects = set(hash_order(projects, "cadretrieval-fusion-val-v1|")[:308])
    training_projects = projects - validation_projects
    splits = {"train": [], "validation": [], "test": []}
    for uid in sorted(catalog):
        ledger = catalog[uid]["record"]
        project = ledger["project_id"]
        official_split = "train" if uid in train_side else "test"
        ledger["autodesk_split"] = official_split
        if official_split == "train":
            split = "validation" if project in validation_projects else "train"
            group = None
        else:
            split = "test"
            group = ("train_seen" if project in training_projects else
                     "validation_only" if project in validation_projects else "strict_unseen")
        ledger.update(split=split, test_project_group=group)
        splits[split].append(uid)
    groups = Counter(catalog[uid]["record"]["test_project_group"] for uid in splits["test"])
    if {key: len(value) for key, value in splits.items()} != EXPECTED_SPLITS or groups != EXPECTED_PROJECTS:
        raise ValueError("Frozen Fusion split disagrees with the registered counts")
    return splits


def query_axes(catalog, ordered_uids, split):
    queries, qrels = [], []
    for index, uid in enumerate(ordered_uids):
        source = catalog[uid]
        entry = source["record"]
        query_id = f"fusion-{split}-{index:04d}"
        queries.append({"query_id": query_id, "query_index": index, "level": "annotation",
                        "text": source["text"], "text_sha256": entry["text_sha256"],
                        "status": entry["annotation_status"]})
        qrels.append({"query_id": query_id, "uid": uid, "test_project_group": entry["test_project_group"]})
    return queries, qrels


def exact_duplicates(catalog, splits):
    result = {}
    for field in ("text_sha256", "minimal_exact_key_sha256"):
        by_split = {split: Counter(catalog[uid]["record"].get(field) for uid in uids)
                    for split, uids in splits.items()}
        for counts in by_split.values():
            counts.pop(None, None)
        train = by_split["train"]
        result[field] = {
            "within_split_repeated_rows": {split: sum(n for n in counts.values() if n > 1)
                                            for split, counts in by_split.items()},
            "validation_rows_with_train_match": sum(n for key, n in by_split["validation"].items() if key in train),
            "test_rows_with_train_match": sum(n for key, n in by_split["test"].items() if key in train),
        }
    return result


def emit_catalog(catalog, splits, official, out):
    validation = splits["validation"]
    panel = hash_order(validation, f"cadretrieval-val-q-v1|{SOURCE}|")[:512]
    background = {catalog[uid]["text"].strip() for uid in validation
                  if catalog[uid]["record"]["annotation_status"] == "ok"}
    background_text = hash_order(background, "cadretrieval-bg-v1|")[:48]
    if len(background_text) != 48:
        raise ValueError("Fusion validation has fewer than 48 distinct nonempty annotations")

    write_jsonl(out / "source_inventory.jsonl", (catalog[uid]["record"] for uid in sorted(catalog)))
    write_jsonl(out / "source_programs.jsonl", (
        {"uid": uid, "program": catalog[uid]["program"]} for uid in sorted(catalog)
        if catalog[uid]["program"] is not None))
    write_jsonl(out / "train_pairs.jsonl", (
        {"uid": uid, "text": catalog[uid]["text"], "text_sha256": catalog[uid]["record"]["text_sha256"],
         "status": catalog[uid]["record"]["annotation_status"]} for uid in splits["train"]))
    for split, queries_for in (("validation", panel), ("test", splits["test"])):
        queries, qrels = query_axes(catalog, queries_for, split)
        write_jsonl(out / f"{split}_queries.jsonl", queries)
        write_jsonl(out / f"{split}_qrels.jsonl", qrels)
        write_jsonl(out / f"{split}_gallery.jsonl", (
            {"cad_index": index, "uid": uid} for index, uid in enumerate(splits[split])))
    write_jsonl(out / "background.jsonl", (
        {"background_index": index, "text": text, "text_sha256": sha256_bytes(text.encode("utf-8"))}
        for index, text in enumerate(background_text)))

    manifest = {
        "source": SOURCE, "CADmium_revision": CADMIUM_REVISION,
        "published_parquet_sha256": PARQUET_SHA256, "autodesk_split_sha256": SPLIT_SHA256,
        "autodesk_official_counts": EXPECTED_OFFICIAL, "frozen_counts": EXPECTED_SPLITS,
        "published_counts_by_autodesk_split": {key: len(uids & catalog.keys()) for key, uids in official.items()},
        "validation_projects": 308, "test_project_groups": EXPECTED_PROJECTS,
        "validation_queries": len(panel), "validation_gallery": len(validation),
        "background_count": len(background_text), "annotation_field": "annotation",
        "query_text_retained_verbatim": True, "background_strip_then_deduplicate": True,
        "minimal_status_by_split": {
            split: dict(Counter(catalog[uid]["record"]["minimal_status"] for uid in uids))
            for split, uids in splits.items()},
        "annotation_status_by_split": {
            split: dict(Counter(catalog[uid]["record"]["annotation_status"] for uid in uids))
            for split, uids in splits.items()},
        "exact_duplicates": exact_duplicates(catalog, splits),
        "conversion_and_independent_geometry_status": "pending_replay",
        "file_sha256": {path.name: sha256_file(path) for path in sorted(out.glob("*.jsonl"))},
    }
    (out / "manifest.json").write_bytes(json_bytes(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    require_sha(args.parquet, PARQUET_SHA256)
    require_sha(args.official_split, SPLIT_SHA256)
    official = load_splits(args.official_split)
    catalog = read_catalog(args.parquet)
    splits = partition(catalog, official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fusion-catalog-", dir=args.output.parent) as stage:
        payload = Path(stage) / "output"
        payload.mkdir()
        manifest = emit_catalog(catalog, splits, official, payload)
        payload.rename(args.output)
    print(json.dumps({"output": str(args.output), "counts": manifest["frozen_counts"],
                      "test_project_groups": manifest["test_project_groups"],
                      "minimal_status": manifest["minimal_status_by_split"]}))


if __name__ == "__main__":
    main()
