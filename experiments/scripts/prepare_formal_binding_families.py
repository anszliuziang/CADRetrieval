#!/usr/bin/env python3
"""Freeze and validate the geometry-only Table 4 fixtures without loading model weights."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from idea01 import formal_binding_families as families
from idea01 import formal_binding_occ as occ
from idea01 import formal_binding_review as review
from idea01 import in_domain_pilot_geometry as pilot


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def wording(path):
    if path is None:
        return families.DRAFT_PARAPHRASES, {"status": "draft_second_wording", "independent_writer": None}
    document = json.loads(path.read_bytes())
    writer = document["writer"]
    variants = document["wordings"]
    if not isinstance(writer, str) or not writer.strip() or set(variants) != set(families.DRAFT_PARAPHRASES):
        raise ValueError("provide a named independent wording author and both rephrased templates")
    return variants, {"status": "provided_for_blind_review", "independent_writer": writer, "sha256": sha(path)}


def exclusions(path, candidates):
    if path is None:
        return {}
    records = json.loads(path.read_bytes())
    allowed = {"ambiguous_english", "paraphrase_mismatch", "anchor_not_visible", "hole_not_visible",
               "geometry_error", "unresolved_reviewer_disagreement"}
    result = {}
    for entry in records:
        candidate = entry["candidate_id"]
        if (candidate in result or candidate not in candidates or entry["reason"] not in allowed
                or not re.fullmatch(r"[0-9a-f]{64}", entry["review_evidence_sha256"])):
            raise ValueError("invalid blind exclusion or missing reviewer evidence")
        result[candidate] = entry
    return result


def history_rows(design, coordinate=False):
    pair_id = design.pair_id + ("-coord" if coordinate else "")
    result = {}
    for name in families.HISTORIES:
        features, cut_order = families.history_features(design, name, coordinate)
        raw = families.raw_program(features)
        result[name] = {"pair_id": pair_id, "history": name, "entity": name[0], "cut_order": cut_order,
                        "raw_program": raw, "raw_program_sha256": pilot.fingerprint(raw), "geometry": {}}
    return result


def pair_views(design, captions, rows):
    if design.slot.kind == "binding":
        axes = product((0, 1), families.CAPTION_VARIANTS)
    else:
        axes = [(design.slot.view_order, design.slot.view_variant)]
    result = []
    for order, variant in axes:
        members = [f"{entity}{order}" for entity in ("A", "B")]
        result.append({"order": order, "variant": variant, "histories": members,
                       "texts": [captions[entity][variant] for entity in ("A", "B")],
                       "geometry_sha256": [rows[member]["raw_program_sha256"] for member in members],
                       "first_cut": [rows[member]["cut_order"][0] for member in members]})
    return result


def operation_signature(raw):
    result = Counter()
    for entity in raw["entities"].values():
        if entity["type"] == "ExtrudeFeature":
            result[entity["operation"]] += 1
        else:
            for profile in entity["profiles"].values():
                for loop in profile["loops"]:
                    result.update(curve["type"] for curve in loop["profile_curves"])
    return result


def planned_candidate(design, paraphrases):
    source = history_rows(design)
    captions = families.captions(design, paraphrases)
    signatures = [operation_signature(source[key]["raw_program"]) for key in families.HISTORIES]
    if design.slot.kind == "binding" and any(value != signatures[0] for value in signatures[1:]):
        raise ValueError("a binding pair changed its operation or primitive tally")
    coords = history_rows(design, coordinate=True) if design.slot.coordinate else {}
    if coords and any(coords[key]["raw_program_sha256"] == source[key]["raw_program_sha256"] for key in source):
        raise ValueError("local-frame intervention left at least one raw program identical")
    return {"candidate_id": design.pair_id, "slot": asdict(design.slot), "slot_id": design.slot.id,
            "backup_rank": design.rank, "dimensions": asdict(design.dimensions),
            "caption_sha256": caption_hashes(captions),
            "raw_program_sha256": {name: row["raw_program_sha256"] for name, row in source.items()},
            "coordinate_raw_program_sha256": {name: row["raw_program_sha256"] for name, row in coords.items()},
            "operation_tallies": dict(zip(families.HISTORIES, signatures, strict=True))}


def caption_hashes(captions):
    return {entity: {variant: sha256(text.encode()).hexdigest() for variant, text in texts.items()}
            for entity, texts in captions.items()}


def replay_panel(implementation, design, rows):
    shapes = {}
    for member, row in rows.items():
        try:
            body, volumes = occ.replay_raw(implementation, row["raw_program"])
            result = occ.measure(body, design.family, design.slot.attribute, design.slot.kind,
                                 design.expected(member[0]))
            result["volume_after_each_feature"] = volumes
            if result["status"] == "ok":
                shapes[member] = body
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        row["geometry"]["raw"] = result
    measurements = {name: row["geometry"]["raw"] for name, row in rows.items()}
    return pilot.geometry_links(measurements, shapes), shapes


def pair_record(design, rows, captions, truth, coordinate=False):
    slot = design.slot
    pair_id = design.pair_id + ("-coord" if coordinate else "")
    return {"pair_id": pair_id, "family": design.family, "attribute": slot.attribute,
            "kind": "coordinate" if coordinate else slot.kind, "primary_slot": slot.id, "backup_rank": design.rank,
            "views": pair_views(design, captions, rows), "captions": captions,
            "caption_sha256": caption_hashes(captions), "truth": truth,
            "fixed_generation_view": {"order": slot.view_order, "variant": slot.view_variant},
            "pose_quarter_turns": slot.quarter_turns}


def local_reading(row, anchor_xy):
    program = row["raw_program"]
    centers = {}
    for role, event in zip(row["cut_order"], program["sequence"][-4::2], strict=True):
        sketch = program["entities"][event["entity"]]
        curves = sketch["profiles"]["profile_1"]["loops"][0]["profile_curves"]
        points = [curve["center_point"] for curve in curves if "center_point" in curve]
        if len(points) not in (1, 2):
            raise ValueError("expected one circle center or two slot arc centers")
        xy = tuple(sum(point[axis] for point in points) / len(points) for axis in "xy")
        centers[role] = xy
    distances = {role: math.dist(center, anchor_xy) for role, center in centers.items()}
    delta = distances["far"] - distances["near"]
    call = "tie" if abs(delta) <= 1e-7 else ("near" if delta > 0 else "far")
    return {"local_centers": centers, "local_distances": distances, "nearest": call}


def coordinate_record(implementation, design, captions, baseline_rows, baseline_shapes):
    transformed_rows = history_rows(design, coordinate=True)
    links, transformed_shapes = replay_panel(implementation, design, transformed_rows)
    same = {}
    shortcut = {}
    for name in families.HISTORIES:
        if name in transformed_shapes:
            try:
                same[name] = occ.compare_same_body(baseline_shapes[name], transformed_shapes[name])
            except Exception as exc:
                same[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        else:
            same[name] = {"status": "geometry_unverified"}
        anchor = baseline_rows[name]["geometry"]["raw"]["anchor_xy"]
        before = local_reading(baseline_rows[name], anchor)
        after = local_reading(transformed_rows[name], anchor)
        shortcut[name] = {"before": before, "after": after}
    geometry_verified = links["status"] == "verified" and all(item["status"] == "verified" for item in same.values())
    conflict = geometry_verified and all(
        state["before"]["nearest"] == "near" and state["after"]["nearest"] == "far" for state in shortcut.values()
    )
    pair = pair_record(design, transformed_rows, captions, {"raw_geometry": links, "same_as_parent": same},
                       coordinate=True)
    return {**pair, "parent_pair_id": design.pair_id, "coordinate_conflict": conflict,
            "coordinate_geometry_verified": geometry_verified, "raw_local_rule": shortcut}, transformed_rows


def group_key(slot):
    return (slot.family, slot.attribute, slot.kind)


def smoke_slots(all_slots):
    selected = {}
    for slot in all_slots:
        if slot.coordinate or slot.kind == "easy":
            selected.setdefault(group_key(slot), slot)
    return list(selected.values())


def preregister(out, all_slots, templates, metadata):
    source = {}
    for module in (families, occ, review, pilot):
        path = Path(module.__file__)
        source[path.name] = sha(path)
    source[Path(__file__).name] = sha(Path(__file__))
    catalog = {
        "schema": "idea01-formal-binding-catalog-v1", "author_source": families.SOURCE, "seed": families.SEED,
        "source_sha256": source, "families": {name: asdict(item) for name, item in families.FAMILIES.items()},
        "excluded_development_families": families.EXCLUDED_DEVELOPMENT_FAMILIES, "attributes": families.ATTRIBUTES,
        "planned_per_family_attribute_kind": families.PRIMARY_COUNT, "backups_per_slot": families.BACKUP_COUNT,
        "coordinate_subset_per_family_attribute": families.COORDINATE_COUNT,
        "anchor_distance": "Euclidean distance from the measured target bore center to the projected anchor in world XY.",
        "raw_local_coordinate_rule": (
            "Compare the two cutter profile centers in local XY to the measured anchor in the fixed baseline reference sketch. "
            "The transformed sketches rotate 180 degrees around their common center midpoint with compensating placement."
        ),
        "tolerances": {"distance_gap_fraction_of_max_extent": occ.MIN_RELATIVE_DISTANCE_GAP,
                       "min_far_over_near": occ.MIN_DISTANCE_RATIO, "min_diameter_ratio": occ.MIN_SIZE_RATIO,
                       "max_equal_diameter_ratio": occ.MAX_EQUAL_SIZE_RATIO,
                       "history_max_relative_symmetric_difference": occ.SAME_BODY_LIMIT,
                       "opposite_min_relative_symmetric_difference": 1e-3, "boolean_fuzzy": occ.BOOLEAN_FUZZY},
        "selection": "Ascending frozen backup rank. Replace only failed raw geometry or documented blind exclusions.",
        "family_independence": (
            "Separate original construction trees and seeded dimensions. All share matched target through cuts. "
            "Exact score-label exchangeability requires a separate justification."
        ),
        "wording": metadata,
    }
    save(out / "catalog.json", catalog)
    ids = set()
    with (out / "candidate_ledger.jsonl").open("xb") as stream:
        for slot in all_slots:
            for rank in range(families.BACKUP_COUNT + 1):
                entry = planned_candidate(families.design_at(slot, rank), templates)
                pilot.jsonl_write(stream, entry)
                if entry["candidate_id"] in ids:
                    raise ValueError("duplicate frozen candidate ID")
                ids.add(entry["candidate_id"])
    return ids


def write_pair(streams, pair, rows):
    pilot.jsonl_write(streams["pairs"], pair)
    for row in rows.values():
        pilot.jsonl_write(streams["histories"], row)


def prepare(out, slots, implementation, templates, excluded, *, previews):
    from contextlib import ExitStack

    review_dir = out / "review_packet"
    (review_dir / "solids").mkdir(parents=True)
    (review_dir / "README.md").write_text(review.REVIEW_INSTRUCTIONS, encoding="utf-8", newline="\n")
    files = {name: out / f"{name}.jsonl" for name in ("pairs", "histories", "geometry_attempts", "review_key")}
    files["items"] = review_dir / "items.jsonl"
    files["decision_template"] = review_dir / "decisions_template.jsonl"
    group_stats = {group_key(slot): Counter() for slot in slots}
    review_failures = []
    with ExitStack() as stack:
        streams = {name: stack.enter_context(path.open("xb")) for name, path in files.items()}
        for slot in slots:
            stats = group_stats[group_key(slot)]
            stats["requested"] += 1
            stats["coordinate_requested"] += slot.coordinate
            for rank in range(families.BACKUP_COUNT + 1):
                design = families.design_at(slot, rank)
                if (entry := excluded.get(design.pair_id)) is not None:
                    pilot.jsonl_write(streams["geometry_attempts"], {"candidate_id": design.pair_id,
                                       "status": "blind_exclusion", "evidence": entry})
                    stats["blind_exclusions"] += 1
                    continue
                rows = history_rows(design)
                truth, shapes = replay_panel(implementation, design, rows)
                pilot.jsonl_write(streams["geometry_attempts"], {"candidate_id": design.pair_id,
                                   "status": truth["status"], "geometry": truth, "histories": {
                                       name: row["geometry"]["raw"] for name, row in rows.items()}})
                if truth["status"] != "verified":
                    stats["geometry_failures"] += 1
                    continue
                stats["verified"] += 1
                stats["replaced"] += rank > 0
                captions = families.captions(design, templates)
                pair = pair_record(design, rows, captions, {"raw_geometry": truth})
                try:
                    item, blank, key = review.export(review_dir, design, pair, shapes, use_previews=previews)
                except Exception as exc:
                    review_failures.append({"pair_id": design.pair_id, "error": f"{type(exc).__name__}: {exc}"})
                else:
                    stats["review_packet"] += 1
                    stats["preview_warnings"] += len(key["preview_warnings"])
                    pair["review_id"] = item["review_id"]
                    for stream, record in (("items", item), ("decision_template", blank), ("review_key", key)):
                        pilot.jsonl_write(streams[stream], record)
                write_pair(streams, pair, rows)
                if slot.coordinate:
                    coordinate, transformed = coordinate_record(implementation, design, captions, rows, shapes)
                    write_pair(streams, coordinate, transformed)
                    stats["coordinate_geometry_verified"] += coordinate["coordinate_geometry_verified"]
                    stats["coordinate_conflict"] += coordinate["coordinate_conflict"]
                break
            else:
                stats["no_valid_candidate"] += 1
    groups = [{"family": family, "attribute": attribute, "kind": kind, **counts}
              for (family, attribute, kind), counts in group_stats.items()]
    files["review_index"] = review.write_index(review_dir)
    return groups, review_failures, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--text2cad-root", type=Path, help="Verified Text2CAD source tree containing CadSeqProc")
    parser.add_argument("--smoke", action="store_true", help="Check one slot per family, attribute and kind")
    parser.add_argument("--pre-register-only", action="store_true", help="Write the frozen catalog and candidate ledger")
    parser.add_argument("--no-previews", action="store_true", help="Export STEP for review without preview images")
    parser.add_argument("--paraphrases", type=Path, help="JSON with writer and both rephrased wording templates")
    parser.add_argument("--review-exclusions", type=Path, help="JSON list of candidate IDs, reasons and evidence hashes")
    args = parser.parse_args()
    if not args.pre_register_only and args.text2cad_root is None:
        parser.error("--text2cad-root is required for OCC replay")
    all_slots = list(families.slots())
    families.check_design_balance(all_slots)
    templates, metadata = wording(args.paraphrases)
    args.out.mkdir(parents=True, exist_ok=False)
    ids = preregister(args.out, all_slots, templates, metadata)
    initial = {"catalog_sha256": sha(args.out / "catalog.json"),
               "candidate_ledger_sha256": sha(args.out / "candidate_ledger.jsonl"), "candidates": len(ids),
               "planned_slots": len(all_slots), "human_review": "pending", "wording": metadata}
    if args.pre_register_only:
        save(args.out / "summary.json", {"status": "registered_only", **initial})
        return 0
    import OCC

    excluded = exclusions(args.review_exclusions, ids)
    implementation, provenance = pilot.activate_text2cad(args.text2cad_root)
    selected = smoke_slots(all_slots) if args.smoke else all_slots
    groups, failures, files = prepare(args.out, selected, implementation, templates, excluded,
                                      previews=not args.no_previews)
    no_valid = sum(row.get("no_valid_candidate", 0) for row in groups)
    coord_unverified = sum(row["coordinate_requested"] - row.get("coordinate_geometry_verified", 0) for row in groups)
    family_coverage = {}
    for name in families.FAMILIES:
        blocks = [row for row in groups if row["family"] == name]
        family_coverage[name] = {key: sum(row.get(key, 0) for row in blocks)
                                 for key in ("requested", "verified", "review_packet", "preview_warnings",
                                             "coordinate_requested", "coordinate_geometry_verified", "coordinate_conflict")}
    summary = {**initial, "mode": "smoke" if args.smoke else "full", "pythonocc": OCC.VERSION,
               "cadseqproc": provenance, "family_coverage": family_coverage, "groups": groups,
               "raw_geometry_missing_slots": no_valid,
               "coordinate_unverified_slots": coord_unverified, "review_export_failures": failures,
               "review_exclusions_sha256": sha(args.review_exclusions) if args.review_exclusions else None,
               "files_sha256": {name: sha(path) for name, path in files.items()}}
    save(args.out / "summary.json", summary)
    print(json.dumps({"output": str(args.out), "planned": len(selected), "raw_geometry_missing": no_valid,
                      "coordinate_unverified": coord_unverified, "review_export_failures": len(failures)}), flush=True)
    return 2 if no_valid or coord_unverified or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
