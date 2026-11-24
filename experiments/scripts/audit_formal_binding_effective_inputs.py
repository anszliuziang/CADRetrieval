#!/usr/bin/env python3
"""Audit fixed Table 4 effective inputs and their OCC replay without model weights."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from idea01 import formal_binding_families as families
from idea01 import formal_binding_occ as occ
from idea01 import in_domain_pilot_geometry as geo
from evaluate_formal_binding import keyed, load_view_axes, read_pinned, require
from run_in_domain_pilot_geometry import (
    activate,
    activate_hiercad,
    attempt,
    encode_hiercad_panel,
    environment,
    load_hiercad_tokenizer,
    unavailable,
)


MODELS = geo.MODELS
MEMBERS = geo.HISTORIES
TEXT2CAD_DEPTH = 8


def read_fixtures(directory, summary_sha):
    receipts = {}
    summary = json.loads(read_pinned(directory / "summary.json", summary_sha, receipts))
    require(summary["mode"] in ("full", "smoke") and summary["human_review"] == "pending",
            "expected a prepared raw geometry fixture receipt")
    catalog = json.loads(read_pinned(directory / "catalog.json", summary["catalog_sha256"], receipts))
    require(set(catalog["families"]) == set(families.FAMILIES)
            and set(catalog["excluded_development_families"]) == set(families.EXCLUDED_DEVELOPMENT_FAMILIES),
            "prepared family catalog differs from the known held-out geometry rules")

    def rows(name, digest):
        return [json.loads(line) for line in read_pinned(directory / f"{name}.jsonl", digest, receipts).splitlines()]

    pairs = keyed(rows("pairs", summary["files_sha256"]["pairs"]), lambda row: row["pair_id"])
    histories = keyed(rows("histories", summary["files_sha256"]["histories"]),
                      lambda row: (row["pair_id"], row["history"]))
    candidates = keyed(rows("candidate_ledger", summary["candidate_ledger_sha256"]),
                       lambda row: row["candidate_id"])
    require(len(candidates) == summary["candidates"], "preregistered candidate count differs")
    require(histories.keys() == set(product(pairs, MEMBERS)), "fixture history axis differs from the pair axis")
    load_view_axes(list(pairs.values()), histories)

    coord_parent_ids = set()
    primary_slots = set()
    verified = Counter()
    coordinate_count = Counter()
    for pid, pair in pairs.items():
        family, attribute, kind = (pair[name] for name in ("family", "attribute", "kind"))
        require(family in catalog["families"] and attribute in geo.ATTRIBUTES, f"unknown axes for {pid}")
        parent_id = pair["parent_pair_id"] if kind == "coordinate" else pid
        candidate = candidates[parent_id]
        slot = candidate["slot"]
        raw_kind = slot["kind"]
        require((family, attribute) == (slot["family"], slot["attribute"])
                and pair["backup_rank"] == candidate["backup_rank"] and pair["primary_slot"] == candidate["slot_id"]
                and pair["caption_sha256"] == candidate["caption_sha256"],
                f"pair differs from the preregistered candidate {pid}")
        if kind == "coordinate":
            require(parent_id in pairs and parent_id not in coord_parent_ids and raw_kind == "binding"
                    and slot["coordinate"] and pairs[parent_id]["kind"] == "binding"
                    and pairs[parent_id]["captions"] == pair["captions"],
                    f"coordinate witness has an incorrect parent for {pid}")
            coord_parent_ids.add(parent_id)
            coordinate_count[family, attribute] += 1
            program_hashes = candidate["coordinate_raw_program_sha256"]
        else:
            require(kind == raw_kind and pair["primary_slot"] not in primary_slots
                    and pair["truth"]["raw_geometry"]["status"] == "verified",
                    f"invalid or duplicate raw primary slot {pid}")
            primary_slots.add(pair["primary_slot"])
            verified[family, attribute, kind] += 1
            program_hashes = candidate["raw_program_sha256"]
        member_rows = {member: histories[pid, member] for member in MEMBERS}
        for member, row in member_rows.items():
            require(row["entity"] == member[0] and tuple(sorted(row["cut_order"])) == ("far", "near")
                    and geo.fingerprint(row["raw_program"]) == row["raw_program_sha256"] == program_hashes[member],
                    f"raw history differs from the preregistered member {pid}/{member}")
            raw = row["geometry"]["raw"]
            if kind != "coordinate" or pair["coordinate_geometry_verified"]:
                require(raw["status"] == "ok" and raw["expected_binding"] == raw["measured_binding"],
                        f"raw truth missing for {pid}/{member}")
            row["registered_binding"] = expected_binding(slot, member[0])
            if raw["status"] == "ok":
                expected = row["registered_binding"]
                require(raw["expected_binding"] == (list(expected) if isinstance(expected, tuple) else expected),
                        f"registered and measured raw labels disagree for {pid}/{member}")
        for left, right in geo.HISTORY_LINKS:
            require(member_rows[left]["cut_order"] == member_rows[right]["cut_order"][::-1]
                    and program_hashes[left] != program_hashes[right],
                    f"cut histories do not reverse as authored for {pid}/{left}")
    for group in summary["groups"]:
        key = (group["family"], group["attribute"], group["kind"])
        require(verified[key] == group.get("verified", 0)
                and (key[2] == "easy" or coordinate_count[key[:2]] <= group["coordinate_requested"]),
                f"prepared coverage does not match the receipt for {key}")
    require(coord_parent_ids == {pid for pid, pair in pairs.items()
                                 if pair["kind"] == "binding" and candidates[pid]["slot"]["coordinate"]},
            "prepared coordinate axis omits a frozen subset parent")
    return pairs, histories, summary, receipts


def expected_binding(slot, entity):
    marked = (entity == "A") == slot["a_marked"]
    if slot["kind"] == "easy":
        return "different" if marked else "same"
    forms = geo.ATTRIBUTES[slot["attribute"]]
    return forms if marked else forms[::-1]


def select_smoke(pairs):
    """One easy pair and one matched binding/coordinate case per family and attribute."""
    coordinate_parents = {pair["parent_pair_id"] for pair in pairs.values() if pair["kind"] == "coordinate"}
    cells = {}
    for pid, pair in pairs.items():
        kind = pair["kind"]
        if kind == "easy" or (kind == "binding" and pid in coordinate_parents):
            cells.setdefault((pair["family"], pair["attribute"], kind), pid)
    selected = set(cells.values())
    return [pair for pid, pair in pairs.items() if pid in selected or pair.get("parent_pair_id") in selected]


def open_assets(config):
    assets = {}
    statuses = {}
    for name, loader in (
        ("text2cad_source", geo.activate_text2cad),
        ("spectrabrep_source", geo.activate_spectrabrep),
        ("text2cad_tokenizer", geo.text2cad_tokenizer),
        ("hiercad_repo", activate_hiercad),
        ("hiercad_tokenizer", load_hiercad_tokenizer),
    ):
        assets[name], statuses[name] = activate(name, config[name], loader)
    require(assets["text2cad_source"] is not None, f"pinned Text2CAD CAD source failed: {statuses['text2cad_source']}")
    return assets, statuses


def effective_sequence(sequence_type, arrays):
    return sequence_type.from_vec(arrays["cad_vec"].numpy().copy(), bit=TEXT2CAD_DEPTH, post_processing=False)


def sketch_center(sketch):
    circles = [curve for curve in sketch.all_curves if curve.curve_type == "circle"]
    arcs = [curve for curve in sketch.all_curves if curve.curve_type == "arc"]
    if len(circles) == 1 and not arcs:
        center = np.asarray(circles[0].metadata["center"], dtype=float)
    elif len(arcs) == 2 and not circles:
        centers = []
        for arc in arcs:
            points = np.asarray([arc.metadata[key] for key in ("start_point", "mid_point", "end_point")], dtype=float)
            squared = (points * points).sum(axis=1)
            centers.append(np.linalg.solve(2 * (points[1:] - points[0]), squared[1:] - squared[0]))
        center = np.mean(centers, axis=0)
    else:
        raise ValueError("effective cutter profile is neither one circle nor two end arcs")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("nonfinite effective sketch center")
    return center.tolist()


def read_local(sequence, cut_order, anchor):
    centers = {role: sketch_center(sketch)
               for role, sketch in zip(cut_order, sequence.sketch_seq[-2:], strict=True)}
    distances = {role: math.dist(center, anchor) for role, center in centers.items()}
    delta = distances["far"] - distances["near"]
    nearest = "tie" if abs(delta) <= 1e-7 else ("near" if delta > 0 else "far")
    return {"status": "ok", "local_centers": centers, "distances_to_effective_anchor": distances, "nearest": nearest}


def measure_sequence(pair, row, sequence):
    shape, volumes = occ.replay_sequence(sequence)
    kind = "binding" if pair["kind"] == "coordinate" else pair["kind"]
    result = occ.measure(shape, pair["family"], pair["attribute"], kind, row["registered_binding"])
    result["volume_after_each_feature"] = volumes
    if result["status"] == "ok":
        local, error = attempt("local_cutter_coordinates", read_local, sequence, row["cut_order"], result["anchor_xy"])
        result["local_rule"] = error or local
    return shape, result


def replay(pair, row, stage, make_sequence):
    record, error = attempt(stage, lambda: measure_sequence(pair, row, make_sequence()))
    return (None, error) if error else record


def run_history(pair, row, assets):
    raw = row["raw_program"]
    sequence_type = assets["text2cad_source"]
    inputs = {}
    geometry = {}
    shapes = {}
    record = {"pair_id": pair["pair_id"], "history": row["history"], "entity": row["entity"],
              "raw_program_sha256": row["raw_program_sha256"], "cut_order": row["cut_order"],
              "geometry": geometry, "inputs": inputs}

    encoded, error = attempt("text2cad_program", geo.text2cad_program, sequence_type, raw)
    if error:
        inputs.update(dict.fromkeys(("text2cad", "command_dual_tower", "spectrabrep"), error))
        geometry["text2cad"] = unavailable("text2cad_replay", "encoder did not produce a CAD sequence")
    else:
        arrays, inputs["text2cad"] = encoded
        shapes["text2cad"], geometry["text2cad"] = replay(
            pair, row, "text2cad_quantized_replay", lambda: effective_sequence(sequence_type, arrays),
        )
        command, error = attempt("command_effective_tensor", geo.dual_tower_program, arrays)
        if command and [item[:2] for item in command["effective_input"]] != inputs["text2cad"]["program_rows"]:
            error = {"status": "failed", "stage": "command_effective_tensor",
                     "detail": "ordered command coordinates differ from the replayed Text2CAD rows"}
        inputs["command_dual_tower"] = error or {
            **command, "replay_stage": "text2cad", "coordinate_columns_match_replayed_rows": True,
        }
        if assets["spectrabrep_source"] is None:
            inputs["spectrabrep"] = unavailable("spectrabrep_source", "released descriptor source unavailable")
        else:
            spectra, error = attempt("spectrabrep_descriptor", geo.spectra_program, assets["spectrabrep_source"], arrays)
            inputs["spectrabrep"] = error or {**spectra, "replay_stage": "text2cad",
                "representation_stage": "released descriptor before checkpoint-specific feature normalization"}

    native = assets["hiercad_repo"]
    if native is None:
        inputs["hiercad"] = geometry["hiercad"] = unavailable("hiercad_repo", "pinned HierCAD source unavailable")
        return record, shapes
    minimal, error = attempt("text2cad_minimal", geo.normalized_minimal, sequence_type, raw)
    if error:
        inputs["hiercad"] = geometry["hiercad"] = error
        return record, shapes
    target, error = attempt("hiercad_target", native.hiercad_target, minimal)
    if error:
        inputs["hiercad"] = geometry["hiercad"] = error
        return record, shapes
    record["hiercad_serialized"] = {
        "status": "ok", "input_sha256": sha256(target.response.encode()).hexdigest(), "response": target.response,
        "replay_program": target.replay_program, "replay_program_sha256": geo.fingerprint(target.replay_program),
    }
    shapes["hiercad"], geometry["hiercad"] = replay(
        pair, row, "hiercad_effective_replay", lambda: geo.minimal_replay_sequence(sequence_type, target.replay_program),
    )
    if assets["hiercad_tokenizer"] is None:
        inputs["hiercad"] = unavailable("hiercad_tokenizer", "tokens unavailable; released serialization recorded")
    else:
        tokenized, error = attempt("hiercad_tokenize_panel", encode_hiercad_panel, native,
                                   assets["hiercad_tokenizer"], target, pair["captions"])
        inputs["hiercad"] = error or tokenized
    return record, shapes


def link_inputs(records):
    checks = {}
    for key, axis in (("same_entity_histories", geo.HISTORY_LINKS), ("opposite_entities", geo.ENTITY_LINKS)):
        comparisons = [{"histories": [left, right], **geo.input_link(records[left], records[right])}
                       for left, right in axis]
        checks[key] = {"comparisons": comparisons, "counts": dict(Counter(item["status"] for item in comparisons))}
    return checks


def before_after(current, parent):
    evidence = {}
    if parent is None:
        return {"status": "unavailable", "effective_coordinate_conflict": False, "per_history": evidence}
    now_rows, now_shapes = current
    old_rows, old_shapes = parent
    for member in MEMBERS:
        if (now_rows[member]["status"] != "ok" or old_rows[member]["status"] != "ok"
                or now_shapes.get(member) is None or old_shapes.get(member) is None):
            evidence[member] = {"status": "geometry_unverified"}
            continue
        equivalent, error = attempt("coordinate_effective_symdiff", occ.compare_same_body,
                                    old_shapes[member], now_shapes[member])
        before = old_rows[member].get("local_rule", {})
        after = now_rows[member].get("local_rule", {})
        evidence[member] = {**(error or equivalent), "before_local_rule": before, "after_local_rule": after,
                            "flipped": before.get("nearest") == "near" and after.get("nearest") == "far"}
    verified = all(item["status"] == "verified" for item in evidence.values())
    return {"status": "verified" if verified else "geometry_unverified", "per_history": evidence,
            "effective_coordinate_conflict": verified and all(item["flipped"] for item in evidence.values())}


def easy_replay_check(bundle, members):
    measurements, shapes = bundle
    left, right = members
    statuses = [measurements[member]["status"] for member in (left, right)]
    if statuses != ["ok", "ok"]:
        return {"status": "geometry_unverified", "members": members, "replay_statuses": statuses}
    witness, error = attempt("easy_effective_symmetric_difference", occ.compare_same_body, shapes[left], shapes[right])
    if error:
        return {"status": "geometry_unverified", "members": members, "error": error}
    ratio = witness["relative_symmetric_difference"]
    return {"status": "verified" if ratio >= 1e-3 else "geometry_mismatch",
            "members": members, "relative_symmetric_difference": ratio}


def model_status(pair, model, stage_links, inputs, written, coord):
    if pair["kind"] == "coordinate" and not pair["coordinate_geometry_verified"]:
        return "failed", "raw_coordinate_geometry_unverified"
    use = written if model == "hiercad" else inputs
    problems = sorted({item["status"] for item in use.values()} - {"ok"})
    if problems:
        return "failed", "candidate_input_" + "+".join(problems)
    if model == "hiercad":
        problems = sorted({item["status"] for item in inputs.values()} - {"ok"})
        if problems:
            return "failed", "hiercad_tokenizer_" + "+".join(problems)
    if stage_links["status"] != "verified":
        return "failed", "effective_geometry_unverified"
    if coord and coord["status"] != "verified":
        return "failed", "coordinate_effective_geometry_unverified"
    return "verified", None


def audit_pair(pair, source, assets, parent):
    rows = {}
    stages = {stage: ({}, {}) for stage in ("text2cad", "hiercad")}
    for member, row in source.items():
        result, shapes = run_history(pair, row, assets)
        rows[member] = result
        for stage, (measurements, bodies) in stages.items():
            measurements[member] = result["geometry"][stage]
            if measurements[member]["status"] == "ok":
                bodies[member] = shapes[stage]
    stage_links = {stage: geo.geometry_links(*bundle) for stage, bundle in stages.items()}
    scoring_members = pair["views"][0]["histories"] if pair["kind"] == "easy" else MEMBERS
    scoring_replay = ({stage: easy_replay_check(bundle, scoring_members) for stage, bundle in stages.items()}
                      if pair["kind"] == "easy" else stage_links)
    coordinate = {}
    if pair["kind"] == "coordinate":
        coordinate = {stage: before_after(bundle, parent.get(stage) if parent else None)
                      for stage, bundle in stages.items()}
    caption_tokens = unavailable("text2cad_tokenizer", "official BERT tokenizer unavailable")
    if assets["text2cad_tokenizer"] is not None:
        output, error = attempt("text2cad_caption_tokens", geo.text2cad_captions,
                                assets["text2cad_tokenizer"], pair["captions"])
        caption_tokens = error or output
    by_model = {}
    written = {member: row.get("hiercad_serialized", row["inputs"]["hiercad"]) for member, row in rows.items()}
    field_counts = {member: row["inputs"]["text2cad"].get("valid_fields") for member, row in rows.items()}
    for model in MODELS:
        source_stage = "hiercad" if model == "hiercad" else "text2cad"
        model_inputs = {member: row["inputs"][model] for member, row in rows.items()}
        coord = coordinate.get(source_stage)
        status, error = model_status(pair, model, scoring_replay[source_stage],
                                     {member: model_inputs[member] for member in scoring_members},
                                     {member: written[member] for member in scoring_members}, coord)
        by_model[model] = {"status": status, "failure_reason": error, "replay_stage": source_stage,
                           "candidate_input_statuses": {member: item["status"] for member, item in model_inputs.items()},
                           "effective_identity": link_inputs(model_inputs),
                           "effective_coordinate_conflict": bool(coord and status == "verified"
                               and pair["coordinate_conflict"] and coord["effective_coordinate_conflict"])}
    by_model["hiercad"]["serialized_identity"] = link_inputs(written)
    return ({
        "pair_id": pair["pair_id"], "family": pair["family"], "attribute": pair["attribute"], "kind": pair["kind"],
        "parent_pair_id": pair.get("parent_pair_id"), "replay": stage_links, "coordinate_replay": coordinate,
        "scoring_replay": scoring_replay, "scoring_members": scoring_members,
        "models": by_model, "text2cad_caption_tokens": caption_tokens, "text2cad_valid_fields": field_counts,
        "text2cad_valid_fields_matched": None not in field_counts.values() and len(set(field_counts.values())) == 1,
    }, rows, stages)


def out_files(out):
    files = {name: out / f"{name}.jsonl" for name in ("pairs", "histories")}
    files.update({model: out / f"effective_geometry_{model}.jsonl" for model in MODELS})
    return files


def run_cases(out, pairs, histories, assets):
    out.mkdir(parents=True, exist_ok=True)
    files = out_files(out)
    planned_companions = {pair["parent_pair_id"] for pair in pairs if pair["kind"] == "coordinate"}
    cache = {}
    group_counts = {}
    with ExitStack() as stack:
        streams = {name: stack.enter_context(path.open("xb")) for name, path in files.items()}
        for pair in pairs:
            pid = pair["pair_id"]
            source = {member: histories[pid, member] for member in MEMBERS}
            parent = cache.pop(pair.get("parent_pair_id"), None)
            result, records, stages = audit_pair(pair, source, assets, parent)
            if pid in planned_companions:
                cache[pid] = stages
            pair_digest = geo.fingerprint(result)
            geo.jsonl_write(streams["pairs"], result)
            for row in records.values():
                geo.jsonl_write(streams["histories"], row)
            for model, model_result in result["models"].items():
                key = (model, pair["family"], pair["attribute"], pair["kind"])
                counts = group_counts.setdefault(key, Counter())
                counts["denominator"] += 1
                counts[model_result["status"]] += 1
                if model_result["failure_reason"] is not None:
                    counts["failure:" + model_result["failure_reason"]] += 1
                counts["effective_coordinate_conflict"] += model_result["effective_coordinate_conflict"]
                for check, cells in model_result["effective_identity"].items():
                    counts.update({f"{check}:{name}": count for name, count in cells["counts"].items()})
                for status in model_result["candidate_input_statuses"].values():
                    counts["candidate_input:" + status] += 1
                geo.jsonl_write(streams[model], {
                    "pair_id": pid, "status": model_result["status"], "failure_reason": model_result["failure_reason"],
                    "evidence": f"pairs.jsonl/{pid}/{model}", "pair_audit_sha256": pair_digest,
                    "effective_coordinate_conflict": model_result["effective_coordinate_conflict"],
                })
    return group_counts, files


def run_family(task):
    out, pairs, histories, config = task
    assets, _ = open_assets(config)
    return run_cases(out, pairs, histories, assets)


def parallel(out, selected, histories, config, workers):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from shutil import copyfileobj
    from tempfile import TemporaryDirectory

    counts = {}
    files = out_files(out)
    with TemporaryDirectory(prefix=".input-audit-", dir=out) as scratch:
        tasks = []
        for family in families.FAMILIES:
            subset = [pair for pair in selected if pair["family"] == family]
            if subset:
                source = {(pair["pair_id"], member): histories[pair["pair_id"], member]
                          for pair in subset for member in MEMBERS}
                tasks.append((Path(scratch) / family, subset, source, config))
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(run_family, tasks))
        for group, _ in results:
            require(counts.keys().isdisjoint(group), "family worker returned a duplicate audit group")
            counts.update(group)
        for name, path in files.items():
            with path.open("xb") as output:
                for _, shard in results:
                    with shard[name].open("rb") as source:
                        copyfileobj(source, output)
    return counts, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--prepared-summary-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--text2cad-source", type=Path, required=True)
    parser.add_argument("--text2cad-tokenizer", type=Path)
    parser.add_argument("--spectrabrep-source", type=Path)
    parser.add_argument("--hiercad-repo", type=Path)
    parser.add_argument("--hiercad-tokenizer", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(families.FAMILIES):
        parser.error(f"--workers must be in 1..{len(families.FAMILIES)}")
    start = time.perf_counter()
    pairs, histories, prepared, receipts = read_fixtures(args.fixtures, args.prepared_summary_sha256)
    selected = select_smoke(pairs) if args.smoke else list(pairs.values())
    require(bool(selected), "no prepared cases selected")
    config = {name: getattr(args, name) for name in
              ("text2cad_source", "text2cad_tokenizer", "spectrabrep_source", "hiercad_repo", "hiercad_tokenizer")}
    assets, statuses = open_assets(config)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.workers == 1:
        group_counts, files = run_cases(args.out, selected, histories, assets)
    else:
        group_counts, files = parallel(args.out, selected, histories, config, args.workers)
    groups = [{"model": model, "family": family, "attribute": attr, "kind": kind, **value}
              for (model, family, attr, kind), value in group_counts.items()]
    model_counts = {}
    for model in MODELS:
        totals = Counter()
        for (name, _, _, _), counts in group_counts.items():
            if name == model:
                totals.update(counts)
        model_counts[model] = totals
    summary = {
        "schema": "idea01-formal-binding-effective-input-audit-v1", "scope": "weight-free CPU input audit",
        "mode": "smoke" if args.smoke else prepared["mode"], "workers": args.workers,
        "preregistered_full_slots": prepared["planned_slots"],
        "raw_fixture_requested_slots": sum(group["requested"] for group in prepared["groups"]),
        "raw_fixture_requested_coordinate_slots": sum(group["coordinate_requested"] for group in prepared["groups"]),
        "raw_fixture_missing_slots": prepared["raw_geometry_missing_slots"], "prepared_pairs": len(pairs),
        "audited_pairs": len(selected), "audited_histories": len(selected) * len(MEMBERS),
        "fixture_sha256": receipts, "assets": statuses, "environment": environment(),
        "groups": groups, "model_counts": model_counts,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "files_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        "wall_seconds": time.perf_counter() - start,
    }
    (args.out / "summary.json").write_bytes(geo.json_bytes(summary) + b"\n")
    print(json.dumps({"audited_pairs": len(selected), "model_counts": model_counts,
                      "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
