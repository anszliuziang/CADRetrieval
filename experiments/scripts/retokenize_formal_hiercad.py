#!/usr/bin/env python3
"""Tokenize previously audited HierCAD trajectories without replaying geometry or scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import time

from audit_formal_binding_effective_inputs import MEMBERS, link_inputs, read_fixtures
from evaluate_formal_binding import keyed, read_pinned, require
from run_in_domain_pilot_geometry import (
    activate_hiercad,
    attempt,
    encode_hiercad_panel,
    load_hiercad_tokenizer,
    unavailable,
)
from idea01 import in_domain_pilot_geometry as geo


def read_audit(directory, audit_sha, fixture_dir):
    receipts = {}
    audit = json.loads(read_pinned(directory / "summary.json", audit_sha, receipts))
    require(audit["schema"] == "idea01-formal-binding-effective-input-audit-v1",
            "unsupported immutable input audit receipt")
    registered = keyed(audit["fixture_sha256"].items(), lambda item: Path(item[0]).name)
    fixtures, original, _, fixture_receipts = read_fixtures(fixture_dir, registered["summary.json"][1])
    require({Path(path).name: digest for path, digest in fixture_receipts.items()} ==
            {name: item[1] for name, item in registered.items()}, "raw fixture hashes differ from the old audit")

    def rows(name, key=None):
        data = read_pinned(directory / name, audit["files_sha256"][key or name.removesuffix(".jsonl")], receipts)
        return [json.loads(line) for line in data.splitlines()]

    pairs = keyed(rows("pairs.jsonl"), lambda row: row["pair_id"])
    histories = keyed(rows("histories.jsonl"), lambda row: (row["pair_id"], row["history"]))
    previous = keyed(rows("effective_geometry_hiercad.jsonl", "hiercad"), lambda row: row["pair_id"])
    require(len(pairs) == audit["audited_pairs"] and len(histories) == audit["audited_histories"]
            and histories.keys() == set(product(pairs, MEMBERS)) and pairs.keys() == previous.keys(),
            "immutable audit pair/history/geometry axes differ")
    for pid, pair in pairs.items():
        raw = fixtures[pid]
        require(all(pair[key] == raw[key] for key in ("family", "attribute", "kind"))
                and pair["parent_pair_id"] == raw.get("parent_pair_id")
                and pair["scoring_members"] == (raw["views"][0]["histories"] if raw["kind"] == "easy" else list(MEMBERS))
                and (raw["kind"] != "coordinate" or "hiercad" in pair["coordinate_replay"])
                and previous[pid]["pair_audit_sha256"] == geo.fingerprint(pair),
                f"previous geometry evidence differs from the raw fixture for {pid}")
    for (pid, member), row in histories.items():
        require(row["entity"] == original[pid, member]["entity"]
                and row["raw_program_sha256"] == original[pid, member]["raw_program_sha256"]
                and row["cut_order"] == original[pid, member]["cut_order"],
                f"saved audited program identity differs for {pid}/{member}")
        target = row.get("hiercad_serialized")
        if target is not None:
            require(target["status"] == "ok"
                    and sha256(target["response"].encode()).hexdigest() == target["input_sha256"]
                    and geo.fingerprint(target["replay_program"]) == target["replay_program_sha256"],
                    f"saved HierCAD response or replay program digest differs for {pid}/{member}")
    return audit, pairs, histories, fixtures, {**receipts, **fixture_receipts}


def select_cases(pairs, cap):
    if cap is None:
        return list(pairs)
    first_family_case = {}
    for pid, row in pairs.items():
        first_family_case.setdefault(row["family"], pid)
    first = list(first_family_case.values())
    remainder = [pid for pid in pairs if pid not in first]
    selected = set((first + remainder)[:cap])
    return [pid for pid in pairs if pid in selected]


def restore_target(native, serialization):
    from hiercad.src.utils.hierarchical_parse import merge_part_jsons, parse_hierarchical_response

    response = serialization["response"]
    parsed = parse_hierarchical_response(response)
    if parsed["parse_errors"]:
        raise ValueError("released parser rejected the saved HierCAD target")
    replay = merge_part_jsons(parsed)
    if geo.fingerprint(replay) != serialization["replay_program_sha256"]:
        raise ValueError("released parser differs from the saved replay-program digest")
    # The pinned encoder uses the supplied response directly. Keep the parsed program for provenance.
    return native.HierCadTarget(response, replay, {})


def tokenize_case(native, tokenizer, fixture, pair, histories):
    pid = pair["pair_id"]
    captions = fixture["captions"]
    if pair["kind"] == "easy":
        variant = fixture["views"][0]["variant"]
        captions = {entity: {variant: texts[variant]} for entity, texts in captions.items()}
    input_records = {}
    records = {}
    for member in MEMBERS:
        audited = histories[pid, member]
        serialization = audited.get("hiercad_serialized")
        if serialization is None:
            tokenized = unavailable("saved_hiercad_target", "immutable audit has no serialized target")
        else:
            target, error = attempt("released_parser", restore_target, native, serialization)
            if error:
                tokenized = error
            else:
                tokens, error = attempt("hiercad_tokenize_panel", encode_hiercad_panel,
                                        native, tokenizer, target, captions)
                tokenized = error or tokens
        input_records[member] = tokenized
        records[member] = {"pair_id": pid, "history": member, "entity": member[0],
                           "raw_program_sha256": audited["raw_program_sha256"],
                           "response_sha256": serialization["input_sha256"] if serialization else None,
                           "replay_program_sha256": serialization["replay_program_sha256"] if serialization else None,
                           "tokens": tokenized}
    scoring_replay = pair["scoring_replay"]["hiercad"]
    coordinate = pair["coordinate_replay"].get("hiercad")
    token_statuses = {member: value["status"] for member, value in input_records.items()}
    active_states = sorted({token_statuses[member] for member in pair["scoring_members"]} - {"ok"})
    if active_states:
        failure = "hiercad_tokenizer_" + "+".join(active_states)
    elif scoring_replay["status"] != "verified":
        failure = "immutable_effective_geometry_unverified"
    elif coordinate and (not fixture["coordinate_geometry_verified"] or coordinate["status"] != "verified"):
        failure = "immutable_coordinate_geometry_unverified"
    else:
        failure = None
    actual_conflict = (failure is None and bool(coordinate) and fixture["coordinate_conflict"]
                       and coordinate["effective_coordinate_conflict"])
    result = {"pair_id": pid, "family": pair["family"], "attribute": pair["attribute"], "kind": pair["kind"],
              "immutable_audit_pair_sha256": geo.fingerprint(pair), "scoring_members": pair["scoring_members"],
              "immutable_scoring_replay": scoring_replay, "immutable_coordinate_replay": coordinate,
              "token_statuses": token_statuses, "token_links": link_inputs(input_records),
              "status": "verified" if failure is None else "failed", "failure_reason": failure,
              "effective_coordinate_conflict": actual_conflict}
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-summary-sha256", required=True)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--hiercad-repo", required=True, type=Path)
    parser.add_argument("--hiercad-tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", nargs="?", const=12, type=int, help="Cap at N preordered pairs, default 12")
    args = parser.parse_args()
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke requires a positive case cap")
    started = time.perf_counter()
    audit, pairs, histories, fixtures, receipts = read_audit(args.audit, args.audit_summary_sha256, args.fixtures)
    native, source = activate_hiercad(args.hiercad_repo)
    tokenizer, tokenizer_source = load_hiercad_tokenizer(args.hiercad_tokenizer)
    old_source = audit["assets"]["hiercad_repo"]
    require(old_source["status"] == "ok" and old_source["revision"] == source["revision"],
            "retokenizer source differs from the serialized HierCAD source")
    old_tokenizer = audit["assets"]["hiercad_tokenizer"]
    if old_tokenizer["status"] == "ok":
        require(all(old_tokenizer[key] == value for key, value in tokenizer_source.items()),
                "retokenizer assets differ from the originally pinned tokenizer files")
    # Surface missing release dependencies once before writing any successor evidence.
    from hiercad.src.tokenize_hier_consistency import build_messages
    from hiercad.src.tokenize_parameter_grounding import build_masked_labels

    require(callable(build_messages) and callable(build_masked_labels), "released tokenization hooks are unavailable")
    selected = select_cases(pairs, args.smoke)
    require(bool(selected), "no immutable audit pairs selected")
    args.out.mkdir(parents=True, exist_ok=False)
    token_path = args.out / "hiercad_tokens.jsonl"
    pair_path = args.out / "pairs.jsonl"
    geometry_path = args.out / "effective_geometry_hiercad.jsonl"
    groups = {}
    with token_path.open("xb") as tokens, pair_path.open("xb") as pair_output, geometry_path.open("xb") as geometry:
        for pid in selected:
            pair, records = tokenize_case(native, tokenizer, fixtures[pid], pairs[pid], histories)
            geo.jsonl_write(pair_output, pair)
            for record in records.values():
                geo.jsonl_write(tokens, record)
            geo.jsonl_write(geometry, {
                "pair_id": pid, "status": pair["status"], "failure_reason": pair["failure_reason"],
                "evidence": f"pairs.jsonl/{pid}", "pair_retokenization_sha256": geo.fingerprint(pair),
                "immutable_audit_pair_sha256": pair["immutable_audit_pair_sha256"],
                "effective_coordinate_conflict": pair["effective_coordinate_conflict"],
            })
            key = (pair["family"], pair["attribute"], pair["kind"])
            group = groups.setdefault(key, Counter())
            group["denominator"] += 1
            group[pair["status"]] += 1
            group["effective_coordinate_conflict"] += pair["effective_coordinate_conflict"]
            if pair["failure_reason"]:
                group[pair["failure_reason"]] += 1
            group.update("token:" + status for status in pair["token_statuses"].values())
            for check, evidence in pair["token_links"].items():
                group.update({f"{check}:{status}": count for status, count in evidence["counts"].items()})
    summary = {
        "schema": "idea01-formal-binding-hiercad-retokenization-v1",
        "scope": "CPU retokenization of immutable HierCAD audit targets",
        "mode": "smoke" if args.smoke is not None else audit["mode"],
        "immutable_pairs": len(pairs), "retokenized_pairs": len(selected), "retokenized_histories": len(selected) * len(MEMBERS),
        "verified_pairs": sum(counts["verified"] for counts in groups.values()),
        "input_sha256": receipts, "hiercad_source": source, "hiercad_tokenizer": tokenizer_source,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "files_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                         for path in (token_path, pair_path, geometry_path)},
        "groups": [{"family": family, "attribute": attribute, "kind": kind, **counts}
                   for (family, attribute, kind), counts in groups.items()],
        "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "summary.json").write_bytes(geo.json_bytes(summary) + b"\n")
    print(json.dumps({"retokenized_pairs": len(selected), "verified_pairs": summary["verified_pairs"],
                      "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
