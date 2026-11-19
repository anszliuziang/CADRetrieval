"""Deterministic design-family selection from audited E1a inventory."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from idea01.text2cad_e1_scoring import read_jsonl, sha256_file, write_json


SEED = "cad-domain-v1"
DOMAINS = ("deepcad", "fusion360")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_key(purpose: str, domain: str, identifier: str) -> str:
    return digest(f"{SEED}|{purpose}|{domain}|{identifier}")


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_rows(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def candidate_groups(domain: str, rows: list[dict]) -> dict[str, list[str]]:
    parent = {row["uid"]: row["uid"] for row in rows}
    if len(parent) != len(rows):
        raise ValueError("duplicate candidate UIDs")

    def find(uid: str) -> str:
        while uid != parent[uid]:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    linked = {}
    for row in rows:
        uid = row["uid"]
        keys = [
            ("program", row["program_sha256"]),
            ("quantized", row["cad_vec_sha256"]),
            ("near_family", row.get("confirmed_family_id")),
            ("geometry_exact", row.get("exact_geometry_sha256")),
        ]
        if domain == "fusion360":
            keys.append(("autodesk_project", row["project"]))
        else:
            keys.append(("verified_design_source", row.get("verified_design_source")))
            keys.append(("quantized", row.get("official_cad_vec_sha256")))
        for kind, value in keys:
            if value is not None:
                previous = linked.setdefault((kind, value), uid)
                parent[find(uid)] = find(previous)
    components = defaultdict(list)
    for uid in sorted(parent):
        components[find(uid)].append(uid)
    return {digest(domain + "\n" + "\n".join(uids)): uids for uids in components.values()}


def group_splits(domain: str, groups: dict[str, list[str]]) -> dict:
    held_out, available = [], []
    order = sorted(groups, key=lambda k: (sample_key("select", domain, k), k))
    for key in order:
        members = groups[key]
        entry = {
            "group_id": key,
            "member_uids": members,
            "uid": min(members, key=lambda uid: (sample_key("member", domain, uid), uid)),
        }
        if int(sample_key("reserve", domain, key), 16) % 4 == 0:
            held_out.append(entry)
        else:
            available.append(entry)
    return {"eligible": available, "reserve": held_out}


def choose_size(source_groups: int, target_groups: int) -> int:
    common = min(source_groups, target_groups)
    if common >= 512:
        return 512
    if common >= 256:
        return 256
    raise ValueError(f"fewer than 256 admitted groups: B={source_groups}, C={target_groups}")


def choose_diagnostic(gallery, domain, purpose, count):
    return [
        row["uid"] for row in sorted(
            gallery, key=lambda row: (sample_key(purpose, domain, row["group_id"]), row["group_id"]),
        )[:count]
    ]


def publish_inventory(out: Path, rows: list[dict], coverage: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = (
        "domain", "uid", "project", "published_split", "text_sha256", "token_ids_sha256",
        "program_sha256", "token_count", "n_valid", "n_extrusions", "geometry_pass",
        "cad_vec_sha256", "official_cad_vec_sha256", "model_input_sha256", "group_id",
        "source_train_near_status", "first_exclusion", "all_reasons",
    )
    parquet_rows = [{key: row.get(key) for key in fields} for row in rows]
    pq.write_table(pa.Table.from_pylist(parquet_rows), out / "inventory.parquet")
    with (out / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("domain", "checkpoint_or_first_exclusion", "count"))
        for domain, counters in coverage.items():
            writer.writerows((domain, reason, count) for reason, count in sorted(counters.items()))

    fields = (
        "domain", "uid", "geometry_pass", "reason", "reference_kind", "reference_member",
        "reference_file_sha256", "vector_relpath", "reference_relpath",
        "rel_volume", "candidate_to_ref_p99", "ref_to_candidate_p99",
        "material_disagreements",
    )
    with (out / "geometry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row.get("geometry_pass") is None:
                continue
            evidence = {**row, **(row.get("geometry") or {})}
            writer.writerow({key: evidence.get(key) for key in fields})


def audit_and_select(prep: Path, geometry_path: Path, out: Path) -> dict:
    gate = json.loads((prep / "geometry_freeze.json").read_bytes())
    if gate.get("development_accepted") is not True or gate.get("n") != 64:
        raise ValueError("source validation geometry has not passed its frozen 64-case gate")
    corroboration_path = prep / "full_text_evidence_compact" / "near_train_corroboration.jsonl"
    corroboration = {
        (row["domain"], row["uid"]): row for row in iter_rows(corroboration_path)
    }
    out.mkdir()
    geometry = {}
    for row in iter_rows(geometry_path):
        key = (row["domain"], row["uid"])
        if key in geometry:
            raise ValueError(f"duplicate geometry witness {key}")
        geometry[key] = row

    source_train = {row["sha256"] for row in iter_rows(prep / "train_vec_hashes.jsonl")}
    source_official = {row["uid"]: row["sha256"] for row in iter_rows(prep / "test_vec_hashes.jsonl")}
    if len(source_official) != 8046:
        raise ValueError("full official test vector coverage is required")
    counters = {domain: Counter() for domain in DOMAINS}
    accepted = {domain: [] for domain in DOMAINS}
    cross_programs, cross_vectors, cross_geometry = set(), set(), set()
    inventory_rows = []
    seed = prep / "full_text_evidence_compact" / "inventory_seed.jsonl"
    if not seed.is_file():
        seed = prep / "inventory_seed.jsonl"
    for row in iter_rows(seed):
        domain = row["domain"]
        if domain not in accepted:
            raise ValueError(f"unexpected domain {domain}")
        key = (domain, row["uid"])
        counters[domain]["published"] += 1
        witness = geometry.pop(key, {})
        if not witness and row["first_exclusion"] is None:
            raise ValueError(f"eligible row without a completed geometry-processing record: {key}")
        row.pop("minimal")
        all_reasons = list(row["all_reasons"])
        first = row["first_exclusion"]
        combined = {**row, **witness}
        near = corroboration.get(key)
        if near is None and first is None:
            raise ValueError(f"eligible row without source shortlist corroboration: {key}")
        combined["source_train_near_status"] = near["status"] if near is not None else "not_screened"
        cad_hash = witness.get("cad_vec_sha256")
        official_hash = source_official.get(row["uid"]) if domain == "deepcad" else None
        combined["official_cad_vec_sha256"] = official_hash
        reason = witness.get("reason")
        if (cad_hash is not None and cad_hash in source_train) or (
            official_hash is not None and official_hash in source_train
        ):
            reason = "source_train_quantized_exact"
        elif witness and witness.get("geometry_pass") is not True:
            reason = reason or "geometry_rejected"
        elif near is not None and near["status"] == "confirmed":
            reason = "confirmed_source_train_copy"
        if reason:
            if reason not in all_reasons:
                all_reasons.append(reason)
            first = first or reason
        if first is None:
            for field in ("cad_vec_sha256", "vector_relpath", "n_valid", "n_extrusions"):
                if not combined.get(field):
                    raise ValueError(f"admitted row lacks {field}: {key}")
            accepted[domain].append(combined)
        combined.update(all_reasons=all_reasons, first_exclusion=first)
        inventory_rows.append(combined)
    if geometry:
        raise ValueError("geometry contains unexpected UIDs")
    for row in accepted["deepcad"]:
        cross_programs.add(row["program_sha256"])
        cross_vectors.add(row["cad_vec_sha256"])
        if row["official_cad_vec_sha256"]:
            cross_vectors.add(row["official_cad_vec_sha256"])
        if row.get("exact_geometry_sha256"):
            cross_geometry.add(row["exact_geometry_sha256"])
    only_target = []
    for row in accepted["fusion360"]:
        if (
            row["program_sha256"] in cross_programs or row["cad_vec_sha256"] in cross_vectors
            or row.get("exact_geometry_sha256") in cross_geometry
        ):
            row["first_exclusion"] = "known_exact_source_test_overlap"
            row["all_reasons"].append(row["first_exclusion"])
        else:
            only_target.append(row)
    accepted["fusion360"] = only_target

    groups = {
        domain: group_splits(domain, candidate_groups(domain, accepted[domain]))
        for domain in DOMAINS
    }
    groups_available = {domain: len(groups[domain]["eligible"]) for domain in DOMAINS}
    size = None
    coverage_error = None
    try:
        size = choose_size(*groups_available.values())
    except ValueError as exc:
        coverage_error = str(exc)

    lookup = {(row["domain"], row["uid"]): row for row in inventory_rows}
    gallery, project_isolation = {}, set()
    for domain in DOMAINS:
        gallery[domain] = []
        selected_groups = groups[domain]["eligible"][:size] if size else []
        chosen = {g["uid"] for g in selected_groups}
        for group in selected_groups:
            selected = lookup[(domain, group["uid"])]
            selected["group_id"] = group["group_id"]
            gallery[domain].append(selected)
        reserved = {uid for g in groups[domain]["reserve"] for uid in g["member_uids"]}
        for row in accepted[domain]:
            if row["uid"] not in chosen:
                reason = "reserved_family" if row["uid"] in reserved else "group_sampling"
                row["first_exclusion"] = reason
                row["all_reasons"].append(reason)
        if domain == "fusion360":
            group_uids = {
                uid for g in selected_groups + groups[domain]["reserve"] for uid in g["member_uids"]
            }
            project_isolation.update(lookup[(domain, uid)]["project"] for uid in group_uids)

    for row in inventory_rows:
        counters[row["domain"]][row["first_exclusion"] or "selected"] += 1
    for domain in DOMAINS:
        counters[domain]["geometry_and_exposure_admitted"] = len(accepted[domain])
        counters[domain]["eligible_groups"] = groups_available[domain]
        counters[domain]["reserved_groups"] = len(groups[domain]["reserve"])
        write_rows(out / f"gallery_{domain}.jsonl", gallery[domain])
    selection = {
        "gallery_size": size, "coverage_error": coverage_error, "seed": SEED,
        "gallery": {domain: [r["uid"] for r in gallery[domain]] for domain in DOMAINS},
        "diagnostic_uids": choose_diagnostic(gallery["fusion360"], "fusion360", "diagnostic", 48),
        "fusion_encoding_uids": choose_diagnostic(
            gallery["fusion360"], "fusion360", "fusion-encoding-check", 32,
        ),
        "deepcad_encoding_uids": choose_diagnostic(
            gallery["deepcad"], "deepcad", "encoding-check", 256,
        ),
        "fusion_project_isolation": sorted(project_isolation),
        "geometry_sha256": sha256_file(geometry_path),
        "inventory_seed_sha256": sha256_file(seed),
        "near_corroboration_sha256": sha256_file(corroboration_path),
        "coverage": {domain: dict(counters[domain]) for domain in DOMAINS},
    }
    write_rows(out / "inventory.jsonl", inventory_rows)
    write_json(out / "groups.json", groups)
    publish_inventory(out, inventory_rows, selection["coverage"])
    write_json(out / "selection.json", selection)
    return selection
