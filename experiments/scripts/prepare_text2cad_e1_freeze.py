"""Select E1 galleries and freeze the complete score input bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from idea01.text2cad_e1_scoring import (
    STAGES, get_arms, read_jsonl, sha256_file, sha256_text, write_json, write_npz,
)
from idea01.text2cad_e1_selection import DOMAINS, audit_and_select, iter_rows, write_rows


PLAN_SHA = "3613921e649e226628f8c6b116775eeb9e29af04"
G4_MATRIX_SHA = "9a379ed75b28fb575c2cabd25c9f1e1e6368a9a0cb0ea0c1e564ec8b293710cf"
G4_SIDECAR_SHA = "9479895007c3b45742ccb39884d321be85d4856cbc2ec123d6ddd07e8e03b004"
G4_CERT_SHA = "afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6"


def rectangular(loop) -> bool:
    if len(loop) != 4 or not all(key.startswith("line_") for key in loop):
        return False
    neighbors = defaultdict(list)
    for segment in loop.values():
        a = tuple(segment["Start Point"])
        b = tuple(segment["End Point"])
        if len(a) != 2 or len(b) != 2 or a == b:
            return False
        neighbors[a].append(b)
        neighbors[b].append(a)
    if len(neighbors) != 4 or any(len(set(v)) != 2 for v in neighbors.values()):
        return False
    for (x, y), ((x1, y1), (x2, y2)) in neighbors.items():
        ax, ay, bx, by = x1 - x, y1 - y, x2 - x, y2 - y
        if abs(ax * bx + ay * by) > 0.03 * math.hypot(ax, ay) * math.hypot(bx, by):
            return False
    return True


def features(minimal: dict) -> dict:
    circle, rectangle, complex_shape = False, False, False
    numbers = set()

    def add(attribute, value, space):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            numbers.add((attribute, float(value), space))

    parts = minimal["parts"]
    add("extrusion", len(parts), "count")
    cuts, circles = 0, 0
    for part in parts.values():
        extrusion = part["extrusion"]
        for k in ("extrude_depth_towards_normal", "extrude_depth_opposite_normal"):
            value = extrusion.get(k)
            if value:
                add("depth", abs(value), "normalized")
        is_cut = extrusion["operation"] == "CutFeatureOperation"
        cuts += int(is_cut)
        complex_shape |= is_cut
        for face in part["sketch"].values():
            complex_shape |= len(face) > 1
            for loop in face.values():
                rectangle |= rectangular(loop)
                circle |= len(loop) == 1 and next(iter(loop)).startswith("circle_")
                for k, primitive in loop.items():
                    if k.startswith("circle_"):
                        circles += 1
                        radius = primitive.get("Radius")
                        if radius is not None:
                            add("radius", radius, "normalized")
                            add("diameter", 2 * radius, "normalized")
    add("cut", cuts, "count")
    add("circle", circles, "count")
    return {
        "has_single_circle": circle, "has_near_rectangle": rectangle,
        "has_multiloop_or_cut": complex_shape,
        "numeric_values": [
            {"attribute": k, "value": value, "value_space": space, "unit": ""}
            for k, value, space in sorted(numbers)
        ],
    }


def project_assets(source: Path, out: Path, row: dict) -> dict:
    fields = (
        "uid", "domain", "group_id", "project", "text", "token_count", "l0_text",
        "token_ids_sha256", "l0_token_count", "n_valid", "n_extrusions", "official_member",
        "cad_vec_sha256", "model_input_sha256", "source_train_near_status",
        "l0_text_train_match_count",
    )
    kept = {k: row[k] for k in fields if k in row}
    vector = row["vector_relpath"]
    vp = out / vector
    vp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / vector, vp)
    kept["vector_relpath"] = vector
    if row["domain"] == "fusion360" and row.get("reference_relpath"):
        asset = row["reference_relpath"]
        sp = out / asset
        sp.parent.mkdir(parents=True, exist_ok=True)
        if not sp.exists():
            shutil.copyfile(source / asset, sp)
        kept["geometry_relpath"] = asset
    return kept


def main_query(row: dict, a_l0: bool = False) -> dict:
    text = row["l0_text" if a_l0 else "text"]
    count = row["l0_token_count" if a_l0 else "token_count"]
    q = {
        "uid": row["uid"], "group_id": row["group_id"], "text": text,
        "token_count": count, "text_sha256": sha256_text(text),
    }
    if not a_l0 and row.get("token_ids_sha256"):
        q["token_ids_sha256"] = row["token_ids_sha256"]
    return q


def g4_inputs(data: Path, runtime: Path):
    g4 = data / "experiments/runs/idea01_g5_sum_replay_20261115"
    axis = runtime / "pair_manifest_v4"
    sidecar = g4 / "sum_scores_sidecar.json"
    matrix = g4 / "sum_scores_full_axis_f64.npy"
    if sha256_file(sidecar) != G4_SIDECAR_SHA or sha256_file(matrix) != G4_MATRIX_SHA:
        raise ValueError("historical G4 replay fingerprint differs")
    pins = json.loads((ROOT / "experiments/configs/text2cad_pair_manifest_expected_20260930.json").read_bytes())
    for name, pinned in pins["artifact_sha256"].items():
        if sha256_file(axis / name) != pinned:
            raise ValueError(f"historical G4 axis drift: {name}")
    side = json.loads(sidecar.read_bytes())
    if (
        side["source_certificate_sha256"] != G4_CERT_SHA
        or side["text_indices"] != list(range(3644))
        or side["cad_indices"] != list(range(8046))
        or matrix.stat().st_size != 3644 * 8046 * 8
    ):
        raise ValueError("historical G4 axes or certificate mismatch")
    mem = np.memmap(matrix, dtype="<f8", mode="r", shape=(3644, 8046))
    return (
        mem, read_jsonl(axis / "texts.jsonl"),
        read_jsonl(axis / "cad_store.jsonl"), read_jsonl(axis / "cad_length_index.jsonl"),
    )


def make_replay(out: Path, g4):
    matrix, texts, members, lengths = g4
    queries = [
        {"uid": f"g4_text_{i:04d}", "text": row["text"], "text_sha256": row["text_sha256"]}
        for i, row in enumerate(texts[:16])
    ]
    gallery = [
        {"uid": row["member"].split("/seq/")[0].removeprefix("cad_seq/"),
         "official_member": row["member"]}
        for row in members[:16]
    ]
    write_rows(out / "queries_replay.jsonl", queries)
    write_rows(out / "gallery_replay.jsonl", gallery)
    write_npz(
        out / "replay_reference.npz",
        query_uids=np.array([row["uid"] for row in queries]),
        cad_uids=np.array([row["uid"] for row in gallery]),
        scores=matrix[:16, :16].copy(),
        counts=np.array([row["n_valid_x"] + row["n_valid_y"] for row in lengths[:16]], dtype=np.int64),
    )


def prepare_rewrites(rewrite_file, diagnostic, c_index):
    expected = {(uid, v) for uid in diagnostic for v in ("Qgeom", "Qvalue")}
    result = read_jsonl(rewrite_file)
    seen = set()
    for row in result:
        key = (row["uid"], row["variant"])
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicated rewrite key")
        seen.add(key)
        if row["status"] not in ("changed_valid", "no_edit", "unverifiable", "overlength"):
            raise ValueError("unrecognized rewrite status")
        original = c_index[row["uid"]]
        if row.get("original_text_sha256") != original["text_sha256"]:
            raise ValueError("rewrite did not bind the frozen query")
        if not row.get("review_evidence"):
            raise ValueError("every rewrite status requires the independent review record")
        if row["status"] == "changed_valid":
            if not row["text"].strip() or row["text"] == original["text"]:
                raise ValueError("changed rewrite requires text and independent review evidence")
        if row["status"] == "no_edit" and row.get("text") != original["text"]:
            raise ValueError("no_edit must preserve the original query exactly")
    if seen != expected:
        raise ValueError(f"{len(expected - seen)} rewrite reviews still pending before freeze")
    return result


def baseline_reference(prep: Path) -> dict:
    from idea01.text2cad_e1_eval import candidate_numeric_values, numeric_rules, query_numeric_values

    geo = {r["uid"]: r for r in iter_rows(prep / "geometry_development.jsonl")}
    counts, tokens, operations = [], [], []
    numeric = Counter()
    rules = numeric_rules()
    for row in iter_rows(prep / "development.jsonl"):
        trial = geo[row["uid"]]
        if trial.get("geometry_pass") and trial.get("n_valid") and trial.get("n_extrusions"):
            tokens.append(row["token_count"])
            counts.append(trial["n_valid"])
            operations.append(trial["n_extrusions"])
            seen_text = query_numeric_values(row["text"], rules)
            observations = features(row["minimal"])["numeric_values"]
            seen_geometry = candidate_numeric_values(observations, rules)
            for space in seen_text:
                if seen_text[space] and seen_geometry[space]:
                    numeric[f"{space}_comparable"] += 1
                    numeric[f"{space}_with_overlap"] += int(bool(seen_text[space] & seen_geometry[space]))
    if not counts:
        raise ValueError("validated source reference distributions are missing")
    return {
        "token_counts": tokens, "n_valid": counts, "n_extrusions": operations,
        "numeric_rules": rules, "source_numeric_coverage": dict(numeric),
    }


def encoding_differences(prep: Path, minimal: dict, original: dict) -> dict:
    result = {}
    with np.load(prep / minimal["vector_relpath"], allow_pickle=False) as left, np.load(
        prep / original["vector_relpath"], allow_pickle=False,
    ) as right:
        for key in ("cad_vec", "flag_vec", "index_vec", "key_padding_mask"):
            a, b = left[key], right[key]
            different = np.flatnonzero(a.reshape(-1) != b.reshape(-1))
            result[key] = {
                "minimal_sha256": sha256_bytes(a.tobytes()), "original_sha256": sha256_bytes(b.tobytes()),
                "different_fields": int(len(different)),
                "first_flat_index": int(different[0]) if len(different) else None,
            }
    result["attention_mask"] = "shared frozen full-length causal and START mask"
    return result


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def package(args) -> dict:
    prep, selection_dir, out = args.prep, args.selection, args.out
    selection = json.loads((selection_dir / "selection.json").read_bytes())
    n = selection["gallery_size"]
    if n not in (256, 512):
        raise ValueError("no admissible E1 gallery size")
    out.mkdir()
    kept = {}
    selected = {
        domain: read_jsonl(selection_dir / f"gallery_{domain}.jsonl")
        for domain in DOMAINS
    }
    wanted = {(domain, uid) for domain in DOMAINS for uid in selection["gallery"][domain]}
    raw_features = {}
    for row in iter_rows(prep / "inventory_seed.jsonl"):
        key = (row["domain"], row["uid"])
        if key in wanted:
            raw_features[key] = features(row["minimal"])
    for domain in DOMAINS:
        items = []
        for row in selected[domain]:
            item = project_assets(prep, out, row)
            item["program_features"] = raw_features[(domain, row["uid"])]
            if domain == "fusion360" and row.get("reference_frame"):
                scale = row["reference_frame"]["step_mm_to_normalized"][0][0]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("invalid independent STEP scale")
                numeric = item["program_features"]["numeric_values"]
                physical = [
                    {**value, "value": value["value"] / scale, "value_space": "physical", "unit": "mm"}
                    for value in numeric
                    if value["value_space"] == "normalized" and value["attribute"] in ("depth", "radius", "diameter")
                ]
                item["program_features"] = {**item["program_features"], "numeric_values": numeric + physical}
            items.append(item)
        kept[domain] = items
        write_rows(out / f"gallery_{domain}.jsonl", items)

    b, c = [main_query(r) for r in kept["deepcad"]], [main_query(r) for r in kept["fusion360"]]
    a = [
        main_query(row, True) for row in kept["deepcad"]
        if row.get("l0_text") and 0 < row.get("l0_token_count", 0) <= 512
    ]
    from tokenizers import Tokenizer

    l0_tokenizer_path = args.data / ".cache/bert_large_uncased/tokenizer.json"
    if sha256_file(l0_tokenizer_path) != "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98":
        raise ValueError("source tokenizer fingerprint drift before freezing L0 query IDs")
    tokenizer = Tokenizer.from_file(str(l0_tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    for row in a:
        ids = tokenizer.encode(row["text"]).ids
        if len(ids) != row["token_count"]:
            raise ValueError("L0 tokenizer count drift")
        row["token_ids_sha256"] = sha256_text(json.dumps(ids, separators=(",", ":")))
    b_enc_ids = set(selection["deepcad_encoding_uids"])
    for name, rows in (("a", a), ("b", b), ("c", c),
                       ("b_official", [r for r in b if r["uid"] in b_enc_ids])):
        write_rows(out / f"queries_{name}.jsonl", rows)
    main_c = {r["uid"]: r for r in kept["fusion360"]}
    alternates = read_jsonl(args.alternate)
    if (
        len(alternates) != 32 or {r["uid"] for r in alternates} != set(selection["fusion_encoding_uids"])
    ):
        raise ValueError("Fusion double encoding audit must contain all frozen 32 UIDs")
    alt_index = {r["uid"]: r for r in alternates}
    source_train = {r["sha256"] for r in iter_rows(prep / "train_vec_hashes.jsonl")}
    replacement, skipped, fusion_comparison = [], {}, []
    for uid, row in main_c.items():
        if uid not in alt_index:
            continue
        candidate = alt_index[uid]
        source_match = candidate.get("cad_vec_sha256") in source_train
        evidence = {
            "uid": uid, "minimal_geometry_pass": True,
            "minimal_model_input_sha256": row["model_input_sha256"],
            "minimal_cad_vec_sha256": row["cad_vec_sha256"],
            "minimal_n_valid": row["n_valid"],
            "alternate": candidate, "alternate_exact_source_train": source_match,
        }
        if candidate.get("vector_relpath"):
            evidence["array_diff"] = encoding_differences(prep, row, candidate)
        fusion_comparison.append(evidence)
        if candidate["geometry_pass"] is not True:
            skipped[uid] = "alternate_geometry_failed"
        elif candidate["model_input_sha256"] == row["model_input_sha256"]:
            skipped[uid] = "model_inputs_identical"
        else:
            replacement.append(project_assets(
                prep, out, {**row, **candidate, "domain": "fusion360", "group_id": row["group_id"]},
            ))
    if replacement:
        write_rows(out / "gallery_fusion_original.jsonl", replacement)
        write_rows(out / "queries_c_alt.jsonl", c)

    by_query = {r["uid"]: r for r in c}
    rewrites = prepare_rewrites(args.rewrites, selection["diagnostic_uids"], by_query)
    for row in rewrites:
        row["group_id"] = by_query[row["uid"]]["group_id"]
    write_rows(out / "queries_c_rewrite.jsonl", rewrites)

    short = min(c, key=lambda r: (r["token_count"], r["uid"]))
    long = max(c, key=lambda r: (r["token_count"], r["uid"]))
    if short["uid"] == long["uid"]:
        long = c[-1] if c[-1]["uid"] != short["uid"] else c[0]
    write_rows(out / "queries_probe.jsonl", [short, long])
    write_rows(out / "gallery_probe.jsonl", kept["fusion360"][:16])
    make_replay(out, g4_inputs(args.data, args.runtime))
    shutil.copyfile(prep / "background.jsonl", out / "background.jsonl")
    write_rows(out / "fusion_encoding.jsonl", fusion_comparison)
    write_json(out / "baseline_reference.json", baseline_reference(prep))
    write_json(out / "project_isolation.json", {
        "projects": selection["fusion_project_isolation"],
        "groups_sha256": sha256_file(selection_dir / "groups.json"),
    })

    arm_names = ["replay", "probe", "b", "b_official", "c"]
    if a:
        arm_names.append("a")
    if replacement:
        arm_names.append("c_alt")
    if any(r["status"] in ("changed_valid", "no_edit") for r in rewrites):
        arm_names.append("c_rewrite")
    freeze = {
        "plan_sha": PLAN_SHA, "gallery_size": n, "scored_arms": arm_names,
        "selection_sha256": sha256_file(selection_dir / "selection.json"),
        "geometry_freeze_sha256": sha256_file(prep / "geometry_freeze.json"),
        "diagnostic_uids": selection["diagnostic_uids"],
        "replacement_uids": [r["uid"] for r in replacement],
        "replacement_skipped": skipped, "coverage": selection["coverage"],
        "score_inputs_sha256": {
            file.relative_to(out).as_posix(): sha256_file(file)
            for file in sorted(out.rglob("*")) if file.is_file()
        },
    }
    freeze["planned_pairs"] = sum(
        arm.pairs for stage in STAGES for arm in get_arms(out, freeze, stage)
    )
    write_json(out / "freeze.json", freeze)
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    select = modes.add_parser("select")
    select.add_argument("--prep", type=Path, required=True)
    select.add_argument("--geometry", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    pack = modes.add_parser("package")
    for flag in ("prep", "selection", "alternate", "data", "runtime", "out"):
        pack.add_argument(f"--{flag}", type=Path, required=True)
    pack.add_argument("--rewrites", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "select":
        selection = audit_and_select(args.prep, args.geometry, args.out)
        print(json.dumps({"gallery_size": selection["gallery_size"], "coverage": selection["coverage"]}))
        return 0 if selection["gallery_size"] else 2
    print(json.dumps({"planned_pairs": package(args)["planned_pairs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
