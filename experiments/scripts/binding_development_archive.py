"""Adapt the original development artifacts to the Table 4 evaluator."""

from collections import Counter
from itertools import product
import json
import re

from experiments.scripts.audit_910a_dev_recognition import HISTORY_SHA, PAIR_SHA, verify_native_identity
from experiments.scripts.evaluate_formal_binding import (
    ATTRIBUTES, CELLS, DEVELOPMENT_FAMILIES, GRID, VARIANTS, collision_view, digest, evaluate_view,
    json_lines, keyed, load_view_axes, pair_result, read_pinned, require, view_key,
)


PILOT = "idea01_in_domain_pilot_20261123_control/final_verified_evidence"
QUALIFICATION = "idea01_yunnao_910a_qualification_20261123_control"
FIXTURE = f"{PILOT}/geometry_40_pairs_occframes"
TEXT2CAD = f"{QUALIFICATION}/native_and_training_evidence/development_finalizer/resumed_preserve"
HIERCAD = f"{QUALIFICATION}/idea01_geometry_dev_hiercad_20261123_evidence/result"
H_COARSE = f"{QUALIFICATION}/idea01_geometry_dev_hiercad_coarse_20261123_evidence/result"
DUAL = f"{QUALIFICATION}/runtime_and_dual_evidence/command_dual_recognition/result"
EXTERNAL = "idea01_dev_geometry_external_20261124"
TOP = "idea01_dev_geometry_clip_top_20261124"
EXTERNAL_INPUT_SHA = "ca16a64f84cd4af8d9c905f3ce77f425ebc3a092c365ad321233a4c84249462f"
EXTERNAL_AXIS_SHA = "e6453105d4cd9621e700380c3b88e3da703b8e2cf6f0a2e527e153ee592fc872"
DUAL_TOKEN_SHA = "69d9710e548d4918b6e7fab8541f7c0a8e6a375d6a48f1e66022df1391d356ef"

EARLY_SCORES = {
    "text2cad_release": (f"{TEXT2CAD}/paired_views.jsonl",
                         "467da185df190261a1daccd795be4531d8750085b92a2f2642797a9a856dd65a"),
    "hiercad_release": (f"{HIERCAD}/views.jsonl",
                        "ca6f734c27e799acbf1cb6fd19e75936752b783ab87734fa50e68fb4a4db700a"),
    "dual_t": (f"{DUAL}/dual_t/views.jsonl", "d3409996549ccfd0821cc02a3ab131eed5ea4a6858a3903b344fa599ec9f99bd"),
    "dual_ct": (f"{DUAL}/dual_ct/views.jsonl", "b81008c6a1b54ed2db66dea0db36d04c8f504251d5fc17ab0b560be370a7fa90"),
}
EXTERNAL_SCORES = {
    "mmcad": (EXTERNAL, "44a03830fc72e38261f1d1fabe22c946601022f08e4117608f8ad384f59b097c",
              "dec2b4b2bfa4ac1e085d54a74ae0dcfb7c8a026bf058fb041eb13ee9405b47ae", ".npz"),
    "openshape": (EXTERNAL, "ea90cba700d9dace63ecde939f75ed474b2577cbe302bdeb872802a1865fb017",
                  EXTERNAL_AXIS_SHA, ".npy"),
    "clip": (EXTERNAL, "03b6835270862a038f65f7c3cb3b22b7f4371f769d3503e75b2fa46e3c4ff2f5", EXTERNAL_AXIS_SHA, ".png"),
    "clip_top": (TOP, "4f9d205419cf51741597773e1c60c28afdbac9b4cc99ecb0de4023c82ab6f302", EXTERNAL_AXIS_SHA, None),
}
PROVENANCE = {
    "text2cad_release": ("Text2CAD S0", "released development checkpoint", "ordered", "archived native scores"),
    "hiercad_release": ("HierCAD S0", "released development checkpoint", "ordered", "archived native scores"),
    "dual_t": ("Command dual tower", "Text2CAD four levels, pilot", "ordered", "archived cosine scores"),
    "dual_ct": ("Command dual tower", "CadTranslator short text, pilot", "ordered", "archived cosine scores"),
    "mmcad": ("MM-CAD v2", "external released checkpoint", "finished", "archived cosine scores"),
    "openshape": ("OpenShape", "external released checkpoint", "finished", "archived cosine scores"),
    "clip": ("CLIP ViT-B/32 isometric", "external released checkpoint", "finished", "archived cosine scores"),
    "clip_top": ("CLIP ViT-B/32 top view", "external released checkpoint", "finished", "archived cosine scores"),
    "first_cut_slot": ("First attribute and first cut", "rule", "rule", "archived rule scores"),
    "local_sketch_distance": ("Local sketch distance", "rule", "rule", "archived rule scores"),
    "bow_collision": ("Word bag input collision", "deterministic unigram feature proof", "unordered", "proof"),
    "spectra_collision": ("SpectraBRep input collision", "released CAD descriptor proof", "unordered", "proof"),
}
# Published counts in the development report. These are checked after reading
# the source artifacts, and serve only as assertions on the replay.
PUBLISHED = {
    "text2cad_release": (20, 0, 162, 0), "hiercad_release": (122, 76, 168, 2),
    "dual_t": (0, 0, 163, 0), "dual_ct": (3, 7, 155, 0),
    "mmcad": (8, 2, 202, 0), "openshape": (14, 2, 142, 0),
    "clip": (2, 2, 142, 0), "clip_top": (14, 2, 142, 0),
    "first_cut_slot": (160, 160, 160, 0), "local_sketch_distance": (0, 0, 0, 0),
    "bow_collision": (0, 0, 0, 0), "spectra_collision": (0, 0, 0, 0),
}


def archived_view(scores, saved):
    result = evaluate_view(scores)
    require(result["cad_margins"] == saved["cad_choose_caption"]
            and result["caption_margins"] == saved["caption_choose_cad"] and result["D"] == saved["D"]
            and result["cad_ties"] + result["caption_ties"] == saved["ties"]
            and (result["cad_choice"] == "correct") == saved["both_cads"]
            and (result["cad_choice"] == "swapped") == saved["swapped_both_cads"]
            and (result["caption_choice"] == "correct") == saved["both_captions"],
            "A recorded view verdict disagrees with its four original scores")
    return result


def replay_fixture(read):
    original = keyed(json_lines(read(f"{FIXTURE}/geometry_pairs.jsonl", PAIR_SHA)), lambda row: row["pair_id"])
    h = keyed(json_lines(read(f"{FIXTURE}/geometry_histories.jsonl", HISTORY_SHA)),
              lambda row: (row["pair_id"], row["history"]))
    require(len(original) == 40 and set(h) == set(product(original, ("A0", "A1", "B0", "B1"))),
            "Development fixture shape changed")
    pairs = {}
    for pid, pair in original.items():
        family, attribute = (pair["design"][key] for key in ("family", "attribute"))
        require(pair["geometry"]["raw"]["status"] == "verified", f"Unverified held-back geometry at {pid}")
        views = []
        for order, variant in GRID:
            members = [f"{entity}{order}" for entity in "AB"]
            views.append({
                "order": order, "variant": variant, "texts": [pair["captions"][e][variant] for e in "AB"],
                "histories": members, "geometry_sha256": [h[pid, m]["raw_program_sha256"] for m in members],
                "first_cut": [h[pid, m]["cut_order"][0] for m in members],
            })
        pairs[pid] = {**pair, "family": family, "attribute": attribute, "views": views, "kind": "binding"}
    require(Counter((pair["family"], pair["attribute"]) for pair in pairs.values()) ==
            {key: 10 for key in product(DEVELOPMENT_FAMILIES, ATTRIBUTES)}, "Development strata changed")
    return pairs, h, load_view_axes(list(pairs.values()), h)


def raw_grid(read, path, pin, axes):
    grid = keyed(json_lines(read(path, pin)), view_key)
    require(grid.keys() == axes.keys(), f"Frozen identity or view coverage changed in {path}")
    return grid


def native_cell_evidence(cells):
    fields = ("input_sha256", "target_sha256", "target_token_sha256", "effective_units",
              "query_tokens", "score_x", "score_y", "n_valid_x", "n_valid_y", "logit_dtype")
    return [{key: cell[key] for key in fields if key in cell} for cell in cells]


def original_collision_proofs(row, pair, histories, view):
    pid = pair["pair_id"]
    tokens = [Counter(re.findall(r"[a-z0-9]+", text.lower())) for text in view["texts"]]
    require(row["bow_identical_caption_input"] is True and tokens[0] == tokens[1],
            f"Word bag equality cannot be proven for {pid}/{view['variant']}")
    bow = collision_view("cad", {"identical_unigram_counts": dict(sorted(tokens[0].items()))})
    cad = [histories[pid, name]["inputs"]["spectrabrep"] for name in view["histories"]]
    comparison, = [item for item in pair["models"]["spectrabrep"]["opposite_entities"]["comparisons"]
                   if item["histories"] == view["histories"]]
    hashes = [item["input_sha256"] for item in cad]
    require(row["spectra_identical_cad_input"] is True and comparison["exact_value_equality"] is True
            and comparison["max_abs_difference"] == 0 and cad[0]["effective_input"] == cad[1]["effective_input"]
            and comparison["input_sha256"] == hashes and hashes[0] == hashes[1]
            and all(item["status"] == "ok" and item["truncated"] is False for item in cad),
            f"SpectraBRep descriptor equality cannot be proven for {pid}/{view['histories']}")
    return bow, collision_view("caption", {"identical_cad_feature_sha256": hashes[0]})


def read_early(read, axes, pairs, histories, token_rows):
    grids = {method: {} for method in (*EARLY_SCORES, "first_cut_slot", "local_sketch_distance",
                                       "bow_collision", "spectra_collision")}
    for method, (path, pin) in EARLY_SCORES.items():
        rows = raw_grid(read, path, pin, axes)
        for key, view in axes.items():
            pid = key[0]
            row, pair = rows[key], pairs[pid]
            if method.startswith("dual_"):
                result = archived_view(row["scores_aa_ab_ba_bb"], row)
                result["effective_inputs"] = {
                    "cad_sha256": [histories[pid, name]["inputs"]["command_dual_tower"]["input_sha256"]
                                   for name in view["histories"]],
                    "text_token_sha256": token_rows.get((pid, view["variant"])),
                    "identity_source": "frozen fixture and the pinned command dual-tower run",
                }
            else:
                native = method == "text2cad_release"
                cells = row["cells"] if native else row["scores"]
                verify_native_identity(row, pair, histories, "native" if native else "hiercad")
                require(all(cell["logit_dtype"] == ("torch.float32" if native else "torch.bfloat16")
                            for cell in cells), f"Historical precision changed for {method}/{key}")
                result = archived_view([cell["s0"] for cell in cells],
                                       row["metrics"]["Text2CAD_S0"] if native else row["metrics"])
                result["effective_inputs"] = native_cell_evidence(cells)
                if native:
                    require((row["family"], row["attribute"]) == (pair["family"], pair["attribute"])
                            and [(cell["query"], cell["cad"]) for cell in cells] == list(CELLS),
                            f"Text2CAD stored identity changed for {key}")
                    for rule in ("first_cut_slot", "local_sketch_distance"):
                        grids[rule][key] = archived_view(row["rule_scores"][rule], row["metrics"][rule])
                    bow, spectra = original_collision_proofs(row, pair, histories, view)
                    grids["bow_collision"][key], grids["spectra_collision"][key] = bow, spectra
            grids[method][key] = result
    return grids


def external_geometry(read, histories):
    manifest = json.loads(read(f"{EXTERNAL}/inputs/inputs.json", EXTERNAL_INPUT_SHA))
    refs = {}
    for cad in manifest["cad"]:
        pid, entity, member = cad["pair_id"], cad["entity"], cad["source_history"]
        require(member == entity + "0" and (pid, entity) not in refs
                and histories[pid, member]["raw_program_sha256"] == cad["raw_program_sha256"],
                f"Finished shape representative changed for {pid}/{entity}")
        refs[pid, entity] = cad
    require(len(refs) == 80, "Wrong number of finished shape representatives")
    return refs, manifest["texts"]


def read_finished(read, axes, histories, visibility):
    cad_refs, texts = external_geometry(read, histories)
    audits = {
        EXTERNAL: json.loads(read(f"{EXTERNAL}/independent_audit.json",
                                  "8627a8711218fe6cd1518873aa279bbfba6825554ddcd86dbb9d8532310fdde4")),
        TOP: json.loads(read(f"{TOP}/independent_audit.json",
                             "7bd2095363fb065cb5693e3bd0a765bedd4b81454f2d0e1e60b4e9886bdeb1b1")),
    }
    grids, verified = {}, {}
    for method, (root, pin, axis_pin, suffix) in EXTERNAL_SCORES.items():
        axis = json.loads(read(f"{root}/{method}/axes.json", axis_pin))
        source = audits[root]
        audit = source["models"][method]
        require(axis["cad"] == list(map(list, cad_refs)) and axis["texts"] == texts
                and axis["input_manifest_sha256"] == EXTERNAL_INPUT_SHA
                and source["all_passed"] is True and audit["views_sha256"] == pin,
                f"Finished shape axes or independent audit changed for {method}")
        rows = raw_grid(read, f"{root}/{method}/paired_views.jsonl", pin, axes)
        for key, view in axes.items():
            pid, order, variant = key
            row = rows[key]
            ti = [texts.index(text) for text in view["texts"]]
            ci = [axis["cad"].index([pid, entity]) for entity in "AB"]
            require(row["text_indices"] == ti and row["cad_indices"] == ci, f"Swapped score identity for {method}/{key}")
            require(row["scores_row_major"] == rows[pid, 1 - order, variant]["scores_row_major"],
                    f"Finished shape scores differ by history for {method}/{key}")
            result = archived_view(row["scores_row_major"], row["metrics"])
            cads = [cad_refs[pid, entity] for entity in "AB"]
            shape_hashes = [cad["files"][suffix] if suffix else visibility.get(cad["stem"]) for cad in cads]
            result["effective_inputs"] = {
                "representative_history": [cad["source_history"] for cad in cads],
                "representative_program_sha256": [cad["raw_program_sha256"] for cad in cads],
                "archived_representation_sha256": shape_hashes,
                "text_indices": ti, "cad_indices": ci, "text_tokens": [axis["text_token_counts"][i] for i in ti],
            }
            grids.setdefault(method, {})[key] = result
        verified[method] = audit
    return grids, verified


def replay_published(pairs, grids, external_audits):
    compared = {}
    for method, grid in grids.items():
        groups = {}
        total = Counter()
        for pair in pairs.values():
            pid = pair["pair_id"]
            rows = {(order, variant): grid[pid, order, variant] for order, variant in GRID}
            result = pair_result(pair, rows)
            counts = Counter(pairs=1, views=8, both_cads=result["cad_correct_views"],
                             both_captions=result["caption_correct_views"],
                             swapped=result["cad_swapped_views"], D_positive=result["D_positive"],
                             joint=result["G"], joint_caption=int(result["caption_correct_views"] == 8),
                             joint_swap=result["G_swap"])
            if method in external_audits:
                counts.update(ties=result["cad_ties"] + result["caption_ties"],
                              cad_correct=sum(m > 0 for row in rows.values() for m in row["cad_margins"]),
                              caption_correct=sum(m > 0 for row in rows.values() for m in row["caption_margins"]),
                              small_nonzero_margins=sum(0 < abs(m) <= 1e-6 for row in rows.values()
                                                        for m in row["cad_margins"] + row["caption_margins"]))
            groups.setdefault(f"{pair['family']}/{pair['attribute']}", Counter()).update(counts)
            total.update(counts)
        keys = ("both_cads", "both_captions", "D_positive", "joint")
        require(tuple(total[key] for key in keys) == PUBLISHED[method] and total["joint_swap"] == 0,
                f"Development counts disagree with the published record for {method}")
        if method in external_audits:
            audit = external_audits[method]
            require(audit["overall"] == total and audit["strata"] == groups,
                    f"Development counts disagree with the original detailed audit for {method}")
        compared[method] = {"counts": dict(total), "strata": groups}
    return compared


def replay_coarse(read, pairs, histories):
    paths = {
        "text2cad_release": (TEXT2CAD, "ed16101953247b98aeebb31b3abadb9dca6dee332a60ccbd37a05cc4ff65ead4"),
        "hiercad_release": (H_COARSE, "984104f349cd34c5a62d6e21c422f7e7b5d8ebeb9432391d4a1b4dc929bab872"),
    }
    by_design = {(p["family"], p["attribute"], p["design"]["index"]): pid for pid, p in pairs.items()}
    all_rows = {}
    for method, (root, pin) in paths.items():
        rows = keyed(json.loads(read(f"{root}/coarse_shape_controls.json", pin)),
                     lambda row: (row["family"], row["index"]))
        require(rows.keys() == set(product(DEVELOPMENT_FAMILIES, range(2))), "Historical easy controls changed")
        for (family, index), row in rows.items():
            members = [by_design[family, attr, index] for attr in ATTRIBUTES]
            native = method == "text2cad_release"
            cells = row["cells"] if native else row["scores"]
            for (query, cad), cell in zip(product(range(2), repeat=2), cells, strict=True):
                target = histories[members[cad], "A0"]
                if native:
                    require(cell["member"] == f"{members[cad]}/A0"
                            and cell["text_sha256"] == digest(row["texts"][query].encode("utf-8")),
                            "Wrong Text2CAD easy-control input")
                else:
                    require(row["pair_ids"] == members
                            and cell["target_sha256"] == target["hiercad_target"]["response_sha256"],
                            "Wrong HierCAD easy-control input")
            result = archived_view([cell["s0"] for cell in cells], row["metrics"])
            entry = all_rows.setdefault((family, index), {"family": family, "index": index, "texts": row["texts"],
                                        "representatives": [f"{pid}/A0" for pid in members], "methods": {}})
            require(entry["texts"] == row["texts"], "Easy-control text changed between methods")
            entry["methods"][method] = result
    for method, expected in (("text2cad_release", (3, 1)), ("hiercad_release", (4, 0))):
        values = [row["methods"][method] for row in all_rows.values()]
        require((sum(v["cad_choice"] == "correct" for v in values),
                 sum(v["caption_choice"] == "correct" for v in values)) == expected,
                f"Historical easy-control counts differ for {method}")
    return list(all_rows.values())


def load_development(root):
    receipts = {}

    def read(relative, pin):
        return read_pinned(root / relative, pin, receipts)

    pairs, histories, axes = replay_fixture(read)
    token_path = f"{PILOT}/retriever_effective_inputs/dual_t.jsonl"
    tokens = {}
    if (root / token_path).is_file():
        for pair in json_lines(read(token_path, DUAL_TOKEN_SHA)):
            for row in pair["caption_inputs"]:
                require(row["used_tokens"] == row["full_tokens"], "Archived dual caption was truncated")
                tokens[pair["pair_id"], row["variant"]] = row["token_sha256"]
        require(tokens.keys() == set(product(pairs, VARIANTS)), "Archived dual token coverage changed")
    top_path = f"{TOP}/clip_top/visibility.json"
    visibility = {}
    if (root / top_path).is_file():
        rendered = json.loads(read(top_path, "2be01bd5427d48565d576e8138faa91fbb4196cb5c24617fb5c176fd77bf1aea"))
        require(rendered["axis"] == "+Z" and len(rendered["renders"]) == 80, "Top-view evidence differs")
        visibility = {row["case"]: row["sha256"] for row in rendered["renders"]}
    grids = read_early(read, axes, pairs, histories, tokens)
    finished, audits = read_finished(read, axes, histories, visibility)
    grids.update(finished)
    grids = {method: grids[method] for method in PROVENANCE}
    provenance = {method: dict(zip(("label", "source", "representation", "evidence"), fields, strict=True))
                  for method, fields in PROVENANCE.items()}
    return {
        "scope": "development", "pairs": list(pairs.values()), "views": axes, "grids": grids,
        "exclusions": {}, "geometry": {}, "provenance": provenance, "input_sha256": receipts,
        "blind_reviews": {}, "exchangeability": None, "exchangeable": False,
        "supplement": {
            "development_only": True, "published_counts": replay_published(pairs, grids, audits),
            "archived_coarse_shape_controls": replay_coarse(read, pairs, histories),
            "optional_effective_identity_coverage": {"dual_caption_pairs": len(tokens) // len(VARIANTS),
                                                     "top_image_representatives": len(visibility)},
        },
    }
