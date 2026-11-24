#!/usr/bin/env python3
"""Evaluate fixed Table 4 grids, or replay the pinned 40-pair development archive.

Formal mode takes two JSON receipts. The freeze receipt supplies the formal
scope, seal time, family IDs, exchangeability declaration, and references to
pairs, histories, blind reviews and preregistration. The runs receipt supplies
one entry per attempted method, with score, model geometry, execution and
applicable training-selection references. Each reference has path and sha256.

Score JSONL positions are AA, AB, BA, BB. Every row includes pair_id, order,
variant, view_sha256, scores, failures, cad_input_sha256 and cell_input_sha256.
The view digest uses the complete frozen view object plus its pair_id.
"""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from itertools import combinations, product
import json
import math
from pathlib import Path
from random import Random
from statistics import mean, median


ATTRIBUTES = ("size_location", "shape_location")
VARIANTS = ("near_first", "far_first", "near_first_rephrased", "far_first_rephrased")
GRID = tuple(product(range(2), VARIANTS))
CELLS = tuple(product("AB", repeat=2))
DEVELOPMENT_FAMILIES = ("rectangular_notch_plate", "v_notch_flange")
BOOTSTRAP_SEED = 20261124
BOOTSTRAP_DRAWS = 5000
ALPHA = 0.0125
FORMAL_METHODS = {
    "first_cut_slot": ("First attribute and first cut", "rule", "none"),
    "local_sketch_distance": ("Local sketch distance", "rule", "none"),
    "bow_t": ("Word bag and command histogram", "unordered", "t2cad-v1.0"),
    "bow_ct": ("Word bag and command histogram", "unordered", "ct-deepcad"),
    "dual_t": ("Command dual tower", "ordered", "t2cad-v1.0"),
    "dual_ct": ("Command dual tower", "ordered", "ct-deepcad"),
    "spectra_t": ("SpectraBRep", "unordered", "t2cad-v1.0"),
    "spectra_ct": ("SpectraBRep", "unordered", "ct-deepcad"),
    "text2cad_t": ("Text2CAD S0", "ordered", "t2cad-v1.0"),
    "hiercad_t": ("HierCAD S0", "ordered", "t2cad-v1.0"),
    "text2cad_ct": ("Text2CAD S0", "ordered", "ct-deepcad"),
    "hiercad_ct": ("HierCAD S0", "ordered", "ct-deepcad"),
    "mmcad": ("MM-CAD v2", "finished", "external"),
    "openshape": ("OpenShape", "finished", "external"),
    "clip": ("CLIP ViT-B/32 isometric", "finished", "external"),
    "clip_top": ("CLIP ViT-B/32 top view", "finished", "external"),
}
CONFIRMATORY = {"text2cad_t", "hiercad_t"}
FLIP_FIELDS = (
    "strict_cad_pass_changes", "strict_caption_pass_changes",
    "cad_choice_oppositions", "caption_choice_oppositions", "cad_choice_changes",
    "caption_choice_changes", "D_sign_oppositions",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def is_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def json_lines(data):
    return [json.loads(line) for line in data.splitlines()]


def read_pinned(path, expected, receipts):
    data = path.read_bytes()
    actual = digest(data)
    require(actual == expected, f"File does not match its SHA-256: {path}")
    receipts[str(path)] = actual
    return data


def read_reference(base, reference, receipts):
    require(is_digest(reference["sha256"]), "A referenced artifact needs a SHA-256")
    return read_pinned(base / reference["path"], reference["sha256"], receipts)


def keyed(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, f"Duplicate record {identity}")
        result[identity] = row
    return result


def view_key(row):
    return row["pair_id"], row["order"], row["variant"]


def pair_view_key(pair, view):
    return pair["pair_id"], view["order"], view["variant"]


def view_identity_sha(view):
    return digest(json.dumps(view, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8"))


def margin(left, right):
    return None if left is None or right is None else left - right


def direction(values):
    if None in values:
        return "failed"
    if 0 in values:
        return "tie"
    if min(values) > 0:
        return "correct"
    if max(values) < 0:
        return "swapped"
    return "same"


def twice_credit(values):
    return sum(2 if value > 0 else int(value == 0) for value in values if value is not None)


def evaluate_view(scores, failures=None):
    """Scores are in AA, AB, BA, BB order. Null cells retain their failure reason."""
    require(len(scores) == 4, "A view needs exactly four cells in AA, AB, BA, BB order")
    if failures is None:
        failures = [None] * 4
    require(len(failures) == 4, "A view needs a failure field for every cell")
    for value, reason in zip(scores, failures, strict=True):
        if value is None:
            require(isinstance(reason, str) and reason.strip(), "A missing cell needs a failure reason")
        else:
            require(type(value) in (float, int) and math.isfinite(value) and reason is None,
                    "A scored cell needs a finite number and no failure reason")
    aa, ab, ba, bb = scores
    cad = [margin(aa, ba), margin(bb, ab)]
    caption = [margin(aa, ab), margin(bb, ba)]
    complete = None not in scores
    contrast = math.fsum((aa, bb, -ab, -ba)) if complete else None
    return {
        "scores": scores, "failures": failures, "kind": "scored",
        "cad_margins": cad, "caption_margins": caption, "D": contrast,
        "cad_choice": direction(cad), "caption_choice": direction(caption),
        "cad_ties": cad.count(0), "caption_ties": caption.count(0),
        "cad_failures": cad.count(None), "caption_failures": caption.count(None),
        "cad_credit_twice": twice_credit(cad), "caption_credit_twice": twice_credit(caption),
        "failed_cells": scores.count(None),
    }


def collision_view(axis, evidence):
    # Equal query rows force two zero CAD margins. Equal CAD columns force two
    # zero caption margins. The other direction has opposite unknown margins.
    require(axis in ("cad", "caption"), "Unknown equality axis")
    return {
        "scores": [None] * 4, "failures": [None] * 4, "kind": "proven_collision", "proof": evidence,
        "cad_margins": [0, 0] if axis == "cad" else [None, None],
        "caption_margins": [0, 0] if axis == "caption" else [None, None], "D": 0,
        "cad_choice": "tie" if axis == "cad" else "tie_or_same",
        "caption_choice": "tie" if axis == "caption" else "tie_or_same",
        "cad_ties": 2 if axis == "cad" else None, "caption_ties": 2 if axis == "caption" else None,
        "cad_failures": None, "caption_failures": None, "failed_cells": None,
        "cad_credit_twice": 2, "caption_credit_twice": 2,
    }


def order_relation(view):
    # Both candidates use the same first hole in the frozen order factorial.
    first = view["variant"].partition("_")[0]
    return "aligned" if view["first_cut"] == [first, first] else "conflict"


def choices(values, field):
    return sum(value[field] for value in values) if all(value[field] is not None for value in values) else None


def summarize_views(values):
    result = dict(views=len(values), items=2 * len(values), score_cells=4 * len(values))
    for axis in ("cad", "caption"):
        counts = Counter(value[f"{axis}_choice"] for value in values)
        for status in ("correct", "swapped", "same", "tie", "failed", "tie_or_same"):
            result[f"{axis}_{status}_views"] = counts[status]
        for field in ("ties", "failures", "credit_twice"):
            key = f"{axis}_{field}"
            result[key] = choices(values, key)
        margins = [m for value in values for m in value[f"{axis}_margins"] if m is not None]
        result[f"{axis}_margin_count"] = len(margins)
        result[f"{axis}_median_abs_margin"] = median(map(abs, margins)) if margins else None
    contrasts = [value["D"] for value in values if value["D"] is not None]
    result.update(
        D_available=len(contrasts), D_positive=sum(value > 0 for value in contrasts),
        D_negative=sum(value < 0 for value in contrasts), D_zero=contrasts.count(0),
        D_mean=mean(contrasts) if contrasts else None, failed_cells=choices(values, "failed_cells"),
        unobserved_cells=4 * sum(value["kind"] == "proven_collision" for value in values),
    )
    return result


def flip_counts(table):
    comparisons = {
        "history": [((0, variant), (1, variant)) for variant in VARIANTS],
        "narration": [((order, "near_first" + suffix), (order, "far_first" + suffix))
                      for order, suffix in product(range(2), ("", "_rephrased"))],
        "wording": [((order, variant), (order, variant + "_rephrased"))
                    for order, variant in product(range(2), ("near_first", "far_first"))],
    }
    result = {}
    for factor, edges in comparisons.items():
        counts = Counter({field: 0 for field in FLIP_FIELDS})
        counts["comparisons"] = len(edges)
        for left, right in edges:
            a, b = table[left], table[right]
            for axis in ("cad", "caption"):
                counts[f"strict_{axis}_pass_changes"] += (a[f"{axis}_choice"] == "correct") != (
                    b[f"{axis}_choice"] == "correct")
                for x, y in zip(a[f"{axis}_margins"], b[f"{axis}_margins"], strict=True):
                    if x is not None and y is not None:
                        counts[f"{axis}_choice_oppositions"] += (x < 0 < y) or (y < 0 < x)
                        counts[f"{axis}_choice_changes"] += ((x > 0) - (x < 0)) != ((y > 0) - (y < 0))
                        counts[f"{axis}_choice_comparisons"] += 1
            if a["D"] is not None and b["D"] is not None:
                counts["D_sign_oppositions"] += (a["D"] < 0 < b["D"]) or (b["D"] < 0 < a["D"])
                counts["D_comparisons"] += 1
        result[factor] = dict(counts)
    return result


def pair_result(pair, table, exclusion=None):
    if pair["kind"] == "easy":
        return summarize_views(list(table.values()))
    require(set(table) == set(GRID), f"Incomplete 2 × 2 × 2 grid for {pair['pair_id']}")
    require(Counter(order_relation(view) for view in pair["views"]) == {"aligned": 4, "conflict": 4},
            f"Unbalanced narration/cut order for {pair['pair_id']}")
    direct = [value["cad_choice"] for value in table.values()]
    g = int(exclusion is None and all(value == "correct" for value in direct))
    swapped = int(exclusion is None and all(value == "swapped" for value in direct))
    aligned = {(view["order"], view["variant"]): order_relation(view) for view in pair["views"]}
    return {
        **summarize_views(list(table.values())), "G": g, "G_swap": swapped, "N": (g + swapped) / 2,
        "primary_exclusion": exclusion, "flip_counts": flip_counts(table),
        "order_relation": {name: summarize_views([value for key, value in table.items() if aligned[key] == name])
                           for name in ("aligned", "conflict")},
    }


def percentile(sorted_values, p):
    index = (len(sorted_values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return sorted_values[low] + (index - low) * (sorted_values[high] - sorted_values[low])


def family_interval(values):
    rng = Random(BOOTSTRAP_SEED)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_DRAWS))
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def exact_sign_flip(block_numerators):
    """The 20-pair blocks share a denominator, so ties can be counted exactly."""
    observed = abs(sum(block_numerators))
    permutations = 2 ** len(block_numerators)
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, block_numerators, strict=True))) >= observed
                  for signs in product((-1, 1), repeat=len(block_numerators)))
    return {"extreme": extreme, "permutations": permutations, "p_two_sided": extreme / permutations}


def method_diagnostics(values, prefix=""):
    def rate(numerator, denominator):
        return numerator / denominator if numerator is not None and denominator else None

    return {
        prefix + "cad_item": rate(values["cad_credit_twice"], 2 * values["items"]),
        prefix + "caption_item": rate(values["caption_credit_twice"], 2 * values["items"]),
        prefix + "reverse_strict": rate(values["caption_correct_views"], values["views"]),
        prefix + "cad_tie": rate(values["cad_ties"], values["items"]),
        prefix + "caption_tie": rate(values["caption_ties"], values["items"]),
        prefix + "cad_failure": rate(values["cad_failures"], values["items"]),
        prefix + "caption_failure": rate(values["caption_failures"], values["items"]),
    }


def summarize_method(method, pairs, grid, exclusions, scope, exchangeable, coordinates=None):
    results = {}
    groups = defaultdict(list)
    for pair in pairs:
        pid = pair["pair_id"]
        table = {pair_view_key(pair, view)[1:]: grid[pair_view_key(pair, view)] for view in pair["views"]}
        results[pid] = pair_result(pair, table, exclusions.get(pid))
        groups[pair["attribute"], pair["family"], pair["kind"]].append(pair)
    by_attribute = {}
    table = []
    for attribute in ATTRIBUTES:
        blocks = {}
        for (attr, family, kind), block in groups.items():
            if (attr, kind) != (attribute, "binding"):
                continue
            values = [results[pair["pair_id"]] for pair in block]
            flat = [grid[pair_view_key(pair, view)] for pair in block for view in pair["views"]]
            easy = [grid[pair_view_key(pair, view)] for pair in groups.get((attribute, family, "easy"), [])
                    for view in pair["views"]]
            g = sum(value["G"] for value in values)
            swapped = sum(value["G_swap"] for value in values)
            counts = summarize_views(flat)
            easy_counts = summarize_views(easy)
            eligible = [value for value in values if value["primary_exclusion"] is None]
            diffs = {
                name: {outcome: sum(value["flip_counts"][name].get(outcome, 0) for value in values)
                       for outcome in (*FLIP_FIELDS, "comparisons", "cad_choice_comparisons",
                                       "caption_choice_comparisons", "D_comparisons")}
                for name in ("history", "narration", "wording")
            }
            blocks[family] = {
                "pairs": len(block), "G_count": g, "G_swap_count": swapped, "N_twice_sum": g + swapped,
                "difference": (g - swapped) / (2 * len(block)),
                "eligible_pairs": len(eligible), "G_on_eligible_count": sum(v["G"] for v in eligible),
                "exclusions": dict(Counter(v["primary_exclusion"] for v in values if v["primary_exclusion"])),
                "binding": counts, "easy": easy_counts, "flip_counts": diffs,
                "order_relation": {
                    name: summarize_views([grid[pair_view_key(pair, v)] for pair in block for v in pair["views"]
                                           if order_relation(v) == name])
                    for name in ("aligned", "conflict")
                },
                "diagnostics": {**method_diagnostics(counts), **method_diagnostics(easy_counts, "easy_")},
            }
        require(bool(blocks), f"No binding pairs for {attribute}")
        blocks = dict(sorted(blocks.items()))
        ordered = list(blocks.values())
        diffs = [block["difference"] for block in ordered]
        is_formal = scope == "formal"
        confirmatory = is_formal and method in CONFIRMATORY
        diag = {key: mean(block["diagnostics"][key] for block in ordered)
                if all(block["diagnostics"][key] is not None for block in ordered) else None
                for key in ordered[0]["diagnostics"]}
        aggregate = {
            "method": method, "attribute": attribute, "scope": scope, "families": len(blocks),
            "pairs": sum(block["pairs"] for block in ordered),
            "G_count": sum(block["G_count"] for block in ordered),
            "G_swap_count": sum(block["G_swap_count"] for block in ordered),
            "G": mean(block["G_count"] / block["pairs"] for block in ordered),
            "N": mean(block["N_twice_sum"] / (2 * block["pairs"]) for block in ordered),
            "difference": mean(diffs), "difference_ci_95": family_interval(diffs) if is_formal else None,
            "leave_one_family_out": {family: mean(other["difference"] for key, other in blocks.items()
                                                   if key != family) for family in blocks},
            "inference": "descriptive",
            "alpha": ALPHA if confirmatory else None, "p_two_sided": None,
            "p_extreme_blocks": None, "p_total_blocks": None,
            "direction": ("positive" if sum(diffs) > 0 else "negative" if sum(diffs) < 0 else "zero"),
            "expanded_binding_views": sum(block["binding"]["views"] for block in ordered),
            "distinct_history_free_views": (
                4 * sum(block["pairs"] for block in ordered)
                if (method in FORMAL_METHODS and FORMAL_METHODS[method][1] == "finished") else None),
            **diag,
        }
        if confirmatory and exchangeable:
            test = exact_sign_flip([block["G_count"] - block["G_swap_count"] for block in ordered])
            aggregate.update(p_two_sided=test["p_two_sided"], p_extreme_blocks=test["extreme"],
                             p_total_blocks=test["permutations"], inference=(
                                 "passed" if test["p_two_sided"] <= ALPHA and sum(diffs) > 0 else "did_not_pass"))
        coordinate_result = None
        if coordinates is not None:
            coordinate_result = summarize_coordinates(pairs, results, coordinates, attribute, aggregate)
        by_attribute[attribute] = {"aggregate": aggregate, "families": blocks, "coordinate": coordinate_result}
        table.append(aggregate)
    return results, by_attribute, table


def summarize_coordinates(pairs, results, coverage, attribute, binding_result):
    groups = defaultdict(list)
    for pair in pairs:
        if pair["kind"] == "coordinate" and pair["attribute"] == attribute:
            groups[pair["family"]].append(pair)
    blocks = {}
    for family, block in groups.items():
        subset = [pair for pair in block if coverage[pair["pair_id"]]["effective_coordinate_conflict"]
                  and results[pair["pair_id"]]["primary_exclusion"] is None]
        before = [results[pair["parent_pair_id"]] for pair in subset]
        after = [results[pair["pair_id"]] for pair in subset]
        blocks[family] = {
            "selected": len(block),
            "raw_confirmed_conflict": sum(pair["coordinate_conflict"] for pair in block),
            "effective_replay_eligible": sum(results[pair["pair_id"]]["primary_exclusion"] is None for pair in block),
            "effective_conflict_eligible": len(subset),
            "G_before_count": sum(row["G"] for row in before),
            "G_after_count": sum(row["G"] for row in after),
            "N_after_twice_sum": sum(int(2 * row["N"]) for row in after),
            "delta_conservative": (
                sum(row["G"] for row in after) - sum(results[p["parent_pair_id"]]["G"] for p in block)
            ) / len(block),
        }
    blocks = dict(sorted(blocks.items()))
    covered = [row for row in blocks.values() if row["effective_conflict_eligible"]]
    before = [row["G_before_count"] / row["effective_conflict_eligible"] for row in covered]
    after = [row["G_after_count"] / row["effective_conflict_eligible"] for row in covered]
    baseline = [row["N_after_twice_sum"] / (2 * row["effective_conflict_eligible"]) for row in covered]
    differences = [a - b for a, b in zip(after, before, strict=True)]
    advantages = [a - b for a, b in zip(after, baseline, strict=True)]
    all_families = len(covered) == 12
    diff_interval = family_interval(differences) if all_families else None
    advantage_interval = family_interval(advantages) if all_families else None
    stable = (binding_result["method"] == "text2cad_t" and binding_result["inference"] == "passed"
              and all_families and advantage_interval[0] > 0 and diff_interval[0] > -0.05)
    return {
        "selected": sum(row["selected"] for row in blocks.values()),
        "raw_confirmed_conflict": sum(row["raw_confirmed_conflict"] for row in blocks.values()),
        "effective_replay_eligible": sum(row["effective_replay_eligible"] for row in blocks.values()),
        "effective_conflict_eligible": sum(row["effective_conflict_eligible"] for row in blocks.values()),
        "covered_families": len(covered), "families": blocks,
        "G_before_on_conflicts": mean(before) if covered else None,
        "G_after_on_conflicts": mean(after) if covered else None,
        "N_after_on_conflicts": mean(baseline) if covered else None,
        "G_change_on_conflicts": mean(differences) if covered else None,
        "G_change_ci_95": diff_interval, "G_after_minus_N_ci_95": advantage_interval,
        "G_change_conservative": mean(row["delta_conservative"] for row in blocks.values()),
        "stability_confirmed": stable,
    }


def load_view_axes(pair_rows, histories):
    views = {}
    for pair in pair_rows:
        pid = pair["pair_id"]
        expected = {(view["order"], view["variant"]) for view in pair["views"]}
        require(len(expected) == len(pair["views"]) and
                (len(expected) == 1 and expected <= set(GRID) if pair["kind"] == "easy" else expected == set(GRID)),
                f"Incorrect frozen view grid for {pid}")
        full = pair["kind"] != "easy"
        for item in pair["views"]:
            order, variant = item["order"], item["variant"]
            members = [f"{entity}{order}" for entity in "AB"]
            require(type(order) is int and item["histories"] == members and len(item["texts"]) == 2
                    and item["texts"][0] != item["texts"][1]
                    and all(isinstance(text, str) and text.strip() for text in item["texts"]),
                    f"Invalid query or CAD axis for {pid}/{order}/{variant}")
            source = [histories[pid, member] for member in members]
            require([history["entity"] for history in source] == ["A", "B"]
                    and all(is_digest(history["raw_program_sha256"]) for history in source)
                    and item["geometry_sha256"] == [history["raw_program_sha256"] for history in source]
                    and item["first_cut"] == [history["cut_order"][0] for history in source],
                    f"Incorrect frozen CAD identity for {pid}/{order}/{variant}")
            if full:
                captions = [pair["captions"][entity][variant] for entity in "AB"]
                require(item["texts"] == captions
                        and [digest(text.encode("utf-8")) for text in captions] ==
                        [pair["caption_sha256"][entity][variant] for entity in "AB"],
                        f"Incorrect frozen text identity for {pid}/{order}/{variant}")
            view = {**item, "pair_id": pid}
            views[view_key(view)] = view
    return views


def load_formal_fixture(freeze_path, receipts):
    source = freeze_path.read_bytes()
    receipts[str(freeze_path)] = digest(source)
    freeze = json.loads(source)
    require(freeze["scope"] == "formal" and bool(freeze["sealed_at_utc"]),
            "Formal evaluation requires a sealed formal fixture")
    preregistration = read_reference(freeze_path.parent, freeze["preregistration"], receipts)
    require(bool(preregistration.strip()), "Preregistration is empty")
    families = set(freeze["families"])
    require(len(freeze["families"]) == len(families) == 12 and families.isdisjoint(DEVELOPMENT_FAMILIES),
            "The sealed fixture must name twelve different held-out families")

    def load_rows(name):
        return json_lines(read_reference(freeze_path.parent, freeze[name], receipts))

    pairs = keyed(load_rows("pairs"), lambda row: row["pair_id"])
    histories = keyed(load_rows("histories"), lambda row: (row["pair_id"], row["history"]))
    reviews = keyed(load_rows("blind_reviews"), lambda row: row["pair_id"])
    require(reviews.keys() == {pid for pid, pair in pairs.items() if pair["kind"] != "coordinate"},
            "Blind approvals must cover exactly the frozen binding and easy pairs")
    expected = {(f, a, k): n for f, a in product(families, ATTRIBUTES)
                for k, n in (("binding", 20), ("easy", 20), ("coordinate", 10))}
    require(Counter((row["family"], row["attribute"], row["kind"]) for row in pairs.values()) == expected,
            "Unexpected formal pair, easy-control, or coordinate denominators")
    views = load_view_axes(list(pairs.values()), histories)
    coordinate_parents = set()
    for pid, pair in pairs.items():
        if pair["kind"] == "coordinate":
            parent = pairs[pair["parent_pair_id"]]
            require(parent["kind"] == "binding" and parent["family"] == pair["family"]
                    and parent["attribute"] == pair["attribute"] and parent["captions"] == pair["captions"]
                    and parent["pair_id"] not in coordinate_parents and type(pair["coordinate_conflict"]) is bool,
                    f"Incorrect matched coordinate fixture for {pid}")
            coordinate_parents.add(parent["pair_id"])
            continue
        approval = reviews[pid]
        raters = approval["reviewers"]
        require(approval["resolution"] == "accepted" and len(raters) == 2
                and all(rater["blind"] is True and rater["decision"] == "approved"
                        and isinstance(rater["reviewer_id"], str) and rater["reviewer_id"].strip()
                        for rater in raters)
                and raters[0]["reviewer_id"] != raters[1]["reviewer_id"],
                f"Two distinct recorded blind approvals are required for {pid}")
    for pid, pair in pairs.items():
        if pair["kind"] == "coordinate":
            parent = pair["parent_pair_id"]
            reviews[pid] = {**reviews[parent], "pair_id": pid, "inherited_from": parent}
    exchange = freeze["exchangeability"]
    exchangeable = (exchange["independent_families"] is True and exchange["label_exchangeable"] is True
                    and bool(exchange["basis_in_preregistration"].strip()))
    return list(pairs.values()), views, reviews, exchange, exchangeable


def method_geometry_run(run, base, pairs, receipts):
    method = run["method"]
    geometry_rows = json_lines(read_reference(base, run["effective_geometry"], receipts))
    geometry = keyed(geometry_rows, lambda row: row["pair_id"])
    require({pair["pair_id"] for pair in pairs} == set(geometry), f"Incomplete geometry coverage for {method}")
    for pair in pairs:
        pid = pair["pair_id"]
        row = geometry[pid]
        require(row["status"] in ("verified", "failed") and isinstance(row["evidence"], str) and row["evidence"].strip(),
                f"Missing effective geometry replay evidence for {method}/{pid}")
        if row["status"] == "failed":
            require(isinstance(row["failure_reason"], str) and row["failure_reason"].strip(),
                    f"Missing geometry replay failure reason for {method}/{pid}")
        if pair["kind"] == "coordinate":
            require(type(row["effective_coordinate_conflict"]) is bool
                    and (not row["effective_coordinate_conflict"] or pair["coordinate_conflict"]),
                    f"Incorrect effective coordinate coverage for {method}/{pid}")
    return geometry


def method_score_run(run, base, pairs, axes, receipts):
    method = run["method"]
    _, encoding, source = FORMAL_METHODS[method]
    require(run["training_source"] == source and run["status"] == "attempted"
            and (encoding == "rule" or is_digest(run["checkpoint_sha256"])),
            f"Incorrect run status, training source, or checkpoint for {method}")
    executed = read_reference(base, run["execution_receipt"], receipts)
    require(bool(executed.strip()), f"Empty execution receipt for {method}")
    if source in ("t2cad-v1.0", "ct-deepcad"):
        selected = read_reference(base, run["training_selection"], receipts)
        require(bool(selected.strip()), f"Empty training-selection receipt for {method}")
    geometry = method_geometry_run(run, base, pairs, receipts)
    raw = keyed(json_lines(read_reference(base, run["score_rows"], receipts)), view_key)
    require(raw.keys() == axes.keys(), f"Score rows must match the entire frozen grid for {method}")
    grid = {}
    cad_hashes = {}
    for key, view in axes.items():
        row = raw[key]
        require(row["view_sha256"] == view_identity_sha(view), f"Wrong frozen input identity for {method}/{key}")
        result = evaluate_view(row["scores"], row["failures"])
        hashes = row["cad_input_sha256"]
        joint_hashes = row["cell_input_sha256"]
        require(len(hashes) == 2 and len(joint_hashes) == 4 and
                all(value is None or is_digest(value) for value in hashes + joint_hashes),
                f"Invalid effective input SHA-256 for {method}/{key}")
        for index, score in enumerate(result["scores"]):
            if score is not None:
                require(is_digest(joint_hashes[index]) and is_digest(hashes[index % 2]),
                        f"A valid score requires actual input hashes for {method}/{key}")
        for member, value in zip(view["histories"], hashes, strict=True):
            identity = key[0], member
            if value is not None:
                require(cad_hashes.get(identity, value) == value, f"Changing CAD input for {method}/{identity}")
                cad_hashes[identity] = value
        result["effective_inputs"] = {"cad_sha256": hashes, "cell_sha256": joint_hashes}
        grid[key] = result

    exclusions = {}
    for pair in pairs:
        pid = pair["pair_id"]
        if geometry[pid]["status"] == "failed":
            exclusions[pid] = geometry[pid]["failure_reason"]
        if pair["kind"] == "easy":
            continue
        per_entity = [[cad_hashes.get((pid, entity + str(order))) for order in range(2)] for entity in "AB"]
        if encoding == "ordered" and pid not in exclusions:
            if any(None in hashes for hashes in per_entity):
                exclusions[pid] = "missing_effective_history"
            elif any(hashes[0] == hashes[1] for hashes in per_entity):
                exclusions[pid] = "indistinguishable_ordered_histories"
        if encoding == "finished":
            require(all(None in hashes or hashes[0] == hashes[1] for hashes in per_entity),
                    f"Finished representation changed with history for {method}/{pid}")
            require(all(grid[pid, 0, variant]["scores"] == grid[pid, 1, variant]["scores"]
                        and grid[pid, 0, variant]["failures"] == grid[pid, 1, variant]["failures"]
                        for variant in VARIANTS), f"Finished scores changed with history for {method}/{pid}")
    return grid, exclusions, geometry


def formal_inputs(freeze_path, runs_path):
    receipts = {}
    pairs, views, reviews, exchange, exchangeable = load_formal_fixture(freeze_path, receipts)
    data = runs_path.read_bytes()
    receipts[str(runs_path)] = digest(data)
    methods = keyed(json.loads(data)["runs"], lambda run: run["method"])
    require(bool(methods) and methods.keys() <= FORMAL_METHODS.keys(), "Unknown or empty formal method list")
    grids, exclusions, geometry = {}, {}, {}
    provenance = {}
    for method, run in methods.items():
        grids[method], exclusions[method], geometry[method] = method_score_run(
            run, runs_path.parent, pairs, views, receipts)
        name, encoding, source = FORMAL_METHODS[method]
        provenance[method] = {**run, "label": name, "representation": encoding, "source": source}
    return {
        "scope": "formal", "pairs": pairs, "views": views, "grids": grids, "exclusions": exclusions,
        "geometry": geometry, "provenance": provenance, "input_sha256": receipts, "blind_reviews": reviews,
        "exchangeability": exchange, "exchangeable": exchangeable, "supplement": None,
    }


def pairwise_counts(unit_rows, outcome):
    result = []
    methods = list(unit_rows[0]["methods"])
    for attribute in (*ATTRIBUTES, "all"):
        rows = [row["methods"] for row in unit_rows if attribute in (row["attribute"], "all")]
        for left, right in combinations(methods, 2):
            matched = [(outcome(row[left]), outcome(row[right])) for row in rows]
            counts = Counter(matched)
            result.append({
                "attribute": attribute, "left": left, "right": right, "denominator": len(rows),
                "both": counts[True, True], "left_only": counts[True, False],
                "right_only": counts[False, True], "neither": counts[False, False],
            })
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=fields)
        output.writeheader()
        for row in rows:
            output.writerow({key: json.dumps(value, separators=(",", ":"))
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def markdown_table(table, provenance, scope):
    def percent(value):
        return "—" if value is None else f"{value * 100:.2f}"

    def measure(row, key):
        return percent(row.get(key))

    title = ("Table 4 · Held-out binding evaluation" if scope == "formal" else
             "Table 4 development replay · 40 previously inspected pairs")
    lines = [f"# {title}", ""]
    if scope == "development":
        lines.extend(["Two development families with historical weights and identical frozen questions. "
                      "All comparisons are descriptive. Proven input collisions carry theoretical outcomes "
                      "and blank raw score cells.", ""])
    lines.extend([
        "Binding results weight each family equally. G and N are percentages. The difference and interval are "
        "percentage points. Diagnostic rates are percentages.", "",
        "| Method | Source | Attribute | Pairs | G | N | G−N | 95% family interval | p | Inference | "
        "Easy CAD item | Binding CAD item | Both captions | CAD ties | CAD failures |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in table:
        name, source = provenance[row["method"]]["label"], provenance[row["method"]]["source"]
        interval = row.get("difference_ci_95")
        span = f"[{percent(interval[0])}, {percent(interval[1])}]" if interval else "—"
        p = row.get("p_two_sided")
        values = [name, source, row["attribute"], row.get("pairs", "—"), measure(row, "G"), measure(row, "N"),
                  measure(row, "difference"), span, f"{p:.6f}" if p is not None else "—", row["inference"],
                  measure(row, "easy_cad_item"), measure(row, "cad_item"), measure(row, "reverse_strict"),
                  measure(row, "cad_tie"), measure(row, "cad_failure")]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines) + "\n"


def write_report(inputs, output):
    scope, pairs, views = (inputs[key] for key in ("scope", "pairs", "views"))
    grids, exclusions, sources = (inputs[key] for key in ("grids", "exclusions", "provenance"))
    details, summaries, table = {}, {}, []
    for method, grid in grids.items():
        require(grid.keys() == views.keys(), f"The scored and frozen view sets differ for {method}")
        details[method], summaries[method], rows = summarize_method(
            method, pairs, grid, exclusions.get(method, {}), scope, inputs["exchangeable"],
            inputs["geometry"].get(method) if scope == "formal" else None)
        table.extend(rows)
    if scope == "formal":
        for method in FORMAL_METHODS:
            if method not in grids:
                name, encoding, source = FORMAL_METHODS[method]
                sources[method] = {"label": name, "source": source, "representation": encoding, "status": "not_scored"}
                table.extend({"method": method, "attribute": attr, "scope": scope, "inference": "not_scored"}
                             for attr in ATTRIBUTES)
        table.sort(key=lambda row: (tuple(FORMAL_METHODS).index(row["method"]), ATTRIBUTES.index(row["attribute"])))
    pair_rows, view_rows = [], []
    for pair in pairs:
        pid = pair["pair_id"]
        description = {key: pair[key] for key in ("pair_id", "family", "attribute", "kind")}
        pair_rows.append({
            "scope": scope, **description, "parent_pair_id": pair.get("parent_pair_id"),
            "blind_review": inputs["blind_reviews"].get(pid),
            "methods": {method: {**values[pid], "geometry": inputs["geometry"].get(method, {}).get(pid)}
                        for method, values in details.items()},
        })
        for item in pair["views"]:
            key = pair_view_key(pair, item)
            view = views[key]
            view_rows.append({
                "scope": scope, **description, **view, "view_sha256": view_identity_sha(view),
                "order_relation": order_relation(view), "methods": {method: grid[key] for method, grid in grids.items()},
            })
    matching_pairs = [row for row in pair_rows if row["kind"] == "binding"]
    matching_views = [row for row in view_rows if row["kind"] == "binding"]
    reason_counts = {
        method: dict(Counter(reason for result in grid.values() for reason in result["failures"] if reason))
        for method, grid in grids.items()
    }
    summary = {
        "scope": scope, "observed_pairs": len(pair_rows), "observed_views": len(view_rows),
        "input_sha256": inputs["input_sha256"], "method_provenance": sources,
        "exchangeability": inputs["exchangeability"], "exchangeability_supported": inputs["exchangeable"],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "paired family"}
        if scope == "formal" else None,
        "failure_reasons": reason_counts, "methods": summaries,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "joined_views.jsonl", view_rows)
    write_jsonl(output / "joined_pairs.jsonl", pair_rows)
    write_csv(output / "table.csv", table)
    (output / "table.md").write_text(markdown_table(table, sources, scope), encoding="utf-8")
    if len(grids) > 1:
        write_csv(output / "concordance_views.csv",
                  pairwise_counts(matching_views, lambda result: result["cad_choice"] == "correct"))
        write_csv(output / "concordance_pairs.csv", pairwise_counts(matching_pairs, lambda result: bool(result["G"])))
    if inputs["supplement"] is not None:
        write_json(output / "archived_replay_checks.json", inputs["supplement"])
    manifest = [f"{digest(path.read_bytes())}  {path.name}" for path in sorted(output.iterdir())]
    (output / "SHA256SUMS").write_text("\n".join(manifest) + "\n", encoding="ascii")
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    development = commands.add_parser("replay-development", help="Replay only the frozen development archive")
    development.add_argument("--archive-root", required=True, type=Path, help="Root mirroring archived evidence paths")
    development.add_argument("--out", required=True, type=Path, help="New output directory")
    formal = commands.add_parser("formal", help="Evaluate sealed, human-reviewed formal fixtures")
    formal.add_argument("--freeze", required=True, type=Path, help="Sealed protocol and fixture references in JSON")
    formal.add_argument("--runs", required=True, type=Path, help="Frozen method score and geometry receipts in JSON")
    formal.add_argument("--out", required=True, type=Path, help="New output directory")
    args = cli.parse_args()
    if args.command == "replay-development":
        from experiments.scripts.binding_development_archive import load_development

        inputs = load_development(args.archive_root)
    else:
        inputs = formal_inputs(args.freeze, args.runs)
    summary = write_report(inputs, args.out)
    print(f"{summary['scope']}: {summary['observed_pairs']} pairs, {summary['observed_views']} views, "
          f"{len(summary['methods'])} methods")


if __name__ == "__main__":
    main()
