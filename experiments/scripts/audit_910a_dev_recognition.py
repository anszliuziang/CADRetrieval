#!/usr/bin/env python3
"""Recompute development binding outcomes from the four saved scores of every view."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import statistics

VARIANTS = ("near_first", "far_first", "near_first_rephrased", "far_first_rephrased")
PAIR_SHA = "9f524f9e05d94c0cee321a26fba452395ad90f2d196422c9108d039271864824"


def read_rows(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def outcomes(values):
    if len(values) != 4 or any(not math.isfinite(item) for item in values):
        raise ValueError("Expected four finite scores per view")
    aa, ab, ba, bb = values
    direct = (aa - ba, bb - ab)
    reverse = (aa - ab, bb - ba)
    return {
        "both_cads": min(direct) > 0,
        "swapped_both_cads": max(direct) < 0,
        "both_captions": min(reverse) > 0,
        "D": math.fsum((aa, bb, -ab, -ba)),
        "ties": sum(delta == 0 for delta in (*direct, *reverse)),
        "direct_margins": direct,
    }


def verify_persisted(result, saved):
    for key in ("both_cads", "swapped_both_cads", "both_captions", "ties"):
        if result[key] != saved[key]:
            raise ValueError(f"Stored outcome does not match persisted scores: {key}")
    if not math.isclose(saved["D"], result["D"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Four-cell contrast disagrees with the persisted scores")


def unpack(row, mode):
    if mode == "native":
        cells = row["cells"]
        if [(cell["query"], cell["cad"]) for cell in cells] != list(product("AB", repeat=2)):
            raise ValueError("Native four-cell identity mismatch")
        if any(cell["logit_dtype"] != "torch.float32" or cell["effective_units"] != cell["n_valid_total"]
               for cell in cells):
            raise ValueError("Unexpected Text2CAD logit dtype or effective length")
        values = {"Text2CAD_S0": [cell["s0"] for cell in cells], **row["rule_scores"]}
        return {name: (scores, row["metrics"][name]) for name, scores in values.items()}
    if mode == "hiercad":
        cells = row["scores"]
        if (any(cell["logit_dtype"] != "torch.bfloat16" for cell in cells)
                or cells[0]["target_token_sha256"] != cells[2]["target_token_sha256"]
                or cells[1]["target_token_sha256"] != cells[3]["target_token_sha256"]):
            raise ValueError("Unexpected HierCAD dtype or query-dependent target")
        return {"HierCAD_S0": ([cell["s0"] for cell in cells], row["metrics"])}
    return {mode: (row["scores_aa_ab_ba_bb"], row)}


def compare_groups(rows, saved):
    for original, recomputed in zip(sorted(saved, key=group_key), sorted(rows, key=group_key), strict=True):
        keys = {"pairs", "views", "scores", "joint", "joint_swap", "N_twice", "both_cads",
                "both_captions", "D_positive", "ties", "fixed_cad_correct"}
        for key in keys & original.keys() & recomputed.keys():
            if original[key] != recomputed[key]:
                raise ValueError(f"Aggregate mismatch for {group_key(original)}, {key}")


def group_key(row):
    return row["family"], row["attribute"], row.get("method", "")


def audit(source, mode, reference):
    path = source / ("paired_views.jsonl" if mode == "native" else "views.jsonl")
    raw = read_rows(path)
    design_by_id = {row["pair_id"]: row["design"] for row in read_rows(reference)}
    if len(raw) != 320 or len(design_by_id) != 40:
        raise ValueError("Incomplete development pair or view count")
    seen = set()
    by_pair = defaultdict(dict)
    direct_margins = defaultdict(list)
    collisions = Counter()
    for row in raw:
        key = row["pair_id"], row["order"], row["variant"]
        if key in seen or key[0] not in design_by_id or key[1] not in (0, 1) or key[2] not in VARIANTS:
            raise ValueError(f"Duplicate or unexpected view: {key}")
        seen.add(key)
        for method, (scores, reported) in unpack(row, mode).items():
            result = outcomes(scores)
            verify_persisted(result, reported)
            by_pair[key[0], method][key[1:]] = result
            direct_margins[method].extend(result["direct_margins"])
        if mode == "native":
            collisions["bow_identical_caption"] += row["bow_identical_caption_input"] is True
            collisions["spectra_identical_cad"] += row["spectra_identical_cad_input"] is True

    counts = defaultdict(Counter)
    for (pair_id, method), views in by_pair.items():
        if set(views) != set(product(range(2), VARIANTS)):
            raise ValueError("One pair does not have the eight specified views")
        design = design_by_id[pair_id]
        win = all(value["both_cads"] for value in views.values())
        swapped = all(value["swapped_both_cads"] for value in views.values())
        row = counts[design["family"], design["attribute"], method]
        row.update(pairs=1, views=8, scores=32, joint=win, joint_swap=swapped,
                   N_twice=int(win) + int(swapped), both_cads=sum(x["both_cads"] for x in views.values()),
                   both_captions=sum(x["both_captions"] for x in views.values()),
                   D_positive=sum(x["D"] > 0 for x in views.values()), ties=sum(x["ties"] for x in views.values()),
                   fixed_cad_correct=sum(delta > 0 for view in views.values() for delta in view["direct_margins"]))
        for variant in VARIANTS:
            row["history_order_flips"] += views[0, variant]["both_cads"] != views[1, variant]["both_cads"]
        for order in range(2):
            for prefix in ("near", "far"):
                row["paraphrase_flips"] += views[order, prefix + "_first"]["both_cads"] != views[
                    order, prefix + "_first_rephrased"]["both_cads"]
            for suffix in ("", "_rephrased"):
                row["narration_order_flips"] += views[order, "near_first" + suffix]["both_cads"] != views[
                    order, "far_first" + suffix]["both_cads"]
    groups = [{"family": f, "attribute": a, "method": name, **value} for (f, a, name), value in counts.items()]
    result = {"mode": mode, "views": len(seen), "groups": groups,
              "abs_margin_medians": {name: statistics.median(map(abs, values))
                                     for name, values in direct_margins.items()},
              "identity_checks": dict(collisions)}
    saved = json.loads((source / "summary.json").read_bytes())
    if mode == "native":
        compare_groups(groups, saved["groups"])
        controls = json.loads((source / "coarse_shape_controls.json").read_bytes())
        if len(controls) != 4 or len({(row["family"], row["index"]) for row in controls}) != 4:
            raise ValueError("Missing coarse shape sanity controls")
        result["coarse_shape_controls"] = [
            outcomes([cell["s0"] for cell in row["cells"]]) for row in controls
        ]
    else:
        compare_groups(groups, [{**group, "method": groups[0]["method"]} for group in saved["groups"]])
    if mode == "hiercad":
        layouts = json.loads((source / "layouts.json").read_bytes())
        if len(layouts) != 8:
            raise ValueError("Missing HierCAD layout diagnostics")
        result["layout_decision_flips"] = {
            "fixed_cad": sum(row["both_cads_changed"] for row in layouts),
            "fixed_caption": sum(row["both_captions_changed"] for row in layouts),
            "max_abs_mean_delta": max(max(row["abs_mean_deltas"]) for row in layouts),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "dual_t", "dual_ct", "hiercad"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.pairs.read_bytes()).hexdigest() != PAIR_SHA:
        raise ValueError("Development ground truth file differs")
    report = audit(args.source, args.mode, args.pairs)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{args.mode}: recomputed {report['views']} views successfully")


if __name__ == "__main__":
    main()
