"""CPU statistics for the frozen E1 instance-retrieval protocol."""

import csv
import hashlib
import json
import math
import platform
import re
import secrets
import shlex
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from text2cad_g5_cpu import bootstrap_statistics, duplicate_ceilings, rank_metrics, sha256, valid_ranks, write_json


KS = (1, 5, 20)
METHODS = ("S0", "S48")
N_BACKGROUND = 48
DRAWS = 5000
PERM_SEED = 2026111901
BOOT_SEED = 2026111902
SEED = "cad-domain-v1"
ENCODING_TOLERANCE_PP = 3.125
PAIR_FLAGS = ("has_single_circle", "has_near_rectangle", "has_multiloop_or_cut")
VARIANTS = ("Qgeom", "Qvalue")
REWRITE_STATUSES = ("changed_valid", "no_edit", "unverifiable", "overlength")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def selection_key(purpose, domain, uid):
    return digest(f"{SEED}|{purpose}|{domain}|{uid}"), uid


def subseed(base, domain, analysis):
    return int.from_bytes(digest(f"{base}|{domain}|{analysis}")[:8], "big")


def log_mean_exp(scores):
    peak = scores.max(axis=0)
    return peak + np.log(np.exp(scores - peak).mean(axis=0))


def score_readouts(scores, background):
    require(scores.ndim == 2 and scores.shape[1] > 0 and np.isfinite(scores).all(), "invalid raw scores")
    require(background.shape == (N_BACKGROUND, scores.shape[1]) and np.isfinite(background).all(),
            "invalid ordered background score matrix")
    offset = log_mean_exp(background)
    return {
        "S0": scores,
        "S48": scores - offset,
        "half_0_23": scores - log_mean_exp(background[:24]),
        "half_24_47": scores - log_mean_exp(background[24:]),
        "background_offset": offset,
    }


def rank_matrix(scores, item_uids, domain, purpose="tie-cad"):
    require(scores.ndim == 2 and scores.shape[1] == len(item_uids) and len(item_uids) > 0
            and len(set(item_uids)) == len(item_uids) and np.isfinite(scores).all(), "invalid rank axis")
    tie_order = sorted(range(len(item_uids)), key=lambda i: selection_key(purpose, domain, item_uids[i]))
    order = np.asarray(tie_order)[np.argsort(-scores[:, tie_order], axis=1, kind="stable")]
    ranks = np.empty(order.shape, dtype=np.int32)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, len(item_uids) + 1), order.shape), axis=1)
    return ranks


def percentile_interval(draws):
    return np.quantile(draws, [0.025, 0.975]).tolist()


def cluster_draws(groups, domain, analysis, draws=DRAWS):
    _, dense_groups = np.unique(groups, return_inverse=True)
    n_groups = len(np.unique(dense_groups))
    seed = subseed(BOOT_SEED, domain, analysis)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=draws)
    return dense_groups, counts, seed


def rank_summary(ranks, gallery_size):
    summary = rank_metrics(ranks)
    return {"n_queries": summary["n"], "n_gallery": gallery_size, "MedR": summary["MedR"],
            "hits": {f"R@{k}": int(np.count_nonzero(np.asarray(ranks) <= k)) for k in KS},
            **{f"R@{k}": summary[f"R@{k}"] for k in KS}}


def summarize_methods(positives, groups, gallery_size, domain, analysis):
    summaries = {method: rank_summary(ranks, gallery_size) for method, ranks in positives.items()}
    if len(groups) == 0:
        return summaries
    rank_array = np.column_stack(list(positives.values()))
    dense, draws, seed = cluster_draws(groups, domain, analysis)
    boot = bootstrap_statistics(rank_array, dense, draws)
    for i, summary in enumerate(summaries.values()):
        summary["ci95"] = {metric: percentile_interval(boot[metric][:, i]) for metric in (*[f"R@{k}" for k in KS], "MedR")}
        summary["bootstrap_seed"] = seed
        summary["bootstrap_draws"] = len(draws)
        summary["n_groups"] = int(draws.shape[1])
    return summaries


def paired_ranks(before, after, groups, domain, analysis):
    before, after = valid_ranks(before), valid_ranks(after)
    require(before.shape == after.shape == (len(groups),), "invalid paired rank axes")
    if not len(groups):
        return {"n_queries": 0, "status": "empty_subset"}
    before_hits, after_hits = before <= 20, after <= 20
    delta = after_hits.astype(np.int8) - before_hits.astype(np.int8)
    dense, draws, seed = cluster_draws(groups, domain, analysis)
    n_groups = draws.shape[1]
    delta_draws = draws @ np.bincount(dense, weights=delta, minlength=n_groups)
    denominators = draws @ np.bincount(dense, minlength=n_groups)
    return {"n_queries": len(groups), "n_groups": n_groups, "hits_before": int(before_hits.sum()),
            "hits_after": int(after_hits.sum()), "added_hits": int(np.count_nonzero(delta > 0)),
            "lost_hits": int(np.count_nonzero(delta < 0)), "net_pp": float(100 * delta.mean()),
            "ci95_pp": percentile_interval(100 * delta_draws / denominators), "bootstrap_seed": seed,
            "bootstrap_draws": len(draws), "median_after_minus_before_rank": float(np.median(after - before)),
            "median_absolute_rank_change": float(np.median(np.abs(after - before))),
            "n_rank_changed": int(np.count_nonzero(after != before))}


def permutation_test(rank_matrices, positives, domain, strata=None, draws=DRAWS, k=20):
    positives = np.asarray(positives)
    n = len(positives)
    require(n > 0 and draws > 0 and k > 0 and rank_matrices and all(ranks.shape == (n, n) for ranks in rank_matrices.values())
            and np.array_equal(np.sort(positives), np.arange(n)), "permutation requires a bijective main gallery")
    analysis = "permutation_unrestricted" if strata is None else "permutation_fields"
    labels = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    require(labels.shape == (n,), "invalid permutation strata")
    blocks = [np.where(labels == label)[0] for label in np.unique(labels)]
    seed = subseed(PERM_SEED, domain, analysis)
    rng = np.random.Generator(np.random.PCG64(seed))
    row = np.arange(n)
    hit_matrices = np.stack([ranks <= k for ranks in rank_matrices.values()])
    observed = hit_matrices[:, row, positives].sum(axis=1)
    null = np.empty((draws, len(rank_matrices)), dtype=np.int32)
    assigned = np.empty_like(positives)
    for d in range(draws):
        for block in blocks:
            assigned[block] = rng.permutation(positives[block])
        null[d] = hit_matrices[:, row, assigned].sum(axis=1)
    methods = {}
    for i, method in enumerate(rank_matrices):
        exceed = int(np.count_nonzero(null[:, i] >= observed[i]))
        methods[method] = {"hits": int(observed[i]), "permutations_ge_observed": exceed,
                           "p": (1 + exceed) / (draws + 1), "null_mean_r": float(null[:, i].mean() / n),
                           "null_central95_r": percentile_interval(null[:, i] / n)}
    if strata is None:
        running = 0.0
        for i, method in enumerate(sorted(methods, key=lambda m: (methods[m]["p"], m))):
            running = max(running, min(1.0, (len(methods) - i) * methods[method]["p"]))
            methods[method]["holm_p"] = running
    return {"analysis": analysis, "seed": seed, "draws": draws, "k": k, "alpha": 0.05 if strata is None else None,
            "stratum_sizes": [len(block) for block in blocks], "methods": methods}


def binomial_tail_boundary(n, start, probability):
    """Invert an increasing binomial upper tail by bisection."""
    terms = [(i, math.comb(n, i)) for i in range(start, n + 1)]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        upper_tail = math.fsum(coefficient * mid**i * (1 - mid)**(n - i) for i, coefficient in terms)
        if upper_tail < probability:
            lo = mid
        else:
            hi = mid
    return (hi + lo) / 2


def clopper_pearson_975(hits, n):
    # Complement the failures' lower bound to avoid an independent CDF implementation.
    alpha_half = 0.0125
    lower = binomial_tail_boundary(n, hits, alpha_half) if hits else 0.0
    upper = 1 - binomial_tail_boundary(n, n - hits, alpha_half) if hits < n else 1.0
    return lower, upper


def encoding_loss(reencoded, official):
    reencoded, official = valid_ranks(reencoded), valid_ranks(official)
    require(reencoded.ndim == 1 and reencoded.shape == official.shape and len(reencoded) > 0,
            "invalid paired encoding rank axes")
    u = int(np.count_nonzero((official <= 20) & (reencoded > 20)))
    v = int(np.count_nonzero((official > 20) & (reencoded <= 20)))
    n = len(reencoded)
    u_ci, v_ci = clopper_pearson_975(u, n), clopper_pearson_975(v, n)
    low, high = 100 * (u_ci[0] - v_ci[1]), 100 * (u_ci[1] - v_ci[0])
    if high <= ENCODING_TOLERANCE_PP:
        decision, status = "excludes_net_loss_above_tolerance", "pass"
    elif low > ENCODING_TOLERANCE_PP:
        decision, status = "encoding_degradation", "fail"
    else:
        decision, status = "unresolved", "inconclusive"
    return {"n_queries": n, "k": 20, "official_only": u, "reencoded_only": v,
            "official_hits": int(np.count_nonzero(official <= 20)),
            "reencoded_hits": int(np.count_nonzero(reencoded <= 20)), "net_loss_pp": 100 * (u - v) / n,
            "ci95_pp": [low, high], "official_only_marginal_ci97_5": list(u_ci),
            "reencoded_only_marginal_ci97_5": list(v_ci), "tolerance_pp": ENCODING_TOLERANCE_PP,
            "decision": decision, "status": status,
            "median_official_minus_reencoded_rank": float(np.median(official - reencoded)),
            "n_rank_changed": int(np.count_nonzero(official != reencoded))}


def patch_columns(scores, background, gallery_uids, alternative_scores, alternative_background, replacement_uids):
    require(len(set(replacement_uids)) == len(replacement_uids) and set(replacement_uids) <= set(gallery_uids),
            "invalid alternative CAD columns")
    positions = [gallery_uids.index(uid) for uid in replacement_uids]
    require(alternative_scores.shape == (scores.shape[0], len(positions))
            and alternative_background.shape == (N_BACKGROUND, len(positions)), "invalid alternative matrices")
    patched_scores, patched_background = scores.copy(), background.copy()
    patched_scores[:, positions] = alternative_scores
    patched_background[:, positions] = alternative_background
    return patched_scores, patched_background


def difficult_pairs(gallery, raw, corrected, raw_ranks, corrected_ranks, domain, fields_quartiles):
    buckets = defaultdict(list)
    missing = 0
    for i, candidate in enumerate(gallery):
        features = candidate.get("program_features") or {}
        flags = tuple(features.get(key) for key in PAIR_FLAGS)
        if any(type(flag) is not bool for flag in flags):
            missing += 1
            continue
        buckets[flags, candidate["n_valid"]].append(i)

    def make_pairs(groups):
        pairs, left = [], {}
        for key, indices in sorted(groups.items()):
            order = sorted(indices, key=lambda i: selection_key("pair", domain, gallery[i]["group_id"]))
            pairs.extend(zip(order[::2], order[1::2]))
            if len(order) % 2:
                left[key] = order[-1]
        return pairs, left

    exact, leftovers = make_pairs(buckets)
    relaxed_buckets = defaultdict(list)
    if len(fields_quartiles):
        for (flags, n_valid), i in leftovers.items():
            relaxed_buckets[flags, int(np.searchsorted(fields_quartiles, n_valid, side="right"))].append(i)
    relaxed, _ = make_pairs(relaxed_buckets)

    def summarize(pairs, analysis):
        rows = []
        for i, j in pairs:
            # Pair query rows are guaranteed to match column order.
            d = float((raw[i, i] - raw[i, j]) + (raw[j, j] - raw[j, i]))
            adjusted_d = float((corrected[i, i] - corrected[i, j]) + (corrected[j, j] - corrected[j, i]))
            require(abs(d - adjusted_d) <= 1e-8, "candidate offsets failed to cancel in D")
            rows.append({"uid_i": gallery[i]["uid"], "uid_j": gallery[j]["uid"], "D": d, "D_corrected": adjusted_d,
                         "pair_hit": 1 if d > 1e-8 else 0 if d < -1e-8 else 0.5,
                         "both_correct": {
                             method: bool(ranks[i, i] < ranks[i, j] and ranks[j, j] < ranks[j, i])
                             for method, ranks in zip(METHODS, (raw_ranks, corrected_ranks))}})
        result = {"n_pairs": len(rows), "pairs": rows}
        if rows:
            values = np.array([[r["pair_hit"], *(r["both_correct"][m] for m in METHODS)] for r in rows])
            _, draws, seed = cluster_draws(np.arange(len(rows)), domain, analysis)
            sampled = (draws @ values) / len(rows)
            result.update({"bootstrap_seed": seed, "bootstrap_draws": len(draws), "rates": {
                name: {"mean": float(values[:, i].mean()), "ci95": percentile_interval(sampled[:, i])}
                for i, name in enumerate(("matched_assignment", "both_correct_S0", "both_correct_S48"))}})
        return result

    return {"n_gallery": len(gallery), "n_missing_flags": missing, "n_unpaired_with_flags":
            len(gallery) - missing - 2 * (len(exact) + len(relaxed)),
            "exact": summarize(exact, "pairs_exact"), "relaxed": summarize(relaxed, "pairs_relaxed"),
            "relaxed_status": "available" if len(fields_quartiles) else "pending_source_field_quartiles"}


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f"{path.name}: JSONL records must be objects")
    return rows


def uids_for(rows):
    uids = [row.get("uid") for row in rows]
    require(all(isinstance(uid, str) and uid for uid in uids), "every frozen record needs a nonempty UID")
    return uids


def frozen_gallery(path):
    rows = read_rows(path)
    uids = uids_for(rows)
    groups = [r.get("group_id") for r in rows]
    require(len(rows) in (256, 512) and len(set(uids)) == len(rows) and len(set(groups)) == len(rows)
            and all(isinstance(group, str) and group for group in groups),
            f"{path.name}: formal gallery requires 256 or 512 unique UIDs and independent groups")
    return rows


def load_scores(path, queries, expected_cads, background=None):
    expected_queries = uids_for(queries)
    expected_variants = [row["variant"] for row in queries] if queries and "variant" in queries[0] else None
    with np.load(path, allow_pickle=False) as arm:
        for field, expected in (("cad_uids", expected_cads), ("query_uids", expected_queries)):
            axis = arm[field]
            label = "CAD axis" if field == "cad_uids" else "query axis"
            require(axis.dtype.kind == "U" and axis.ndim == 1 and axis.tolist() == list(expected),
                    f"{path.name}: {label} differs from the frozen axis")
        if expected_variants is not None:
            require("query_variants" in arm.files, f"{path.name}: rewrite rows require variant axis")
            versions = arm["query_variants"]
            require(versions.dtype.kind == "U" and versions.ndim == 1 and versions.tolist() == expected_variants,
                    f"{path.name}: query variants differ from the frozen axis")
        scores = arm["scores"]
        bg = arm["background"] if "background" in arm.files else background
    require(scores.dtype == np.float64 and scores.shape == (len(queries), len(expected_cads))
            and np.isfinite(scores).all(), f"{path.name}: scores must be finite float64 on the frozen axes")
    require(bg is not None and bg.dtype == np.float64 and bg.shape == (N_BACKGROUND, len(expected_cads))
            and np.isfinite(bg).all(), f"{path.name}: requires all 48 frozen background rows as finite float64")
    if background is not None:
        require(np.array_equal(bg, background), f"{path.name}: background differs from the same-encoding gallery")
    return scores, bg


def query_file(input_dir, name, fallback=None):
    path = input_dir / f"queries_{name}.jsonl"
    if path.is_file():
        return read_rows(path)
    require(fallback is not None, f"missing frozen {path.name}")
    return fallback


def verify_frozen_files(input_dir, freeze, names):
    hashes = freeze.get("score_inputs_sha256")
    if hashes is None:
        return
    for name in names:
        path = input_dir / name
        if name in hashes or path.is_file():
            require(name in hashes and path.is_file() and sha256(path) == hashes[name],
                    f"{name}: evaluation input differs from the frozen fingerprint")


def validate_queries(queries, gallery, *, complete=False):
    by_uid = {row["uid"]: row for row in gallery}
    query_uids = uids_for(queries)
    if complete:
        require(query_uids == list(by_uid), "main query UIDs must match frozen gallery order")
    require(len(query_uids) == len(set(query_uids)), "duplicate main/subset query UID")
    for query in queries:
        uid = query["uid"]
        require(uid in by_uid and query.get("group_id") == by_uid[uid]["group_id"], f"unlinked query group for {uid}")
        text = query.get("text")
        require(isinstance(text, str) and bool(text.strip()), f"empty query for {uid}")


def encoding_check(input_dir, scores_dir, freeze, gallery, queries_b, raw_b, background_b):
    """Compare exact same source query UIDs in full B with both CAD encodings."""
    gallery_uids = uids_for(gallery)
    selected = {row["uid"] for row in sorted(gallery, key=lambda r: selection_key("encoding-check", "deepcad", r["group_id"]))[:256]}
    declared = freeze.get("b_official_uids")
    original_by_uid = {row["uid"]: row for row in queries_b}
    fallback_order = [uid for uid in gallery_uids if uid in selected] if declared is None else declared
    require(len(fallback_order) == 256 and set(fallback_order) == selected, "B official selection disagrees with frozen protocol")
    queries = query_file(input_dir, "b_official", [original_by_uid[uid] for uid in fallback_order])
    validate_queries(queries, gallery)
    query_uids = uids_for(queries)
    require(len(query_uids) == 256 and set(query_uids) == selected, "B encoding gate requires the 256 preselected source groups")
    if declared is not None:
        require(query_uids == declared, "B official manifests disagree on query order")
    for query in queries:
        require(query["text"] == original_by_uid[query["uid"]]["text"], "official gate queries must use identical B text")
    official_raw, official_bg = load_scores(scores_dir / "b_official.npz", queries, gallery_uids)
    pos = np.asarray([gallery_uids.index(uid) for uid in query_uids])
    reencoded = score_readouts(raw_b[pos], background_b)
    official = score_readouts(official_raw, official_bg)
    results, evidence = {}, []
    for method in METHODS:
        before = rank_matrix(reencoded[method], gallery_uids, "deepcad")[np.arange(len(queries)), pos]
        after = rank_matrix(official[method], gallery_uids, "deepcad")[np.arange(len(queries)), pos]
        results[method] = encoding_loss(before, after)
        results[method]["n_gallery"] = len(gallery)
        if len(gallery) == len(queries):
            results[method]["finite_gallery_net_loss_pp"] = results[method]["net_loss_pp"]
        evidence.extend(rank_comparison_rows("B_official", method, queries, before, after))
    statuses = {result["status"] for result in results.values()}
    status = "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"
    return {"status": status, "methods": results, "query_uids": query_uids}, evidence


def rank_comparison_rows(analysis, method, queries, before, after):
    return [
        {"analysis": analysis, "method": method, "query_uid": query["uid"], "group_id": query["group_id"],
         "variant": query.get("variant"), "rank_before": int(x), "rank_after": int(y),
         "hit_before": bool(x <= 20), "hit_after": bool(y <= 20), "top20_flip": bool((x <= 20) != (y <= 20))}
        for query, x, y in zip(queries, before, after)
    ]


def source_analysis(input_dir, scores_dir):
    freeze = json.loads((input_dir / "freeze.json").read_text(encoding="utf-8-sig"))
    verify_frozen_files(input_dir, freeze, ("gallery_deepcad.jsonl", "queries_b.jsonl", "queries_b_official.jsonl"))
    gallery = frozen_gallery(input_dir / "gallery_deepcad.jsonl")
    require(freeze.get("gallery_size", len(gallery)) == len(gallery), "B gallery size differs from freeze")
    queries = query_file(input_dir, "b", gallery)
    validate_queries(queries, gallery, complete=True)
    for query, row in zip(queries, gallery):
        if "text" in row:
            require(query["text"] == row["text"], "B main query text disagrees with frozen gallery record")
    raw, bg = load_scores(scores_dir / "b.npz", queries, uids_for(gallery))
    gate, evidence = encoding_check(input_dir, scores_dir, freeze, gallery, queries, raw, bg)
    return freeze, gallery, queries, raw, bg, gate, evidence


def numeric_rules():
    """Candidate rules for source validation before scoring target text."""
    return {
        "bin_width": {"normalized": 0.001, "ratio": 0.01, "mm": 0.1, "degree": 0.5},
        "bin_rule": "floor(value / width + 0.5), bins anchored at zero",
        "millimeters": {"mm": 1, "cm": 10, "m": 1000, "in": 25.4},
        "degrees": {"degree": 1, "rad": 180 / math.pi},
        "dimensions": {
            "depth": ["extrusion depth", "cut depth", "depth"], "diameter": ["diameter", "dia"],
            "radius": ["radius"], "width": ["width"], "height": ["height"], "length": ["length"],
            "thickness": ["thickness"], "angle": ["angle"], "spacing": ["spacing", "separation"],
            "offset": ["offset"], "ratio": ["proportion", "ratio"],
        },
        "count_terms": {"hole": ["hole", "holes"], "slot": ["slot", "slots"], "part": ["part", "parts"],
                        "rib": ["rib", "ribs"], "boss": ["boss", "bosses"], "body": ["body", "bodies"],
                        "extrusion": ["extrusion", "extrusions"], "cut": ["cut", "cuts"],
                        "circle": ["circle", "circles"], "fillet": ["fillet", "fillets"]},
        "count_words": {word: i for i, word in enumerate(
            ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"))},
    }


VALUE_SPACES = ("count", "ratio", "normalized", "physical", "literal")
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
UNIT = (
    r"millimeters?|millimetres?|centimeters?|centimetres?|meters?|metres?|"
    r"inches|inch|degrees?|radians?|mm|cm|m|in(?=\s*(?:[.,;:)\]]|$))|rad|deg"
)
UNIT_ALIASES = {"millimeter": "mm", "millimetre": "mm", "centimeter": "cm", "centimetre": "cm",
                "meter": "m", "metre": "m", "inch": "in", "inche": "in", "deg": "degree",
                "radian": "rad", "°": "degree"}


def candidate_numeric_values(entries, rules):
    result = {space: set() for space in VALUE_SPACES}
    for entry in entries:
        space, attribute, value = entry["value_space"], entry["attribute"], entry["value"]
        if space == "scale_unknown":
            continue
        require(space in result and isinstance(attribute, str) and isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value), "invalid candidate numeric observation")
        if space == "count":
            require(value == int(value) and value >= 0, "candidate count must be a nonnegative integer")
            result[space].add((attribute, "integer", int(value)))
            result["literal"].add(float(value))
            continue
        if space != "physical":
            result["literal"].add(float(value))
        if space == "literal":
            continue
        unit = space
        if space == "physical":
            unit = entry.get("unit")
            if unit in rules["millimeters"]:
                value, unit = value * rules["millimeters"][unit], "mm"
            elif unit in rules["degrees"]:
                value, unit = value * rules["degrees"][unit], "degree"
            else:
                raise ValueError(f"numeric observation has unknown physical unit {unit!r}")
        bucket = math.floor(value / rules["bin_width"][unit] + 0.5)
        result[space].add((attribute, unit, bucket))
    return result


def query_numeric_values(text, rules):
    text = text.casefold().replace("−", "-")
    observations = []
    for attribute, terms in rules["count_terms"].items():
        counts = "|".join(rules["count_words"])
        for match in re.finditer(rf"(?<![\w.])(?P<n>\d+|{counts})\s+(?:{'|'.join(terms)})\b", text):
            token = match["n"]
            value = int(token) if token.isdigit() else rules["count_words"][token]
            observations.append({"attribute": attribute, "value_space": "count", "value": value})
    for attribute, terms in rules["dimensions"].items():
        term = "|".join(re.escape(t) for t in terms)
        value_expression = rf"(?P<n>{NUMBER})(?:\s*:\s*(?P<den>{NUMBER})|[ ]*(?P<unit>°|(?:{UNIT})\b)|(?P<pct>%))?"
        reverse_expression = rf"(?P<n>{NUMBER})(?:[ ]*(?P<unit>°|(?:{UNIT})\b))?\s*(?:-\s*)?(?:{term})\b"
        forward = re.finditer(rf"\b(?:{term})\b\s*(?:(?:of|is|equals?)\s+|[=:]\s*)?{value_expression}(?!\w|\.\d)", text)
        reverse = re.finditer(rf"(?<![\w.]){reverse_expression}", text)
        for match in (*forward, *reverse):
            fields = match.groupdict()
            value = float(fields["n"])
            unit = fields.get("unit")
            if unit is not None:
                unit = UNIT_ALIASES.get(unit.rstrip("s"), unit.rstrip("s"))
                observation = {"attribute": attribute, "value_space": "physical", "unit": unit, "value": value}
            elif fields.get("den") is not None:
                denominator = float(fields["den"])
                if denominator == 0:
                    continue
                observation = {"attribute": attribute, "value_space": "ratio", "value": value / denominator}
            elif fields.get("pct") or attribute == "ratio":
                observation = {"attribute": attribute, "value_space": "ratio", "value": value / 100 if fields.get("pct") else value}
            else:
                observation = {"attribute": attribute, "value_space": "normalized", "value": value}
            observations.append(observation)
    result = candidate_numeric_values(observations, rules)
    result["literal"] = {float(n) for n in re.findall(rf"(?<![\w.]){NUMBER}(?!\d)", text)}
    return result


def source_reference(input_dir):
    path = input_dir / "baseline_reference.json"
    if not path.is_file():
        return {"status": "pending_source_validation"}
    ref = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {"status": "available", "sha256": sha256(path), "samples": {}, "quartiles": {}}
    for name in ("token_counts", "n_valid", "n_extrusions"):
        values = ref.get(name)
        if values is None:
            continue
        require(isinstance(values, list) and values and all(type(v) is int and v > 0 for v in values),
                f"baseline_reference.json: invalid source sample for {name}")
        ordered = np.sort(values)
        result["samples"][name] = ordered
        result["quartiles"][name] = np.quantile(ordered, [0.25, 0.50, 0.75])
    if "numeric_rules" in ref:
        require(ref["numeric_rules"] == numeric_rules(), "source numeric parser rules do not match this evaluator")
        result["numeric_rules"] = ref["numeric_rules"]
    return result


def quantity_array(rows, field):
    values = [row.get(field) for row in rows]
    return np.array(values, dtype=np.int32) if all(type(value) is int and value > 0 for value in values) else None


def baseline_metrics(scores, comparable, gallery_uids, positive, groups, model_ranks, domain, analysis):
    target_ranks = rank_matrix(scores, gallery_uids, domain)[np.arange(len(positive)), positive]
    result = rank_summary(target_ranks, len(gallery_uids))
    result.update({"comparable_queries": int(comparable.sum()),
                   "n_all_candidates_equal": int(np.count_nonzero(np.ptp(scores, axis=1) == 0)),
                   "n_ground_truth_tied": int(np.count_nonzero(
                       np.count_nonzero(scores == scores[np.arange(len(positive)), positive, None], axis=1) > 1))})
    if comparable.any():
        subset = {analysis: target_ranks[comparable], **{m: ranks[comparable] for m, ranks in model_ranks.items()}}
        result["comparable_subset"] = summarize_methods(subset, groups[comparable], len(gallery_uids), domain, analysis)
    return result


def surface_baselines(queries, gallery, positive, model_ranks, domain, reference):
    samples = reference.get("samples", {})
    lengths = quantity_array(queries, "token_count")
    fields = quantity_array(gallery, "n_valid")
    uids, groups = uids_for(gallery), np.asarray([q["group_id"] for q in queries])
    result = {"length": {"status": "pending_source_validation_or_length_metadata"},
              "numeric": {"status": "pending_source_validation_or_candidate_observations"}}
    if lengths is not None and fields is not None and {"token_counts", "n_valid"} <= samples.keys():
        q_pct = np.searchsorted(samples["token_counts"], lengths, side="right") / len(samples["token_counts"])
        c_pct = np.searchsorted(samples["n_valid"], fields, side="right") / len(samples["n_valid"])
        scores = -np.abs(q_pct[:, None] - c_pct)
        result["length"] = {"status": "available", **baseline_metrics(
            scores, np.ones(len(queries), dtype=bool), uids, positive, groups, model_ranks, domain, "length_percentile")}
    rules = reference.get("numeric_rules")
    entries = [c.get("numeric_values", (c.get("program_features") or {}).get("numeric_values")) for c in gallery]
    if rules is not None and any(value is not None for value in entries):
        q_terms = [query_numeric_values(row["text"], rules) for row in queries]
        c_terms = [candidate_numeric_values(entry or [], rules) for entry in entries]
        result["numeric"] = {"status": "available", "n_candidates_with_observed_values_record": sum(e is not None for e in entries),
                             "rules": rules, "spaces": {}}
        for space in VALUE_SPACES:
            q_sets = [terms[space] for terms in q_terms]
            c_sets = [terms[space] for terms in c_terms]
            scores = np.zeros((len(queries), len(gallery)))
            both_possible = np.zeros_like(scores, dtype=bool)
            empty_union_pairs = 0
            for i, q_set in enumerate(q_sets):
                for j, c_set in enumerate(c_sets):
                    union_size = len(q_set | c_set)
                    if union_size:
                        scores[i, j] = len(q_set & c_set) / union_size
                    else:
                        empty_union_pairs += 1
                    if space == "literal":
                        both_possible[i, j] = bool(q_set and c_set)
                    else:
                        both_possible[i, j] = any((a[:2] == b[:2]) for a in q_set for b in c_set)
            # A comparable query must have a common attribute/unit with its registered target.
            comparable = both_possible[np.arange(len(positive)), positive]
            result["numeric"]["spaces"][space] = {
                **baseline_metrics(scores, comparable, uids, positive, groups, model_ranks, domain, f"numeric_{space}"),
                "n_empty_union_pairs": empty_union_pairs, "n_cad_with_values": sum(bool(s) for s in c_sets),
                "n_query_with_values": sum(bool(s) for s in q_sets), "n_comparable_pairs": int(both_possible.sum()),
            }
    return result


def duplication(queries):
    if all(isinstance(row.get("token_ids"), list) and all(type(i) is int for i in row["token_ids"]) for row in queries):
        keys = [tuple(row["token_ids"]) for row in queries]
        basis = "encoder_token_ids"
    elif all(isinstance(row.get("token_ids_sha256"), str) for row in queries):
        keys = [row["token_ids_sha256"] for row in queries]
        basis = "encoder_token_ids_sha256"
    else:
        keys = [row["text"] for row in queries]
        basis = "exact_text_proxy_pending_encoder_token_ids"
    counts = list(Counter(keys).values())
    ceilings = duplicate_ceilings(counts)
    return {"basis": basis, "n_distinct_inputs": len(counts), "n_duplicated_input_groups": sum(n > 1 for n in counts),
            "n_queries_in_duplicated_groups": sum(n for n in counts if n > 1),
            "ceiling": {f"R@{k}": ceilings[f"R@{k}"] for k in KS}}


def reverse_ranking(raw, corrected, query_uids, domain):
    native = rank_matrix(raw.T, query_uids, domain, "tie-query")
    direct = rank_matrix(corrected.T, query_uids, domain, "tie-query")
    max_spread = 0.0
    n_inversions = 0
    changed = np.flatnonzero(np.any(native != direct, axis=1))
    for c in changed:
        left, right = np.triu_indices(len(query_uids), 1)
        flipped = (native[c, left] < native[c, right]) != (direct[c, left] < direct[c, right])
        spread = np.abs(raw[left[flipped], c] - raw[right[flipped], c])
        max_spread = max(max_spread, float(spread.max(initial=0)))
        n_inversions += len(spread)
        high = np.max(np.abs(np.concatenate((raw[:, c], corrected[:, c]))))
        require((spread <= 8 * np.spacing(high)).all(), "reverse ranking changed beyond floating-point near ties")
    return native, {"n_columns_with_floating_tie_reordering": int(len(changed)), "n_inverted_query_pairs": n_inversions,
                    "max_native_spread_in_reordered_pair": max_spread,
                    "ranking_source": "shared_S0_column_order"}


def exclude_suspected(queries, gallery, raw, corrected, domain):
    require(raw.shape == corrected.shape == (len(queries), len(gallery))
            and np.isfinite(raw).all() and np.isfinite(corrected).all(), "invalid near-training sensitivity axes or scores")
    validate_queries(queries, gallery, complete=True)
    states = [row.get("source_train_near_status") for row in gallery]
    require(set(states) <= {None, "not_screened", "no_corroboration", "suspected"},
            "formal gallery contains a confirmed training neighbor or unknown screening status")
    missing = states.count(None) + states.count("not_screened")
    keep = np.array([state != "suspected" for state in states])
    n_retained = int(keep.sum())
    result = {"n_original_queries": len(queries), "n_original_gallery": len(gallery), "n_suspected_excluded": states.count("suspected"),
              "n_status_missing_or_unscreened_retained": missing, "n_retained_queries": n_retained,
              "n_retained_gallery": n_retained, "methods": {}}
    if missing == len(gallery):
        return {**result, "status": "pending_screening"}
    if n_retained == 0:
        return {**result, "status": "empty_after_exclusion"}
    uids = np.asarray(uids_for(gallery))[keep].tolist()
    groups = np.asarray([row["group_id"] for row in queries])[keep]
    ranks = {m: np.diag(rank_matrix(scores[np.ix_(keep, keep)], uids, domain))
             for m, scores in zip(METHODS, (raw, corrected))}
    result["methods"] = summarize_methods(ranks, groups, n_retained, domain, "exclude_suspected")
    result["uniform_random_R@20"] = min(20, n_retained) / n_retained
    return {**result, "status": "available" if result["n_suspected_excluded"] else "no_suspected_candidates"}


def domain_analysis(queries, gallery, raw, background, domain, reference, arm):
    gallery_uids = uids_for(gallery)
    query_uids = uids_for(queries)
    groups = [q["group_id"] for q in queries]
    positive = np.array([gallery_uids.index(uid) for uid in query_uids], dtype=np.int32)
    readouts = score_readouts(raw, background)
    matrices = {m: rank_matrix(readouts[m], gallery_uids, domain) for m in (*METHODS, "half_0_23", "half_24_47")}
    ranks = {m: matrix[np.arange(len(positive)), positive] for m, matrix in matrices.items()}
    summary = {"n_queries": len(queries), "n_gallery": len(gallery), "uniform_random_R@20": min(20, len(gallery)) / len(gallery),
               "methods": summarize_methods(ranks, groups, len(gallery), domain, f"{arm}_forward")}
    summary["top20_S48_minus_S0"] = paired_ranks(*[ranks[m] for m in METHODS], groups, domain, f"{arm}_correction")
    summary["halfsets"] = {m: paired_ranks(ranks["S48"], ranks[m], groups, domain, f"{arm}_{m}")
                           for m in ("half_0_23", "half_24_47")}
    summary["duplicate_inputs"] = duplication(queries)
    summary["popular_candidates"] = {}
    for method in METHODS:
        counts = (matrices[method] <= 20).sum(axis=0)
        chosen = sorted(range(len(gallery)), key=lambda i: (-counts[i], selection_key("tie-cad", domain, gallery_uids[i])))[:10]
        summary["popular_candidates"][method] = [
            {"uid": gallery_uids[i], "top20_appearances": int(counts[i]), "n_valid": gallery[i].get("n_valid"),
             "background_offset": float(readouts["background_offset"][i]),
             "registered_hits": int(np.count_nonzero((positive == i) & (ranks[method] <= 20)))} for i in chosen]
    evidence = rank_comparison_rows(f"{arm}_S48_minus_S0", "S48_vs_S0", queries, ranks["S0"], ranks["S48"])
    for method in ("half_0_23", "half_24_47"):
        evidence.extend(rank_comparison_rows(f"{arm}_halfsets", method, queries, ranks["S48"], ranks[method]))
    if arm in ("B", "C"):
        reverse, reverse_check = reverse_ranking(raw, readouts["S48"], query_uids, domain)
        reverse_ranks = reverse[positive, np.arange(len(positive))]
        summary["reverse"] = {**reverse_check, **summarize_methods({"both": reverse_ranks}, groups, len(gallery), domain,
                                                                   f"{arm}_reverse")["both"]}
        evidence.extend(rank_comparison_rows(f"{arm}_reverse", "shared_S0_S48", queries, reverse_ranks, reverse_ranks))
        summary["difficult_pairs"] = difficult_pairs(
            gallery, raw, readouts["S48"], matrices["S0"], matrices["S48"], domain, reference.get("quartiles", {}).get("n_valid", []))
        summary["surface_baselines"] = surface_baselines(queries, gallery, positive, {m: ranks[m] for m in METHODS}, domain, reference)
        summary["exclude_suspected"] = exclude_suspected(queries, gallery, raw, readouts["S48"], domain)
    elif arm == "A":
        exposures = [gallery[i].get("l0_text_train_match_count") for i in positive]
        tags = []
        for value in exposures:
            if type(value) is not int or value < 0:
                tags.append("pending_text_screening")
            else:
                tags.append("seen_exact_train_text" if value else "no_exact_train_text_match")
        summary["training_L0_text_overlap"] = {
            tag: {m: rank_summary(ranks[m][np.equal(tags, tag)], len(gallery)) for m in METHODS} for tag in sorted(set(tags))
        }
    return summary, ranks, matrices, evidence


def compare_a_b(queries_a, queries_b, ranks_a, ranks_b, gallery_size):
    lookup = {query["uid"]: i for i, query in enumerate(queries_b)}
    positions = np.array([lookup[query["uid"]] for query in queries_a], dtype=int)
    groups = [q["group_id"] for q in queries_a]
    texts = [q["text"] for q in queries_a]
    text_groups, text_draws, seed = cluster_draws(texts, "deepcad", "AB_equal_L0_text")
    group_sizes = np.bincount(text_groups)
    methods, evidence = {}, []
    for method in METHODS:
        before, after = ranks_a[method], ranks_b[method][positions]
        equal_text = {}
        for k in KS:
            a_group = np.bincount(text_groups, weights=(before <= k)) / group_sizes
            b_group = np.bincount(text_groups, weights=(after <= k)) / group_sizes
            differences = text_draws @ (b_group - a_group) / len(group_sizes)
            equal_text[f"R@{k}"] = {"A": float(a_group.mean()), "B": float(b_group.mean()),
                                    "B_minus_A_pp": float(100 * (b_group - a_group).mean()),
                                    "ci95_pp": percentile_interval(100 * differences)}
        methods[method] = {"A": rank_summary(before, gallery_size), "B": rank_summary(after, gallery_size),
                           "paired_B_minus_A": paired_ranks(before, after, groups, "deepcad", "AB_expert"),
                           "equal_L0_text": equal_text}
        evidence.extend(rank_comparison_rows("B_minus_A", method, queries_a, before, after))
    return {"n_common_queries": len(queries_a), "n_distinct_L0_texts": len(group_sizes), "n_gallery": gallery_size,
            "equal_L0_text_bootstrap_seed": seed, "methods": methods}, evidence


def source_bins(queries, gallery, reference):
    fields = {"token_counts": quantity_array(queries, "token_count"), "n_valid": quantity_array(gallery, "n_valid"),
              "n_extrusions": quantity_array(gallery, "n_extrusions")}
    return {name: np.searchsorted(boundaries, fields[name], side="right")
            for name, boundaries in reference.get("quartiles", {}).items() if fields[name] is not None}


def compare_b_c(ranks_b, ranks_c, b_bins, c_bins, reference):
    result = {"status": "available", "methods": {}}
    b_hits = np.column_stack([ranks_b[m] <= 20 for m in METHODS])
    c_hits = np.column_stack([ranks_c[m] <= 20 for m in METHODS])
    _, b_draws, b_seed = cluster_draws(np.arange(len(b_hits)), "deepcad", "BC_raw")
    _, c_draws, c_seed = cluster_draws(np.arange(len(c_hits)), "fusion360", "BC_raw")
    b_boot = (b_draws @ b_hits) / len(b_hits)
    c_boot = (c_draws @ c_hits) / len(c_hits)
    for i, method in enumerate(METHODS):
        result["methods"][method] = {
            "raw_C_minus_B_pp": float(100 * (c_hits[:, i].mean() - b_hits[:, i].mean())),
            "raw_ci95_pp": percentile_interval(100 * (c_boot[:, i] - b_boot[:, i])),
            "raw_bootstrap_seeds": {"deepcad": b_seed, "fusion360": c_seed},
        }
    common_binning = tuple(name for name in ("token_counts", "n_valid", "n_extrusions") if name in b_bins and name in c_bins)
    result["source_cutoffs"] = {key: reference["quartiles"][key].tolist() for key in common_binning}
    result["strata"] = {}
    for names in ([common_binning] if len(common_binning) > 1 else []) + [(name,) for name in common_binning]:
        b_labels = list(zip(*(b_bins[name].tolist() for name in names)))
        c_labels = list(zip(*(c_bins[name].tolist() for name in names)))
        observed = sorted(set(b_labels) | set(c_labels))
        raw_strata = [(label, np.array([i for i, group in enumerate(b_labels) if group == label], dtype=int),
                       np.array([i for i, group in enumerate(c_labels) if group == label], dtype=int)) for label in observed]
        common = [(bi, ci) for _, bi, ci in raw_strata if len(bi) and len(ci)]
        n_b_support = sum(len(bi) for bi, _ in common)
        n_c_support = sum(len(ci) for _, ci in common)
        name = "+".join(names)
        stratum_result = {"n_B_in_common_support": n_b_support, "n_C_in_common_support": n_c_support,
                          "n_shared_strata_with_fewer_than_five_per_domain": sum(min(len(bi), len(ci)) < 5 for bi, ci in common),
                          "stratum_rows": [{"labels": list(label), "n_B": len(bi), "n_C": len(ci),
                                            "methods": {m: {"B_hits": int(np.count_nonzero(ranks_b[m][bi] <= 20)),
                                                            "C_hits": int(np.count_nonzero(ranks_c[m][ci] <= 20))} for m in METHODS}}
                                           for label, bi, ci in raw_strata], "methods": {}}
        if common:
            b_seed = subseed(BOOT_SEED, "deepcad", f"BC_strata_{name}")
            c_seed = subseed(BOOT_SEED, "fusion360", f"BC_strata_{name}")
            rng_b, rng_c = (np.random.Generator(np.random.PCG64(seed)) for seed in (b_seed, c_seed))
            b_mean, c_mean = np.zeros(len(METHODS)), np.zeros(len(METHODS))
            deltas = np.zeros((DRAWS, len(METHODS)))
            for bi, ci in common:
                weight = len(bi) / n_b_support
                b_mean += weight * b_hits[bi].mean(axis=0)
                c_mean += weight * c_hits[ci].mean(axis=0)
                b_counts = rng_b.multinomial(len(bi), np.full(len(bi), 1 / len(bi)), size=DRAWS)
                c_counts = rng_c.multinomial(len(ci), np.full(len(ci), 1 / len(ci)), size=DRAWS)
                deltas += weight * ((c_counts @ c_hits[ci]) / len(ci) - (b_counts @ b_hits[bi]) / len(bi))
            for i, method in enumerate(METHODS):
                stratum_result["methods"][method] = {"B_weighted_R@20": float(b_mean[i]), "C_weighted_R@20": float(c_mean[i]),
                                                      "C_minus_B_pp": float(100 * (c_mean[i] - b_mean[i])),
                                                      "ci95_pp": percentile_interval(100 * deltas[:, i]),
                                                      "bootstrap_seeds": {"deepcad": b_seed, "fusion360": c_seed}}
        result["strata"][name] = stratum_result
    result["stratification_status"] = "available" if common_binning else "pending_source_validation_or_metadata"
    result["standardization_scope"] = "fixed B weights on observed shared strata, conditional independent group bootstrap within strata"
    return result


def target_encoding(input_dir, scores_dir, freeze, queries, gallery, raw, background, original_ranks):
    replacements = freeze.get("replacement_uids")
    if replacements is None:
        require(not (scores_dir / "c_alt.npz").exists(), "C encoding columns must be frozen before patching scores")
        return {"status": "pending_frozen_encoding_comparison"}, []
    require(isinstance(replacements, list) and len(replacements) <= 32, "invalid frozen C replacement column list")
    if not replacements:
        require(not (scores_dir / "c_alt.npz").exists(), "unexpected C alternative scores for zero replacement columns")
        return {"status": "no_frozen_replacement_columns", "n_replaced_columns": 0, "n_target_unreplaced": len(queries)}, []
    alt_gallery = read_rows(input_dir / "gallery_fusion_original.jsonl")
    alt_uids = uids_for(alt_gallery)
    require(len(alt_uids) == len(set(alt_uids)) and set(replacements) <= set(alt_uids),
            "replacement columns must map to frozen original-history candidates")
    alt_queries = query_file(input_dir, "c_alt", queries)
    require(uids_for(alt_queries) == uids_for(queries) and all(a["text"] == q["text"] for a, q in zip(alt_queries, queries)),
            "C alternative must use the unchanged full C query axis")
    alt_raw, alt_bg = load_scores(scores_dir / "c_alt.npz", alt_queries, replacements)
    cad_uids = uids_for(gallery)
    patched = patch_columns(raw, background, cad_uids, alt_raw, alt_bg, replacements)
    readouts = score_readouts(*patched)
    affected = np.isin(cad_uids, replacements)
    result = {"status": "available", "n_replaced_columns": len(replacements), "replacement_uids": replacements,
              "n_target_replaced": int(affected.sum()), "n_target_unreplaced": int((~affected).sum()), "methods": {}}
    evidence = []
    groups = np.array([q["group_id"] for q in queries])
    for method in METHODS:
        before = original_ranks[method]
        after = np.diag(rank_matrix(readouts[method], cad_uids, "fusion360"))
        partitions = {"all": np.ones_like(affected), "target_replaced": affected, "target_unreplaced": ~affected}
        result["methods"][method] = {partition: paired_ranks(
            before[mask], after[mask], groups[mask], "fusion360", f"C_native_{partition}") for partition, mask in partitions.items()}
        rows = rank_comparison_rows("C_native_columns", method, queries, before, after)
        evidence.extend({**row, "target_column_replaced": bool(selected)} for row, selected in zip(rows, affected))
    return result, evidence


def diagnostic_uids(freeze, gallery):
    chosen = sorted(gallery, key=lambda row: selection_key("diagnostic", "fusion360", row["group_id"]))[:48]
    expected = uids_for(chosen)
    ordered = freeze.get("diagnostic_uids", expected)
    require(isinstance(ordered, list) and len(ordered) == len(expected) and set(ordered) == set(expected),
            "frozen C diagnostic queries disagree with the 48 preselected groups")
    return ordered


def rewrite_analysis(input_dir, scores_dir, selected, queries, gallery, raw, background, original_ranks):
    orig = {q["uid"]: (i, q) for i, q in enumerate(queries)}
    rows = query_file(input_dir, "c_rewrite", [])
    ledger_path = input_dir / "rewrites.jsonl"
    ledger = read_rows(ledger_path) if ledger_path.is_file() else rows
    accepted = {"changed_valid", "no_edit"}
    by_key = {}
    for item in ledger:
        uid, variant = item["uid"], item["variant"]
        require(uid in selected and variant in VARIANTS and (uid, variant) not in by_key,
                "rewrite ledger includes duplicate or non-diagnostic query")
        status = item.get("status")
        require(status in REWRITE_STATUSES, "rewrite needs frozen review status")
        original = orig[uid][1]
        item = {**item, "group_id": item.get("group_id", original["group_id"])}
        require(item["group_id"] == original["group_id"], "rewrite ledger group does not match original query")
        if status == "no_edit":
            require(item.get("text", original["text"]) == original["text"], "no_edit must match original frozen text")
            item = {**original, **item}
        elif status == "changed_valid":
            require(isinstance(item.get("text"), str) and item["text"].strip() and item["text"] != original["text"],
                    "changed_valid needs distinct, nonempty frozen text")
        by_key[uid, variant] = item
    for row in rows:
        key = row["uid"], row["variant"]
        require(key in by_key and row.get("text") == by_key[key].get("text")
                and row.get("status", by_key[key]["status"]) == by_key[key]["status"],
                "rewrite scoring manifest disagrees with the frozen ledger")
    scored_rows = [by_key[q["uid"], q["variant"]] for q in rows if by_key[q["uid"], q["variant"]]["status"] in accepted]
    path = scores_dir / "c_rewrite.npz"
    score_by_key = {}
    cad_uids = uids_for(gallery)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            archive_keys = list(zip(data["query_uids"].tolist(), data["query_variants"].tolist()))
        archive_set = set(archive_keys)
        expected = [q for q in scored_rows if (q["uid"], q["variant"]) in archive_set]
        values, _ = load_scores(path, expected, cad_uids, background)
        score_by_key = dict(zip(archive_keys, values))
    require(all(key in score_by_key for key, row in by_key.items() if row["status"] == "changed_valid"),
            "a valid changed rewrite is missing its frozen score row")
    result = {"n_original_queries_in_ledger": len(selected), "variants": {}, "common_three_version_comparison": {}}
    all_ranks, evidence = {}, []
    for variant in VARIANTS:
        subset = [by_key[uid, variant] for uid in selected if (uid, variant) in by_key]
        states = Counter(q["status"] for q in subset)
        available = [q for q in subset if q["status"] in accepted]
        changed = np.asarray([q["status"] == "changed_valid" for q in available], dtype=bool)
        variant_result = {"statuses": {**{name: states[name] for name in REWRITE_STATUSES},
                                       "pending_missing_ledger": len(selected) - len(subset)},
                          "methods": {}, "n_after_token_counts_available": sum(type(q.get("token_count")) is int for q in available)}
        result["variants"][variant] = variant_result
        all_ranks[variant] = {}
        if not available:
            continue
        originals = [orig[q["uid"]] for q in available]
        positions = np.array([i for i, _ in originals], dtype=int)
        groups = np.array([q["group_id"] for q in available])
        rewritten_raw = np.stack([score_by_key[q["uid"], variant] if q["status"] == "changed_valid" else raw[i]
                                  for q, (i, _) in zip(available, originals)])
        for q, (i, _) in zip(available, originals):
            if q["status"] == "no_edit" and (q["uid"], variant) in score_by_key:
                require(np.allclose(score_by_key[q["uid"], variant], raw[i], rtol=1e-6, atol=1e-3),
                        "no_edit scoring row differs from unchanged original")
        scored = score_readouts(rewritten_raw, background)
        token_changes = [q["token_count"] - original["token_count"] for q, (_, original) in zip(available, originals)
                         if q["status"] == "changed_valid" and type(q.get("token_count")) is int and type(original.get("token_count")) is int]
        variant_result["changed_token_count_delta"] = {"n": len(token_changes), "median": float(np.median(token_changes)) if token_changes else None}
        for method in METHODS:
            before = original_ranks[method][positions]
            after = rank_matrix(scored[method], cad_uids, "fusion360")[np.arange(len(positions)), positions]
            variant_result["methods"][method] = paired_ranks(before[changed], after[changed], groups[changed], "fusion360", variant)
            all_ranks[variant][method] = dict(zip(uids_for(available), after))
            rank_rows = rank_comparison_rows(f"C_{variant}", method, available, before, after)
            for row, rewritten, (_, original) in zip(rank_rows, available, originals):
                n_constraints = rewritten.get("n_constraints")
                if n_constraints is None and isinstance(rewritten.get("constraints"), list):
                    n_constraints = len(rewritten["constraints"])
                evidence.append({**row, "n_tokens_before": original.get("token_count"), "n_tokens_after": rewritten.get("token_count"),
                                 "n_constraints_after": n_constraints})
    common = [uid for uid in selected if all(uid in all_ranks[v].get("S0", {}) for v in VARIANTS)]
    result["common_three_version_comparison"]["n_queries"] = len(common)
    if common:
        common_groups = [orig[uid][1]["group_id"] for uid in common]
        for method in METHODS:
            methods = {"Qorig": original_ranks[method][[orig[uid][0] for uid in common]],
                       **{v: np.array([all_ranks[v][method][uid] for uid in common]) for v in VARIANTS}}
            result["common_three_version_comparison"][method] = summarize_methods(
                methods, common_groups, len(gallery), "fusion360", "C_common_rewrite")
    return result, evidence


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def blind_asset(input_dir, row, public_dir, candidate_code):
    source = row.get("geometry_relpath")
    if not source:
        return None
    path = (input_dir / source).resolve()
    require(path.is_relative_to(input_dir.resolve()), "blind geometry paths must resolve within the frozen input")
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix not in (".brep", ".stl", ".step", ".stp", ".obj"):
        return None
    data = path.read_bytes()
    if suffix == ".brep":
        try:
            lines = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
        except UnicodeError:
            return None
        if lines and lines[0] == "DBRep_DrawableShape":
            lines = lines[1:]
        if (not lines or not re.fullmatch(r"CASCADE Topology V[123], \(c\) (?:Open Cascade|Matra-Datavision)", lines[0], re.IGNORECASE)
                or not all(line.isascii() for line in lines) or row["uid"].encode() in data):
            return None
    elif suffix in (".step", ".stp"):
        data = re.sub(rb"FILE_NAME\s*\(.*?\);",
                      b"FILE_NAME('part','',(''),(''),'','','');", data, flags=re.IGNORECASE | re.DOTALL)
        data = re.sub(rb"'(?:[^']|'')*'", lambda m: m[0].replace(row["uid"].encode(), b"part"), data)
    elif suffix == ".obj":
        data = b"\n".join(line for line in data.splitlines() if not line.lstrip().startswith(
            (b"#", b"o ", b"g ", b"mtllib ", b"usemtl "))) + b"\n"
    elif len(data) >= 84 and len(data) == 84 + 50 * struct.unpack_from("<I", data, 80)[0]:
        data = bytes(80) + data[80:]
    else:
        data = re.sub(rb"(?mi)^([ \t]*)(solid|endsolid)\b[^\r\n]*", rb"\1\2 part", data)
    relative = Path("assets") / (candidate_code + suffix)
    (public_dir / relative).parent.mkdir(exist_ok=True)
    (public_dir / relative).write_bytes(data)
    return relative.as_posix()


def blind_packets(input_dir, output, queries, gallery, matrices, selected):
    public_dir = output / "blind_packets" / "public"
    private_dir = output / "blind_packets" / "private"
    public_dir.mkdir(parents=True)
    private_dir.mkdir()
    query_index = {q["uid"]: i for i, q in enumerate(queries)}
    forms, packets, sealed = [], [], []
    random_seed = subseed(BOOT_SEED, "fusion360", "blind_random")
    rng = np.random.Generator(np.random.PCG64(random_seed))
    presentation_seed = subseed(BOOT_SEED, "fusion360", "blind_presentation")
    presentation = np.random.Generator(np.random.PCG64(presentation_seed))
    random_indices = [int(rng.integers(len(gallery) - 1)) for _ in selected]
    for position in presentation.permutation(len(selected)):
        uid = selected[position]
        i = query_index[uid]
        q = queries[i]
        query_code = secrets.token_hex(8)
        chosen = defaultdict(dict)
        random_cad = random_indices[position] + (random_indices[position] >= i)
        for method in METHODS:
            for j in np.flatnonzero(matrices[method][i] <= 3):
                chosen[j][method] = int(matrices[method][i, j])
        chosen[i]["registered"] = True
        chosen[random_cad]["random"] = True
        candidates = []
        for j in presentation.permutation(sorted(chosen)):
            code = secrets.token_hex(8)
            candidate = gallery[j]
            asset = blind_asset(input_dir, candidate, public_dir, code)
            candidates.append({"candidate_code": code, "asset": asset})
            forms.append({"query_code": query_code, "candidate_code": code})
            sealed.append({"query_code": query_code, "candidate_code": code, "query_uid": uid,
                           "candidate_uid": candidate["uid"], "roles": chosen[j]})
        packets.append({"query_code": query_code, "text": q["text"], "candidates": candidates})
    write_jsonl(public_dir / "packets.jsonl", packets)
    write_jsonl(private_dir / "id_map.jsonl", sealed)
    columns = ["rater_code", "query_code", "candidate_code", "label", "constraint_evidence", "process_only"]
    for path in (public_dir / "rater_1.csv", public_dir / "rater_2.csv", output / "annotations.csv"):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            if path.name != "annotations.csv":
                writer.writerows(forms)
    (public_dir / "RATING_GUIDE.md").write_text(
        "# Independent geometry ratings\n\n"
        "Read the original text and inspect the candidate using its geometry asset. Judge absolute dimensions only "
        "when the description units and geometry scale can be established. Record evidence for each verifiable constraint. "
        "Use `process_only` for requirements that can only be judged from modeling history. Fill one rater form independently.\n\n"
        "- `all_met` — all explicit verifiable geometric constraints have supporting evidence.\n"
        "- `partial` — some are met and some cannot be verified, with no known conflict.\n"
        "- `conflict` — at least one verifiable explicit constraint is contradicted. Identify it in the evidence.\n"
        "- `insufficient` — no constraint can be established with the available views, scale, or evidence.\n",
        encoding="utf-8", newline="\n",
    )
    n_assets = sum(bool(c["asset"]) for packet in packets for c in packet["candidates"])
    return {"status": "pending_two_independent_annotators", "n_queries": len(packets), "n_rating_units": len(sealed),
            "n_units_with_asset": n_assets, "n_units_pending_asset": len(sealed) - n_assets, "n_annotations": 0,
            "random_control_seed": random_seed, "presentation_seed": presentation_seed,
            "blind_id_generation": "opaque random codes regenerated on each run; preserve the original restricted map for ratings",
            "shareable_packet_directory": "blind_packets/public", "restricted_uid_map": "blind_packets/private/id_map.jsonl"}


def evaluate_target(input_dir, scores_dir, output, freeze, gallery_b, queries_b, raw_b, background_b, metrics, evidence):
    verify_frozen_files(input_dir, freeze, ("queries_a.jsonl", "queries_c.jsonl", "gallery_fusion360.jsonl",
                                           "baseline_reference.json", "gallery_fusion_original.jsonl", "queries_c_alt.jsonl",
                                           "queries_c_rewrite.jsonl", "rewrites.jsonl"))
    reference = source_reference(input_dir)
    summary_b, ranks_b, _, rows = domain_analysis(queries_b, gallery_b, raw_b, background_b, "deepcad", reference, "B")
    evidence.extend(rows)
    b_uids = uids_for(gallery_b)
    l0_nonempty = [r for r in gallery_b if isinstance(r.get("l0_text"), str) and r["l0_text"].strip()]
    l0_rows = [{"uid": row["uid"], "group_id": row["group_id"], "text": row["l0_text"],
                "token_count": row.get("l0_token_count"), "token_ids": row.get("l0_token_ids")}
               for row in l0_nonempty if row.get("l0_token_count") is None or 0 < row["l0_token_count"] <= 512]
    queries_a = query_file(input_dir, "a", l0_rows)
    validate_queries(queries_a, gallery_b)
    if all("l0_text" in row for row in gallery_b):
        expected_l0 = {r["uid"]: r["text"] for r in l0_rows}
        require({r["uid"]: r["text"] for r in queries_a} == expected_l0, "A must use every fully encodable frozen L0 text")
    summary_a = {"n_queries": 0, "n_gallery": len(b_uids), "status": "no_eligible_frozen_L0_queries"}
    comparison_ab = {"n_common_queries": 0, "status": "no_eligible_frozen_L0_queries"}
    if queries_a:
        raw_a, bg_a = load_scores(scores_dir / "a.npz", queries_a, b_uids, background_b)
        summary_a, ranks_a, _, rows = domain_analysis(queries_a, gallery_b, raw_a, bg_a, "deepcad", reference, "A")
        evidence.extend(rows)
        comparison_ab, rows = compare_a_b(queries_a, queries_b, ranks_a, ranks_b, len(b_uids))
        evidence.extend(rows)
    if all("l0_text" in row for row in gallery_b):
        summary_a["coverage"] = {"n_nonempty": len(l0_nonempty), "n_missing_or_empty": len(b_uids) - len(l0_nonempty),
                                 "n_nonempty_excluded_for_token_limit": len(l0_nonempty) - len(l0_rows),
                                 "n_distinct_nonempty_L0_texts": len({r["l0_text"] for r in l0_nonempty})}
    gallery_c = frozen_gallery(input_dir / "gallery_fusion360.jsonl")
    require(len(gallery_b) == len(gallery_c), "B and C must use the same formal gallery size")
    queries_c = query_file(input_dir, "c", gallery_c)
    validate_queries(queries_c, gallery_c, complete=True)
    for query, row in zip(queries_c, gallery_c):
        if "text" in row:
            require(query["text"] == row["text"], "C main query text disagrees with frozen gallery record")
    raw_c, background_c = load_scores(scores_dir / "c.npz", queries_c, uids_for(gallery_c))
    summary_c, ranks_c, matrices_c, rows = domain_analysis(queries_c, gallery_c, raw_c, background_c, "fusion360", reference, "C")
    evidence.extend(rows)
    selected = diagnostic_uids(freeze, gallery_c)
    target_gate, rows = target_encoding(input_dir, scores_dir, freeze, queries_c, gallery_c, raw_c, background_c, ranks_c)
    evidence.extend(rows)
    rewrites, rows = rewrite_analysis(input_dir, scores_dir, selected, queries_c, gallery_c, raw_c, background_c, ranks_c)
    evidence.extend(rows)
    c_main_matrices = {m: matrices_c[m] for m in METHODS}
    c_bins = source_bins(queries_c, gallery_c, reference)
    field_test = {"status": "pending_source_validation_or_field_counts"}
    if "n_valid" in c_bins:
        field_test = permutation_test(c_main_matrices, np.arange(len(gallery_c)), "fusion360", c_bins["n_valid"])
    metrics.update({
        "phase": "formal_main",
        "main": {"A": summary_a, "B": summary_b, "C": summary_c},
        "primary_C": permutation_test(c_main_matrices, np.arange(len(gallery_c)), "fusion360"),
        "diagnostic_C_field_permutation": field_test,
        "expert_minus_L0_A_B": comparison_ab,
        "external_minus_source_B_C": compare_b_c(ranks_b, ranks_c, source_bins(queries_b, gallery_b, reference), c_bins, reference),
        "c_encoding_sensitivity": target_gate,
        "c_rewrites": rewrites,
        "semantic_review": blind_packets(input_dir, output, queries_c, gallery_c, c_main_matrices, selected),
        "source_reference": {"status": reference["status"], "sha256": reference.get("sha256"),
                             "n_validation_samples": {key: len(rows) for key, rows in reference.get("samples", {}).items()},
                             "numeric_rules_frozen": "numeric_rules" in reference},
    })
    return metrics, evidence


def write_rank_evidence(output, rows):
    columns = {"analysis": "string", "method": "string", "query_uid": "string", "group_id": "string", "variant": "string",
               "rank_before": "int32", "rank_after": "int32", "hit_before": "bool", "hit_after": "bool", "top20_flip": "bool",
               "target_column_replaced": "bool", "n_tokens_before": "int32", "n_tokens_after": "int32", "n_constraints_after": "int32"}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        with (output / "ranks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        return {"file": "ranks.csv", "n_rows": len(rows), "parquet_status": "pending_pyarrow_runtime"}
    schema = pa.schema([(name, pa.type_for_alias(dtype)) for name, dtype in columns.items()])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output / "ranks.parquet")
    return {"file": "ranks.parquet", "n_rows": len(rows), "parquet_status": "available"}


def score_manifest(input_dir, scores_dir, metrics):
    names = ["b", "b_official"]
    if metrics["phase"] == "formal_main":
        names.append("c")
        if metrics["main"]["A"]["n_queries"]:
            names.append("a")
        if metrics["c_encoding_sensitivity"]["status"] == "available":
            names.append("c_alt")
        if (scores_dir / "c_rewrite.npz").is_file():
            names.append("c_rewrite")
    evaluator = Path(__file__).resolve()
    cli = evaluator.parents[2] / "scripts" / "eval_text2cad_e1.py"
    result = {"freeze_sha256": sha256(input_dir / "freeze.json"), "archive_directory": str(scores_dir.resolve()),
              "evaluator_sha256": sha256(evaluator), "evaluator_cli_sha256": sha256(cli),
              "S0": "sum of true non-padding field log probabilities", "S48": "S0 minus logmeanexp of 48 ordered background rows",
              "aggregation": "logmix_l1", "archives": {}}
    for name in names:
        path = scores_dir / f"{name}.npz"
        with np.load(path, allow_pickle=False) as data:
            scores = data["scores"]
            result["archives"][name] = {"file": path.name, "sha256": sha256(path), "dtype": str(scores.dtype),
                                        "query_shape": list(scores.shape), "archive_has_background": "background" in data.files,
                                        "effective_background_shape": [N_BACKGROUND, scores.shape[1]],
                                        "query_uid_axis": "query_uids", "cad_uid_axis": "cad_uids",
                                        "candidate_field_counts": "counts" if "counts" in data.files else None}
            if "query_variants" in data.files:
                result["archives"][name]["query_variant_axis"] = "query_variants"
    return result


def write_results(output, metrics, input_dir, scores_dir, source_only):
    rows = ["# E1 retrieval evaluation", "", f"Evaluation phase `{metrics['phase']}`.", ""]
    if "main" in metrics:
        rows.extend(("| Domain | Query text | Score | Queries | Gallery | R@1 | R@5 | R@20 | MedR | 95% CI for R@20 |",
                     "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
        for arm, label, domain in (("A", "Official L0", "DeepCAD"), ("B", "Expert", "DeepCAD"), ("C", "Expert", "Fusion 360")):
            for method in METHODS:
                score = metrics["main"][arm].get("methods", {}).get(method)
                if score is None:
                    continue
                hits = " | ".join(f"{score[f'R@{k}']:.2%} ({score['hits'][f'R@{k}']})" for k in KS)
                lo, hi = score["ci95"]["R@20"]
                rows.append(f"| {domain} | {label} ({arm}) | {method} | {score['n_queries']} | {score['n_gallery']} | {hits} | {score['MedR']:g} | {lo:.2%} to {hi:.2%} |")
        rows.extend(("", "## Confirmatory Fusion 360 test", ""))
        for method, test in metrics["primary_C"]["methods"].items():
            rows.append(f"- {method} had {test['hits']} Top-20 hits. Raw permutation p = {test['p']:.6g}. Holm-adjusted p = {test['holm_p']:.6g}.")
        rows.extend(("", "## Paired Top-20 change from S0 to S48", ""))
        for arm, summary in metrics["main"].items():
            delta = summary.get("top20_S48_minus_S0")
            if delta:
                lo, hi = delta["ci95_pp"]
                rows.append(f"- {arm} gained {delta['added_hits']} hits and lost {delta['lost_hits']} hits. Net {delta['net_pp']:+.2f} pp (95% CI {lo:+.2f} to {hi:+.2f} pp).")
        rows.extend(("", "## Fusion 360 diagnostics", ""))
        sensitivity = metrics["c_encoding_sensitivity"]
        rows.append(f"- Native-history replacement status `{sensitivity['status']}`. Frozen replacement columns {sensitivity.get('n_replaced_columns', 'pending')}.")
        for variant, result in metrics["c_rewrites"]["variants"].items():
            counts = ", ".join(f"{name} {count}" for name, count in result["statuses"].items())
            rows.append(f"- {variant} rewrite ledger counts are {counts}.")
            for method, delta in result["methods"].items():
                if delta["n_queries"]:
                    lo, hi = delta["ci95_pp"]
                    rows.append(f"  - {method} gained {delta['added_hits']} hits and lost {delta['lost_hits']} hits among {delta['n_queries']} changed descriptions. Net {delta['net_pp']:+.2f} pp (95% CI {lo:+.2f} to {hi:+.2f} pp).")
        semantic = metrics["semantic_review"]
        rows.append(f"- Two independent semantic ratings are pending. The blinded set contains {semantic['n_rating_units']} units, with {semantic['n_units_with_asset']} geometry assets available.")
    rows.extend(("", "## DeepCAD encoding check", ""))
    for method, check in metrics["encoding_check"]["methods"].items():
        lo, hi = check["ci95_pp"]
        rows.append(f"- {method} had {check['official_only']} official-only and {check['reencoded_only']} reencoded-only hits in {check['n_queries']} queries. Estimated net encoding loss {check['net_loss_pp']:+.3f} pp (95% CI {lo:+.3f} to {hi:+.3f} pp). Decision `{check['status']}` against {check['tolerance_pp']} pp tolerance.")
    rows += ["", "Intervals condition on the frozen galleries. Each recall counts a match to the registered CAD UID. "
             "The analysis compares one generator across the covered text and candidate representations.", "", "## Reproduction", ""]
    script = Path(__file__).resolve().parents[2] / "scripts" / "eval_text2cad_e1.py"
    invocation = [sys.executable, str(script), "--input", str(input_dir.resolve()), "--scores", str(scores_dir.resolve()),
                  "--output", str(output.resolve().with_name(output.name + "-reproduced"))]
    if source_only:
        invocation.append("--source-only")
    cmd = subprocess.list2cmdline(invocation) if sys.platform == "win32" else shlex.join(invocation)
    rows.extend((f"Python {platform.python_version()} and NumPy {np.__version__} on {platform.system()}.", "", "```text", cmd, "```", ""))
    (output / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8", newline="\n")


def write_checksums(output):
    entries = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in sorted(output.rglob("*"))
               if path.is_file() and path != output / "SHA256SUMS"]
    (output / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")


def evaluate(input_dir, scores_dir, output, source_only=False):
    input_dir, scores_dir, output = map(Path, (input_dir, scores_dir, output))
    require(not output.exists() or not any(output.iterdir()), f"evaluation output must be an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        freeze, gallery, queries, raw, background, gate, evidence = source_analysis(input_dir, scores_dir)
    except Exception as error:
        write_json(output / "evaluation_failure.json", {"stage": "source_encoding", "error": f"{type(error).__name__}: {error}"})
        raise
    metrics = {"phase": "source_only", "encoding_check": gate}
    write_json(output / "encoding_check.json", gate)
    if not source_only and gate["status"] != "fail":
        try:
            metrics, evidence = evaluate_target(input_dir, scores_dir, output, freeze, gallery, queries, raw,
                                                background, metrics, evidence)
        except Exception as error:
            write_json(output / "evaluation_failure.json", {"stage": "target_evaluation", "error": f"{type(error).__name__}: {error}"})
            raise
    elif not source_only:
        metrics["phase"] = "source_encoding_degradation"
        metrics["target_status"] = "pending_source_validation_refreeze"
    metrics["rank_evidence"] = write_rank_evidence(output, evidence)
    write_json(output / "metrics.json", metrics)
    write_json(output / "score_manifest.json", score_manifest(input_dir, scores_dir, metrics))
    write_results(output, metrics, input_dir, scores_dir, source_only)
    write_checksums(output)
    return metrics
