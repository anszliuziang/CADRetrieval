"""Independent status, raw matrix, rank and confidence-interval witness."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify(condition, label):
    if not condition:
        raise ValueError(label)


def inspect_method(cfg, block, text, counts, background, configs):
    total, grouped = block[:, 0], block[:, 1:]
    kind = cfg["kind"]
    if kind == "sum":
        return total
    if kind == "count_asc":
        return -counts.sum(axis=1)
    if kind == "mean":
        return total / counts.sum(axis=1)
    refs = background[:24] if cfg.get("b24") else background
    if kind == "resid":
        return total - cfg["lambda"] * refs[:, :, 0].mean(axis=0)
    if kind == "logmix":
        array = refs[:, :, 0]
        vertex = array.max(axis=0)
        mix = vertex + np.log(np.exp(array - vertex).sum(axis=0)) - np.log(len(array))
        return total - cfg["lambda"] * mix
    if kind == "fusion":
        components = []
        for member in cfg["members"]:
            score = inspect_method(configs[member], block, text, counts, background, configs)
            order = np.lexsort((np.arange(len(score)), -score))
            inverse = np.empty_like(order)
            inverse[order] = np.arange(1, len(order) + 1)
            components.append(1.0 / (60 + inverse))
        return (components[0] + components[1]) / 2
    fields = grouped if kind == "fit_raw" else grouped - refs[:, :, 1:].mean(axis=0)
    weight = np.array(cfg["weights"])
    verify(weight.shape == (6,) and np.isfinite(weight).all() and (weight >= 0).all(), cfg["id"])
    if kind == "gate":
        gate_word = r"\b(?:holes?|cuts?|slots?|subtract|subtracts|subtracted|subtracting)\b"
        if "word" in cfg["gates"] and re.search(gate_word, text.casefold()):
            weight[3] *= 2
        if "digit" in cfg["gates"] and re.search(r"[0-9]", text):
            weight[:3] *= 2
    return (fields * (1.0 / np.maximum(1, counts))) @ weight


def check_metrics(rows, values):
    by_method = {}
    for name, metric in values.items():
        ranks = np.array([row["ranks"][name] for row in rows])
        verify(len(ranks) == metric["n"], f"{name}: query count")
        verify(float(np.median(ranks)) == metric["MedR"], f"{name}: median")
        verify(float(ranks.mean()) == metric["MeanR"], f"{name}: mean")
        for k in (1, 5, 10, 20):
            hits = int(np.count_nonzero(ranks <= k))
            verify(hits == metric["hit_counts"][f"hit@{k}"], f"{name}: hits at {k}")
            verify(hits / len(ranks) == metric[f"R@{k}"], f"{name}: R@{k}")
        by_method[name] = ranks
    return by_method


def check_intervals(rows, rank_arrays, interval_file, selected=None):
    intervals = read(interval_file)
    text_ids, positions = np.unique([row["text_index"] for row in rows], return_inverse=True)
    counts = np.zeros((5000, len(text_ids)), dtype=np.int64)
    rng = np.random.default_rng(20261118)
    for draw in counts:
        np.add.at(draw, rng.integers(len(text_ids), size=len(text_ids)), 1)
    denominators = counts @ np.bincount(positions)
    checked = 0
    for k in (1, 5, 10, 20):
        draws = {
            mode: (counts @ np.bincount(positions[ranks <= k], minlength=len(text_ids))) / denominators
            for mode, ranks in rank_arrays.items()
        }
        for mode, values in draws.items():
            key = f"R@{k}"
            verify(np.allclose(np.percentile(values, [2.5, 97.5]),
                               intervals["method_ci95"][mode][key], rtol=0, atol=5e-15),
                   f"{mode}: bootstrap {key}")
            checked += 1
            for baseline in [intervals["baseline"], intervals.get("best_background_config")]:
                if baseline and mode != baseline:
                    paired = np.percentile(values - draws[baseline], [2.5, 97.5])
                    stored = intervals[f"paired_gain_vs_{baseline}_ci95"][mode][key]
                    verify(np.allclose(paired, stored, rtol=0, atol=5e-15),
                           f"{mode}: paired {baseline} {key}")
                    checked += 1
    primary = {intervals["baseline"], intervals.get("best_background_config"), selected} - {None}
    medians = {}
    for method in primary:
        ranks = rank_arrays[method]
        order = np.argsort(ranks, kind="stable")
        ordered_ranks = ranks[order]
        samples = []
        for multiplicities, n in zip(counts, denominators):
            cumulative = np.cumsum(multiplicities[positions[order]])
            lo = np.searchsorted(cumulative, (n + 1) // 2, side="left")
            hi = np.searchsorted(cumulative, n // 2 + 1, side="left")
            samples.append((ordered_ranks[lo] + ordered_ranks[hi]) / 2)
        medians[method] = np.asarray(samples)
        verify(np.array_equal(np.percentile(samples, [2.5, 97.5]),
                              intervals["method_ci95"][method]["MedR"]), f"{method}: median interval")
    if selected:
        for baseline in primary - {selected}:
            paired = np.percentile(medians[selected] - medians[baseline], [2.5, 97.5])
            stored = intervals[f"paired_gain_vs_{baseline}_ci95"][selected]["MedR"]
            verify(np.array_equal(paired, stored), f"selected median gain vs {baseline}")
    return {"recall_intervals": checked, "median_intervals": len(medians)}


def fit_differences(block, counts, background, positives, rng, matrix):
    indices = np.arange(len(counts))
    positive_free = np.setdiff1d(indices, positives, assume_unique=True)
    length = counts.sum(axis=1)
    by_sum = np.lexsort((indices, -block[:, 0]))
    first = by_sum[~np.isin(by_sum, positives)][:16]
    pool = np.setdiff1d(positive_free, first, assume_unique=True)
    reciprocal = 1.0 / np.maximum(1, counts)
    raw = block[:, 1:] * reciprocal
    calibrated = (block[:, 1:] - background[:, :, 1:].mean(axis=0)) * reciprocal
    for pos in sorted(positives):
        near = pool[np.lexsort((pool, abs(length[pool] - length[pos])))[:16]]
        hard = np.r_[first, near]
        random = np.sort(rng.choice(positive_free, 32, replace=False))
        matrix["hard", "raw"].append(raw[pos] - raw[hard])
        matrix["hard", "bg"].append(calibrated[pos] - calibrated[hard])
        matrix["random", "bg"].append(calibrated[pos] - calibrated[random])


def check_fit(matrix, fit_report, methods):
    tested = 0
    strongest = 0.0
    for (source, variant), blocks in matrix.items():
        values = np.concatenate(blocks)
        report = fit_report["sources"][source]["variants"][variant]
        verify(digest(values.astype("<f8").tobytes()) == report["diffs_sha256"], f"{source}/{variant}: exact D")
        rms = np.sqrt(np.mean(values ** 2, axis=0))
        rms[rms == 0] = 1
        verify(np.array_equal(rms, report["scales"]), f"{source}/{variant}: RMS")
        standardized = values / rms
        for fit in report["fits"].values():
            w = np.asarray(fit["weights"])
            alpha = fit["alpha"]
            logits = standardized @ w
            loss = np.logaddexp(0.0, -logits).mean() + alpha * (w @ w)
            verify(abs(loss - fit["final_loss"]) < 1e-11 and loss <= np.log(2) + 1e-12, "fit objective")
            grad = -(standardized / (1 + np.exp(logits))[:, None]).mean(axis=0) + 2 * alpha * w
            error = float(np.max(np.abs(w - np.maximum(w - grad, 0))))
            strongest = max(strongest, error)
            verify(np.isfinite(w).all() and np.all(w >= 0) and error <= 1e-6, "fit optimality")
            for candidate in methods.values():
                if candidate.get("source") == source and candidate.get("variant") == variant and candidate.get("alpha") == alpha:
                    verify(np.array_equal(w / rms, candidate["weights"]), "applied RMS coefficients")
            tested += 1
    return {"fits_checked": tested, "max_projected_gradient": strongest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=["validation", "test"], required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--historical-g5", type=Path)
    parser.add_argument("--approved-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.run
    stage = root / args.phase
    score = stage / "scores"
    evdir = stage / "evaluation"
    receipt = read(stage / "receipt.json")
    verify((root / f"{args.phase}_controller.exit").read_text().strip() == "0", "controller exit")
    verify(receipt["status"] == "complete" and receipt["approved_sha"] == args.approved_sha, "receipt")
    verify(receipt["evaluation_exit_code"] == 0 and receipt["worker_exit_codes"] == [0] * 4, "child exits")
    prepared = root / "prepared" / args.phase
    input_bytes = (prepared / "inputs.json").read_bytes()
    eval_bytes = (prepared / "eval.json").read_bytes()
    verify(digest(input_bytes) == receipt["inputs_sha256"], "prepared input digest")
    verify(digest(eval_bytes) == receipt["eval_sha256"], "prepared eval digest")
    source, ground = json.loads(input_bytes), json.loads(eval_bytes)
    gallery = len(source["members"])
    text_stop = receipt["planned_texts"]
    verify(receipt["planned_pairs"] == text_stop * gallery, "pair arithmetic")
    sums = {}
    for name in evdir.iterdir():
        if name.name != "SHA256SUMS" and name.is_file():
            sums[name.name] = digest(name.read_bytes())
    declared = {name: sha for sha, name in (line.split(None, 1) for line in (evdir / "SHA256SUMS").read_text().splitlines())}
    verify(sums == declared, "evaluation artifact digest")

    test = args.phase == "test"
    metadata = read(evdir / ("evaluation_summary.json" if test else "selection_table.json"))
    frozen_path = args.frozen if test else evdir / "frozen.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    if test:
        verify(digest(frozen_bytes) == receipt["frozen_sha256"] == metadata["frozen_sha256"], "test freeze")
    verify(frozen["background_texts_sha256"] == [digest(t.encode()) for t in source["texts"][:48]], "same background")
    methods = {cfg["id"]: cfg for cfg in frozen["configs"]}
    counts = np.load(score / "counts.npy", allow_pickle=False)
    verify(counts.shape == (gallery, 6) and counts.dtype == np.int64, "count axis")
    verify(np.all(counts >= 0) and np.all(counts[:, 0] == counts[:, 1]) and np.all(counts[:, -1] == 1), "valid fields")
    counts_id = digest(counts.astype("<i8").tobytes())
    selected_queries = [q for q in ground["queries"] if q["text_index"] < text_stop and
                        ground["text_roles"][q["text_index"]] == ("test" if test else "select")]
    expected = {q["query_id"]: (q["text_index"], q["cad_index"]) for q in selected_queries}
    rows = lines(evdir / ("query_ranks.jsonl" if test else "select_query_ranks.jsonl"))
    verify(len(rows) == len(expected) and {r["query_id"]: (r["text_index"], r["cad_index"])
                                         for r in rows} == expected, "query identity")
    by_text = defaultdict(list)
    for row in rows:
        by_text[row["text_index"]].append(row)
        if test:
            verify(row["cad_uid"] == ground["cad_uids"][row["cad_index"]], "positive uid")
    tops = {row["text_index"]: row["cad_indices"] for row in lines(evdir / "top20.jsonl")} if test else {}
    rank_arrays = check_metrics(rows, metadata["metrics"] if test else metadata["point_metrics"])
    history = None
    if test and args.historical_g5:
        expected_old = "cebe4c8a361eeb7974ef125dd6e3fd991a330ac9473adb8de7b0ac3cb322b869"
        verify(digest(args.historical_g5.read_bytes()) == expected_old, "certified historical G5")
        baseline = {x["query_id"]: x for x in lines(args.historical_g5)}
        drift = []
        for q in rows:
            previous = baseline[q["query_id"]]
            verify(previous["cad_id"] == q["cad_uid"] and previous["cad_index"] == q["cad_index"], "historical positive identity")
            if previous["ranks"]["sum"] != q["ranks"]["sum"]:
                drift.append((q["query_id"], previous["ranks"]["sum"], q["ranks"]["sum"]))
        history = {"ranks_matched": len(rows) - len(drift), "n_queries": len(rows), "drift_examples": drift[:10]}
    if not test:
        keys = {n: (-v["R@20"], -v["R@1"], -v["R@5"], v["MedR"]) for n, v in metadata["point_metrics"].items()}
        winner = min(metadata["enumeration"], key=keys.get)
        verify(winner == frozen["selected_config"], "validation selection order")
    declared_chunks = {}
    for shard in range(4):
        log = stage / f"worker_{shard}.log"
        assignments = set()
        for entry in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if not entry.startswith("{"):
                continue
            row = json.loads(entry)
            if row.get("phase") == "text_scored":
                index = row["text_index"]
                verify(index not in declared_chunks and index % 4 == shard, "shard ownership")
                declared_chunks[index] = row["chunk_sha256"]
                assignments.add(index)
        summary = read(score / f"shard_summary_s{shard}.json")
        verify(summary["git"]["head"] == args.approved_sha, "worker source")
        verify(summary["scoring"]["n_texts_scored"] == len(assignments), "shard count")
    verify(set(declared_chunks) == set(range(text_stop)), "complete text logs")

    bg = []
    fit_positives = defaultdict(list)
    fit_matrix = defaultdict(list)
    fit_rng = np.random.default_rng(20261118)
    if not test:
        for q in ground["queries"]:
            if ground["text_roles"][q["text_index"]] == "fit":
                fit_positives[q["text_index"]].append(q["cad_index"])
    n_checked_ranks = 0
    identifiers = np.arange(gallery)
    for index in range(text_stop):
        name = score / "chunks" / f"t{index:05d}.npz"
        verify(digest(name.read_bytes()) == declared_chunks[index], f"{index}: scorer digest")
        with np.load(name, allow_pickle=False) as obj:
            block = obj["scores"]
            verify(obj["text_index"].item() == index and obj["counts_sha256"].item() == counts_id and
                   obj["text_sha256"].item() == digest(source["texts"][index].encode()) and
                   obj["inputs_sha256"].item() == receipt["inputs_sha256"], f"{index}: identities")
        verify(block.shape == (gallery, 7) and block.dtype == np.float64 and np.isfinite(block).all(),
               f"{index}: finite fields")
        verify(np.max(np.abs(block[:, 1:].sum(axis=1) - block[:, 0])) <= 1e-8 and
               np.all(block[:, 1:][counts == 0] == 0), f"{index}: partition")
        if index < 48:
            bg.append(block)
            continue
        if index == 48:
            bg = np.stack(bg)
        if index in fit_positives:
            fit_differences(block, counts, bg, fit_positives[index], fit_rng, fit_matrix)
        if index not in by_text:
            continue
        for mode in by_text[index][0]["ranks"]:
            if mode not in methods:
                verify(not test, "unknown test method")
                continue
            vec = inspect_method(methods[mode], block, source["texts"][index], counts, bg, methods)
            for row in by_text[index]:
                pos = row["cad_index"]
                measured = 1 + np.count_nonzero((vec > vec[pos]) | ((vec == vec[pos]) & (identifiers < pos)))
                verify(measured == row["ranks"][mode], f"rank {index}/{pos}/{mode}")
                n_checked_ranks += 1
            if test:
                verify(np.lexsort((identifiers, -vec))[:20].tolist() == tops[index][mode], f"top20 {index}/{mode}")
    interval_count = check_intervals(
        rows, rank_arrays, evdir / ("bootstrap.json" if test else "select_bootstrap.json"),
        frozen["selected_config"],
    )
    fit_check = check_fit(fit_matrix, read(evdir / "fit_report.json"), methods) if not test else None
    report = {
        "status": "passed", "axis": args.phase, "source_sha": args.approved_sha,
        "n_pairs": text_stop * gallery, "n_queries": len(rows), "n_raw_ranks_recomputed": n_checked_ranks,
        "point_metric_methods": len(rank_arrays), "verified_intervals": interval_count,
        "selected_config": frozen["selected_config"], "frozen_sha256": digest(frozen_bytes),
        "fit_recomputed": fit_check, "historical_g5": history,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
