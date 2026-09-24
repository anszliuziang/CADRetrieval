"""P1 pre-registered analysis (plan sec.3). Pure numpy/stdlib; runs on the raw
per-pair records (pairs.jsonl) plus the fixed selection manifest — the summary
is always recomputed from raw per-pair files, never from a pre-aggregated table.

Diagnostics (all labelled exact-UID pairing sanity, NOT semantic retrieval):
  1. same-candidate margin: for each of the 32 selected programs x,
     margin(x) = S(own,x) - mean_j S(other_j,x) over its 3 offset queries.
     Candidate response tokens are identical across the 4 queries, so the
     margin cannot be produced by candidate length alone. Bootstrap 95% CI
     resamples candidate UIDs.
  2. 8-candidate ranking: rank of the paired UID, strict top-1 rate, pairwise
     win rate vs the 7 others, for three score arms: raw sum, mean (length
     diagnostic only), corrected S(q,x)-S(empty,x) (known candidate-prior
     correction, not a new method). Bootstrap 95% CI resamples queries.
  3. Baselines on the same 8-candidate sets: fixed-seed random, query-
     independent S(empty,x), response-length ascending, char 3-gram TF-IDF
     cosine (IDF fitted on THIS round's distinct candidate programs only;
     no qrels used). Per-baseline wallclock is recorded by the caller.
  4. Score-independent near-duplicate screen: program token Jaccard between
     each positive and its 7 co-candidates; exact-identical program text
     between different UIDs is a data-admission error (flagged, not deleted).
"""

from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass

BOOTSTRAP_SEED = 20260925
N_RESAMPLES = 10000  # plan requires >= 2000
RANDOM_BASELINE_SEED = 20260925
HIGH_JACCARD_THRESHOLD = 0.5  # heuristic flag only; nothing is deleted

SCORE_ARMS = ("sum", "mean", "corrected")

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class AnalysisError(Exception):
    """Raw records inconsistent with the fixed plan (missing/failed pairs)."""


# ---------------------------------------------------------------- raw records

def validate_pair_records(records: list[dict], expected_pair_ids: list[str]) -> None:
    """Every planned pair must be present exactly once with status ok and
    finite scores; a success-only subset is not acceptable (plan sec.4)."""
    ids = [r.get("pair_id") for r in records]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise AnalysisError(f"duplicate pair_id in records: {sorted(dup)[:5]}")
    have = set(ids)
    missing = [p for p in expected_pair_ids if p not in have]
    extra = [p for p in ids if p not in set(expected_pair_ids)]
    if missing or extra:
        raise AnalysisError(f"pair set mismatch: missing={missing[:5]} extra={extra[:5]}")
    bad = [r["pair_id"] for r in records if r.get("status") != "ok" or not r.get("all_finite")]
    if bad:
        raise AnalysisError(f"non-ok/non-finite pair records present: {bad[:5]}")


def score_lookup(records: list[dict]) -> dict[tuple[str, str | None, str], dict]:
    """(kind, query_uid|None, candidate_uid) -> record."""
    out = {}
    for r in records:
        out[(r["kind"], r.get("query_uid"), r["candidate_uid"])] = r
    return out


# ---------------------------------------------------------------- bootstrap

def bootstrap_ci(values: list[float], stat_fn, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile bootstrap 95% CI, resampling len(values) units with
    replacement. A fresh random.Random(seed) per call keeps every CI
    reproducible independently of call order."""
    n = len(values)
    if n == 0:
        raise AnalysisError("bootstrap over empty values")
    point = stat_fn(values)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[min(n_resamples - 1, int(0.025 * n_resamples))]
    hi = stats[min(n_resamples - 1, int(0.975 * n_resamples))]
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_units": n, "n_resamples": n_resamples, "seed": seed}


# ------------------------------------------------------------ primary margin

def same_candidate_margins(table: dict, selection: dict) -> dict:
    """margin(x) = S(own,x) - mean_j S(other_j,x) using raw sum scores."""
    per = []
    for qp in selection["queries"]:
        uid = qp["uid"]
        own = table.get(("conditional", uid, uid))
        if own is None:
            raise AnalysisError(f"missing self pair for {uid}")
        others = [
            r for (kind, _q, cuid), r in table.items()
            if kind == "conditional" and cuid == uid and r.get("candidate_source") == "selected-offset"
        ]
        if len(others) != len(selection["offsets"]):
            raise AnalysisError(f"candidate {uid}: expected {len(selection['offsets'])} offset pairs, found {len(others)}")
        margin = own["sum_logprob"] - statistics.mean(o["sum_logprob"] for o in others)
        per.append(
            {
                "candidate_uid": uid,
                "query_index": qp["query_index"],
                "own_sum": own["sum_logprob"],
                "other_sums": [o["sum_logprob"] for o in others],
                "margin": margin,
                "n_response_tokens": qp["n_response_tokens"],
            }
        )
    margins = [p["margin"] for p in per]
    return {
        "per_candidate": per,
        "n": len(margins),
        "mean_margin": bootstrap_ci(margins, statistics.mean),
        "median_margin": bootstrap_ci(margins, statistics.median),
        "n_positive": sum(1 for m in margins if m > 0),
        "resampling_unit": "candidate uid (32 programs)",
    }


# ------------------------------------------------------------ 8-candidate ranking

def rank_of_self(scores: list[float], self_index: int = 0) -> tuple[int, bool, float]:
    """scores: 8 candidate scores, self at self_index (higher = better).
    Returns (rank, top1_strict, win_rate): rank = 1 + #strictly greater;
    top1_strict = self strictly greater than all others; win_rate counts
    ties as 0.5 over the 7 pairwise comparisons."""
    s = scores[self_index]
    others = [x for k, x in enumerate(scores) if k != self_index]
    rank = 1 + sum(1 for x in others if x > s)
    top1 = all(s > x for x in others)
    wins = sum(1.0 if s > x else 0.5 if s == x else 0.0 for x in others)
    return rank, top1, wins / len(others)


def _arm_score(rec: dict, arm: str, empty_sums: dict[str, float]) -> float:
    if arm == "sum":
        return rec["sum_logprob"]
    if arm == "mean":
        return rec["mean_logprob"]
    if arm == "corrected":
        return rec["sum_logprob"] - empty_sums[rec["candidate_uid"]]
    raise ValueError(f"unknown score arm {arm!r}")


def empty_score_map(records: list[dict]) -> dict[str, float]:
    out = {}
    for r in records:
        if r["kind"] == "empty":
            if r["candidate_uid"] in out:
                raise AnalysisError(f"duplicate empty pair for {r['candidate_uid']}")
            out[r["candidate_uid"]] = r["sum_logprob"]
    return out


def ranking_analysis(table: dict, selection: dict, empty_sums: dict[str, float], arm: str,
                     n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    per = []
    for qp in selection["queries"]:
        scores = []
        for c in qp["candidates"]:
            rec = table.get(("conditional", qp["uid"], c["uid"]))
            if rec is None:
                raise AnalysisError(f"missing pair q{qp['query_index']:02d} cand {c['uid']}")
            scores.append(_arm_score(rec, arm, empty_sums))
        rank, top1, win = rank_of_self(scores, 0)
        per.append({"query_uid": qp["uid"], "query_index": qp["query_index"], "rank": rank, "top1_strict": top1, "win_rate": win})
    ranks = [p["rank"] for p in per]
    top1s = [1.0 if p["top1_strict"] else 0.0 for p in per]
    wins = [p["win_rate"] for p in per]
    return {
        "arm": arm,
        "per_query": per,
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "top1_strict_rate": bootstrap_ci(top1s, statistics.mean, n_resamples, seed),
        "mean_win_rate": bootstrap_ci(wins, statistics.mean, n_resamples, seed),
        "resampling_unit": "query (32 queries)",
    }


# ---------------------------------------------------------------- baselines

def _order_summary(orders: list[list[str]], selection: dict) -> dict:
    """orders: per query, candidate uids ordered best-first. Same summary shape
    as ranking_analysis (deterministic total order => ties impossible)."""
    per = []
    for qp, order in zip(selection["queries"], orders):
        rank = order.index(qp["uid"]) + 1
        per.append(
            {
                "query_uid": qp["uid"],
                "query_index": qp["query_index"],
                "rank": rank,
                "top1_strict": rank == 1,
                "win_rate": (8 - rank) / 7,
            }
        )
    ranks = [p["rank"] for p in per]
    top1s = [1.0 if p["top1_strict"] else 0.0 for p in per]
    wins = [p["win_rate"] for p in per]
    return {
        "per_query": per,
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "top1_strict_rate": bootstrap_ci(top1s, statistics.mean),
        "mean_win_rate": bootstrap_ci(wins, statistics.mean),
        "resampling_unit": "query (32 queries)",
    }


def baseline_random(selection: dict, seed: int = RANDOM_BASELINE_SEED) -> dict:
    """Fixed-seed uniform permutation per query (one rng stream for all)."""
    rng = random.Random(seed)
    orders = []
    for qp in selection["queries"]:
        uids = [c["uid"] for c in qp["candidates"]]
        rng.shuffle(uids)
        orders.append(uids)
    return {"baseline": "random_fixed_seed", "seed": seed, **_order_summary(orders, selection)}


def baseline_length(selection: dict) -> dict:
    """Shorter response first; ties by uid ascending."""
    orders = []
    for qp in selection["queries"]:
        cands = sorted(qp["candidates"], key=lambda c: (c["n_response_tokens"], c["uid"]))
        orders.append([c["uid"] for c in cands])
    return {"baseline": "response_length_ascending", **_order_summary(orders, selection)}


def baseline_empty(selection: dict, empty_sums: dict[str, float]) -> dict:
    """Query-independent candidate prior S(empty,x), higher first; ties uid asc."""
    orders = []
    for qp in selection["queries"]:
        cands = sorted(qp["candidates"], key=lambda c: (-empty_sums[c["uid"]], c["uid"]))
        orders.append([c["uid"] for c in cands])
    return {"baseline": "empty_query_score", **_order_summary(orders, selection)}


def char_ngrams(text: str, n: int = 3) -> list[str]:
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def tfidf_cosine(query: str, doc: str, idf: dict[str, float], n: int = 3) -> float:
    qg = char_ngrams(query, n)
    dg = char_ngrams(doc, n)
    qtf: dict[str, int] = {}
    for g in qg:
        qtf[g] = qtf.get(g, 0) + 1
    dtf: dict[str, int] = {}
    for g in dg:
        dtf[g] = dtf.get(g, 0) + 1
    dot = qn = dn = 0.0
    for g, c in qtf.items():
        w = c * idf.get(g, 0.0)
        qn += w * w
        if g in dtf:
            dot += w * (dtf[g] * idf[g])
    for g, c in dtf.items():
        w = c * idf.get(g, 0.0)
        dn += w * w
    if qn == 0.0 or dn == 0.0:
        return 0.0
    return dot / math.sqrt(qn * dn)


def baseline_tfidf(selection: dict, query_texts: dict[str, str], programs: dict[str, str],
                   n: int = 3) -> dict:
    """Char 3-gram TF-IDF cosine(query, raw program). IDF fitted ONLY on this
    round's distinct candidate programs (selection['distinct_candidate_uids']);
    sklearn-style smoothing idf = ln((1+N)/(1+df)) + 1; raw-count TF."""
    docs = selection["distinct_candidate_uids"]
    N = len(docs)
    df: dict[str, int] = {}
    for uid in docs:
        for g in set(char_ngrams(programs[uid], n)):
            df[g] = df.get(g, 0) + 1
    idf = {g: math.log((1 + N) / (1 + d)) + 1.0 for g, d in df.items()}
    orders = []
    for qp in selection["queries"]:
        q = query_texts[qp["uid"]]
        scored = [(tfidf_cosine(q, programs[c["uid"]], idf, n), c["uid"]) for c in qp["candidates"]]
        scored.sort(key=lambda t: (-t[0], t[1]))
        orders.append([u for _s, u in scored])
    return {
        "baseline": "char3_tfidf_cosine",
        "idf_corpus": "distinct candidate programs of this round only",
        "n_docs_idf": N,
        **_order_summary(orders, selection),
    }


# ------------------------------------------------------- near-duplicate screen

def program_token_set(program: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(program))


def near_duplicate_screen(selection: dict, programs: dict[str, str],
                          threshold: float = HIGH_JACCARD_THRESHOLD) -> dict:
    """Score-independent program token Jaccard between each positive and its 7
    co-candidates. exact_same_text across different uids => data admission error."""
    rows = []
    n_exact = 0
    for qp in selection["queries"]:
        pos_tok = program_token_set(programs[qp["uid"]])
        for c in qp["candidates"]:
            if c["source"] == "self":
                continue
            ct = program_token_set(programs[c["uid"]])
            union = pos_tok | ct
            j = len(pos_tok & ct) / len(union) if union else 0.0
            exact = programs[qp["uid"]] == programs[c["uid"]]
            if exact:
                n_exact += 1
            rows.append(
                {
                    "query_uid": qp["uid"],
                    "candidate_uid": c["uid"],
                    "candidate_source": c["source"],
                    "token_jaccard": j,
                    "exact_same_text": exact,
                    "flag_high": j >= threshold,
                }
            )
    rows.sort(key=lambda r: (-r["token_jaccard"], r["query_uid"], r["candidate_uid"]))
    return {
        "pairs": rows,
        "n_pairs": len(rows),
        "n_high": sum(1 for r in rows if r["flag_high"]),
        "threshold": threshold,
        "n_exact_same_text": n_exact,
        "data_admission_error": n_exact > 0,
        "note": "near-duplicate hint only; NOT a geometric family proof; nothing deleted",
    }


# -------------------------------------------------------- stratified diagnostics

def stratified_diagnostics(table: dict, selection: dict, margins: dict, programs: dict[str, str],
                           comment_ratio_fn, export_fn) -> dict:
    """Per-query stratification table (positive length, method similarity,
    comment ratio, export boilerplate, EOS logprob) plus median-split group
    means of the same-candidate margin. Descriptive only."""
    margin_by_uid = {p["candidate_uid"]: p["margin"] for p in margins["per_candidate"]}
    rows = []
    for qp in selection["queries"]:
        uid = qp["uid"]
        prog = programs[uid]
        own = table[("conditional", uid, uid)]
        method_sims = [c["method_similarity"] for c in qp["candidates"] if c["source"] == "method-neighbor"]
        rows.append(
            {
                "query_uid": uid,
                "n_response_tokens_positive": qp["n_response_tokens"],
                "max_method_neighbor_similarity": max(method_sims) if method_sims else None,
                "comment_line_ratio_positive": comment_ratio_fn(prog),
                "has_export_boilerplate_positive": export_fn(prog),
                "eos_logprob_self": own.get("eos_logprob"),
                "same_candidate_margin": margin_by_uid[uid],
            }
        )

    def split_mean(key):
        vals = sorted(r[key] for r in rows if r[key] is not None)
        med = statistics.median(vals)
        lo = [r["same_candidate_margin"] for r in rows if r[key] is not None and r[key] <= med]
        hi = [r["same_candidate_margin"] for r in rows if r[key] is not None and r[key] > med]
        return {
            "split_key": key,
            "median": med,
            "low_group": {"n": len(lo), "mean_margin": statistics.mean(lo) if lo else None},
            "high_group": {"n": len(hi), "mean_margin": statistics.mean(hi) if hi else None},
        }

    return {
        "per_query": rows,
        "median_splits": [
            split_mean("n_response_tokens_positive"),
            split_mean("max_method_neighbor_similarity"),
            split_mean("comment_line_ratio_positive"),
        ],
        "note": "descriptive stratification; no thresholds tuned on results",
    }
