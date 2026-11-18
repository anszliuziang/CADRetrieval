#!/usr/bin/env python
"""Frozen Text2CAD retrieval-accuracy night: fit / evaluate.

Plan: experiments/plans/IDEA01_TEXT2CAD_RETRIEVAL_ACCURACY_2026-11-18.md (run
order table assigns this script the fit/evaluate stages;
experiments/scripts/launch_text2cad_accuracy_stage.py invokes
  fit      on the validation axis after the four-card scoring completes, and
  evaluate on the test axis with the SHA-pinned frozen configuration).

Inputs (the interface fixed by the plan's artifact schema; produced by
run_text2cad_accuracy_night.py prepare/score):
  --inputs     <prepared>/<axis>/inputs.json   texts + members only
  --eval       <prepared>/<axis>/eval.json     cad_uids, text_roles, queries
  --score-dir  shard-shared scoring directory with counts.npy (int64 (n_cads, 6)
               candidate field-group counts) and chunks/t{index:05d}.npz
               (scores float64 (n_cads, 7): column 0 historical SUM, columns
               1..6 the fixed field-group sums sketch_x, sketch_y,
               extrude_numeric, extrude_boolean, delimiter, terminal).

fit (validation axis only; the full preregistered 48 background + 256 fit +
256 select text prefix is mandatory):
  1. reads every chunk with explicit reader checks (schema, shapes, binding
     fields, finiteness, |sum(groups) - SUM| <= 1e-8);
  2. builds the arm-0..7 configurations from the plan's "每臂可执行定义与
     优先级" table: native/simple baselines, background residual (lambda = 1
     fixed plus a lambda grid), log-mix background, fixed field weights,
     non-negative linear learned rankers fitted on the 256 fit texts
     (top-16-by-SUM + 16-nearest-valid-count hard negatives per positive,
     seed-20261118 equal-count random control), text-conditional gates,
     leave-one-group-out masks and the top-2-family reciprocal-rank fusion;
  3. selects by validation select-query R@20, then R@1, R@5, smaller MedR,
     then fixed enumeration order; every arm family keeps one representative;
  4. runs the exploratory 5,000-replicate paired text-group bootstrap
     (seed 20261118, original queries equally weighted);
  5. writes frozen.json (the exact configuration contract for the test axis),
     selection_table.json, fit_report.json, select_query_ranks.jsonl,
     select_bootstrap.json and SHA256SUMS. NO test artifact is read here.

evaluate (test axis): verifies --frozen against --expected-frozen-sha, loads
the frozen configuration WITHOUT any refitting, recomputes every frozen
configuration on the test gallery from the same 48 background texts, and
evaluates all fixed baselines and representatives under identical conditions:
per-query ranks, hit counts, point metrics, stratification, rerank deltas vs
SUM and the paired 5,000-replicate text-group bootstrap (seed 20261118).
--text-stop selects a preregistered test prefix (48 + {128,256,512,768,1024});
the gallery always stays complete.

Isolation: this script reads qrels-derived linkage ONLY through eval.json and
never writes anything a scorer consumes. numpy + stdlib only; no torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import text2cad_g5_cpu as g5  # noqa: E402  (audited rank/median kernels, tie rule)

SCHEMA_INPUTS = "text2cad-accuracy-inputs-v1"
SCHEMA_EVAL = "text2cad-accuracy-eval-v1"
SCHEMA_FROZEN = "text2cad-accuracy-frozen-v1"
SCHEMA_FIT_REPORT = "text2cad-accuracy-fit-report-v1"
SCHEMA_SELECTION = "text2cad-accuracy-selection-v1"
SCHEMA_BOOTSTRAP = "text2cad-accuracy-bootstrap-v1"
SCHEMA_EVAL_SUMMARY = "text2cad-accuracy-eval-summary-v1"

GROUP_NAMES = ("sketch_x", "sketch_y", "extrude_numeric", "extrude_boolean",
               "delimiter", "terminal")
N_GROUPS = len(GROUP_NAMES)
GROUP_TOTAL_TOL = 1e-8
N_BACKGROUND = 48
N_FIT = 256
N_SELECT = 256
KS = (1, 5, 10, 20)
SEED = 20261118
REPLICATES = 5000
LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25)
ALPHA_GRID = (0.01, 0.1, 1.0)
FIXED_WEIGHT_SETS = (
    ("fixed_equal", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    ("fixed_struct", (1.0, 1.0, 1.0, 2.0, 2.0, 1.0)),
    ("fixed_boolheavy", (1.0, 1.0, 1.0, 3.0, 3.0, 1.0)),
)
#: arm 5 text gates (plan: a word-gate hit doubles the boolean group's weight,
#: a digit-gate hit doubles the first three groups' weights; both may compose)
GATE_REGEX = r"\b(?:holes?|cuts?|slots?|subtract|subtracts|subtracted|subtracting)\b"
GATE_WORD_GROUP = 3            # extrude_boolean
GATE_WORD_MULTIPLIER = 2.0
GATE_DIGIT_GROUPS = (0, 1, 2)  # sketch_x, sketch_y, extrude_numeric
GATE_DIGIT_MULTIPLIER = 2.0
FUSION_K = 60
FIT_ITERATIONS = 2000
NEG_TOP_SUM = 16
NEG_NEAREST_COUNT = 16
B24_SIZE = 24  # fixed background-sensitivity subset (first 24 background texts)
TEST_PREFIXES = (1024, 768, 512, 256, 128)

ROLE_BACKGROUND, ROLE_FIT, ROLE_SELECT, ROLE_TEST = ("background", "fit", "select", "test")
BOOTSTRAP_PERCENTILES = (0.025, 0.975)


class EvalError(RuntimeError):
    """Hard refusal of a fit/evaluate stage; exit non-zero, nothing published."""


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_canonical_json(path, obj) -> str:
    path = Path(path)
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return sha256_bytes(data)


def write_jsonl(path, rows) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_out_dir(path) -> Path:
    out = Path(path)
    if out.exists():
        raise EvalError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    return out


def softplus_neg(z):
    return np.logaddexp(0.0, -z)


def sigmoid_neg(z):
    """Numerically stable sigmoid(-z)."""
    return np.exp(-np.logaddexp(0.0, z))


def logsumexp_axis0(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=0)
    return m + np.log(np.exp(x - m[None, :]).sum(axis=0))


def dense_query_groups(queries: list[dict]) -> np.ndarray:
    """Dense 0..g-1 text-group ids over a query list (bootstrap contract)."""
    distinct = sorted({q["text_index"] for q in queries})
    index = {t: i for i, t in enumerate(distinct)}
    return np.array([index[q["text_index"]] for q in queries], dtype=np.int64)


# ---------------------------------------------------------------------------
# artifact loading (explicit reader checks per the plan)
# ---------------------------------------------------------------------------


def load_axis(inputs_path: Path, eval_path: Path) -> dict:
    inputs_bytes = inputs_path.read_bytes()
    inputs_sha = sha256_bytes(inputs_bytes)
    payload = json.loads(inputs_bytes)
    if payload.get("schema") != SCHEMA_INPUTS:
        raise EvalError(f"inputs.json schema {payload.get('schema')!r} unexpected")
    axis = payload.get("axis")
    if axis not in ("validation", "test"):
        raise EvalError(f"inputs.json axis {axis!r} unknown")
    texts, members = payload.get("texts"), payload.get("members")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t for t in texts):
        raise EvalError("inputs.json texts must be non-empty strings")
    if not isinstance(members, list) or not members or not all(isinstance(m, str) and m for m in members):
        raise EvalError("inputs.json members must be non-empty strings")
    if len(set(texts)) != len(texts) or len(set(members)) != len(members):
        raise EvalError("inputs.json contains duplicated texts or candidate members")

    eval_bytes = eval_path.read_bytes()
    ev = json.loads(eval_bytes)
    if ev.get("schema") != SCHEMA_EVAL:
        raise EvalError(f"eval.json schema {ev.get('schema')!r} unexpected")
    if ev.get("axis") != axis:
        raise EvalError(f"eval.json axis {ev.get('axis')!r} != inputs axis {axis!r}")
    if ev.get("inputs_sha256") != inputs_sha:
        raise EvalError("eval.json inputs_sha256 does not match inputs.json bytes")
    cad_uids, roles, queries = ev.get("cad_uids"), ev.get("text_roles"), ev.get("queries")
    if (not isinstance(cad_uids, list) or len(cad_uids) != len(members)
            or not all(isinstance(uid, str) and uid for uid in cad_uids)
            or len(set(cad_uids)) != len(cad_uids)):
        raise EvalError("eval.json cad_uids must align with inputs members")
    if not isinstance(roles, list) or len(roles) != len(texts):
        raise EvalError("eval.json text_roles must align with inputs texts")
    if not isinstance(queries, list):
        raise EvalError("eval.json queries must be a list")
    seen_qids = set()
    seen_pairs = set()
    for q in queries:
        if not isinstance(q, dict):
            raise EvalError("eval.json queries must be objects")
        qid = q.get("query_id")
        if not isinstance(qid, str) or not qid or qid in seen_qids:
            raise EvalError(f"eval.json query_id {qid!r} empty or duplicated")
        seen_qids.add(qid)
        if type(q.get("text_index")) is not int or type(q.get("cad_index")) is not int:
            raise EvalError(f"eval.json {qid}: text_index/cad_index must be ints")
        if not 0 <= q["cad_index"] < len(members):
            raise EvalError(f"eval.json {qid}: cad_index outside the gallery")
        if not 0 <= q["text_index"] < len(texts):
            raise EvalError(f"eval.json {qid}: text_index outside the text axis")
        pair = (q["text_index"], q["cad_index"])
        if pair in seen_pairs or roles[pair[0]] == ROLE_BACKGROUND:
            raise EvalError(f"eval.json {qid}: duplicate positive or background qrel")
        seen_pairs.add(pair)
    if {t for t, _ in seen_pairs} != {i for i, role in enumerate(roles) if role != ROLE_BACKGROUND}:
        raise EvalError("eval.json does not provide positives for every non-background text")
    return {"axis": axis, "texts": texts, "members": members, "cad_uids": cad_uids,
            "roles": roles, "queries": queries, "inputs_sha256": inputs_sha,
            "eval_sha256": sha256_bytes(eval_bytes)}


def load_counts(score_dir: Path, n_cads: int) -> tuple[np.ndarray, str]:
    path = Path(score_dir) / "counts.npy"
    if not path.is_file():
        raise EvalError(f"counts.npy missing under {score_dir}")
    counts = np.load(path, allow_pickle=False)
    if counts.dtype != np.dtype("<i8") or counts.shape != (n_cads, N_GROUPS):
        raise EvalError(
            f"counts.npy must be int64 ({n_cads}, {N_GROUPS}), got {counts.dtype} {counts.shape}"
        )
    if (counts < 0).any() or (counts[:, 0] != counts[:, 1]).any() or (counts[:, -1] != 1).any():
        raise EvalError("counts.npy violates the field-count contract")
    sha = sha256_bytes(np.ascontiguousarray(counts, dtype=np.dtype("<i8")).tobytes(order="C"))
    return counts, sha


def load_chunks(score_dir: Path, text_indices, texts, inputs_sha: str,
                counts_sha: str, counts) -> dict:
    """Read chunks/t{index:05d}.npz with explicit binding/shape checks; no gaps."""
    chunks_dir = Path(score_dir) / "chunks"
    n_cads = len(counts)
    expected_names = {f"t{i:05d}.npz" for i in text_indices}
    if {path.name for path in chunks_dir.glob("*.npz")} != expected_names:
        raise EvalError("score chunks do not cover precisely the frozen text axis")
    absent = counts == 0
    n = len(text_indices)
    col0 = np.empty((n, n_cads), dtype=np.float64)
    groups = np.empty((n, n_cads, N_GROUPS), dtype=np.float64)
    chunk_shas = {}
    for row, text_index in enumerate(text_indices):
        path = chunks_dir / f"t{text_index:05d}.npz"
        if not path.is_file():
            raise EvalError(f"missing chunk for text {text_index}: {path}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"scores", "text_index", "text_sha256",
                                   "inputs_sha256", "counts_sha256"}:
                raise EvalError(f"{path.name}: unexpected keys {sorted(data.files)}")
            scores = data["scores"]
            text_slot = data["text_index"]
            if text_slot.ndim != 0 or text_slot.dtype.kind not in "iu":
                raise EvalError(f"{path.name}: expected an integer scalar text_index")
            hashes = tuple(data[field] for field in ("text_sha256", "inputs_sha256", "counts_sha256"))
            if any(slot.ndim != 0 or slot.dtype.kind != "U" for slot in hashes):
                raise EvalError(f"{path.name}: expected Unicode scalar digests")
            stored = (text_slot.item(), *(slot.item() for slot in hashes))
        if scores.dtype != np.float64 or scores.shape != (n_cads, 1 + N_GROUPS):
            raise EvalError(
                f"{path.name}: scores must be float64 ({n_cads}, {1 + N_GROUPS}), "
                f"got {scores.dtype} {scores.shape}"
            )
        if stored[0] != text_index:
            raise EvalError(f"{path.name}: text_index {stored[0]} != {text_index}")
        if stored[1] != hashlib.sha256(texts[text_index].encode("utf-8")).hexdigest():
            raise EvalError(f"{path.name}: text_sha256 does not match inputs.json texts")
        if stored[2] != inputs_sha:
            raise EvalError(f"{path.name}: inputs_sha256 mismatch")
        if stored[3] != counts_sha:
            raise EvalError(f"{path.name}: counts_sha256 mismatch")
        if not np.isfinite(scores).all():
            raise EvalError(f"{path.name}: non-finite scores")
        if np.any(scores[:, 1:][absent] != 0):
            raise EvalError(f"{path.name}: nonzero score in an absent group")
        drift = float(np.abs(scores[:, 1:].sum(axis=1) - scores[:, 0]).max())
        if drift > GROUP_TOTAL_TOL:
            raise EvalError(
                f"{path.name}: |sum(groups) - SUM| = {drift:.3e} > {GROUP_TOTAL_TOL}"
            )
        col0[row] = scores[:, 0]
        groups[row] = scores[:, 1:]
        chunk_shas[text_index] = sha256_file(path)
    return {"sum": col0, "groups": groups, "chunk_sha256": chunk_shas}


def require_roles(roles, axis: str) -> None:
    got = list(roles)
    if axis == "validation":
        want = ([ROLE_BACKGROUND] * N_BACKGROUND + [ROLE_FIT] * N_FIT
                + [ROLE_SELECT] * N_SELECT)
        if got != want:
            raise EvalError("validation text_roles must be exactly 48 background / "
                            "256 fit / 256 select")
    else:
        if len(got) <= N_BACKGROUND or any(r != ROLE_BACKGROUND for r in got[:N_BACKGROUND]):
            raise EvalError("test text_roles must start with exactly 48 background texts")
        if any(r != ROLE_TEST for r in got[N_BACKGROUND:]):
            raise EvalError("test text_roles after the background prefix must all be 'test'")


class ScoringContext:
    """Per-axis inputs for every arm: SUM/group columns, counts, backgrounds."""

    def __init__(self, matrix: dict, counts: np.ndarray, texts, text_indices) -> None:
        self.sum = matrix["sum"]                      # (T, n_cads)
        self.groups = matrix["groups"]                # (T, n_cads, 6)
        self.counts = counts                          # (n_cads, 6)
        self.n_total = counts.sum(axis=1)             # (n_cads,)
        self.inv = 1.0 / np.maximum(1, counts)        # (n_cads, 6)
        self.texts = list(texts)
        self.text_indices = list(text_indices)
        self.text_row = {t: r for r, t in enumerate(self.text_indices)}
        self.n_cads = counts.shape[0]
        if len(self.text_indices) <= N_BACKGROUND:
            raise EvalError("scored range must include the 48 background texts")
        bg = slice(0, N_BACKGROUND)
        self.background = self.sum[bg].mean(axis=0)                       # (n_cads,)
        self.background_log = logsumexp_axis0(self.sum[bg]) - np.log(N_BACKGROUND)
        self.background_g = self.groups[bg].mean(axis=0)                  # (n_cads, 6)
        self.background24 = self.sum[:B24_SIZE].mean(axis=0)
        self.background24_log = logsumexp_axis0(self.sum[:B24_SIZE]) - np.log(B24_SIZE)
        self.background24_g = self.groups[:B24_SIZE].mean(axis=0)         # (n_cads, 6)

    def raw_means(self, t: int) -> np.ndarray:
        """A_g = G_g / max(1, n_g) for text row t (absent groups score 0)."""
        return self.groups[t] * self.inv

    def bg_means(self, t: int) -> np.ndarray:
        return (self.groups[t] - self.background_g) * self.inv

    def bg24_means(self, t: int) -> np.ndarray:
        return (self.groups[t] - self.background24_g) * self.inv


# ---------------------------------------------------------------------------
# configurations (the arm table) and their score vectors
# ---------------------------------------------------------------------------


def gate_multipliers(text: str, gates) -> np.ndarray:
    mult = np.ones(N_GROUPS, dtype=np.float64)
    if "word" in gates and re.search(GATE_REGEX, text.casefold()):
        mult[GATE_WORD_GROUP] *= GATE_WORD_MULTIPLIER
    if "digit" in gates and re.search(r"[0-9]", text):
        for g in GATE_DIGIT_GROUPS:
            mult[g] *= GATE_DIGIT_MULTIPLIER
    return mult


def config_score_vector(cfg: dict, t: int, ctx: ScoringContext) -> np.ndarray:
    """Full-gallery score vector for one configuration at text row t.

    Tie handling happens downstream (g5.score_ranks breaks score ties by
    candidate index ascending, per the plan).
    """
    kind = cfg["kind"]
    if kind == "sum":
        return ctx.sum[t].copy()
    if kind == "mean":
        return ctx.sum[t] / ctx.n_total
    if kind == "count_asc":
        return -ctx.n_total.astype(np.float64)
    if kind == "resid":
        background = ctx.background24 if cfg.get("b24") else ctx.background
        return ctx.sum[t] - cfg["lambda"] * background
    if kind == "logmix":
        background = ctx.background24_log if cfg.get("b24") else ctx.background_log
        return ctx.sum[t] - cfg["lambda"] * background
    if kind in ("fixed", "fit_bg", "loo"):
        feats = ctx.bg24_means(t) if cfg.get("b24") else ctx.bg_means(t)
        return feats @ np.asarray(cfg["weights"], dtype=np.float64)
    if kind == "fit_raw":
        return ctx.raw_means(t) @ np.asarray(cfg["weights"], dtype=np.float64)
    if kind == "gate":
        feats = ctx.bg_means(t)
        mult = gate_multipliers(ctx.texts[t], cfg["gates"])
        return feats @ (np.asarray(cfg["weights"], dtype=np.float64) * mult)
    if kind == "fusion":
        parts = []
        for member in cfg["member_cfg"]:      # injected; order matters
            vec = config_score_vector(member, t, ctx)
            parts.append(1.0 / (FUSION_K + g5.score_ranks(vec)))
        return np.mean(parts, axis=0)
    raise EvalError(f"config {cfg.get('id')!r}: unknown kind {kind!r}")


def point_metrics(ranks: np.ndarray) -> dict:
    metrics = g5.rank_metrics(ranks)
    metrics["hit_counts"] = {f"hit@{k}": int((np.asarray(ranks) <= k).sum()) for k in KS}
    return metrics


def selection_key(metrics: dict) -> tuple:
    """Plan's selection order: R@20 desc, R@1 desc, R@5 desc, MedR asc, enum."""
    return (-metrics["R@20"], -metrics["R@1"], -metrics["R@5"], metrics["MedR"])


def select_best(metrics: dict, ids: list[str]) -> str:
    """Fixed enumeration order = position inside the given id list."""
    best = None
    for position, cid in enumerate(ids):
        key = (selection_key(metrics[cid]), position)
        if best is None or key < best[0]:
            best = (key, cid)
    return best[1]


def evaluate_configs(configs: list[dict], ctx: ScoringContext,
                     queries: list[dict], *, include_top=False) -> dict:
    """Per-query gallery ranks for every config + point metrics + hit counts."""
    by_id = {cfg["id"]: cfg for cfg in configs}
    for cfg in configs:
        if cfg["kind"] == "fusion":
            members = [by_id[mid] for mid in cfg["members"]]
            if len(members) != len(cfg["members"]):
                raise EvalError(f"fusion {cfg['id']}: member configs missing")
            cfg["member_cfg"] = members
    n_queries = len(queries)
    rank_matrix = np.empty((n_queries, len(configs)), dtype=np.int64)
    top_by_text = {}
    for ci, cfg in enumerate(configs):
        last_t = None
        ranks = None
        for i, q in enumerate(queries):
            if q["text_index"] != last_t:
                last_t = q["text_index"]
                ranks = g5.score_ranks(config_score_vector(cfg, ctx.text_row[last_t], ctx))
                if include_top:
                    top = np.flatnonzero(ranks <= 20)
                    top_by_text.setdefault(last_t, {})[cfg["id"]] = top[np.argsort(ranks[top])].tolist()
            rank_matrix[i, ci] = ranks[q["cad_index"]]
    methods = [cfg["id"] for cfg in configs]
    metrics = {m: point_metrics(rank_matrix[:, j]) for j, m in enumerate(methods)}
    return {"rank_matrix": rank_matrix, "methods": methods, "metrics": metrics,
            "top_by_text": top_by_text,
            "query_groups": dense_query_groups(queries)}


def stratified_metrics(rank_matrix: np.ndarray, methods: list[str],
                       queries: list[dict], ctx: ScoringContext) -> dict:
    """Text-group-size and positive-valid-count strata (descriptive)."""
    positives = np.array([q["cad_index"] for q in queries], dtype=np.int64)
    size_counter = Counter(q["text_index"] for q in queries)
    sizes = np.array([size_counter[q["text_index"]] for q in queries], dtype=np.int64)
    n_total = ctx.n_total[positives]
    cuts = np.quantile(n_total, [0.25, 0.5, 0.75])
    bins = np.searchsorted(cuts, n_total, side="left")
    masks = {"unique_text_group": sizes == 1, "repeated_text_group": sizes > 1}
    for i in range(4):
        masks[f"positive_valid_count_quartile_{i + 1}"] = bins == i
    return {
        "gallery_valid_count_quartile_cutpoints": [float(c) for c in cuts],
        "cut_rule": "lower bin includes the exact boundary (np.searchsorted left)",
        "strata": {name: {m: point_metrics(rank_matrix[mask, j])
                          for j, m in enumerate(methods)}
                   for name, mask in masks.items()},
    }


# ---------------------------------------------------------------------------
# fit-stage model fitting (arm 4 / arm 6)
# ---------------------------------------------------------------------------


def text_positives(queries: list[dict]) -> dict:
    out: dict[int, set] = {}
    for q in queries:
        out.setdefault(q["text_index"], set()).add(q["cad_index"])
    return out


def build_negative_pairs(ctx: ScoringContext, fit_queries: list[dict],
                         positives: dict, source: str,
                         rng: np.random.Generator) -> list[dict]:
    """Fixed-order negative picks; uniform count per positive.

    hard: top-16 remaining by the same query's SUM, then 16 further remaining
    nearest in valid-field count (ties by candidate index ascending; the two
    blocks are disjoint so every positive carries exactly 32). Near negatives
    are defined by registered UID and may still match the coarse description.
    random: seed-controlled equal-count draw without replacement from the
    positive-free pool (seed 20261118 per the plan).
    """
    pairs = []
    for q in sorted(fit_queries, key=lambda r: (r["text_index"], r["cad_index"])):
        t, pos = q["text_index"], q["cad_index"]
        pos_set = positives[t]
        pool = np.array([c for c in range(ctx.n_cads) if c not in pos_set], dtype=np.int64)
        if source == "random":
            picks = np.sort(rng.choice(pool, size=NEG_TOP_SUM + NEG_NEAREST_COUNT,
                                       replace=False))
            negs = [("random", int(c)) for c in picks]
        else:
            order = np.lexsort((pool, -ctx.sum[t][pool]))
            block1 = pool[order[:NEG_TOP_SUM]]
            chosen = {int(c) for c in block1}
            rest = np.array([c for c in pool if c not in chosen], dtype=np.int64)
            diff = np.abs(ctx.n_total[rest] - ctx.n_total[pos])
            block2 = rest[np.lexsort((rest, diff))[:NEG_NEAREST_COUNT]]
            negs = [("sum_hard", int(c)) for c in block1] + \
                   [("count_hard", int(c)) for c in block2]
        for neg_kind, neg in negs:
            pairs.append({"text_index": t, "positive": pos, "negative": neg,
                          "negative_kind": neg_kind})
    return pairs


def pair_features(ctx: ScoringContext, pairs: list[dict], variant: str) -> np.ndarray:
    """(n_pairs, 6) positive-minus-negative feature differences."""
    feats = np.empty((len(pairs), N_GROUPS), dtype=np.float64)
    blocks: dict = {}
    for i, p in enumerate(pairs):
        t = p["text_index"]
        if t not in blocks:
            blocks[t] = ctx.raw_means(t) if variant == "raw" else ctx.bg_means(t)
        block = blocks[t]
        feats[i] = block[p["positive"]] - block[p["negative"]]
    return feats


def rms_scales(diffs: np.ndarray) -> np.ndarray:
    """Per-dimension RMS standardization; constant-zero dimensions scale 1."""
    rms = np.sqrt((np.asarray(diffs, dtype=np.float64) ** 2).mean(axis=0))
    return np.where(rms > 0.0, rms, 1.0)


def fit_nonnegative_weights(diffs: np.ndarray, alpha: float,
                            iterations: int = FIT_ITERATIONS) -> dict:
    """Projected gradient on mean softplus(-Dw) + alpha*||w||^2 with w >= 0.

    Step size from the logistic-loss curvature bound
    L = 0.25 * lambda_max(D^T D / n) + 2*alpha (per-sample Hessian
    sigma(z)sigma(-z) dd^T <= 0.25 dd^T); fixed iteration count; convergence
    recorded as the final gradient infinity-norm plus a loss trace. RMS
    standardization happens upstream (fit-set only; frozen for select/test).
    """
    d = np.ascontiguousarray(diffs, dtype=np.float64)
    n = d.shape[0]
    if n == 0:
        raise EvalError("fit refused: empty pair set")
    lam_max = float(np.linalg.eigvalsh((d.T @ d) / n)[-1])
    step = 1.0 / (0.25 * lam_max + 2.0 * alpha)
    w = np.zeros(d.shape[1], dtype=np.float64)
    trace = []
    for it in range(iterations):
        z = d @ w
        grad = -(d * sigmoid_neg(z)[:, None]).mean(axis=0) + 2.0 * alpha * w
        if it % 500 == 0 or it == iterations - 1:
            trace.append({"iteration": it,
                          "loss": float(softplus_neg(z).mean() + alpha * float(w @ w))})
        w = np.maximum(w - step * grad, 0.0)
    z = d @ w
    grad = -(d * sigmoid_neg(z)[:, None]).mean(axis=0) + 2.0 * alpha * w
    return {
        "weights": [float(v) for v in w],
        "alpha": float(alpha),
        "iterations": int(iterations),
        "step_size": float(step),
        "curvature_bound_lipschitz": float(0.25 * lam_max + 2.0 * alpha),
        "final_loss": float(softplus_neg(z).mean() + alpha * float(w @ w)),
        "final_grad_inf_norm": float(np.abs(grad).max()),
        "loss_trace": trace,
    }


def mechanism_diagnostics(ctx: ScoringContext, hard_pairs: list[dict]) -> dict:
    """Per-group positive/negative difference median, strict win rate, ties."""
    out = {}
    for variant in ("raw", "bg"):
        feats = pair_features(ctx, hard_pairs, variant)
        out[variant] = {}
        for g, name in enumerate(GROUP_NAMES):
            col = feats[:, g]
            out[variant][name] = {
                "median_diff": float(np.median(col)),
                "strict_win_rate": float((col > 0).mean()),
                "tie_rate": float((col == 0).mean()),
            }
    return out


def fit_all(ctx: ScoringContext, fit_queries: list[dict]) -> dict:
    """Fit every (negative source, feature variant, alpha) weight vector."""
    positives = text_positives(fit_queries)
    pairs = {
        "hard": build_negative_pairs(ctx, fit_queries, positives, "hard",
                                     np.random.default_rng(SEED)),
        "random": build_negative_pairs(ctx, fit_queries, positives, "random",
                                       np.random.default_rng(SEED)),
    }
    variants = {"hard": ("raw", "bg"), "random": ("bg",)}
    report = {
        "seed": SEED,
        "negatives_per_positive": NEG_TOP_SUM + NEG_NEAREST_COUNT,
        "neg_top_sum": NEG_TOP_SUM,
        "neg_nearest_count": NEG_NEAREST_COUNT,
        "alpha_grid": [float(a) for a in ALPHA_GRID],
        "iterations": FIT_ITERATIONS,
        "n_fit_queries": len(fit_queries),
        "n_fit_groups": len({q["text_index"] for q in fit_queries}),
        "pair_weighting": "uniform per original query (each positive is one query row)",
        "feature_definition": {
            "raw": "A_g = G_g / max(1, n_g)",
            "bg": "(G_g - B_g) / max(1, n_g), B_g = mean over the 48 background texts",
        },
        "sources": {},
        "mechanism_diagnostics": mechanism_diagnostics(ctx, pairs["hard"]),
    }
    weights = {}
    for source in ("hard", "random"):
        entry = {"n_pairs": len(pairs[source]), "variants": {}}
        for variant in variants[source]:
            diffs = pair_features(ctx, pairs[source], variant)
            scales = rms_scales(diffs)
            std = diffs / scales[None, :]
            sub = {
                "scales": [float(s) for s in scales],
                "diffs_sha256": sha256_bytes(np.ascontiguousarray(diffs).tobytes(order="C")),
                "fits": {f"a{alpha:g}": fit_nonnegative_weights(std, alpha)
                         for alpha in ALPHA_GRID},
            }
            entry["variants"][variant] = sub
            for alpha in ALPHA_GRID:
                standardized = np.asarray(sub["fits"][f"a{alpha:g}"]["weights"], dtype=np.float64)
                weights[(source, variant, alpha)] = (standardized / scales).tolist()
        report["sources"][source] = entry
    return {"report": report, "weights": weights, "pairs": pairs}


# ---------------------------------------------------------------------------
# bootstrap (paired text-group resampling, vectorized weighted medians)
# ---------------------------------------------------------------------------


def draw_counts_matrix(n_groups: int, replicates: int, seed: int) -> np.ndarray:
    """Same draw construction as the certified G5 clustered bootstrap."""
    rng = np.random.default_rng(seed)
    counts = np.empty((replicates, n_groups), dtype=np.int16)
    for i in range(replicates):
        counts[i] = np.bincount(rng.integers(0, n_groups, size=n_groups),
                                minlength=n_groups)
    return counts


def bootstrap_samples(rank_matrix: np.ndarray, groups: np.ndarray,
                      draws: np.ndarray) -> dict:
    ranks = np.asarray(rank_matrix, dtype=np.int64)
    grp = np.asarray(groups, dtype=np.int64)
    n_methods = ranks.shape[1]
    n_groups = int(grp.max()) + 1
    group_sizes = np.bincount(grp, minlength=n_groups).astype(np.int64)
    counts64 = draws.astype(np.int64)
    denominators = counts64 @ group_sizes
    if (denominators <= 0).any():
        raise EvalError("bootstrap draw produced an empty resample")
    samples: dict = {}
    for k in KS:
        hits = np.zeros((n_groups, n_methods), dtype=np.int64)
        np.add.at(hits, grp, (ranks <= k).astype(np.int64))
        samples[f"R@{k}"] = (counts64 @ hits) / denominators[:, None]
    medians = np.empty((draws.shape[0], n_methods), dtype=np.float64)
    for j in range(n_methods):
        sorder = np.argsort(ranks[:, j], kind="stable")
        sorted_ranks = ranks[sorder, j]
        sgroups = grp[sorder]
        cum = np.cumsum(counts64[:, sgroups], axis=1)
        total = denominators[:, None]
        lo = (cum >= (total + 1) // 2).argmax(axis=1)
        hi = (cum >= total // 2 + 1).argmax(axis=1)
        medians[:, j] = (sorted_ranks[lo] + sorted_ranks[hi]) / 2.0
        del sgroups
    samples["MedR"] = medians
    samples["denominators"] = denominators
    return samples


def summarize_bootstrap(samples: dict, methods: list[str], baseline: str) -> dict:
    baseline_index = methods.index(baseline)
    out = {
        "replicates": int(samples["R@1"].shape[0]),
        "seed": SEED,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "resampling_unit": "unique_text_group",
        "query_weighted": True,
        "baseline": baseline,
        "denominator_min": int(samples["denominators"].min()),
        "denominator_max": int(samples["denominators"].max()),
    }
    ci, gain = {}, {}
    for j, name in enumerate(methods):
        entry = {f"R@{k}": np.quantile(samples[f"R@{k}"][:, j], BOOTSTRAP_PERCENTILES).tolist()
                 for k in KS}
        entry["MedR"] = np.quantile(samples["MedR"][:, j], BOOTSTRAP_PERCENTILES).tolist()
        ci[name] = entry
        if j != baseline_index:
            gain[name] = {
                f"R@{k}": np.quantile(samples[f"R@{k}"][:, j] - samples[f"R@{k}"][:, baseline_index],
                                      BOOTSTRAP_PERCENTILES).tolist()
                for k in KS
            }
            gain[name]["MedR"] = np.quantile(
                samples["MedR"][:, j] - samples["MedR"][:, baseline_index], BOOTSTRAP_PERCENTILES
            ).tolist()
    out["method_ci95"] = ci
    out[f"paired_gain_vs_{baseline}_ci95"] = gain
    return out


# ---------------------------------------------------------------------------
# configuration construction (fixed enumeration order)
# ---------------------------------------------------------------------------


def make_config(config_id: str, family: int, kind: str, **params) -> dict:
    if "lambda_" in params:
        params["lambda"] = params.pop("lambda_")
    return {"id": config_id, "family": int(family), "kind": kind, **params}


def build_base_configs(fitted: dict) -> list[dict]:
    """Arms 0-4 and 6 in fixed enumeration order."""
    configs = [
        make_config("sum", 0, "sum"),
        make_config("mean", 0, "mean"),
        make_config("count_asc", 0, "count_asc"),
        make_config("resid_l1", 1, "resid", lambda_=1.0),
    ]
    configs += [make_config(f"resid_grid_lam{lam:g}", 1, "resid", lambda_=float(lam))
                for lam in LAMBDA_GRID]
    configs.append(make_config("logmix_l1", 2, "logmix", lambda_=1.0))
    configs += [make_config(f"logmix_grid_lam{lam:g}", 2, "logmix", lambda_=float(lam))
                for lam in LAMBDA_GRID]
    configs += [make_config(cid, 3, "fixed", weights=[float(w) for w in weights])
                for cid, weights in FIXED_WEIGHT_SETS]
    for alpha in ALPHA_GRID:
        configs.append(make_config(f"fit_raw_a{alpha:g}", 4, "fit_raw",
                                   weights=fitted["weights"][("hard", "raw", alpha)],
                                   alpha=float(alpha), source="hard", variant="raw"))
        configs.append(make_config(f"fit_bg_a{alpha:g}", 4, "fit_bg",
                                   weights=fitted["weights"][("hard", "bg", alpha)],
                                   alpha=float(alpha), source="hard", variant="bg"))
        configs.append(make_config(f"fit_bg_rand_a{alpha:g}", 6, "fit_bg",
                                   weights=fitted["weights"][("random", "bg", alpha)],
                                   alpha=float(alpha), source="random", variant="bg"))
    return configs


def derive_late_configs(base_metrics: dict, base_configs: list[dict]) -> list[dict]:
    """Arms 5 and 7, derived from the base selection (validation-defined)."""
    metrics = base_metrics
    family_ids: dict = {}
    for cfg in base_configs:
        family_ids.setdefault(cfg["family"], []).append(cfg["id"])
    representatives = {f: select_best(metrics, ids) for f, ids in sorted(family_ids.items())}

    bg_ids = [c["id"] for c in base_configs if c["kind"] == "fit_bg" and c["source"] == "hard"]
    bg_best_id = select_best(metrics, bg_ids)
    gate_weights = list(next(c["weights"] for c in base_configs if c["id"] == bg_best_id))
    late = [
        make_config("gate_word", 5, "gate", weights=gate_weights, gates=["word"]),
        make_config("gate_digit", 5, "gate", weights=gate_weights, gates=["digit"]),
        make_config("gate_both", 5, "gate", weights=gate_weights, gates=["word", "digit"]),
    ]
    for g in range(N_GROUPS):
        loo_weights = list(gate_weights)
        loo_weights[g] = 0.0
        late.append(make_config(f"loo_drop{g}", 7, "loo", weights=loo_weights,
                                dropped_group=GROUP_NAMES[g],
                                base_config=bg_best_id))

    return late, {
        "family_representatives_base": {str(f): representatives[f]
                                        for f in sorted(representatives)},
        "background_group_model": bg_best_id,
    }


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def run_fit(args) -> int:
    started = time.time()
    out = prepare_out_dir(args.out)
    axis = load_axis(Path(args.inputs), Path(args.eval))
    if axis["axis"] != "validation":
        raise EvalError("fit runs on the validation axis only")
    require_roles(axis["roles"], "validation")

    counts, counts_sha = load_counts(Path(args.score_dir), len(axis["members"]))
    all_indices = list(range(N_BACKGROUND + N_FIT + N_SELECT))
    matrix = load_chunks(Path(args.score_dir), all_indices, axis["texts"],
                         axis["inputs_sha256"], counts_sha, counts)
    ctx = ScoringContext(matrix, counts, axis["texts"], all_indices)

    queries = axis["queries"]
    fit_queries = [q for q in queries
                   if N_BACKGROUND <= q["text_index"] < N_BACKGROUND + N_FIT]
    select_queries = [q for q in queries
                      if N_BACKGROUND + N_FIT <= q["text_index"] < N_BACKGROUND + N_FIT + N_SELECT]
    if not fit_queries or not select_queries:
        raise EvalError("eval.json holds no fit/select queries")

    fitted = fit_all(ctx, fit_queries)
    base_configs = build_base_configs(fitted)
    base_eval = evaluate_configs(base_configs, ctx, select_queries)

    late_configs, derive_info = derive_late_configs(base_eval["metrics"], base_configs)
    configs = base_configs + late_configs
    late_eval = evaluate_configs(configs, ctx, select_queries)
    family_ids: dict = {}
    for cfg in configs:
        family_ids.setdefault(cfg["family"], []).append(cfg["id"])
    reps = {f: select_best(late_eval["metrics"], ids) for f, ids in sorted(family_ids.items())}
    fusion_members = sorted(
        (name for family, name in reps.items() if family != 7),
        key=lambda name: (selection_key(late_eval["metrics"][name]), late_eval["methods"].index(name)),
    )[:2]
    configs.append(make_config("fusion_top2", 7, "fusion", members=fusion_members))
    family_ids[7].append("fusion_top2")
    derive_info["fusion_members"] = fusion_members
    late_eval = evaluate_configs(configs, ctx, select_queries)
    metrics = late_eval["metrics"]
    methods = late_eval["methods"]
    representatives = {f: select_best(metrics, ids) for f, ids in sorted(family_ids.items())}
    selected_config = select_best(metrics, list(methods))

    # fixed first-24-background sensitivity for the tuned arm-1/arm-2 selections
    sensitivity = {}
    sensitivity_configs = []
    for base, grid_prefix, family in (("resid", "resid_grid", 1), ("logmix", "logmix_grid", 2)):
        grid_ids = [c["id"] for c in configs if c["id"].startswith(grid_prefix)]
        tuned = select_best(metrics, grid_ids)
        lam = float(next(c["lambda"] for c in configs if c["id"] == tuned))
        b24_cfg = make_config(f"b24_{base}_lam{lam:g}", family, base,
                              lambda_=lam, b24=True)
        sensitivity_configs.append(b24_cfg)
        b24_eval = evaluate_configs([b24_cfg], ctx, select_queries)
        sensitivity[base] = {
            "tuned_config": tuned,
            "tuned_lambda": lam,
            "b24_config": b24_cfg["id"],
            "b24_metrics": b24_eval["metrics"][b24_cfg["id"]],
        }

    # exploratory paired text-group bootstrap over the select split
    draws = draw_counts_matrix(int(late_eval["query_groups"].max()) + 1, REPLICATES, SEED)
    samples = bootstrap_samples(late_eval["rank_matrix"], late_eval["query_groups"], draws)
    boot_summary = summarize_bootstrap(samples, methods, "sum")

    select_rows = []
    for i, q in enumerate(select_queries):
        select_rows.append({
            "query_id": q["query_id"], "text_index": q["text_index"],
            "cad_index": q["cad_index"],
            "ranks": {m: int(late_eval["rank_matrix"][i, j])
                      for j, m in enumerate(methods)},
        })
    write_jsonl(out / "select_query_ranks.jsonl", select_rows)

    selection_table = {
        "schema": SCHEMA_SELECTION,
        "selection_rule": "R@20 desc, R@1 desc, R@5 desc, MedR asc, fixed enumeration order",
        "n_select_queries": len(select_queries),
        "n_select_groups": len({q["text_index"] for q in select_queries}),
        "point_metrics": metrics,
        **derive_info,
        "family_representatives": {str(f): representatives[f]
                                   for f in sorted(representatives)},
        "selected_config": selected_config,
        "background24_sensitivity": sensitivity,
        "enumeration": [c["id"] for c in configs],
    }
    selection_sha = write_canonical_json(out / "selection_table.json", selection_table)

    fit_report = {
        "schema": SCHEMA_FIT_REPORT,
        **fitted["report"],
        "hard_negative_pairs": len(fitted["pairs"]["hard"]),
        "random_negative_pairs": len(fitted["pairs"]["random"]),
        "fit_query_axis_fingerprint": sha256_bytes(np.asarray(
            [[q["text_index"], q["cad_index"]] for q in sorted(
                fit_queries, key=lambda r: (r["text_index"], r["cad_index"]))],
            dtype=np.int64).tobytes(order="C")),
        "validation_inputs_sha256": axis["inputs_sha256"],
        "validation_counts_sha256": counts_sha,
        "n_scored_chunks": len(all_indices),
    }
    fit_sha = write_canonical_json(out / "fit_report.json", fit_report)

    # Freeze the controls and one representative for each tuned comparison.
    best_background = select_best(metrics, [item["tuned_config"] for item in sensitivity.values()])
    controls = {"sum", "mean", "count_asc", "resid_l1", "logmix_l1", "fixed_equal",
                "fixed_struct", "fixed_boolheavy", "gate_word", "gate_digit", "gate_both",
                "fusion_top2", selected_config, *fusion_members, *representatives.values()}
    controls.update(f"loo_drop{g}" for g in range(N_GROUPS))
    controls.update(item["tuned_config"] for item in sensitivity.values())
    for prefix in ("fit_raw_", "fit_bg_a", "fit_bg_rand_"):
        controls.add(select_best(metrics, [m for m in methods if m.startswith(prefix)]))
    export_configs = [
        {k: v for k, v in cfg.items() if k != "member_cfg"}
        for cfg in configs if cfg["id"] in controls
    ] + sensitivity_configs
    frozen = {
        "schema": SCHEMA_FROZEN,
        "plan": "experiments/plans/IDEA01_TEXT2CAD_RETRIEVAL_ACCURACY_2026-11-18.md",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_inputs_sha256": axis["inputs_sha256"],
        "validation_eval_sha256": axis["eval_sha256"],
        "background_texts_sha256": [
            sha256_bytes(t.encode("utf-8")) for t in axis["texts"][:N_BACKGROUND]
        ],
        "validation_counts_sha256": counts_sha,
        "n_validation_chunks": len(all_indices),
        "group_names": list(GROUP_NAMES),
        "seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "lambda_grid": [float(l) for l in LAMBDA_GRID],
        "alpha_grid": [float(a) for a in ALPHA_GRID],
        "fixed_weight_sets": {cid: [float(w) for w in weights]
                              for cid, weights in FIXED_WEIGHT_SETS},
        "gate_spec": {
            "regex": GATE_REGEX,
            "word_group": GATE_WORD_GROUP,
            "word_multiplier": GATE_WORD_MULTIPLIER,
            "digit_groups": list(GATE_DIGIT_GROUPS),
            "digit_multiplier": GATE_DIGIT_MULTIPLIER,
            "application": "casefolded regex hit doubles the word group's weight; "
                           "an ASCII digit hit doubles the digit groups' weights; both compose",
        },
        "configs": export_configs,
        "best_background_config": best_background,
        **derive_info,
        "family_representatives": {str(f): representatives[f]
                                   for f in sorted(representatives)},
        "selected_config": selected_config,
        "select_point_metrics": metrics,
        "selection_rule": selection_table["selection_rule"],
    }
    frozen_sha = write_canonical_json(out / "frozen.json", frozen)
    boot_sha = write_canonical_json(out / "select_bootstrap.json", {
        "schema": SCHEMA_BOOTSTRAP, "stage": "validation_select_exploratory",
        **boot_summary})

    lines = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({
        "status": "fit_complete", "out": str(out),
        "n_configs": len(configs),
        "selected_config": selected_config,
        "selected_r20": metrics[selected_config]["R@20"],
        "sum_r20": metrics["sum"]["R@20"],
        "family_representatives": {str(f): representatives[f]
                                   for f in sorted(representatives)},
        "frozen_sha256": frozen_sha,
        "selection_table_sha256": selection_sha,
        "fit_report_sha256": fit_sha,
        "bootstrap_sha256": boot_sha,
        "elapsed_seconds": round(time.time() - started, 1),
    }, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def run_evaluate(args) -> int:
    started = time.time()
    frozen_sha = sha256_file(args.frozen)
    if frozen_sha != args.expected_frozen_sha:
        raise EvalError(
            f"frozen.json sha256 {frozen_sha} != expected {args.expected_frozen_sha}"
        )
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    if frozen.get("schema") != SCHEMA_FROZEN:
        raise EvalError(f"frozen schema {frozen.get('schema')!r} unexpected")
    if frozen.get("group_names") != list(GROUP_NAMES):
        raise EvalError("frozen group column order differs from this evaluator")
    configs = frozen.get("configs")
    if not isinstance(configs, list) or not configs:
        raise EvalError("frozen configuration set is empty")

    out = prepare_out_dir(args.out)
    axis = load_axis(Path(args.inputs), Path(args.eval))
    if axis["axis"] != "test":
        raise EvalError("evaluate runs on the test axis only")
    if frozen.get("background_texts_sha256") != [
        sha256_bytes(t.encode("utf-8")) for t in axis["texts"][:N_BACKGROUND]
    ]:
        raise EvalError("test gallery background differs from the frozen validation background")

    text_stop = int(args.text_stop)
    if text_stop <= N_BACKGROUND or (text_stop - N_BACKGROUND) not in TEST_PREFIXES:
        raise EvalError(
            f"--text-stop minus {N_BACKGROUND} must be one of {TEST_PREFIXES}, "
            f"got {text_stop - N_BACKGROUND}"
        )
    if text_stop > len(axis["texts"]):
        raise EvalError(f"--text-stop {text_stop} exceeds the test axis ({len(axis['texts'])})")
    require_roles(axis["roles"][:text_stop], "test")

    n_cads = len(axis["members"])
    counts, counts_sha = load_counts(Path(args.score_dir), n_cads)
    all_indices = list(range(text_stop))
    matrix = load_chunks(Path(args.score_dir), all_indices, axis["texts"],
                         axis["inputs_sha256"], counts_sha, counts)
    ctx = ScoringContext(matrix, counts, axis["texts"], all_indices)
    queries = [q for q in axis["queries"] if q["text_index"] < text_stop]
    if not queries:
        raise EvalError("no test queries inside the requested prefix")

    eval_result = evaluate_configs(configs, ctx, queries, include_top=True)
    metrics = eval_result["metrics"]
    methods = eval_result["methods"]
    if "sum" not in methods:
        raise EvalError("frozen configuration set lacks the 'sum' baseline")

    draws = draw_counts_matrix(int(eval_result["query_groups"].max()) + 1,
                               int(frozen.get("bootstrap_replicates", REPLICATES)),
                               int(frozen.get("seed", SEED)))
    samples = bootstrap_samples(eval_result["rank_matrix"], eval_result["query_groups"], draws)
    boot = summarize_bootstrap(samples, methods, "sum")
    best_background = frozen["best_background_config"]
    boot.update({
        f"paired_gain_vs_{best_background}_ci95": summarize_bootstrap(
            samples, methods, best_background
        )[f"paired_gain_vs_{best_background}_ci95"],
        "best_background_config": best_background,
    })
    strata = stratified_metrics(eval_result["rank_matrix"], methods, queries, ctx)

    selected = frozen["selected_config"]
    if selected not in methods:
        raise EvalError(f"frozen selected_config {selected!r} is not among the configs")
    sum_index = methods.index("sum")
    selected_index = methods.index(selected)
    deltas = eval_result["rank_matrix"][:, selected_index] - eval_result["rank_matrix"][:, sum_index]
    moved_up = int((deltas < 0).sum())
    moved_down = int((deltas > 0).sum())

    rows = []
    for i, q in enumerate(queries):
        rows.append({
            "query_id": q["query_id"], "text_index": q["text_index"],
            "cad_index": q["cad_index"], "cad_uid": axis["cad_uids"][q["cad_index"]],
            "ranks": {m: int(eval_result["rank_matrix"][i, j])
                      for j, m in enumerate(methods)},
            "delta_vs_sum": {
                m: int(eval_result["rank_matrix"][i, sum_index] - eval_result["rank_matrix"][i, j])
                for j, m in enumerate(methods) if m != "sum"
            },
        })
    write_jsonl(out / "query_ranks.jsonl", rows)
    write_jsonl(out / "top20.jsonl", [
        {"text_index": index, "cad_indices": by_method}
        for index, by_method in sorted(eval_result["top_by_text"].items())
    ])

    summary = {
        "schema": SCHEMA_EVAL_SUMMARY,
        "plan": frozen.get("plan"),
        "axis": "test",
        "text_stop": text_stop,
        "n_test_groups": text_stop - N_BACKGROUND,
        "n_gallery": n_cads,
        "n_queries": len(queries),
        "frozen_sha256": frozen_sha,
        "selected_config": selected,
        "analytic_random": {f"R@{k}": k / n_cads for k in KS},
        "metrics": metrics,
        "rerank_vs_sum": {
            "selected_config": selected,
            "n_queries": len(queries),
            "n_moved_up": moved_up,
            "n_moved_down": moved_down,
            "n_unchanged": len(queries) - moved_up - moved_down,
        },
        "inputs_sha256": axis["inputs_sha256"],
        "counts_sha256": counts_sha,
        "chunk_sha256": matrix["chunk_sha256"],
        "elapsed_seconds": round(time.time() - started, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_sha = write_canonical_json(out / "evaluation_summary.json", summary)
    boot_sha = write_canonical_json(out / "bootstrap.json", {
        "schema": SCHEMA_BOOTSTRAP, "stage": "test", **boot})
    strata_sha = write_canonical_json(out / "stratification.json", strata)

    lines = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({
        "status": "evaluate_complete", "out": str(out),
        "n_queries": len(queries), "text_stop": text_stop,
        "n_configs": len(methods),
        "selected_config": selected,
        "selected_r20": metrics[selected]["R@20"],
        "sum_r20": metrics["sum"]["R@20"],
        "rerank_vs_sum": summary["rerank_vs_sum"],
        "evaluation_summary_sha256": summary_sha,
        "bootstrap_sha256": boot_sha,
        "stratification_sha256": strata_sha,
        "elapsed_seconds": round(time.time() - started, 1),
    }, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p_fit = sub.add_parser("fit", help="fit + select on the validation axis; writes frozen.json")
    p_fit.add_argument("--inputs", required=True)
    p_fit.add_argument("--eval", required=True)
    p_fit.add_argument("--score-dir", required=True)
    p_fit.add_argument("--out", required=True)
    p_fit.set_defaults(func=lambda a: run_fit(a))

    p_eval = sub.add_parser("evaluate", help="frozen evaluation on the test axis")
    p_eval.add_argument("--inputs", required=True)
    p_eval.add_argument("--eval", required=True)
    p_eval.add_argument("--score-dir", required=True)
    p_eval.add_argument("--out", required=True)
    p_eval.add_argument("--frozen", required=True)
    p_eval.add_argument("--expected-frozen-sha", required=True)
    p_eval.add_argument("--text-stop", type=int, required=True)
    p_eval.set_defaults(func=lambda a: run_evaluate(a))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except EvalError as exc:
        print(f"ACCURACY-NIGHT-EVAL FATAL: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001 - every refusal exits non-zero with the reason
        print(f"ACCURACY-NIGHT-EVAL UNEXPECTED {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
