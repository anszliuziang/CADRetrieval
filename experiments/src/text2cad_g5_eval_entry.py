"""Thin external-scores entry onto the frozen G5 CPU evaluation protocol.

Plan: IDEA01_G5_EXTERNAL_BASELINES_2026-11-15.md section 2. The historical
formal CLI (experiments/scripts/run_text2cad_g5_cpu.py + experiments/src/
text2cad_g5_cpu.py) is kept untouched as the original run evidence; this
module reuses its audited building blocks (link_inputs, score_ranks,
rank_metrics, bootstrap_statistics, duplicate_ceilings, manifest_entries)
and adds:

  * identity gates  — declared row/col identities of an external score set
    must match the frozen unique-text / gallery axes exactly (dense or a
    declared common sub-library);
  * coverage gates  — every declared (text, cad) pair exactly once per text,
    no duplicates, no missing rows; per-row cad identity lists must line up
    with the score values they carry (the JSONL chunk format detects
    arrangement errors; the matrix format detects shape mismatches —
    content-level mislabeling is out of scope for the evaluator and is
    covered by input fingerprints plus the replay reconciliation);
  * value gates     — all scores finite; direction declared ('higher' or
    'lower'); ranking always float64, exact ties broken by cad_index
    ascending (g5.score_ranks semantics);
  * metrics         — per-query hit counts, R@1/5/10/20 / MedR / MeanR,
    unique/repeated and gallery-length strata, duplicate UID ceilings;
  * paired arm      — a second score set (e.g. SUM sliced from the G4
    matrix) re-ranked on the SAME declared axis with SHARED cluster
    bootstrap draws, reporting paired deltas and 95% CIs;
  * evidence        — per-query ranks (query_id, text_index, positive
    cad_id, rank) and per-text Top-20 written to the output directory
    together with input fingerprints and a results.json.

Zero model forwards; torch-free; runs anywhere the inputs live.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import text2cad_g5_cpu as g5  # noqa: E402  (audited G5 evaluation module)

SCHEMA = "text2cad-g5-external-eval-v1"
SCORE_CHUNK_SCHEMA = "text2cad-method-scores-v1"
DIRECTIONS = ("higher", "lower")


class EntryError(Exception):
    """Refusal of an external score set (identity/coverage/value gates)."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise EntryError(message)


def sha256_file(path: Path) -> str:
    return g5.sha256(Path(path))


# ---------------------------------------------------------------- score sets


class ScoreSet:
    """An external score set over (declared unique texts) x (declared cads).

    rows must be iterable in declared-text order; each row yields
    (text_index, cad_indices_list, scores_list). The matrix convenience
    constructor expands a dense array into that shape.
    """

    def __init__(self, *, direction: str, dtype: str, meta: dict, rows) -> None:
        _require(direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}, got {direction!r}")
        self.direction = direction
        self.dtype = str(dtype)
        self.meta = dict(meta or {})
        self._rows = list(rows)
        _require(len(self._rows) > 0, "empty score set")

    def __iter__(self):
        return iter(self._rows)

    @classmethod
    def from_matrix(cls, scores, text_indices, cad_indices, *, direction="higher",
                    dtype="float64", meta=None) -> "ScoreSet":
        a = np.asarray(scores)
        ti = list(int(t) for t in text_indices)
        ci = [int(c) for c in cad_indices]
        _require(a.ndim == 2 and a.shape == (len(ti), len(ci)),
                 f"score matrix shape {a.shape} does not match declared axes "
                 f"{(len(ti), len(ci))} (row/col misalignment)")
        meta = dict(meta or {})
        meta["format"] = "dense_matrix"
        return cls(direction=direction, dtype=dtype, meta=meta,
                   rows=((t, ci, a[i]) for i, t in enumerate(ti)))

    @classmethod
    def from_jsonl(cls, path) -> "ScoreSet":
        """Streaming chunk file: one header record + one row record per text.

        Header: {record_type: method_score_header, schema, method, direction,
        dtype, n_texts, n_cads_per_text, meta{...}}.  Row: {record_type:
        method_scores, text_index, cad_indices: [...], scores: [...]}.
        Per-row cad identity lists are validated against the declared axis,
        so a mis-arranged row is refused instead of silently reindexed.
        """
        path = Path(path)
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        _require(lines, "empty score chunk file")
        header = json.loads(lines[0])
        _require(header.get("record_type") == "method_score_header",
                 "first record must be method_score_header")
        _require(header.get("schema") == SCORE_CHUNK_SCHEMA,
                 f"score chunk schema must be {SCORE_CHUNK_SCHEMA}")
        _require(header.get("direction") in DIRECTIONS,
                 f"header direction invalid: {header.get('direction')!r}")
        rows = []
        for line in lines[1:]:
            row = json.loads(line)
            _require(row.get("record_type") == "method_scores",
                     "non-row record inside score chunk file")
            ci = row.get("cad_indices")
            sc = row.get("scores")
            _require(isinstance(ci, list) and isinstance(sc, list) and len(ci) == len(sc),
                     f"text {row.get('text_index')}: cad_indices/scores length mismatch "
                     f"(row/col misalignment)")
            rows.append((int(row["text_index"]), ci, sc))
        _require(len(rows) == int(header["n_texts"]),
                 f"declared n_texts {header['n_texts']} != row records {len(rows)}")
        meta = dict(header.get("meta") or {})
        meta["format"] = "jsonl_chunks"
        meta["file_sha256"] = hashlib.sha256(raw).hexdigest()
        meta["file"] = str(path)
        return cls(direction=header["direction"], dtype=str(header["dtype"]),
                   meta=meta, rows=rows)


# ---------------------------------------------------------------- inputs


def load_frozen_protocol(repo, data_root, runtime_root, *, require_certificate=True):
    """Verify the frozen G5 input pins and return the linked protocol."""
    repo, data_root, runtime_root = Path(repo), Path(data_root), Path(runtime_root)
    config = g5.read_json(repo / "experiments/configs/text2cad_g5_cpu_frozen_20261003.json")
    fingerprints = {}
    if require_certificate:
        g5.require_certificate(repo / config["certificate_path"], g5.CERTIFICATE_SHA)
    for name, pin in config["inputs"].items():
        path = (data_root if pin["root"] == "data" else runtime_root) / pin["path"]
        digest = sha256_file(path)
        _require(digest == pin["sha256"], f"frozen input SHA differs: {name}")
        fingerprints[name] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
    linked = g5.link_inputs(
        **{name: g5.jsonl((data_root if p["root"] == "data" else runtime_root) / p["path"])
           for name, p in config["inputs"].items()
           if name in ("queries", "gallery", "qrels", "texts", "store", "lengths")},
        mapping=g5.read_json((data_root if config["inputs"]["mapping"]["root"] == "data"
                              else runtime_root) / config["inputs"]["mapping"]["path"]))
    g5.require(len(linked["query_ids"]) == 7934 and len(linked["group_sizes"]) == 3644
               and len(linked["cad_ids"]) == 8046, "full official counts differ")
    return dict(config=config, fingerprints=fingerprints, linked=linked)


# ---------------------------------------------------------------- evaluation


def _axis_and_scores(score_set: ScoreSet, linked: dict, cad_subset):
    """Validate declared identities against the frozen axes; collect scores.

    Returns (text_indices, cad_positions (dense indices), scores matrix
    (n_texts, n_declared_cads) float64)."""
    n_texts_total = len(linked["group_sizes"])
    n_cads_total = len(linked["cad_ids"])

    ti_declared: list[int] = []
    cad_declared: list[int] | None = None
    out_texts: list[int] = []
    out_rows: list[np.ndarray] = []
    for text_index, cad_indices, scores in score_set:
        _require(0 <= int(text_index) < n_texts_total,
                 f"text_index {text_index} outside 0..{n_texts_total - 1}")
        _require(int(text_index) not in ti_declared,
                 f"duplicate text row for text_index {text_index}")
        ci = [int(c) for c in cad_indices]
        if cad_declared is None:
            cad_declared = ci
            _require(len(set(ci)) == len(ci), "duplicate cad in declared axis")
            _require(all(0 <= c < n_cads_total for c in ci),
                     "declared cad index outside gallery")
            if cad_subset is None:
                _require(sorted(ci) == list(range(n_cads_total)),
                         "declared cad axis is not the full dense gallery "
                         "(pass cad_subset explicitly for sub-library mode)")
            else:
                _require(sorted(ci) == sorted(int(c) for c in cad_subset),
                         "declared cad axis does not equal the declared cad_subset")
        else:
            _require(ci == cad_declared,
                     f"text {text_index}: per-row cad identity list differs from the "
                     f"declared axis (row/col misalignment)")
        _require(len(scores) == len(ci),
                 f"text {text_index}: scores length != declared cad axis length")
        ti_declared.append(int(text_index))
        a = np.asarray(scores, dtype=np.float64)
        _require(a.shape == (len(ci),), f"text {text_index}: score conversion mismatch")
        _require(bool(np.isfinite(a).all()),
                 f"text {text_index}: non-finite score(s) refused")
        out_texts.append(int(text_index))
        out_rows.append(-a if score_set.direction == "lower" else a)
    _require(cad_declared is not None, "no score rows")
    want_texts = list(range(n_texts_total))
    _require(sorted(ti_declared) == want_texts,
             f"declared text axis {len(ti_declared)} row(s) is not the full unique-text "
             f"axis 0..{n_texts_total - 1} (missing texts must be declared as a sub-library "
             f"evaluation instead of silently omitted)")
    return np.array(out_texts, dtype=np.int64), np.array(cad_declared, dtype=np.int64), \
        np.vstack(out_rows)


def _rank_axis(scores: np.ndarray) -> np.ndarray:
    """Rank every row with g5.score_ranks (exact ties -> cad_index ascending)."""
    out = np.empty(scores.shape, dtype=np.int64)
    for i in range(scores.shape[0]):
        out[i] = g5.score_ranks(scores[i])
    return out


def _strata(pos_ranks, qs_per_query, positives, lengths, methods):
    qs = qs_per_query
    masks = {"unique_q1": qs == 1, "repeated_q2p": qs > 1}
    cuts = np.quantile(lengths, [.25, .5, .75])
    bins = np.searchsorted(cuts, lengths[positives], side="left")
    masks.update({f"length_quartile_{i + 1}": bins == i for i in range(4)})
    strata = {}
    for name, mask in masks.items():
        strata[name] = {m: g5.rank_metrics(pos_ranks[mask, j]) for j, m in enumerate(methods)}
        strata[name]["n_queries"] = int(mask.sum())
    return dict(gallery_length_quartile_cutpoints=cuts.tolist(),
                length_boundary_rule="lower bin includes exact boundary", strata=strata)


def _shared_draw_counts(groups, seed: int, replicates: int) -> np.ndarray:
    gg = int(groups.max()) + 1
    rng = np.random.default_rng(seed)
    counts = np.empty((replicates, gg), dtype=np.int16)
    for i in range(replicates):
        counts[i] = np.bincount(rng.integers(0, gg, size=gg), minlength=gg)
    return counts


def evaluate(score_set: ScoreSet, *, protocol: dict, output, method_meta: dict,
             cad_subset=None, paired_sum_set: ScoreSet | None = None,
             bootstrap: dict | None = None) -> dict:
    """Evaluate one external score set under the frozen protocol.

    cad_subset: None (full 8,046 gallery) or an explicit dense-index subset
    (common sub-library).  Queries whose positive is outside the subset are
    frozen out BEFORE any metric is computed, and the kept query set is
    reported.  paired_sum_set must declare the SAME axis; both arms are
    re-ranked on that axis and compared with shared bootstrap draws.
    """
    started = time.time()
    linked = protocol["linked"]
    output = Path(output)
    _require(not output.exists(), "existing output directory")
    sizes_all, groups_all, positives_all = (linked[k] for k in ("group_sizes", "query_groups", "positive_cads"))
    lengths = linked["lengths"].astype(np.float64)

    texts, cads, scores = _axis_and_scores(score_set, linked, cad_subset)
    method_ranks = _rank_axis(scores)  # (n_texts, n_axis_cads)

    # subset handling: map dense cad_index -> axis position
    pos_of = {int(c): i for i, c in enumerate(cads)}
    if cad_subset is None:
        kept = np.ones(len(groups_all), dtype=bool)
    else:
        subset_set = set(int(c) for c in cad_subset)
        kept = np.array([int(p) in subset_set for p in positives_all], dtype=bool)

    groups = groups_all[kept]
    positives = positives_all[kept]
    # link_inputs' group_sizes are PER UNIQUE TEXT — re-dense them over the
    # kept text groups (sub-library mode may empty some texts entirely)
    kept_text_ids = np.unique(groups)
    sizes_by_group = sizes_all[kept_text_ids]
    remap = np.full(len(sizes_all), -1, dtype=np.int64)
    remap[kept_text_ids] = np.arange(len(kept_text_ids))
    group_col = remap[groups]
    qs_per_query = sizes_by_group[group_col]
    query_ids = [qid for qid, k in zip(linked["query_ids"], kept) if k]

    pos_method = np.empty((len(query_ids), 1), dtype=np.int64)
    for t_idx, t in enumerate(texts):
        qi = np.flatnonzero(groups == t)
        axis = np.array([pos_of[int(p)] for p in positives[qi]], dtype=np.int64)
        pos_method[qi, 0] = method_ranks[t_idx][axis]

    methods = [str(method_meta.get("name", "method"))]
    metrics = {methods[0]: g5.rank_metrics(pos_method[:, 0])}
    hit_counts = {methods[0]: {f"hit@{k}": int((pos_method[:, 0] <= k).sum()) for k in g5.KS}}
    denominators = {methods[0]: int(len(query_ids))}

    paired_block = None
    pos_sum = None
    if paired_sum_set is not None:
        s_texts, s_cads, s_scores = _axis_and_scores(paired_sum_set, linked, cad_subset)
        _require(np.array_equal(s_texts, texts) and np.array_equal(s_cads, cads),
                 "paired SUM arm declares a different axis")
        sum_ranks = _rank_axis(s_scores)
        pos_sum = np.empty((len(query_ids), 1), dtype=np.int64)
        for t_idx, t in enumerate(texts):
            qi = np.flatnonzero(groups == t)
            axis = np.array([pos_of[int(p)] for p in positives[qi]], dtype=np.int64)
            pos_sum[qi, 0] = sum_ranks[t_idx][axis]
        sum_name = str(paired_sum_set.meta.get("name", "sum_paired"))
        methods.append(sum_name)
        metrics[sum_name] = g5.rank_metrics(pos_sum[:, 0])
        hit_counts[sum_name] = {f"hit@{k}": int((pos_sum[:, 0] <= k).sum()) for k in g5.KS}
        denominators[sum_name] = int(len(query_ids))
        pos_all = np.concatenate([pos_method, pos_sum], axis=1)

    strata = _strata(pos_all if pos_sum is not None else pos_method, qs_per_query,
                     positives, lengths, methods)
    ceilings = g5.duplicate_ceilings(sizes_by_group)

    bootstrap_block = None
    if bootstrap is not None:
        _require(pos_sum is not None, "paired bootstrap requires paired_sum_set")
        seed = int(bootstrap.get("seed", 20261112))
        replicates = int(bootstrap.get("replicates", 5000))
        # group_col is already dense over kept groups (sub-library mode may
        # empty some unique texts; g5.bootstrap_statistics requires dense)
        counts = _shared_draw_counts(group_col, seed, replicates)
        samples = g5.bootstrap_statistics(pos_all, group_col, counts)
        paired = {}
        for k in g5.KS:
            deltas = samples[f"R@{k}"][:, 0] - samples[f"R@{k}"][:, 1]
            paired[f"R@{k}"] = dict(point=float(metrics[methods[0]][f"R@{k}"] - metrics[methods[1]][f"R@{k}"]),
                                    ci95=np.quantile(deltas, [.025, .975]).tolist())
        dm = samples["MedR"][:, 0] - samples["MedR"][:, 1]
        paired["MedR"] = dict(point=float(metrics[methods[0]]["MedR"] - metrics[methods[1]]["MedR"]),
                              ci95=np.quantile(dm, [.025, .975]).tolist())
        bootstrap_block = dict(seed=seed, replicates=replicates,
                               resampling_unit="unique_text_group", query_weighted=True,
                               shared_draws=True, paired_vs=methods[1], paired_ci95=paired,
                               draw_counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())

    output.mkdir(parents=True)
    axis_cad_ids = [linked["cad_ids"][int(c)] for c in cads]
    with (output / "query_ranks.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for i, qid in enumerate(query_ids):
            rec = dict(query_id=qid, text_index=int(groups[i]), cad_index=int(positives[i]),
                       cad_id=linked["cad_ids"][int(positives[i])],
                       group_size=int(qs_per_query[i]),
                       n_valid_total=int(round(lengths[positives[i]])),
                       ranks={methods[0]: int(pos_method[i, 0])})
            if pos_sum is not None:
                rec["ranks"][methods[1]] = int(pos_sum[i, 0])
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "text_top20.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for t_idx, t in enumerate(texts):
            rr = method_ranks[t_idx]
            order = np.argsort(rr)[:20]
            f.write(json.dumps(dict(
                text_index=int(t), axis="subset" if cad_subset is not None else "full",
                top20=[dict(rank=int(rr[int(p)]), cad_index=int(cads[int(p)]),
                            cad_id=axis_cad_ids[int(p)], score=float(scores[t_idx][int(p)]))
                       for p in order]), ensure_ascii=False, allow_nan=False) + "\n")

    fingerprints = dict(protocol["fingerprints"])
    fingerprints["score_source"] = score_set.meta
    if paired_sum_set is not None:
        fingerprints["paired_sum_source"] = paired_sum_set.meta

    results = dict(schema=SCHEMA, method=method_meta, score_direction=score_set.direction,
                   score_dtype=score_set.dtype, axis=dict(
                       n_texts=len(texts), n_cads=int(len(cads)),
                       full_library=cad_subset is None,
                       cad_index_first=int(cads[0]), cad_index_last=int(cads[-1])),
                   kept_queries=int(len(query_ids)), dropped_queries=int((~kept).sum()),
                   coverage_note="every declared (text, cad) pair present exactly once; all finite",
                   tie_rule="exact score then cad_index ascending",
                   metric_units="fraction for recall, one-based rank",
                   metrics=metrics, hit_counts=hit_counts, denominators=denominators,
                   duplicate_uid_ceilings=ceilings, stratification=strata,
                   statistics=bootstrap_block, fingerprints=fingerprints,
                   zero_npu=True, score_forwards=0, elapsed_seconds=time.time() - started)
    body = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (output / "results.json").write_text(body, encoding="utf-8", newline="\n")
    lines = []
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(output).as_posix())
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return results
