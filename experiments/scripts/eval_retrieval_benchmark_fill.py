"""Independently rank registered CAD instances from frozen scores or embeddings."""

from __future__ import annotations

import argparse
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prepare_retrieval_benchmark_fill import (
    ASSETS,
    CONDITIONS,
    G5_REPORT_DIR,
    load_json,
    read_csv,
    read_jsonl,
    sha256,
    text_digest,
    verify_freeze,
    verify_sha256,
    write_json,
    write_jsonl,
)


K_LEVELS = (1, 5, 10, 20)
CORE_METHODS = ("clip", "openshape", "mmcad_v2", "sum", "s48")
METHOD_TRAINING = {
    "clip": "external frozen CLIP ViT-B/32",
    "openshape": "external frozen OpenShape point-cloud adaptation",
    "mmcad_v2": "external frozen MM-CAD joint checkpoint, strict v2 loader",
    "sum": "official Text2CAD v1.0 generator checkpoint",
    "s48": "official Text2CAD v1.0 generator checkpoint, 48 frozen source-validation texts",
    "shortest_first": "none",
}
HIST_RANK_SHA = "0af7debb4aeb62f66e53a502cfb9cd4dd3559866e910860b8b0797fbd1f37e21"
TEXT2CAD_CHECKPOINT_SHA = "235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598"
CLIP_HIST_WEIGHT_RECEIPT_SHA = "b98580bb804a47417bb34bec202d645fc53e537b2ac71dc1b1befd2920e2c0d9"
HIST_WEIGHT_PINS = {
    "openshape": ("pointbert_sha256", "34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9"),
    "mmcad_v2": ("ckpt_sha256", "009b23ce18f2baeca9b14bc8043e19147c244b495b3265500bba419eee5b2b84"),
}
HIST_EXPECTED = {
    "clip": ([11, 29, 57, 94], 2049),
    "openshape": ([17, 69, 119, 189], 1367),
    "mmcad_v2": ([9, 39, 64, 117], 1724),
    "sum": ([16, 73, 128, 202], 3300.5),
    "s48": ([23, 126, 233, 437], 593),
}
HIST_COLUMNS = {
    "clip": "common_clip",
    "openshape": "common_openshape",
    "mmcad_v2": "common_mmcad",
    "sum": "common_sum",
    "s48": "common_logmix",
}
HIST_MATRICES = {
    "clip": (
        "runtime", "idea01_clip_g5_20261115", "clip_scores", "clip_scores_subset_axis_f64.npy",
        "c0711c197fcb999776d304274da64e9ea77a2d53054ceda933ab9c1cf09275f9",
        "d60d8f9cc9f647e16d9478b0287d243429e8c41d5d4aafb66e6115a08ac44045",
    ),
    "openshape": (
        "runs", "idea01_openshape_g5_20261116", "openshape_scores",
        "openshape_scores_subset_axis_f64.npy",
        "1c8ebc4616d652d3ad5c9e7755f3b33ebefdecf8bb6b86e8c39310561744640e",
        "26d1471316d7821b8eaa6f34f53cd7bb376c7d1e4ea2d17c4064d12c5537fee7",
    ),
    "mmcad_v2": (
        "runs", "idea01_mmcad_g5_v2_20261118", "mmcad_v2_scores",
        "mmcad_v2_scores_subset_axis_f64.npy",
        "c08d47a997b92188ab3437facb8b4f86e6bffeb956ee42b5f779b3592af2ca51",
        "7e35bcc36a36849c2a9cab278cd616b34214e398ff063ec3eb9b9906987921ef",
    ),
    "sum": (
        "runs", "idea01_g5_sum_replay_20261115", "sum_scores", "sum_scores_full_axis_f64.npy",
        "9479895007c3b45742ccb39884d321be85d4856cbc2ec123d6ddd07e8e03b004",
        "9a379ed75b28fb575c2cabd25c9f1e1e6368a9a0cb0ea0c1e564ec8b293710cf",
    ),
}


@dataclass
class FrozenCondition:
    directory: Path
    name: str
    manifest: dict
    gallery: list[dict]
    texts: list[dict]
    queries: list[dict]
    queries_by_text: dict
    text_counts: Counter
    positions: dict[int, int]

    @classmethod
    def read(cls, directory: Path, condition: str) -> FrozenCondition:
        manifest = verify_freeze(directory)
        gallery = read_jsonl(directory / "gallery.jsonl")
        texts = read_jsonl(directory / f"texts_{condition}.jsonl")
        queries = read_jsonl(directory / f"queries_{condition}.jsonl")
        cad_indices = [row["cad_index"] for row in gallery]
        if (len(gallery) != manifest["gallery_count"] or cad_indices != sorted(set(cad_indices))
                or len({row["uid"] for row in gallery}) != len(gallery)):
            raise ValueError("Invalid frozen gallery UID or index axis")
        if any(
            row["text_index"] != i or not row["text"] or row["text"] != row["text"].strip()
            or row["text_sha256"] != text_digest(row["text"])
            for i, row in enumerate(texts)
        ) or len({row["text_sha256"] for row in texts}) != len(texts):
            raise ValueError(f"Invalid frozen text axis for {condition}")
        positions = {index: position for position, index in enumerate(cad_indices)}
        by_text = defaultdict(list)
        for row in queries:
            if (row["cad_index"] not in positions or not 0 <= row["text_index"] < len(texts)
                    or row["uid"] != gallery[positions[row["cad_index"]]]["uid"]):
                raise ValueError(f"Invalid target link for {condition}: {row}")
            by_text[row["text_index"]].append(row)
        if (len(queries) != len({row["uid"] for row in queries})
                or set(by_text) != set(range(len(texts)))):
            raise ValueError("Frozen query axis has a duplicated UID or an unlinked text")
        counts = Counter({index: len(group) for index, group in by_text.items()})
        stats = manifest["conditions"][condition]
        if (len(queries), len(texts), sum(count == 1 for count in counts.values()), max(counts.values())) != (
            stats["queries"], stats["unique_texts"], stats["singleton_queries"], stats["largest_group"]
        ):
            raise ValueError(f"Registered counts differ from the freeze for {condition}")
        return cls(directory, condition, manifest, gallery, texts, queries, by_text, counts, positions)

    @property
    def gallery_sha(self) -> str:
        return self.manifest["files"]["gallery.jsonl"]

    @property
    def text_sha(self) -> str:
        return self.manifest["files"][f"texts_{self.name}.jsonl"]

    @property
    def query_sha(self) -> str:
        return self.manifest["files"][f"queries_{self.name}.jsonl"]


def strict_rank(scores: np.ndarray, candidate_valid: np.ndarray, position: int) -> int:
    """Compute the positive rank on an ascending historical CAD axis."""
    if not candidate_valid[position]:
        return len(scores) + 1
    score = scores[position]
    higher = np.count_nonzero(candidate_valid & (scores > score))
    earlier_equal = np.count_nonzero(candidate_valid[:position] & (scores[:position] == score))
    return int(1 + higher + earlier_equal)


def score_summary(ranks, gallery_size: int | None = None) -> dict:
    ranks = np.asarray(ranks, dtype=np.int64)
    hits = [int(np.count_nonzero(ranks <= (k if gallery_size is None else min(k, gallery_size))))
            for k in K_LEVELS]
    n = len(ranks)
    return {
        "queries": n,
        "hits": hits,
        "recall_percent": [100 * hit / n if n else None for hit in hits],
        "median_rank": float(np.median(ranks)) if n else None,
    }


def hit_contrast(before: list[dict], after: list[dict]) -> dict:
    if not before or [r["uid"] for r in before] != [r["uid"] for r in after]:
        raise ValueError("Paired comparison requires a shared ordered positive UID axis")
    x = np.array([top20_hit(row) for row in before])
    y = np.array([top20_hit(row) for row in after])
    gained = int(np.count_nonzero(y & ~x))
    lost = int(np.count_nonzero(x & ~y))
    return {
        "queries": len(before),
        "both_hits": int(np.count_nonzero(x & y)),
        "gained": gained,
        "lost": lost,
        "difference_percentage_points": 100 * (gained - lost) / len(before),
    }


def top20_hit(row: dict) -> bool:
    return row["rank"] <= 20 and not row.get("failed", False)


def grouped_r20_contrast(before: list[dict], after: list[dict],
                         repetitions: int = 5000, seed: int = 20261120) -> dict:
    if [row["text_index"] for row in before] != [row["text_index"] for row in after]:
        raise ValueError("A same-condition contrast must share the exact native text groups")
    contrast = hit_contrast(before, after)
    # Frozen text_index order is SHA256 order, also used by the published L0 bootstrap.
    group_n = Counter(row["text_index"] for row in before)
    group_delta = Counter()
    for x, y in zip(before, after, strict=True):
        group_delta[x["text_index"]] += int(top20_hit(y)) - int(top20_hit(x))
    group_order = sorted(group_n)
    sizes = np.array([group_n[group] for group in group_order], dtype=np.int64)
    deltas = np.array([group_delta[group] for group in group_order], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sample = np.empty(repetitions, dtype=np.float64)
    for lo in range(0, repetitions, 32):
        chunk = sample[lo:lo + 32]
        draws = rng.integers(len(sizes), size=(len(chunk), len(sizes)))
        chunk[:] = 100 * deltas[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {
        **contrast,
        "bootstrap_percentile_95_pp": np.percentile(sample, [2.5, 97.5]).tolist(),
        "unit": "exact native text, with every target UID in each sampled group",
        "groups": len(sizes),
        "repetitions": repetitions,
        "seed": seed,
    }


def rank_text(frozen: FrozenCondition, text_index: int, scores: np.ndarray, query_status: dict,
              gallery_status: list[dict], candidate_valid: np.ndarray) -> list[dict]:
    n = len(frozen.gallery)
    if query_status["valid"]:
        if scores.dtype != np.dtype("<f8") or scores.shape != (n,):
            raise ValueError(f"Malformed score row for {frozen.name}/{text_index}")
        if not np.isfinite(scores[candidate_valid]).all():
            raise ValueError(f"Non-finite candidate scores for {frozen.name}/{text_index}")
    results = []
    for query in frozen.queries_by_text[text_index]:
        position = frozen.positions[query["cad_index"]]
        query_failure = None if query_status["valid"] else query_status["reason"]
        candidate_failure = None if candidate_valid[position] else gallery_status[position]["reason"]
        failed = query_failure is not None or candidate_failure is not None
        results.append({
            "domain": frozen.manifest["domain"],
            "condition": frozen.name,
            **query,
            "text_sha256": frozen.texts[text_index]["text_sha256"],
            "rank": n + 1 if failed else strict_rank(scores, candidate_valid, position),
            "positive_score": None if failed else float(scores[position]),
            "failed": failed,
            "query_failure": query_failure,
            "candidate_failure": candidate_failure,
            "truncated": query_status["truncated"],
        })
    return results


def valid_gallery(status: list[dict]) -> np.ndarray:
    return np.array([row["valid"] for row in status], dtype=bool)


def gallery_accepted(frozen: FrozenCondition) -> list[dict]:
    return [{"valid": True}] * len(frozen.gallery)


def read_rank_results(path: Path, frozen: FrozenCondition) -> list[dict]:
    results_file = path / "query_ranks.jsonl"
    receipt = load_json(path / "metrics.json")
    if (receipt["gallery_sha256"] != frozen.gallery_sha or receipt["text_axis_sha256"] != frozen.text_sha
            or receipt["query_axis_sha256"] != frozen.query_sha
            or receipt["condition"] != frozen.name
            or receipt["queries"] != len(frozen.queries)):
        raise ValueError(f"Rank receipt points to a different benchmark input: {path}")
    verify_sha256(results_file, receipt["query_ranks_sha256"])
    rows = read_jsonl(results_file)
    if [(r["uid"], r["cad_index"], r["text_index"]) for r in rows] != [
        (query["uid"], query["cad_index"], query["text_index"]) for query in frozen.queries
    ] or any(
        r["text_sha256"] != frozen.texts[r["text_index"]]["text_sha256"]
        or type(r["rank"]) is not int or not 1 <= r["rank"] <= len(frozen.gallery) + 1
        or type(r["failed"]) is not bool
        or r["failed"] != (r["rank"] == len(frozen.gallery) + 1)
        or r["failed"] != bool(r["query_failure"] or r["candidate_failure"])
        or (r["truncated"] is not None and type(r["truncated"]) is not bool)
        or (r["positive_score"] is not None if r["failed"] else not np.isfinite(r["positive_score"]))
        for r in rows
    ):
        raise ValueError(f"Rank rows differ from the registered queries: {path}")
    return rows


def publish_result(output: Path, frozen: FrozenCondition, method: str, rows: list[dict],
                   gallery_status: list[dict], source: dict) -> dict:
    by_uid = {row["uid"]: row for row in rows}
    if len(rows) != len(by_uid) or set(by_uid) != {query["uid"] for query in frozen.queries}:
        raise ValueError("The result does not cover the registered query denominator")
    rows = [by_uid[query["uid"]] for query in frozen.queries]
    singleton = [row for row in rows if frozen.text_counts[row["text_index"]] == 1]
    n_truncated = sum(row["truncated"] is True for row in rows)
    n_gallery = len(frozen.gallery)
    gallery_failures = Counter(row["reason"] for row in gallery_status if not row["valid"])
    query_failures = Counter(row["query_failure"] for row in rows if row["query_failure"] is not None)
    positive_failures = Counter(row["candidate_failure"] for row in rows if row["candidate_failure"] is not None)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluating-", dir=output) as temp:
        rank_file = Path(temp) / "query_ranks.jsonl"
        metrics_file = Path(temp) / "metrics.json"
        write_jsonl(rank_file, rows)
        receipt = {
            "schema_version": 1,
            "table": 2 if frozen.name == "ct_deepcad" else 1,
            "domain": frozen.manifest["domain"],
            "condition": frozen.name,
            "method": method,
            "training_source": METHOD_TRAINING.get(method, source.get("training_source")),
            "gallery": n_gallery,
            "gallery_sha256": frozen.gallery_sha,
            "text_axis_sha256": frozen.text_sha,
            "query_axis_sha256": frozen.query_sha,
            "frozen_manifest_sha256": sha256(frozen.directory / "freeze.json"),
            "query_ranks_sha256": sha256(rank_file),
            "run_directory": str(output.resolve()),
            "k": list(K_LEVELS),
            **score_summary([row["rank"] for row in rows], n_gallery),
            "singleton": score_summary([row["rank"] for row in singleton], n_gallery),
            "distinct_original_texts": len(frozen.texts),
            "repeated_original_text_queries": len(rows) - len(singleton),
            "truncated_queries": n_truncated if all(row["truncated"] is not None for row in rows) else None,
            "known_truncation_queries": sum(row["truncated"] is not None for row in rows),
            "failed_queries": sum(row["failed"] for row in rows),
            "gallery_failures": {"count": sum(gallery_failures.values()), "reasons": gallery_failures},
            "query_text_failures": {"count": sum(query_failures.values()), "reasons": query_failures},
            "positive_cad_failures": {"count": sum(positive_failures.values()), "reasons": positive_failures},
            "source": source,
        }
        write_json(metrics_file, receipt)
        rank_file.replace(output / rank_file.name)
        metrics_file.replace(output / metrics_file.name)
    return receipt


def open_historical_matrix(folder: Path, prefix: str, matrix_name: str,
                           sidecar_sha: str, matrix_sha: str):
    path = folder / matrix_name
    sidecar_path = folder / (prefix + "_sidecar.json")
    verify_sha256(sidecar_path, sidecar_sha)
    sidecar = load_json(sidecar_path)
    shape = (len(sidecar["text_indices"]), len(sidecar["cad_indices"]))
    if (sidecar["dtype"] != "float64" or sidecar["direction"] != "higher"
            or sidecar["matrix_sha256"] != matrix_sha or sidecar["text_indices"] != list(range(3644))
            or sidecar["matrix_bytes"] != path.stat().st_size or np.prod(shape) * 8 != path.stat().st_size):
        raise ValueError(f"Historical matrix and sidecar disagree: {path}")
    verify_sha256(path, matrix_sha)
    # These files have .npy names but store raw binary per the historical sidecars.
    return np.memmap(path, dtype="<f8", mode="r", shape=shape), sidecar


def replay_l0(frozen: FrozenCondition, main_repo: Path, asset_runtime: Path, output: Path) -> dict:
    rank_csv = main_repo / G5_REPORT_DIR / "query_ranks.csv"
    verify_sha256(rank_csv, HIST_RANK_SHA)
    ranks = {row["uid"]: row for row in read_csv(rank_csv)}
    old_texts_path = asset_runtime / ASSETS["g5_text_axis"][1]
    verify_sha256(old_texts_path, ASSETS["g5_text_axis"][2])
    old_texts = read_jsonl(old_texts_path)
    previous_by_digest = {row["text_sha256"]: row["text_index"] for row in old_texts}
    old_row = [previous_by_digest[row["text_sha256"]] for row in frozen.texts]
    if any(
        ranks[query["uid"]]["cad_index"] != str(query["cad_index"])
        or ranks[query["uid"]]["text_index"] != str(old_row[query["text_index"]])
        for query in frozen.queries
    ) or any(
        original["text_index"] != i or original["text_sha256"] != text_digest(original["text"])
        for i, original in enumerate(old_texts)
    ):
        raise ValueError("Frozen L0 text or UID membership differs from historical G5")
    roots = {"runtime": asset_runtime, "runs": main_repo / "experiments/runs"}
    background = np.load(frozen.directory / "background.npy", allow_pickle=False)
    if background.shape != (len(frozen.gallery),) or background.dtype != np.dtype("<f8"):
        raise ValueError("Frozen S48 background does not match the gallery")
    all_good = gallery_accepted(frozen)
    candidate_valid = valid_gallery(all_good)
    expected_cad = [row["cad_index"] for row in frozen.gallery]
    receipts = {}
    for method, (anchor, run, prefix, matrix_name, sidecar_sha, matrix_sha) in HIST_MATRICES.items():
        folder = roots[anchor] / run
        matrix, sidecar = open_historical_matrix(folder, prefix, matrix_name, sidecar_sha, matrix_sha)
        if method == "sum":
            if sidecar["cad_indices"] != list(range(8046)):
                raise ValueError("Unexpected complete Text2CAD historical gallery axis")
            columns = np.asarray(expected_cad, dtype=int)
        else:
            if sidecar["cad_indices"] != expected_cad:
                raise ValueError(f"Different historical gallery axis for {method}")
            columns = slice(None)
        methods = ("sum", "s48") if method == "sum" else (method,)
        rows = {m: [] for m in methods}
        status = {"valid": True, "truncated": False if method == "sum" else None}
        for text_index, previous in enumerate(old_row):
            native = matrix[previous, columns]
            for m in methods:
                scores = native - background if m == "s48" else native
                rows[m].extend(rank_text(
                    frozen, text_index, scores, status, all_good, candidate_valid,
                ))
        del matrix
        source = {
            "mode": "historical raw matrix, independently counted rank for every registered UID",
            "path": str((folder / matrix_name).resolve()),
            "matrix_sha256": matrix_sha,
            "sidecar_sha256": sidecar_sha,
            "pinned_rank_csv_sha256": HIST_RANK_SHA,
            "historical_text_axis_sha256": ASSETS["g5_text_axis"][2],
        }
        if method == "sum":
            source["checkpoint_sha256"] = TEXT2CAD_CHECKPOINT_SHA
        else:
            meta = sidecar["meta"]
            source["historical_model"] = {k: meta[k] for k in (
                "method", "weights", "pointbert_sha256", "ckpt_sha256", "text_tower",
                "text_prompt", "spec_sha256",
            ) if k in meta}
            if method == "clip":
                weight_path = folder / "weights_version_receipt.json"
                verify_sha256(weight_path, CLIP_HIST_WEIGHT_RECEIPT_SHA)
                weights = load_json(weight_path)
                source["weights_receipt_sha256"] = CLIP_HIST_WEIGHT_RECEIPT_SHA
                source["checkpoint_sha256"] = weights["model_safetensors_sha256"]
            else:
                prefix_key, checkpoint_hash = HIST_WEIGHT_PINS[method]
                if not checkpoint_hash.startswith(meta[prefix_key]):
                    raise ValueError(f"Historical sidecar weight fingerprint differs for {method}")
                source["checkpoint_sha256"] = checkpoint_hash
            source["historical_truncation"] = "per-text truncation receipts absent from archived matrix"
        for m in methods:
            column = HIST_COLUMNS[m]
            if any(int(ranks[row["uid"]][column]) != row["rank"] for row in rows[m]):
                raise ValueError(f"Historical L0 target rank mismatch for {m}")
            metrics = score_summary([row["rank"] for row in rows[m]])
            if (metrics["hits"], metrics["median_rank"]) != HIST_EXPECTED[m]:
                raise ValueError(f"Historical L0 counts differ from plan for {m}: {metrics}")
            m_source = source if m != "s48" else {
                **source, "background_sha256": frozen.manifest["files"]["background.npy"],
            }
            receipts[m] = publish_result(output / frozen.name / m, frozen, m, rows[m], all_good, m_source)
        print(f"Verified historical L0 ranks for {', '.join(methods)}", flush=True)
    replay = {
        "gallery": len(frozen.gallery),
        "queries": len(frozen.queries),
        "matched_ranks": len(frozen.queries) * len(receipts),
        "methods": {name: {
            "hits": result["hits"],
            "median_rank": result["median_rank"],
            "metrics_sha256": sha256(output / frozen.name / name / "metrics.json"),
            "query_ranks_sha256": result["query_ranks_sha256"],
        } for name, result in receipts.items()},
        "rank_csv_sha256": HIST_RANK_SHA,
        "rank_formula": "1 + count(strictly higher valid scores) + count(exact ties at earlier cad_index)",
        "freeze_sha256": sha256(frozen.directory / "freeze.json"),
    }
    write_json(output / "l0_replay.json", replay)
    return replay


def read_axis_status(path: Path, frozen_rows: list[dict], is_text: bool) -> list[dict]:
    rows = read_jsonl(path)
    index_key = "text_index" if is_text else "cad_index"
    identity_key = "text_sha256" if is_text else "uid"
    if len(rows) != len(frozen_rows):
        raise ValueError(f"Status coverage differs for {path}")
    for status, expected in zip(rows, frozen_rows, strict=True):
        if (status[index_key] != expected[index_key]
                or (identity_key in status and status[identity_key] != expected[identity_key])
                or type(status["valid"]) is not bool
                or (is_text and type(status["truncated"]) is not bool)
                or (not status["valid"]
                    and (not isinstance(status.get("reason"), str) or not status["reason"]))):
            raise ValueError(f"Invalid {index_key} status in {path}: {status}")
    return rows


def array_artifact(path: Path, meta: dict, possible_keys: tuple[str, ...]) -> tuple[np.ndarray, str]:
    digest = next((meta[key] for key in possible_keys if key in meta), None)
    if not isinstance(digest, str):
        raise ValueError(f"Missing pinned array digest: {path}")
    verify_sha256(path, digest)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype("<f8") or array.ndim != 2 or not array.flags.c_contiguous:
        raise ValueError(f"Expected a C-order little-endian float64 matrix: {path}")
    return array, digest


def verify_axis_meta(path: Path, expected_axis_sha: str, axis_key: str) -> dict:
    meta = load_json(path)
    declared = meta.get("axis_sha256", meta.get(axis_key))
    if declared != expected_axis_sha:
        raise ValueError(f"Wrong axis SHA in {path}: {declared}")
    return meta


def verify_unit_vectors(matrix: np.ndarray, valid: np.ndarray, path: Path) -> None:
    if matrix.shape[0] != len(valid) or matrix.shape[1] == 0:
        raise ValueError(f"Embedding array does not match its registered axis: {path}")
    for lo in range(0, len(valid), 256):
        block = matrix[lo:lo+256]
        keep = valid[lo:lo+256]
        square_norms = np.einsum("ij,ij->i", block, block)
        if (not np.isfinite(square_norms).all()
                or not np.allclose(square_norms[keep], 1.0, atol=5e-6, rtol=0)
                or np.any(square_norms[~keep] != 0)):
            raise ValueError(f"Embedding rows must be finite unit vectors or registered zero failures: {path}")


def metadata_info(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items()
            if key.endswith(("_sha256", "_revision")) or key in {
                "weights", "checkpoint", "model", "training_source", "text_prompt", "preprocessing",
            }}


def external_rows(frozen: FrozenCondition, folder: Path, method: str, encoding: str, output: Path) -> dict:
    text_name = f"texts_{frozen.name}"
    gallery_status_path = folder / "gallery_status.jsonl"
    text_status_path = folder / f"{text_name}_status.jsonl"
    gallery_status = read_axis_status(gallery_status_path, frozen.gallery, is_text=False)
    text_status = read_axis_status(text_status_path, frozen.texts, is_text=True)
    candidate_valid = valid_gallery(gallery_status)
    text_valid = np.array([status["valid"] for status in text_status], dtype=bool)
    provenance = {
        "mode": encoding,
        "path": str(folder.resolve()),
        "gallery_status_sha256": sha256(gallery_status_path),
        "text_status_sha256": sha256(text_status_path),
    }
    if encoding == "embeddings":
        for axis, path, expected_sha, axis_key in (
            ("gallery", folder / "gallery_meta.json", frozen.gallery_sha, "gallery_sha256"),
            ("text", folder / f"{text_name}_meta.json", frozen.text_sha, "text_sha256"),
        ):
            meta = verify_axis_meta(path, expected_sha, axis_key)
            status_sha = meta.get("status_sha256")
            if status_sha is not None and status_sha != provenance[f"{axis}_status_sha256"]:
                raise ValueError(f"Wrong status SHA in {path}")
            provenance[f"{axis}_meta_sha256"] = sha256(path)
            provenance[f"{axis}_meta"] = metadata_info(meta)
            if axis == "gallery":
                gallery_meta = meta
            else:
                text_meta = meta
        gallery_path = folder / "gallery.npy"
        text_path = folder / f"{text_name}.npy"
        if (gallery_meta.get("method") != method or text_meta.get("method") != method
                or gallery_meta.get("weights") != text_meta.get("weights")):
            raise ValueError(f"Gallery and text embeddings use different model weights: {folder}")
        if method in CORE_METHODS[:3]:
            replay_path = folder / "l0_replay.json"
            replay_sha = sha256(replay_path)
            replay = load_json(replay_path)
            if (gallery_meta.get("l0_replay_sha256") != replay_sha
                    or text_meta.get("l0_replay_sha256") != replay_sha
                    or replay.get("passed") is not True or replay.get("gallery_sha256") != frozen.gallery_sha
                    or replay.get("method") != method):
                raise ValueError(f"Missing passed historical G5 replay for {method}")
            provenance["l0_replay_sha256"] = replay_sha
        digests = ("vectors_sha256", "array_sha256", "embedding_sha256", "embeddings_sha256")
        gallery, provenance["gallery_embeddings_sha256"] = array_artifact(gallery_path, gallery_meta, digests)
        texts, provenance["text_embeddings_sha256"] = array_artifact(text_path, text_meta, digests)
        if gallery.shape[1] != texts.shape[1]:
            raise ValueError("Gallery and text embedding dimensions differ")
        verify_unit_vectors(gallery, candidate_valid, gallery_path)
        verify_unit_vectors(texts, text_valid, text_path)
        provenance["training_source"] = text_meta.get("training_source")

        def scores(lo, hi):
            return texts[lo:hi] @ gallery.T

    else:
        meta_path = folder / f"scores_{frozen.name}_meta.json"
        meta = verify_axis_meta(meta_path, frozen.gallery_sha, "gallery_sha256")
        if meta.get("text_sha256", meta.get("text_axis_sha256")) != frozen.text_sha:
            raise ValueError(f"Wrong text axis SHA in {meta_path}")
        dense, digest = array_artifact(
            folder / f"scores_{frozen.name}.npy", meta, ("matrix_sha256", "array_sha256"),
        )
        if dense.shape != (len(frozen.texts), len(frozen.gallery)):
            raise ValueError(f"Dense score shape differs from frozen axes for {method}/{frozen.name}")
        provenance.update(matrix_sha256=digest, meta_sha256=sha256(meta_path), meta=metadata_info(meta),
                          training_source=meta.get("training_source"))

        def scores(lo, hi):
            return dense[lo:hi]

    rows = []
    for lo in range(0, len(frozen.texts), 64):
        hi = min(lo + 64, len(frozen.texts))
        for text_index, row in enumerate(scores(lo, hi), start=lo):
            rows.extend(rank_text(
                frozen, text_index, row, text_status[text_index], gallery_status, candidate_valid,
            ))
    return publish_result(output / frozen.name / method, frozen, method, rows, gallery_status, provenance)


def text2cad_rows(frozen: FrozenCondition, run: Path, output: Path) -> dict:
    auth_path = run / "run_authorization.json"
    completion_path = run / f"{frozen.name}_complete.json"
    auth = load_json(auth_path)
    complete = load_json(completion_path)
    freeze_sha = sha256(frozen.directory / "freeze.json")
    if (auth["mode"] != "score" or auth["status"] != "pilot_authorized"
            or auth["freeze_sha256"] != freeze_sha or frozen.name not in auth["conditions"]
            or not re.fullmatch(r"[0-9a-f]{40}", auth["approved_sha"])
            or not re.fullmatch(r"[0-9a-f]{64}", auth["probe_result_sha256"])):
        raise ValueError("Text2CAD authorization differs from the frozen scoring contract")
    # Independently reconstruct the controller's binding for each saved row.
    binding = text_digest("\n".join((
        auth["approved_sha"], freeze_sha, frozen.name, auth["probe_result_sha256"], "fp32/full272/b16",
    )))
    n = len(frozen.gallery)
    if (complete["condition"] != frozen.name or complete["complete_rows"] != len(frozen.texts)
            or complete["gallery"] != n or complete["binding"] != binding
            or complete["pairs_charged_total"] < n * len(frozen.texts)
            or complete["pairs_charged_total"] > auth["max_pairs"]
            or not 0 <= complete["cumulative_card_seconds"] <= auth["max_card_seconds"]):
        raise ValueError("Text2CAD condition is incomplete or has inconsistent accounting")
    folder = run / "scores" / frozen.name
    token_path = folder / "text_tokens.json"
    token_sha = sha256(token_path)
    if "token_metadata_sha256" in complete and complete["token_metadata_sha256"] != token_sha:
        raise ValueError("Text2CAD token metadata fingerprint differs from the condition receipt")
    tokens = load_json(token_path)
    if len(tokens) != len(frozen.texts):
        raise ValueError("Text2CAD token metadata has incomplete coverage")
    if any(
        meta["text_index"] != index or meta["text_sha256"] != text["text_sha256"]
        or type(meta["token_count"]) is not int or meta["token_count"] < 2
        or type(meta["truncated"]) is not bool or meta["truncated"] != (meta["token_count"] > 512)
        or meta["effective_token_count"] != min(512, meta["token_count"])
        or not re.fullmatch(r"[0-9a-f]{64}", meta["ids_sha256"])
        for index, (text, meta) in enumerate(zip(frozen.texts, tokens, strict=True))
    ):
        raise ValueError("Text2CAD token metadata differs from the frozen input/prefix truncation policy")
    expected_truncations = 1369 if frozen.name == "l3" else 0
    if sum(meta["truncated"] for meta in tokens) != expected_truncations:
        raise ValueError("Text2CAD token truncation count differs from the registered tokenizer audit")
    background = np.load(frozen.directory / "background.npy", allow_pickle=False)
    if background.shape != (n,) or background.dtype != np.dtype("<f8") or not np.isfinite(background).all():
        raise ValueError("Invalid frozen source-validation logmeanexp vector")
    all_good = gallery_accepted(frozen)
    candidate_valid = valid_gallery(all_good)
    rows = {"sum": [], "s48": []}
    row_digests = []
    for text, meta in zip(frozen.texts, tokens, strict=True):
        index = text["text_index"]
        path = folder / f"t{index:05d}.npy"
        digest = sha256(path)
        receipt = load_json(path.with_suffix(".json"))
        if receipt != {
            "text_index": index, "text_sha256": text["text_sha256"],
            "binding": binding, "scores_sha256": digest,
        }:
            raise ValueError(f"Score row receipt mismatch at {path}")
        row_digests.append(digest)
        native = np.load(path, allow_pickle=False)
        status = {"valid": True, "truncated": meta["truncated"]}
        rows["sum"].extend(rank_text(frozen, index, native, status, all_good, candidate_valid))
        rows["s48"].extend(rank_text(frozen, index, native - background, status, all_good, candidate_valid))
    source = {
        "mode": "Text2CAD authorized per-text raw sum shards",
        "path": str(folder.resolve()),
        "authorization_sha256": sha256(auth_path),
        "completion_sha256": sha256(completion_path),
        "implementation_sha": auth["approved_sha"],
        "probe_result_sha256": auth["probe_result_sha256"],
        "checkpoint_sha256": TEXT2CAD_CHECKPOINT_SHA,
        "token_metadata_sha256": token_sha,
        "ordered_score_row_digest": text_digest("\n".join(row_digests)),
    }
    return {method: publish_result(
        output / frozen.name / method, frozen, method, method_rows, all_good,
        source if method == "sum" else {**source, "background_sha256": frozen.manifest["files"]["background.npy"]},
    ) for method, method_rows in rows.items()}


def references(frozen: FrozenCondition, output: Path) -> dict:
    n, q = len(frozen.gallery), len(frozen.queries)
    fields = np.load(frozen.directory / "gallery_fields.npy", allow_pickle=False)
    if fields.shape != (n,) or fields.dtype != np.dtype("<i4") or not np.all(fields > 0):
        raise ValueError("Frozen valid-field counts disagree with the candidate axis")
    gallery_status = gallery_accepted(frozen)
    candidate_valid = valid_gallery(gallery_status)
    scores = -fields.astype("<f8")
    query_status = {"valid": True, "truncated": False}
    rank_rows = []
    for index in range(len(frozen.texts)):
        rank_rows.extend(rank_text(frozen, index, scores, query_status, gallery_status, candidate_valid))
    shortest = score_summary([row["rank"] for row in rank_rows], n)
    if frozen.name == "ct_deepcad":
        expected = ([1, 5, 10, 20], 4017)
        singleton = score_summary([row["rank"] for row in rank_rows
                                   if frozen.text_counts[row["text_index"]] == 1], n)
        if singleton["hits"] != [0, 1, 1, 1] or singleton["median_rank"] != 5669:
            raise ValueError("CadTranslator–DeepCAD singleton valid-field baseline disagrees with pinned evidence")
    else:
        expected = ([1, 4, 9, 19], 3969.5)
    if (shortest["hits"], shortest["median_rank"]) != expected:
        raise ValueError(f"Valid-field baseline disagrees with published receipt: {shortest}")
    shortest = publish_result(output / frozen.name / "shortest_first", frozen, "shortest_first",
                              rank_rows, gallery_status, {
                                  "mode": "valid-fields ascending",
                                  "native_field_counts_sha256": frozen.manifest["files"]["gallery_fields.npy"],
                              })
    expected_fraction = [min(k, n) / n for k in K_LEVELS]
    upper_ranks = [rank for count in frozen.text_counts.values() for rank in range(1, count + 1)]
    reference = {
        "condition": frozen.name,
        "gallery": n,
        "gallery_sha256": frozen.gallery_sha,
        "query_axis_sha256": frozen.query_sha,
        "queries": q,
        "random": {
            "mode": "analytical uniform ranking",
            "expected_hits": [q * fraction for fraction in expected_fraction],
            "recall_percent": [100 * fraction for fraction in expected_fraction],
            "population_median_rank": (n + 1) / 2,
            "singleton_expected_hits_at_20": sum(count == 1 for count in frozen.text_counts.values()) * expected_fraction[-1],
        },
        "shortest_first": shortest,
        "same_text_upper_bound": score_summary(upper_ranks, n),
    }
    write_json(output / frozen.name / "references.json", reference)
    return reference


def contrast_between_conditions(before: list[dict], after: list[dict],
                                uids: list[str]) -> dict:
    before_uid = {row["uid"]: row for row in before}
    after_uid = {row["uid"]: row for row in after}
    before_paired = [before_uid[uid] for uid in uids]
    after_paired = [after_uid[uid] for uid in uids]
    return {
        **hit_contrast(before_paired, after_paired),
        "before": score_summary([row["rank"] for row in before_paired]),
        "after": score_summary([row["rank"] for row in after_paired]),
    }


def summarize(directory: Path, output: Path) -> dict:
    conditions = {arm: FrozenCondition.read(directory, arm) for arm in CONDITIONS}
    available = defaultdict(dict)
    paired_s48_vs_sum = {}
    all_methods_full_text = {}
    manifest = conditions["l0"].manifest
    ct_uid = {row["uid"] for row in conditions["ct_deepcad"].queries}
    native_uid = {row["uid"] for row in conditions["l0"].queries}
    arms = ("l0", "l1", "l2", "l3")
    if any(native_uid != {row["uid"] for row in conditions[arm].queries} for arm in arms):
        raise ValueError("The four native Text2CAD conditions have different registered UID sets")
    paired_uid = sorted(ct_uid & native_uid)
    if len(paired_uid) != manifest["paired_uid_count"]:
        raise ValueError("Text2CAD/CT–DeepCAD paired query intersection differs from the freeze")
    for arm, frozen in conditions.items():
        for method in CORE_METHODS:
            result = output / arm / method
            if (result / "metrics.json").is_file():
                available[arm][method] = read_rank_results(result, frozen)
        per_method = available[arm]
        if {"sum", "s48"} <= per_method.keys():
            paired_s48_vs_sum[arm] = grouped_r20_contrast(per_method["sum"], per_method["s48"])
        if set(CORE_METHODS) == per_method.keys() and all(
            row["truncated"] is not None for rows in per_method.values() for row in rows
        ):
            # A geometry failure still counts on the common full-text panel.
            selected = set.intersection(*[
                {row["uid"] for row in rows if not row["truncated"] and row["query_failure"] is None}
                for rows in per_method.values()
            ])
            all_methods_full_text[arm] = {method: score_summary(
                [row["rank"] for row in rows if row["uid"] in selected], len(frozen.gallery),
            ) for method, rows in per_method.items()}

    paired_panel = {}
    for method in CORE_METHODS:
        present = {arm: available[arm][method] for arm in CONDITIONS if method in available[arm]}
        if not present:
            continue
        keep = set(paired_uid)
        paired_metrics = {arm: score_summary([row["rank"] for row in rows if row["uid"] in keep])
                          for arm, rows in present.items()}
        differences = {}
        if "ct_deepcad" in present:
            differences = {arm: contrast_between_conditions(present[arm], present["ct_deepcad"], paired_uid)
                           for arm in arms if arm in present}
        paired_panel[method] = {"metrics": paired_metrics, "ct_deepcad_minus_native_r20": differences}
    native_panel = {method: {
        arm: contrast_between_conditions(available["l0"][method], available[arm][method], sorted(native_uid))
        for arm in arms[1:] if method in available[arm]
    } for method in CORE_METHODS if method in available["l0"]}
    result = {
        "freeze_sha256": sha256(directory / "freeze.json"),
        "gallery": manifest["gallery_count"],
        "gallery_sha256": conditions["l0"].gallery_sha,
        "available_core_methods": {arm: list(methods) for arm, methods in available.items()},
        "s48_minus_sum_r20": paired_s48_vs_sum,
        "all_core_methods_full_text": all_methods_full_text,
        "ct_deepcad_common_uid_panel": {
            "queries": len(paired_uid),
            "ordered_uid_digest": text_digest("\n".join(paired_uid)),
            "methods": paired_panel,
        },
        "text2cad_native_level_minus_l0": {"queries": len(native_uid), "methods": native_panel},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparisons.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path,
                        help="Run directory containing frozen inputs and receiving evaluation results.")
    sub = parser.add_subparsers(dest="command", required=True)
    l0 = sub.add_parser("replay-l0", help="Verify all five historical L0 results from their raw scores.")
    l0.add_argument("--main-repo", required=True, type=Path)
    l0.add_argument("--asset-runtime", required=True, type=Path)
    external = sub.add_parser("external", help="Evaluate two frozen normalized embeddings or a dense matrix.")
    external.add_argument("--input", type=Path, help="Defaults to runtime/external/<method>.")
    external.add_argument("--condition", required=True, choices=CONDITIONS)
    external.add_argument("--method", required=True)
    external.add_argument("--encoding", choices=("embeddings", "scores"), default="embeddings")
    generator = sub.add_parser("text2cad", help="Evaluate a completed Text2CAD score condition.")
    generator.add_argument("--input", type=Path, help="Defaults to runtime/full.")
    generator.add_argument("--condition", required=True, choices=CONDITIONS[1:])
    sub.add_parser("references", help="Calculate random, valid-field-count, and exact-text reference rows.")
    sub.add_parser("summarize", help="Aggregate verified per-query results and pair matching UIDs.")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    directory = runtime / "frozen"
    output = runtime / "evaluation"
    if args.command == "replay-l0":
        replay = replay_l0(FrozenCondition.read(directory, "l0"), args.main_repo.resolve(),
                           args.asset_runtime.resolve(), output)
        print(f"Verified {replay['matched_ranks']} historical target ranks")
    elif args.command == "external":
        if not re.fullmatch(r"[a-z][a-z0-9_]*", args.method):
            raise ValueError("Invalid external method name")
        if args.method in ("sum", "s48", "shortest_first"):
            raise ValueError(f"Reserved method name: {args.method}")
        if args.condition == "l0" and args.method in CORE_METHODS[:3]:
            raise ValueError("The L0 core rows are pinned to historical replay")
        frozen = FrozenCondition.read(directory, args.condition)
        result = external_rows(frozen, args.input or runtime / "external" / args.method,
                               args.method, args.encoding, output)
        print(f"{args.condition}/{args.method}: {result['hits']}, MedR {result['median_rank']}")
    elif args.command == "text2cad":
        frozen = FrozenCondition.read(directory, args.condition)
        results = text2cad_rows(frozen, args.input or runtime / "full", output)
        for method, result in results.items():
            print(f"{args.condition}/{method}: {result['hits']}, MedR {result['median_rank']}")
    elif args.command == "references":
        for condition in CONDITIONS:
            result = references(FrozenCondition.read(directory, condition), output)
            print(f"{condition}: valid-field hits {result['shortest_first']['hits']}")
    else:
        result = summarize(directory, output)
        print(f"Updated {output / 'comparisons.json'} with {result['available_core_methods']}")


if __name__ == "__main__":
    main()
