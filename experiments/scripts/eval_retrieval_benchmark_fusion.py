"""Validate the final CadTranslator-Fusion cohort and compute its CPU reference rows."""

from __future__ import annotations

import argparse
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval_retrieval_benchmark_fill import (
    K_LEVELS, FrozenCondition, gallery_accepted, rank_text, score_summary, valid_gallery,
)
from prepare_retrieval_benchmark_fill import (
    load_json, read_jsonl, sha256, text_digest, verify_sha256, write_json, write_jsonl,
)


CONDITION = "ct_fusion"
DOMAIN = "Fusion360"
REGISTERED_CANDIDATES = 6757
REGISTERED_QUERIES = 6502
MAX_VALID_FIELDS = (272 - 1) * 2  # The two heads score target positions after the start row.
PAYLOADS = frozenset({
    "gallery.jsonl", "texts_ct_fusion.jsonl", "queries_ct_fusion.jsonl", "qrels.jsonl", "gallery_fields.npy",
})


def verify_fusion_manifest(directory: Path, expected_sha256: str) -> dict:
    manifest_path = directory / "freeze.json"
    verify_sha256(manifest_path, expected_sha256)
    manifest = load_json(manifest_path)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("domain") != DOMAIN or manifest.get("gallery_order") != "ascending Fusion UID"
            or manifest.get("candidate_program_source") != "converted_h5"
            or manifest.get("candidate_member_root") != "preparation_run_directory"
            or set(manifest.get("conditions", {})) != {CONDITION}):
        raise ValueError("Expected the final CadTranslator-Fusion freeze schema")
    replay = manifest.get("source_replay", {})
    if (not isinstance(replay, dict) or replay.get("scope") != "full"
            or replay.get("complete_source_index") is not True
            or replay.get("n_candidates") != REGISTERED_CANDIDATES
            or any(not isinstance(replay.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", replay[key])
                   for key in ("replay_pin_sha256", "objects_sha256"))):
        raise ValueError("Expected a certified full Fusion replay over the registered inventory")
    outcomes = replay.get("outcomes", {})
    n_gallery = manifest.get("gallery_count")
    if (not isinstance(outcomes, dict) or type(n_gallery) is not int
            or not 0 < n_gallery <= REGISTERED_CANDIDATES
            or any(type(n) is not int or n < 0 for n in outcomes.values())
            or outcomes.get("accepted") != n_gallery or sum(outcomes.values()) != REGISTERED_CANDIDATES):
        raise ValueError("Final eligible gallery count differs from the full Fusion replay ledger")

    entries = list(directory.iterdir())
    files = manifest.get("files", {})
    if (set(files) != PAYLOADS or {path.name for path in entries} != PAYLOADS | {"freeze.json"}
            or any(path.is_symlink() or not path.is_file() for path in entries)):
        raise ValueError("Unexpected Fusion freeze file coverage")
    for name, expected in files.items():
        verify_sha256(directory / name, expected)
    return manifest


def read_fusion_condition(directory: Path, expected_sha256: str) -> tuple[FrozenCondition, np.ndarray]:
    manifest = verify_fusion_manifest(directory, expected_sha256)
    gallery = read_jsonl(directory / "gallery.jsonl")
    texts = read_jsonl(directory / f"texts_{CONDITION}.jsonl")
    queries = read_jsonl(directory / f"queries_{CONDITION}.jsonl")
    qrels = read_jsonl(directory / "qrels.jsonl")
    if any(not isinstance(row, dict) for axis in (gallery, texts, queries, qrels) for row in axis):
        raise ValueError("Expected object rows on all Fusion axes")

    n_gallery = manifest["gallery_count"]
    uids = [row.get("uid") for row in gallery]
    if (len(gallery) != n_gallery
            or any(not isinstance(uid, str) or not re.fullmatch(r"[0-9]{4}/[0-9]{8}", uid) for uid in uids)
            or uids != sorted(set(uids))
            or any(type(row.get("cad_index")) is not int or row["cad_index"] != index
                   or row.get("member") != f"objects/{uid.replace('/', '_')}/program.npz"
                   or row.get("h5_member") != f"dataset/fusion_360_txt/cad_vec/{uid}.h5"
                   for index, (row, uid) in enumerate(zip(gallery, uids)))):
        raise ValueError("Invalid Fusion gallery UID, CAD index or program member axis")

    digests = [row.get("text_sha256") for row in texts]
    if (not texts or any(type(row.get("text_index")) is not int or row["text_index"] != index
                         or not isinstance(row.get("text"), str) or not row["text"]
                         or row["text"].strip() != row["text"] or digest != text_digest(row["text"])
                         for index, (row, digest) in enumerate(zip(texts, digests)))
            or digests != sorted(set(digests))):
        raise ValueError("Invalid ordered Fusion text axis or original text digest")

    by_text = defaultdict(list)
    previous_index = -1
    for row in queries:
        index, text_index = row.get("cad_index"), row.get("text_index")
        if (set(row) != {"uid", "cad_index", "text_index"}
                or type(index) is not int or not previous_index < index < n_gallery
                or row["uid"] != uids[index]
                or type(text_index) is not int or not 0 <= text_index < len(texts)):
            raise ValueError("Invalid Fusion query axis or positive CAD link")
        previous_index = index
        by_text[text_index].append(row)
    if set(by_text) != set(range(len(texts))):
        raise ValueError("Invalid Fusion query axis with an unlinked text")
    expected_qrels = [{"uid": row["uid"], "candidate_uid": row["uid"], "relevance": 1} for row in queries]
    if qrels != expected_qrels or any(type(row["relevance"]) is not int for row in qrels):
        raise ValueError("Fusion qrels differ from the ordered query positive UID links")

    counts = Counter({index: len(group) for index, group in by_text.items()})
    expected_counts = {
        "queries": len(queries), "unique_texts": len(texts), "largest_group": max(counts.values()),
        "singleton_queries": sum(n == 1 for n in counts.values()),
    }
    stats = manifest["conditions"][CONDITION]
    if (stats != expected_counts or any(type(n) is not int for n in stats.values())
            or len(queries) > REGISTERED_QUERIES or n_gallery - len(queries) > REGISTERED_CANDIDATES - REGISTERED_QUERIES):
        raise ValueError("Final eligible Fusion query and captionless counts differ from the freeze")

    fields = np.load(directory / "gallery_fields.npy", allow_pickle=False)
    if (fields.dtype != np.dtype("<i4") or fields.shape != (n_gallery,)
            or np.any(fields < 1) or np.any(fields > MAX_VALID_FIELDS)):
        raise ValueError("Frozen Fusion valid field counts do not match the gallery axis")
    positions = {index: index for index in range(n_gallery)}
    return FrozenCondition(directory, CONDITION, manifest, gallery, texts, queries, by_text, counts, positions), fields


def uniform_reference(n_gallery: int, n_queries: int) -> dict:
    fractions = [min(k, n_gallery) / n_gallery for k in K_LEVELS]
    return {
        "mode": "analytical uniform ranking",
        "queries": n_queries,
        "expected_hits": [n_queries * fraction for fraction in fractions],
        "recall_percent": [100 * fraction if n_queries else None for fraction in fractions],
        "population_median_rank": (n_gallery + 1) / 2 if n_queries else None,
    }


def export_fusion_references(directory: Path, expected_sha256: str, output: Path) -> dict:
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Reference output must be outside the frozen input directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Reference output already exists: {output}")
    frozen, fields = read_fusion_condition(directory, expected_sha256)
    n_gallery, n_queries = len(frozen.gallery), len(frozen.queries)
    gallery_status = gallery_accepted(frozen)
    valid = valid_gallery(gallery_status)
    scores = -fields.astype("<f8")
    query_status = {"valid": True, "truncated": False}
    by_uid = {row["uid"]: row for index in range(len(frozen.texts))
              for row in rank_text(frozen, index, scores, query_status, gallery_status, valid)}
    ranks = [by_uid[query["uid"]] for query in frozen.queries]
    singleton = [row["rank"] for row in ranks if frozen.text_counts[row["text_index"]] == 1]
    upper_ranks = [rank for count in frozen.text_counts.values() for rank in range(1, count + 1)]
    n_singleton = len(singleton)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fusion-references-", dir=output.parent) as temp:
        stage = Path(temp) / "payload"
        stage.mkdir()
        rank_path = stage / "shortest_first_query_ranks.jsonl"
        write_jsonl(rank_path, ranks)
        summary = {
            "schema_version": 1, "table": 3, "domain": DOMAIN, "condition": CONDITION,
            "k": list(K_LEVELS), "gallery": n_gallery, "queries": n_queries,
            "captionless_candidates": n_gallery - n_queries,
            "distinct_original_texts": len(frozen.texts), "singleton_queries": n_singleton,
            "largest_same_text_group": max(frozen.text_counts.values()),
            "frozen_manifest_sha256": expected_sha256,
            "gallery_sha256": frozen.gallery_sha, "query_axis_sha256": frozen.query_sha,
            "text_axis_sha256": frozen.text_sha, "qrels_sha256": frozen.manifest["files"]["qrels.jsonl"],
            "source_replay": frozen.manifest["source_replay"],
            "random": {**uniform_reference(n_gallery, n_queries),
                       "singleton": uniform_reference(n_gallery, n_singleton)},
            "shortest_first": {
                "mode": "valid-fields ascending, then CAD index ascending",
                **score_summary([row["rank"] for row in ranks], n_gallery),
                "singleton": score_summary(singleton, n_gallery), "failed_queries": 0,
                "native_field_counts_sha256": frozen.manifest["files"]["gallery_fields.npy"],
                "query_ranks_file": rank_path.name, "query_ranks_sha256": sha256(rank_path),
            },
            "same_text_upper_bound": {
                "mode": "independent ideal ordering per distinct original text",
                **score_summary(upper_ranks, n_gallery),
                "singleton": score_summary([1] * n_singleton, n_gallery),
            },
        }
        write_json(stage / "references.json", summary)
        stage.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path, help="Final Fusion freeze with replay certification")
    parser.add_argument("--freeze-sha256", required=True, help="Verified SHA-256 of freeze.json from preparation")
    parser.add_argument("--output", required=True, type=Path, help="New directory for reference artifacts")
    args = parser.parse_args()
    report = export_fusion_references(args.frozen, args.freeze_sha256, args.output)
    print(f"Verified {report['gallery']} Fusion candidates and {report['queries']} queries. "
          f"References saved to {args.output / 'references.json'}")


if __name__ == "__main__":
    main()
