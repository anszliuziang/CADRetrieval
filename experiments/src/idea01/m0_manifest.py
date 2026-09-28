"""M0: fixed Text2CAD-v1.0 retrieval benchmark query/gallery/qrels manifest builder.

This is a CPU-only, deterministic manifest generator for the Idea 01 retrieval
benchmark described in
`experiments/plans/IDEA01_TEXT2CAD_BENCHMARK_2026-09-28.md` (sections 2 and 5).

What it produces (caller-specified output dir, never the Git checkout by
default):

  query.jsonl    one record per non-empty L0 `abstract` query on the official
                 v1.0 test split. It contains the text to score and an opaque
                 query_id, but never the paired CAD UID.
  gallery.jsonl  one record per official test UID (the full 8,046 CAD gallery).
                 Fields: cad_id (== uid), cad_seq_member, minimal_json_member,
                 cadmium_native_present, n_representations. No paired caption,
                 caption-derived presence flag, or qrels.
  qrels.jsonl    one record per query: query_id, cad_id (paired UID),
                 relevance=1. Read ONLY by the evaluation module, never by the
                 scorer (input isolation contract, CLAUDE.md sec.5.1).
  audit.json     coverage, duplicate-text groups, recognisability ceiling,
                 per-representation coverage, cross-split leakage checks,
                 CADmium-ds coverage, fixed-ordering and manifest hashes, plus
                 a `ledger` of all dropped / unscoreable items with reasons.

Numbers from the plan/audit (8046 gallery, 7934 non-empty L0, 3644 unique
texts, 639 duplicate groups, 45.93% R@1 ceiling, etc.) are treated as
EXPECTATIONS TO BE TESTED, never hardcoded as results. The code recomputes
every figure from the source files and the audit records the recomputed value
alongside the expected value so a mismatch is loud.

Design rules enforced here:
  * query / gallery / qrels are physically separate files; gallery carries no
    paired text and no field that maps back to a description.
  * fixed ordering: gallery is sorted by CAD UID and queries by a stable
    pseudorandom hash of paired UID, so query rank does not reveal CAD rank.
    Re-runs are byte-identical and manifest hashes are recorded.
  * a missing query is recorded in the ledger, never fabricated as a retrieval
    failure; a missing/unscoreable CAD stays in the gallery with a coverage
    flag rather than being silently dropped.
  * input paths come from CLI args; nothing is hardcoded to `.cache`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# CSV rows can be large (all_level_data / nli_data columns).
csv.field_size_limit(2**31 - 1)

# ---- expected values from the plan / asset audit (TESTED, not trusted) ----
EXPECTED = {
    "n_test_uid": 8046,
    "n_train_uid": 159049,
    "n_val_uid": 8925,
    "n_nonempty_l0_query": 7934,
    "n_unique_l0_text": 3644,
    "n_duplicate_groups": 639,
    "max_duplicate_group": 371,
    "r1_ceiling_pct": 45.93,
    "csv_test_uid_covered": 7946,  # CSV covers 7946 of 8046 test UIDs
    "csv_test_l0_empty": 12,       # 12 covered test UIDs have empty L0
    "cadmium_test_rows": 8046,
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Source readers
# --------------------------------------------------------------------------- #
@dataclass
class SplitFile:
    path: str
    sha256: str
    size_bytes: int


def load_split_json(path: Path) -> dict:
    """Return {"train": [...], "validation": [...], "test": [...]} UID lists."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    for k in ("train", "validation", "test"):
        if k not in d:
            raise ValueError(f"{path}: missing split {k!r}")
        if not isinstance(d[k], list):
            raise ValueError(f"{path}: split {k!r} is not a list")
    return d


def stream_csv_uid_l0(csv_path: Path) -> dict[str, dict]:
    """Stream the v1.0 CSV once, returning {uid: {"abstract": str|None, ...}}.

    Only `uid` and `abstract` (L0) are retained in memory; the other caption
    levels and the large all_level_data / nli_data columns are discarded to
    keep memory bounded. The CSV is the only place the audit reads paired
    captions; nothing caption-like is written to gallery.jsonl.
    """
    out: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "uid" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: CSV missing 'uid' column")
        if "abstract" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: CSV missing 'abstract' (L0) column")
        for row in reader:
            uid = row["uid"]
            abstract = row.get("abstract")
            # Freeze the only L0 normalisation used by this benchmark. The
            # official CSV contains captions differing solely at the edges.
            abstract = (abstract.strip() or None) if abstract is not None else None
            # Keep last occurrence; UIDs are unique in this CSV but be defensive.
            out[uid] = {"abstract": abstract}
    return out


def index_cad_seq_zip(path: Path) -> dict[str, str]:
    """Map uid -> cad_seq .pth member path inside cad_seq.zip.

    Member layout: cad_seq/<pref>/<uid_tail>/seq/<uid_tail>.pth where
    uid == "<pref>/<uid_tail>". We do not read the .pth bytes (CPU/IO bound);
    presence + ZipInfo size are enough for the gallery coverage manifest.
    """
    idx: dict[str, str] = {}
    with zipfile.ZipFile(path) as zf:
        for m in zf.namelist():
            if m.endswith("/"):
                continue
            parts = m.split("/")
            # cad_seq / pref / tail / seq / tail.pth
            if len(parts) != 5 or parts[0] != "cad_seq" or parts[3] != "seq":
                continue
            uid = f"{parts[1]}/{parts[2]}"
            idx[uid] = m
    return idx


def index_minimal_json_zip(path: Path) -> dict[str, str]:
    """Map uid -> minimal_json member path inside a minimal_json ZIP.

    Member layout: <pref>/<tail>/minimal_json/<tail>.json where
    uid == "<pref>/<tail>".
    """
    idx: dict[str, str] = {}
    with zipfile.ZipFile(path) as zf:
        for m in zf.namelist():
            if not m.endswith(".json"):
                continue
            parts = m.split("/")
            # pref / tail / minimal_json / tail.json
            if len(parts) != 4 or parts[2] != "minimal_json":
                continue
            uid = f"{parts[0]}/{parts[1]}"
            idx[uid] = m
    return idx


def index_cadmium_parquet(path: Path, py_ok: bool) -> dict[str, dict]:
    """Map uid -> {"annotation_present": bool, "json_desc_present": bool}.

    CADmium-ds parquet supplies the *native target* JSON for each official test
    CAD (json_desc) plus CADmium's own annotation text. We only record presence
    flags here; the annotation text is NOT a query (queries come from Text2CAD
    L0) and must not leak into the gallery as a description.
    """
    idx: dict[str, dict] = {}
    if not py_ok:
        return idx
    import pyarrow.parquet as pq  # local import; only when available

    t = pq.read_table(str(path), columns=["uid", "annotation", "json_desc"])
    rows = t.to_pylist()
    for r in rows:
        idx[r["uid"]] = {
            "annotation_present": bool(r.get("annotation")),
            "json_desc_present": bool(r.get("json_desc")),
        }
    return idx


# --------------------------------------------------------------------------- #
# Manifest records
# --------------------------------------------------------------------------- #
@dataclass
class QueryRecord:
    query_id: str       # "Q" + zero-padded rank
    text: str           # L0 text after strip(); the scorer's actual input
    text_sha256: str
    n_chars: int
    level: str          # "L0"


@dataclass
class GalleryRecord:
    cad_id: str         # == uid
    cad_seq_member: str | None
    minimal_json_member: str | None
    cadmium_native_present: bool   # json_desc present in CADmium-ds test parquet
    n_representations: int         # count of present native representations


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
@dataclass
class BuildResult:
    audit: dict
    queries: list[QueryRecord]
    gallery: list[GalleryRecord]
    qrels: list[dict]


def _record_source(path: Path) -> SplitFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return SplitFile(path=str(path), sha256=sha256_file(path), size_bytes=path.stat().st_size)


def build_manifest(
    split_json_path: Path,
    csv_path: Path,
    cad_seq_zip_path: Path,
    minimal_json_zip_path: Path,
    cadmium_test_parquet_path: Path,
) -> BuildResult:
    """Build query/gallery/qrels + audit. Pure CPU; no model, no network."""
    # 1. Source file hashes (revision evidence).
    split_src = _record_source(split_json_path)
    csv_src = _record_source(csv_path)
    cad_seq_src = _record_source(cad_seq_zip_path)
    mj_src = _record_source(minimal_json_zip_path)
    cadmium_src = _record_source(cadmium_test_parquet_path)

    # 2. Splits.
    splits = load_split_json(split_json_path)
    train_uids = list(splits["train"])
    val_uids = list(splits["validation"])
    test_uids = list(splits["test"])
    n_train, n_val, n_test = len(train_uids), len(val_uids), len(test_uids)

    # Cross-split UID intersection must be zero (held-out integrity).
    tr, vr, te = set(train_uids), set(val_uids), set(test_uids)
    uid_intersections = {
        "train__validation": len(tr & vr),
        "train__test": len(tr & te),
        "validation__test": len(vr & te),
    }
    dup_within_test = n_test - len(te)
    dup_within_train = n_train - len(tr)
    dup_within_val = n_val - len(vr)

    # 3. CSV L0 captions for test UIDs.
    csv_index = stream_csv_uid_l0(csv_path)
    # CSV coverage on test:
    test_in_csv = [u for u in test_uids if u in csv_index]
    test_l0_nonempty = [u for u in test_in_csv if csv_index[u]["abstract"]]
    test_l0_empty = [u for u in test_in_csv if not csv_index[u]["abstract"]]
    test_not_in_csv = [u for u in test_uids if u not in csv_index]

    # 4. Native representation coverage on the full test gallery.
    cad_seq_idx = index_cad_seq_zip(cad_seq_zip_path)
    mj_idx = index_minimal_json_zip(minimal_json_zip_path)
    # CADmium native JSON: optional if pyarrow present.
    try:
        import pyarrow  # noqa: F401

        py_ok = True
    except Exception:
        py_ok = False
    cadmium_idx = index_cadmium_parquet(cadmium_test_parquet_path, py_ok)

    cad_seq_covered = [u for u in test_uids if u in cad_seq_idx]
    mj_covered = [u for u in test_uids if u in mj_idx]
    # CADmium native coverage requires a present json_desc, not mere row
    # presence (a parquet row may carry a null native target).
    cadmium_covered = [u for u in test_uids if cadmium_idx.get(u, {}).get("json_desc_present")]

    # 5. Duplicate-text groups on non-empty L0 queries (recognisability ceiling).
    # The ceiling premise is "identical visible text -> identical ranking", so
    # captions are normalised with .strip() before grouping/hashing. This makes
    # leading/trailing-whitespace variants of the same caption collapse into one
    # group (the v1.0 CSV carries such variants); the raw bytes are still
    # recoverable from the CSV at score time. This matches the plan's reported
    # figures (3644 unique / 639 groups / max 371).
    text_to_uids: dict[str, list[str]] = {}
    for u in test_l0_nonempty:
        norm = csv_index[u]["abstract"].strip()
        text_to_uids.setdefault(norm, []).append(u)
    # Stable ordering: sort each group's uids, then sort groups by (size desc,
    # first uid asc) so the largest group gets the lowest group id and the
    # ordering is deterministic and reproducible.
    groups_sorted = sorted(
        text_to_uids.values(),
        key=lambda g: (-len(g), sorted(g)[0]),
    )
    n_unique_text = len(text_to_uids)
    duplicate_groups = [g for g in groups_sorted if len(g) > 1]
    n_duplicate_groups = len(duplicate_groups)
    max_dup_group = max((len(g) for g in duplicate_groups), default=0)
    n_queries = sum(len(g) for g in groups_sorted)  # == len(test_l0_nonempty)

    # Map uid -> (group_id, group_size). group_id 0 reserved for singletons.
    uid_to_group: dict[str, tuple[int, int]] = {}
    for gid, g in enumerate(groups_sorted, start=1):
        for u in g:
            uid_to_group[u] = (gid if len(g) > 1 else 0, len(g))

    # Recognisability ceiling: with one registered positive UID per query and a
    # fixed gallery, identical text yields identical ranking, so for a group of
    # size s the best achievable hits@K is min(K, s) summed over groups.
    def ceiling(k: int) -> float:
        return 100.0 * sum(min(k, len(g)) for g in groups_sorted) / n_queries

    ceilings = {f"R@{k}": round(ceiling(k), 2) for k in (1, 2, 5, 10, 20)}

    # 6. Build query records. The public query order must not be the paired
    #    CAD UID order; only qrels carries that mapping.
    queries: list[QueryRecord] = []
    qrels: list[dict] = []
    query_strata: list[dict] = []
    query_order = sorted(
        test_l0_nonempty,
        key=lambda u: (sha256_text("M0-query-order:" + u), u),
    )
    for rank, u in enumerate(query_order):
        text = csv_index[u]["abstract"]
        gid, gsize = uid_to_group[u]
        query_id = f"Q{rank:05d}"
        queries.append(
            QueryRecord(
                query_id=query_id,
                text=text,
                text_sha256=sha256_text(text),
                n_chars=len(text),
                level="L0",
            )
        )
        qrels.append({"query_id": query_id, "cad_id": u, "relevance": 1})
        query_strata.append({
            "query_id": query_id,
            "duplicate_group_id": gid,
            "duplicate_group_size": gsize,
        })

    # 7. Build gallery records (stable order: by uid asc). Full 8,046 gallery;
    #    no caption, no UID-derivable description. Missing representations are
    #    flagged, never silently dropped.
    gallery: list[GalleryRecord] = []
    ledger: list[dict] = []
    for u in sorted(test_uids):
        has_cadseq = u in cad_seq_idx
        has_mj = u in mj_idx
        has_cad = cadmium_idx.get(u, {}).get("json_desc_present", False)
        nrep = sum([has_cadseq, has_mj, has_cad])
        gallery.append(
            GalleryRecord(
                cad_id=u,
                cad_seq_member=cad_seq_idx.get(u),
                minimal_json_member=mj_idx.get(u),
                cadmium_native_present=has_cad,
                n_representations=nrep,
            )
        )
        if not has_cadseq:
            ledger.append({"cad_id": u, "issue": "missing_cad_seq", "severity": "unscoreable_for_deepcad_arm"})
        if not has_mj:
            ledger.append({"cad_id": u, "issue": "missing_minimal_json", "severity": "unscoreable_for_cadmium_arm"})
        if not has_cad:
            ledger.append({"cad_id": u, "issue": "missing_cadmium_native", "severity": "unscoreable_for_cadmium_arm"})
    # Queries with empty / missing L0 go to the ledger (not fabricated as
    # retrieval failures).
    for u in test_l0_empty:
        ledger.append({"cad_id": u, "issue": "l0_empty", "severity": "query_dropped"})
    for u in test_not_in_csv:
        ledger.append({"cad_id": u, "issue": "uid_not_in_csv", "severity": "query_dropped"})

    # 8. qrels was built alongside queries: one positive per query. It is
    #    written to a separate file and must never be opened by the scorer.

    # 9. Duplicate-group distribution (size -> count) for stratified reporting.
    dup_size_dist = Counter(len(g) for g in duplicate_groups)

    # 10. Assemble audit.
    def _check(name: str, got, exp) -> dict:
        return {"name": name, "expected": exp, "got": got, "match": got == exp}

    audit = {
        "schema_version": "m0-v1",
        "sources": {
            "split_json": asdict(split_src),
            "csv": asdict(csv_src),
            "cad_seq_zip": asdict(cad_seq_src),
            "minimal_json_zip": asdict(mj_src),
            "cadmium_test_parquet": asdict(cadmium_src),
        },
        "pyarrow_available": py_ok,
        "splits": {
            "n_train": n_train,
            "n_validation": n_val,
            "n_test": n_test,
            "uid_intersections": uid_intersections,
            "dup_within_train": dup_within_train,
            "dup_within_validation": dup_within_val,
            "dup_within_test": dup_within_test,
        },
        "csv_coverage_on_test": {
            "test_uid_in_csv": len(test_in_csv),
            "test_l0_nonempty": len(test_l0_nonempty),
            "test_l0_empty": len(test_l0_empty),
            "test_uid_not_in_csv": len(test_not_in_csv),
        },
        "representation_coverage_on_test": {
            "cad_seq": len(cad_seq_covered),
            "minimal_json": len(mj_covered),
            "cadmium_native_json": len(cadmium_covered),
        },
        "queries": {
            "n_queries": n_queries,
            "n_unique_text": n_unique_text,
            "n_duplicate_groups": n_duplicate_groups,
            "max_duplicate_group": max_dup_group,
            "duplicate_size_distribution": dict(sorted(dup_size_dist.items())),
            "recognisability_ceilings_pct": ceilings,
        },
        # Evaluation metadata; the scorer must not open audit.json.
        "query_strata": query_strata,
        "gallery": {
            "n_gallery": len(gallery),
            "n_with_cad_seq": sum(1 for g in gallery if g.cad_seq_member),
            "n_with_minimal_json": sum(1 for g in gallery if g.minimal_json_member),
            "n_with_cadmium_native": sum(1 for g in gallery if g.cadmium_native_present),
            "n_with_abstract": len(test_l0_nonempty),
        },
        "expectations_checked": [
            _check("n_test_uid", n_test, EXPECTED["n_test_uid"]),
            _check("n_train_uid", n_train, EXPECTED["n_train_uid"]),
            _check("n_val_uid", n_val, EXPECTED["n_val_uid"]),
            _check("n_nonempty_l0_query", n_queries, EXPECTED["n_nonempty_l0_query"]),
            _check("n_unique_l0_text", n_unique_text, EXPECTED["n_unique_l0_text"]),
            _check("n_duplicate_groups", n_duplicate_groups, EXPECTED["n_duplicate_groups"]),
            _check("max_duplicate_group", max_dup_group, EXPECTED["max_duplicate_group"]),
            _check("csv_test_uid_covered", len(test_in_csv), EXPECTED["csv_test_uid_covered"]),
            _check("csv_test_l0_empty", len(test_l0_empty), EXPECTED["csv_test_l0_empty"]),
            _check("cadmium_test_rows", len(cadmium_idx), EXPECTED["cadmium_test_rows"]),
        ],
        "ledger": ledger,
        "ledger_counts": dict(Counter(e["issue"] for e in ledger)),
    }
    return BuildResult(audit=audit, queries=queries, gallery=gallery, qrels=qrels)


# --------------------------------------------------------------------------- #
# Output writer (caller-specified dir; deterministic, hashed)
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records: list) -> None:
    buf = io.StringIO()
    for r in records:
        d = asdict(r) if hasattr(r, "__dataclass_fields__") else r
        buf.write(json.dumps(d, ensure_ascii=False, sort_keys=True))
        buf.write("\n")
    path.write_text(buf.getvalue(), encoding="utf-8")


def write_outputs(out_dir: Path, res: BuildResult) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / "query.jsonl"
    g_path = out_dir / "gallery.jsonl"
    r_path = out_dir / "qrels.jsonl"
    a_path = out_dir / "audit.json"
    _write_jsonl(q_path, res.queries)
    _write_jsonl(g_path, res.gallery)
    _write_jsonl(r_path, res.qrels)
    a_path.write_text(
        json.dumps(res.audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # Stable manifest hashes (over the exact bytes written).
    hashes = {
        "query.jsonl": sha256_file(q_path),
        "gallery.jsonl": sha256_file(g_path),
        "qrels.jsonl": sha256_file(r_path),
        "audit.json": sha256_file(a_path),
    }
    return {"out_dir": str(out_dir), "manifest_hashes": hashes, "n_query": len(res.queries),
            "n_gallery": len(res.gallery), "n_qrels": len(res.qrels)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build fixed Text2CAD-v1.0 retrieval benchmark manifests (M0, CPU only)."
    )
    p.add_argument("--split-json", required=True, type=Path,
                   help="official train_test_val.json")
    p.add_argument("--csv", required=True, type=Path,
                   help="official text2cad_v1.0.csv (L0 abstract source)")
    p.add_argument("--cad-seq-zip", required=True, type=Path,
                   help="official cad_seq.zip (DeepCAD command vectors)")
    p.add_argument("--minimal-json-zip", required=True, type=Path,
                   help="official minimal_json ZIP (CADmium native JSON)")
    p.add_argument("--cadmium-test-parquet", required=True, type=Path,
                   help="CADmium-ds test parquet (native target JSON)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="output directory for manifests (NOT the Git checkout)")
    args = p.parse_args(argv)
    res = build_manifest(
        args.split_json, args.csv, args.cad_seq_zip,
        args.minimal_json_zip, args.cadmium_test_parquet,
    )
    summary = write_outputs(args.out_dir, res)
    # Keep CLI output short (<= a few lines) to control context/cost.
    print(json.dumps(summary, ensure_ascii=False))
    # Exit non-zero if any expectation check failed, so CI/tests notice.
    failed = [c for c in res.audit["expectations_checked"] if not c["match"]]
    if failed:
        print("EXPECTATION_MISMATCH: " + ", ".join(f"{c['name']}(got={c['got']},exp={c['expected']})" for c in failed))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
