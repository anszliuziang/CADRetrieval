"""Tests for the M0 Text2CAD-v1.0 manifest builder.

Uses small synthetic fixtures that mimic the official file layouts so the
builder can be verified without touching the real (large) assets. The real
assets are exercised once, by hand, via the CLI in M0_IMPLEMENTATION.md — not
in the unit tests (CLAUDE.md: keep tests light; no model forward).
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from experiments.src.idea01 import m0_manifest as M


# --------------------------------------------------------------------------- #
# Fixture builders (synthetic, mirror official layouts)
# --------------------------------------------------------------------------- #
def _make_split_json(path: Path, train, val, test):
    path.write_text(json.dumps({"train": train, "validation": val, "test": test}), encoding="utf-8")


def _make_csv(path: Path, rows: list[tuple[str, str | None]]):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["uid", "abstract", "beginner", "intermediate", "expert", "all_level_data", "nli_data"])
    for uid, abstract in rows:
        w.writerow([uid, abstract if abstract is not None else "", "b", "i", "e", "ald", "nli"])
    path.write_text(buf.getvalue(), encoding="utf-8")


def _make_cad_seq_zip(path: Path, uids: list[str]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for uid in uids:
            pref, tail = uid.split("/")
            zf.writestr(f"cad_seq/{pref}/{tail}/seq/{tail}.pth", b"\x00\x01\x02")


def _make_minimal_json_zip(path: Path, uids: list[str]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for uid in uids:
            pref, tail = uid.split("/")
            zf.writestr(f"{pref}/{tail}/minimal_json/{tail}.json", json.dumps({"parts": {}}))


def _make_cadmium_parquet(path: Path, rows: list[tuple[str, str | None, str | None]]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "uid": [r[0] for r in rows],
            "annotation": [r[1] for r in rows],
            "json_desc": [r[2] for r in rows],
        }
    )
    pq.write_table(table, str(path))


@pytest.fixture
def synthetic(tmp_path: Path):
    train = ["0000/00000001", "0000/00000002"]
    val = ["0000/00000003"]
    # 6 test UIDs. Duplicate-text groups:
    #   text "shared" -> 3 UIDs (a duplicate group of size 3)
    #   text "pair"  -> 2 UIDs (group of size 2)
    #   text "unique1" -> 1 UID (singleton)
    #   one UID with empty L0 -> query dropped
    #   one UID not in CSV -> query dropped, but stays in gallery
    test = [
        "0000/00000010",  # shared
        "0000/00000011",  # shared
        "0000/00000012",  # shared
        "0000/00000013",  # pair
        "0000/00000014",  # pair
        "0000/00000015",  # unique1
        "0000/00000016",  # empty L0
        "0000/00000017",  # not in CSV
    ]
    split = tmp_path / "split.json"
    _make_split_json(split, train, val, test)

    csv_path = tmp_path / "t2c.csv"
    csv_rows = [
        ("0000/00000010", "shared"),
        ("0000/00000011", " shared "),  # edge whitespace is not a new query
        ("0000/00000012", "shared"),
        ("0000/00000013", "pair"),
        ("0000/00000014", "pair"),
        ("0000/00000015", "unique1"),
        ("0000/00000016", ""),  # empty L0
        # 0000/00000017 deliberately absent from CSV
    ]
    _make_csv(csv_path, csv_rows)

    cad_seq = tmp_path / "cad_seq.zip"
    _make_cad_seq_zip(cad_seq, test)  # all test UIDs covered

    mj = tmp_path / "minimal_json.zip"
    _make_minimal_json_zip(mj, test)  # all covered

    cadmium = tmp_path / "cadmium_test.parquet"
    # 0000/00000016 missing json_desc (unscoreable for cadmium arm)
    cad_rows = [
        ("0000/00000010", "ann", '{"parts": {}}'),
        ("0000/00000011", "ann", '{"parts": {}}'),
        ("0000/00000012", "ann", '{"parts": {}}'),
        ("0000/00000013", "ann", '{"parts": {}}'),
        ("0000/00000014", "ann", '{"parts": {}}'),
        ("0000/00000015", "ann", '{"parts": {}}'),
        ("0000/00000016", "ann", None),     # missing native json_desc
        ("0000/00000017", "ann", '{"parts": {}}'),
    ]
    _make_cadmium_parquet(cadmium, cad_rows)

    out_dir = tmp_path / "out"
    return {
        "split": split, "csv": csv_path, "cad_seq": cad_seq,
        "mj": mj, "cadmium": cadmium, "out_dir": out_dir, "test": test,
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_gallery_full_and_no_caption_leak(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    # Full gallery: all 8 test UIDs, including the empty-L0 and not-in-CSV ones.
    assert len(res.gallery) == 8
    cad_ids = {g.cad_id for g in res.gallery}
    assert cad_ids == set(synthetic["test"])
    # No gallery field carries paired caption text or a UID-derivable description.
    for g in res.gallery:
        d = M.asdict(g)
        for forbidden in ("abstract", "annotation", "caption", "intermediate", "expert"):
            assert forbidden not in d
        # members are paths, not descriptions; cad_seq_member must not contain
        # the raw caption.
        assert g.cad_seq_member is None or g.cad_seq_member.startswith("cad_seq/")


def test_query_qrels_physical_separation(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    # 6 non-empty L0 queries (shared x3 + pair x2 + unique1).
    assert len(res.queries) == 6
    assert len(res.qrels) == 6
    # The scorer sees the actual query text, with no paired CAD identifier.
    for q in res.queries:
        d = M.asdict(q)
        assert d["text"] in {"shared", "pair", "unique1"}
        assert "uid" not in d and "cad_id" not in d
        assert "duplicate_group_id" not in d and "duplicate_group_size" not in d
        assert d["level"] == "L0"
        assert d["n_chars"] > 0
        assert len(d["text_sha256"]) == 64
    # qrels only carry ids + relevance.
    for qr in res.qrels:
        assert set(qr.keys()) == {"query_id", "cad_id", "relevance"}
        assert qr["relevance"] == 1
    # Every query's paired UID appears only in qrels, exactly once.
    assert {qr["cad_id"] for qr in res.qrels} == set(synthetic["test"][:6])
    assert {qr["query_id"] for qr in res.qrels} == {q.query_id for q in res.queries}


def test_duplicate_groups_and_ceiling(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    a = res.audit["queries"]
    assert a["n_queries"] == 6
    assert a["n_unique_text"] == 3          # shared, pair, unique1
    assert a["n_duplicate_groups"] == 2     # shared(3), pair(2)
    assert a["max_duplicate_group"] == 3
    # Ceiling: group sizes [3,2,1], n=6.
    # R@1 = (min(1,3)+min(1,2)+min(1,1))/6 = 3/6 = 50.0
    # R@2 = (2+2+1)/6 = 5/6 = 83.33
    # R@5 = (3+2+1)/6 = 6/6 = 100.0
    assert a["recognisability_ceilings_pct"]["R@1"] == pytest.approx(50.0, abs=0.01)
    assert a["recognisability_ceilings_pct"]["R@2"] == pytest.approx(83.33, abs=0.01)
    assert a["recognisability_ceilings_pct"]["R@5"] == 100.0
    # Group sizes are evaluation strata in audit.json, outside scorer input.
    strata = {x["query_id"]: x["duplicate_group_size"] for x in res.audit["query_strata"]}
    sizes_by_text = {q.text: strata[q.query_id] for q in res.queries}
    assert sizes_by_text["unique1"] == 1
    assert sizes_by_text["shared"] == 3


def test_ledger_records_drops_not_fabricated(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    ledger = res.audit["ledger"]
    issues = res.audit["ledger_counts"]
    # empty L0 and not-in-CSV are query drops; missing cadmium native for 0000/00000016.
    assert issues.get("l0_empty") == 1
    assert issues.get("uid_not_in_csv") == 1
    assert issues.get("missing_cadmium_native") == 1
    # No query fabricated for the dropped UIDs.
    q_uids = {qr["cad_id"] for qr in res.qrels}
    assert "0000/00000016" not in q_uids
    assert "0000/00000017" not in q_uids
    # Dropped UIDs still present in gallery (not silently shrunk).
    g_uids = {g.cad_id for g in res.gallery}
    assert "0000/00000016" in g_uids
    assert "0000/00000017" in g_uids


def test_representation_coverage(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    cov = res.audit["representation_coverage_on_test"]
    assert cov["cad_seq"] == 8
    assert cov["minimal_json"] == 8
    assert cov["cadmium_native_json"] == 7  # one missing json_desc
    g = res.audit["gallery"]
    assert g["n_with_cadmium_native"] == 7
    assert g["n_with_abstract"] == 6        # 6 non-empty L0


def test_cross_split_zero_intersection(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    inter = res.audit["splits"]["uid_intersections"]
    assert inter["train__test"] == 0
    assert inter["validation__test"] == 0
    assert inter["train__validation"] == 0


def test_deterministic_byte_identical(synthetic):
    def build_once():
        res = M.build_manifest(
            synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
            synthetic["mj"], synthetic["cadmium"],
        )
        s = M.write_outputs(synthetic["out_dir"], res)
        h = s["manifest_hashes"]
        return h, s

    h1, s1 = build_once()
    h2, s2 = build_once()
    assert h1 == h2, "manifest hashes must be byte-identical across runs"
    # Stable query IDs, without exposing the paired UID in query records.
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    assert [q.query_id for q in res.queries] == [f"Q{i:05d}" for i in range(6)]
    assert all(not hasattr(q, "uid") for q in res.queries)
    assert all(not hasattr(g, "abstract_present") for g in res.gallery)
    assert [r["cad_id"] for r in res.qrels] != sorted(r["cad_id"] for r in res.qrels)


def test_expectation_check_reports_mismatch(synthetic):
    res = M.build_manifest(
        synthetic["split"], synthetic["csv"], synthetic["cad_seq"],
        synthetic["mj"], synthetic["cadmium"],
    )
    # Synthetic counts intentionally differ from the real-asset expectations.
    mismatches = [c for c in res.audit["expectations_checked"] if not c["match"]]
    assert len(mismatches) > 0  # e.g. n_test_uid 8 != 8046
    # The expected values from the plan are recorded alongside the got values.
    names = {c["name"] for c in mismatches}
    assert "n_test_uid" in names


def test_cli_writes_files_and_returns_zero_on_match(tmp_path, synthetic, monkeypatch):
    # Point expectations at the synthetic counts so CLI returns 0.
    monkeypatch.setattr(M, "EXPECTED", {
        "n_test_uid": 8, "n_train_uid": 2, "n_val_uid": 1,
        "n_nonempty_l0_query": 6, "n_unique_l0_text": 3, "n_duplicate_groups": 2,
        "max_duplicate_group": 3, "r1_ceiling_pct": 50.0,
        "csv_test_uid_covered": 7, "csv_test_l0_empty": 1, "cadmium_test_rows": 8,
    })
    out = tmp_path / "cli_out"
    rc = M.main([
        "--split-json", str(synthetic["split"]),
        "--csv", str(synthetic["csv"]),
        "--cad-seq-zip", str(synthetic["cad_seq"]),
        "--minimal-json-zip", str(synthetic["mj"]),
        "--cadmium-test-parquet", str(synthetic["cadmium"]),
        "--out-dir", str(out),
    ])
    assert rc == 0
    assert (out / "query.jsonl").exists()
    assert (out / "gallery.jsonl").exists()
    assert (out / "qrels.jsonl").exists()
    assert (out / "audit.json").exists()
