"""CPU tests for the Text2CAD outer pair/index builder (M1c stage B2).

Covers, without any model forward or network:

  * frozen-M0 input validation: exact schemas, field integrity (n_chars /
    text_sha256 / strip-form), duplicate ids, ordering, canonical member
    paths, expectation counts -- every refusal path with its stable reason;
  * canonical unique-text indexing: content-addressed order independent of
    query order, byte-identical rebuilds;
  * the implicit pair space: 3,644 x 8,046 = 29,319,624 pairs (real dims),
    pair_index <-> (text_index, cad_index) round-trips, shard disjointness /
    completeness / contiguity for many shard counts, chunk coverage;
  * resume verification: complete chunk accepted, every corruption refused;
  * artifact loaders: schema strictness, tamper (sha) detection, contiguity;
  * zip cross-check and the zero-forward length scan on a synthetic zip;
  * isolation: worker-facing artifacts carry exactly their allowed keys and
    no UID/query_id/qrels semantics; the committed asset-SHA config pins the
    same zip hash this module uses;
  * (skipUnless the pinned local assets exist) an end-to-end rebuild from the
    frozen M0 manifests whose artifact hashes match the committed reviewed
    fingerprints.

Runs on CPU only; torch is used solely for the synthetic-zip scan tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_SEQ_ZIP_SHA256,
    CHUNK_BINDING_FIELDS,
    EXPECTED_N_GALLERY,
    EXPECTED_N_PAIRS,
    EXPECTED_N_QUERY,
    EXPECTED_N_UNIQUE_TEXT,
    PAIR_MANIFEST_FILE,
    PairIndexError,
    PairSpace,
    build_pair_manifest,
    chunk_file_name,
    load_cad_length_index,
    load_cad_store,
    load_gallery_records,
    load_pair_manifest,
    load_query_records,
    load_texts,
    scan_cad_lengths,
    sha256_file,
    sha256_text,
    verify_chunk_records,
    verify_zip_members,
    write_pair_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_CONFIG = REPO_ROOT / "experiments/configs/text2cad_original_asset_sha256_20260930.json"
EXPECTED_CONFIG = REPO_ROOT / "experiments/configs/text2cad_pair_manifest_expected_20260930.json"
M0_DIR = REPO_ROOT / ".cache/m0_real_run_v3"
CAD_SEQ_ZIP = REPO_ROOT / ".cache/text2cad_official/cad_seq.zip"


# ---------------------------------------------------------------------------
# Synthetic frozen-M0 fixtures
# ---------------------------------------------------------------------------


def _query_record(rank: int, text: str) -> dict:
    return {
        "level": "L0",
        "n_chars": len(text),
        "query_id": f"Q{rank:05d}",
        "text": text,
        "text_sha256": sha256_text(text),
    }


def _gallery_record(cad_id: str) -> dict:
    tail = cad_id.split("/")[-1]
    return {
        "cad_id": cad_id,
        "cad_seq_member": f"cad_seq/{cad_id}/seq/{tail}.pth",
        "cadmium_native_present": True,
        "minimal_json_member": f"{cad_id}/minimal_json/{tail}.json",
        "n_representations": 3,
    }


#: 5 queries, 3 unique texts ("alpha" x2, "beta", "gamma x" variant), 4 CADs.
TEXTS = ["alpha", "beta", "gamma"]
QUERIES = ["alpha", "beta", "alpha", "gamma", "beta"]
CAD_IDS = ["0000/00000001", "0000/00000002", "0001/00010000", "0099/00999999"]


def _write_mini_m0(root: Path) -> tuple[Path, Path]:
    query_path = root / "query.jsonl"
    gallery_path = root / "gallery.jsonl"
    query_path.write_text(
        "".join(json.dumps(_query_record(i, t), sort_keys=True) + "\n" for i, t in enumerate(QUERIES)),
        encoding="utf-8",
    )
    gallery_path.write_text(
        "".join(json.dumps(_gallery_record(u), sort_keys=True) + "\n" for u in CAD_IDS),
        encoding="utf-8",
    )
    return query_path, gallery_path


def _build_mini(root: Path, **kwargs):
    query_path, gallery_path = _write_mini_m0(root)
    return build_pair_manifest(
        query_path,
        gallery_path,
        expected_n_query=len(QUERIES),
        expected_n_unique_text=len(TEXTS),
        expected_n_gallery=len(CAD_IDS),
        **kwargs,
    )


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _expect_reason(self, reason: str, fn) -> None:
        with self.assertRaises(PairIndexError) as ctx:
            fn()
        self.assertEqual(ctx.exception.reason, reason, str(ctx.exception))

    def test_valid_mini_manifest_loads(self):
        query_path, _ = _write_mini_m0(self.root)
        records = load_query_records(query_path)
        self.assertEqual(len(records), len(QUERIES))

    def test_query_extra_key_refused(self):
        query_path, _ = _write_mini_m0(self.root)
        bad = _query_record(0, "alpha")
        bad["uid"] = "0000/00000001"  # relevance-linkable field must refuse
        query_path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("schema_mismatch", lambda: load_query_records(query_path))

    def test_query_missing_key_refused(self):
        query_path, _ = _write_mini_m0(self.root)
        bad = _query_record(0, "alpha")
        del bad["n_chars"]
        query_path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("schema_mismatch", lambda: load_query_records(query_path))

    def test_query_duplicate_json_key_refused(self):
        query_path, _ = _write_mini_m0(self.root)
        query_path.write_text(
            '{"level": "L0", "n_chars": 5, "query_id": "Q00000", "text": "alpha", '
            '"text": "beta", "text_sha256": "x"}\n',
            encoding="utf-8",
        )
        self._expect_reason("duplicate_key", lambda: load_query_records(query_path))

    def test_query_bad_json_refused(self):
        self._expect_reason(
            "bad_json", lambda: load_query_records(self.root / "missing.jsonl")
        )

    def test_query_id_format_and_uniqueness(self):
        query_path, _ = _write_mini_m0(self.root)
        bad = [_query_record(0, "alpha"), _query_record(0, "beta")]
        query_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in bad), encoding="utf-8"
        )
        self._expect_reason("duplicate_id", lambda: load_query_records(query_path))
        worse = _query_record(0, "alpha")
        worse["query_id"] = "Q0"  # not Q%05d
        query_path.write_text(json.dumps(worse, sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("field_mismatch", lambda: load_query_records(query_path))

    def test_query_integrity_fields_checked(self):
        query_path, _ = _write_mini_m0(self.root)
        bad = _query_record(0, "alpha")
        bad["n_chars"] = 999
        query_path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("field_mismatch", lambda: load_query_records(query_path))
        bad = _query_record(0, "alpha")
        bad["text_sha256"] = "0" * 64
        query_path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("field_mismatch", lambda: load_query_records(query_path))
        bad = _query_record(0, " alpha ")  # not in strip() form
        bad["n_chars"] = len(bad["text"])
        bad["text_sha256"] = sha256_text(bad["text"])
        query_path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("text_drift", lambda: load_query_records(query_path))

    def test_gallery_validation_paths(self):
        _, gallery_path = _write_mini_m0(self.root)
        # unsorted
        recs = [_gallery_record(u) for u in reversed(CAD_IDS)]
        gallery_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8"
        )
        self._expect_reason("order_mismatch", lambda: load_gallery_records(gallery_path))
        # duplicate
        recs = [_gallery_record(CAD_IDS[0]), _gallery_record(CAD_IDS[0])]
        gallery_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8"
        )
        self._expect_reason("duplicate_id", lambda: load_gallery_records(gallery_path))
        # bad cad_id pattern
        recs = [_gallery_record("0000/00000001")]
        recs[0]["cad_id"] = "0/1"
        gallery_path.write_text(json.dumps(recs[0], sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("field_mismatch", lambda: load_gallery_records(gallery_path))
        # member pattern drift
        recs = [_gallery_record(CAD_IDS[0])]
        recs[0]["cad_seq_member"] = "other/path.pth"
        gallery_path.write_text(json.dumps(recs[0], sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("member_mismatch", lambda: load_gallery_records(gallery_path))
        # member not canonical for its cad_id
        recs = [_gallery_record(CAD_IDS[0])]
        recs[0]["cad_seq_member"] = "cad_seq/0000/00000001/seq/00000002.pth"
        gallery_path.write_text(json.dumps(recs[0], sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("member_mismatch", lambda: load_gallery_records(gallery_path))
        # member null (unscoreable in this arm)
        recs = [_gallery_record(CAD_IDS[0])]
        recs[0]["cad_seq_member"] = None
        gallery_path.write_text(json.dumps(recs[0], sort_keys=True) + "\n", encoding="utf-8")
        self._expect_reason("member_mismatch", lambda: load_gallery_records(gallery_path))

    def test_expectation_mismatch_is_hard_failure(self):
        query_path, gallery_path = _write_mini_m0(self.root)
        with self.assertRaises(PairIndexError) as ctx:
            build_pair_manifest(query_path, gallery_path)  # real expectations (7934...)
        self.assertEqual(ctx.exception.reason, "count_mismatch")
        self.assertIn("n_query", str(ctx.exception))


class TextIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_canonical_order_is_content_addressed(self):
        built = _build_mini(self.root)
        entries = built["text_entries"]
        got = [e.text for e in entries]
        expected = sorted(set(QUERIES), key=lambda t: (sha256_text(t), t))
        self.assertEqual(got, expected)
        # Shuffling query order cannot change the text index.
        shuffled = list(reversed(QUERIES))
        query_path = self.root / "query.jsonl"
        query_path.write_text(
            "".join(
                json.dumps(_query_record(i, t), sort_keys=True) + "\n" for i, t in enumerate(shuffled)
            ),
            encoding="utf-8",
        )
        built2 = build_pair_manifest(
            query_path,
            self.root / "gallery.jsonl",
            expected_n_query=len(QUERIES),
            expected_n_unique_text=len(TEXTS),
            expected_n_gallery=len(CAD_IDS),
        )
        self.assertEqual([e.text for e in built2["text_entries"]], expected)

    def test_query_to_text_mapping(self):
        built = _build_mini(self.root)
        text_of = {e.text: e.text_index for e in built["text_entries"]}
        for i, text in enumerate(QUERIES):
            self.assertEqual(built["query_to_text"][f"Q{i:05d}"], text_of[text])

    def test_byte_identical_rebuild(self):
        built = _build_mini(self.root)
        out_a = self.root / "out_a"
        out_b = self.root / "out_b"
        ra = write_pair_manifest(out_a, built)
        rb = write_pair_manifest(out_b, built)
        self.assertEqual(ra["summary"]["artifact_sha256"], rb["summary"]["artifact_sha256"])
        for name in ra["summary"]["artifact_sha256"]:
            self.assertEqual(
                (out_a / name).read_bytes(), (out_b / name).read_bytes(), f"{name} not byte-identical"
            )

    def test_artifacts_use_lf_bytes_on_every_host(self):
        built = _build_mini(self.root)
        out = self.root / "out_lf"
        write_pair_manifest(out, built)
        for path in out.iterdir():
            self.assertNotIn(b"\r\n", path.read_bytes(), f"{path.name} must use canonical LF")

    def test_input_paths_do_not_change_manifest_bytes(self):
        original = _build_mini(self.root)
        relocated = self.root / "relocated"
        relocated.mkdir()
        shutil.copyfile(self.root / "query.jsonl", relocated / "query.jsonl")
        shutil.copyfile(self.root / "gallery.jsonl", relocated / "gallery.jsonl")
        rebuilt = build_pair_manifest(
            relocated / "query.jsonl", relocated / "gallery.jsonl",
            expected_n_query=len(QUERIES),
            expected_n_unique_text=len(TEXTS),
            expected_n_gallery=len(CAD_IDS),
        )
        a = write_pair_manifest(self.root / "out_a", original)
        b = write_pair_manifest(self.root / "out_b", rebuilt)
        self.assertEqual(a["summary"]["pair_manifest_sha256"],
                         b["summary"]["pair_manifest_sha256"])
        self.assertEqual((self.root / "out_a" / "pair_manifest.json").read_bytes(),
                         (self.root / "out_b" / "pair_manifest.json").read_bytes())

    def test_refuses_overwrite_and_checkout(self):
        built = _build_mini(self.root)
        out = self.root / "out"
        write_pair_manifest(out, built)
        with self.assertRaises(PairIndexError) as ctx:
            write_pair_manifest(out, built)
        self.assertEqual(ctx.exception.reason, "output_exists")
        with self.assertRaises(PairIndexError) as ctx:
            write_pair_manifest(self.root / "repo" / "sub", built, repo_root=self.root / "repo")
        self.assertEqual(ctx.exception.reason, "output_inside_checkout")

    def test_worker_artifacts_carry_no_uid_semantics(self):
        built = _build_mini(self.root)
        out = self.root / "out"
        write_pair_manifest(out, built)
        for line in (out / "texts.jsonl").read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            self.assertEqual(set(obj), {"text_index", "text", "text_sha256", "n_chars"})
            self.assertNotIn("/", str(obj["text_index"]))
        for line in (out / "cad_store.jsonl").read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            self.assertEqual(set(obj), {"cad_index", "member"})
        eval_map = json.loads((out / "evaluation_map.json").read_text(encoding="utf-8"))
        self.assertEqual(set(eval_map), {"note", "query_to_text", "cad_index_to_cad_id"})
        self.assertIn("never reads", eval_map["note"])
        # The evaluation map must NOT contain relevance: query_id maps to an
        # integer text_index only (never to a cad_id), and vice versa.
        self.assertTrue(all(isinstance(v, int) for v in eval_map["query_to_text"].values()))
        self.assertTrue(all(isinstance(v, str) for v in eval_map["cad_index_to_cad_id"]))

    def test_loaders_roundtrip_and_tamper_detection(self):
        built = _build_mini(self.root)
        out = self.root / "out"
        result = write_pair_manifest(out, built)
        texts = load_texts(out)
        members = load_cad_store(out)
        self.assertEqual(len(texts), len(TEXTS))
        self.assertEqual(len(members), len(CAD_IDS))
        manifest = load_pair_manifest(out)
        self.assertEqual(manifest["counts"]["n_pairs"], len(TEXTS) * len(CAD_IDS))
        # reviewed-fingerprint gate accepts the true hashes ...
        fingerprints = dict(result["summary"]["artifact_sha256"])
        fingerprints[PAIR_MANIFEST_FILE] = result["summary"]["pair_manifest_sha256"]
        load_pair_manifest(out, expected_sha256=fingerprints)
        # The scorer must not even open the evaluation-only UID linkage.
        (out / "evaluation_map.json").unlink()
        load_pair_manifest(out, expected_sha256=fingerprints)
        # ... and refuses drift
        bad = dict(fingerprints)
        bad["texts.jsonl"] = "0" * 64
        with self.assertRaises(PairIndexError) as ctx:
            load_pair_manifest(out, expected_sha256=bad)
        self.assertEqual(ctx.exception.reason, "asset_sha256_mismatch")
        # on-disk tamper is caught even without the reviewed config
        texts_path = out / "texts.jsonl"
        lines = texts_path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"alpha"', '"alpha!"')  # bytes drift, sha + integrity
        texts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(PairIndexError):
            load_pair_manifest(out)
        with self.assertRaises(PairIndexError):
            load_texts(out)

    def test_loader_contiguity_and_duplicates(self):
        built = _build_mini(self.root)
        out = self.root / "out"
        write_pair_manifest(out, built)
        store_path = out / "cad_store.jsonl"
        lines = store_path.read_text(encoding="utf-8").splitlines()
        dup = json.loads(lines[1])
        lines.append(json.dumps(dup))  # duplicate cad_index
        store_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(PairIndexError) as ctx:
            load_cad_store(out)
        self.assertEqual(ctx.exception.reason, "duplicate_id")


class PairSpaceTest(unittest.TestCase):
    def test_real_dimensions(self):
        space = PairSpace(EXPECTED_N_UNIQUE_TEXT, EXPECTED_N_GALLERY)
        self.assertEqual(space.n_pairs, EXPECTED_N_PAIRS)
        self.assertEqual(EXPECTED_N_PAIRS, 29_319_624)
        self.assertEqual(space.coords_of(0), (0, 0))
        self.assertEqual(space.coords_of(EXPECTED_N_PAIRS - 1), (3643, 8045))
        self.assertEqual(space.pair_index_of(3643, 8045), EXPECTED_N_PAIRS - 1)
        # Round-trip over a deterministic spread of pair indices.
        for p in (1, 8045, 8046, 8047, 29_319_623 // 2, 12_345_678):
            t, c = space.coords_of(p)
            self.assertEqual(space.pair_index_of(t, c), p)

    def test_range_checks(self):
        space = PairSpace(3, 4)
        for bad in (-1, 12, 10**9):
            with self.assertRaises(PairIndexError):
                space.coords_of(bad)
        with self.assertRaises(PairIndexError):
            space.pair_index_of(3, 0)
        with self.assertRaises(PairIndexError):
            PairSpace(0, 4)

    def test_shards_disjoint_complete_contiguous(self):
        space = PairSpace(11, 5)
        for n_shards in (1, 2, 3, 7, 11):
            covered: list[int] = []
            pair_ranges = []
            for shard in range(n_shards):
                t0, t1 = space.shard_text_range(n_shards, shard)
                self.assertLess(t0, t1)
                covered.extend(range(t0, t1))
                pair_ranges.append(space.shard_pair_range(n_shards, shard))
            self.assertEqual(covered, list(range(11)), f"n_shards={n_shards} leaves gaps")
            self.assertEqual(
                pair_ranges, [(r[0], r[1]) for r in sorted(pair_ranges)],
            )
            self.assertEqual(pair_ranges[0][0], 0)
            self.assertEqual(pair_ranges[-1][1], space.n_pairs)
            for (_, end), (start, _) in zip(pair_ranges, pair_ranges[1:]):
                self.assertEqual(end, start)  # contiguous, no overlap, no gap

    def test_shard_args_checked(self):
        space = PairSpace(4, 3)
        with self.assertRaises(PairIndexError):
            space.shard_text_range(0, 0)
        with self.assertRaises(PairIndexError):
            space.shard_text_range(2, 2)
        with self.assertRaises(PairIndexError):
            space.shard_text_range(5, 0)  # more shards than texts

    def test_chunks_cover_shard_exactly_once(self):
        space = PairSpace(11, 5)
        for texts_per_chunk in (1, 2, 3, 4):
            pairs: list[int] = []
            for _ordinal, t0, t1, pair_start in space.iter_chunks(3, 1, texts_per_chunk):
                self.assertEqual(pair_start, t0 * 5)
                chunk_pairs = list(space.chunk_pair_indices(t0, t1))
                self.assertEqual(chunk_pairs, list(range(t0 * 5, t1 * 5)))
                pairs.extend(chunk_pairs)
            lo, hi = space.shard_pair_range(3, 1)
            self.assertEqual(pairs, list(range(lo, hi)))
        with self.assertRaises(PairIndexError):
            list(space.iter_chunks(1, 0, 0))


class ChunkVerificationTest(unittest.TestCase):
    def _make_chunk(self, space: PairSpace, shard: int, t0: int, t1: int, binding: dict):
        header = {
            "record_type": "chunk_header",
            "shard_id": shard,
            "text_range": [t0, t1],
            "n_pairs": (t1 - t0) * space.n_cads,
            "n_scored": (t1 - t0) * space.n_cads,
            "n_rejected": 0,
            "pair_index_first": t0 * space.n_cads,
            "pair_index_last": t1 * space.n_cads - 1,
            **{f: binding.get(f) for f in CHUNK_BINDING_FIELDS},
        }
        records = [header]
        for pair_index in space.chunk_pair_indices(t0, t1):
            text_index, cad_index = space.coords_of(pair_index)
            records.append(
                {
                    "record_type": "pair",
                    "pair_index": pair_index,
                    "text_index": text_index,
                    "cad_index": cad_index,
                    "status": "scored",
                    "score_x": -1.0,
                    "score_y": -2.0,
                    "score_total": -3.0,
                    "mean_x": -1.0,
                    "mean_y": -2.0,
                    "mean_total": -1.5,
                    "n_valid_x": 1,
                    "n_valid_y": 1,
                    "n_valid_total": 2,
                    "valid_seq_len": 2,
                    "n_real_tokens": 2,
                }
            )
        return records

    def _binding(self) -> dict:
        return {f: f"v_{f}" for f in CHUNK_BINDING_FIELDS}

    def test_complete_chunk_accepted(self):
        space = PairSpace(4, 3)
        records = self._make_chunk(space, 0, 1, 3, self._binding())
        self.assertEqual(verify_chunk_records(records, space, 0, 1, 3, self._binding()), [])

    def test_every_corruption_refused(self):
        space = PairSpace(4, 3)
        binding = self._binding()
        good = self._make_chunk(space, 0, 1, 3, binding)

        def problems_for(mutate) -> list[str]:
            records = json.loads(json.dumps(good))  # deep copy
            mutate(records)
            return verify_chunk_records(records, space, 0, 1, 3, binding)

        self.assertTrue(problems_for(lambda r: r.pop()))  # missing record
        self.assertTrue(problems_for(lambda r: r.append(dict(r[-1]))))  # extra record
        self.assertTrue(problems_for(lambda r: r.__setitem__(1, r[2])))  # order break
        self.assertTrue(problems_for(lambda r: r[1].update(status="error")))  # non-final
        self.assertTrue(problems_for(lambda r: r[0].update(git_sha="tampered")))  # binding drift
        self.assertTrue(problems_for(lambda r: r[0].update(text_range=[1, 2])))
        self.assertTrue(problems_for(lambda r: r[0].update(shard_id=1)))
        self.assertTrue(problems_for(lambda r: r[1].update(cad_index=99)))
        self.assertTrue(problems_for(lambda r: r[1].update(score_total=None)))
        self.assertTrue(problems_for(lambda r: r[1].update(score_x=float("nan"))))
        self.assertTrue(problems_for(lambda r: r[1].update(status="rejected")))
        self.assertTrue(problems_for(lambda r: r[0].update(n_scored=0)))
        self.assertTrue(problems_for(lambda r: r.__setitem__(0, {"record_type": "pair"})))
        self.assertEqual(verify_chunk_records([], space, 0, 1, 3, binding), ["empty chunk file"])


class ZipAndScanTest(unittest.TestCase):
    """Synthetic cad_seq.zip: member cross-check + zero-forward length scan."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make_zip(self, members: list[str], corrupt_index: int | None = None) -> Path:
        import torch

        from idea01.text2cad_port.model import generate_attention_mask

        zip_path = self.root / "cad_seq.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for i, member in enumerate(members):
                rows = [[1, 0], [139, 139], [139, 266], [5, 0], [4, 0], [3, 0], [2, 0]]
                rows += [[17 + j, 0] for j in range(9)] + [[8, 0], [6, 0]] + [[1, 0]]
                n_real = len(rows)
                n_pad = 272 - n_real
                flags = [0] * 7 + [1] + list(range(1, 11)) + [0]
                indices = [0] * n_real
                rows += [[0, 0]] * n_pad
                flags += [11] * n_pad
                indices += [1] * n_pad
                cad_vec = torch.tensor(rows, dtype=torch.int32)
                sample = {
                    "vec": {
                        "cad_vec": cad_vec,
                        "flag_vec": torch.tensor(flags, dtype=torch.int32),
                        "index_vec": torch.tensor(indices, dtype=torch.int32),
                    },
                    "mask_cad_dict": {
                        "attn_mask": generate_attention_mask(271),
                        "key_padding_mask": cad_vec == 0,
                    },
                }
                if corrupt_index is not None and i == corrupt_index:
                    sample["vec"]["cad_vec"] = cad_vec.clone()
                    sample["vec"]["cad_vec"][5] = torch.tensor([0, 9], dtype=torch.int32)
                buffer = io.BytesIO()
                torch.save(sample, buffer)
                archive.writestr(member, buffer.getvalue())
            archive.writestr("cad_seq/0000/00000077/seq/00000077.pth", b"extra-train-member")
        return zip_path

    def test_member_crosscheck_and_scan(self):
        members = [f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth" for i in range(3)]
        zip_path = self._make_zip(members)
        report = verify_zip_members(zip_path, members, expected_sha256=sha256_file(zip_path))
        self.assertEqual(report["n_store_members"], 3)
        self.assertEqual(report["n_extra_pth_members"], 1)
        lengths = scan_cad_lengths(zip_path, members)
        self.assertEqual([r["n_real_tokens"] for r in lengths], [19, 19, 19])
        self.assertEqual([r["cad_index"] for r in lengths], [0, 1, 2])

    def test_missing_member_and_sha_drift_refused(self):
        members = ["cad_seq/0000/00000000/seq/00000000.pth"]
        zip_path = self._make_zip(members)
        with self.assertRaises(PairIndexError) as ctx:
            verify_zip_members(zip_path, members, expected_sha256="0" * 64)
        self.assertEqual(ctx.exception.reason, "asset_sha256_mismatch")
        with self.assertRaises(PairIndexError) as ctx:
            verify_zip_members(zip_path, members + ["cad_seq/0000/00000009/seq/00000009.pth"],
                               expected_sha256=sha256_file(zip_path))
        self.assertEqual(ctx.exception.reason, "zip_member_missing")

    def test_scan_refuses_corrupt_candidate(self):
        members = [f"cad_seq/0000/0000000{i}/seq/0000000{i}.pth" for i in range(2)]
        zip_path = self._make_zip(members, corrupt_index=1)
        with self.assertRaises(PairIndexError) as ctx:
            scan_cad_lengths(zip_path, members)
        self.assertEqual(ctx.exception.reason, "member_mismatch")
        self.assertIn("cad_index 1", str(ctx.exception))


class PinnedConstantTest(unittest.TestCase):
    def test_zip_sha_matches_committed_asset_config(self):
        config = json.loads(ASSET_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["cad_sequence"]["cad_seq.zip"], CAD_SEQ_ZIP_SHA256)

    def test_committed_expected_config_shape(self):
        config = json.loads(EXPECTED_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["counts"]["n_pairs"], EXPECTED_N_PAIRS)
        self.assertEqual(config["counts"]["n_unique_text"], EXPECTED_N_UNIQUE_TEXT)
        self.assertEqual(config["counts"]["n_gallery"], EXPECTED_N_GALLERY)
        self.assertEqual(config["counts"]["n_query"], EXPECTED_N_QUERY)
        for name in ("texts.jsonl", "cad_store.jsonl", "cad_length_index.jsonl"):
            self.assertEqual(len(config["artifact_sha256"][name]), 64)
        self.assertEqual(len(config["evaluation_artifact_sha256"]["evaluation_map.json"]), 64)
        self.assertEqual(len(config["pair_manifest_sha256"]), 64)


@unittest.skipUnless(
    (M0_DIR / "query.jsonl").is_file() and (M0_DIR / "gallery.jsonl").is_file()
    and CAD_SEQ_ZIP.is_file() and shutil.which("git"),
    "pinned local M0 manifests and cad_seq.zip required",
)
class RealAssetIntegrationTest(unittest.TestCase):
    """Rebuild from the frozen M0 manifests; artifacts must byte-match the
    committed reviewed fingerprints (non-mirror: the fingerprints were
    produced by an independent earlier run and committed)."""

    def test_rebuild_matches_committed_fingerprints(self):
        expected = json.loads(EXPECTED_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rebuilt"
            built = build_pair_manifest(M0_DIR / "query.jsonl", M0_DIR / "gallery.jsonl")
            members = [e.member for e in built["cad_entries"]]
            zip_report = verify_zip_members(CAD_SEQ_ZIP, members)
            self.assertEqual(zip_report["n_store_members"], EXPECTED_N_GALLERY)
            result = write_pair_manifest(out, built, zip_report=zip_report)
            got = result["summary"]["artifact_sha256"]
            for name in ("texts.jsonl", "cad_store.jsonl"):
                self.assertEqual(
                    got[name],
                    expected["artifact_sha256"][name],
                    f"{name} drifted from the reviewed fingerprint",
                )
            self.assertEqual(
                result["summary"]["evaluation_artifact_sha256"]["evaluation_map.json"],
                expected["evaluation_artifact_sha256"]["evaluation_map.json"],
            )
            self.assertEqual(built["pair_space"].n_pairs, EXPECTED_N_PAIRS)


if __name__ == "__main__":
    unittest.main()
