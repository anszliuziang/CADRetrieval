"""Regression tests for the split-local SpectraBRep descriptor cache."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_retrieval_benchmark_spectrabrep as cache  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_member(uid):
    return f"cad_seq/{uid}/seq/{uid.split('/')[-1]}.pth"


class SpectraCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.upstream = self.root / "upstream"
        package = self.upstream / "src" / "cadret"
        package.mkdir(parents=True)
        (package / "config.py").write_text("DESCRIPTOR_DIM = 280\n", encoding="utf-8")
        (package / "geometry.py").write_text(
            """import numpy as np
from .config import DESCRIPTOR_DIM

def describe_sequence(cad_vec, index_vec):
    if cad_vec[0, 0] == 0:
        return {}
    return {"value": cad_vec[0, 0] + index_vec[1]}

def descriptor_vector(info):
    value = info.get("value", 0)
    return np.full(DESCRIPTOR_DIM, np.nan if value == 13 else value, dtype=np.float32)
""",
            encoding="utf-8",
        )
        lines = [
            f"{path.relative_to(self.upstream).as_posix()}\0{digest(path)}\n"
            for path in sorted(package.iterdir())
        ]
        tree_digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
        self.train_uids = [f"0000/{n:08d}" for n in (9, 2, 5, 4, 6, 3)]
        self.val_uids = [f"0001/{n:08d}" for n in (10002, 10001)]
        self.test_uid = "0002/00020000"
        self.splits = {"train": self.train_uids, "validation": self.val_uids, "test": [self.test_uid]}
        self.split = self.root / "split.json"
        save_json(self.split, self.splits)
        self.archive = self.root / "cad_seq.zip"
        with ZipFile(self.archive, "w") as archive:
            self.write_program(archive, self.train_uids[0], 1, 10)
            self.write_program(archive, self.train_uids[1], 2, 20)
            self.write_program(archive, self.train_uids[2], 0, 0)
            self.write_program(archive, self.train_uids[3], 3, 10)
            self.write_program(archive, self.train_uids[4], 1, 1, malformed=True)
            self.write_program(archive, self.val_uids[0], 5, 30)
            self.write_program(archive, self.val_uids[1], 1, 40)
            archive.writestr(fixture_member(self.test_uid), b"held out")
        self.supervision = self.root / "supervision"
        self.supervision.mkdir()
        train = [
            self.registration(self.train_uids[4], {"l0": "invalid program"}, "short"),
            self.registration(self.train_uids[0], {"l0": "part", "l3": "helpful"}, "very short"),
            self.registration(self.train_uids[2], {"l1": "empty geometry"}, None),
            self.registration(self.train_uids[3], {"l2": "nonfinite"}, "short"),
            self.registration(self.train_uids[5], {"l0": "missing member"}, "short"),
        ]
        val = [
            self.registration(self.val_uids[1], {"l1": "second"}, "CT validation short"),
            self.registration(self.val_uids[0], {"l0": "first", "l2": "more detail"}, None),
        ]
        frozen_hashes = {}
        stats = {}
        for fold, rows in (("train", train), ("validation", val)):
            path = self.supervision / f"{fold}.jsonl"
            save_rows(path, rows)
            frozen_hashes[path.name] = digest(path)
            stats[fold] = {
                "official_uids": len(self.splits[fold]),
                "source_ct_caption_uids": sum(row["ct_deepcad_text"] is not None for row in rows),
                "native_uids": len(rows),
                "native_four_level_uids": 0,
                "common_native_and_ct_uids": sum(row["ct_deepcad_text"] is not None for row in rows),
                "native_pairs_per_level": {
                    level: sum(level in row["native_texts"] for row in rows)
                    for level in ("l0", "l1", "l2", "l3")
                },
                "common_native_pairs_per_level": {
                    level: sum(level in row["native_texts"] and row["ct_deepcad_text"] is not None for row in rows)
                    for level in ("l0", "l1", "l2", "l3")
                },
            }
        replacement_pins = dict(cache.ASSETS)
        for name, path in (("text2cad_program_zip", self.archive), ("text2cad_split_json", self.split)):
            replacement_pins[name] = ("repo", path.name, digest(path))
        self.manifest = {
            "schema_version": 1,
            "files": frozen_hashes.copy(),
            "official_test_uids_excluded": 1,
            "splits": stats,
            "sources": {name: {"sha256": replacement_pins[name][2]} for name in (
                "text2cad_prompt_csv", "text2cad_split_json", "text2cad_program_zip", "cadtranslator_dataset_zip"
            )},
        }
        save_json(self.supervision / "manifest.json", self.manifest)
        context = patch.multiple(
            cache,
            AUTHOR_SOURCE_SHA256=tree_digest,
            FROZEN_SUPERVISION_SHA256=frozen_hashes,
            ASSETS=replacement_pins,
            EXPECTED_SOURCE_SPLIT={key: len(value) for key, value in self.splits.items()},
        )
        context.start()
        self.addCleanup(context.stop)
        self.output = self.root / "result"

    @staticmethod
    def registration(uid, native_texts, text):
        return {"uid": uid, "member": fixture_member(uid), "native_texts": native_texts, "ct_deepcad_text": text}

    @staticmethod
    def write_program(archive, uid, first, index, malformed=False):
        cad_vec = torch.zeros((272, 2), dtype=torch.long)
        cad_vec[0, 0] = first
        index_vec = torch.zeros(272, dtype=torch.long)
        index_vec[1] = index
        payload = {"vec": {"cad_vec": cad_vec}}
        if not malformed:
            payload["vec"]["index_vec"] = index_vec
        stream = io.BytesIO()
        torch.save(payload, stream)
        archive.writestr(fixture_member(uid), stream.getvalue())

    def build(self, **kwargs):
        return cache.build_cache(self.upstream, self.archive, self.split, self.supervision, self.output, **kwargs)

    def rows(self, fold, artifact="rows.jsonl"):
        return [json.loads(row) for row in (self.output / fold / artifact).read_text(encoding="utf-8").splitlines()]

    def test_official_order_with_valid_matrix_and_eligibility(self):
        result = self.build(workers=2)
        self.assertEqual(result["mode"], "full")
        train = self.rows("train")
        self.assertEqual([row["uid"] for row in train], self.train_uids)
        self.assertEqual([row["vector_index"] for row in train], [0, 1, None, None, None, None])
        self.assertEqual([row["status"] for row in train], [
            "ok", "ok", "empty_geometry", "nonfinite_descriptor", "invalid_program", "missing_program"
        ])
        self.assertEqual([row["uid"] for row in self.rows("validation")], self.val_uids)
        for fold, values in (("train", [11, 22]), ("validation", [35, 41])):
            vectors = np.load(self.output / fold / "vectors.npy", allow_pickle=False)
            self.assertEqual(vectors.dtype, np.float32)
            np.testing.assert_array_equal(vectors, np.repeat(np.array(values)[:, None], 280, axis=1))
        self.assertEqual(self.rows("train", "native_eligible.jsonl"), [self.train_uids[0]])
        self.assertEqual(self.rows("train", "matched_eligible.jsonl"), [self.train_uids[0]])
        self.assertEqual(self.rows("validation", "native_eligible.jsonl"), self.val_uids)
        self.assertEqual(self.rows("validation", "matched_eligible.jsonl"), self.val_uids[1:])
        self.assertFalse((self.output / "test").exists())
        self.assertNotIn("very short", (self.output / "train" / "rows.jsonl").read_text())
        self.assertEqual(result["files"]["train/vectors.npy"], digest(self.output / "train" / "vectors.npy"))

    def test_pilot_is_explicitly_limited_and_records_official_counts(self):
        manifest = self.build(workers=1, pilot_per_fold=1)
        self.assertEqual(manifest["mode"], "pilot")
        self.assertEqual(manifest["folds"]["train"]["official_uids"], 6)
        self.assertEqual([row["uid"] for row in self.rows("train")], self.train_uids[:1])
        self.assertEqual([row["uid"] for row in self.rows("validation")], self.val_uids[:1])

    def test_artifacts_are_never_published_with_a_tampered_author_file(self):
        (self.upstream / "src" / "cadret" / "geometry.py").write_text(
            "raise RuntimeError('unauthenticated code should not run')\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "author source"):
            self.build(workers=1)
        self.assertFalse(self.output.exists())

    def test_rejects_self_consistent_but_unfrozen_supervision(self):
        train_path = self.supervision / "train.jsonl"
        train_path.write_bytes(train_path.read_bytes() + b"\n")
        self.manifest["files"]["train.jsonl"] = digest(train_path)
        save_json(self.supervision / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "supervision"):
            self.build(workers=1)
        self.assertFalse(self.output.exists())

    def test_rejects_held_out_uid_even_under_a_matching_file_hash(self):
        train_path = self.supervision / "train.jsonl"
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        rows[0] = self.registration(self.test_uid, {"l0": "held out"}, "held out")
        save_rows(train_path, rows)
        self.manifest["files"]["train.jsonl"] = digest(train_path)
        save_json(self.supervision / "manifest.json", self.manifest)
        with patch.object(cache, "FROZEN_SUPERVISION_SHA256", self.manifest["files"]):
            with self.assertRaisesRegex(ValueError, "train"):
                self.build(workers=1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
