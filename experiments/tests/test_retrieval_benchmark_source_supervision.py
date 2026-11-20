"""Verify the two supervision sources use the registered UID splits."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_retrieval_benchmark_source_supervision import source_rows  # noqa: E402


class SourceSupervisionTests(unittest.TestCase):
    def test_train_and_validation_keep_native_levels_and_align_short_captions(self):
        train, validation, test, unmatched = [f"0000/{n:08}" for n in range(1, 5)]
        assignments = {train: "train", validation: "validation", unmatched: "train"}
        captions = {train: "a short / caption", test: "held out", unmatched: "short only"}
        rows = [
            {"uid": train, "abstract": " simple ", "beginner": " N/A ",
             "intermediate": "two  holes", "expert": "precise"},
            {"uid": validation, "abstract": "val", "beginner": "beginner", "intermediate": "", "expert": ""},
            {"uid": test},
            {"uid": unmatched, "abstract": "", "beginner": "nan", "intermediate": "", "expert": ""},
        ]
        outputs = list(source_rows(rows, assignments, captions))
        self.assertEqual([split for split, _ in outputs], ["train", "validation"])
        self.assertEqual(outputs[0][1]["native_texts"], {"l0": "simple", "l2": "two  holes", "l3": "precise"})
        self.assertEqual(outputs[0][1]["ct_deepcad_text"], "a short / caption")
        self.assertIsNone(outputs[1][1]["ct_deepcad_text"])
        self.assertEqual(outputs[0][1]["member"], f"cad_seq/{train}/seq/00000001.pth")

    def test_duplicate_source_uid_is_rejected_even_with_one_empty_row(self):
        uid = "0000/00000001"
        row = {"uid": uid, "abstract": "", "beginner": "", "intermediate": "", "expert": ""}
        with self.assertRaisesRegex(ValueError, "Duplicate native source annotation UID"):
            list(source_rows([row, dict(row, abstract="text")], {uid: "train"}, {}))
