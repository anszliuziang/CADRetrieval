"""CPU tests for the Text2CAD G2 production-batch bench protocol (M1c G2 repair).

Contract: experiments/plans/IDEA01_M1C_G2_PROTOCOL_REPAIR_2026-10-01.md.
Symptoms under repair (Codex G1 batch review 2026-10-01):

  * nominal batch 32 was actually 20 pairs (strata cells smaller than the
    tier); the new protocol must produce REAL full batches of adjacent CAD
    indices plus an honestly-modelled tail batch;
  * a global measurement window cannot prove per-cell concurrency; the new
    barrier + ETA require per-(tier, cell, rep) windows with >=80% overlap of
    the shorter window;
  * the pooled two-card rate ignored the fixed text shards; the new ETA takes
    the SLOWER card's completion time over exact shard x cell weights;
  * char-length text strata may not represent real BERT token lengths; the
    new selection stratifies by audited tokenizer lengths;
  * --allow-large-batch silently appended unrequested 16/32 tiers.

All tests run on CPU with synthetic fixtures -- no real weights, no NPU,
no qrels.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.cadmium_runner import RunnerError  # noqa: E402
from idea01.text2cad_pair_index import (  # noqa: E402
    PairSpace,
    sha256_file,
    sha256_text,
)
from idea01.text2cad_speed_bench import (  # noqa: E402
    ETA_SAFETY_FACTOR,
    FULL_MATRIX_PAIRS,
    SEVEN_DAY_SECONDS,
    generate_selection,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures (deterministic; token lengths crafted to DIFFER from the
# n_chars ranking so tests prove the BERT-token stratification is used)
# ---------------------------------------------------------------------------


def _text_records(n: int, char_lens: list[int] | None = None) -> list[dict]:
    out = []
    for i in range(n):
        n_chars = char_lens[i] if char_lens else (i * 37) % 180 + 10
        text = f"t{i} " + "x" * (n_chars - len(f"t{i} "))
        out.append(
            {
                "text_index": i,
                "text": text,
                "text_sha256": sha256_text(text),
                "n_chars": len(text),
            }
        )
    return out


def _cad_length_records(n: int, real_lens: list[int] | None = None) -> list[dict]:
    out = []
    for i in range(n):
        n_real = real_lens[i] if real_lens else (i * 13) % 190 + 19
        out.append(
            {
                "cad_index": i,
                "n_real_tokens": n_real,
                "n_valid_x": n_real - 1,
                "n_valid_y": 2 + (i % 5),
            }
        )
    return out


def _token_lengths(text_records: list[dict], lens: list[int] | None = None) -> dict[int, int]:
    if lens is not None:
        return {rec["text_index"]: lens[rec["text_index"]] for rec in text_records}
    return {rec["text_index"]: (rec["text_index"] * 11) % 90 + 5 for rec in text_records}


#: The 5-text / 37-CAD fixture from the contract's regression test.
TEXT_CHARS_5 = [10, 50, 20, 90, 30]  # n_chars rank: 0,2,4,1,3
TOKEN_LENS_5 = [5, 12, 8, 40, 20]  # token rank:    0,2,1,4,3 (different!)
CAD_REALS_37 = [(i * 13) % 190 + 19 for i in range(37)]


def _fixture_5x37():
    texts = _text_records(5, TEXT_CHARS_5)
    cads = _cad_length_records(37, CAD_REALS_37)
    tokens = _token_lengths(texts, TOKEN_LENS_5)
    return texts, cads, tokens


# ---------------------------------------------------------------------------
# Task 1: production batch construction + exact shard/cell weights
# ---------------------------------------------------------------------------


class G2SelectionStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto
        self.texts, self.cads, self.tokens = _fixture_5x37()
        self.sel = proto.build_g2_selection(self.texts, self.cads, self.tokens)

    def test_schema_and_pair_space(self):
        self.assertEqual(self.sel["schema_version"], self.proto.G2_SELECTION_SCHEMA_VERSION)
        self.assertEqual(self.sel["counts"]["n_texts"], 5)
        self.assertEqual(self.sel["counts"]["n_cads"], 37)
        self.assertEqual(self.sel["counts"]["n_pairs"], 5 * 37)
        self.assertEqual(
            self.sel["rules"]["text_strata"],
            "rank by (n_bert_tokens, text_index) into 3 terciles",
        )

    def test_text_shards_match_pair_space_disjoint_complete(self):
        space = PairSpace(5, 37)
        expected = {s: space.shard_text_range(2, s) for s in range(2)}
        self.assertEqual(expected[0], (0, 2))  # round(2.5) = 2 (banker-free pin)
        self.assertEqual(expected[1], (2, 5))
        shards = self.sel["text_shards"]
        self.assertEqual([s["shard_id"] for s in shards], [0, 1])
        covered: list[int] = []
        for s, (t0, t1) in expected.items():
            self.assertEqual(shards[s]["text_range"], [t0, t1])
            covered.extend(range(t0, t1))
        self.assertEqual(sorted(covered), list(range(5)))  # disjoint + complete

    def test_batch32_real_full_and_tail_batches(self):
        tier = self.sel["tiers"]["32"]
        windows = tier["windows"]
        self.assertEqual(windows["n_full"], 1)
        self.assertEqual(windows["tail_size"], 5)
        # the full window is a REAL adjacent slice of 32 distinct CADs
        full = windows["full_layers"]
        present = [layer for layer in full if layer["n_windows"]]
        self.assertEqual(len(present), 1)
        layer = present[0]
        self.assertEqual(layer["layer"], "d0")
        self.assertEqual(layer["batch_size"], 32)
        # sample windows carry the actual adjacent cad ranges
        for cell in tier["cells"].values():
            for w in cell["samples"]["windows"]:
                if cell["layer"] == "tail":
                    self.assertEqual((w["cad_start"], w["cad_end"]), (32, 37))
                else:
                    self.assertEqual((w["cad_start"], w["cad_end"]), (0, 32))
                self.assertEqual(w["cad_end"] - w["cad_start"], cell["batch_size"])

    def test_pair_counts_exact_per_tier_and_tail_honest(self):
        n_pairs = 5 * 37
        for tier in ("1", "2", "4", "8", "16", "32"):
            cells = self.sel["tiers"][tier]["cells"]
            total = sum(c["n_pairs"] for c in cells.values())
            self.assertEqual(total, n_pairs, tier)
            self.assertEqual(
                sum(c["n_batches"] for c in cells.values()),
                sum(c["n_texts"] * c["n_windows"] for c in cells.values()),
                tier,
            )
        # tail windows never drop or duplicate CADs: union of all windows is
        # exactly range(37), disjoint
        for tier in (1, 2, 4, 8, 16, 32):
            windows = self.proto.production_cad_windows(37, tier)
            seen: list[int] = []
            for start, end in windows:
                self.assertEqual(end - start <= tier, True)
                seen.extend(range(start, end))
            self.assertEqual(sorted(seen), list(range(37)))
            self.assertEqual(len(seen), 37)  # no duplicate padding CADs

    def test_token_stratification_uses_bert_tokens_not_chars(self):
        # token rank order of the 5 texts is [0,2,1,4,3]; terciles by rank:
        # rank0->0, rank1->0, rank2->1, rank3->1, rank4->2
        expected_membership = {0: 0, 2: 0, 1: 1, 4: 1, 3: 2}
        for tier in self.sel["tiers"].values():
            for key, cell in tier["cells"].items():
                for t in cell["samples"]["texts"]:
                    self.assertEqual(expected_membership[t], cell["text_tier"], key)
        # shard x tier composition (shard0={0,1}, shard1={2,3,4})
        comp = {
            (c["shard_id"], c["text_tier"]): c["n_texts"]
            for c in self.sel["tiers"]["1"]["cells"].values()
        }
        self.assertEqual(comp[(0, 0)], 1)  # {0}
        self.assertEqual(comp[(0, 1)], 1)  # {1}
        self.assertEqual(comp[(1, 0)], 1)  # {2}
        self.assertEqual(comp[(1, 1)], 1)  # {4}
        self.assertEqual(comp[(1, 2)], 1)  # {3}
        self.assertNotIn((0, 2), comp)  # empty shard x tier cell omitted

    def test_cell_weights_hand_computed_tier32(self):
        cells = self.sel["tiers"]["32"]["cells"]
        # 5 non-empty shard x tier combos x 2 layers (d0 full + tail)
        self.assertEqual(len(cells), 10)
        for key, cell in cells.items():
            self.assertEqual(cell["n_texts"], 1)
            if cell["layer"] == "d0":
                self.assertEqual(cell["n_windows"], 1)
                self.assertEqual(cell["batch_size"], 32)
                self.assertEqual(cell["n_batches"], 1)
                self.assertEqual(cell["n_pairs"], 32)
            else:
                self.assertEqual(cell["layer"], "tail")
                self.assertEqual(cell["batch_size"], 5)
                self.assertEqual(cell["n_pairs"], 5)
            # batch max CAD length endpoints recorded for the layer
            self.assertGreaterEqual(cell["batch_max_len_max"], cell["batch_max_len_min"])

    def test_sample_endpoint_shortfalls_recorded(self):
        cells = self.sel["tiers"]["32"]["cells"]
        # every fixture cell has exactly one text and the d0 layer one window
        for cell in cells.values():
            self.assertEqual(len(cell["samples"]["texts"]), 1)
            shortfall = cell["samples"]["shortfall"]
            self.assertTrue(any("text" in s for s in shortfall))
            if cell["layer"] in ("d0", "tail"):
                self.assertTrue(any("window" in s for s in shortfall))

    def test_spot_and_ten_pair_identical_to_legacy_selection(self):
        legacy = generate_selection(
            self.texts, self.cads,
            texts_sha256="a", cad_length_sha256="b", pair_manifest_sha256="c",
        )
        self.assertEqual(self.sel["spot"], legacy["spot"])
        self.assertEqual(self.sel["ten_pair"], legacy["ten_pair"])

    def test_profile64_grid_uses_token_and_cad_quantiles(self):
        profile = self.sel["profile64"]
        self.assertEqual(len(profile), 64)
        quantiles = self.proto.PROFILE_QUANTILES
        token_order = sorted(range(5), key=lambda i: (TOKEN_LENS_5[i], i))
        cad_order = sorted(range(37), key=lambda i: (CAD_REALS_37[i], i))
        for k, entry in enumerate(profile):
            tq = quantiles[k // 8]
            cq = quantiles[k % 8]
            self.assertEqual(entry["text_quantile"], tq)
            self.assertEqual(entry["cad_quantile"], cq)
            self.assertEqual(entry["text_index"], token_order[round(4 * tq / 100)])
            self.assertEqual(entry["cad_index"], cad_order[round(36 * cq / 100)])
            self.assertEqual(entry["n_bert_tokens"], TOKEN_LENS_5[entry["text_index"]])
            self.assertEqual(entry["n_real_tokens"], CAD_REALS_37[entry["cad_index"]])
        # q100 picks the REAL longest text and longest CAD
        last = profile[-1]
        self.assertEqual(last["text_index"], 3)  # 40 tokens is the max
        self.assertEqual(last["cad_index"], max(range(37), key=lambda i: (CAD_REALS_37[i], i)))

    def test_rejects_bad_token_length_inputs(self):
        texts, cads, tokens = _fixture_5x37()
        with self.assertRaisesRegex(RunnerError, "token length"):
            self.proto.build_g2_selection(texts, cads, {0: 5, 1: 12})  # incomplete
        bad = dict(tokens)
        bad[0] = 0
        with self.assertRaisesRegex(RunnerError, "token length"):
            self.proto.build_g2_selection(texts, cads, bad)
        bad = dict(tokens)
        bad[2] = 4.5
        with self.assertRaisesRegex(RunnerError, "token length"):
            self.proto.build_g2_selection(texts, cads, bad)

    def test_rejects_non_adjacent_or_wrong_tier_requests(self):
        texts, cads, tokens = _fixture_5x37()
        with self.assertRaisesRegex(RunnerError, "batch tier"):
            self.proto.build_g2_selection(texts, cads, tokens, batch_tiers=(1, 3))
        with self.assertRaisesRegex(RunnerError, "texts_per_chunk"):
            self.proto.build_g2_selection(texts, cads, tokens, texts_per_chunk=2)
        with self.assertRaisesRegex(RunnerError, "n_shards"):
            self.proto.build_g2_selection(texts, cads, tokens, n_shards=3)


class G2SelectionVerifyTest(unittest.TestCase):
    """verify_g2_selection re-derives every weight from live facts."""

    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto
        self.texts, self.cads, self.tokens = _fixture_5x37()
        self.sel = proto.build_g2_selection(self.texts, self.cads, self.tokens)

    def test_accepts_self_consistent_selection(self):
        self.proto.verify_g2_selection(self.sel, self.texts, self.cads, self.tokens)

    def test_refuses_schema_drift_and_old_schema_not_interchangeable(self):
        from idea01.text2cad_speed_bench import verify_selection_against_manifest

        bad = json.loads(json.dumps(self.sel))
        bad["schema_version"] = "text2cad-g1g2-selection-v1"
        with self.assertRaisesRegex(RunnerError, "schema"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)
        # and the legacy verifier rejects the NEW schema (no cross-acceptance
        # for G2 ETA purposes)
        with self.assertRaisesRegex(RunnerError, "schema"):
            verify_selection_against_manifest(self.sel, Path("/nonexistent"), PairSpace(5, 37))

    def test_refuses_weight_drift(self):
        bad = json.loads(json.dumps(self.sel))
        cell = bad["tiers"]["32"]["cells"]["shard0_tt0_d0"]
        cell["n_batches"] += 1
        with self.assertRaisesRegex(RunnerError, "n_batches"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)

    def test_refuses_token_length_drift(self):
        bad = json.loads(json.dumps(self.sel))
        bad["text_token_lengths"][0] += 1
        with self.assertRaisesRegex(RunnerError, "token"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)

    def test_refuses_profile_drift(self):
        bad = json.loads(json.dumps(self.sel))
        bad["profile64"][0]["cad_index"] += 1
        with self.assertRaisesRegex(RunnerError, "profile64"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)

    # -- R2 fix 5: strict key sets (top-level, nested, no injected inputs) ---

    def test_r2_refuses_injected_top_level_inputs(self):
        for key in ("qrels", "evaluation_map", "captions"):
            bad = json.loads(json.dumps(self.sel))
            bad[key] = {}
            with self.subTest(key=key):
                with self.assertRaisesRegex(RunnerError, key):
                    self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)

    def test_r2_refuses_injected_nested_keys(self):
        bad = json.loads(json.dumps(self.sel))
        bad["tiers"]["32"]["cells"]["shard0_tt0_d0"]["evaluation_map"] = {"x": 1}
        with self.assertRaisesRegex(RunnerError, "evaluation_map"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)
        bad = json.loads(json.dumps(self.sel))
        bad["profile64"][0]["qrels"] = [1]
        with self.assertRaisesRegex(RunnerError, "qrels"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)

    def test_r2_missing_nested_key_still_refused(self):
        bad = json.loads(json.dumps(self.sel))
        del bad["tiers"]["32"]["cells"]["shard0_tt0_d0"]["batch_size"]
        with self.assertRaisesRegex(RunnerError, "batch_size"):
            self.proto.verify_g2_selection(bad, self.texts, self.cads, self.tokens)


class G2ProductionSizeTest(unittest.TestCase):
    """The real matrix shape: 3,644 texts x 8,046 CADs, exact pair accounting."""

    def test_full_size_weights_sum_exactly(self):
        from idea01 import text2cad_bench_protocol as proto

        texts = _text_records(3644)
        cads = _cad_length_records(8046)
        tokens = _token_lengths(texts)
        sel = proto.build_g2_selection(texts, cads, tokens)
        self.assertEqual(sel["counts"]["n_pairs"], FULL_MATRIX_PAIRS)
        for tier in ("1", "2", "4", "8", "16", "32"):
            cells = sel["tiers"][tier]["cells"]
            self.assertEqual(sum(c["n_pairs"] for c in cells.values()), FULL_MATRIX_PAIRS, tier)
            # production: every shard x tercile x layer cell is non-empty and
            # fully sampled (2 endpoint texts, 2 endpoint windows for full layers)
            for key, cell in cells.items():
                self.assertEqual(len(cell["samples"]["texts"]), 2, key)
                if cell["layer"] == "tail":
                    self.assertEqual(len(cell["samples"]["windows"]), 1, key)
                else:
                    self.assertEqual(len(cell["samples"]["windows"]), 2, key)
                    self.assertEqual(cell["samples"]["shortfall"], [], key)
        # tier 1 and 2 divide 8046 exactly -> no tail layer at all
        for tier in ("1", "2"):
            self.assertFalse(
                any(c["layer"] == "tail" for c in sel["tiers"][tier]["cells"].values())
            )


class G2ProductionSizeTest(unittest.TestCase):
    """The real matrix shape: 3,644 texts x 8,046 CADs, exact pair accounting."""

    def test_full_size_weights_sum_exactly(self):
        from idea01 import text2cad_bench_protocol as proto

        texts = _text_records(3644)
        cads = _cad_length_records(8046)
        tokens = _token_lengths(texts)
        sel = proto.build_g2_selection(texts, cads, tokens)
        self.assertEqual(sel["counts"]["n_pairs"], FULL_MATRIX_PAIRS)
        for tier in ("1", "2", "4", "8", "16", "32"):
            cells = sel["tiers"][tier]["cells"]
            self.assertEqual(sum(c["n_pairs"] for c in cells.values()), FULL_MATRIX_PAIRS, tier)
            # production: every shard x tercile x layer cell is non-empty and
            # fully sampled (2 endpoint texts, 2 endpoint windows for full layers)
            for key, cell in cells.items():
                self.assertEqual(len(cell["samples"]["texts"]), 2, key)
                if cell["layer"] == "tail":
                    self.assertEqual(len(cell["samples"]["windows"]), 1, key)
                else:
                    self.assertEqual(len(cell["samples"]["windows"]), 2, key)
                    self.assertEqual(cell["samples"]["shortfall"], [], key)
        # tier 1 and 2 divide 8046 exactly -> no tail layer at all
        for tier in ("1", "2"):
            self.assertFalse(
                any(c["layer"] == "tail" for c in sel["tiers"][tier]["cells"].values())
            )


# ---------------------------------------------------------------------------
# Task 1 CLI: token audit + select-g2 (pure CPU; fake tokenizer only)
# ---------------------------------------------------------------------------


class TokenAuditFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto
        self.texts = _text_records(4)
        self.counter = lambda text: len(text.split()) + 2  # word count + [CLS]/[SEP]
        self.bert_shas = {name: f"{i:064x}" for i, name in enumerate(
            ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
        )}

    def _audit(self):
        return self.proto.build_text_token_audit(
            self.texts, self.counter,
            texts_jsonl_sha256="a" * 64, bert_file_sha256=self.bert_shas,
        )

    def test_build_and_validate_roundtrip(self):
        audit = self._audit()
        self.assertEqual(audit["schema_version"], self.proto.TOKEN_AUDIT_SCHEMA_VERSION)
        self.assertFalse(audit["provenance"]["model_loaded"])
        self.assertEqual(len(audit["lengths"]), 4)
        expected = [len(r["text"].split()) + 2 for r in self.texts]
        self.assertEqual([e["n_bert_tokens"] for e in audit["lengths"]], expected)
        out = self.proto.token_lengths_from_audit(
            audit, expected_texts_sha256="a" * 64,
            expected_bert_file_sha256=self.bert_shas, n_texts=4,
        )
        self.assertEqual(out, {i: expected[i] for i in range(4)})

    def test_over_max_seq_len_refused(self):
        with self.assertRaisesRegex(RunnerError, "512"):
            self.proto.build_text_token_audit(
                self.texts, lambda text: 600,
                texts_jsonl_sha256="a" * 64, bert_file_sha256=self.bert_shas,
            )

    def test_binding_drift_refused(self):
        audit = self._audit()
        with self.assertRaisesRegex(RunnerError, "texts"):
            self.proto.token_lengths_from_audit(
                audit, expected_texts_sha256="b" * 64,
                expected_bert_file_sha256=self.bert_shas, n_texts=4,
            )
        with self.assertRaisesRegex(RunnerError, "BERT"):
            self.proto.token_lengths_from_audit(
                audit, expected_texts_sha256="a" * 64,
                expected_bert_file_sha256={"config.json": "0" * 64}, n_texts=4,
            )
        bad = json.loads(json.dumps(audit))
        bad["lengths"] = bad["lengths"][:3]
        with self.assertRaisesRegex(RunnerError, "covers"):
            self.proto.token_lengths_from_audit(
                bad, expected_texts_sha256="a" * 64,
                expected_bert_file_sha256=self.bert_shas, n_texts=4,
            )


class _FakeCliTokenizer:
    """Stands in for BertTokenizer; word count + special tokens, never truncates."""

    @classmethod
    def from_pretrained(cls, path, local_files_only=False):
        assert local_files_only is True
        return cls()

    def __call__(self, text):
        return {"attention_mask": [1] * (len(text.split()) + 2)}


class G2CliModesTest(unittest.TestCase):
    """audit-text-lengths and select-g2 through the real CLI parser."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.manifest_dir = self.root / "pair_manifest"
        self.manifest_dir.mkdir()
        self.out_dir = self.root / "out"
        self.out_dir.mkdir()
        self.texts, self.cads, self.tokens = _fixture_5x37()
        from idea01.text2cad_pair_index import CAD_LENGTH_FILE, PAIR_MANIFEST_FILE, TEXTS_FILE

        (self.manifest_dir / TEXTS_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.texts), encoding="utf-8"
        )
        (self.manifest_dir / CAD_LENGTH_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.cads), encoding="utf-8"
        )
        from idea01.text2cad_pair_index import sha256_file

        texts_sha = sha256_file(self.manifest_dir / TEXTS_FILE)
        cad_sha = sha256_file(self.manifest_dir / CAD_LENGTH_FILE)
        (self.manifest_dir / PAIR_MANIFEST_FILE).write_text(
            json.dumps({
                "schema_version": "text2cad-pair-manifest-v1",
                "counts": {"n_unique_text": 5, "n_gallery": 37, "n_pairs": 185},
                "artifacts": {
                    TEXTS_FILE: {"file": TEXTS_FILE, "sha256": texts_sha},
                    CAD_LENGTH_FILE: {"file": CAD_LENGTH_FILE, "sha256": cad_sha},
                },
            }),
            encoding="utf-8",
        )
        from idea01 import text2cad_bench_protocol as proto
        from idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256

        self.proto = proto
        self.bert_shas = dict(PINNED_BERT_FILE_SHA256)
        self.audit = proto.build_text_token_audit(
            self.texts, lambda t: TOKEN_LENS_5[self.texts.index(next(r for r in self.texts if r["text"] == t))],
            texts_jsonl_sha256=texts_sha, bert_file_sha256=self.bert_shas,
        )
        self.audit_path = self.root / "token_audit.json"
        self.audit_path.write_text(json.dumps(self.audit, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    def _run_main(self, argv):
        from idea01.text2cad_speed_bench import main

        return main(argv)

    def test_select_g2_happy_then_refuse_overwrite(self):
        out = self.out_dir / "g2_selection.json"
        rc = self._run_main([
            "--mode", "select-g2",
            "--pair-manifest-dir", str(self.manifest_dir),
            "--token-audit", str(self.audit_path),
            "--out", str(out),
        ])
        self.assertEqual(rc, 0)
        raw = out.read_bytes()
        self.assertNotIn(b"\r\n", raw)  # LF output
        config = json.loads(raw)
        self.assertEqual(config["schema_version"], self.proto.G2_SELECTION_SCHEMA_VERSION)
        self.assertEqual(len(config["profile64"]), 64)
        self.assertEqual(config["provenance"]["token_audit_sha256"], sha256_file(self.audit_path))
        self.proto.verify_g2_selection(config, self.texts, self.cads, self.tokens)
        # second run over the same --out is refused (exit 2, never overwrite)
        rc = self._run_main([
            "--mode", "select-g2",
            "--pair-manifest-dir", str(self.manifest_dir),
            "--token-audit", str(self.audit_path),
            "--out", str(out),
        ])
        self.assertEqual(rc, 2)

    def test_select_g2_refuses_foreign_audit(self):
        from types import SimpleNamespace

        from idea01.text2cad_speed_bench import run_select_g2

        bad = json.loads(json.dumps(self.audit))
        bad["provenance"]["texts_jsonl_sha256"] = "0" * 64
        self.audit_path.write_text(json.dumps(bad), encoding="utf-8")
        args = SimpleNamespace(
            pair_manifest_dir=str(self.manifest_dir),
            token_audit=str(self.audit_path),
            out=str(self.out_dir / "g2.json"),
        )
        with self.assertRaisesRegex(RunnerError, "texts"):
            run_select_g2(args)

    def test_audit_text_lengths_with_fake_tokenizer(self):
        import types
        from unittest import mock

        from idea01.text2cad_pair_index import TEXTS_FILE, sha256_file

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BertTokenizer = _FakeCliTokenizer
        bert_dir = self.root / "bert"
        bert_dir.mkdir()
        repo = self.root / "checkout"  # fake checkout; outputs must be OUTSIDE it
        repo.mkdir()
        out = self.out_dir / "token_lengths.json"
        argv = [
            "--mode", "audit-text-lengths",
            "--pair-manifest-dir", str(self.manifest_dir),
            "--bert-dir", str(bert_dir),
            "--out", str(out),
            "--repo-root", str(repo),
        ]
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}), mock.patch(
            "idea01.text2cad_speed_bench.verify_bert_asset_shas",
            return_value={n: {"sha256": s} for n, s in self.bert_shas.items()},
        ):
            rc = self._run_main(argv)
        self.assertEqual(rc, 0)
        audit = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(audit["schema_version"], self.proto.TOKEN_AUDIT_SCHEMA_VERSION)
        self.assertFalse(audit["provenance"]["model_loaded"])
        self.assertEqual(
            audit["provenance"]["texts_jsonl_sha256"],
            sha256_file(self.manifest_dir / TEXTS_FILE),
        )
        expected = [len(r["text"].split()) + 2 for r in self.texts]
        self.assertEqual([e["n_bert_tokens"] for e in audit["lengths"]], expected)

    def test_audit_text_lengths_refuses_existing_out_before_any_tokenizer(self):
        from types import SimpleNamespace

        from idea01.text2cad_speed_bench import run_audit_text_lengths

        out = self.out_dir / "token_lengths.json"
        out.write_text("{}", encoding="utf-8")
        args = SimpleNamespace(
            pair_manifest_dir=str(self.manifest_dir),
            bert_dir=str(self.root / "bert"),
            out=str(out),
            repo_root=str(self.root),
        )
        # transformers is NOT importable in this environment; reaching the
        # tokenizer before this refusal would raise ModuleNotFoundError
        with self.assertRaisesRegex(RunnerError, "overwrite"):
            run_audit_text_lengths(args)

    def test_audit_text_lengths_refuses_output_inside_checkout(self):
        from types import SimpleNamespace

        from idea01.text2cad_speed_bench import run_audit_text_lengths

        inside = self.root / "repo"
        (inside / "experiments").mkdir(parents=True)
        args = SimpleNamespace(
            pair_manifest_dir=str(self.manifest_dir),
            bert_dir=str(self.root / "bert"),
            out=str(inside / "audit.json"),
            repo_root=str(inside),
        )
        with self.assertRaisesRegex(RunnerError, "outside"):
            run_audit_text_lengths(args)


# ---------------------------------------------------------------------------
# Task 2: tier plans, real-batch driver inputs, sync barrier, rep windows
# ---------------------------------------------------------------------------


class TierPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto
        self.texts, self.cads, self.tokens = _fixture_5x37()
        self.sel = proto.build_g2_selection(self.texts, self.cads, self.tokens)

    def test_plans_carry_prebuilt_real_batches_in_canonical_order(self):
        plans = self.proto.build_tier_plans(self.sel, [1, 32], shard_id=0)
        self.assertEqual([p["batch_size"] for p in plans], [1, 32])
        tier32 = plans[1]
        # shard0 has tt0 and tt1 texts only; canonical order: (text_tier, layer)
        keys = [c["cell"] for c in tier32["cells"]]
        self.assertEqual(
            keys,
            ["shard0_tt0_d0", "shard0_tt0_tail", "shard0_tt1_d0", "shard0_tt1_tail"],
        )
        points = [c["point"] for c in tier32["cells"]]
        self.assertEqual(points, ["tt0_d0", "tt0_tail", "tt1_d0", "tt1_tail"])
        for cell_plan in tier32["cells"]:
            for batch in cell_plan["batches"]:
                self.assertEqual(len(batch), cell_plan["batch_size"])
                cad_indices = [c for _, c in batch]
                self.assertEqual(cad_indices, list(range(cad_indices[0], cad_indices[0] + len(batch))))
        # the tier-32 d0 batch is a REAL 32-pair adjacent slice (not 20, not padded)
        d0 = tier32["cells"][0]
        self.assertEqual(d0["batch_size"], 32)
        self.assertEqual(len(d0["batches"][0]), 32)
        # plan hash is deterministic and changes with the plan
        self.assertEqual(
            self.proto.tier_plans_sha256(plans), self.proto.tier_plans_sha256(plans)
        )
        other = self.proto.build_tier_plans(self.sel, [1], shard_id=0)
        self.assertNotEqual(self.proto.tier_plans_sha256(plans), self.proto.tier_plans_sha256(other))

    def test_budget_plan_counts_warmups_and_reps_from_same_enumeration(self):
        plans = self.proto.build_tier_plans(self.sel, [1, 32], shard_id=0)
        planned = self.proto.plan_production_forwards(plans, warmup_passes=2, reps=3)
        # independent hand count: R2 fix 7 warms up EVERY cell (2 passes over
        # its first two sample batches), then 3 timed reps over all batches
        total = 0
        for plan in plans:
            warmup_pairs = sum(
                len(b) for c in plan["cells"] for b in c["batches"][:2]
            )
            rep_pairs = sum(len(b) for c in plan["cells"] for b in c["batches"])
            total += 2 * warmup_pairs + 3 * rep_pairs
        self.assertEqual(planned, total)
        # 5x37 fixture: tier1 = 20 cells x (2p x 2b x 1) + 3 x 40 = 200;
        # tier32 = 2 x (32+5+32+5) + 3 x 74 = 370
        self.assertEqual(planned, 570)

    def test_unknown_tier_or_shard_refused(self):
        with self.assertRaisesRegex(RunnerError, "tier"):
            self.proto.build_tier_plans(self.sel, [3], shard_id=0)
        with self.assertRaisesRegex(RunnerError, "shard"):
            self.proto.build_tier_plans(self.sel, [1], shard_id=2)


class SyncBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.sync_dir = Path(self._tmp.name)
        self.binding = {"git_sha": "a" * 40, "plan_sha256": "b" * 64}

    def _barrier(self, device, **kwargs):
        params = dict(timeout_s=2.0, poll_interval_s=0.02)
        params.update(kwargs)
        return self.proto.SyncBarrier(
            self.sync_dir, "grp", device, self.binding, **params
        )

    def test_two_peers_exchange_payloads(self):
        import threading

        results: dict[str, Any] = {}
        errors: list[Exception] = []

        def arrive(device, payload):
            try:
                results[device] = self._barrier(device).arrive("t32_tt0_d0_r0", payload)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        ta = threading.Thread(target=arrive, args=("npu:0", {"card": 0}))
        tb = threading.Thread(target=arrive, args=("npu:1", {"card": 1}))
        ta.start(); tb.start(); ta.join(); tb.join()
        self.assertEqual(errors, [])
        self.assertEqual(results["npu:0"], {"card": 1})
        self.assertEqual(results["npu:1"], {"card": 0})

    def test_binding_mismatch_refused(self):
        import threading

        errors: list[Exception] = []

        def arrive(barrier):
            try:
                barrier.arrive("p", {})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        other = dict(self.binding)
        other["plan_sha256"] = "c" * 64
        ba = self._barrier("npu:0")
        bb = self.proto.SyncBarrier(
            self.sync_dir, "grp", "npu:1", other, timeout_s=2.0, poll_interval_s=0.02
        )
        ta = threading.Thread(target=arrive, args=(ba,))
        tb = threading.Thread(target=arrive, args=(bb,))
        ta.start(); tb.start(); ta.join(); tb.join()
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("binding" in str(e) for e in errors))

    def test_same_device_refused(self):
        import threading

        errors: list[Exception] = []

        def arrive(name):
            try:
                self._barrier("npu:0").arrive("p", {})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        ta = threading.Thread(target=arrive, args=("a",))
        tb = threading.Thread(target=arrive, args=("b",))
        ta.start(); tb.start(); ta.join(); tb.join()
        self.assertEqual(len(errors), 2)
        # two processes claiming the same device collide on the same marker
        # path (fail-closed "exists") or are refused as same-device
        self.assertTrue(any(("device" in str(e)) or ("exists" in str(e)) for e in errors))

    def test_stale_schema_marker_refused(self):
        # a foreign/old marker sitting at the sync point is never accepted
        stale = self.sync_dir / "grp__p__npu1.json"
        stale.write_text(json.dumps({"schema_version": "old", "device": "npu:1"}), encoding="utf-8")
        with self.assertRaisesRegex(RunnerError, "schema"):
            self._barrier("npu:0").arrive("p", {})

    def test_peer_failure_marker_stops_other_side(self):
        failed = self._barrier("npu:1")
        failed.signal_failure("t32_tt1_d0_r1", "oom", "simulated peer OOM")
        with self.assertRaisesRegex(RunnerError, "peer"):
            self._barrier("npu:0").arrive("t32_tt0_d0_r0", {})
        # failure markers are evidence: never deleted by an arrival
        self.assertTrue(list(self.sync_dir.glob("*.FAILED.json")))

    def test_timeout_refused_and_marker_preserved(self):
        with self.assertRaisesRegex(RunnerError, "timeout"):
            self.proto.SyncBarrier(
                self.sync_dir, "grp", "npu:0", self.binding,
                timeout_s=0.1, poll_interval_s=0.02,
            ).arrive("lonely", {})
        self.assertTrue((self.sync_dir / "grp__lonely__npu0.json").exists())

    def test_rerun_at_same_point_refused(self):
        import threading

        errors: list[Exception] = []

        def arrive(device):
            try:
                self._barrier(device).arrive("p", {})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        ta = threading.Thread(target=arrive, args=("npu:0",))
        tb = threading.Thread(target=arrive, args=("npu:1",))
        ta.start(); tb.start(); ta.join(); tb.join()
        self.assertEqual(errors, [])
        with self.assertRaisesRegex(RunnerError, "rerun|exists"):
            self._barrier("npu:0").arrive("p", {})

    def test_default_poll_interval_is_sub_skew_resolution(self):
        # Contract IDEA01_M1C_G2_B16_2026-10-02: the default poll interval is
        # 0.001 s so a realistic arrival skew (0.03 s) is resolved well inside
        # a short (0.10 s) measured rep window. The old 0.2 s default let one
        # card sleep through the whole window (CPU witness overlap = 0).
        barrier = self.proto.SyncBarrier(
            self.sync_dir, "grp", "npu:0", self.binding, timeout_s=2.0
        )
        self.assertEqual(barrier.poll_interval_s, 0.001)

    def test_default_poll_short_window_real_overlap(self):
        # Real-thread regression of the 2026-10-02 CPU witness geometry:
        # 0.03 s injected arrival skew, 0.10 s measured window, DEFAULT poll
        # interval (no override). The sync wait stays outside the window;
        # overlap of the two windows must pass the existing >=80% gate.
        import threading

        skew_s, window_s = 0.03, 0.10
        windows: dict[str, dict[str, float]] = {}
        errors: list[Exception] = []

        def run(device, delay):
            try:
                barrier = self.proto.SyncBarrier(
                    self.sync_dir, "grp", device, self.binding, timeout_s=10.0
                )
                if delay:
                    time.sleep(delay)
                barrier.arrive("t16_tt0_d0_r0", {})
                start = time.time()
                time.sleep(window_s)
                windows[device] = {"start_epoch": start, "end_epoch": time.time()}
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        ta = threading.Thread(target=run, args=("npu:0", 0.0))
        tb = threading.Thread(target=run, args=("npu:1", skew_s))
        ta.start(); tb.start(); ta.join(); tb.join()
        self.assertEqual(errors, [])
        self.assertEqual(set(windows), {"npu:0", "npu:1"})
        overlap = self.proto.window_overlap_fraction(windows["npu:0"], windows["npu:1"])
        self.assertGreaterEqual(overlap, self.proto.MIN_WINDOW_OVERLAP_FRACTION)


class RepWindowOverlapTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto

    def test_overlap_fraction_math(self):
        wa = {"start_epoch": 0.0, "end_epoch": 100.0}
        wb = {"start_epoch": 90.0, "end_epoch": 110.0}
        # overlap 10, shorter 20 -> 0.5
        self.assertAlmostEqual(self.proto.window_overlap_fraction(wa, wb), 0.5)
        wc = {"start_epoch": 0.0, "end_epoch": 100.0}
        self.assertAlmostEqual(self.proto.window_overlap_fraction(wa, wc), 1.0)

    def test_concurrent_windows_accepted(self):
        a = [{"point": "tt0_d0", "window": {"start_epoch": 0.0, "end_epoch": 10.0}}]
        b = [{"point": "tt0_d0", "window": {"start_epoch": 1.0, "end_epoch": 10.5}}]
        self.proto.verify_concurrent_rep_windows(a, b)  # 9/9.5 overlap >= 80%

    def test_global_overlap_but_staggered_reps_refused(self):
        # symptom: the OLD global windows overlap fully (both span [0, 100])
        # while the actual per-rep windows never do -- the new verifier must
        # refuse what the legacy combine accepted
        from idea01.text2cad_speed_bench import combine_two_card_results

        legacy_a = {
            "schema_version": "text2cad-bench-result-v1",
            "measurement_window": {"start_epoch": 0.0, "end_epoch": 100.0},
            "cell_e2e_pairs_per_s": {"c": 10.0},
        }
        legacy_b = {
            "schema_version": "text2cad-bench-result-v1",
            "measurement_window": {"start_epoch": 0.0, "end_epoch": 100.0},
            "cell_e2e_pairs_per_s": {"c": 10.0},
        }
        combine_two_card_results(legacy_a, legacy_b)  # legacy accepted this
        a = [{"point": "tt0_d0", "window": {"start_epoch": 0.0, "end_epoch": 10.0}}]
        b = [{"point": "tt0_d0", "window": {"start_epoch": 90.0, "end_epoch": 100.0}}]
        with self.assertRaisesRegex(RunnerError, "overlap"):
            self.proto.verify_concurrent_rep_windows(a, b)

    def test_reversed_missing_nonfinite_refused(self):
        good = [{"point": "p", "window": {"start_epoch": 0.0, "end_epoch": 10.0}}]
        with self.assertRaisesRegex(RunnerError, "window"):
            self.proto.verify_concurrent_rep_windows(
                good, [{"point": "p", "window": {"start_epoch": 10.0, "end_epoch": 0.0}}]
            )
        with self.assertRaisesRegex(RunnerError, "missing"):
            self.proto.verify_concurrent_rep_windows(good, [])
        with self.assertRaisesRegex(RunnerError, "window"):
            self.proto.verify_concurrent_rep_windows(
                good, [{"point": "p", "window": {"start_epoch": 0.0, "end_epoch": float("nan")}}]
            )
        with self.assertRaisesRegex(RunnerError, "point"):
            self.proto.verify_concurrent_rep_windows(
                good, [{"point": "other", "window": {"start_epoch": 0.0, "end_epoch": 10.0}}]
            )


# ---------------------------------------------------------------------------
# Task 3: sharded slower-card ETA
# ---------------------------------------------------------------------------


class ShardedETATest(unittest.TestCase):
    """estimate_sharded_eta over the production-size (3,644 x 8,046) shape."""

    @classmethod
    def setUpClass(cls) -> None:
        from idea01 import text2cad_bench_protocol as proto

        cls.proto = proto
        texts = _text_records(3644)
        cads = _cad_length_records(8046)
        tokens = _token_lengths(texts)
        cls.selection = proto.build_g2_selection(texts, cads, tokens)

    def _result(
        self,
        device: str,
        per_batch_s: float,
        *,
        cold: float = 10.0,
        reps: int = 3,
        invocation: str | None = None,
        window_shift: float = 0.0,
        drop_cell: str | None = None,
        drop_last_rep: bool = False,
        schema: str | None = None,
        git_sha: str = "a" * 40,
        sync_enabled: bool = True,
        sync_group: str = "g2run",
        common_plan_sha256: str | None = None,
        local_plan_sha256: str | None = None,
        peak_hbm: int | None = 1_000_000,
        env_overrides: dict | None = None,
        rep_indices: list[int] | None = None,
        tamper_batch_order: bool = False,
        extra_cell: bool = False,
        batch_tiers: list[int] | None = None,
    ) -> dict:
        """Fabricate an R2-format v2 throughput result consistent with the selection."""
        proto = self.proto
        shard = {"npu:0": 0, "npu:1": 1}[device]
        tiers = batch_tiers if batch_tiers is not None else [32]
        binding = proto.dual_shard_plan_binding(self.selection, tiers)
        tier_entry = self.selection["tiers"]["32"]
        cells: dict[str, Any] = {}
        for cell_key, cell in tier_entry["cells"].items():
            if cell["shard_id"] != shard or cell_key == drop_cell:
                continue
            sample_batches = proto.sample_batches_for_cell(cell)
            n_sample = len(sample_batches)
            indices = rep_indices if rep_indices is not None else list(range(reps))
            if drop_last_rep:
                indices = indices[:-1]
            rep_list = []
            for r in indices:
                e2e = per_batch_s * n_sample
                start = 1000.0 + window_shift + r * 100.0
                batch_records = []
                for k, batch in enumerate(sample_batches):
                    pairs = [[t, c] for t, c in batch]
                    if tamper_batch_order and k == 0 and len(pairs) >= 2:
                        pairs = [pairs[1], pairs[0], *pairs[2:]]
                    batch_records.append(
                        {"batch_ordinal": k, "size": len(batch), "pairs": pairs}
                    )
                rep_list.append({
                    "rep": r,
                    "point": f"tt{cell['text_tier']}_{cell['layer']}",
                    "window": {"start_epoch": start, "end_epoch": start + e2e},
                    "n_batches": n_sample,
                    "n_pairs": n_sample * cell["batch_size"],
                    "e2e_s": e2e,
                    "batches": batch_records,
                })
            cells[cell_key] = {
                "point": f"tt{cell['text_tier']}_{cell['layer']}",
                "shard_id": shard,
                "text_tier": cell["text_tier"],
                "layer": cell["layer"],
                "batch_size": cell["batch_size"],
                "is_tail": cell["layer"] == "tail",
                "reps": rep_list,
                "per_batch_e2e_s_max": per_batch_s,
            }
        if extra_cell:
            cells["shardX_foreign"] = {
                "point": "tt9_d9", "shard_id": shard, "text_tier": 9, "layer": "d9",
                "batch_size": 1, "is_tail": False, "reps": [], "per_batch_e2e_s_max": 1.0,
            }
        env = {
            "device": device,
            "git_sha": git_sha,
            "checkpoint_sha256": "c" * 64,
            "pair_manifest_sha256": "m" * 64,
            "torch_dtype": "float32",
            "trim_mode": "none",
            "text_cache": False,
            "score_name": "two_head_factorized_logprob_sum",
            "spec_version": "text2cad-vector-score-v1",
        }
        env.update(env_overrides or {})
        return {
            "schema_version": schema or "text2cad-bench-result-v2",
            "mode": "throughput",
            "invocation_id": invocation or f"inv-{device}",
            "cold_start_s": cold,
            "selection_sha256": "s" * 64,
            "environment": env,
            "batch_tiers": tiers,
            "sync": {
                "enabled": sync_enabled,
                "group": sync_group,
                "timeout_s": 180.0,
                "common_plan_sha256": common_plan_sha256 or binding["common_plan_sha256"],
                "local_plan_sha256": (
                    local_plan_sha256 or binding["local_plan_sha256_by_shard"][shard]
                ),
                "shard_id": shard,
            },
            "tiers": [{
                "batch_size": 32,
                "status": "ok",
                "cells": cells,
                "peak_device_memory_bytes": peak_hbm,
                "hbm_device": device,
            }],
        }

    def test_slower_card_is_the_wall_clock_not_pooled(self):
        # card0 at 40 pairs/s (tier32 -> 0.8 s/batch), card1 at 20 pairs/s
        r0 = self._result("npu:0", 0.8)
        r1 = self._result("npu:1", 1.6)
        est = self.proto.estimate_sharded_eta(
            self.selection, {"npu:0": r0, "npu:1": r1}, 32
        )
        # shard loads: 1,822 texts x 252 windows (251 full + 1 tail) each
        n_batches_per_card = 1822 * 252
        self.assertAlmostEqual(
            est["per_card"]["npu:0"]["compute_seconds"], n_batches_per_card * 0.8
        )
        self.assertAlmostEqual(
            est["per_card"]["npu:1"]["compute_seconds"], n_batches_per_card * 1.6
        )
        self.assertEqual(est["slower_card"], "npu:1")
        expected_wall = n_batches_per_card * 1.6 + 10.0
        self.assertAlmostEqual(est["wall_seconds_presafety"], expected_wall)
        self.assertAlmostEqual(est["eta_seconds"], expected_wall * 1.30)
        # NOT the pooled ideal-2x: that would be (FULL/60 + 10) * 1.3
        pooled = (FULL_MATRIX_PAIRS / 60.0 + 10.0) * 1.30
        self.assertGreater(est["eta_seconds"], pooled * 1.4)
        self.assertFalse(est["passes_seven_day_gate"])
        self.assertEqual(est["full_matrix_pairs"], FULL_MATRIX_PAIRS)

    def test_exact_cell_weights_not_averaged(self):
        # card1's TAIL cells are 1250x slower than its full cells; the ETA must
        # price every cell at its own measured cost
        def cost_for(cell_key):
            return 1000.0 if cell_key.endswith("_tail") else 0.8

        r0 = self._result("npu:0", 0.8)
        r1 = self._result("npu:1", 0.8)
        # rebuild card1 with per-cell costs
        tier_entry = self.selection["tiers"]["32"]
        for cell_key, cell_record in r1["tiers"][0]["cells"].items():
            cost = cost_for(cell_key)
            for rep in cell_record["reps"]:
                rep["e2e_s"] = cost * rep["n_batches"]
            cell_record["per_batch_e2e_s_max"] = cost
        est = self.proto.estimate_sharded_eta(
            self.selection, {"npu:0": r0, "npu:1": r1}, 32
        )
        expected = 0.0
        for cell_key, cell in tier_entry["cells"].items():
            if cell["shard_id"] != 1:
                continue
            expected += cell["n_batches"] * cost_for(cell_key)
        self.assertAlmostEqual(est["per_card"]["npu:1"]["compute_seconds"], expected)
        # an average-cost (pooled) estimate would underprice the tail
        n_batches = sum(
            c["n_batches"] for c in tier_entry["cells"].values() if c["shard_id"] == 1
        )
        naive_average = n_batches * 0.8
        self.assertGreater(est["per_card"]["npu:1"]["compute_seconds"], naive_average * 2)

    def test_overhead_rules(self):
        r0 = self._result("npu:0", 0.8, cold=12.0)
        r1 = self._result("npu:1", 0.8, cold=8.0)
        est = self.proto.estimate_sharded_eta(self.selection, {"npu:0": r0, "npu:1": r1}, 32)
        self.assertEqual(est["extra_overhead_seconds"], 0.0)  # cold starts kept
        est2 = self.proto.estimate_sharded_eta(
            self.selection, {"npu:0": r0, "npu:1": r1}, 32, overhead_seconds=120.0
        )
        self.assertAlmostEqual(est2["extra_overhead_seconds"], 120.0 - 20.0)
        self.assertAlmostEqual(
            est2["eta_seconds"], (est["wall_seconds_presafety"] + 100.0) * 1.30
        )
        for bad in (0.0, 19.9, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(RunnerError, "overhead"):
                self.proto.estimate_sharded_eta(
                    self.selection, {"npu:0": r0, "npu:1": r1}, 32, overhead_seconds=bad
                )

    def test_fail_closed_refusals(self):
        r0 = self._result("npu:0", 0.8)
        r1 = self._result("npu:1", 0.8)
        with self.assertRaisesRegex(RunnerError, "safety"):
            self.proto.estimate_sharded_eta(
                self.selection, {"npu:0": r0, "npu:1": r1}, 32, safety_factor=1.0
            )
        with self.assertRaisesRegex(RunnerError, "cell"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, drop_cell="shard1_tt0_d0")},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "tier"):
            self.proto.estimate_sharded_eta(self.selection, {"npu:0": r0, "npu:1": r1}, 16)
        dup = self._result("npu:1", 0.8, invocation="same")
        with self.assertRaisesRegex(RunnerError, "invocation"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": self._result("npu:0", 0.8, invocation="same"), "npu:1": dup},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "git_sha"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, git_sha="b" * 40)},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "schema"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, schema="text2cad-bench-result-v1")},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "overlap"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, window_shift=500.0)},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "rep"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, drop_last_rep=True)},
                32,
            )

    def test_nonfinite_and_wrong_profile_refused(self):
        r0 = self._result("npu:0", 0.8)
        bad = self._result("npu:1", 0.8)
        bad["tiers"][0]["cells"]["shard1_tt0_d0"]["reps"][0]["e2e_s"] = float("nan")
        with self.assertRaisesRegex(RunnerError, "finite"):
            self.proto.estimate_sharded_eta(
                self.selection, {"npu:0": r0, "npu:1": bad}, 32
            )
        bad2 = self._result("npu:1", 0.8)
        bad2["tiers"][0]["cells"]["shard1_tt0_d0"]["reps"][0]["n_batches"] += 1
        with self.assertRaisesRegex(RunnerError, "profile"):
            self.proto.estimate_sharded_eta(
                self.selection, {"npu:0": r0, "npu:1": bad2}, 32
            )

    def test_pair_sum_must_be_the_exact_full_matrix(self):
        sel = json.loads(json.dumps(self.selection))
        sel["tiers"]["32"]["cells"]["shard0_tt0_d0"]["n_pairs"] += 1
        r0 = self._result("npu:0", 0.8)
        r1 = self._result("npu:1", 0.8)
        with self.assertRaisesRegex(RunnerError, "29,319,624|full matrix"):
            self.proto.estimate_sharded_eta(sel, {"npu:0": r0, "npu:1": r1}, 32)

    # -- R2 fix 3/4 + supplement: complete two-card validation ---------------

    def test_r2_per_card_reference_config_and_identity(self):
        r0 = self._result("npu:0", 0.8)
        for overrides in (
            {"torch_dtype": "bfloat16"},
            {"trim_mode": "batch_max"},
            {"text_cache": True},
        ):
            r1 = self._result("npu:1", 0.8, env_overrides=overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(RunnerError, "reference"):
                    self.proto.estimate_sharded_eta(
                        self.selection, {"npu:0": r0, "npu:1": r1}, 32
                    )
        # card identity: each result filed under a device its env does not claim
        with self.assertRaisesRegex(RunnerError, "identity"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": self._result("npu:1", 0.8), "npu:1": self._result("npu:0", 0.8)},
                32,
            )

    def test_r2_plan_binding_checked(self):
        r0 = self._result("npu:0", 0.8)
        with self.assertRaisesRegex(RunnerError, "common plan"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, common_plan_sha256="0" * 64)},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "local plan"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, local_plan_sha256="0" * 64)},
                32,
            )

    def test_r2_sync_evidence_mandatory(self):
        r0 = self._result("npu:0", 0.8)
        with self.assertRaisesRegex(RunnerError, "sync"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, sync_enabled=False)},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "group"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, sync_group="other")},
                32,
            )
        missing = self._result("npu:1", 0.8)
        del missing["sync"]
        with self.assertRaisesRegex(RunnerError, "sync"):
            self.proto.estimate_sharded_eta(
                self.selection, {"npu:0": r0, "npu:1": missing}, 32
            )

    def test_r2_tier_and_cell_sets_exact(self):
        r0 = self._result("npu:0", 0.8)
        with self.assertRaisesRegex(RunnerError, "cell"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, extra_cell=True)},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "tier"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, batch_tiers=[16])},
                32,
            )
        dup = self._result("npu:1", 0.8)
        dup["tiers"] = dup["tiers"] + [json.loads(json.dumps(dup["tiers"][0]))]
        with self.assertRaisesRegex(RunnerError, "tier"):
            self.proto.estimate_sharded_eta(
                self.selection, {"npu:0": r0, "npu:1": dup}, 32
            )

    def test_r2_reps_unique_contiguous_at_least_three(self):
        r0 = self._result("npu:0", 0.8)
        with self.assertRaisesRegex(RunnerError, "rep"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, rep_indices=[0, 1, 1])},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "rep"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, rep_indices=[0, 2, 3])},
                32,
            )
        # BOTH cards missing the same rep is still incomplete evidence
        with self.assertRaisesRegex(RunnerError, "rep"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {
                    "npu:0": self._result("npu:0", 0.8, reps=2),
                    "npu:1": self._result("npu:1", 0.8, reps=2),
                },
                32,
            )

    def test_r2_batch_pairs_order_and_peak_hbm_checked(self):
        r0 = self._result("npu:0", 0.8)
        with self.assertRaisesRegex(RunnerError, "pair order|profile"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, tamper_batch_order=True)},
                32,
            )
        with self.assertRaisesRegex(RunnerError, "HBM|peak"):
            self.proto.estimate_sharded_eta(
                self.selection,
                {"npu:0": r0, "npu:1": self._result("npu:1", 0.8, peak_hbm=None)},
                32,
            )


# ---------------------------------------------------------------------------
# R2 fix 1 + supplement: dual-shard common plan binding; mandatory sync
# ---------------------------------------------------------------------------


def _balanced_6x37_fixture():
    """6 texts / 37 CADs where BOTH shards cover all three token terciles."""
    char_lens = [10, 50, 20, 90, 30, 60]
    token_lens = [5, 12, 20, 8, 15, 40]  # terciles: t0={0,3}, t1={1,4}, t2={2,5}
    cad_reals = [(i * 13) % 190 + 19 for i in range(37)]
    texts = []
    for i, n_chars in enumerate(char_lens):
        text = f"t{i} " + "x" * (n_chars - len(f"t{i} "))
        texts.append({"text_index": i, "text": text, "text_sha256": sha256_text(text),
                      "n_chars": len(text)})
    cads = [{"cad_index": i, "n_real_tokens": n, "n_valid_x": n - 1, "n_valid_y": 2}
            for i, n in enumerate(cad_reals)]
    tokens = {i: token_lens[i] for i in range(6)}
    return texts, cads, tokens


class DualShardPlanBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        from idea01 import text2cad_bench_protocol as proto

        self.proto = proto
        texts, cads, tokens = _balanced_6x37_fixture()
        self.selection = proto.build_g2_selection(texts, cads, tokens)

    def test_balanced_fixture_covers_all_terciles_both_shards(self):
        for shard in (0, 1):
            tiers_seen = {
                c["text_tier"]
                for c in self.selection["tiers"]["32"]["cells"].values()
                if c["shard_id"] == shard
            }
            self.assertEqual(tiers_seen, {0, 1, 2}, shard)

    def test_common_binding_fixed_shard_order_local_shas_differ(self):
        binding = self.proto.dual_shard_plan_binding(self.selection, [32])
        common = binding["common_plan_sha256"]
        local0 = binding["local_plan_sha256_by_shard"][0]
        local1 = binding["local_plan_sha256_by_shard"][1]
        self.assertNotEqual(common, local0)
        self.assertNotEqual(common, local1)
        self.assertNotEqual(local0, local1)  # the two shards really differ
        # fixed shard order: rebuilding must not change the common hash
        again = self.proto.dual_shard_plan_binding(self.selection, [32])
        self.assertEqual(common, again["common_plan_sha256"])
        # the common hash commits to BOTH shards: tampering either side moves it
        import copy

        for shard in (0, 1):
            tampered = copy.deepcopy(binding["plans_by_shard"])
            tampered[shard][0]["cells"][0]["batches"][0] = (
                tampered[shard][0]["cells"][0]["batches"][0][:31]
            )
            self.assertNotEqual(
                self.proto.combined_tier_plans_sha256(tampered[0], tampered[1]),
                common,
                shard,
            )


if __name__ == "__main__":
    unittest.main()
