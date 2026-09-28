"""Unit tests for the CADmium native JSON scorer (M1a).

Covers (contract item 4): template prefix, end token + newline, shift, mask,
non-finite values, 4096 boundary, batch consistency, verbatim JSON target,
no qrels read. Uses a fake tokenizer + synthetic logits so it runs without
torch or weights. A real-tokenizer small CPU sample is included and skipped
when ``.cache/cadmium_model`` is absent.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.cadmium_scorer import (  # noqa: E402
    EXPECTED_SYSTEM_MESSAGE_SHA256,
    MAX_TRAINING_CONTEXT,
    REASON_BAD_TERMINATOR,
    REASON_EMPTY_TARGET,
    REASON_NONFINITE,
    REASON_OUT_OF_TRAINING_CONTEXT,
    REASON_PREFIX_MISMATCH,
    REASON_TOKENIZER_MISMATCH,
    ScoreLedger,
    ScoredPair,
    build_cadmium_sequence,
    fingerprint_tokenizer,
    load_system_message,
    score_batch_consistency,
    score_pair_from_logits as official_score_pair_from_logits,
    _score_pair_from_logits_unverified as score_pair_from_logits,
)
from idea01.scoring import ScoringError  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments" / "configs" / "cadmium_system_message.json"
CADMIUM_CACHE = REPO / ".cache" / "cadmium_model"


# ---------------------------------------------------------------------------
# Fake tokenizer mimicking Qwen2.5 special-token behavior
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Deterministic char/special tokenizer for unit tests.

    Small token ids keep synthetic logits and the 4096 boundary test cheap.
    The fake implements apply_chat_template itself; production must not silently
    fall back to a hand-written renderer if the real template fails.
    """

    vocab_size = 256
    eos_token_id = 201
    pad_token_id = 202
    chat_template = "test-only-template"

    IM_START_ID = 200
    IM_END_ID = 201
    NEWLINE_ID = 10

    _SPECIALS = {
        "<|im_start|>": IM_START_ID,
        "<|im_end|>": IM_END_ID,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIALS.get(token, -1)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # add_special_tokens is ignored: the template emits markers as text.
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for special, sid in self._SPECIALS.items():
                if text.startswith(special, i):
                    ids.append(sid)
                    i += len(special)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            ids.append(self.NEWLINE_ID if ch == "\n" else ord(ch))
            i += 1
        return ids

    def decode(self, ids) -> str:
        out = []
        for sid in ids:
            if sid == self.IM_START_ID:
                out.append("<|im_start|>")
            elif sid == self.IM_END_ID:
                out.append("<|im_end|>")
            elif sid == self.NEWLINE_ID:
                out.append("\n")
            else:
                out.append(chr(sid))
        return "".join(out)


SYS = (
    "Generate CAD model JSON EXACTLY matching this schema:\n"
    "{ \"parts\": {} }\n"
)
QUERY = "make a 10mm cube"
CANDIDATE = '{"parts":{"part_1":null}}'


def _built(tok):
    built, reason = build_cadmium_sequence(QUERY, CANDIDATE, SYS, tok)
    assert built is not None, reason
    return built


def _one_hot_logits(L: int, V: int, target_ids, target_start: int) -> np.ndarray:
    """Logits where row (p-1) is one-hot on target_ids[p-target_start]."""
    logits = np.full((L, V), -20.0, dtype=np.float32)
    for k, tid in enumerate(target_ids):
        pos = target_start + k
        logits[pos - 1, tid] = 20.0
    return logits


class TestSystemMessageSHA(unittest.TestCase):
    def test_load_matches_expected_sha(self):
        msg = load_system_message(CONFIG)
        self.assertEqual(len(msg), 2152)
        import hashlib
        self.assertEqual(
            hashlib.sha256(msg.encode("utf-8")).hexdigest(),
            EXPECTED_SYSTEM_MESSAGE_SHA256,
        )

    def test_tampered_config_raises(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
            cfg["system_message"] = "tampered"
            p = Path(d) / "bad.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ScoringError):
                load_system_message(p)

    def test_tampered_message_and_config_hash_still_raise(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
            cfg["system_message"] = "tampered"
            cfg["system_message_sha256"] = hashlib.sha256(b"tampered").hexdigest()
            p = Path(d) / "bad.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ScoringError):
                load_system_message(p)


class TestFingerprint(unittest.TestCase):
    def test_fake_cannot_masquerade_as_cadmium_tokenizer(self):
        fp = fingerprint_tokenizer(FakeTokenizer())
        self.assertFalse(fp.matches)
        self.assertEqual(fp.im_end_token_id, 201)
        self.assertEqual(fp.eos_token_id, 201)

    def test_mismatch_on_wrong_vocab(self):
        class Bad(FakeTokenizer):
            vocab_size = 999
        self.assertFalse(fingerprint_tokenizer(Bad()).matches)


class TestBuildSequence(unittest.TestCase):
    def test_template_failure_does_not_fall_back(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("template unavailable")
        with self.assertRaises((ScoringError, RuntimeError)):
            build_cadmium_sequence(QUERY, CANDIDATE, SYS, Broken())

    def test_prefix_is_strict_token_prefix(self):
        b = _built(FakeTokenizer())
        self.assertEqual(b.full_ids[: b.prefix_len], b.prefix_ids)
        self.assertGreater(b.target_len, 0)
        self.assertEqual(b.target_len, b.full_len - b.prefix_len)

    def test_target_ends_with_im_end_then_newline(self):
        b = _built(FakeTokenizer())
        self.assertEqual(b.target_ids[-2], FakeTokenizer.IM_END_ID)
        self.assertEqual(b.target_ids[-1], FakeTokenizer.NEWLINE_ID)

    def test_no_extra_eos_appended(self):
        # The template's <|im_end|> is the terminator; no further EOS token is
        # appended. full_ids must end with the newline, not a duplicated EOS.
        b = _built(FakeTokenizer())
        self.assertEqual(b.full_ids[-1], FakeTokenizer.NEWLINE_ID)
        self.assertEqual(b.full_ids[-2], FakeTokenizer.IM_END_ID)

    def test_empty_candidate_rejected(self):
        built, reason = build_cadmium_sequence(QUERY, "", SYS, FakeTokenizer())
        self.assertIsNone(built)
        self.assertEqual(reason, REASON_EMPTY_TARGET)
        built, reason = build_cadmium_sequence(QUERY, "   \n  ", SYS, FakeTokenizer())
        self.assertIsNone(built)
        self.assertEqual(reason, REASON_EMPTY_TARGET)

    def test_prefix_mismatch_when_assistant_content_breaks_boundary(self):
        # A tokenizer that encodes the boundary differently would break the
        # strict-prefix property; simulate by using a tokenizer whose newline
        # token merges with the following char.
        class MergeNewline(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                # encode '\n' together with the next char into one id (300+ord)
                ids = []
                i = 0
                while i < len(text):
                    for sp, sid in self._SPECIALS.items():
                        if text.startswith(sp, i):
                            ids.append(sid)
                            i += len(sp)
                            break
                    else:
                        if text[i] == "\n" and i + 1 < len(text):
                            ids.append(300 + ord(text[i + 1]))
                            i += 2
                        else:
                            ids.append(ord(text[i]))
                            i += 1
                return ids
        built, reason = build_cadmium_sequence(QUERY, CANDIDATE, SYS, MergeNewline())
        # The merge corrupts the assistant-header newline boundary in prefix vs
        # full, so the strict prefix check should fail.
        if built is None:
            self.assertEqual(reason, REASON_PREFIX_MISMATCH)
        else:
            # If merge happened to keep prefix consistent, terminator check
            # still must pass; either way no silent scoring of a broken pair.
            self.assertIn(built.target_ids[-2], (FakeTokenizer.IM_END_ID,))


class TestScoringShiftAndMask(unittest.TestCase):
    def test_public_scoring_rejects_nonofficial_tokenizer(self):
        tok = FakeTokenizer()
        b = _built(tok)
        logits = _one_hot_logits(b.full_len, 256, b.target_ids, b.prefix_len)
        result = official_score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.reason, REASON_TOKENIZER_MISMATCH)

    def test_correct_shift_gives_near_zero_sum(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = _one_hot_logits(b.full_len, V, b.target_ids, b.prefix_len)
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertEqual(r.status, "scored")
        self.assertEqual(r.target_len, b.target_len)
        self.assertAlmostEqual(r.sum_logprob, 0.0, places=3)
        self.assertTrue(r.all_finite)

    def test_wrong_shift_changes_sum(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        # Offset predictions by one row: row p predicts target at p+1.
        logits = np.full((b.full_len, V), -20.0, dtype=np.float32)
        for k, tid in enumerate(b.target_ids):
            pos = b.prefix_len + k
            # use row pos (wrong) instead of pos-1 (correct)
            if pos < b.full_len:
                logits[pos, tid] = 20.0
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertEqual(r.status, "scored")
        self.assertLess(r.sum_logprob, -5.0)

    def test_uniform_logits_match_log_vocab(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = np.zeros((b.full_len, V), dtype=np.float32)
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        import math
        self.assertAlmostEqual(r.sum_logprob, -b.target_len * math.log(V), places=2)

    def test_nonfinite_rejected(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = _one_hot_logits(b.full_len, V, b.target_ids, b.prefix_len)
        logits[b.prefix_len, :] = np.nan
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertEqual(r.status, "rejected")
        self.assertEqual(r.reason, REASON_NONFINITE)

    def test_logits_length_mismatch_rejected(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = np.zeros((b.full_len - 1, V), dtype=np.float32)
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertEqual(r.status, "rejected")
        self.assertIn("mismatch", r.reason)

    def test_mean_is_sum_over_n(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = np.zeros((b.full_len, V), dtype=np.float32)
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertAlmostEqual(r.mean_logprob, r.sum_logprob / r.target_len, places=6)

    def test_mismatched_caller_input_ids_are_rejected(self):
        tok = FakeTokenizer()
        b = _built(tok)
        logits = _one_hot_logits(b.full_len, 256, b.target_ids, b.prefix_len)
        wrong_ids = np.asarray(b.full_ids, dtype=np.int64)
        wrong_ids[b.prefix_len] = (wrong_ids[b.prefix_len] + 1) % 256
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits, input_ids=wrong_ids)
        self.assertEqual(r.status, "rejected")
        self.assertIsNone(r.sum_logprob)

    def test_malformed_scalar_input_ids_are_rejected(self):
        tok = FakeTokenizer()
        b = _built(tok)
        logits = _one_hot_logits(b.full_len, 256, b.target_ids, b.prefix_len)
        r = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits, input_ids=np.asarray(1))
        self.assertEqual(r.status, "rejected")


class TestContext4096(unittest.TestCase):
    def test_over_4096_rejected_not_truncated(self):
        tok = FakeTokenizer()
        # Pad candidate with many chars so full_len exceeds 4096.
        big = '{"parts":{"part_1":null}}' + ("x" * 4096)
        built, reason = build_cadmium_sequence(QUERY, big, SYS, tok)
        self.assertIsNotNone(built)
        self.assertGreater(built.full_len, MAX_TRAINING_CONTEXT)
        # score_pair_from_logits must reject, not truncate.
        V = 256
        logits = np.zeros((built.full_len, V), dtype=np.float32)
        r = score_pair_from_logits(QUERY, big, SYS, tok, logits)
        self.assertEqual(r.status, "rejected")
        self.assertEqual(r.reason, REASON_OUT_OF_TRAINING_CONTEXT)
        self.assertFalse(r.within_training_context)
        self.assertIsNone(r.sum_logprob)

    def test_exactly_4096_is_scored(self):
        tok = FakeTokenizer()
        b = _built(tok)
        # Build a candidate whose full_len is exactly 4096 by padding.
        # target_len currently = b.target_len; we want full_len = 4096.
        target_len_now = b.target_len
        prefix_len = b.prefix_len
        need = MAX_TRAINING_CONTEXT - prefix_len - 2  # leave room for im_end+newline
        pad_chars = max(0, need - (len(CANDIDATE)))
        cand = CANDIDATE + ("x" * pad_chars)
        built, reason = build_cadmium_sequence(QUERY, cand, SYS, tok)
        self.assertIsNotNone(built)
        self.assertEqual(built.full_len, MAX_TRAINING_CONTEXT)
        V = 256
        logits = _one_hot_logits(built.full_len, V, built.target_ids, built.prefix_len)
        r = score_pair_from_logits(QUERY, cand, SYS, tok, logits)
        self.assertEqual(r.status, "scored", r.reason)


class TestBatchConsistency(unittest.TestCase):
    def test_identical_scores_pass(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = _one_hot_logits(b.full_len, V, b.target_ids, b.prefix_len)
        a = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        c = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        self.assertTrue(score_batch_consistency([a], [c]))

    def test_different_scores_fail(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = _one_hot_logits(b.full_len, V, b.target_ids, b.prefix_len)
        a = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits)
        logits2 = logits.copy()
        first_target = b.target_ids[0]
        logits2[b.prefix_len - 1, first_target] = -20.0
        logits2[b.prefix_len - 1, (first_target + 1) % V] = 20.0
        c = score_pair_from_logits(QUERY, CANDIDATE, SYS, tok, logits2)
        self.assertFalse(score_batch_consistency([a], [c]))


class TestVerbatimAndIsolation(unittest.TestCase):
    def test_candidate_json_preserved_verbatim(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        logits = _one_hot_logits(b.full_len, V, b.target_ids, b.prefix_len)
        weird = '{"parts":{"part_1":"has \\"quotes\\" and {braces}"}}'
        built, reason = build_cadmium_sequence(QUERY, weird, SYS, tok)
        self.assertIsNotNone(built)
        r = score_pair_from_logits(QUERY, weird, SYS, tok,
                                   _one_hot_logits(built.full_len, V, built.target_ids, built.prefix_len))
        self.assertEqual(r.candidate_json_desc, weird)

    def test_scorer_takes_no_filesystem_inputs(self):
        # The scoring path must not open qrels/audit/CSV. build_cadmium_sequence
        # accepts only (query, candidate, system_message, tokenizer); no path.
        import inspect
        sig = inspect.signature(build_cadmium_sequence)
        self.assertEqual(
            list(sig.parameters),
            ["query_text", "candidate_json_desc", "system_message", "tokenizer"],
        )
        sig2 = inspect.signature(official_score_pair_from_logits)
        self.assertEqual(
            list(sig2.parameters),
            ["query_text", "candidate_json_desc", "system_message", "tokenizer", "logits", "input_ids"],
        )


class TestLedger(unittest.TestCase):
    def test_ledger_counts(self):
        tok = FakeTokenizer()
        b = _built(tok)
        V = 256
        good = score_pair_from_logits(
            QUERY, CANDIDATE, SYS, tok,
            _one_hot_logits(b.full_len, V, b.target_ids, b.prefix_len))
        empty = score_pair_from_logits(QUERY, "", SYS, tok, np.zeros((3, V), dtype=np.float32))
        big = '{"parts":{"part_1":null}}' + ("x" * 4096)
        built_big, _ = build_cadmium_sequence(QUERY, big, SYS, tok)
        ooc = score_pair_from_logits(QUERY, big, SYS, tok,
                                     np.zeros((built_big.full_len, V), dtype=np.float32))
        led = ScoreLedger()
        for r in (good, empty, ooc):
            led.add(r)
        self.assertEqual(led.n_total, 3)
        self.assertEqual(led.n_scored, 1)
        self.assertEqual(led.n_rejected, 2)
        self.assertEqual(led.by_reason[REASON_EMPTY_TARGET], 1)
        self.assertEqual(led.by_reason[REASON_OUT_OF_TRAINING_CONTEXT], 1)
        self.assertEqual(led.n_out_of_context, 1)
        d = led.to_dict()
        self.assertEqual(d["n_scored"], 1)


@unittest.skipUnless(CADMIUM_CACHE.exists(), "real CADmium tokenizer not cached locally")
class TestRealTokenizerCPU(unittest.TestCase):
    """Small CPU sample with the real verified CADmium adapter tokenizer.

    No weights are loaded and no model forward runs -- only tokenization. This
    validates the contract's prefix / terminator / length checks against the
    actual Qwen2.5-Coder chat template shipped with the adapter.
    """

    def setUp(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(CADMIUM_CACHE), trust_remote_code=True)
        self.sysmsg = load_system_message(CONFIG)

    def test_fingerprint_matches(self):
        fp = fingerprint_tokenizer(self.tok)
        self.assertTrue(fp.matches, fp)

    def test_prefix_strict_and_terminator(self):
        built, reason = build_cadmium_sequence(
            "create a cylindrical pin 5mm diameter 20mm long",
            '{"parts":{"part_1":null}}',
            self.sysmsg,
            self.tok,
        )
        self.assertIsNotNone(built, reason)
        self.assertEqual(built.full_ids[: built.prefix_len], built.prefix_ids)
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(built.target_ids[-2], im_end)
        # trailing token must decode to a newline-bearing string
        tail = self.tok.decode([built.target_ids[-1]])
        self.assertIn("\n", tail)

    def test_no_extra_eos(self):
        built, _ = build_cadmium_sequence("q", '{"parts":{"part_1":null}}', self.sysmsg, self.tok)
        self.assertEqual(built.full_ids[-2], self.tok.convert_tokens_to_ids("<|im_end|>"))

    def test_lengths_within_survey_range(self):
        # Codex's CPU survey: prefixes 665-761 tokens, targets 182-3446.
        # We only assert our one prefix is in a plausible band, not hardcode.
        built, _ = build_cadmium_sequence("a small bracket", '{"parts":{"part_1":null}}', self.sysmsg, self.tok)
        self.assertGreater(built.prefix_len, 600)
        self.assertLess(built.prefix_len, 800)


if __name__ == "__main__":
    unittest.main()
