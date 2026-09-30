"""CPU synthetic tests for the ported Text2CAD forward architecture and strict loader.

M1c stage B1 scope: NO real model forward, NO official weights on the forward
path, NO NPU. Everything below runs on CPU with small random weights or
hand-built tensors. The real checkpoint appears ONLY as a structure reference:
the committed manifest ``experiments/configs/text2cad_checkpoint_expected_keys_20260930.json``
(key/shape/dtype/zero-tensor facts dumped by
``experiments/scripts/dump_text2cad_expected_keys.py``, itself a
``weights_only=True`` structure audit) is cross-checked against the ported
architecture -- loading the real ``.pth`` happens in
``experiments/scripts/audit_text2cad_model_load.py``, not in this suite.

Covers:
  * mask semantics: causal + START-column shield, cross-attention additive mask;
  * sin/cos positional encoding values and non-persistence;
  * CAD token embedder (4-table sum, padding zeroing);
  * MultiHeadAttention vs a hand-written float64 reference (self-attention and
    encoder-decoder branches), exact masked-weight zeros, 3D-mask conventions;
  * CrossAttention quirks preserved (additive-mask-only padding, num_heads=1);
  * architecture key-set/shape parity with the 172-key manifest;
  * dual-head logits shape/independence, eval determinism, dropout presence;
  * end-to-end causal/padding invariance through the full decoder;
  * teacher-forcing slicing glue (train.py:257-270) without input mutation;
  * adapted TextEmbedder with fakes: device routing, local_files_only wiring,
    >512 token rejection (no silent truncation), mask semantics;
  * strict loader: happy path + every rejection reason;
  * source hygiene: no CUDA literals / no transformers import / no OCC imports
    in the torch-only architecture files;
  * random-weight integration with the stage-A scorer incl. batch consistency.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_vector_scorer import (  # noqa: E402
    MAX_CAD_SEQUENCE_LENGTH,
    ONE_HOT_SIZE,
    two_head_factorized_score,
    validate_cad_vec,
)
from idea01.text2cad_port.model import (  # noqa: E402
    CAD_CLASS_INFO,
    MAX_CAD_SEQUENCE_LENGTH as PORT_MAX_CAD_SEQUENCE_LENGTH,
    OFFICIAL_INFERENCE_CONFIG,
    CADDecoder,
    CADDecoderLayer,
    CADSequenceEmbedder,
    CrossAttention,
    FeedForwardLayer,
    MultiHeadAttention,
    PortInputError,
    PositionalEncodingSinCos,
    Text2CAD,
    TextAdaptiveLayer,
    build_teacher_forcing_inputs,
    generate_attention_mask,
    prepare_cross_attention_mask_batch,
    teacher_forced_logits,
)
from idea01.text2cad_port.text_embed import (  # noqa: E402
    REASON_NO_TEXT_MODEL,
    REASON_TEXT_TOO_LONG,
    TextEmbedder,
)
from idea01.text2cad_port.loader import (  # noqa: E402
    EXPECTED_ZERO_TENSOR_KEYS,
    PINNED_BERT_FILE_SHA256,
    AssetVerificationError,
    REASON_ASSET_MISSING,
    REASON_BAD_CHECKPOINT_STRUCTURE,
    REASON_BERT_TENSOR_IN_CHECKPOINT,
    REASON_EXPECTED_ZERO_VIOLATION,
    REASON_MANIFEST_MISMATCH,
    REASON_MISSING_KEYS,
    REASON_SHAPE_MISMATCH,
    REASON_SHA_MISMATCH,
    REASON_UNEXPECTED_KEYS,
    REASON_UNEXPECTED_ZERO_TENSOR,
    build_text2cad,
    load_verified_checkpoint,
    nonbert_state_dict_keys,
    strict_load_frozen_weights,
    verify_bert_asset_shas,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"
)
PORT_DIR = REPO_ROOT / "experiments/src/idea01/text2cad_port"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class StubEmbedder(torch.nn.Module):
    """Text embedder placeholder for architecture/loader tests (no forward)."""

    def get_embedding(self, texts):  # pragma: no cover - must never be called
        raise NotImplementedError("stub embedder must not be called")


def build_official_model(device="cpu") -> Text2CAD:
    return build_text2cad(StubEmbedder(), device=device)


def make_circle_like_candidate(pad_to=MAX_CAD_SEQUENCE_LENGTH):
    """Hand-built official-format candidate (1 circle + 1 extrusion).

    Layout mirrors the real ``0000/00000093``-style vectors audited in stage A:
    [START,0], 6 sketch rows (2 coord pairs + END_CURVE/END_LOOP/END_FACE/
    END_SKETCH), 11 extrusion rows (9 params, 1 boolean, END_EXTRUSION),
    terminal [START,0], tail [0,0] padding. flag/index follow
    CADSequence.to_vec (cad_sequence.py:242-274).
    """
    rows = [[1, 0]]
    flags = [0]
    indices = [0]
    sketch = [[139, 139], [139, 266], [5, 0], [4, 0], [3, 0], [2, 0]]
    rows += sketch
    flags += [0] * len(sketch)
    extrusion = [[17 + i, 0] for i in range(9)] + [[8, 0], [6, 0]]
    assert len(extrusion) == 11
    rows += extrusion
    flags += [1] + list(range(1, 11))
    indices += [0] * (len(sketch) + len(extrusion))
    rows.append([1, 0])
    flags.append(0)
    indices.append(0)
    n_real = len(rows)
    n_pad = pad_to - n_real
    rows += [[0, 0]] * n_pad
    flags += [11] * n_pad
    indices += [max(indices) + 1] * n_pad
    cad_vec = torch.tensor(rows, dtype=torch.int32)
    flag_vec = torch.tensor(flags, dtype=torch.int32)
    index_vec = torch.tensor(indices, dtype=torch.int32)
    key_padding_mask = cad_vec == 0
    # Sanity: the stage-A validator must accept this candidate.
    validate_cad_vec(cad_vec, key_padding_mask, flag_vec, index_vec)
    return cad_vec, flag_vec, index_vec, key_padding_mask


def make_batch(vec, flag, index, mask, batch=1):
    vec_dict = {
        "cad_vec": vec.unsqueeze(0).repeat(batch, 1, 1),
        "flag_vec": flag.unsqueeze(0).repeat(batch, 1),
        "index_vec": index.unsqueeze(0).repeat(batch, 1),
    }
    mask_dict = {
        "attn_mask": generate_attention_mask(vec.shape[0] - 1),
        "key_padding_mask": mask.unsqueeze(0).repeat(batch, 1, 1),
    }
    return vec_dict, mask_dict


def manual_mha_reference(x_q, x_k, x_v, W, b, Wo, bo, num_heads,
                         attn_mask=None, key_padding_mask=None):
    """Independent float64 reference of multi_head_attention_forward.

    x_q (L,B,E), x_k/x_v (S,B,E); W (3E,E), b (3E,), Wo (E,E), bo (E,).
    Returns (output (B,L,E), weights (B,H,L,S)) mirroring the module's return.
    """
    L, B, E = x_q.shape
    S = x_k.shape[0]
    H = num_heads
    D = E // H
    scale = D ** -0.5
    W64, b64 = W.to(torch.float64), b.to(torch.float64)
    q = (x_q.to(torch.float64) @ W64[:E].T + b64[:E]) * scale
    k = x_k.to(torch.float64) @ W64[E:2 * E].T + b64[E:2 * E]
    v = x_k.to(torch.float64) @ W64[2 * E:].T + b64[2 * E:]
    weights = torch.zeros(B, H, L, S, dtype=torch.float64)
    out = torch.zeros(B, L, E, dtype=torch.float64)
    for bi in range(B):
        for h in range(H):
            qh = q[:, bi, h * D:(h + 1) * D]  # (L,D)
            kh = k[:, bi, h * D:(h + 1) * D]  # (S,D)
            vh = v[:, bi, h * D:(h + 1) * D]
            scores = qh @ kh.T  # (L,S)
            if attn_mask is not None:
                scores = scores + attn_mask.to(torch.float64)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[bi].unsqueeze(0), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            weights[bi, h] = w
            out[bi, :, h * D:(h + 1) * D] = w @ vh
    out = out @ Wo.to(torch.float64).T + bo.to(torch.float64)
    return out, weights


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


class TestMasks(unittest.TestCase):
    def test_causal_mask_with_start_column_shield(self):
        mask = generate_attention_mask(4)
        self.assertEqual(tuple(mask.shape), (4, 4))
        # row 0 attends only itself
        self.assertEqual(float(mask[0, 0]), 0.0)
        self.assertTrue(torch.isinf(mask[0, 1:]).all())
        # START column hidden from every later position (utils.py:1801-1802)
        self.assertTrue(torch.isinf(mask[1:, 0]).all())
        # lower triangle (except column 0) visible, upper triangle hidden
        self.assertEqual(float(mask[1, 1]), 0.0)
        self.assertEqual(float(mask[2, 1]), 0.0)
        self.assertEqual(float(mask[3, 2]), 0.0)
        self.assertTrue(torch.isinf(mask[1, 2]))
        self.assertTrue(torch.isinf(mask[2, 3]))
        self.assertEqual(mask.dtype, torch.float32)

    def test_mask_start_token_flag(self):
        mask = generate_attention_mask(4, mask_start_token=False)
        self.assertEqual(float(mask[1, 0]), 0.0)
        self.assertEqual(float(mask[3, 0]), 0.0)

    def test_rectangular_mask(self):
        mask = generate_attention_mask(3, 5)
        self.assertEqual(tuple(mask.shape), (3, 5))

    def test_cross_attention_mask_batch_single(self):
        kpm = torch.tensor([[False, False, True, True]])
        mask = prepare_cross_attention_mask_batch(kpm, cad_seq_len=3)
        self.assertEqual(tuple(mask.shape), (1, 3, 4))
        self.assertTrue((mask[:, :, :2] == 0).all())
        self.assertTrue(torch.isinf(mask[:, :, 2:]).all())
        self.assertTrue((mask < 0).all(dim=None) is not None)  # sanity
        self.assertTrue(mask.dtype.is_floating_point)

    def test_cross_attention_mask_batch_multi_matches_unsqueeze(self):
        kpm = torch.tensor([[False, True, False, True], [True, False, False, False]])
        batched = prepare_cross_attention_mask_batch(kpm, cad_seq_len=5)
        self.assertEqual(tuple(batched.shape), (2, 5, 4))
        # the B>1 reshape branch must equal per-sample unsqueeze tiling
        for i in range(2):
            single = prepare_cross_attention_mask_batch(kpm[i : i + 1], cad_seq_len=5)
            self.assertTrue(torch.equal(batched[i : i + 1], single))


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding(unittest.TestCase):
    def test_values_match_formula(self):
        d, n = 16, 7
        pe = PositionalEncodingSinCos(embedding_size=d, max_seq_len=272, device="cpu")
        out = pe(n)
        self.assertEqual(tuple(out.shape), (1, n, d))
        pos = torch.arange(n, dtype=torch.float64).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float64) * (-math.log(10000.0) / d))
        ref = torch.zeros(n, d, dtype=torch.float64)
        ref[:, 0::2] = torch.sin(pos * div)
        ref[:, 1::2] = torch.cos(pos * div)
        self.assertTrue(torch.allclose(out[0].to(torch.float64), ref, atol=1e-6))

    def test_not_persisted_in_state_dict(self):
        # upstream keeps pos_enc as a plain attribute, so the checkpoint has no
        # positional-encoding keys; the port must not turn it into a buffer.
        model = build_official_model()
        pe_keys = [k for k in model.state_dict() if "pos_enc" in k or k.endswith(".pe")]
        self.assertEqual(pe_keys, [])

    def test_table_stays_off_forward_device_contract(self):
        pe = PositionalEncodingSinCos(embedding_size=8, max_seq_len=272, device="cpu")
        out = pe(3)
        self.assertEqual(out.device, torch.device("cpu"))


# ---------------------------------------------------------------------------
# CAD token embedder
# ---------------------------------------------------------------------------


class TestCADSequenceEmbedder(unittest.TestCase):
    def test_four_table_sum_and_padding_zeroing(self):
        torch.manual_seed(0)
        emb = CADSequenceEmbedder(
            one_hot_size=ONE_HOT_SIZE, flag_size=12, index_size=11, d_model=8, device="cpu"
        )
        vec_dict = {
            "cad_vec": torch.tensor([[[1, 0], [37, 94], [0, 0]]]),
            "flag_vec": torch.tensor([[0, 0, 11]]),
            "index_vec": torch.tensor([[0, 0, 1]]),
        }
        kpm = torch.tensor([[[False, False], [False, False], [True, True]]])
        out = emb(vec_dict, kpm)
        manual = (
            emb.sf(vec_dict["flag_vec"])
            + emb.si(vec_dict["index_vec"])
            + emb.cx(torch.tensor([[1, 37, 0]]))
            + emb.cy(torch.tensor([[0, 94, 0]]))
        )
        self.assertTrue(torch.equal(out, manual))

    def test_garbage_values_at_padded_positions_are_zeroed(self):
        torch.manual_seed(1)
        emb = CADSequenceEmbedder(
            one_hot_size=ONE_HOT_SIZE, flag_size=12, index_size=11, d_model=8, device="cpu"
        )
        kpm = torch.tensor([[[False, False], [True, True]]])
        base = {
            "cad_vec": torch.tensor([[[1, 0], [0, 0]]]),
            "flag_vec": torch.tensor([[0, 11]]),
            "index_vec": torch.tensor([[0, 1]]),
        }
        dirty = {
            "cad_vec": torch.tensor([[[1, 0], [266, 266]]]),
            "flag_vec": torch.tensor([[0, 11]]),
            "index_vec": torch.tensor([[0, 1]]),
        }
        self.assertTrue(torch.equal(emb(base, kpm), emb(dirty, kpm)))


# ---------------------------------------------------------------------------
# MultiHeadAttention vs manual reference
# ---------------------------------------------------------------------------


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.E, self.H = 16, 4
        self.mha = MultiHeadAttention(input_dim=self.E, embed_dim=self.E, num_heads=self.H, dropout=0.0)
        self.mha.eval()

    def test_self_attention_matches_float64_reference(self):
        L, B, S = 5, 3, 5
        torch.manual_seed(1)
        x = torch.randn(B, L, self.E)
        attn_mask = generate_attention_mask(L)
        kpm = torch.zeros(B, S, dtype=torch.bool)
        kpm[0, 4] = True
        kpm[2, 3:] = True  # key 0 stays visible for every row
        with torch.no_grad():
            out, weights = self.mha(x, x, x, key_padding_mask=kpm, attn_mask=attn_mask)
        ref_out, ref_w = manual_mha_reference(
            x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1),
            self.mha.in_proj_weight.detach(), self.mha.in_proj_bias.detach(),
            self.mha.out_proj.weight.detach(), self.mha.out_proj.bias.detach(),
            self.H, attn_mask=attn_mask, key_padding_mask=kpm,
        )
        self.assertTrue(torch.allclose(out.to(torch.float64), ref_out, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(weights.to(torch.float64), ref_w, atol=1e-5))
        # masked positions get EXACTLY zero weight (padding + causal + START col)
        self.assertEqual(float(weights[0, :, :, 4].abs().max()), 0.0)
        self.assertEqual(float(weights[2, :, :, 3:].abs().max()), 0.0)
        self.assertEqual(float(weights[:, :, 1:, 0].abs().max()), 0.0)  # START column
        self.assertTrue(torch.isclose(weights.sum(dim=-1), torch.ones(()), atol=1e-5).all())

    def test_encoder_decoder_branch_matches_reference(self):
        # query differs from key==value: exercises the split-projection branch
        L, S, B = 4, 6, 2
        torch.manual_seed(2)
        q = torch.randn(B, L, self.E)
        ctx = torch.randn(B, S, self.E)
        additive = torch.zeros(L, S)
        additive[:, 5] = float("-inf")
        with torch.no_grad():
            out, weights = self.mha(q, ctx, ctx, attn_mask=additive)
        ref_out, ref_w = manual_mha_reference(
            q.transpose(0, 1), ctx.transpose(0, 1), ctx.transpose(0, 1),
            self.mha.in_proj_weight.detach(), self.mha.in_proj_bias.detach(),
            self.mha.out_proj.weight.detach(), self.mha.out_proj.bias.detach(),
            self.H, attn_mask=additive,
        )
        self.assertTrue(torch.allclose(out.to(torch.float64), ref_out, atol=1e-4, rtol=1e-4))
        self.assertEqual(float(weights[..., 5].abs().max()), 0.0)

    def test_batched_3d_mask_collapses_to_first_entry_without_use_3d(self):
        L, B = 4, 2
        torch.manual_seed(3)
        x = torch.randn(B, L, self.E)
        mask2d = generate_attention_mask(L)
        mask3d = mask2d.unsqueeze(0).repeat(B, 1, 1)
        with torch.no_grad():
            out_a, _ = self.mha(x, x, x, attn_mask=mask2d)
            out_b, _ = self.mha(x, x, x, attn_mask=mask3d)
        self.assertTrue(torch.equal(out_a, out_b))

    def test_3d_mask_shape_check(self):
        mha1 = MultiHeadAttention(input_dim=8, embed_dim=8, num_heads=2, dropout=0.0)
        x = torch.randn(2, 3, 8)
        bad = torch.zeros(2, 3, 3)  # B*num_heads == 4 != 2
        with self.assertRaises(RuntimeError):
            mha1(x, x, x, attn_mask=bad, use_3d=True)


class TestCrossAttention(unittest.TestCase):
    def test_padding_suppressed_by_additive_mask_only(self):
        torch.manual_seed(4)
        ca = CrossAttention(input_dim_list=[16, 32], output_dim=16, num_heads=1)
        ca.eval()
        B, N1, L = 2, 5, 6
        x = torch.randn(B, N1, 16)
        ctx = torch.randn(B, L, 16)
        kpm = torch.zeros(B, L, dtype=torch.bool)
        kpm[:, 4:] = True
        additive = prepare_cross_attention_mask_batch(kpm, cad_seq_len=N1)
        with torch.no_grad():
            out_masked, _ = ca(x, ctx, ctx, attn_mask=additive)
            # valid-only computation (context truncated, no mask)
            out_valid, _ = ca(x, ctx[:, :4], ctx[:, :4], attn_mask=None)
        self.assertTrue(torch.allclose(out_masked, out_valid, atol=1e-5, rtol=1e-5))

    def test_key_padding_mask_argument_is_deliberately_ignored(self):
        # upstream CrossAttention.forward does not forward key_padding_mask to
        # the inner MHA; the port preserves that exactly.
        torch.manual_seed(5)
        ca = CrossAttention(input_dim_list=[16, 32], output_dim=16, num_heads=1)
        ca.eval()
        x = torch.randn(2, 5, 16)
        ctx = torch.randn(2, 6, 16)
        with torch.no_grad():
            out_none, _ = ca(x, ctx, ctx, attn_mask=None, key_padding_mask=None)
            out_junk, _ = ca(
                x, ctx, ctx, attn_mask=None,
                key_padding_mask=torch.ones(2, 6, dtype=torch.bool),
            )
        self.assertTrue(torch.equal(out_none, out_junk))


# ---------------------------------------------------------------------------
# Architecture key/shape parity with the pinned manifest
# ---------------------------------------------------------------------------


class TestArchitectureKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        torch.manual_seed(0)
        cls.model = build_official_model()

    def test_manifest_self_consistency(self):
        manifest = self.manifest
        self.assertEqual(manifest["n_tensors"], 172)
        self.assertEqual(len(manifest["tensors"]), 172)
        self.assertEqual(sorted(manifest["zero_tensor_keys"]), sorted(EXPECTED_ZERO_TENSOR_KEYS))
        self.assertEqual(manifest["n_zero_tensors"], 2)
        for key, entry in manifest["tensors"].items():
            self.assertEqual(entry["dtype"], "torch.float32")
        self.assertEqual(
            manifest["provenance"]["checkpoint_sha256"],
            "235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598",
        )

    def test_model_key_set_matches_manifest(self):
        keys = nonbert_state_dict_keys(self.model)
        self.assertEqual(len(keys), 172)
        self.assertEqual(keys, set(self.manifest["tensors"]))

    def test_model_shapes_match_manifest(self):
        state = self.model.state_dict()
        for key, entry in self.manifest["tensors"].items():
            self.assertEqual(list(state[key].shape), entry["shape"], key)

    def test_get_trainable_state_dict_matches_manifest(self):
        # text2cad.py:123-127 is the official checkpoint save path; its key
        # set on the ported model must equal the 172 checkpoint keys.
        self.assertEqual(
            set(self.model.get_trainable_state_dict()), set(self.manifest["tensors"])
        )

    def test_structural_facts(self):
        dec = self.model.cad_decoder
        self.assertEqual(dec.use_ca, [False, False, True, True, True, True, True, True])
        for i, layer in enumerate(dec.cad_decoder_layers):
            self.assertEqual(hasattr(layer, "ca_ze_seq"), i >= 2)
            self.assertTrue(hasattr(layer, "downsampler"))  # created for every layer
            self.assertTrue(hasattr(layer, "norm_seq"))
            if i >= 2:
                self.assertEqual(layer.ca_ze_seq.mha.num_heads, 1)
            self.assertEqual(layer.sa_seq.num_heads, 8)
        # official config in_dim == out_dim == 1024 -> no adaptive downsampler
        self.assertFalse(hasattr(self.model.adaptive_layer, "downsampler"))
        # embedder table sizes (macro.py:40-44)
        self.assertEqual(dec.seq_embed.cx.embed.num_embeddings, 267)
        self.assertEqual(dec.seq_embed.cy.embed.num_embeddings, 267)
        self.assertEqual(dec.seq_embed.sf.embed.num_embeddings, 12)
        self.assertEqual(dec.seq_embed.si.embed.num_embeddings, 11)
        self.assertEqual(dec.seq_output_x[0].out_features, 267)
        self.assertEqual(dec.seq_output_y[0].out_features, 267)
        self.assertEqual(self.model.cad_seq_len, 271)

    def test_expected_zero_keys_exist_and_init_to_zero(self):
        state = self.model.state_dict()
        for key in EXPECTED_ZERO_TENSOR_KEYS:
            self.assertIn(key, state)
            # fresh LayerNorm bias init is all-zero -- the same value the
            # untrained-by-forward keys kept in the real checkpoint
            self.assertTrue(bool((state[key] == 0).all()), key)
        self.assertTrue(
            all("norm_3.bias" in key for key in EXPECTED_ZERO_TENSOR_KEYS)
        )


# ---------------------------------------------------------------------------
# Forward behavior (random weights, full official-size decoder)
# ---------------------------------------------------------------------------


class DeterministicStubEmbedder(torch.nn.Module):
    """Per-text deterministic pseudo-BERT for end-to-end plumbing tests.

    Valid-token rows depend only on the text (hash-seeded), padded rows are
    zeros with mask True -- emulating BERT's property that valid-token outputs
    do not depend on batch composition.
    """

    def __init__(self, hidden=1024):
        super().__init__()
        self.hidden = hidden

    def get_embedding(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            n = (len(text) % 7) + 3
            gen = torch.Generator().manual_seed(
                int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
            )
            rows.append(torch.randn(n, self.hidden, generator=gen))
        width = max(r.shape[0] for r in rows)
        emb = torch.zeros(len(texts), width, self.hidden)
        kpm = torch.ones(len(texts), width, dtype=torch.bool)
        for i, r in enumerate(rows):
            emb[i, : r.shape[0]] = r
            kpm[i, : r.shape[0]] = False
        return emb, kpm


class TestForwardBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = build_text2cad(DeterministicStubEmbedder(), device="cpu")
        cls.model.eval()
        cls.cad_vec, cls.flag_vec, cls.index_vec, cls.kpm = make_circle_like_candidate()

    def _forward(self, texts, batch=1):
        # the model contract consumes teacher-forcing-sliced inputs
        # (train.py:264-270): 271 rows in, the stored 271x271 attn_mask as-is.
        vec_dict, mask_dict = make_batch(
            self.cad_vec, self.flag_vec, self.index_vec, self.kpm, batch=batch
        )
        inputs_vec, inputs_mask, _, _ = build_teacher_forcing_inputs(vec_dict, mask_dict)
        with torch.no_grad():
            pred, _ = self.model(vec_dict=inputs_vec, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
        return pred

    def test_dual_head_logits_shape(self):
        pred = self._forward(["a simple bracket"])
        self.assertEqual(tuple(pred.shape), (1, 271, 2, ONE_HOT_SIZE))
        self.assertTrue(torch.isfinite(pred).all())

    def test_heads_are_independent_linears(self):
        pred = self._forward(["a simple bracket"])
        dec = self.model.cad_decoder
        with torch.no_grad():
            dec.seq_output_x[0].weight.add_(0.01)
            pred_perturbed = self._forward(["a simple bracket"])
        self.assertFalse(torch.allclose(pred[..., 0, :], pred_perturbed[..., 0, :]))
        self.assertTrue(torch.equal(pred[..., 1, :], pred_perturbed[..., 1, :]))

    def test_eval_determinism_and_train_mode_dropout(self):
        texts = ["a simple bracket"]
        out1 = self._forward(texts)
        out2 = self._forward(texts)
        self.assertTrue(torch.equal(out1, out2))
        self.model.train()
        try:
            out3 = self._forward(texts)
            out4 = self._forward(texts)
        finally:
            self.model.eval()
        self.assertFalse(torch.equal(out3, out4))  # dropout 0.1 active in train mode

    def test_start_column_shield_end_to_end(self):
        # Perturbing position-0 inputs must not change any later-position
        # logits: key 0 is shielded from every later query at every layer.
        texts = ["a simple bracket"]
        vec_dict, mask_dict = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        inputs_vec, inputs_mask, _, _ = build_teacher_forcing_inputs(vec_dict, mask_dict)
        with torch.no_grad():
            base, _ = self.model(vec_dict=inputs_vec, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
        perturbed = {k: v.clone() for k, v in inputs_vec.items()}
        perturbed["flag_vec"] = perturbed["flag_vec"].clone()
        perturbed["flag_vec"][0, 0] = 5  # change ONLY the position-0 embedding
        with torch.no_grad():
            alt, _ = self.model(vec_dict=perturbed, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
        self.assertFalse(torch.equal(base[0, 0], alt[0, 0]))  # position 0 itself reacts
        self.assertTrue(torch.equal(base[:, 1:], alt[:, 1:]))  # positions >= 1 do not

    def test_tail_padding_invisible_to_valid_positions(self):
        texts = ["a simple bracket"]
        n_real = int((~self.kpm[:, 0]).sum())
        vec_dict, mask_dict = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        inputs_vec, inputs_mask, _, _ = build_teacher_forcing_inputs(vec_dict, mask_dict)
        with torch.no_grad():
            base, _ = self.model(vec_dict=inputs_vec, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
        dirty = {k: v.clone() for k, v in inputs_vec.items()}
        dirty["cad_vec"] = dirty["cad_vec"].clone()
        dirty["cad_vec"][0, n_real:] = 200  # garbage inside padded rows
        dirty["flag_vec"] = dirty["flag_vec"].clone()
        dirty["flag_vec"][0, n_real:] = 3
        dirty["index_vec"] = dirty["index_vec"].clone()
        dirty["index_vec"][0, n_real:] = 2
        with torch.no_grad():
            alt, _ = self.model(vec_dict=dirty, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
        # predictions targeting real positions (rows 0..n_real-2 of pred) unchanged
        self.assertTrue(torch.equal(base[:, : n_real - 1], alt[:, : n_real - 1]))

    def test_v0_rows_participate_in_attention(self):
        # rows [v, 0] (extrusion-like) are NOT self-attention-masked (the mask
        # is all(axis=2)); changing them must change later-position logits.
        texts = ["a simple bracket"]
        vec_dict, mask_dict = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        inputs_vec, inputs_mask, _, _ = build_teacher_forcing_inputs(vec_dict, mask_dict)
        with torch.no_grad():
            base, _ = self.model(vec_dict=inputs_vec, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
        alt_vec = {k: v.clone() for k, v in inputs_vec.items()}
        alt_vec["cad_vec"] = alt_vec["cad_vec"].clone()
        alt_vec["cad_vec"][0, 7, 0] = 60  # first extrusion param row [v,0]
        alt_mask = {
            "attn_mask": inputs_mask["attn_mask"],
            "key_padding_mask": alt_vec["cad_vec"] == 0,
        }
        with torch.no_grad():
            alt, _ = self.model(vec_dict=alt_vec, texts=texts, mask_cad_dict=alt_mask, metadata=False)
        self.assertFalse(torch.equal(base[:, 7:], alt[:, 7:]))
        self.assertTrue(torch.equal(base[:, :7], alt[:, :7]))  # causality: earlier rows unaffected

    def test_batched_attn_mask_accepted(self):
        # the dataloader stacks the (271,271) mask to (B,271,271); the MHA
        # collapses it to its first entry -- must equal the 2D call bitwise.
        texts = ["a simple bracket"]
        vec_dict, mask_dict = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        inputs_vec, inputs_mask, _, _ = build_teacher_forcing_inputs(vec_dict, mask_dict)
        batched_mask = {
            "attn_mask": inputs_mask["attn_mask"].unsqueeze(0).repeat(1, 1, 1),
            "key_padding_mask": inputs_mask["key_padding_mask"],
        }
        with torch.no_grad():
            out2d, _ = self.model(vec_dict=inputs_vec, texts=texts, mask_cad_dict=inputs_mask, metadata=False)
            out3d, _ = self.model(vec_dict=inputs_vec, texts=texts, mask_cad_dict=batched_mask, metadata=False)
        self.assertTrue(torch.equal(out2d, out3d))


# ---------------------------------------------------------------------------
# Teacher-forcing glue
# ---------------------------------------------------------------------------


class TestTeacherForcingGlue(unittest.TestCase):
    def setUp(self):
        self.cad_vec, self.flag_vec, self.index_vec, self.kpm = make_circle_like_candidate()
        self.vec_dict, self.mask_dict = make_batch(
            self.cad_vec, self.flag_vec, self.index_vec, self.kpm, batch=2
        )

    def test_slicing_matches_train_py(self):
        inputs_vec, inputs_mask, target, shifted = build_teacher_forcing_inputs(
            self.vec_dict, self.mask_dict
        )
        for key in ("cad_vec", "flag_vec", "index_vec"):
            self.assertTrue(torch.equal(inputs_vec[key], self.vec_dict[key][:, :-1]))
        self.assertTrue(torch.equal(inputs_mask["key_padding_mask"], self.mask_dict["key_padding_mask"][:, :-1]))
        self.assertTrue(torch.equal(inputs_mask["attn_mask"], self.mask_dict["attn_mask"]))
        self.assertTrue(torch.equal(target, self.vec_dict["cad_vec"][:, 1:]))
        self.assertTrue(torch.equal(shifted, self.mask_dict["key_padding_mask"][:, 1:]))
        # target is a clone exactly as upstream (train.py:259)
        self.assertFalse(target.data_ptr() == self.vec_dict["cad_vec"].data_ptr())

    def test_inputs_not_mutated(self):
        before_vec = {k: v.clone() for k, v in self.vec_dict.items()}
        before_mask = {k: v.clone() for k, v in self.mask_dict.items()}
        build_teacher_forcing_inputs(self.vec_dict, self.mask_dict)
        for key in self.vec_dict:
            self.assertTrue(torch.equal(self.vec_dict[key], before_vec[key]))
        for key in self.mask_dict:
            self.assertTrue(torch.equal(self.mask_dict[key], before_mask[key]))

    def test_shifted_mask_alignment_with_scorer(self):
        _, _, target, shifted = build_teacher_forcing_inputs(self.vec_dict, self.mask_dict)
        # scorer validity is target != 0; the official loss receives ~shifted
        self.assertTrue(torch.equal(target != 0, ~shifted))

    def test_teacher_forced_logits_full_path(self):
        torch.manual_seed(0)
        model = build_text2cad(DeterministicStubEmbedder(), device="cpu")
        model.eval()
        texts = ["first query text", "a much longer second query text!!"]
        with torch.no_grad():
            pred, target, shifted = teacher_forced_logits(model, texts, self.vec_dict, self.mask_dict)
        self.assertEqual(tuple(pred.shape), (2, 271, 2, ONE_HOT_SIZE))
        self.assertEqual(tuple(target.shape), (2, 271, 2))
        self.assertEqual(tuple(shifted.shape), (2, 271, 2))


# ---------------------------------------------------------------------------
# Adapted TextEmbedder (fakes only; no transformers, no real BERT)
# ---------------------------------------------------------------------------


class FakeBatchEncoding(dict):
    def to(self, device):
        return FakeBatchEncoding(
            {k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()}
        )


class FakeTokenizer:
    def __init__(self, length_fn=None):
        self.calls = []
        self.length_fn = length_fn or (lambda t: len(t.split()) + 2)

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        self.calls.append(
            {
                "texts": list(texts),
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        lens = [self.length_fn(t) for t in texts]
        width = max(lens)
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attn = torch.zeros(len(texts), width, dtype=torch.long)
        for i, n in enumerate(lens):
            input_ids[i, :n] = torch.arange(101, 101 + n)
            attn[i, :n] = 1
        return FakeBatchEncoding({"input_ids": input_ids, "attention_mask": attn})


class FakeBertModel(torch.nn.Module):
    def __init__(self, vocab=700, hidden=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.seen_devices = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.seen_devices.append(input_ids.device)
        return (self.emb(input_ids),)


class TestTextEmbedder(unittest.TestCase):
    def test_device_routing_and_tokenizer_contract(self):
        fake_model = FakeBertModel()
        fake_tok = FakeTokenizer()
        embedder = TextEmbedder(max_seq_len=512, device="cpu", model=fake_model, tokenizer=fake_tok)
        emb, kpm = embedder.get_embedding(["draw a hole", "plate"])
        self.assertEqual(fake_model.seen_devices[-1], torch.device("cpu"))
        call = fake_tok.calls[-1]
        self.assertEqual(call["return_tensors"], "pt")
        self.assertEqual(call["padding"], True)
        self.assertEqual(call["truncation"], False)  # never silent truncation
        # token counts: 3+2=5 and 1+2=3 -> width 5, mask True only on pads
        self.assertEqual(tuple(emb.shape), (2, 5, 16))
        self.assertTrue(torch.equal(kpm, torch.tensor([[False] * 5, [False] * 3 + [True] * 2])))

    def test_bare_string_is_wrapped(self):
        embedder = TextEmbedder(max_seq_len=512, device="cpu", model=FakeBertModel(), tokenizer=FakeTokenizer())
        emb, kpm = embedder.get_embedding("single query")
        self.assertEqual(emb.shape[0], 1)

    def test_length_gate_rejects_over_512(self):
        tok = FakeTokenizer(length_fn=lambda t: 513 if "long" in t else 10)
        embedder = TextEmbedder(max_seq_len=512, device="cpu", model=FakeBertModel(), tokenizer=tok)
        with self.assertRaises(PortInputError) as ctx:
            embedder.get_embedding(["ok text", "this is long"])
        self.assertEqual(ctx.exception.reason, REASON_TEXT_TOO_LONG)
        self.assertIn("(1, 513)", str(ctx.exception))
        # exactly 512 passes
        tok_ok = FakeTokenizer(length_fn=lambda t: 512)
        embedder_ok = TextEmbedder(max_seq_len=512, device="cpu", model=FakeBertModel(), tokenizer=tok_ok)
        embedder_ok.get_embedding(["boundary text"])

    def test_unloaded_embedder_rejects(self):
        embedder = TextEmbedder(max_seq_len=512, device="cpu")
        with self.assertRaises(PortInputError) as ctx:
            embedder.get_embedding(["anything"])
        self.assertEqual(ctx.exception.reason, REASON_NO_TEXT_MODEL)

    def test_from_local_uses_local_files_only_and_freezes(self):
        fake_transformers = types.ModuleType("transformers")
        created = {}

        class FakeBert:
            @staticmethod
            def from_pretrained(path, **kwargs):
                created["model_args"] = (path, kwargs)
                return FakeBertModel(), {
                    "missing_keys": [],
                    "unexpected_keys": ["cls.predictions.bias"],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }

        class FakeTok:
            @staticmethod
            def from_pretrained(path, **kwargs):
                created["tokenizer_args"] = (path, kwargs)
                return FakeTokenizer()

        fake_transformers.BertModel = FakeBert
        fake_transformers.BertTokenizer = FakeTok
        previous = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        try:
            embedder = TextEmbedder.from_local("/pinned/bert", max_seq_len=512, device="cpu")
        finally:
            if previous is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = previous
        self.assertEqual(created["model_args"][0], "/pinned/bert")
        self.assertEqual(created["model_args"][1]["local_files_only"], True)
        self.assertEqual(created["model_args"][1]["output_loading_info"], True)
        self.assertEqual(created["tokenizer_args"][1]["local_files_only"], True)
        self.assertEqual(embedder.loading_info["unexpected_keys"], ["cls.predictions.bias"])
        self.assertFalse(embedder.model.training)
        self.assertTrue(all(not p.requires_grad for p in embedder.model.parameters()))


# ---------------------------------------------------------------------------
# Strict loader (synthetic checkpoints; torch-only)
# ---------------------------------------------------------------------------


class TestStrictLoader(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_official_model()
        # A fresh init has 56 all-zero tensors (LayerNorm biases, MHA
        # in_proj_bias / out_proj.bias all start at 0). The real checkpoint
        # has exactly the two documented ones, so randomize every parameter
        # and then re-zero precisely the expected-zero keys.
        with torch.no_grad():
            for param in self.model.parameters():
                param.normal_(0, 0.05)
            for key in EXPECTED_ZERO_TENSOR_KEYS:
                self.model.state_dict()[key].zero_()
        self.state = {
            "module." + key: value.detach().clone()
            for key, value in self.model.get_trainable_state_dict().items()
        }
        self.tmp = Path(tempfile.mkdtemp(prefix="t2c_loader_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_checkpoint(self, state, name="ckpt.pth", extra_top_level=None):
        blob = {"model_state_dict": state}
        if extra_top_level:
            blob.update(extra_top_level)
        path = self.tmp / name
        torch.save(blob, path)
        return path

    def test_happy_path_round_trip(self):
        path = self._write_checkpoint(self.state)
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded, report = load_verified_checkpoint(path, expected_sha256=sha)
        self.assertEqual(report["n_tensors"], 172)
        self.assertEqual(report["zero_tensor_keys"], sorted(EXPECTED_ZERO_TENSOR_KEYS))
        fresh = build_official_model()
        load_report = strict_load_frozen_weights(fresh, loaded, expected_keys_manifest=MANIFEST_PATH)
        self.assertTrue(load_report["numeric_equality"])
        self.assertEqual(load_report["n_loaded_tensors"], 172)
        post = fresh.state_dict()
        for key, value in loaded.items():
            self.assertTrue(torch.equal(post[key], value), key)

    def test_sha_mismatch_rejected(self):
        path = self._write_checkpoint(self.state)
        with self.assertRaises(AssetVerificationError) as ctx:
            load_verified_checkpoint(path, expected_sha256="0" * 64)
        self.assertEqual(ctx.exception.reason, REASON_SHA_MISMATCH)

    def test_missing_file_rejected(self):
        with self.assertRaises(AssetVerificationError) as ctx:
            load_verified_checkpoint(self.tmp / "nope.pth", expected_sha256="0" * 64)
        self.assertEqual(ctx.exception.reason, REASON_ASSET_MISSING)

    def test_top_level_structure_rejected(self):
        path = self._write_checkpoint(self.state, extra_top_level={"epoch": 150})
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(AssetVerificationError) as ctx:
            load_verified_checkpoint(path, expected_sha256=sha)
        self.assertEqual(ctx.exception.reason, REASON_BAD_CHECKPOINT_STRUCTURE)

    def test_unprefixed_key_rejected(self):
        state = dict(self.state)
        first = sorted(state)[0]
        state[first.removeprefix("module.")] = state.pop(first)
        path = self._write_checkpoint(state)
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(AssetVerificationError) as ctx:
            load_verified_checkpoint(path, expected_sha256=sha)
        self.assertEqual(ctx.exception.reason, REASON_BAD_CHECKPOINT_STRUCTURE)

    def test_bert_tensor_in_checkpoint_rejected(self):
        state = dict(self.state)
        state["module.base_text_embedder.model.embeddings.word_embeddings.weight"] = torch.zeros(2, 2)
        path = self._write_checkpoint(state)
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(AssetVerificationError) as ctx:
            load_verified_checkpoint(path, expected_sha256=sha)
        self.assertEqual(ctx.exception.reason, REASON_BERT_TENSOR_IN_CHECKPOINT)

    def test_missing_key_rejected(self):
        state = {k.removeprefix("module."): v for k, v in self.state.items()}
        state.pop("cad_decoder.seq_output_x.0.weight")
        fresh = build_official_model()
        with self.assertRaises(AssetVerificationError) as ctx:
            strict_load_frozen_weights(fresh, state)
        self.assertEqual(ctx.exception.reason, REASON_MISSING_KEYS)

    def test_unexpected_key_rejected(self):
        state = {k.removeprefix("module."): v for k, v in self.state.items()}
        state["cad_decoder.bogus.weight"] = torch.zeros(3)
        fresh = build_official_model()
        with self.assertRaises(AssetVerificationError) as ctx:
            strict_load_frozen_weights(fresh, state)
        self.assertEqual(ctx.exception.reason, REASON_UNEXPECTED_KEYS)

    def test_shape_mismatch_rejected(self):
        state = {k.removeprefix("module."): v for k, v in self.state.items()}
        state["cad_decoder.seq_output_x.0.bias"] = torch.full((267,), 0.5)[:266]
        fresh = build_official_model()
        with self.assertRaises(AssetVerificationError) as ctx:
            strict_load_frozen_weights(fresh, state)
        self.assertEqual(ctx.exception.reason, REASON_SHAPE_MISMATCH)

    def test_unexpected_zero_tensor_rejected(self):
        state = {k: v.clone() for k, v in self.state.items()}
        state["module.cad_decoder.seq_output_y.0.bias"].zero_()
        stripped = {k.removeprefix("module."): v for k, v in state.items()}
        fresh = build_official_model()
        with self.assertRaises(AssetVerificationError) as ctx:
            strict_load_frozen_weights(fresh, stripped)
        self.assertEqual(ctx.exception.reason, REASON_UNEXPECTED_ZERO_TENSOR)

    def test_expected_zero_violation_rejected(self):
        state = {k: v.clone() for k, v in self.state.items()}
        key = "module.cad_decoder.cad_decoder_layers.0.norm_seq.norm_3.bias"
        state[key] = torch.ones_like(state[key])
        stripped = {k.removeprefix("module."): v for k, v in state.items()}
        fresh = build_official_model()
        with self.assertRaises(AssetVerificationError) as ctx:
            strict_load_frozen_weights(fresh, stripped)
        self.assertEqual(ctx.exception.reason, REASON_EXPECTED_ZERO_VIOLATION)

    def test_manifest_mismatch_rejected(self):
        stripped = {k.removeprefix("module."): v for k, v in self.state.items()}
        bad_manifest = {
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in stripped.items()
            },
            "zero_tensor_keys": sorted(EXPECTED_ZERO_TENSOR_KEYS),
            "n_tensors": len(stripped),
        }
        bad_manifest["tensors"]["cad_decoder.seq_output_x.0.weight"]["shape"] = [1, 1]
        fresh = build_official_model()
        with self.assertRaises(AssetVerificationError) as ctx:
            strict_load_frozen_weights(fresh, stripped, expected_keys_manifest=bad_manifest)
        self.assertEqual(ctx.exception.reason, REASON_MANIFEST_MISMATCH)

    def test_verify_bert_asset_shas(self):
        bert_dir = self.tmp / "bert"
        bert_dir.mkdir()
        for name, sha in PINNED_BERT_FILE_SHA256.items():
            (bert_dir / name).write_bytes(b"pinned-bytes" if name != "vocab.txt" else b"other")
        good_shas = {
            name: hashlib.sha256((bert_dir / name).read_bytes()).hexdigest()
            for name in PINNED_BERT_FILE_SHA256
        }
        report = verify_bert_asset_shas(bert_dir, good_shas)
        self.assertEqual(sorted(report), sorted(PINNED_BERT_FILE_SHA256))
        with self.assertRaises(AssetVerificationError) as ctx:
            verify_bert_asset_shas(bert_dir, PINNED_BERT_FILE_SHA256)
        self.assertEqual(ctx.exception.reason, REASON_SHA_MISMATCH)
        (bert_dir / "vocab.txt").unlink()
        with self.assertRaises(AssetVerificationError) as ctx:
            verify_bert_asset_shas(bert_dir, good_shas)
        self.assertEqual(ctx.exception.reason, REASON_ASSET_MISSING)


# ---------------------------------------------------------------------------
# Source hygiene: the port must not reintroduce CUDA literals / OCC imports
# ---------------------------------------------------------------------------


class TestSourceHygiene(unittest.TestCase):
    def _source(self, name):
        return (PORT_DIR / name).read_text(encoding="utf-8")

    def test_no_cuda_literals_in_port(self):
        for name in ("model.py", "text_embed.py", "loader.py", "__init__.py"):
            src = self._source(name)
            for literal in (
                '.to("cuda")',
                ".to('cuda')",
                'torch.device("cuda")',
                "torch.device('cuda')",
                "torch.cuda.is_available()",
                'device="cuda"',
                "device='cuda'",
            ):
                self.assertNotIn(literal, src, f"{name} contains {literal}")

    def test_model_py_has_no_transformers_or_occ_imports(self):
        src = self._source("model.py")
        self.assertNotIn("import transformers", src)
        self.assertNotIn("from transformers", src)
        for name in ("model.py", "text_embed.py", "loader.py"):
            for line in self._source(name).splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    self.assertFalse(
                        stripped.startswith("import OCC") or stripped.startswith("from OCC"),
                        f"{name}: OCC import line: {line}",
                    )


# ---------------------------------------------------------------------------
# Integration with the stage-A scorer (random weights)
# ---------------------------------------------------------------------------


class TestScorerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = build_text2cad(DeterministicStubEmbedder(), device="cpu")
        cls.model.eval()
        cls.cad_vec, cls.flag_vec, cls.index_vec, cls.kpm = make_circle_like_candidate()

    def test_score_plumbing_and_valid_counts(self):
        vec_dict, mask_dict = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        with torch.no_grad():
            pred, target, shifted = teacher_forced_logits(
                self.model, ["draw a circle plate"], vec_dict, mask_dict
            )
        results = two_head_factorized_score(pred, self.cad_vec.unsqueeze(0), self.kpm.unsqueeze(0))
        self.assertEqual(len(results), 1)
        result = results[0]
        diag = validate_cad_vec(self.cad_vec, self.kpm, self.flag_vec, self.index_vec)
        self.assertEqual(result["n_valid_x"], diag.n_valid_x)
        self.assertEqual(result["n_valid_y"], diag.n_valid_y)
        n_real = int((~self.kpm[:, 0]).sum())
        self.assertEqual(result["n_valid_x"], n_real - 1)
        self.assertEqual(result["n_valid_y"], 2)  # the two coordinate pairs
        self.assertTrue(math.isfinite(result["score_total"]))
        self.assertLess(result["score_total"], 0.0)

    def test_batch_consistency(self):
        # (textA, cand) scored alone vs the same pair inside a mixed batch:
        # text padding is fully masked, so the pair's score must agree.
        text_a = "draw a circle plate"
        text_b = "extrude a very different and longer shape description!!"
        vec_single, mask_single = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        with torch.no_grad():
            pred_single, _, _ = teacher_forced_logits(self.model, [text_a], vec_single, mask_single)
        other_vec, other_flag, other_index, other_kpm = make_circle_like_candidate()
        vec_batch = {
            "cad_vec": torch.stack([self.cad_vec, other_vec]),
            "flag_vec": torch.stack([self.flag_vec, other_flag]),
            "index_vec": torch.stack([self.index_vec, other_index]),
        }
        mask_batch = {
            "attn_mask": mask_single["attn_mask"],
            "key_padding_mask": torch.stack([self.kpm, other_kpm]),
        }
        with torch.no_grad():
            pred_batch, _, _ = teacher_forced_logits(
                self.model, [text_a, text_b], vec_batch, mask_batch
            )
        score_single = two_head_factorized_score(
            pred_single, self.cad_vec.unsqueeze(0), self.kpm.unsqueeze(0)
        )[0]
        score_batched = two_head_factorized_score(
            pred_batch, torch.stack([self.cad_vec, other_vec]), torch.stack([self.kpm, other_kpm])
        )[0]
        self.assertEqual(score_single["n_valid_total"], score_batched["n_valid_total"])
        self.assertTrue(
            torch.allclose(pred_single[0], pred_batch[0], atol=1e-4, rtol=1e-4),
            float((pred_single[0] - pred_batch[0]).abs().max()),
        )
        self.assertAlmostEqual(
            score_single["score_total"], score_batched["score_total"], places=3
        )

    def test_query_changes_score(self):
        vec_dict, mask_dict = make_batch(self.cad_vec, self.flag_vec, self.index_vec, self.kpm)
        with torch.no_grad():
            pred_a, _, _ = teacher_forced_logits(self.model, ["draw a circle plate"], vec_dict, mask_dict)
            pred_b, _, _ = teacher_forced_logits(self.model, ["mill a rectangular pocket"], vec_dict, mask_dict)
        score_a = two_head_factorized_score(pred_a, self.cad_vec.unsqueeze(0))[0]["score_total"]
        score_b = two_head_factorized_score(pred_b, self.cad_vec.unsqueeze(0))[0]["score_total"]
        self.assertGreater(abs(score_a - score_b), 1e-6)


if __name__ == "__main__":
    unittest.main()
