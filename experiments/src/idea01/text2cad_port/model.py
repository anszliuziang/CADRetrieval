"""Occ-free, explicit-device port of the frozen Text2CAD forward architecture.

Upstream source (read-only mirror, Git SHA 669a81472ede2269837c22a4d00070eaac360c82,
license CC BY-NC-SA 4.0 -- attribution: Mohammad Sadil Khan et al., Text2CAD,
NeurIPS 2024, DFKI GmbH):

  * ``CadSeqProc/utility/macro.py``           -> constants (CAD_CLASS_INFO etc.)
  * ``CadSeqProc/utility/utils.py``           -> :func:`generate_attention_mask`
  * ``Cad_VLM/models/layers/text_embed.py``   -> :func:`prepare_cross_attention_mask_batch`
  * ``Cad_VLM/models/layers/embedder.py``     -> PositionalEncodingSinCos / Embedder /
                                                 CADSequenceEmbedder
  * ``Cad_VLM/models/layers/functional.py``   -> multi_head_attention_forward /
                                                 FeedForwardLayer
  * ``Cad_VLM/models/layers/attention.py``    -> MultiHeadAttention / CrossAttention
  * ``Cad_VLM/models/decoder.py``             -> CADDecoder / CADDecoderLayer
  * ``Cad_VLM/models/layers/adaptive_layer.py`` -> TextAdaptiveLayer
  * ``Cad_VLM/models/text2cad.py``            -> Text2CAD (teacher-forcing forward only)
  * ``Cad_VLM/train.py``                      -> teacher-forcing input slicing
  * ``Cad_VLM/config/inference.yaml``         -> OFFICIAL_INFERENCE_CONFIG values

This file is an adaptation for research evaluation and remains under
CC BY-NC-SA 4.0. Every deviation from the upstream files is enumerated in
``NOTICE.md`` next to this file. In brief, and only here:

  * the top-level OCC-dependent import chains (``CadSeqProc.utility.utils``,
    ``Cad_VLM.models.layers.utils_decode``, ``Cad_VLM.models.utils``) are NOT
    imported; the three helper functions actually needed on the teacher-forcing
    path are copied verbatim into this module;
  * every upstream device selection (CUDA-availability checks and hard-coded
    CUDA string literals) is replaced by an explicit ``device`` argument;
    no CUDA literal remains;
  * autoregressive generation (``CADDecoder.decode``, ``Text2CAD.test_decode``)
    and parameter pretty-printing are out of scope and not ported;
  * class names, attribute names and forward mathematics are unchanged so the
    172 non-BERT checkpoint parameter paths (``adaptive_layer.*``,
    ``cad_decoder.*``) match exactly.

Scope restrictions (M1c stage B1): this module performs no file I/O, loads no
weights, and never sees qrels/UID mappings. Running the *real* checkpoint is
NOT authorized here; tests use small random weights on CPU only.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter

#: Git SHA of the pinned upstream mirror this port was made from.
OFFICIAL_SOURCE_GIT_SHA = "669a81472ede2269837c22a4d00070eaac360c82"

# ---------------------------------------------------------------------------
# Constants (CadSeqProc/utility/macro.py @ OFFICIAL_SOURCE_GIT_SHA)
# ---------------------------------------------------------------------------

#: macro.py:14 -- padded CAD sequence length used by all official assets.
MAX_CAD_SEQUENCE_LENGTH = 272

#: macro.py:40-44 -- vocabulary / embedding table sizes.
#: one_hot_size = END_PAD(7) + BOOLEAN_PAD(4) + 2**N_BIT(8) = 267
#: index_size   = MAX_EXTRUSION(10) + 1 = 11
#: flag_size    = ONE_EXT_SEQ_LENGTH(10) + 2 = 12
CAD_CLASS_INFO = {"one_hot_size": 267, "index_size": 11, "flag_size": 12}

#: Values of ``Cad_VLM/config/inference.yaml`` @ OFFICIAL_SOURCE_GIT_SHA.
#: ``cad_seq_len`` is not in the yaml; train.py:80 / test.py:76 /
#: test_user_input.py:76 inject ``MAX_CAD_SEQUENCE_LENGTH`` before building the
#: model, which this dict mirrors. ``text_embedder.cache_dir`` ("YOUR DIR") is
#: dropped: the adapted TextEmbedder loads from a pinned local directory
#: (see text_embed.py and NOTICE.md).
OFFICIAL_INFERENCE_CONFIG = {
    "text_encoder": {
        "text_embedder": {"model_name": "bert_large_uncased", "max_seq_len": 512},
        "adaptive_layer": {"in_dim": 1024, "out_dim": 1024, "num_heads": 8, "dropout": 0.1},
    },
    "cad_decoder": {
        "tdim": 1024,
        "cdim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.1,
        "ca_level_start": 2,
        "cad_seq_len": MAX_CAD_SEQUENCE_LENGTH,
    },
}


class PortInputError(ValueError):
    """Input contract violation inside the ported forward path.

    ``reason`` is a stable machine-readable string so callers can ledger
    rejections without parsing messages.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def generate_attention_mask(
    sz1: int, sz2: int = None, device="cpu", mask_start_token=True
):
    """Copied verbatim from CadSeqProc/utility/utils.py:1790-1803.

    Upper-triangular causal mask (0 / -inf) where the START column is
    additionally hidden from every later position (``mask[1:, 0] = -inf``).
    This is exactly the mask json2vec.py:261 precomputes into every official
    ``mask_cad_dict["attn_mask"]``; it is re-implemented here so tests and
    synthetic inputs do not need the OCC-importing upstream utils module.
    """
    if sz2 is None:
        sz2 = sz1
    mask = torch.triu(torch.full((sz1, sz2), float("-inf"), device=device), diagonal=1)

    # Add Mask for start token
    if mask_start_token:
        mask[1:, 0] = float("-inf")
    return mask


def prepare_cross_attention_mask_batch(mask, cad_seq_len=271):
    """Copied verbatim from Cad_VLM/models/layers/text_embed.py:8-18.

    Turns the BERT padding mask (B, L_text), True = PAD, into an additive
    (B, cad_seq_len, L_text) float mask with 0 / -inf entries. This additive
    mask is the ONLY padding suppression inside the official cross-attention
    (CrossAttention.forward does not forward its key_padding_mask argument);
    keep it bit-exact.
    """
    if mask.shape[0] > 1:
        length = mask.shape[1]
        batch_size = mask.shape[0]
        mask = mask.reshape(batch_size, 1, length)
    mask = torch.tile(mask, (1, cad_seq_len, 1))  # (512) -> (271, 512)
    mask = torch.where(
        mask, -torch.inf, 0
    )  # Changing the [True,False] format to [0,-inf] format

    return mask


# ---------------------------------------------------------------------------
# Embedders (Cad_VLM/models/layers/embedder.py)
# ---------------------------------------------------------------------------


class PositionalEncodingSinCos(nn.Module):
    """embedder.py:11-33. Deterministic sin/cos table; ``device`` stays an
    explicit constructor argument exactly as upstream (the only change is that
    callers must now pass a real device -- upstream's *callers* picked one via
    a CUDA-availability check, see NOTICE.md)."""

    def __init__(self, embedding_size: int, max_seq_len: int, device: str):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        # create a matrix of shape (max_seq_len, embedding_size/2)
        self.pos_enc = torch.zeros(max_seq_len, embedding_size)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.pos_enc = self.pos_enc.unsqueeze(0)

    def forward(self, seq_len):
        # x has shape (batch_size, seq_len, embedding_size)
        # add positional encoding to x
        x = self.pos_enc[:, :seq_len, :]
        return x.to(self.device)


class Embedder(nn.Module):
    """embedder.py:58-64, verbatim."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class CADSequenceEmbedder(nn.Module):
    """embedder.py:67-103, verbatim (one commented-out dead line removed).

    Token embedding = sf(flag) + si(index) + cx(x) + cy(y); x/y values at
    PADDING positions are zeroed via ``~key_padding_mask`` (a redundant
    identity for official data, where padded values are already 0 -- kept for
    fidelity). ``self.device`` is stored but, as upstream, unused in forward.
    """

    def __init__(self, one_hot_size, flag_size, index_size, d_model, device="cpu"):
        super(CADSequenceEmbedder, self).__init__()

        self.si = Embedder(index_size, d_model)  # CAD Sequence Index Encoding
        self.sf = Embedder(flag_size, d_model)  # CAD Sequence Flag Encoding
        self.cx = Embedder(one_hot_size, d_model)  # x-coordinate embedding
        self.cy = Embedder(one_hot_size, d_model)  # y-coordinate embedding
        self.device = torch.device(device)

    def forward(self, vec_dict, key_padding_mask):
        """
        vec_dict: contains key "cad_vec","flag_vec" and "index_vec"
        key_padding_mask: Tensor. Shape (N,2). Must be same with cad_vec
        """
        x_seq = vec_dict["cad_vec"][:, :, 0] * (~key_padding_mask[:, :, 0] * 1)
        y_seq = vec_dict["cad_vec"][:, :, 1] * (~key_padding_mask[:, :, 1] * 1)

        return (
            self.sf(vec_dict["flag_vec"])
            + self.si(vec_dict["index_vec"])
            + self.cx(x_seq)
            + self.cy(y_seq)
        )


# ---------------------------------------------------------------------------
# Attention (Cad_VLM/models/layers/functional.py + attention.py)
# ---------------------------------------------------------------------------


def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 # type: Optional[Tensor]
                                 bias_k,
                                 bias_v,
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 # type: Optional[Tensor]
                                 key_padding_mask=None,
                                 need_weights=True,               # type: bool
                                 # type: Optional[Tensor]
                                 attn_mask=None,
                                 use_separate_proj_weight=False,  # type: bool
                                 # type: Optional[Tensor]
                                 q_proj_weight=None,
                                 k_proj_weight=None,
                                 v_proj_weight=None,
                                 static_k=None,
                                 static_v=None
                                 ):
    # type: (...) -> tuple
    r"""Copied verbatim from Cad_VLM/models/layers/functional.py:9-283.

    (Itself a copy of the classic ``torch.nn.functional.multi_head_attention_forward``
    reference implementation.) The verbatim ``torch.equal(query, key)`` /
    ``torch.equal(key, value)`` branch selection is kept on purpose: the
    self-attention branch computes q/k/v with one fused GEMM while the
    encoder-decoder branch splits the projections, which is numerically
    distinguishable at the last-ulp level -- pruning the branches would risk
    changing the frozen model's math.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input proj. weights.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where N is the batch size, S is the source sequence length,
          E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)` where N is the batch size, S is the source sequence length,
          E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check, f"{embed_dim} is not same as {embed_dim_to_check}"
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight,
                               in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt,
                         in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt,
                         in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt,
                         in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError(
                    f'The size of the 2D attn_mask is not correct. Size is {attn_mask.shape}. Query size is {query.shape}, Key size is {key.shape}')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError(f'The size of the 3D attn_mask is not correct. Expected ({bsz * num_heads, query.size(0), key.size(0)}) but got {attn_mask.size()}')
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len, f"key padding mask {key_padding_mask.size(1)} size mismatch with src length {src_len}"

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()
                      [2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()
                      [2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [
        bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)

    #! Note: Adding dropout here changes the sum of the attention weights to 1.11, so force it 0.

    attn_output_weights = F.dropout(
        attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len)
        return attn_output.transpose(0, 1), attn_output_weights
    else:
        return attn_output.transpose(0, 1), None


class MultiHeadAttention(nn.Module):
    r"""Cad_VLM/models/layers/attention.py:15-218 (classic fairseq-style MHA).

    Ported unchanged except for dropping unused imports (``itemgetter``,
    ``layer_utils`` star-import). Forward keeps the upstream ``use_3d``
    convention: a >2D ``attn_mask`` is collapsed to its first batch entry
    unless ``use_3d=True`` (the cross-attention path relies on this).
    """

    __constants__ = [
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "in_proj_weight",
    ]

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiHeadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_3d=False,
    ):
        r"""
        Args:
            query: input Tensor of shape (batch_size,n_samples,n_features) for query generation
            key: input Tensor of shape (batch_size,n_samples,n_features) for key generation
            value: input Tensor of shape (batch_size,n_samples,n_features) for value generation
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        Shape:
            - Inputs:
            - query: :math:`(B, N, E)` where N is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key: :math:`(B, N, E)` where N is the target sequence length, B is the batch size, E is
            the embedding dimension.
            - vaue: :math:`(B, N, E)` where N is the target sequence length, B is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
            - attn_mask: 2D mask :math:`(N, S)` where N is the target sequence length, S is the source sequence length.
              3D mask :math:`(B*num_heads, N, S)` where B is the batch size, N is the target sequence length,
              S is the source sequence length.
            - Outputs:
            - attn_output: :math:`(N, B, E)` where N is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - attn_output_weights: :math:`(B, N, S)` where B is the batch size,
              N is the target sequence length, S is the source sequence length.
        """

        query = query.transpose(0, 1)  # (N,B,E)
        key = key.transpose(0, 1)  # (N,B,E)
        value = value.transpose(0, 1)  # (N,B,E)

        # Atttention mask is same for all the batches
        if attn_mask is not None and attn_mask.dim() > 2 and not use_3d:
            attn_mask = attn_mask[0]  # (N,S)

        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )


class CrossAttention(nn.Module):
    """Cad_VLM/models/layers/attention.py:220-272, verbatim.

    IMPORTANT (upstream behavior, preserved deliberately): ``forward`` accepts
    ``key_padding_mask`` but does NOT pass it to the inner MHA -- padding of
    the text context is suppressed exclusively by the additive ``attn_mask``
    produced by :func:`prepare_cross_attention_mask_batch`. "Fixing" this
    would change the frozen model's math. Note ``num_heads`` defaults to 1:
    the additive mask is (B, N1, L_text) and the 3D-mask shape check requires
    ``B * num_heads == B``.
    """

    def __init__(
        self,
        input_dim_list,
        output_dim=512,
        num_heads=1,
        query_name="pc",
        context_1_name="cad",
        dropout=0,
        block_level=1,
    ):
        super(CrossAttention, self).__init__()
        self.query_name = query_name
        self.context_1_name = context_1_name
        self.block_level = block_level

        self.mha = MultiHeadAttention(
            input_dim=input_dim_list[0],
            embed_dim=output_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

    def forward(self, X, Y, Z, attn_mask=None, key_padding_mask=None):
        """
        input:
        X: Input Embedding of shape. (B,N1,E)
        Y: Context Input Embedding of shape (B,N2,E)
        mask_dict: dictionary containing two keys "key_padding_mask" and "attn_mask"

        output:
        X: Input Embedding of shape. (B,N1,output_dim)
        cross_attention_scores: dict
        """
        attention_output, attention_scores = self.mha(
            X, Y, Z, attn_mask=attn_mask, use_3d=True
        )

        cross_attention_score = {
            f"{self.query_name}_{self.context_1_name}_{self.block_level}": attention_scores
        }

        return attention_output, cross_attention_score


class FeedForwardLayer(nn.Module):
    """Cad_VLM/models/layers/functional.py:286-309, verbatim.

    Args:
        input_dim: Input Embedding Dimension
        d_ff: Feedforward dimension. Default=2048
        dropout: float
    """

    def __init__(self, input_dim, d_ff=2048, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear_1 = nn.Linear(input_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, input_dim)

    def forward(self, x):
        """
        input:
            x: tensors of shape (B,N,input_dim)
        output:
            x: tensors of shape (B,N,input_dim)
        """
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# ---------------------------------------------------------------------------
# CAD decoder (Cad_VLM/models/decoder.py)
# ---------------------------------------------------------------------------


class CADDecoderLayer(nn.Module):
    """decoder.py:294-421, forward logic verbatim.

    Changes (see NOTICE.md): the ``device`` constructor argument is now
    required (no implicit CUDA default); the rich/prettytable helpers are not
    imported. ``norm_seq["norm_3"]`` and ``dp_seq["dropout_2"]`` are created
    for every layer but only *used* when ``use_ca`` is true -- this is exactly
    why the checkpoint's two all-zero tensors are the ``norm_3.bias`` of the
    two layers without cross-attention (see loader.EXPECTED_ZERO_TENSOR_KEYS).
    """

    def __init__(
        self,
        tdim=128,
        cdim=128,
        num_heads=8,
        block_level=0,
        dropout=0.1,
        use_ca=True,
        device=None,
    ):
        super(CADDecoderLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        #! Note: Dropout is set to 0 for attention otherwise the sum of attention weights > 1
        self.sa_seq = MultiHeadAttention(
            input_dim=cdim,
            embed_dim=cdim,
            dropout=0,
            num_heads=num_heads,
        )
        self.use_ca = use_ca

        if use_ca:
            # Cross Attention between discrete latent vectors and cad features
            self.ca_ze_seq = CrossAttention(
                input_dim_list=[cdim, tdim],
                output_dim=cdim,
                query_name="cad",
                context_1_name="vq",
                dropout=0,
                block_level=block_level,
            )

        # Text Embedding Downsampler
        self.downsampler = nn.Linear(tdim, cdim)

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {
                "norm_1": nn.LayerNorm(cdim),
                "norm_2": nn.LayerNorm(cdim),
                "norm_3": nn.LayerNorm(cdim),
            }
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {
                "dropout_1": nn.Dropout(dropout),
                "dropout_2": nn.Dropout(dropout),
                "dropout_3": nn.Dropout(dropout),
            }
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=cdim)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        S: Optional[torch.Tensor],
        ZE: Optional[torch.Tensor],
        mask_cad_dict: dict,
        cross_attn_mask_dict: dict = {"attn_mask": None, "key_padding_mask": None},
        metadata: bool = False,
    ):
        """
        S: tensor of shape (bs, num_seq, emb_dim)
        ZE: tensor of shape (bs, num_code, emd_dim)
        mask_cad_dict: dictionary with keys "attn_mask" and "key_padding_mask"
        cross_attn_mask_dict: dictionary with keys "attn_mask" and "key_padding_mask"
        metadata: boolean. To save attention weights
        """

        self_attn_mask_dict = mask_cad_dict.copy()
        self_attn_mask_dict["key_padding_mask"] = torch.all(
            mask_cad_dict["key_padding_mask"], axis=2
        )

        # ? <----------  CAD SEQUENCE SELF-ATTENTION  ---------->
        S2 = self.norm_seq["norm_1"](S)  # (bs,num_seq,emb_dim)
        S2, S_score = self.sa_seq(
            S2,
            S2,
            S2,
            key_padding_mask=self_attn_mask_dict["key_padding_mask"],
            attn_mask=self_attn_mask_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)

        # (bs,num_seq,emb_dim) (Dropout + Addition)
        S = S + self.dp_seq["dropout_1"](S2)
        S2 = self.norm_seq["norm_2"](S)  # (bs,num_seq,emb_dim) (Normalization)

        # ? <----------  CROSS-ATTENTION BETWEEN LATENT EMBEDDING AND CAD EMBEDDING  ---------->
        if self.use_ca:
            if hasattr(self, "downsampler"):
                ZE = self.downsampler(ZE)
            S3, ZE_S_score = self.ca_ze_seq(
                S2,
                ZE,
                ZE,
                key_padding_mask=cross_attn_mask_dict["key_padding_mask"],
                attn_mask=cross_attn_mask_dict["attn_mask"],
            )

            # ? <----------  (CROSS-ATTENDED FEATURES + SELF-ATTENDED FEATURES) + DROPOUT + NORMALIZATION LAYER  ---------->
            S = S + self.dp_seq["dropout_2"](S3)
            S2 = self.norm_seq["norm_3"](S)  # (bs,num_seq,emb_dim)
            self.attention_scores["ca"] = ZE_S_score

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION +    ---------->
        S = S + self.dp_seq["dropout_3"](self.ffl_seq(S2))

        # Add the cross attention scores (metadata)
        self.attention_scores["sa"] = S_score

        return S, self.attention_scores


class CADDecoder(nn.Module):
    """decoder.py:22-164, forward logic verbatim; autoregressive ``decode()``
    (decoder.py:166-282) is NOT ported -- teacher-forced scoring never calls
    it, and porting it would drag in the OCC-importing helpers
    ``create_flag_vec`` / ``create_index_vec`` / ``top_p_sampling``.

    Args:
        cad_class_info (dict, required): A dictionary containing information about CAD classes. Check `DataProc.CadSeqProc.utility.macro`
        tdim (int, required): Dimensionality of the continuous latent space.
        cdim (int, required): Dimensionality of the cad embedding.
        num_layers (int, required): Number of transformer layers.
        num_heads (int, required): Number of attention heads in the transformer.
        dropout (float, required): Dropout probability.
        ca_level_start (int, required): The starting level of class attention layers.
        device (str, required): Device to run the model on, e.g., "npu:0" or "cpu".
            MUST be passed explicitly (upstream's from_config defaulted to the
            CUDA device whenever one was available; see NOTICE.md).
    """

    def __init__(
        self,
        cad_class_info: dict,
        tdim: int,
        cdim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ca_level_start: int,
        device: str,
    ):
        super(CADDecoder, self).__init__()

        self.num_layers = num_layers

        # For Initial Sequence Embedding
        self.seq_embed = CADSequenceEmbedder(
            one_hot_size=cad_class_info["one_hot_size"],
            flag_size=cad_class_info["flag_size"],
            index_size=cad_class_info["index_size"],
            d_model=cdim,
            device=device,
        )

        # Positional Encoding
        self.pe = PositionalEncodingSinCos(
            embedding_size=cdim, max_seq_len=MAX_CAD_SEQUENCE_LENGTH, device=device
        )

        # List of booleans indicating which decoder layers support cross-attention
        self.use_ca = [False] * ca_level_start + [True] * (num_layers - ca_level_start)

        # A stack of num_layers x Decoder Layer
        # (upstream builds these with copy.deepcopy of a fresh layer; direct
        # per-layer construction is structurally identical -- see NOTICE.md)
        self.cad_decoder_layers = nn.ModuleList(
            [
                CADDecoderLayer(
                    tdim=tdim,
                    cdim=cdim,
                    num_heads=num_heads,
                    block_level=i,
                    dropout=dropout,
                    use_ca=self.use_ca[i],
                    device=device,
                )
                for i in range(self.num_layers)
            ]
        )

        # Out Layer for Sequence Prediction
        self.seq_output_x = nn.Sequential(
            nn.Linear(in_features=cdim, out_features=cad_class_info["one_hot_size"])
        )
        self.seq_output_y = nn.Sequential(
            nn.Linear(in_features=cdim, out_features=cad_class_info["one_hot_size"])
        )

        # Metadata
        self.attention_scores = dict()

    def forward(
        self,
        vec_dict: dict,
        ZE: Optional[torch.Tensor],
        mask_cad_dict: dict,
        cross_attn_mask_dict: dict = {"attn_mask": None, "key_padding_mask": None},
        metadata: bool = False,
    ):
        """
        vec_dict: dictionary with keys "cad_vec", "flag_vec", "index_vec"
        ZE: tensor of shape (batch, num_code, emd_dim). Context
        mask_cad_dict: dictionary with keys "key_padding_mask" and "attn_mask". SELF ATTENTION MASK
        cross_attn_mask_dict: dictionary with keys "attn_mask" and "key_padding_mask". CROSS ATTENTION MASK
        metadata: boolean indicating whether attention weights are saved. Turn off during training
        """
        num_seq = vec_dict["cad_vec"].shape[1]
        # Token Embedding and positional encoding
        S = self.pe(num_seq) + self.seq_embed(
            vec_dict, mask_cad_dict["key_padding_mask"]
        )  # (B,N1,cdim)
        # Pass through Decoder Layers
        for i in range(self.num_layers):
            S, self.attention_scores[f"block_level_{i}"] = self.cad_decoder_layers[i](
                S,
                ZE=ZE,
                mask_cad_dict=mask_cad_dict,
                cross_attn_mask_dict=cross_attn_mask_dict,
                metadata=metadata,
            )
        Sx = self.seq_output_x(S).unsqueeze(dim=2)  # (B,1,N1,one_hot_size)
        Sy = self.seq_output_y(S).unsqueeze(dim=2)  # (B,N1,1,one_hot_size)

        S = torch.cat([Sx, Sy], dim=2)  # (B,N1,2*one_hot_size

        # Save the metadata
        if metadata:
            self.metadata = {"attention_scores": self.attention_scores}
            return S, self.metadata
        else:
            return S, None

    @staticmethod
    def from_config(config, device):
        """Initialize the CADDecoder class from a config file.

        Change vs decoder.py:151-164: ``device`` is a required explicit
        argument instead of upstream's CUDA-availability fallback.
        """
        cad_decoder = CADDecoder(
            cad_class_info=CAD_CLASS_INFO,
            tdim=config["tdim"],
            cdim=config["cdim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            ca_level_start=config["ca_level_start"],
            device=device,
        )
        return cad_decoder


# ---------------------------------------------------------------------------
# Text adaptive layer (Cad_VLM/models/layers/adaptive_layer.py)
# ---------------------------------------------------------------------------


class TextAdaptiveLayer(nn.Module):
    """adaptive_layer.py:18-109, forward logic verbatim.

    Changes (NOTICE.md): the top-level imports of ``utils_decode`` (OCC chain;
    only used in upstream's ``__main__`` demo) and ``models.utils``
    (prettytable parameter printer) are dropped; ``total_parameters`` is not
    ported. With the official config (in_dim == out_dim == 1024) no
    ``downsampler`` submodule exists -- the checkpoint's ``adaptive_layer.*``
    keys confirm this.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super(TextAdaptiveLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        self.sa_seq = MultiHeadAttention(
            input_dim=in_dim,
            embed_dim=in_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {"norm_1": nn.LayerNorm(in_dim), "norm_2": nn.LayerNorm(in_dim)}
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {"dropout_1": nn.Dropout(dropout), "dropout_2": nn.Dropout(dropout)}
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=in_dim)

        if in_dim != out_dim:
            # Downsampler
            self.downsampler = nn.Linear(in_dim, out_dim)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        T: Optional[torch.Tensor],
        mask_prompt_dict: dict,
        metadata: bool = False,
    ):
        """
        T: tensor of shape (bs, num_seq, emb_dim). Text Embedding
        mask_prompt_dict: dictionary with keys "attn_mask", "key_padding_mask"
        metadata: boolean. To save attention weights
        """

        # ? <----------  TEXT EMBEDDING SELF-ATTENTION  ---------->
        T2 = self.norm_seq["norm_1"](T)  # (bs,num_seq,emb_dim)
        T2, T_score = self.sa_seq(
            T2,
            T2,
            T2,
            key_padding_mask=mask_prompt_dict["key_padding_mask"],
            attn_mask=mask_prompt_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)

        # (bs,num_seq,emb_dim) (Dropout + Addition)
        T = T + self.dp_seq["dropout_1"](T2)
        T2 = self.norm_seq["norm_2"](T)  # (bs,num_seq,emb_dim) (Normalization)

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION + DOWN-SAMPLER    ---------->
        T = T + self.dp_seq["dropout_2"](self.ffl_seq(T2))

        if hasattr(self, "downsampler"):
            T = self.downsampler(T)

        if metadata:
            # Add the cross attention scores (metadata)
            self.attention_scores["text_sattn"] = T_score
            return T, self.attention_scores
        else:
            return T, None

    @staticmethod
    def from_config(config):
        return TextAdaptiveLayer(**config)


# ---------------------------------------------------------------------------
# Full model shell (Cad_VLM/models/text2cad.py) + teacher-forcing glue
# ---------------------------------------------------------------------------


class Text2CAD(nn.Module):
    """text2cad.py:14-127, teacher-forcing forward verbatim.

    Changes (NOTICE.md): the constructor receives pre-built submodules
    (composition root moved to the loader) instead of building a
    hub-downloading TextEmbedder from a config dict; attribute names
    (``base_text_embedder`` / ``adaptive_layer`` / ``cad_decoder``) and the
    forward body are unchanged, so state_dict paths are identical to
    upstream's. ``test_decode`` (autoregressive generation) is out of scope
    and not ported. ``cad_seq_len`` follows text2cad.py:31 (config value - 1).
    """

    def __init__(self, base_text_embedder, adaptive_layer, cad_decoder, cad_seq_len: int):
        super().__init__()

        # Base Text Embedder
        self.base_text_embedder = base_text_embedder
        # Adaptive Layer to fine tune text embeddings
        self.adaptive_layer = adaptive_layer

        # Transformer Decoder for CAD sequence generation
        self.cad_decoder = cad_decoder

        self.cad_seq_len = cad_seq_len - 1

        self.attention_scores = dict()

    def forward(
        self,
        vec_dict: dict,
        texts: list,
        mask_cad_dict: dict,
        metadata: bool = False,
    ):
        """
        vec_dict: dict contains cad_vec, flag_vec, and index_vec
        texts: list of text prompts
        mask_cad_dict: dict contains attention mask and key_padding_mask for CAD sequence
        metadata: bool to return attention scores
        """
        # ------------ Get the initial text embeddings ------------ #

        T, key_padding_mask = self.base_text_embedder.get_embedding(texts)
        ca_mask = {"attn_mask": prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=self.cad_seq_len),
                   "key_padding_mask": key_padding_mask}

        # ------------ Pass the text embedding through the adaptive layer ------------ #
        T, text_attn_scores = self.adaptive_layer(
            T,
            {
                "attn_mask": None,
                "key_padding_mask": ca_mask["key_padding_mask"],
            },
            metadata,
        )
        if text_attn_scores is not None:
            self.attention_scores.update(text_attn_scores)

        # ------------ Pass the text embedding through the CAD Decoder as context ------------ #
        S_output, cad_attn_scores = self.cad_decoder(
            vec_dict, T, mask_cad_dict, ca_mask, metadata
        )
        if cad_attn_scores is not None:
            self.attention_scores.update(cad_attn_scores)

        if metadata:
            return S_output, self.attention_scores
        else:
            return S_output, None

    def get_trainable_state_dict(self):
        # Get the state dict of the model which are trainable parameters
        return {
            k: v for k, v in self.state_dict().items() if "base_text_embedder" not in k.split(".")
        }


def build_teacher_forcing_inputs(vec_dict: dict, mask_cad_dict: dict):
    """Replicate the official teacher-forcing slicing (train.py:257-270)
    WITHOUT mutating the caller's dicts.

    Returns ``(inputs_vec_dict, inputs_mask_cad_dict, cad_vec_target,
    shifted_key_padding_mask)`` where

      * ``inputs_vec_dict[k] = vec_dict[k][:, :-1]`` for every key
        (cad_vec / flag_vec / index_vec sliced alike, train.py:264-265);
      * ``inputs_mask_cad_dict`` keeps ``attn_mask`` untouched (train.py never
        slices it -- it is the precomputed 271x271 mask) and slices
        ``key_padding_mask[:, :-1]`` (train.py:268-270);
      * ``cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()``
        (train.py:259);
      * ``shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]``
        (train.py:257) -- the mask of the TARGET positions, handed to the
        loss inverted upstream and consumed as ``target != 0`` by the stage-A
        scorer.

    Sliced tensors are views (as upstream's in-place rebinding produced); the
    target is a clone exactly as upstream.
    """
    shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]
    cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()
    inputs_vec_dict = {key: value[:, :-1] for key, value in vec_dict.items()}
    inputs_mask_cad_dict = dict(mask_cad_dict)
    inputs_mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][:, :-1]
    return inputs_vec_dict, inputs_mask_cad_dict, cad_vec_target, shifted_key_padding_mask


def teacher_forced_logits(model: Text2CAD, texts: list, vec_dict: dict, mask_cad_dict: dict):
    """Run one official teacher-forcing forward and return the raw logits.

    Mirrors the model invocation of train.py:273-278 (``metadata=False``);
    the forward itself is NOT wrapped in no_grad here (upstream wasn't) --
    the evaluation runner wraps it. Returns ``(pred, cad_vec_target,
    shifted_key_padding_mask)`` with ``pred`` of shape (B, N-1, 2, 267), the
    exact tensor the stage-A scorer consumes. This function does no I/O, no
    pairing and no scoring loops; the pair runner belongs to stage B2.
    """
    inputs_vec_dict, inputs_mask_cad_dict, cad_vec_target, shifted_key_padding_mask = (
        build_teacher_forcing_inputs(vec_dict, mask_cad_dict)
    )
    pred, _ = model(
        vec_dict=inputs_vec_dict,
        texts=texts,
        mask_cad_dict=inputs_mask_cad_dict,
        metadata=False,
    )
    return pred, cad_vec_target, shifted_key_padding_mask
