"""Adapted Text2CAD text embedder (explicit device, pinned local BERT, no silent truncation).

Upstream source: ``Cad_VLM/models/layers/text_embed.py`` @ Git SHA
669a81472ede2269837c22a4d00070eaac360c82, license CC BY-NC-SA 4.0
(attribution: Mohammad Sadil Khan et al., Text2CAD, NeurIPS 2024, DFKI GmbH).
This file is an adaptation for research evaluation and remains under
CC BY-NC-SA 4.0; every deviation is enumerated in ``NOTICE.md``.

Deviations from upstream (summary -- line-level list in NOTICE.md):

  * ``BertTokenizer`` / ``BertModel`` load from a pinned LOCAL directory with
    ``local_files_only=True`` instead of the hub id ``google-bert/bert-large-uncased``
    plus ``cache_dir`` (text_embed.py:21-30);
  * the device is an explicit constructor argument: upstream chose the model
    device via a CUDA-availability check (text_embed.py:24) and moved the
    token batch with a hard-coded CUDA string literal (text_embed.py:42),
    which crashes both on CPU-only hosts and on Ascend 910 (torch_npu) hosts;
  * ``truncation=True`` (silent truncation at ``max_seq_len``) is replaced by
    ``truncation=False`` plus an explicit per-input length gate that REJECTS
    any text with more than ``max_seq_len`` BERT tokens. For every accepted
    input the produced token tensors are identical to upstream's
    (same tokenizer, same special tokens, same right padding; upstream's
    truncation simply never triggered at or below the limit). The task
    contract forbids silent truncation (ACTIVE_TASK stop condition);
  * the redundant ``max_position_embeddings=max_seq_len`` kwarg
    (text_embed.py:28-29) is dropped -- it equals the pinned config's default
    (512) and the strict loader asserts that config value instead;
  * the loaded BERT is switched to ``eval()`` and frozen
    (``requires_grad_(False)``), matching train.py:86-87's freeze for the
    forward pass; forward math is unchanged.

Scope restrictions (M1c stage B1): importing this module does NOT import
``transformers`` (the import happens lazily inside :meth:`from_local`), so the
torch-only test suite runs without it. ``get_embedding`` performs the BERT
forward; tests inject small fakes. Running the REAL BERT forward is not
authorized in this stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import PortInputError

#: Rejection reason for the >max_seq_len token gate (stable string, asserted
#: in tests and ledgers).
REASON_TEXT_TOO_LONG = "text_too_long"
REASON_NO_TEXT_MODEL = "text_model_not_loaded"

#: Kept from text_embed.py:6 for reference; the adapted loader never resolves
#: hub ids -- it always receives a pinned local directory.
MODEL_NAME_DICT = {"bert_large_uncased": "google-bert/bert-large-uncased"}


class TextEmbedder(nn.Module):
    """Port of text_embed.py:20-56 with the deviations listed in the module docstring.

    The submodule attribute names (``model`` for the BertModel, ``tokenizer``
    for the BertTokenizer) are unchanged so that, had BERT been part of the
    checkpoint, parameter paths would match upstream's
    ``base_text_embedder.model.*`` (it is not -- the checkpoint carries no BERT
    tensors; BERT loads separately from the pinned base weights).
    """

    def __init__(self, max_seq_len: int, device, model=None, tokenizer=None):
        super(TextEmbedder, self).__init__()
        self.max_seq_len = int(max_seq_len)
        self.device = torch.device(device)
        self.model = model
        self.tokenizer = tokenizer
        #: populated by from_local(); gated by the strict loader
        #: (missing/mismatched/error keys empty, unexpected keys restricted to
        #: the unused ``cls.*`` MLM head).
        self.loading_info = None

    @classmethod
    def from_local(cls, bert_dir, max_seq_len: int, device):
        """Build from a pinned local BERT directory (no network, no hub id)."""
        from transformers import BertModel, BertTokenizer  # lazy: keeps the
        # torch-only architecture importable without transformers installed.

        tokenizer = BertTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
        model, loading_info = BertModel.from_pretrained(
            str(bert_dir), local_files_only=True, output_loading_info=True
        )
        model = model.to(torch.device(device))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        instance = cls(max_seq_len=max_seq_len, device=device, model=model, tokenizer=tokenizer)
        instance.loading_info = loading_info
        return instance

    def get_embedding(self, texts: list):
        """text_embed.py:32-50 with explicit device and the length gate.

        Returns ``(embedding, key_padding_mask)``: the full-sequence BERT
        ``last_hidden_state`` (NOT the [CLS] pooler) and the boolean padding
        mask (True = PAD), exactly as upstream.
        """
        if isinstance(texts, str):
            texts = [texts]
        if self.model is None or self.tokenizer is None:
            raise PortInputError(
                REASON_NO_TEXT_MODEL,
                "TextEmbedder has no BERT model/tokenizer; use from_local() "
                "or inject test doubles",
            )
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=False,  # never silently truncate; gate below rejects
        )
        # attention_mask sums are the true per-input token counts (including
        # [CLS]/[SEP]), independent of right padding.
        lengths = encoded["attention_mask"].sum(dim=1)
        over = [
            (index, int(count))
            for index, count in enumerate(lengths.tolist())
            if int(count) > self.max_seq_len
        ]
        if over:
            raise PortInputError(
                REASON_TEXT_TOO_LONG,
                f"{len(over)} text(s) exceed {self.max_seq_len} BERT tokens "
                f"(index, token_count): {over[:5]}; refusing to truncate silently",
            )
        inputs = encoded.to(self.device)
        with torch.no_grad():
            all_output = self.model(**inputs)

            embedding = all_output[0]
            key_padding_mask = inputs["attention_mask"] == 0

        return embedding, key_padding_mask
