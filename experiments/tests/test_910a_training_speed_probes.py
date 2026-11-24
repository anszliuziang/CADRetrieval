import pytest
import torch
from transformers import BertTokenizer

from experiments.scripts import run_formal_text2cad_training as native
from experiments.scripts import run_in_domain_pilot_retrievers as retrieval


@pytest.mark.parametrize("multiple", [None, 32, 64])
def test_text_width_probe_only_changes_right_padding_request(multiple):
    calls = []

    def tokenizer(texts, **options):
        calls.append((texts, options))
        return object()

    native.encode_training_texts(tokenizer, ["short", "long"], multiple)
    expected = dict(return_tensors="pt", padding=True, truncation=True, max_length=512)
    if multiple is not None:
        expected["pad_to_multiple_of"] = multiple
    assert calls == [(["short", "long"], expected)]


def test_spectra_probe_counts_real_pairs_in_the_prefix():
    batches = [(0, 16), (16, 32), (32, 46)]
    assert retrieval.spectra_timing_prefix(batches, None) == batches
    assert retrieval.spectra_timing_prefix(batches, 2) == batches[:2]
    with pytest.raises(ValueError, match="strict prefix"):
        retrieval.spectra_timing_prefix(batches, 3)


def test_epoch_end_cursor_revalidates_after_resume_at_the_update_limit():
    pending = dict(completed=False, updates=155, next_batch=155)
    assert native.continue_training(pending, 155, 155)
    assert not native.continue_training(dict(pending, next_batch=0), 155, 155)
    assert not native.continue_training(dict(pending, completed=True), 155, 155)


def test_validation_padding_matches_native_effective_tokens_and_rejects_long_text(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab))

    class TokenEcho(torch.nn.Module):
        def forward(self, input_ids, **unused):
            return (input_ids.unsqueeze(-1).float(),)

    embedder = native.TrainingEmbedder(512, "cpu", TokenEcho(), tokenizer)
    texts = ["a b a b", "b"]
    original, original_pad = embedder.get_embedding(texts)
    embedder.validation_pad_multiple = 64
    padded, padded_mask = embedder.get_embedding(iter(texts))
    assert padded.shape[1] == 64
    assert torch.equal(padded[~padded_mask], original[~original_pad])
    with pytest.raises(ValueError, match="complete texts"):
        embedder.get_embedding(["a " * 512])


def test_actual_tokenizer_keeps_effective_ids_and_truncation(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    texts = ["a b a b", "b"]
    natural = native.encode_training_texts(tokenizer, texts, None)
    for multiple in (32, 64):
        fixed = native.encode_training_texts(tokenizer, texts, multiple)
        assert fixed["input_ids"].shape[1] == multiple
        assert torch.equal(fixed["attention_mask"].sum(1), natural["attention_mask"].sum(1))
        assert torch.equal(fixed["input_ids"][fixed["attention_mask"].bool()],
                           natural["input_ids"][natural["attention_mask"].bool()])
        long_text = ["a " * 530, "b"]
        truncated = native.encode_training_texts(tokenizer, long_text, multiple)
        assert truncated["input_ids"].shape[1] == 512
        assert truncated["input_ids"][0, -1] == tokenizer.sep_token_id


def test_distributed_spectra_timing_keeps_global_batch_and_full_gate():
    assert retrieval.valid_formal_spectra_layout(None, 1, None)
    assert retrieval.valid_formal_spectra_layout(16, 1, None)
    assert not retrieval.valid_formal_spectra_layout(4, 4, None)
    assert retrieval.valid_formal_spectra_layout(4, 4, 4)
    assert not retrieval.valid_formal_spectra_layout(16, 4, 4)
    assert not retrieval.valid_formal_spectra_layout(None, 4, 4)
