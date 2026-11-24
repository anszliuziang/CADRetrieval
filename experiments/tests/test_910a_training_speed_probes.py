import pytest

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
