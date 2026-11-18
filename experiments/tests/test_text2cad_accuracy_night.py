"""Regressions for the reviewed Text2CAD accuracy-night execution paths."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load_script("run_text2cad_accuracy_night")
evaluate = load_script("eval_text2cad_accuracy_night")


def test_sorted_groups_keep_their_own_text():
    groups = {"first text": [1, 2], "other text": [3], "third text": [4]}
    selected = score.sort_groups_by_salt(groups, "test\n")
    assert dict(selected) == groups
    assert [score.salted_rank_key("test\n", text) for text, _ in selected] == sorted(
        score.salted_rank_key("test\n", text) for text in groups
    )


@pytest.mark.parametrize("batch_size", [1, 14, 16])
def test_one_forward_preserves_native_sum_and_all_field_groups(monkeypatch, batch_size):
    from idea01.text2cad_frozen_runner import generate_attention_mask
    from idea01.text2cad_port import model as port
    from idea01.text2cad_vector_scorer import two_head_factorized_score

    vec = torch.zeros((272, 2), dtype=torch.int32)
    vec[:6] = torch.tensor([[1, 0], [11, 12], [13, 0], [7, 0], [2, 0], [1, 0]])
    flag = torch.zeros(272, dtype=torch.int32)
    flag[6:] = 11
    index = torch.zeros(272, dtype=torch.int32)
    index[6:] = 1
    sample = {"cad_vec": vec, "flag_vec": flag, "index_vec": index,
              "key_padding_mask": vec == 0, "n_real_tokens": 6}
    store = SimpleNamespace(canonical_attn_mask=generate_attention_mask(271),
                            get=lambda _index: sample)
    layout = score.build_gallery_group_layout(store, batch_size)
    logits = torch.randn(batch_size, 271, 2, 267, generator=torch.Generator().manual_seed(17))
    called = []

    def one_forward(*args, **kwargs):
        called.append((len(args[1]), kwargs["cad_seq_len"]))
        return logits, None, None

    monkeypatch.setattr(port, "teacher_forced_logits", one_forward)
    total, groups = score.grouped_batch_scores(
        None, store, "toy", range(batch_size), torch.device("cpu"), layout,
    )
    original = two_head_factorized_score(
        logits, vec.repeat(batch_size, 1, 1), (vec == 0).repeat(batch_size, 1, 1)
    )
    assert called == [(batch_size, 271)]
    np.testing.assert_array_equal(total, [row["score_total"] for row in original])
    np.testing.assert_allclose(total, groups.sum(axis=1), rtol=0, atol=1e-8)
    np.testing.assert_array_equal(layout["counts_numpy"], np.ones((batch_size, 6), dtype=np.int64))


def test_gate_variants_are_distinct_and_lambda_survives_serialization():
    np.testing.assert_array_equal(evaluate.gate_multipliers("2 holes", ["word"]), [1, 1, 1, 2, 1, 1])
    np.testing.assert_array_equal(evaluate.gate_multipliers("2 holes", ["digit"]), [2, 2, 2, 1, 1, 1])
    np.testing.assert_array_equal(evaluate.gate_multipliers("2 holes", ["word", "digit"]), [2, 2, 2, 2, 1, 1])
    assert evaluate.make_config("resid_l1", 1, "resid", lambda_=1.0)["lambda"] == 1.0


def test_invalid_frozen_does_not_read_test_labels(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}", encoding="utf-8")

    def fail(*_):
        pytest.fail("read test axes before verifying the frozen digest")

    monkeypatch.setattr(evaluate, "load_axis", fail)
    with pytest.raises(evaluate.EvalError, match="sha256"):
        evaluate.run_evaluate(argparse.Namespace(frozen=frozen, expected_frozen_sha="0" * 64))


def test_replay_comparison_handles_serialized_counts_and_rejects_drift():
    baseline = {"text_index": 0, "dense_cad_index": 0, "cad_index": 0, "reference_sum": -7,
                "sum": -7, "n_valid_total": 3, "reference_n_valid_total": 3, "group_total": -7,
                "group_counts": [1, 1, 0, 0, 0, 1]}
    assert score.compare_replay_pairs([baseline])["n_violations"] == 0
    changed = {**baseline, "sum": -7.5, "group_total": -7.5}
    assert score.compare_replay_pairs([changed])["n_violations"] == 1
