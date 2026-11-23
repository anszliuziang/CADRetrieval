from math import log

import torch

from experiments.src.idea01.in_domain_pilot_retrievers import (
    direct_registration,
    effective_command_input,
    known_positive_counts,
    registered_positive_mask,
    symmetric_positive_set_loss,
)


def test_direct_registrations_cross_logical_rank_boundaries_without_transitivity():
    registration = direct_registration([("x", "A"), ("x", "B"), ("y", "B")])
    rank_zero = ("x", "A")
    rank_one = ("y", "B")
    texts, uids = zip(rank_zero, rank_one)
    mask = registered_positive_mask(texts, uids, registration)

    assert mask.tolist() == [[True, True], [False, True]]
    logits = torch.zeros(2, 2, requires_grad=True)
    loss = symmetric_positive_set_loss(logits, mask)
    torch.testing.assert_close(loss, torch.tensor(log(2) / 2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-0.125, 0.0], [0.25, -0.125]]))

    counts = known_positive_counts(mask, texts, uids)
    assert counts["known_off_diagonal_positives"] == 1
    assert counts["off_diagonal_cells"] == 2
    assert counts["known_off_diagonal_other_registration"] == 1


def test_same_uid_and_identical_text_are_exact_registrations_in_the_batch():
    pairs = [("x", "A"), ("x", "B"), ("y", "B"), ("X", "C")]
    texts, uids = zip(*pairs)
    mask = registered_positive_mask(texts, uids, direct_registration(pairs))
    assert mask.tolist() == [
        [True, True, True, False],
        [True, True, True, False],
        [False, True, True, False],
        [False, False, False, True],
    ]
    counts = known_positive_counts(mask, texts, uids)
    assert counts["known_off_diagonal_positives"] == 5
    assert counts["known_off_diagonal_same_uid"] == 2
    assert counts["known_off_diagonal_same_text"] == 2
    assert counts["known_off_diagonal_other_registration"] == 1


def test_official_sequence_input_preserves_order_terminals_flags_and_indices():
    cad_vec = torch.tensor([[1, 0], [19, 20], [5, 0], [1, 0], [0, 0]])
    payload = {
        "cad_vec": cad_vec,
        "key_padding_mask": cad_vec == 0,
        "flag_vec": torch.tensor([0, 0, 1, 0, 11]),
        "index_vec": torch.tensor([0, 0, 0, 1, 2]),
    }
    command = effective_command_input(payload)
    assert command.dtype == torch.int32
    assert command.tolist() == [[1, 0, 0, 0], [19, 20, 0, 0], [5, 0, 1, 0], [1, 0, 0, 1]]
