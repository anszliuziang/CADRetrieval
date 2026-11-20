"""Use a fake model forward to exercise exact full-gallery worker behavior."""

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_retrieval_benchmark_fusion_score as scorer
from prepare_retrieval_benchmark_fill import load_json, read_jsonl, sha256, text_digest, write_json, write_jsonl
from test_retrieval_benchmark_fusion_score_inputs import program


def repin(input_root, manifest):
    manifest["score_inputs_sha256"] = {path.relative_to(input_root).as_posix(): sha256(path)
                                      for path in input_root.rglob("*") if path.is_file() and path.name != "freeze.json"}
    write_json(input_root / "freeze.json", manifest)
    return sha256(input_root / "freeze.json")


@pytest.fixture
def portable(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    query = sorted(["short caption", "long " * 520 + "end"], key=text_digest)
    background = [f"background {index}" for index in range(48)]
    counts = np.asarray([2 + index % 3 for index in range(18)], dtype="<i4")
    gallery = []
    for index, n_valid in enumerate(counts):
        uid = f"0001/{10000 + index:08}"
        path = f"objects/{uid.replace('/', '_')}/program.npz"
        (root / path).parent.mkdir(parents=True)
        np.savez(root / path, **program(int(n_valid)))
        gallery.append({"uid": uid, "cad_index": index, "vector_relpath": path, "n_valid": int(n_valid)})
    write_jsonl(root / "gallery.jsonl", gallery)
    score_rows = []
    for role, texts, axis, index_key in (("query", query, "texts_ct_fusion.jsonl", "text_index"),
                                         ("background", background, "background.jsonl", "index")):
        write_jsonl(root / axis, [{index_key: i, "text": text, "text_sha256": text_digest(text)}
                                  for i, text in enumerate(texts)])
        for index, text in enumerate(texts):
            score_rows.append({"row_index": len(score_rows), "role": role, "axis_index": index,
                               "text_sha256": text_digest(text)})
    write_jsonl(root / "score_rows.jsonl", score_rows)
    background_digest = text_digest("\n".join(map(text_digest, background)))
    monkeypatch.setattr(scorer, "BACKGROUND_ORDER_SHA256", background_digest)
    manifest = {
        "schema_version": 1, "kind": "fusion_score_only_inputs", "condition": "ct_fusion", "domain": "Fusion360",
        "gallery_size": 18, "unique_query_texts": 2, "background_count": 48,
        "score_shape": [50, 18], "planned_pairs": 900, "pairs_by_role": {"query": 36, "background": 864},
        "role_axes": {"query": "texts_ct_fusion.jsonl", "background": "background.jsonl"},
        "source_pins": {"background_ordered_digest": background_digest,
                        "text_axis_sha256": sha256(root / "texts_ct_fusion.jsonl")},
        "field_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
    }
    return root, repin(root, manifest)


class FakeTokenizer:
    def __call__(self, text, truncation=False, max_length=512):
        ids = [101, *range(200, 200 + len(text.split())), 102]
        return {"input_ids": ids[:max_length - 1] + ids[-1:] if truncation and len(ids) > max_length else ids}


def test_preflight_admits_exact_short_ids_and_frozen_prefix_for_long_text(portable):
    root, pin = portable
    view = scorer.read_score_view(root, pin)
    meta = scorer.token_receipts(FakeTokenizer(), view)
    assert (len(meta), [row["role"] for row in meta[:3]]) == (50, ["query", "query", "background"])
    short = next(row for row in meta if row["role"] == "query" and not row["truncated"])
    long = next(row for row in meta if row["truncated"])
    assert (short["token_count"], short["effective_token_count"]) == (4, 4)
    assert short["ids_sha256"] == text_digest("[101, 200, 201, 102]")
    assert (long["token_count"], long["effective_token_count"]) == (523, 512)


@pytest.mark.parametrize("corruption", ("missing", "mutated", "role", "qrels"))
def test_preflight_rejects_incomplete_or_inconsistent_inputs(portable, corruption):
    root, pin = portable
    path = root / "score_rows.jsonl"
    if corruption == "missing":
        path.unlink()
    elif corruption == "mutated":
        path.write_bytes(path.read_bytes() + b"\n")
    elif corruption == "qrels":
        write_jsonl(root / "qrels.jsonl", [{"uid": "not for the scorer"}])
    else:
        rows = read_jsonl(path)
        rows[2]["role"] = "query"
        write_jsonl(path, rows)
        pin = repin(root, load_json(root / "freeze.json"))
    with pytest.raises(ValueError, match="coverage|SHA256|role"):
        scorer.read_score_view(root, pin)


@pytest.fixture
def fake_worker(portable, tmp_path, monkeypatch):
    root, pin = portable
    view = scorer.read_score_view(root, pin)
    run = tmp_path / "fusion_scores_test"
    binding = scorer.WorkerBinding(
        run_id=run.name, freeze_sha256=pin, implementation_sha="a" * 40, authorization_sha="b" * 40,
        measured_probe_sha256="c" * 64, deadline_epoch=time.time() + 600, shards=2,
        max_pairs=900, max_card_seconds=120.0,
    )
    calls, charges = [], []

    def forward(model, store, text, indices, device, *, trim_mode):
        assert all(item == text[0] for item in text)
        assert trim_mode == "none"
        calls.append((text[0], tuple(indices)))
        return [{"n_valid_total": view.counts[index], "score_total": -0.5 - index} for index in indices]

    monkeypatch.setitem(sys.modules, "idea01.text2cad_frozen_runner", SimpleNamespace(forward_and_score_batch=forward))
    ledger = SimpleNamespace(path=run / "ledger.json", approved_sha=binding.implementation_sha,
                             max_pairs=binding.max_pairs, max_card_seconds=binding.max_card_seconds,
                             heartbeat=lambda invocation: None, charge_pairs=lambda invocation, n: charges.append(n))
    store = SimpleNamespace(counts=view.counts)
    return view, run, binding, calls, charges, ledger, store


def test_fake_forward_scores_every_role_once_and_preserves_tail(fake_worker):
    view, run, binding, calls, charges, ledger, store = fake_worker
    for shard in (0, 1):
        summary = scorer.score_fusion_shard(view, run, binding, shard, None, store, "fake", ledger, "active")
        assert (summary["written"], summary["verified_previous"]) == (25, 0)
        if shard == 0:
            with pytest.raises(ValueError, match="Missing frozen Fusion score row"):
                scorer.verify_complete_rows(view, run, binding)
    assert [len(columns) for _, columns in calls] == [16, 2] * 50
    assert all(columns == tuple(range(16, 18)) for _, columns in calls[1::2])
    assert charges == [18] * 50
    assert scorer.verify_complete_rows(view, run, binding) == {"query": 2, "background": 48}
    summary = scorer.score_fusion_shard(view, run, binding, 0, None, store, "fake", ledger, "active")
    assert (summary["written"], summary["verified_previous"]) == (0, 25)
    assert len(calls) == 100 and sum(charges) == 900


@pytest.mark.parametrize("change", ("incomplete", "altered", "missing_claim"))
def test_incomplete_or_altered_row_blocks_rescoring(fake_worker, change):
    view, run, binding, calls, charges, ledger, store = fake_worker
    scorer.score_fusion_shard(view, run, binding, 0, None, store, "fake", ledger, "active")
    row = run / "scores/ct_fusion/t00000.npy"
    if change == "incomplete":
        row.with_suffix(".json").unlink()
    elif change == "missing_claim":
        write_json(run / "ledger.json", {"pairs_charged": sum(charges)})
        row.with_suffix(".json").unlink()
        row.with_suffix(".claim.json").unlink()
        row.unlink()
    else:
        row.write_bytes(row.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="incomplete|receipt|durable row claim"):
        scorer.score_fusion_shard(view, run, binding, 0, None, store, "fake", ledger, "active")
    assert len(calls) == 50 and sum(charges) == 450


def test_worker_checks_deadline_before_any_forward(fake_worker):
    view, run, binding, calls, _, ledger, store = fake_worker
    expired = replace(binding, deadline_epoch=time.time() - 1)
    with pytest.raises(TimeoutError, match="deadline"):
        scorer.score_fusion_shard(view, run, expired, 0, None, store, "fake", ledger, "active")
    assert not calls and not run.exists()
