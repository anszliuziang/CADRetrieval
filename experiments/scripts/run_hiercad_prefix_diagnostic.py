#!/usr/bin/env python3
"""Diagnose failed native prefix invariance on the five original HierCAD validation cases."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/src"))
from idea01 import in_domain_pilot_native as native


LEVELS = {"t2cad-v1.0": ("L0", "L1", "L2", "L3"), "ct-deepcad": ("CT",)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replay_case(backend, original, manifest):
    proof = original["causal_prefix_probe"]
    i, j = proof["pair_indices"]
    query = original["query_axis"][0]["text"]
    candidates = [backend.prepare(manifest[original["uid_axis"][k]], query) for k in (i, j)]
    if [row["input_sha256"] for row in candidates] != proof["full_input_sha256"]:
        raise ValueError("Original causal probe input bytes differ")
    pos = proof["first_differing_full_token_index"]
    batch = backend.batch([row["encoded"] for row in candidates])
    del batch["labels"]
    if (not torch.equal(batch["input_ids"][0, :pos], batch["input_ids"][1, :pos])
            or not bool(batch["attention_mask"][:, :pos + 1].all())):
        raise ValueError("Audit prefix or attention differs before the causal split")
    return batch, pos, proof, candidates


@torch.inference_mode()
def test_backend(backend, batch, pos, full_controls):
    def forward(values):
        return backend.model(**values, use_cache=False).logits[:, pos - 1].float().cpu()

    original = forward(batch)
    outputs = {"original_a": original[0], "original_b": original[1]}
    if full_controls:
        swapped = forward({key: value.flip(0) for key, value in batch.items()})
        duplicate_a = forward({key: value[[0, 0]] for key, value in batch.items()})
        duplicate_b = forward({key: value[[1, 1]] for key, value in batch.items()})
        one_row_a = forward({key: value[:1] for key, value in batch.items()})
        one_row_b = forward({key: value[1:] for key, value in batch.items()})
        outputs.update(swapped_a=swapped[1], swapped_b=swapped[0],
                       duplicate_a0=duplicate_a[0], duplicate_a1=duplicate_a[1],
                       duplicate_b0=duplicate_b[0], duplicate_b1=duplicate_b[1],
                       single_a=one_row_a[0], single_b=one_row_b[0])

    # Change only the first supervised target token in row A. Keep every mask,
    # shape and peer row byte identical, and read the same prediction position.
    changed = {key: value.clone() for key, value in batch.items()}
    changed["input_ids"][0, pos] = batch["input_ids"][1, pos]
    variant = forward(changed)
    outputs.update(one_future_token_changed_a=variant[0], peer_b_unchanged=variant[1])
    checks = {
        "full_real_candidate_suffixes": ("original_a", "original_b"),
        "one_future_token_same_batch_slot_and_mask": ("original_a", "one_future_token_changed_a"),
        "untouched_peer_in_same_batch": ("original_b", "peer_b_unchanged"),
    }
    if full_controls:
        checks.update(
            exact_duplicate_a_rows=("duplicate_a0", "duplicate_a1"),
            exact_duplicate_b_rows=("duplicate_b0", "duplicate_b1"),
            original_a_after_batch_swap=("original_a", "swapped_a"),
            original_b_after_batch_swap=("original_b", "swapped_b"),
            separate_calls_same_padded_width=("single_a", "single_b"),
        )
    log_probs = {name: torch.log_softmax(values, dim=-1) for name, values in outputs.items()}
    comparisons = {}
    for name, (a, b) in checks.items():
        delta = (log_probs[a] - log_probs[b]).abs()
        comparisons[name] = {
            "whole_log_distribution_atol": native.H_CAUSAL_ATOL,
            "max_log_probability_abs_delta": float(delta.max()),
            "n_over_tolerance": int((delta > native.H_CAUSAL_ATOL).sum()),
            "max_total_variation": float((log_probs[a].exp() - log_probs[b].exp()).abs().sum() / 2),
            "passed": bool((delta <= native.H_CAUSAL_ATOL).all()),
        }
    return outputs, comparisons


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("original-stage", "manifests", "minimal-zip", "hiercad-repo", "base", "adapter", "out"):
        cli.add_argument(f"--{key}", required=True, type=Path)
    cli.add_argument("--device", default="npu:0")
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    torch.npu.set_device(device)
    revision = native.use_hiercad_repo(args.hiercad_repo)
    tokenizer = native.load_hiercad_tokenizer(args.base)
    model = native.hiercad_scoring_model(args.base, args.adapter, device, torch.bfloat16)
    report = {"device": torch.npu.get_device_name(device), "dtype": "bfloat16",
              "hiercad_source": revision, "unchanged_tolerance": native.H_CAUSAL_ATOL,
              "qualification": "Diagnostic only, no test or validation gallery reranking",
              "cases": {}}
    archive = native.MinimalArchive(args.minimal_zip)
    try:
        backend = native.HierCadReadout(model, tokenizer, archive, device)
        for source, levels in LEVELS.items():
            with (args.manifests / source / "validation_targets.jsonl").open(encoding="utf-8") as stream:
                manifest = {row["uid"]: row for row in map(json.loads, stream)}
            for condition in levels:
                original = read(args.original_stage / f"pilot_{source}" / source / f"{condition}_readout.json")
                batch, pos, proof, candidates = replay_case(backend, original, manifest)
                values = {"original": proof, "causal_step": pos, "effective_prefix_sha256":
                          native.digest_tensor(batch["input_ids"][0, :pos]), "real_context_lengths":
                          [len(row["encoded"].input_ids) for row in candidates],
                          "attention_mask_sha256": native.digest_tensor(batch["attention_mask"]),
                          "backends": {}}
                trace = {}
                for implementation in ("sdpa", "eager"):
                    model.get_base_model().set_attn_implementation(implementation)
                    if model.config._attn_implementation != implementation:
                        raise RuntimeError(f"Attention implementation was not applied: {implementation}")
                    native.synchronize(device)
                    started = time.perf_counter()
                    logits, checks = test_backend(backend, batch, pos, full_controls=implementation == "sdpa")
                    native.synchronize(device)
                    values["backends"][implementation] = {"comparisons": checks,
                                                          "wall_seconds": time.perf_counter() - started}
                    trace[implementation] = logits
                torch.save(trace, args.out / f"{condition}_controls.pt")
                report["cases"][condition] = values
                print(condition, json.dumps({k: v["comparisons"] for k, v in values["backends"].items()}),
                      flush=True)
        native.write_json(args.out / "diagnostic_summary.json", report)
    finally:
        archive.close()


if __name__ == "__main__":
    main()
