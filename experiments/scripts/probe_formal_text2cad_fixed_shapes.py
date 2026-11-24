#!/usr/bin/env python3
"""Check masked text padding and filling the final CAD batch on a fixed validation tail."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch_npu

from experiments.scripts.probe_910a_text2cad_november_path import compare
from experiments.scripts.probe_910a_text2cad_shared_text import (
    RECEIPT, model_loaded, raw_reference, shared_embedding, verify,
)
from experiments.scripts.probe_formal_text2cad_length_buckets import CAPS, score_batch
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "checkpoint", "bert", "cad-zip", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prior, prior_sha = raw_reference(args.reference, "L2", "tail_steady")
    verify(len(set(prior["uids"])) == prior["pairs"] == len(prior["uids"]) == 14, "Expected the complete prior L2 tail")
    pin = args.manifests / "manifest_receipt.json"
    verify(pilot.sha256_file(pin) == RECEIPT, "Frozen validation manifest differs")
    pins = json.loads(pin.read_bytes())["file_sha256"]
    names = [f"t2cad-v1.0/{name}.jsonl" for name in ("validation_targets", "validation_gallery")]
    verify(all(pilot.sha256_file(args.manifests / name) == pins[name] for name in names), "Frozen validation axis differs")
    queries, gallery = [{row["uid"]: row for row in pilot.read_jsonl(args.manifests / name)} for name in names]
    query = queries[prior["validation_query_uid"]]["texts"]["L2"]
    candidates = [gallery[uid] for uid in prior["uids"]]
    members = [row["cad_member"] for row in candidates]
    verify(pilot.sha256_file(args.cad_zip) == pilot.INPUT_HASHES["official_zip"], "Native CAD archive drift")
    args.out.mkdir()
    torch.set_num_threads(2)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model, tensors = model_loaded(args, device)
    store = ZipCandidateStore(args.cad_zip, members)
    verify(store.preload()["n_loaded"] == len(members), "Candidate preload incomplete")
    indices = list(range(len(members)))
    axis = [row["cad_index"] for row in candidates]
    native = Text2CadReadout(model, store, dict(zip(members, indices, strict=True)), device)
    for row, old in zip(candidates, prior["scores"], strict=True):
        actual = native.prepare(row, query)
        verify(all(actual[name] == old[name] for name in ("input_sha256", "target_sha256", "query_tokens")),
               "Validation input changed from the preceding native path")
    width = CAPS[bisect_left(CAPS, max(store.get(i)["n_real_tokens"] for i in indices))]
    embedder = model.base_text_embedder
    tokenizer = embedder.tokenizer
    verify(tokenizer.padding_side == "right", "Expected right-padding native tokenizer")

    def padded_tokenizer(*pos, **kwargs):
        return tokenizer(*pos, pad_to_multiple_of=32, **kwargs)

    original = tokenizer([query], return_tensors="pt", padding=True, truncation=False)
    padded = padded_tokenizer([query], return_tensors="pt", padding=True, truncation=False)
    raw_length = original["input_ids"].shape[1]
    padded_length = padded["input_ids"].shape[1]
    verify(raw_length == prior["query_tokens"] < padded_length <= 512 and padded_length % 32 == 0,
           "Right-padding did not give the registered width")
    verify(all(torch.equal(original[key], padded[key][:, :raw_length]) for key in original)
           and not bool(padded["attention_mask"][:, raw_length:].any()), "Text inputs changed at an active position")

    started = time.monotonic()
    unpadded = shared_embedding(model, query)
    torch.npu.synchronize(device)
    first = time.monotonic() - started
    started = time.monotonic()
    with patch.object(embedder, "tokenizer", padded_tokenizer):
        prepared = shared_embedding(model, query)
    torch.npu.synchronize(device)
    second = time.monotonic() - started
    verify(not bool(prepared[1][:, :raw_length].any()) and bool(prepared[1][:, raw_length:].all()),
           "New cross-attention key padding does not match the original query")

    base, base_time = score_batch(model, store, indices, unpadded, device, width)
    text_only, text_time = score_batch(model, store, indices, prepared, device, width)
    full, full_time = score_batch(model, store, [*indices, indices[-1], indices[-1]], prepared, device, width)
    checks = {"native_layout": compare(prior["scores"], base, axis),
              "text_right_padding": compare(base, text_only, axis),
              "text_padding_and_two_dummy_lanes": compare(base, full[:len(indices)], axis)}
    torch.testing.assert_close(prepared[0][:, :raw_length].cpu(), unpadded[0].cpu(), rtol=5e-4, atol=2e-4)
    result = {"scope": "one fixed validation tail, score and order gate", "uids": prior["uids"],
              "query_uid": prior["validation_query_uid"], "reference_row_sha256": prior_sha,
              "source_assets": tensors, "full_cad_rows": width, "raw_text_tokens": raw_length,
              "padded_text_tokens": padded_length, "real_rows": len(indices), "dummy_rows": 2,
              "text_prep_seconds": {"native": first, "right_padded": second},
              "timings": {"native": base_time, "right_padded": text_time, "both_fixed": full_time},
              "comparisons": checks, "passed": all(check["passed"] for check in checks.values())}
    pilot.write_json(args.out / "report.json", result)
    print(json.dumps({"passed": result["passed"], "comparisons": checks, "actual_tokens": [raw_length, padded_length]}),
          flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
