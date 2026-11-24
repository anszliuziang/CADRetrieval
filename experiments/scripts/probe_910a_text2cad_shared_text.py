#!/usr/bin/env python3
"""Check whether one exact text encoding can be reused across a scored gallery batch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from experiments.scripts.probe_formal_text2cad_cost import RECEIPT, model_loaded
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore, assemble_batch
from experiments.src.idea01.text2cad_port.model import (
    build_teacher_forcing_inputs,
    prepare_cross_attention_mask_batch,
)
from experiments.src.idea01.text2cad_vector_scorer import two_head_factorized_score


def verify(ok, message):
    if not ok:
        raise ValueError(message)


def raw_reference(record_file):
    with record_file.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if (row["stage"], row["source"], row["level"], row["kind"]) == (
                "score", "t2cad-v1.0", "L3", "full_steady1"
            ):
                return row, pilot.sha256_bytes(line)
    raise ValueError("Need the completed long L3 reference timing row")


@torch.inference_mode()
def shared_embedding(model, text):
    t, m = model.base_text_embedder.get_embedding([text])
    result, _ = model.adaptive_layer(t, {"attn_mask": None, "key_padding_mask": m}, False)
    return result, m


@torch.inference_mode()
def score_with_shared_text(model, store, indices, prepared, device):
    vec, mask, width = assemble_batch(store, indices, device)
    v, m, _, _ = build_teacher_forcing_inputs(vec, mask)
    repeated_text = prepared[0].expand(len(indices), -1, -1).contiguous()
    kpm = prepared[1].expand(len(indices), -1).contiguous()
    ca = {"attn_mask": prepare_cross_attention_mask_batch(kpm, cad_seq_len=width), "key_padding_mask": kpm}
    pred, _ = model.cad_decoder(v, repeated_text, m, ca, False)
    verify(pred.dtype == torch.float32, f"CAD logits changed precision: {pred.dtype}")
    return two_head_factorized_score(pred.cpu(), vec["cad_vec"].cpu(), mask["key_padding_mask"].cpu())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "checkpoint", "bert", "cad-zip", "reference", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    reference, reference_sha = raw_reference(a.reference)
    verify(len(reference["uids"]) == len(set(reference["uids"])) == reference["pairs"] == 16,
           "Need the actual 16 distinct historical validation CADs")
    receipt = a.manifests / "manifest_receipt.json"
    verify(pilot.sha256_file(receipt) == RECEIPT, "Changed pilot manifest")
    pins = json.loads(receipt.read_bytes())["file_sha256"]
    paths = ("t2cad-v1.0/validation_targets.jsonl", "t2cad-v1.0/validation_gallery.jsonl")
    verify(all(pilot.sha256_file(a.manifests / name) == pins[name] for name in paths), "Changed validation identity")
    val, cad = [dict((r["uid"], r) for r in pilot.read_jsonl(a.manifests / name)) for name in paths]
    text = val[reference["validation_query_uid"]]["texts"]["L3"]
    ordered = [cad[uid] for uid in reference["uids"]]
    members = [row["cad_member"] for row in ordered]
    verify(pilot.sha256_file(a.cad_zip) == pilot.INPUT_HASHES["official_zip"], "Changed native CAD zip")
    a.out.mkdir()
    torch.set_num_threads(2)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model, provenance = model_loaded(a, device)
    store = ZipCandidateStore(a.cad_zip, members)
    verify(store.preload()["n_loaded"] == len(members), "Native validation CADs failed preload")
    indices = list(range(len(members)))
    backend = Text2CadReadout(model, store, dict(zip(members, indices, strict=True)), device)
    for row, old in zip(ordered, reference["scores"], strict=True):
        pair = backend.prepare(row, text)
        verify(all(pair[k] == old[k] for k in ("input_sha256", "target_sha256", "query_tokens")),
               "Current teacher-forcing input differs from non-cached reference")
    tic = time.monotonic()
    features = shared_embedding(model, text)
    torch.npu.synchronize(device)
    preparation_s = time.monotonic() - tic
    original_order = sorted(range(16), key=lambda i: (-reference["scores"][i]["s0"], ordered[i]["cad_index"]))
    windows, checks = [], []
    for step in range(3):
        tic = time.monotonic()
        rows = score_with_shared_text(model, store, indices, features, device)
        torch.npu.synchronize(device)
        elapsed = time.monotonic() - tic
        actual_order = sorted(range(16), key=lambda i: (-rows[i]["score_total"], ordered[i]["cad_index"]))
        windows.append({"seconds": elapsed, "pairs_per_s": len(rows) / elapsed,
                        "whole_order_equal": original_order == actual_order,
                        "phase": "cold" if step == 0 else "steady"})
        for row, old, identity in zip(rows, reference["scores"], ordered, strict=True):
            s0, original = row["score_total"], old["s0"]
            error = abs(s0 - original)
            tol = max(0.001, 1e-6 * abs(original))
            counts_match = all(row[key] == old[key] for key in ("n_valid_x", "n_valid_y", "n_real_tokens"))
            checks.append({"uid": identity["uid"], "step": step + 1, "original_s0": original,
                           "shared_text_s0": s0, "absolute_delta": error, "tolerance": tol,
                           "counts_match": counts_match, "passed": math.isfinite(s0) and error <= tol and counts_match})
        print(json.dumps(windows[-1]), flush=True)
    timing = statistics.median(x["seconds"] for x in windows if x["phase"] == "steady")
    same_order = all(x["whole_order_equal"] for x in windows)
    passed = all(check["passed"] for check in checks) and same_order
    pilot.write_json(a.out / "report.json", {
        "scope": "one complete validation text, exact CAD inputs, engineering witness",
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reference_timing_record_sha256": reference_sha, "provenance": provenance,
        "text_tokens": reference["query_tokens"], "query_uid": reference["validation_query_uid"],
        "preparation_s": preparation_s, "windows": windows, "comparison": checks,
        "original_historical_batch_s": reference["seconds"],
        "historical_batch_over_cached_steady_batch": reference["seconds"] / timing,
        "ratio_excludes_one_time_text_preparation": True,
        "whole_order_equal": same_order, "all_passed": passed,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
