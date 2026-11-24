#!/usr/bin/env python3
"""Measure exact reusable Text2CAD input identities before allocating full scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_native import _tensor_candidate_hashes, check_text2cad_flags
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore
from experiments.src.idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256


def checked_jsonl(root, file, manifest):
    path = root / file
    if pilot.sha256_file(path) != manifest[file]:
        raise ValueError(f"Not the frozen score input: {file}")
    return pilot.read_jsonl(path)


def grouped_texts(tokenizer, rows):
    identities, failed, full_widths = {}, [], Counter()
    for start in range(0, len(rows), 256):
        part = rows[start:start + 256]
        for j, row in enumerate(part, start):
            if row["text_index"] != j or pilot.sha256_bytes(row["text"].encode()) != row["text_sha256"]:
                raise ValueError("Text axis identity drift")
        model_inputs = tokenizer([row["text"] for row in part], truncation=False, padding=False,
                                 return_token_type_ids=True, return_attention_mask=True)
        for i, row in enumerate(part):
            width = len(model_inputs["input_ids"][i])
            full_widths[width] += 1
            if width > 512:
                failed.append(row["text_index"])
                continue
            effective = {key: values[i] for key, values in model_inputs.items()}
            identities[row["text_index"]] = pilot.sha256_bytes(json.dumps(
                effective, sort_keys=True, separators=(",", ":")).encode())
    return identities, failed, full_widths


def main():
    # Match TextEmbedder.from_local directly. AutoTokenizer brings in its
    # model factory, which imports accelerate and torch_npu on this CPU path.
    from transformers import BertTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "cad-zip", "bert", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir()
    start = time.monotonic()
    manifest_path = a.frozen / "files_sha256.json"
    manifest = json.loads(manifest_path.read_bytes())
    if pilot.sha256_file(a.cad_zip) != pilot.INPUT_HASHES["official_zip"]:
        raise ValueError("Official CAD zip differs")
    bert_files = ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt")
    if any(pilot.sha256_file(a.bert / name) != PINNED_BERT_FILE_SHA256[name] for name in bert_files):
        raise ValueError("Tokenizer differs from the pinned model")
    tokenizer = BertTokenizer.from_pretrained(str(a.bert), local_files_only=True)
    if set(tokenizer.model_input_names) != {"input_ids", "attention_mask", "token_type_ids"}:
        raise ValueError("BERT input fields differ from the ported model")
    torch.set_num_threads(1)
    candidates = checked_jsonl(a.frozen, "candidate_inputs.jsonl", manifest)
    cad_members = [r["cad_member"] for r in candidates]
    if len(cad_members) != len(set(cad_members)) or len(cad_members) != 8046:
        raise ValueError("Changed test candidate coverage")
    store = ZipCandidateStore(a.cad_zip, cad_members)
    preload = store.preload()
    if preload["n_loaded"] != len(cad_members):
        raise ValueError(f"Failed to admit the complete original CAD axis: {preload}")
    program_keys, cad_rows = set(), []
    for i, row in enumerate(candidates):
        item = store.get(i)
        if row["cad_index"] != i:
            raise ValueError("Frozen order differs")
        check_text2cad_flags(item)
        fields = _tensor_candidate_hashes(item, store.canonical_attn_mask)
        key = pilot.sha256_bytes(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        program_keys.add(key)
        cad_rows.append({"cad_index": i, "effective_sha256": key, "fields": fields})
    pilot.write_jsonl(a.out / "candidates_effective.jsonl", cad_rows)
    levels = pilot.SOURCE_LEVELS
    results = {}
    for source, schema in levels.items():
        per_level, union, entries = {}, set(), []
        for level in (*schema, "background"):
            name = (f"{source}/validation_background_score_queries.jsonl" if level == "background"
                    else f"{source}/test/{level}_score_queries.jsonl")
            raw = checked_jsonl(a.frozen, name, manifest)
            keys, failures, widths = grouped_texts(tokenizer, raw)
            if level != "background":
                union.update(keys.values())
            m = len(set(keys.values()))
            per_level[level] = {"text_rows": len(raw), "over_512": len(failures),
                                "distinct_complete_token_inputs": m,
                                "token_widths_histogram": widths,
                                "potential_exact_pairs_after_dedup": m * len(program_keys)}
            entries.extend({"role": level, "text_index": i, "effective_sha256": keys.get(i),
                            "exclusion": None if i in keys else "over_512"}
                           for i in range(len(raw)))
        pilot.write_jsonl(a.out / f"{source}_text_inputs.jsonl", entries)
        background = {r["effective_sha256"] for r in entries if r["role"] == "background"
                      and r["effective_sha256"]}
        results[source] = {"levels": per_level, "test_token_inputs_across_levels": len(union),
                           "test_and_background_token_inputs": len(union | background),
                           "potential_exact_pairs_including_background": len(union | background) * len(program_keys)}
    pilot.write_json(a.out / "summary.json", {
        "scope": "CPU-only identical inputs audit, no scores or ground truth used",
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_is_fast": tokenizer.is_fast,
        "candidate_rows": len(candidates), "distinct_complete_model_candidates": len(program_keys),
        "frozen_manifest_sha256": pilot.sha256_file(manifest_path),
        "sources": results, "numeric_equivalence_of_optimization_admitted": False,
        "seconds": time.monotonic() - start,
    })
    print(json.dumps({"distinct_cads": len(program_keys),
                      "source_work": {k: v["potential_exact_pairs_including_background"] for k, v in results.items()},
                      "scope": "CPU input identities"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
