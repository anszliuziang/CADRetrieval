#!/usr/bin/env python3
"""Check both trained retrievers' actual input transforms on the frozen development pairs."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


PAIR_LINKS = {
    "same_entity_histories": (("A0", "A1"), ("B0", "B1")),
    "opposite_entities": (("A0", "B0"), ("A0", "B1"), ("A1", "B0"), ("A1", "B1")),
}


def read_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def file_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def fingerprint_ids(value):
    return sha256(np.asarray(value, dtype="<i4").tobytes()).hexdigest()


def caption_inputs(tokenizer, config, captions):
    result = []
    limit = config["max_text_tokens"]
    for variant, left in captions["A"].items():
        prompts = [config["instruction"] + text for text in (left, captions["B"][variant])]
        full = tokenizer(prompts, padding=False, truncation=False, return_attention_mask=False,
                         verbose=False)["input_ids"]
        used = [row[-limit:] if tokenizer.truncation_side == "left" else row[:limit] for row in full]
        result.append({"variant": variant, "full_tokens": [len(row) for row in full],
                       "used_tokens": [len(row) for row in used], "collision": used[0] == used[1],
                       "token_sha256": [fingerprint_ids(row) for row in used]})
    return result


def spectra_inputs(histories, moments):
    mean, std = moments
    values = {}
    for history, row in histories.items():
        feature = row["inputs"]["spectrabrep"]
        if feature["status"] != "ok":
            values[history] = None
            continue
        raw = np.asarray(feature["effective_input"], dtype="<f4")
        if sha256(raw.tobytes()).hexdigest() != feature["input_sha256"]:
            raise ValueError(f"Spectra descriptor bytes differ for {row['pair_id']} {history}")
        vec = (torch.from_numpy(raw) - mean) / std
        if vec.shape != (280,) or not torch.isfinite(vec).all():
            raise ValueError("Invalid FP32 normalized SpectraBRep feature")
        values[history] = vec
    records = {}
    for check, links in PAIR_LINKS.items():
        records[check] = []
        for left, right in links:
            a, b = values[left], values[right]
            record = {"histories": [left, right]}
            if a is None or b is None:
                record["status"] = "unavailable"
            else:
                record.update(status="collision" if torch.equal(a, b) else "distinct",
                              max_abs_difference=float((a - b).abs().max()))
            records[check].append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pair_path = args.geometry / "geometry_pairs.jsonl"
    histories_path = args.geometry / "geometry_histories.jsonl"
    geometry = json.loads((args.geometry / "geometry_summary.json").read_text())
    for path in (pair_path, histories_path):
        if file_sha(path) != geometry["artifacts_sha256"][path.name]:
            raise ValueError(f"Geometry proof bytes differ for {path.name}")
    pairs = read_lines(pair_path)
    histories = {}
    for history in read_lines(histories_path):
        histories.setdefault(history["pair_id"], {})[history["history"]] = history
    if len(pairs) != 40 or len(histories) != 40 or any(len(x) != 4 for x in histories.values()):
        raise ValueError("Expected forty frozen pairs and four histories per pair")
    totals = {}
    for tag in ("dual_t", "dual_ct", "spectra_t", "spectra_ct"):
        model = args.models / tag
        config_path = model / "config.json"
        config = json.loads(config_path.read_text())
        model_run = json.loads((model / "run.json").read_text())
        if file_sha(config_path) != model_run["files"]["config.json"]:
            raise ValueError(f"Unregistered model configuration for {tag}")
        tokenizer_root = model / ("adapter" if tag.startswith("spectra") else "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, trust_remote_code=False)
        if tokenizer.truncation_side != config["tokenizer"]["truncation_side"]:
            raise ValueError(f"Tokenizer truncation side drifted for {tag}")
        tokens_file = tokenizer_root / "tokenizer.json"
        tokenizer_pin = {p.name: file_sha(p) for p in (
            tokens_file, tokenizer_root / "tokenizer_config.json") if p.is_file()}
        moments = None
        checkpoint_sha = None
        if tag.startswith("spectra"):
            checkpoint = model / "retrieval.pt"
            checkpoint_sha = file_sha(checkpoint)
            if checkpoint_sha != model_run["files"]["retrieval.pt"]:
                raise ValueError(f"Unregistered feature normalization for {tag}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            moments = state["mean"], state["std"]
        checks = []
        for pair in pairs:
            check = {"pair_id": pair["pair_id"],
                     "caption_inputs": caption_inputs(tokenizer, config, pair["captions"])}
            if moments is not None:
                check["normalized_spectra"] = spectra_inputs(histories[pair["pair_id"]], moments)
            checks.append(check)
        counts = Counter(link["collision"] for row in checks for link in row["caption_inputs"])
        summary = {"pairs": len(checks), "caption_comparisons": counts.total(), "caption_collisions": counts[True],
                   "caption_truncated": sum(row["full_tokens"] != row["used_tokens"]
                                            for check in checks for row in check["caption_inputs"]),
                   "tokenizer_sha256": tokenizer_pin, "checkpoint_sha256": checkpoint_sha,
                   "config_sha256": file_sha(config_path)}
        if moments is not None:
            summary["normalized_spectra"] = {
                check: dict(Counter(link["status"] for row in checks for link in row["normalized_spectra"][check]))
                for check in PAIR_LINKS
            }
        with (args.out / f"{tag}.jsonl").open("x", encoding="utf-8") as stream:
            for check in checks:
                stream.write(json.dumps(check, ensure_ascii=False, allow_nan=False) + "\n")
        totals[tag] = summary
    with (args.out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump({"models": totals, "geometry_artifacts_sha256": geometry["artifacts_sha256"]},
                  stream, allow_nan=False, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
