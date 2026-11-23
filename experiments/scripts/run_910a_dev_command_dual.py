#!/usr/bin/env python3
"""Run the two frozen command dual-tower controls on the 40 development pairs."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from experiments.scripts.audit_in_domain_pilot_effective_inputs import caption_inputs
from experiments.scripts.run_910a_text2cad_dev_recognition import (
    FAMILIES,
    ATTRIBUTES,
    HISTORIES,
    VARIANTS,
    FixtureStore,
    fixture_design_key,
    load_fixtures,
    member_name,
    metrics,
    write_json,
    write_row,
)
from experiments.scripts.run_in_domain_pilot_retrievers import RetrievalTowers, TextInputs
from experiments.src.idea01.in_domain_pilot_retrievers import CommandTower, OUTPUT_DIM, effective_command_input
from experiments.src.idea01.in_domain_pilot_geometry import fingerprint
from experiments.src.idea01.text2cad_port.loader import verify_bert_asset_shas

CHECKPOINTS = {
    "dual_t": ("t2cad-v1.0", "46f73b934fe02e44a143f63a54f3cf237d341d8abfa2c468793bd59256677a2c"),
    "dual_ct": ("ct-deepcad", "c7e0892c1a7a9f30a68924cd07f2ceadfb75424f47d580948a4f48f1d650b7cf"),
}
CAPTION_EVIDENCE_SHA = "69d9710e548d4918b6e7fab8541f7c0a8e6a375d6a48f1e66022df1391d356ef"


def file_sha(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def verified_tokens(reference, pairs, tokenizer, config, tag):
    evidence = reference / f"retriever_effective_inputs/{tag}.jsonl"
    if file_sha(evidence) != CAPTION_EVIDENCE_SHA:
        raise ValueError(f"{tag} prior tokenizer evidence differs")
    rows = [json.loads(line) for line in evidence.read_bytes().splitlines()]
    by_pair = {row["pair_id"]: row["caption_inputs"] for row in rows}
    if len(rows) != 40 or len(by_pair) != 40:
        raise ValueError("Expected all 40 frozen token records")
    for pair in pairs:
        if caption_inputs(tokenizer, config, pair["captions"]) != by_pair[pair["pair_id"]]:
            raise ValueError(f"{tag} text tokenization mismatch at {pair['pair_id']}")


def verified_commands(store, histories):
    result = []
    for row in histories:
        candidate = store.get(store.by_member[member_name(row["pair_id"], row["history"])])
        seq = effective_command_input(candidate)
        saved = row["inputs"]["command_dual_tower"]
        if (saved["status"] != "ok" or saved["truncated"] or len(seq) != saved["real_rows"]
                or seq.tolist() != saved["effective_input"] or fingerprint(seq.tolist()) != saved["input_sha256"]):
            raise ValueError("Command-tower effective inputs differ from the prior witness")
        padded = torch.zeros((272, 4), dtype=torch.int32)
        padded[:len(seq)] = seq
        result.append(padded)
    return torch.stack(result)


def load_heads(path, tag, reference, backbone):
    source, expected = CHECKPOINTS[tag]
    if file_sha(path) != expected:
        raise ValueError(f"{tag} checkpoint file differs")
    audited_config = json.loads((reference / f"models/{tag}/config.json").read_bytes())
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if (json.loads(json.dumps(config)) != audited_config or config["source"] != source
            or config["method"] != "dual-tower" or config["backbone_precision"] != "bf16"
            or config["head_dtype"] != "float32" or config["attention"] != "eager"
            or config["text_pooling"] != "masked mean of frozen last hidden states"):
        raise ValueError(f"{tag} trained method config differs")
    width = backbone.config.hidden_size
    towers = RetrievalTowers(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, OUTPUT_DIM)), CommandTower())
    towers.cad.load_state_dict(checkpoint["cad_tower"], strict=True)
    towers.text.load_state_dict(checkpoint["text_head"], strict=True)
    return towers.eval().requires_grad_(False), config


@torch.inference_mode()
def evaluate(towers, config, tokenizer, backbone, texts, pairs, commands, store, output):
    batch = config["per_device_evaluation_batch"]
    text_inputs = TextInputs(tokenizer, texts, config["instruction"], config["max_text_tokens"], torch.device("cpu"))
    pooled, indexes = text_inputs.frozen(backbone, texts, batch)
    query = torch.cat([towers.encode_text(values) for values in pooled.split(batch)])
    cad = torch.cat([towers.encode_cad(values) for values in commands.split(batch)])
    if not bool(torch.isfinite(query).all() and torch.isfinite(cad).all()):
        raise ValueError("Non-finite command dual-tower embeddings")
    combined = query @ cad.T
    groups = defaultdict(Counter)
    with (output / "views.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            pair_id = pair["pair_id"]
            group = fixture_design_key(pair)[:2]
            diagnostics = []
            for n, variant in product(range(2), VARIANTS):
                q = [indexes[pair["captions"][entity][variant]] for entity in "AB"]
                c = [store.by_member[member_name(pair_id, f"{entity}{n}")] for entity in "AB"]
                raw = [float(combined[i, j]) for i, j in product(q, c)]
                result = metrics(raw)
                diagnostics.append(result)
                write_row(stream, {"pair_id": pair_id, "order": n, "variant": variant,
                                   "scores_aa_ab_ba_bb": raw, **result})
            joint = all(result["both_cads"] for result in diagnostics)
            swapped = all(result["swapped_both_cads"] for result in diagnostics)
            groups[group].update(pairs=1, views=len(diagnostics), scores=4 * len(diagnostics),
                joint=joint, joint_swap=swapped, N_twice=int(joint) + int(swapped),
                both_cads=sum(result["both_cads"] for result in diagnostics),
                both_captions=sum(result["both_captions"] for result in diagnostics),
                D_positive=sum(result["D"] > 0 for result in diagnostics),
                ties=sum(result["ties"] for result in diagnostics))
    if sum(result["scores"] for result in groups.values()) != 1280:
        raise ValueError("Command dual-tower view coverage disagrees with the plan")
    return [{"family": f, "attribute": a, **groups[f, a]} for f, a in product(FAMILIES, ATTRIBUTES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Use the clean deployed implementation")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pairs, histories, _ = load_fixtures(args.reference)
    store = FixtureStore(histories)
    commands = verified_commands(store, histories)
    texts = list(dict.fromkeys(text for pair in pairs
                               for entity in "AB" for text in pair["captions"][entity].values()))
    bert_files = verify_bert_asset_shas(args.bert)
    tokenizer = AutoTokenizer.from_pretrained(args.bert, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    backbone = AutoModel.from_pretrained(
        args.bert, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().requires_grad_(False)
    if {value.dtype for value in backbone.parameters()} != {torch.bfloat16}:
        raise ValueError("The frozen BERT did not load in BF16")
    write_json(args.out / "runtime.json", {"device": "cpu", "git_sha": commit,
               "bert_files": bert_files, "bert_dtype": "bfloat16", "text_count": len(texts),
               "command_count": len(commands), "torch": torch.__version__,
               "checkpoints": CHECKPOINTS})
    for tag in CHECKPOINTS:
        target = args.out / tag
        target.mkdir()
        towers, config = load_heads(args.checkpoints / f"models/{tag}/retrieval.pt", tag, args.reference, backbone)
        verified_tokens(args.reference, pairs, tokenizer, config, tag)
        results = evaluate(towers, config, tokenizer, backbone, texts, pairs, commands, store, target)
        write_json(target / "summary.json", {"groups": results, "missing_scores": 0})
        print(f"{tag}: 1,280 development scores, complete", flush=True)
    write_json(args.out / "summary.json", {"completed": list(CHECKPOINTS),
                                          "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
