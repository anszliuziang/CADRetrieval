"""Check FP32 pad-64 validation on the frozen Fusion step-128 checkpoint."""

from argparse import ArgumentParser
from collections import defaultdict
import json
import math
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import torch

from experiments.scripts import run_formal_text2cad_training as native


WEIGHT_SHA = "09780095d201a064363e6cd07d89cfb42c35cff3b5b43e59d53e6370e91cbefb"
ARGV_SHA = "ccce05f184bc7b447d06d31ce850c7b3b2534c16ec03d21d20dc93a5ea78f0e7"
SUPPORT_SHA = "408d3891f8f48bb20940abec09fb9cbe12d2b1210ef77cbe52d2b7defeb25c7a"


def select_witness_batches(widths):
    by_pad_width = defaultdict(list)
    for batch, width in enumerate(widths):
        by_pad_width[math.ceil(width / 64) * 64].append(batch)
    for group in by_pad_width.values():
        first = group[0]
        second = next((index for index in group[1:] if widths[index] != widths[first]), None)
        if second is not None:
            return (first, second)
    raise ValueError("A validation witness needs two native widths in the same padded bucket")


@torch.no_grad()
def score(model, store, indices, targets, rows, device, pad):
    embedder = model.base_text_embedder
    embedder.truncate = False
    embedder.validation_pad_multiple = 64 if pad else None
    texts = [targets[row["index"]].texts[0] for row in rows]
    tokenized = native.encode_training_texts(embedder.tokenizer, texts, 64 if pad else None)
    token_count = tokenized["attention_mask"].sum(axis=1).tolist()
    native.require(token_count == [row["tokens"] for row in rows] and max(token_count) <= 512,
                   "Validation padding changed effective tokens or attempted truncation")
    vec, masks, seq = native.assemble_batch(store, [indices[row["member"]] for row in rows], device)
    native.synchronize(device)
    started = monotonic()
    logits, _, _ = native.teacher_forced_logits(model, texts, vec, masks, cad_seq_len=seq)
    native.synchronize(device)
    native.require(logits.dtype == torch.float32, "Expected FP32 validation logits")
    values = native.two_head_factorized_score(logits.cpu(), vec["cad_vec"].cpu(), masks["key_padding_mask"].cpu())
    return dict(seconds=monotonic() - started, width=tokenized["input_ids"].shape[1], scores=values,
                effective_tokens=[{key: value[i, :n].tolist() for key, value in tokenized.items()}
                                  for i, n in enumerate(token_count)])


def check_scores(reference, candidate):
    native.require(reference["effective_tokens"] == candidate["effective_tokens"], "Validation tokens changed")
    largest = 0.0
    for ref, cand in zip(reference["scores"], candidate["scores"], strict=True):
        for head in ("x", "y", "total"):
            fields = ref["n_valid_" + head]
            native.require(fields == cand["n_valid_" + head], "Validation fields changed")
            scale = max(1, fields)
            a, b = ref["score_" + head] / scale, cand["score_" + head] / scale
            native.require(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5), f"Validation {head} score changed")
            largest = max(largest, abs(a - b))
    return largest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    argv = json.loads(native.verified(args.snapshot / "terminal_evidence/actual_argv.json", ARGV_SHA).read_bytes())

    def option(name):
        return argv[argv.index("--" + name) + 1]

    native.require(option("source") == native.FUSION, "Wrong validation source")
    need = SimpleNamespace(source=native.FUSION, train_only=False, device="npu",
                           manifests=Path(option("manifests")), cad_zip=Path(option("cad-zip")),
                           fusion_validation_qrels=Path(option("fusion-validation-qrels")))
    bert = Path(option("bert"))
    train, targets, archive_hash = native.axes(need)
    levels = native.LEVELS[native.FUSION]
    members = sorted({row.member for row in train + targets if row.member})
    indices = {name: i for i, name in enumerate(members)}
    store = native.LazyCandidates(need.cad_zip, members, archive_hash)
    device, rank, world = native.runtime(need)
    native.require((rank, world) == (0, 1), "This is a single-card validation witness")
    native.verify_bert_asset_shas(bert)
    embedder = native.TrainingEmbedder.from_local(bert, 512, device)
    native.verify_bert_model_load(embedder, bert)
    native.require(embedder.tokenizer.padding_side == embedder.tokenizer.truncation_side == "right",
                   "Expected the registered right-padding tokenizer")
    frozen = json.loads(native.verified(args.snapshot / "validation-support.json", SUPPORT_SHA).read_bytes())
    native.require(native.validation_support(store, indices, targets, levels, embedder) == frozen,
                   "Fusion validation coverage changed")
    checkpoint = torch.load(native.verified(args.snapshot / "progress-a.pt", WEIGHT_SHA), map_location="cpu",
                            weights_only=False)
    native.require(checkpoint["recipe"]["source"] == native.FUSION and checkpoint["state"]["updates"] == 128,
                   "Wrong Fusion recovery checkpoint")
    batch_size = checkpoint["recipe"]["validation_batch_per_rank"]
    native.require(batch_size == 8, "Unexpected Fusion validation batch size")
    native.seed_all(native.SEED, device)
    model = native.build_text2cad(embedder, device).to(device).eval()
    for name in ("adaptive_layer", "cad_decoder"):
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in checkpoint["weights"].items()
                                             if k.startswith(prefix)}, strict=True)
    native.require(native.weight_sha(native.cpu_weights(model)) == checkpoint["weight_sha256"],
                   "Fusion validation weights differ from the checkpoint")
    valid = [row for row in frozen if row["status"] == "ok"]
    batches = [valid[start:start + batch_size] for start in range(0, len(valid), batch_size)]
    widths = [max(row["tokens"] for row in batch) for batch in batches]
    selected = select_witness_batches(widths)
    native.require(all(len(batches[batch]) == batch_size for batch in selected), "The witness requires two full B8 batches")
    trials = []
    for batch in selected:
        ref = score(model, store, indices, targets, batches[batch], device, False)
        cand = score(model, store, indices, targets, batches[batch], device, True)
        native.require(cand["width"] == math.ceil(ref["width"] / 64) * 64, "Wrong validation bucket")
        largest = check_scores(ref, cand)
        trials.append(dict(batch=batch, query_indices=[row["index"] for row in batches[batch]], reference=ref,
                           padded=cand, max_abs_per_field_nll_error=largest))
        print(json.dumps(dict(batch=batch, original_width=ref["width"], padded_width=cand["width"],
                              max_abs_per_field_nll_error=largest)), flush=True)
    native.write_json(args.out / "receipt.json", dict(
        passed=True, source=native.FUSION, scope="validation-only padding witness, no training or selection",
        frozen_checkpoint_sha256=WEIGHT_SHA, validation_panel_sha256=SUPPORT_SHA,
        validation_rows=len(frozen), eligible_rows=len(valid), batches=len(batches),
        native_distinct_widths=len(set(widths)), padded_distinct_widths=len({math.ceil(w / 64) * 64 for w in widths}),
        trials=trials))


if __name__ == "__main__":
    main()
