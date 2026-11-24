"""Score a validation-only engineering subset with the fresh CT update-two checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
import math
from pathlib import Path
from time import perf_counter

from experiments.scripts.probe_formal_text2cad_length_buckets import file_hash, read_rows, verify
from experiments.src.idea01 import in_domain_pilot as pilot


SOURCE = "ct-deepcad"
SUBSET_SIZE = 48
CHECKPOINT_SHA = "8a65e721c045dc940d7e8eff227acda8b1e532eff07f06ebccd7cd74049a96d1"
WEIGHT_SHA = "f4f1f2ab086dcea14d74e7a4df69d327d89d1de3084d2dd0029dceb54220a54b"


def load_model(args, device):
    import torch

    from experiments.scripts import run_formal_text2cad_training as native
    from experiments.src.idea01.text2cad_port import loader
    from experiments.src.idea01.text2cad_port.text_embed import TextEmbedder

    with args.checkpoint.open("rb") as stream:
        verify(file_digest(stream, "sha256").hexdigest() == args.checkpoint_sha256 == CHECKPOINT_SHA,
               "Checkpoint differs from the registered CT update-two artifact")
        stream.seek(0)
        # The exact pinned native artifact also carries the training RNG and optimizer.
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    recipe, state, weights = (checkpoint[key] for key in ("recipe", "state", "weights"))
    expected = dict(source=SOURCE, levels=("CT",), seed=17, world=2, device="npu", global_batch=16,
                    train_pairs=148046, updates_per_epoch=9253, train_only=False, audit_batch=None,
                    smoke_validation_count=0, formal_receipt=native.DEEPCAD_RECEIPT,
                    cad_zip_sha256=pilot.INPUT_HASHES["official_zip"], bert_sha256=loader.PINNED_BERT_FILE_SHA256)
    verify(all(recipe[key] == value for key, value in expected.items()), "Registered CT recipe changed")
    verify((state["epoch"], state["updates"], state["next_batch"], state["exposure"], state["completed"])
           == (1, 2, 2, 32, False) and state["stats"]["by_level"] == {"CT": 32}, "Wrong update cursor")
    verify(checkpoint["weight_sha256"] == args.weight_sha256 == WEIGHT_SHA, "Registered weight SHA changed")
    del checkpoint

    bert_files = loader.verify_bert_asset_shas(args.bert)
    embedder = TextEmbedder.from_local(args.bert, max_seq_len=512, device=device)
    bert_load = loader.verify_bert_model_load(embedder, args.bert)
    model = loader.build_text2cad(embedder, device)
    schema = model.get_trainable_state_dict()
    verify(weights.keys() == schema.keys(), "Fresh weight keys differ from the native model shell")
    for key, tensor in weights.items():
        verify(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and
               tensor.shape == schema[key].shape and tensor.dtype == schema[key].dtype == torch.float32 and
               bool(torch.isfinite(tensor).all()), f"Invalid fresh tensor: {key}")
    verify(native.weight_sha(weights) == WEIGHT_SHA, "Fresh weight digest differs")
    for name in ("adaptive_layer", "cad_decoder"):
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k, v in weights.items()
                                             if k.startswith(prefix)}, strict=True)
    model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    verify(native.weight_sha(native.cpu_weights(model)) == WEIGHT_SHA, "Loaded FP32 weight digest differs")
    return model, dict(recipe=recipe, updates=state["updates"], exposure=state["exposure"],
                       checkpoint_sha256=CHECKPOINT_SHA, weight_sha256=WEIGHT_SHA,
                       bert_files=bert_files, bert_load=bert_load)


def validation_inputs(root):
    from experiments.scripts.run_formal_text2cad_training import DEEPCAD_RECEIPT

    receipt = root / "manifest_receipt.json"
    verify(file_hash(receipt) == DEEPCAD_RECEIPT, "Frozen manifest receipt changed")
    pins = json.loads(receipt.read_bytes())["file_sha256"]
    files = {f"{SOURCE}/validation_{name}.jsonl": pins[f"{SOURCE}/validation_{name}.jsonl"]
             for name in ("targets", "gallery", "background")}
    for name, digest in files.items():
        verify(file_hash(root / name) == digest, f"Frozen CT validation input changed: {name}")
    targets, gallery, background = (read_rows(root / name) for name in files)
    verify(len(targets) == len({row["uid"] for row in targets}) == pilot.PANEL_LENGTH and
           [row["cad_index"] for row in gallery] == list(range(pilot.GALLERY_LENGTH)) and
           len({row["uid"] for row in gallery}) == len({row["cad_member"] for row in gallery}) == pilot.GALLERY_LENGTH
           and len(background) == len({row["text"] for row in background}) == pilot.BACKGROUND_LENGTH,
           "Incomplete frozen CT validation axes")
    return targets, gallery, background, {"manifest_receipt.json": DEEPCAD_RECEIPT, **files}


def subset(targets, gallery, embedder):
    groups = defaultdict(list)
    for row in targets:
        groups[row["texts"]["CT"]].append(row["uid"])
    ordered = sorted((len(embedder.tokenizer(text, truncation=False, verbose=False)["input_ids"]), text)
                     for text in groups)
    eligible = [text for width, text in ordered if width <= embedder.max_seq_len]
    repeated = next((text for text in eligible if 2 <= len(groups[text]) < SUBSET_SIZE), None)
    singleton = next((text for text in eligible if len(groups[text]) == 1), None)
    verify(repeated is not None and singleton is not None, "Need a complete repeated group and a singleton")
    queries = [{"uid": uid, "level": "CT", "text": text} for text in (repeated, singleton)
               for uid in sorted(groups[text])]
    positives = {row["uid"] for row in queries}
    selected = sorted(gallery, key=lambda row: (row["uid"] not in positives, row["cad_index"]))[:SUBSET_SIZE]
    verify(positives <= {row["uid"] for row in selected}, "Missing a registered positive CAD")
    return queries, sorted(selected, key=lambda row: row["cad_index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifests", "checkpoint", "bert", "cad-zip", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--weight-sha256", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch_npu

    from experiments.scripts.probe_formal_text2cad_persistent import gallery_batches, run_query
    from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
    from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore

    started = perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    targets, gallery, background, files = validation_inputs(args.manifests)
    torch.set_num_threads(2)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    model, provenance = load_model(args, device)
    queries, gallery = subset(targets, gallery, model.base_text_embedder)
    members = [row["cad_member"] for row in gallery]
    store = ZipCandidateStore(args.cad_zip, members, expected_sha256=pilot.INPUT_HASHES["official_zip"])
    verify(store.preload()["n_loaded"] == SUBSET_SIZE, "Incomplete validation candidate preload")
    backend = Text2CadReadout(model, store, {member: i for i, member in enumerate(members)}, device)
    text_ids = {text: i for i, text in enumerate(dict.fromkeys(row["text"] for row in [*queries, *background]))}
    texts = [dict(query_uid=f"text-{i:02}", text=text, text_sha256=pilot.sha256_bytes(text.encode("utf-8")),
                  full_input_tokens=len(backend.text_tokens(text))) for text, i in text_ids.items()]
    pilot.write_json(args.out / "axes.json", dict(candidates=gallery, texts=texts,
                     queries=[dict(row, text_i=text_ids[row["text"]]) for row in queries],
                     background=[dict(row, text_i=text_ids[row["text"]]) for row in background]))

    batches, seen, timings, vectors = gallery_batches(store, gallery), set(), [], []
    with (args.out / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as log, \
            (args.out / "s0.jsonl").open("x", encoding="utf-8", newline="\n") as raw:
        for text_i, text in enumerate(texts):
            scores, timing = run_query(model, store, batches, gallery, text, device, "fixed", seen, log)
            for cad, score in zip(gallery, scores, strict=True):
                pair = backend.prepare(cad, text["text"])
                row = dict(score, text_i=text_i, text_sha256=text["text_sha256"], cad_index=cad["cad_index"],
                           s0=score["score_total"], input_sha256=pair["input_sha256"],
                           target_sha256=pair["target_sha256"], query_tokens=pair["query_tokens"])
                raw.write(json.dumps(row, allow_nan=False) + "\n")
            raw.flush()
            vectors.append([score["score_total"] for score in scores])
            timings.append(dict(text_i=text_i, **timing))

    scores = np.asarray(vectors, dtype=np.float64)
    s0 = scores[[text_ids[row["text"]] for row in queries]]
    bg = scores[[text_ids[row["text"]] for row in background]]
    prior = np.logaddexp.reduce(bg, axis=0) - math.log(pilot.BACKGROUND_LENGTH)
    s48 = s0 - prior
    verify(np.isfinite(scores).all() and np.isfinite(s48).all(), "Nonfinite readout")
    delta = float(np.max(np.abs((s0[0] - s0[-1]) - (s48[0] - s48[-1]))))
    verify(delta <= 1e-9 and np.all(s48[:-1] == s48[0]), "Calibration or repeated-text restoration changed")
    ranks0, _ = pilot.evaluate_rankings(s0, queries, gallery)
    ranks48, _ = pilot.evaluate_rankings(s48, queries, gallery)
    for r0, r48 in zip(ranks0, ranks48, strict=True):
        r0.update(s0_rank=r0.pop("rank"), s48_rank=r48["rank"])
    pilot.write_jsonl(args.out / "ranks.jsonl", ranks0)
    np.savez(args.out / "scores.npz", s0=s0, background_s0=bg, background_logmeanexp=prior, s48=s48)
    pilot.write_json(args.out / "summary.json", dict(
        scope="engineering-only validation subset", source=SOURCE, level="CT", layout="fixed FP32",
        probe_sha256=file_hash(Path(__file__)), manifest_files=files, cad_zip=store.zip_report, model=provenance,
        candidate_count=len(gallery), query_rows=len(queries), distinct_scored_texts=len(texts),
        repeated_query_uids=[row["uid"] for row in queries[:-1]], background_count=len(background),
        fixed_cad_delta_max=delta, distinct_batch_shapes=len(seen), timings=timings,
        artifact_sha256={path.name: file_hash(path) for path in sorted(args.out.iterdir())},
        peak_device_memory_bytes=torch.npu.max_memory_allocated(device), elapsed_seconds=perf_counter() - started,
    ))


if __name__ == "__main__":
    main()
