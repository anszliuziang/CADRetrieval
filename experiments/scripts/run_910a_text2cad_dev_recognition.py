#!/usr/bin/env python3
"""Qualify released Text2CAD against the historical singleton replay, then score the development pairs."""

from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import argparse
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import time

import torch

from experiments.src.idea01.in_domain_pilot import cad_member
from experiments.src.idea01.in_domain_pilot_geometry import (
    ATTRIBUTES,
    FAMILIES,
    HISTORIES,
    fingerprint,
)
from experiments.src.idea01.in_domain_pilot_native import Text2CadReadout
from experiments.src.idea01.text2cad_frozen_runner import ZipCandidateStore
from experiments.src.idea01.text2cad_port.loader import load_frozen_text2cad
from experiments.src.idea01.text2cad_port.model import generate_attention_mask
from experiments.src.idea01.text2cad_vector_scorer import validate_cad_vec

REPLAY = "models/text2cad_native/ct-deepcad/CT_readout.json"
GEOMETRY = "geometry_40_pairs_occframes/"
INPUT_SHA = {
    REPLAY: "d1ff2ce940df7acc653a5a1ed02883f301c34ea5c3bd164fe7f026701a68fee8",
    GEOMETRY + "geometry_pairs.jsonl": "9f524f9e05d94c0cee321a26fba452395ad90f2d196422c9108d039271864824",
    GEOMETRY + "geometry_histories.jsonl": "b7767bf41d902d86b506df48f20b153061b16dc85d9277852729fd33103f2b3a",
}
VARIANTS = ("near_first", "far_first", "near_first_rephrased", "far_first_rephrased")
ENTITIES = ("A", "B")
PINS = Path("experiments/configs/text2cad_original_asset_sha256_20260930.json")


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_row(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def read_pinned(root, relative):
    data = (root / relative).read_bytes()
    if sha256(data).hexdigest() != INPUT_SHA[relative]:
        raise ValueError(f"Evidence SHA mismatch: {relative}")
    return data


def replay_reference(model, device, archive, root):
    reference = json.loads(read_pinned(root, REPLAY))
    members = [cad_member(uid) for uid in reference["uid_axis"]]
    store = ZipCandidateStore(archive, members)
    preload = store.preload()
    if len(members) != 4 or preload["n_loaded"] != 4:
        raise ValueError(f"Invalid historical replay candidates: {preload}")
    backend = Text2CadReadout(model, store, dict(zip(members, range(4), strict=True)), device)
    expected = reference["single"]["pairs"]
    keys = {(row["query_i"], row["candidate_i"]) for row in expected}
    if len(expected) != 16 or keys != set(product(range(4), repeat=2)):
        raise ValueError("Historical panel lacks the 16 unique singleton pairs")
    results = []
    for row in expected:
        query = reference["query_axis"][row["query_i"]]
        member = members[row["candidate_i"]]
        if (query["uid"] != row["query_uid"] or sha256(query["text"].encode()).hexdigest()
                != row["query_text_sha256"] or reference["uid_axis"][row["candidate_i"]]
                != row["candidate_uid"]):
            raise ValueError("Historical query/CAD identity mismatch")
        scores, _ = backend.score([backend.prepare({"cad_member": member}, query["text"])])
        result = scores[0]
        delta = abs(result["s0"] - row["s0"])
        tolerance = max(0.001, 0.000001 * abs(row["s0"]))
        identity = all(result[key] == row[key] for key in (
            "effective_units", "query_tokens", "input_sha256", "target_sha256",
        ))
        passed = math.isfinite(delta) and delta <= tolerance and identity and result["logit_dtype"] == "torch.float32"
        results.append({"query_i": row["query_i"], "candidate_i": row["candidate_i"],
                        "reference_s0": row["s0"], "actual": result, "abs_delta": delta,
                        "tolerance": tolerance, "identity_exact": identity, "passed": passed})
    return {"passed": all(row["passed"] for row in results),
            "unique_pairs": len(results), "max_abs_delta": max(row["abs_delta"] for row in results),
            "pairs": results}


class FixtureStore:
    def __init__(self, histories):
        self.canonical_attn_mask = generate_attention_mask(271)
        mask_sha = sha256(self.canonical_attn_mask.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        self.by_member = {}
        self.candidates = []
        for history in histories:
            source = history["inputs"]["text2cad"]
            n = source["real_rows"]
            key = member_name(history["pair_id"], history["history"])
            if (source["status"] != "ok" or source["truncated"] or not 2 <= n <= 272
                    or key in self.by_member or mask_sha != source["fixed_causal_mask_sha256"]):
                raise ValueError(f"Unqualified fixture: {key}")
            vectors = {
                "cad_vec": source["program_rows"] + [[0, 0]] * (272 - n),
                "flag_vec": source["flag_vec"] + [11] * (272 - n),
                "index_vec": source["index_vec"] + [source["index_vec"][-1] + 1] * (272 - n),
            }
            if fingerprint(vectors) != source["padded_program_sha256"]:
                raise ValueError(f"Padded programming vectors differ: {key}")
            tensors = {name: torch.tensor(vec, dtype=torch.int32) for name, vec in vectors.items()}
            cad = tensors["cad_vec"]
            pad_mask = cad.eq(0)
            diag = validate_cad_vec(cad, pad_mask, tensors["flag_vec"], tensors["index_vec"])
            active = ~pad_mask[1:n]
            effective = {"target_rows": cad[1:n].tolist(), "valid_fields": active.tolist()}
            if (diag.n_real_tokens != n or int(active.sum()) != source["valid_fields"]
                    or effective != source["effective_target"]
                    or fingerprint(effective) != source["input_sha256"]):
                raise ValueError(f"Shifted effective target differs: {key}")
            self.by_member[key] = len(self.candidates)
            self.candidates.append({**tensors, "key_padding_mask": pad_mask, "n_real_tokens": n})

    def get(self, index):
        return self.candidates[index]


def member_name(pair_id, history):
    return f"{pair_id}/{history}"


def fixture_design_key(pair):
    item = pair["design"]
    return item["family"], item["attribute"], item["index"]


def load_fixtures(root):
    pairs = [json.loads(line) for line in read_pinned(root, GEOMETRY + "geometry_pairs.jsonl").splitlines()]
    histories = [json.loads(line) for line in read_pinned(root, GEOMETRY + "geometry_histories.jsonl").splitlines()]
    by_pair = {row["pair_id"]: row for row in pairs}
    indexed = {(row["pair_id"], row["history"]): row for row in histories}
    if (len(pairs) != 40 or len(by_pair) != 40 or len(histories) != 160 or len(indexed) != 160
            or {fixture_design_key(row) for row in pairs} != set(product(FAMILIES, ATTRIBUTES, range(10)))
            or set(indexed) != set(product(by_pair, HISTORIES))):
        raise ValueError("Development fixture axes disagree with the fixed plan")
    for pair in pairs:
        truth = pair["replay_caption_truth"]["text2cad"]
        model = pair["models"]["text2cad"]
        if (pair["geometry"]["text2cad"]["status"] != "verified"
                or any(model[key]["status"] != "pass" for key in (
                    "same_entity_histories", "opposite_entities", "opposite_captions",
                ))
                or not pair["text2cad_field_count_audit"]["matched"]
                or any(truth[history] != {entity: history[0] == entity for entity in ENTITIES}
                       for history in HISTORIES)):
            raise ValueError(f"Text2CAD truth or effective input check failed for {pair['pair_id']}")
    return pairs, histories, indexed


def native_cells(backend, texts, members, token_records=None):
    cells = []
    for query_i, cad_i in product(range(2), repeat=2):
        text = texts[query_i]
        pair = backend.prepare({"cad_member": members[cad_i]}, text)
        if token_records is not None:
            tokens = token_records[query_i]
            effective = tokens["effective_input"]
            if (tokens["truncated"] or fingerprint(effective) != tokens["input_sha256"]
                    or tuple(effective["input_ids"]) != backend.text_tokens(text)
                    or effective["attention_mask"] != [1] * pair["query_tokens"]
                    or effective["token_type_ids"] != [0] * pair["query_tokens"]):
                raise ValueError("BERT valid tokens differ from the audited caption input")
        scores, _ = backend.score([pair])
        score = scores[0]
        if score["logit_dtype"] != "torch.float32" or not math.isfinite(score["s0"]):
            raise ValueError("Invalid released Text2CAD FP32 logit or SUM")
        cells.append({"query": ENTITIES[query_i], "cad": ENTITIES[cad_i], "member": members[cad_i],
                      "text_sha256": sha256(text.encode()).hexdigest(), **score})
    return cells


def metrics(scores):
    aa, ab, ba, bb = scores
    if any(value is None for value in scores):
        return {"cad_choose_caption": [None, None], "caption_choose_cad": [None, None],
                "both_cads": False, "both_captions": False, "swapped_both_cads": False,
                "ties": 0, "D": None}
    cad = [aa - ba, bb - ab]
    text = [aa - ab, bb - ba]
    return {"cad_choose_caption": cad, "caption_choose_cad": text,
            "both_cads": all(value > 0 for value in cad),
            "both_captions": all(value > 0 for value in text),
            "swapped_both_cads": all(value < 0 for value in cad),
            "ties": sum(value == 0 for value in cad + text),
            "D": math.fsum((aa, bb, -ab, -ba))}


def sketch_cuts(history):
    sketches = [part["sketch"]["face_1"]["loop_1"]
                for part in history["normalized_minimal"]["parts"].values()]
    if len(sketches) != 3:
        raise ValueError("Expected one stock and two cut sketches")
    outline = [curve["Start Point"] for curve in sketches[0].values()]
    area = math.fsum(ax * by - ay * bx for (ax, ay), (bx, by)
                     in zip(outline, outline[1:] + outline[:1], strict=True))
    reflex = []
    for (ax, ay), (bx, by), (cx, cy) in zip(outline[-1:] + outline[:-1], outline,
                                          outline[1:] + outline[:1], strict=True):
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross * area < -1e-12:
            reflex.append((bx, by))
    if len(reflex) not in (1, 2):
        raise ValueError("Local stock sketch does not contain a recognisable notch")
    anchor = [math.fsum(axis) / len(reflex) for axis in zip(*reflex, strict=True)]
    cuts = []
    for sketch in sketches[1:]:
        kinds = Counter(name.split("_", 1)[0] for name in sketch)
        if kinds == {"circle": 1}:
            curve = next(iter(sketch.values()))
            kind, center, radius = "round", curve["Center"], curve["Radius"]
        elif kinds == {"arc": 2, "line": 2}:
            mids = [curve["Mid Point"] for name, curve in sketch.items() if name.startswith("arc_")]
            center = [(a + b) / 2 for a, b in zip(*mids, strict=True)]
            kind, radius = "oblong", None
        else:
            raise ValueError(f"Unexpected cut topology: {kinds}")
        cuts.append((kind, radius, math.dist(anchor, center)))
    return cuts


def attribute_words(cuts, attribute):
    if attribute == "shape_location":
        return tuple(kind for kind, _, _ in cuts)
    r0, r1 = (radius for _, radius, _ in cuts)
    if r0 == r1:
        return None
    return ("larger", "smaller") if r0 > r1 else ("smaller", "larger")


def rule_scores(texts, drawings, attribute, variant):
    word_pattern = r"\b(" + "|".join(ATTRIBUTES[attribute]) + r")\b"
    caption = [tuple(re.findall(word_pattern, query)) for query in texts]
    if any(len(words) != 2 for words in caption):
        raise ValueError("Expected both attributes in each frozen caption")
    slot = [attribute_words(cuts, attribute) for cuts in drawings]
    local = []
    for cuts in drawings:
        if abs(cuts[0][2] - cuts[1][2]) <= 1e-12:
            local.append(None)
            continue
        ordered = sorted(cuts, key=lambda cut: cut[2], reverse=variant.startswith("far"))
        local.append(attribute_words(ordered, attribute))

    def compare(word_axis):
        return [float(words == target) if target is not None else 0.0
                for words, target in product(caption, word_axis)]

    return {"first_cut_slot": compare(slot), "local_sketch_distance": compare(local)}


def recognise(model, device, reference, output):
    pairs, all_histories, indexed = load_fixtures(reference)
    store = FixtureStore(all_histories)
    backend = Text2CadReadout(model, store, store.by_member, device)
    groups = defaultdict(Counter)
    all_group = Counter()
    with (output / "paired_views.jsonl").open("x", encoding="utf-8", newline="\n") as views, (
        output / "paired_outcomes.jsonl"
    ).open("x", encoding="utf-8", newline="\n") as outcomes:
        for index, pair in enumerate(pairs, start=1):
            pair_id = pair["pair_id"]
            family, attribute, _ = fixture_design_key(pair)
            drawings = {history: sketch_cuts(indexed[pair_id, history]) for history in HISTORIES}
            spectrum = {tuple(case["histories"]): case for case in
                        pair["models"]["spectrabrep"]["opposite_entities"]["comparisons"]}
            per_method = defaultdict(list)
            for order, variant in product(range(2), VARIANTS):
                ids = [f"{entity}{order}" for entity in ENTITIES]
                texts = [pair["captions"][entity][variant] for entity in ENTITIES]
                if any(sha256(text.encode()).hexdigest() != pair["caption_sha256"][entity][variant]
                       for entity, text in zip(ENTITIES, texts, strict=True)):
                    raise ValueError("Caption text hash differs")
                cells = native_cells(
                    backend, texts, [member_name(pair_id, item) for item in ids],
                    [pair["text2cad_caption_inputs"][entity][variant] for entity in ENTITIES],
                )
                rules = rule_scores(texts, [drawings[item] for item in ids], attribute, variant)
                methods = {"Text2CAD_S0": metrics([cell["s0"] for cell in cells]),
                           **{name: metrics(values) for name, values in rules.items()}}
                for name, result in methods.items():
                    per_method[name].append(result)
                feature_check = spectrum[tuple(ids)]
                bow_collision = Counter(re.findall(r"[a-z0-9]+", texts[0].lower())) == Counter(
                    re.findall(r"[a-z0-9]+", texts[1].lower())
                )
                spectral_collision = (feature_check.get("exact_value_equality") is True
                                      and len(set(feature_check["input_sha256"])) == 1)
                write_row(views, {"pair_id": pair_id, "family": family, "attribute": attribute,
                                  "order": order, "variant": variant, "cells": cells,
                                  "metrics": methods, "rule_scores": rules,
                                  "bow_identical_caption_input": bow_collision,
                                  "spectra_identical_cad_input": spectral_collision})
                all_group.update(views=1, scores=len(cells), bow_collisions=bow_collision,
                                 spectra_collisions=spectral_collision)
            summary = {}
            for name, results in per_method.items():
                win = all(result["both_cads"] for result in results)
                swap = all(result["swapped_both_cads"] for result in results)
                counts = {"pairs": 1, "joint": int(win), "joint_swap": int(swap),
                          "N_twice": int(win) + int(swap), "views": len(results),
                          "both_cads": sum(result["both_cads"] for result in results),
                          "both_captions": sum(result["both_captions"] for result in results),
                          "D_positive": sum(result["D"] > 0 for result in results),
                          "ties": sum(result["ties"] for result in results),
                          "fixed_cad_correct": sum(value > 0 for row in results for value in row["cad_choose_caption"])}
                summary[name] = counts
                groups[(family, attribute, name)].update(counts)
            write_row(outcomes, {"pair_id": pair_id, "methods": summary})
            if index % 10 == 0:
                print(f"Text2CAD completed {index}/40 pairs", flush=True)

    by_design = {fixture_design_key(row): row["pair_id"] for row in pairs}
    controls = []
    for family, index in product(FAMILIES, range(2)):
        ids = [by_design[(family, attr, index)] for attr in ATTRIBUTES]
        for pair_id, expected in zip(ids, (["round", "round"], ["oblong", "round"]), strict=True):
            geometry = indexed[pair_id, "A0"]["geometry"]["text2cad"]
            if geometry["status"] != "ok" or sorted(geometry[loc]["kind"] for loc in ("near", "far")) != expected:
                raise ValueError("Coarse control geometry disagrees with its literal caption")
        part = FAMILIES[family][0]
        texts = [f"A {part} has two round through holes.",
                 f"A {part} has one round through hole and one oblong through hole."]
        cells = native_cells(backend, texts, [member_name(pair_id, "A0") for pair_id in ids])
        controls.append({"family": family, "index": index, "texts": texts, "cells": cells,
                         "metrics": metrics([cell["s0"] for cell in cells])})
    write_json(output / "coarse_shape_controls.json", controls)
    if all_group["views"] != 320 or all_group["scores"] != 1280:
        raise ValueError("Development matrix was incomplete")
    return {"global": dict(all_group), "coarse_shape_views": len(controls), "missing_scores": 0,
            "groups": [{"family": family, "attribute": attr, "method": name, **counts}
                       for (family, attr, name), counts in groups.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("npu:0", "cpu"), required=True)
    parser.add_argument("--precision", choices=("default", "preserve"), default="default")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if args.device == "cpu" and args.precision != "default":
        parser.error("preserve applies only to NPU")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Use the clean deployed revision")
    args.out.mkdir(exist_ok=False, parents=True)
    device = torch.device(args.device) if args.device == "cpu" else None
    if args.device.startswith("npu:"):
        import torch_npu

        if args.precision == "preserve":
            torch_npu.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        device = torch.device(args.device)
        torch.npu.set_device(device)
    torch.set_num_threads(8 if device.type == "cpu" else 2)
    pins = json.loads(PINS.read_bytes())
    data_revision = pins["dataset_revision"]
    asset_root = args.assets
    reference = asset_root / "reference"
    archive = asset_root / f"datasets/Text2CAD-v1.0/{data_revision}/cad_seq.zip"
    checkpoint = asset_root / f"models/Text2CAD-v1.0/{data_revision}/Text2CAD_1.0.pth"
    bert = asset_root / "models/bert-large-uncased" / pins["bert_revision"]
    metadata = {"git_sha": commit, "host": socket.gethostname(), "device": str(device),
                "precision": args.precision, "torch": version("torch"),
                "transformers": version("transformers"), "asset_pins": pins, "evidence_sha256": INPUT_SHA,
                "rows": 272, "batch_size": 1}
    if device.type == "npu":
        metadata.update(torch_npu=version("torch-npu"), hardware=torch.npu.get_device_name(device))
    write_json(args.out / "runtime.json", metadata)
    start = time.perf_counter()
    model, info = load_frozen_text2cad(checkpoint, bert, device)
    write_json(args.out / "load.json", info)
    replay = replay_reference(model, device, archive, reference)
    write_json(args.out / "replay.json", replay)
    print(f"Replay passed={replay['passed']} max_abs_delta={replay['max_abs_delta']:.9g}", flush=True)
    if not replay["passed"]:
        return 3
    if not args.replay_only:
        result = recognise(model, device, reference, args.out)
        elapsed = time.perf_counter() - start
        write_json(args.out / "summary.json", {**result, "wall_seconds": elapsed})
        print(f"Completed full development recognition after {elapsed:.2f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
