#!/usr/bin/env python3
"""Check external baseline inputs and recompute all strict recognition outcomes from disk."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

FIXTURE_PINS = {
    "geometry_pairs.jsonl": "9f524f9e05d94c0cee321a26fba452395ad90f2d196422c9108d039271864824",
    "geometry_histories.jsonl": "b7767bf41d902d86b506df48f20b153061b16dc85d9277852729fd33103f2b3a",
}
VARIANTS = ("near_first", "far_first", "near_first_rephrased", "far_first_rephrased")
DIMS = {"mmcad": 768, "openshape": 1280, "clip": 512, "clip_top": 512}


def check(ok, context):
    if not ok:
        raise ValueError(context)


def read_json(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def geometry_axes(fixture, prepared):
    for name, expected in FIXTURE_PINS.items():
        check(file_hash(fixture / name) == expected, name)
    pairs = {x["pair_id"]: x for x in read_rows(fixture / "geometry_pairs.jsonl")}
    histories = {(x["pair_id"], x["history"]): x for x in read_rows(fixture / "geometry_histories.jsonl")}
    check(len(pairs) == 40 and set(histories) == set(product(pairs, ("A0", "A1", "B0", "B1"))),
          "Frozen fixture identities")
    inputs = read_json(prepared / "inputs.json")
    cad = [(x["pair_id"], x["entity"]) for x in inputs["cad"]]
    check(cad == list(product(pairs, ("A", "B"))), "Wrong CAD coverage or ordering")
    texts = set()
    for pair in pairs.values():
        for entity, variant in product(("A", "B"), VARIANTS):
            text = pair["captions"][entity][variant]
            check(sha256(text.encode()).hexdigest() == pair["caption_sha256"][entity][variant],
                  "Incorrect caption text fingerprint")
            texts.add(text)
    check(inputs["texts"] == sorted(texts), "Wrong caption coverage or ordering")

    count = 0
    for row in inputs["cad"]:
        entity, pair_id = row["entity"], row["pair_id"]
        check(row["source_history"] == entity + "0" and row["stem"] == f"{pair_id}-{entity}",
              "Representative history or filename changed")
        old = histories[pair_id, entity + "0"]
        measured = row["measured_in_step"]
        check(row["raw_program_sha256"] == old["raw_program_sha256"]
              and measured["status"] == "ok"
              and measured["measured_binding"] == old["geometry"]["raw"]["measured_binding"],
              "The exported solid's relation does not agree with the fixture")
        matched = row["verified_history_links"]
        check(all(r["status"] == "verified" and r["relative_symmetric_difference"] <= 2e-6
                  for r in matched["histories"])
              and all(r["status"] == "verified" and r["relative_symmetric_difference"] >= 1e-3
                      for r in matched["opposite_entities"])
              and len(matched["histories"]) == 2 and len(matched["opposite_entities"]) == 4
              and row["step_bbox_delta"] <= 1e-5 and row["step_relative_symmetric_difference"] <= 2e-6,
              "Missing truth or round-trip geometry evidence")
        topology = row["topology"]
        check(topology["retained_edge_occurrences"] == topology["attached_edges"]
              and topology["two_face_entries"] > 0
              and not topology["truncated_faces"] and not topology["truncated_edges"], "Bad B-Rep topology")
        check(set(row["files"]) == {".brep", ".step", ".stl", ".npz", ".png", ".npy"}, "Missing geometry artifact")
        for suffix, expected in row["files"].items():
            check(file_hash(prepared / (row["stem"] + suffix)) == expected, "Altered geometry file")
            count += 1
    return pairs, cad, sorted(texts), count


def audit_method(method, output, prepared, pairs, cad, texts, exit_file):
    dim = DIMS[method]
    check(exit_file.read_text(encoding="utf-8").strip() == "0", f"{method} did not exit successfully")
    axis = read_json(output / "axes.json")
    result = read_json(output / "result.json")
    check(axis["input_manifest_sha256"] == file_hash(prepared / "inputs.json")
          and axis["cad"] == [list(x) for x in cad] and axis["texts"] == texts, "Changed score axes")
    check(result["coverage_cads"] == 80 and result["unique_texts"] == len(texts)
          and result["cpu"] and result["encoder_dtype"] == "float32", "Incomplete model coverage")
    options = result["encoder"]
    check(len(axis["text_token_counts"]) == len(texts)
          and all(0 < x <= options["window"] for x in axis["text_token_counts"]), "Caption truncation")
    if method == "mmcad":
        check(options["strict"]["missing"] == 0 and options["strict"]["unexpected"] == 0
              and len(options["gate"]) == 3 and all(x["cosine"] > .999 for x in options["gate"]),
              "MM-CAD weight/reference mismatch")
    if method == "clip_top":
        from PIL import Image

        path = output / "visibility.json"
        check(file_hash(path) == options["visibility_sha256"], "Changed top-view manifest")
        visibility = read_json(path)
        rows = visibility["renders"]
        check(visibility["axis"] == "+Z" and len(rows) == len(cad), "Wrong top-view grid")
        geometry = read_json(prepared / "inputs.json")["cad"]
        for info, solid in zip(rows, geometry, strict=True):
            path = output / "top_renders" / (solid["stem"] + ".png")
            check(info["case"] == solid["stem"] and info["mesh_sha256"] == solid["files"][".stl"]
                  and file_hash(path) == info["sha256"], "Changed top-view input")
            with Image.open(path) as image:
                check(image.size == (448, 448), "Wrong rendered image size")
                image = np.asarray(image)
            check(len(info["hole_center_pixels"]) == 2 and all(
                np.all(image[y-1:y+2, x-1:x+2] == 255) for x, y in info["hole_center_pixels"]
            ), "Through-hole centers not visible")
    with np.load(output / "embeddings.npz", allow_pickle=False) as z:
        x, y, scores = z["text"], z["cad"], z["similarities"]
    check(x.shape == (len(texts), dim) and y.shape == (len(cad), dim) and scores.shape == (len(texts), len(cad)),
          "Wrong embedding axes")
    check(all(a.dtype == np.float64 and np.isfinite(a).all() for a in (x, y, scores)), "Wrong score dtype/numerics")
    for m in (x, y):
        check(np.allclose(np.sum(m * m, axis=1), 1, atol=2e-14, rtol=0), "Embeddings are not unit vectors")
    deltas = [abs(math.fsum((x[i] * y[j]).tolist()) - scores[i, j])
              for i, j in product(range(len(texts)), range(len(cad)))]
    check(max(deltas) <= 2e-14, "Recorded cosine differs from independent math.fsum")
    check(result["identity_collisions_cad_pairs"] == int(np.sum(np.all(y[::2] == y[1::2], axis=1)))
          and result["score_collisions_cad_pairs"] == int(np.sum(np.all(scores[:, ::2] == scores[:, 1::2], axis=0))),
          "Incorrect representation collision counts")

    views = read_rows(output / "paired_views.jsonl")
    table = {(r["pair_id"], r["order"], r["variant"]): r for r in views}
    check(len(views) == len(table) == 320 and set(table) == set(product(pairs, range(2), VARIANTS)),
          "Non-exact recognition coverage")
    groups, overall = defaultdict(Counter), Counter()
    for pair_id, pair in pairs.items():
        tallies = Counter(pairs=1)
        cj = [cad.index((pair_id, e)) for e in ("A", "B")]
        for order, variant in product(range(2), VARIANTS):
            ti = [texts.index(pair["captions"][e][variant]) for e in ("A", "B")]
            row = table[pair_id, order, variant]
            aa, ab, ba, bb = [float(scores[t, c]) for t, c in product(ti, cj)]
            check(row["text_indices"] == ti and row["cad_indices"] == cj
                  and row["scores_row_major"] == [aa, ab, ba, bb], "Swapped caption/candidate identity")
            left, right, up, down = aa - ba, bb - ab, aa - ab, bb - ba
            ds = (left, right, up, down)
            both_cads, both_captions = aa > ba and bb > ab, aa > ab and bb > ba
            swap = aa < ba and bb < ab
            m = row["metrics"]
            check(m["cad_choose_caption"] == list(ds[:2]) and m["caption_choose_cad"] == list(ds[2:])
                  and m["ties"] == ds.count(0) and m["D"] == math.fsum((aa, bb, -ba, -ab))
                  and m["both_cads"] == both_cads and m["both_captions"] == both_captions
                  and m["swapped_both_cads"] == swap, "Incorrect strict view verdict")
            tallies.update(views=1, both_cads=both_cads, both_captions=both_captions, swapped=swap,
                           ties=ds.count(0), D_positive=math.fsum((aa, bb, -ba, -ab)) > 0,
                           cad_correct=(aa > ba) + (bb > ab), caption_correct=(aa > ab) + (bb > ba),
                           small_nonzero_margins=sum(0 < abs(d) <= 1e-6 for d in ds))
        tallies["joint"] = int(tallies["both_cads"] == 8)
        tallies["joint_caption"] = int(tallies["both_captions"] == 8)
        tallies["joint_swap"] = int(tallies["swapped"] == 8)
        groups[f"{pair['design']['family']}/{pair['design']['attribute']}"].update(tallies)
        overall.update(tallies)
    check(result["scores"] == {"overall": overall, "strata": groups}, "Published totals do not reproduce")
    return {"model": method, "checks": {"cosines": len(deltas), "expanded_cells": len(views) * 4},
            "maximum_cosine_delta": max(deltas), "overall": overall, "strata": groups,
            "embeddings_sha256": file_hash(output / "embeddings.npz"),
            "views_sha256": file_hash(output / "paired_views.jsonl"), "result_sha256": file_hash(output / "result.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    for method in DIMS:
        p.add_argument(f"--{method.replace('_', '-')}", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    pairs, cad, texts, count = geometry_axes(a.fixture, a.prepared)
    results = {method: audit_method(method, path, a.prepared, pairs, cad, texts,
                                   path.parent / f"{method}.exit")
               for method in DIMS for path in (getattr(a, method),) if path is not None}
    check(results, "Select at least one score artifact")
    with a.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"all_passed": True, "geometry_files_checked": count, "models": results}, stream,
                  indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {count} geometry files and {len(results)} score artifacts", flush=True)


if __name__ == "__main__":
    main()
