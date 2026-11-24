#!/usr/bin/env python3
"""Replay frozen development solids and score three released retrieval baselines."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.src.idea01 import in_domain_pilot_geometry as geo
from run_910a_text2cad_dev_recognition import ENTITIES, VARIANTS, load_fixtures, metrics

E1_REV = "17293e1c6bb7b840b40718203181abedf1343582"
CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_WEIGHT = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
POINT_REV = "d771a992218de10b967eea690e8f474aa96dd758"
BIGG_REV = "743c27bd53dfe508a0ade0f50698f99b39d03bec"
DEMO_REV = "70dbc29fa30520cb78b4982de671f90600c08685"
POINT_WEIGHT = "34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9"
BIGG_WEIGHT = "0d5318839ad03607c48055c45897c655a14c0276a79f6b867934ddd073760e39"
FPS_SHIM = {
    "dgl/__init__.py": "f87298c80354d0be5e606272c3f070f32da25b1d7f9213743947950ea2fd013f",
    "dgl/geometry.py": "fa375e11c3e97be3d6683ede1ea5acb6d6d330ae08cd7e107e221cb0b813cdd7",
}


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    h = sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def log(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def revision(repo):
    return subprocess.check_output(("git", "-C", str(repo), "rev-parse", "HEAD"), text=True).strip()


def prior_mmcad(runtime):
    repo = runtime / "worktrees/codex-e1-mmcad-20261120"
    status = subprocess.check_output(("git", "-C", str(repo), "status", "--porcelain"), text=True)
    if revision(repo) != E1_REV or status:
        raise ValueError("Expected the clean, previously reviewed E1 MM-CAD runner")
    sys.path.insert(0, str(repo / "experiments/scripts"))
    from run_mmcad_e1_20261120 import load_model, step_from_brep, topology, checked_features
    return SimpleNamespace(load_model=load_model, step_from_brep=step_from_brep, topology=topology,
                           checked_features=checked_features, revision=E1_REV)


def fixture_axes(reference):
    pairs, _, histories = load_fixtures(reference)
    texts = sorted({pair["captions"][entity][view] for pair in pairs
                    for entity, view in product(ENTITIES, VARIANTS)})
    for pair in pairs:
        for entity, view in product(ENTITIES, VARIANTS):
            text = pair["captions"][entity][view]
            if sha256(text.encode()).hexdigest() != pair["caption_sha256"][entity][view]:
                raise ValueError("Frozen text identity differs")
    return pairs, histories, texts


def raw_body(sequence_type, fixture, design):
    raw = fixture["raw_program"]
    if geo.fingerprint(raw) != fixture["raw_program_sha256"]:
        raise ValueError("Frozen raw program fingerprint differs")
    sequence = sequence_type.from_dict(raw)
    for sketch in sequence.sketch_seq:
        frame = sketch.coordsystem.metadata
        for channel in ("origin", "x_axis", "y_axis", "z_axis"):
            frame[channel] = frame[channel].astype("float64", copy=False)
    body, _ = geo.strict_replay(sequence)
    actual = geo.measure_binding(body, design, fixture["entity"])
    expected = fixture["geometry"]["raw"]
    if (actual["status"] != "ok" or actual["status"] != expected["status"]
            or list(actual["measured_binding"]) != expected["measured_binding"]
            or not np.allclose(actual["bounds"], expected["bounds"], rtol=0, atol=1e-9)
            or not np.isclose(actual["volume"], expected["volume"], rtol=1e-10)):
        raise ValueError(f"Raw replay differs for {fixture['pair_id']}/{fixture['history']}")
    return body, actual


def render_and_sample(body, base, seed):
    import trimesh
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    from run_clip_g5_20261115 import render_mesh_png

    mesher = BRepMesh_IncrementalMesh(body, 0.001, False, 0.5, False)
    if not mesher.IsDone() or not StlAPI_Writer().Write(body, str(base.with_suffix(".stl"))):
        raise ValueError(f"OCC tessellation failed for {base.name}")
    mesh = trimesh.load_mesh(base.with_suffix(".stl"))
    if (not mesh.is_watertight or not np.isfinite(mesh.vertices).all() or mesh.area <= 0):
        raise ValueError(f"Invalid triangle mesh for {base.name}")
    render_mesh_png(mesh, base.with_suffix(".png"))
    points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=seed)
    lo, hi = points.min(axis=0), points.max(axis=0)
    points = (points - (hi + lo) / 2) / (np.max(hi - lo) / 2)
    if points.shape != (10000, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid point cloud for {base.name}")
    np.save(base.with_suffix(".npy"), points.astype(np.float32), allow_pickle=False)
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "sample_seed": seed}


def prepare(args):
    import OCC
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepTools import breptools
    from OCC.Core.gp import gp_Trsf
    from step_to_brep_fixed_20261118 import MAX_EDGES, MAX_FACES, extract

    prior = prior_mmcad(args.runtime)
    pairs, histories, texts = fixture_axes(args.reference)
    designs = {design.pair_id: design for design in geo.pilot_designs()}
    vendor = args.runtime / "pilot_assets_20261123/vendor/text2cad"
    sequence_type, provenance = geo.activate_text2cad(vendor)
    outputs = []
    for index, pair in enumerate(pairs):
        pair_id = pair["pair_id"]
        design = designs[pair_id]
        present = {key: raw_body(sequence_type, histories[pair_id, key], design)
                   for key in geo.HISTORIES}
        shapes = {key: value[0] for key, value in present.items()}
        measured = {key: value[1] for key, value in present.items()}
        links = geo.geometry_links(measured, shapes)
        if links["status"] != "verified":
            raise ValueError(f"Geometric history relationships failed for {pair_id}")
        bounds = np.array([measured[entity + "0"]["bounds"] for entity in ENTITIES])
        if not np.allclose(bounds[0], bounds[1], rtol=0, atol=1e-8):
            raise ValueError("Expected the same stock bounds for both entities")
        lo, hi = bounds[0, :3], bounds[0, 3:]
        scale = 1.0 / np.max(hi - lo)
        trsf = gp_Trsf()
        x, y, z = -lo * scale
        trsf.SetValues(scale, 0, 0, x, 0, scale, 0, y, 0, 0, scale, z)
        for entity in ENTITIES:
            source = entity + "0"
            fixture = histories[pair_id, source]
            base = args.out / f"{pair_id}-{entity}"
            body = BRepBuilderAPI_Transform(shapes[source], trsf, True).Shape()
            before = geo.measure_binding(body, design, entity)
            brep = base.with_suffix(".brep")
            step = base.with_suffix(".step")
            if not breptools.Write(body, str(brep)):
                raise ValueError(f"B-Rep export failed for {base.name}")
            _, reopened = prior.step_from_brep(brep, step)
            after = geo.measure_binding(reopened, design, entity)
            diff = sum(geo._volume(geo._cut(a, b)) for a, b in ((body, reopened), (reopened, body)))
            bbox_delta = max(abs(a - b) for a, b in zip(before["bounds"], after["bounds"], strict=True))
            ratio = diff / before["volume"]
            if (before["status"] != "ok" or after["status"] != "ok"
                    or bbox_delta > 1e-5 or ratio > 2e-6):
                raise ValueError(f"STEP solid roundtrip failed for {base.name}")
            features, topology = prior.checked_features(extract(str(step)), prior.topology(reopened),
                                                        MAX_FACES, MAX_EDGES)
            if topology["truncated_faces"] or topology["truncated_edges"]:
                raise ValueError(f"Topology truncated for {base.name}")
            np.savez_compressed(base.with_suffix(".npz"), **features)
            mesh = render_and_sample(reopened, base, 20261124 + index)
            outputs.append({"pair_id": pair_id, "entity": entity, "source_history": source,
                            "raw_program_sha256": fixture["raw_program_sha256"],
                            "stem": base.name, "verified_history_links": links, "raw_bounds": bounds[0].tolist(),
                            "scale": float(scale), "measured_in_step": after, "step_bbox_delta": bbox_delta,
                            "step_relative_symmetric_difference": ratio, "topology": topology, "mesh": mesh,
                            "files": {ext: digest(base.with_suffix(ext)) for ext in
                                      (".brep", ".step", ".npz", ".stl", ".png", ".npy")}})
        log(stage="geometry", complete_pairs=index + 1)
    save_json(args.out / "inputs.json", {"cad": outputs, "texts": texts, "cadseqproc": provenance,
              "pythonocc": OCC.VERSION, "trimesh": version("trimesh"), "loader_revision": prior.revision})


def unit_rows(rows, count, dimension):
    if rows.shape != (count, dimension) or rows.dtype != np.float32:
        raise ValueError(f"Unexpected embedding shape or dtype {rows.shape}/{rows.dtype}")
    rows = rows.astype(np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Encoder returned zero or non-finite vectors")
    return rows / norms


def mmcad(args, paths, texts):
    import torch
    prior = prior_mmcad(args.runtime)
    weights = args.runtime / "mmcad_weights"
    config = SimpleNamespace(upstream=args.runtime / "MMCAD-official", threads=args.threads,
                             checkpoint=weights / "baseline_trimodal_v4.pth",
                             stats=weights / "brep_norm_stats_v4.npz",
                             text_model=weights / "embeddinggemma-300m")
    _, mm, model, strict = prior.load_model(config)
    stats = mm.load_brep_stats(config.stats)

    def embed(raw):
        return mm.embed_brep(model, mm.prepare_brep(raw, *stats), dim=768, device="cpu").numpy()

    with torch.inference_mode():
        gate_root = args.runtime / "e1_mmcad_inputs/idea01_e1_mmcad_20261120/references"
        prior_gate = args.runtime / "runs/idea01_e1_mmcad_20261120/reference_gate.json"
        accepted = {row["file"]: row["sha256"] for row in json.loads(prior_gate.read_bytes())["references"]}
        if len(accepted) != 3:
            raise ValueError("Expected the three previously checked MM-CAD embedding fixtures")
        gate = []
        for name, pin in sorted(accepted.items()):
            path = gate_root / name
            if digest(path) != pin:
                raise ValueError(f"MM-CAD reference fixture differs: {name}")
            with np.load(path, allow_pickle=False) as z:
                features = {k[4:]: z[k] for k in z.files if k.startswith("raw_")}
                known = z["embedding"].reshape(-1).astype(np.float64)
                result = unit_rows(embed(features), 1, 768)[0]
                cosine = float(result @ (known / np.linalg.norm(known)))
            gate.append({"name": name, "sha256": pin, "cosine": cosine})
        if any(not r["cosine"] > 0.999 for r in gate):
            raise ValueError(f"MM-CAD reference replay failed: {gate}")
        log(stage="mmcad_reference_gate", references=gate, strict=strict)
        prompts = [f"title: none | text: {text}" for text in texts]
        tokenizer = model.text_encoder.tokenizer
        token_counts = [len(row) for row in tokenizer(prompts, truncation=False)["input_ids"]]
        t = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        text = mm.truncate(model.text_encoder(t["input_ids"], t["attention_mask"]), 768).numpy()
        chunks = []
        for i, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as z:
                chunks.append(embed(dict(z)))
            if (i + 1) % 20 == 0:
                log(stage="mmcad_cad", rows=i + 1)
    return np.concatenate(chunks), text, token_counts, 768, {"strict": strict, "gate": gate,
            "window": 512, "weight_sha256": digest(config.checkpoint), "stats_sha256": digest(config.stats)}


def openshape(args, paths, texts):
    import torch
    import open_clip
    root = args.runtime / "hf_home/hub"
    demo = root / f"models--OpenShape--openshape-demo-support/snapshots/{DEMO_REV}"
    point = root / f"models--OpenShape--openshape-pointbert-vitg14-rgb/snapshots/{POINT_REV}/model.pt"
    bigg = root / ("models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/"
                   f"{BIGG_REV}/open_clip_pytorch_model.bin")
    shim = args.runtime / "probe_shims"
    if (digest(point) != POINT_WEIGHT or digest(bigg) != BIGG_WEIGHT
            or any(digest(shim / name) != pin for name, pin in FPS_SHIM.items())):
        raise ValueError("OpenShape assets differ from the pinned benchmark assets")
    sys.path[:0] = [str(demo), str(shim)]
    import dgl
    import openshape as official
    if not Path(dgl.__file__).resolve().is_relative_to(shim):
        raise ValueError("Historical OpenShape FPS shim not in use")
    point_model = official.G14(torch.load(point, map_location="cpu", weights_only=True)).eval()
    text_model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=str(bigg), device="cpu")
    text_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    token_counts = [len(tokenizer.encode(text)) + 2 for text in texts]
    with torch.inference_mode():
        text = text_model.encode_text(tokenizer(texts)).numpy()
        chunks = []
        for i, path in enumerate(paths):
            xyz = np.load(path, allow_pickle=False)
            rgb = np.full_like(xyz, 0.5)
            point_tensor = torch.from_numpy(np.concatenate((xyz[:, [0, 2, 1]], rgb), axis=1).T.copy()[None])
            chunks.append(point_model(point_tensor).numpy())
            if (i + 1) % 20 == 0:
                log(stage="openshape_cad", rows=i + 1)
    return np.concatenate(chunks), text, token_counts, 1280, {
        "window": 77, "point_rev": POINT_REV, "text_rev": BIGG_REV, "demo_rev": DEMO_REV,
        "point_weight_sha256": POINT_WEIGHT, "text_weight_sha256": BIGG_WEIGHT,
        "demo_sources": {p.relative_to(demo).as_posix(): digest(p) for p in sorted(demo.rglob("*.py"))},
        "fps_shim_sources": FPS_SHIM,
    }


def clip(args, paths, texts):
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

    root = args.runtime / "binding_clip_assets_20261124" / CLIP_REV
    weights = digest(root / "pytorch_model.bin")
    if weights != CLIP_WEIGHT:
        raise ValueError("CLIP weight differs from the copied snapshot")
    model, loading = CLIPModel.from_pretrained(root, local_files_only=True, use_safetensors=False,
                                               output_loading_info=True)
    unexpected = sorted(loading.get("unexpected_keys", ()))
    if (loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs")
            or any(not key.endswith(".position_ids") for key in unexpected)):
        raise ValueError(f"CLIP weight load incomplete: {loading}")
    model.eval().to("cpu")
    processor = CLIPImageProcessor.from_pretrained(root, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(root, local_files_only=True)
    token_counts = [len(row) for row in tokenizer(texts, truncation=False)["input_ids"]]

    def projected(result):
        return (result if isinstance(result, torch.Tensor) else result.pooler_output).numpy()

    with torch.inference_mode():
        text = projected(model.get_text_features(**tokenizer(texts, padding=True, truncation=True,
                                                              max_length=77, return_tensors="pt")))
        chunks = []
        for offset in range(0, len(paths), 16):
            images = []
            for path in paths[offset:offset + 16]:
                with Image.open(path) as image:
                    content = image.convert("RGB")
                    if all(ch[0] == ch[1] for ch in content.getextrema()):
                        raise ValueError(f"Empty render: {path}")
                    images.append(content)
            chunks.append(projected(model.get_image_features(**processor(images=images, return_tensors="pt"))))
            log(stage="clip_cad", rows=offset + len(images))
    return np.concatenate(chunks), text, token_counts, 512, {"snapshot": CLIP_REV, "window": 77,
            "unexpected_position_buffers": unexpected, "weight_sha256": weights,
            "snapshot_sha256": {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix != ".bin"},
            "render_spec_sha256": digest(ROOT / "experiments/configs/clip_g5_render_spec_20261115.json")}


def top_images(args, cad_rows):
    """Produce one fixed XY projection, then use the measured centers to check visibility."""
    import trimesh
    from PIL import Image, ImageDraw

    directory = args.out / "top_renders"
    directory.mkdir()
    paths, evidence = [], []
    size = 448
    for row in cad_rows:
        source = args.inputs / (row["stem"] + ".stl")
        if digest(source) != row["files"][".stl"]:
            raise ValueError(f"Changed source mesh: {source}")
        mesh = trimesh.load_mesh(source)
        xy = np.asarray(mesh.vertices)[:, :2]
        low, high = xy.min(axis=0), xy.max(axis=0)
        center = (low + high) / 2
        scale = size * 0.8 / (high - low).max()

        def project(points):
            return (points - center) * (scale * np.array([1, -1])) + (size - 1) / 2

        canvas = Image.new("RGB", (2 * size, 2 * size), "white")
        draw = ImageDraw.Draw(canvas)
        uv = 2 * project(xy)
        for face in mesh.faces:
            draw.polygon([tuple(uv[i]) for i in face], fill="#9db8d9")
        rendered = canvas.resize((size, size), Image.Resampling.LANCZOS)
        path = directory / (row["stem"] + ".png")
        rendered.save(path)
        result = np.asarray(rendered)
        measured = row["measured_in_step"]
        centers = [np.rint(project(np.asarray(measured[k]["center_xy"]))).astype(int).tolist()
                   for k in ("near", "far")]
        visible_pixels = [bool((result[y-1:y+2, x-1:x+2] == 255).all()) for x, y in centers]
        if not all(visible_pixels) or all(k[0] == k[1] for k in rendered.getextrema()):
            raise ValueError(f"Unverified visible through holes: {path}")
        paths.append(path)
        evidence.append({"case": row["stem"], "sha256": digest(path),
                         "mesh_sha256": row["files"][".stl"], "hole_center_pixels": centers,
                         "center_3x3_white": visible_pixels})
    save_json(args.out / "visibility.json", {"axis": "+Z", "size": 448, "stock_fraction": 0.8,
                                             "supersample": 2, "renders": evidence})
    return paths


def evaluate(pairs, texts, cad_rows, similarities, output):
    positions = {(row["pair_id"], row["entity"]): i for i, row in enumerate(cad_rows)}
    text_index = {text: i for i, text in enumerate(texts)}
    totals, by_stratum = Counter(), defaultdict(Counter)
    with (output / "paired_views.jsonl").open("xb") as stream:
        for pair in pairs:
            pair_id = pair["pair_id"]
            columns = [positions[pair_id, entity] for entity in ENTITIES]
            counts = Counter(pairs=1)
            for order, view in product(range(2), VARIANTS):
                indices = [text_index[pair["captions"][entity][view]] for entity in ENTITIES]
                scores = similarities[np.ix_(indices, columns)].ravel().tolist()
                m = metrics(scores)
                margins = m["cad_choose_caption"] + m["caption_choose_cad"]
                counts.update(views=1, both_cads=m["both_cads"], both_captions=m["both_captions"],
                              swapped=m["swapped_both_cads"], ties=m["ties"], D_positive=m["D"] > 0,
                              cad_correct=sum(v > 0 for v in m["cad_choose_caption"]),
                              caption_correct=sum(v > 0 for v in m["caption_choose_cad"]),
                              small_nonzero_margins=sum(0 < abs(v) <= 1e-6 for v in margins))
                stream.write((json.dumps({"pair_id": pair_id, "order": order, "variant": view,
                                          "text_indices": indices, "cad_indices": columns,
                                          "scores_row_major": scores, "metrics": m}, allow_nan=False) + "\n").encode())
            counts["joint"] = int(counts["both_cads"] == len(VARIANTS) * 2)
            counts["joint_caption"] = int(counts["both_captions"] == len(VARIANTS) * 2)
            counts["joint_swap"] = int(counts["swapped"] == len(VARIANTS) * 2)
            totals.update(counts)
            by_stratum["/".join((pair["design"]["family"], pair["design"]["attribute"]))].update(counts)
    return dict(overall=totals, strata=by_stratum)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("prepare", "mmcad", "openshape", "clip", "clip_top"))
    cli.add_argument("--runtime", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--inputs", type=Path)
    cli.add_argument("--threads", type=int, default=12)
    args = cli.parse_args()
    args.out.mkdir()
    started = time.monotonic()
    if args.stage == "prepare":
        prepare(args)
    else:
        import torch
        torch.set_num_threads(args.threads)
        pairs, _, texts = fixture_axes(args.reference)
        prepared = json.loads((args.inputs / "inputs.json").read_bytes())
        cad_rows = prepared["cad"]
        keys = [(row["pair_id"], row["entity"]) for row in cad_rows]
        if prepared["texts"] != texts or keys != [(p["pair_id"], e) for p, e in product(pairs, ENTITIES)]:
            raise ValueError("Prepared geometry/text axes differ")
        if args.stage == "clip_top":
            paths, encoder = top_images(args, cad_rows), clip
        else:
            suffix, encoder = {"mmcad": (".npz", mmcad), "openshape": (".npy", openshape),
                               "clip": (".png", clip)}[args.stage]
            paths = [args.inputs / (row["stem"] + suffix) for row in cad_rows]
            if any(digest(p) != row["files"][suffix] for p, row in zip(paths, cad_rows, strict=True)):
                raise ValueError("Prepared encoder input differs from the measured geometry manifest")
        cad, text, counts, dimension, details = encoder(args, paths, texts)
        if args.stage == "clip_top":
            details.pop("render_spec_sha256")
            details.update(render="stock-oriented single +Z view",
                           visibility_sha256=digest(args.out / "visibility.json"))
        cad, text = unit_rows(cad, len(keys), dimension), unit_rows(text, len(texts), dimension)
        if len(counts) != len(texts) or any(n > details["window"] for n in counts):
            raise ValueError(f"Frozen captions exceed {args.stage} context: {counts}")
        similarities = text @ cad.T
        np.savez_compressed(args.out / "embeddings.npz", cad=cad, text=text, similarities=similarities)
        save_json(args.out / "axes.json", {"cad": keys, "texts": texts, "text_token_counts": counts,
                                          "input_manifest_sha256": digest(args.inputs / "inputs.json")})
        evaluation = evaluate(pairs, texts, cad_rows, similarities, args.out)
        within_pair = similarities[:, 0::2] - similarities[:, 1::2]
        save_json(args.out / "result.json", {"stage": args.stage, "dimension": dimension, "cpu": True,
                  "encoder_dtype": "float32", "cosine_dtype": "float64", "coverage_cads": len(keys),
                  "unique_texts": len(texts), "identity_collisions_cad_pairs": int((cad[::2] == cad[1::2]).all(1).sum()),
                  "score_collisions_cad_pairs": int((within_pair == 0).all(0).sum()), "encoder": details,
                  "scores": evaluation, "elapsed_s": time.monotonic() - started,
                  "git_sha": revision(ROOT), "torch": torch.__version__})
    log(stage=args.stage, status="complete", elapsed_s=time.monotonic() - started)


if __name__ == "__main__":
    main()
