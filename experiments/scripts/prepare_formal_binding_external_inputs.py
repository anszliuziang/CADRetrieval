#!/usr/bin/env python3
"""Prepare fixed STEP, B-Rep features, point clouds and images for Table 4 on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from audit_formal_binding_effective_inputs import read_fixtures
from evaluate_formal_binding import keyed, read_pinned, require, verify_review_item
from idea01 import formal_binding_families as families
from idea01 import formal_binding_occ as occ
from idea01 import in_domain_pilot_geometry as geo
import run_dev_geometry_external_baselines as development
from run_in_domain_pilot_geometry import attempt


METHODS = ("mmcad", "openshape", "clip", "clip_top")
ENTITIES = ("A", "B")
SURFACE_POINT_SEED = 20261124
SOURCE_BOUNDS_TOL = 1e-7
STEP_BOUNDS_TOL = 1e-5


def mmcad_helpers(checkout):
    checkout = checkout.resolve(strict=True)
    status = subprocess.check_output(("git", "-C", str(checkout), "status", "--porcelain"), text=True)
    require(development.revision(checkout) == development.E1_REV and not status,
            "expected the clean, pinned E1 MM-CAD checkout")
    scripts = checkout / "experiments" / "scripts"
    sys.path.insert(0, str(scripts))
    import run_mmcad_e1_20261120 as prior

    require(Path(prior.__file__).resolve().parent == scripts, "MM-CAD helper loaded from a different checkout")
    return prior


def load_sources(fixtures, prepared_sha, smoke):
    pairs, histories, prepared, receipts = read_fixtures(fixtures, prepared_sha)
    packet = fixtures / "review_packet"
    items = keyed(development_inputs(packet / "items.jsonl", prepared["files_sha256"]["items"], receipts),
                  lambda row: row["review_id"])
    private = keyed(development_inputs(fixtures / "review_key.jsonl", prepared["files_sha256"]["review_key"], receipts),
                    lambda row: row["review_id"])
    parents = [pair for pair in pairs.values() if pair["kind"] != "coordinate"]
    require(items.keys() == private.keys() == {pair["review_id"] for pair in parents},
            "the public and private review axes must cover every parent pair")
    if prepared["mode"] == "full":
        require(len(parents) == 960, "expected 960 prepared parent pairs in the full bank")
    if smoke is None:
        selected = parents
    else:
        representatives = {}
        for pair in parents:
            representatives.setdefault(pair["family"], pair["pair_id"])
        first = list(representatives.values())
        priority = first + [pair["pair_id"] for pair in parents if pair["pair_id"] not in first]
        chosen = set(priority[:smoke])
        selected = [pair for pair in parents if pair["pair_id"] in chosen]
    indices = {pair["pair_id"]: index for index, pair in enumerate(parents)}
    sources = []
    for pair in selected:
        pid, rid = pair["pair_id"], pair["review_id"]
        item, private_item = items[rid], private[rid]
        error = None
        parts = {}
        try:
            verify_review_item(pair, histories, item, private_item, packet, receipts)
            for part in item["parts"]:
                mapping = private_item["parts"][part["part_id"]]
                entity = mapping["entity"]
                source = histories[pid, mapping["source_history"]]
                parts[entity] = {"path": str(packet / part["step"]), "step_sha256": part["step_sha256"],
                                 "source_history": mapping["source_history"],
                                 "raw_program_sha256": source["raw_program_sha256"],
                                 "registered_binding": source["registered_binding"], "raw": source["geometry"]["raw"]}
        except (KeyError, OSError, ValueError) as exc:
            error = {"status": "failed", "stage": "blinded_step_identity", "detail": f"{type(exc).__name__}: {exc}"}
        sources.append((pair, parts, error, SURFACE_POINT_SEED + indices[pid]))
    selected_ids = {pair["pair_id"] for pair in selected}
    axis = [pair for pair in pairs.values() if pair["pair_id"] in selected_ids or pair.get("parent_pair_id") in selected_ids]
    return sources, axis, prepared, receipts


def development_inputs(path, digest, receipts):
    return [json.loads(line) for line in read_pinned(path, digest, receipts).splitlines()]


def read_step_source(source, pair):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    path = Path(source["path"])
    require(development.digest(path) == source["step_sha256"], "input STEP differed before replay")
    reader = STEPControl_Reader()
    require(reader.ReadFile(str(path)) == IFSelect_RetDone and reader.TransferRoots() == 1,
            "checked review STEP no longer reads as a single root")
    shape = reader.OneShape()
    require(development.digest(path) == source["step_sha256"], "input STEP changed during replay")
    actual = measure(shape, pair, source)
    raw = source["raw"]
    require(np.allclose(actual["bounds"], raw["bounds"], rtol=0, atol=SOURCE_BOUNDS_TOL)
            and math.isclose(actual["volume"], raw["volume"], rel_tol=1e-7),
            "review STEP bounds or volume differs from the sealed raw geometry")
    return shape, actual


def measure(shape, pair, source):
    result = occ.measure(shape, pair["family"], pair["attribute"], pair["kind"], source["registered_binding"])
    require(result["status"] == "ok", "solid differs from the registered hole relation")
    return result


def shared_pose(source_shapes):
    from OCC.Core.gp import gp_Trsf

    bounds = np.asarray([source_shapes[entity][1]["bounds"] for entity in ENTITIES])
    require(np.allclose(bounds[0], bounds[1], rtol=0, atol=1e-8), "expected identical A/B stock bounds")
    lo, hi = bounds[0, :3], bounds[0, 3:]
    scale = float(1.0 / np.max(hi - lo))
    transform = gp_Trsf()
    x, y, z = -lo * scale
    transform.SetValues(scale, 0, 0, x, 0, scale, 0, y, 0, 0, scale, z)
    return transform, bounds[0].tolist(), scale


def normalized_step(prior, shape, pose, pair, source, base):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepTools import breptools

    body = BRepBuilderAPI_Transform(shape, pose, True).Shape()
    before = measure(body, pair, source)
    brep, step = (base.with_suffix(ext) for ext in (".brep", ".step"))
    require(breptools.Write(body, str(brep)), "B-Rep write failed")
    _, reopened = prior.step_from_brep(brep, step)
    after = measure(reopened, pair, source)
    witness = occ.compare_same_body(body, reopened)
    delta = max(abs(a - b) for a, b in zip(before["bounds"], after["bounds"], strict=True))
    require(delta <= STEP_BOUNDS_TOL and witness["status"] == "verified",
            "normalized STEP roundtrip changed bounds or solid")
    return reopened, after, {"bbox_max_delta": delta, **witness}


def mmcad_features(prior, body, base):
    from step_to_brep_fixed_20261118 import MAX_EDGES, MAX_FACES, extract

    features, topology = prior.checked_features(extract(str(base.with_suffix(".step"))), prior.topology(body),
                                                MAX_FACES, MAX_EDGES)
    require(not topology["truncated_faces"] and not topology["truncated_edges"], "MM-CAD topology truncated")
    np.savez_compressed(base.with_suffix(".npz"), **features)
    return {"status": "verified", "topology": topology}


def make_mesh(body, path):
    import trimesh
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    mesher = BRepMesh_IncrementalMesh(body, 0.001, False, 0.5, False)
    require(mesher.IsDone() and StlAPI_Writer().Write(body, str(path)), "pinned OCC tessellation failed")
    mesh = trimesh.load_mesh(path)
    require(mesh.is_watertight and np.isfinite(mesh.vertices).all() and mesh.area > 0, "invalid triangle mesh")
    return mesh


def point_cloud(mesh, path, seed):
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=seed)
    lo, hi = points.min(axis=0), points.max(axis=0)
    radius = np.max(hi - lo) / 2
    require(radius > 0, "empty sampled bounding box")
    points = (points - (hi + lo) / 2) / radius
    require(points.shape == (10000, 3) and np.isfinite(points).all(), "invalid 10k surface sample")
    np.save(path, points.astype(np.float32), allow_pickle=False)
    return {"status": "verified", "sample_seed": seed, "points": len(points)}


def iso_render(mesh, path, measurement):
    from PIL import Image
    from run_clip_g5_20261115 import render_mesh_png

    render_mesh_png(mesh, path)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    require(pixels.size and np.any(pixels != pixels[0, 0]), "empty isometric render")
    visibility, error = attempt("clip_iso_visibility", iso_opening_pixels, mesh, pixels, measurement)
    return {"status": "verified", "pixel_shape": list(pixels.shape), "visibility": error or visibility}


def iso_opening_pixels(mesh, pixels, measurement):
    """Project the STEP-derived openings with the frozen isometric camera."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=200)
    try:
        axes = fig.add_subplot(111, projection="3d")
        axes.add_collection3d(Poly3DCollection(vertices[faces], facecolor="#9db8d9", edgecolor="none"))
        center = vertices.mean(axis=0)
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        for setter, coordinate in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim), center, strict=True):
            setter(coordinate - radius, coordinate + radius)
        axes.view_init(elev=30, azim=45)
        axes.set_axis_off()
        axes.set_box_aspect((1, 1, 1))
        fig.canvas.draw()
        crop = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
        height, width = pixels.shape[:2]
        require(abs(crop.width - width) <= 2 and abs(crop.height - height) <= 2,
                "could not align the isometric projection with the frozen cropped image")
        checks = {}
        for role in ("near", "far"):
            opening = measurement[role]
            x, y, _ = proj3d.proj_transform(*opening["center_xy"], opening["z"], axes.get_proj())
            u, v = axes.transData.transform((x, y))
            col, row = round(u - crop.x0), round(crop.y1 - v)
            inside = 1 <= row < height - 1 and 1 <= col < width - 1
            white = inside and bool((pixels[row - 1:row + 2, col - 1:col + 2] == 255).all())
            checks[role] = {"pixel_xy": [col, row], "inside_image": inside, "white_center_3x3": white}
        return {"status": "both_white" if all(item["white_center_3x3"] for item in checks.values()) else "limited",
                "camera": {"azimuth_degrees": 45, "elevation_degrees": 30}, "opening_checks": checks}
    finally:
        plt.close(fig)


def top_render(base, measurement, mesh_digest):
    with TemporaryDirectory(prefix=".top-", dir=base.parent) as work:
        work = Path(work)
        row = {"stem": base.name, "files": {".stl": mesh_digest}, "measured_in_step": measurement}
        paths, error = attempt("clip_top_visibility", development.top_images,
                               SimpleNamespace(out=work, inputs=base.parent), [row])
        if error:
            diagnostic = work / "top_renders" / (base.name + ".png")
            if diagnostic.is_file():
                diagnostic.replace(base.with_suffix(".top.png"))
            return {**error, "visibility": {"status": "unverified"}}
        rendered, = paths
        rendered.replace(base.with_suffix(".top.png"))
        evidence, = json.loads((work / "visibility.json").read_bytes())["renders"]
    return {"status": "verified", "visibility": {"status": "both_white", **evidence}}


def failed_modalities(error):
    return {name: dict(error) for name in METHODS}


def prepare_pair(out, prior, pair, sources, identity_error, seed):
    pid, family = pair["pair_id"], pair["family"]
    source_shapes, error = (None, identity_error)
    if error is None:
        source_shapes, error = attempt("source_step", lambda: {
            entity: read_step_source(sources[entity], pair) for entity in ENTITIES
        })
    if error is None:
        shared, error = attempt("shared_pose", shared_pose, source_shapes)
    if error:
        return report_pair(pair, {}, error)
    pose, bounds, scale = shared
    rows = {}
    normalized_shapes = {}
    for entity in ENTITIES:
        stem = f"cad/{family}/{pid}-{entity}"
        base = out / stem
        base.parent.mkdir(parents=True, exist_ok=True)
        prepared, error = attempt("normalized_step", normalized_step, prior, source_shapes[entity][0], pose,
                                  pair, sources[entity], base)
        row = {"pair_id": pid, "entity": entity, "source_history": sources[entity]["source_history"],
               "raw_program_sha256": sources[entity]["raw_program_sha256"],
               "source_step_sha256": sources[entity]["step_sha256"], "stem": stem,
               "raw_bounds": bounds, "scale": scale}
        if error:
            row["models"] = failed_modalities(error)
        else:
            reopened, measured, roundtrip = prepared
            normalized_shapes[entity] = reopened
            row.update(measured_in_step=measured, normalized_step_roundtrip=roundtrip)
            mmcad, mmcad_error = attempt("mmcad_checked_features", mmcad_features, prior, reopened, base)
            models = {"mmcad": mmcad_error or mmcad}
            mesh, mesh_error = attempt("shared_tessellation", make_mesh, reopened, base.with_suffix(".stl"))
            if mesh_error:
                models.update({name: dict(mesh_error) for name in METHODS[1:]})
            else:
                points, point_error = attempt("openshape_surface_sample", point_cloud, mesh, base.with_suffix(".npy"), seed)
                models["openshape"] = point_error or points
                image, image_error = attempt("clip_iso_render", iso_render, mesh, base.with_suffix(".png"), measured)
                models["clip"] = image_error or image
                top, top_error = attempt("clip_top_render", top_render, base, measured,
                                         development.digest(base.with_suffix(".stl")))
                models["clip_top"] = top_error or top
                row["mesh"] = {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}
            row["models"] = models
        row["files"] = {ext: development.digest(base.with_suffix(ext))
                        for ext in (".step", ".brep", ".npz", ".npy", ".stl", ".png", ".top.png")
                        if base.with_suffix(ext).is_file()}
        rows[entity] = row
    separation = None
    if len(normalized_shapes) == 2:
        evidence, error = attempt("normalized_opposition", occ.compare_same_body,
                                  normalized_shapes["A"], normalized_shapes["B"])
        if error is None:
            separation = evidence["relative_symmetric_difference"]
            if separation < 1e-3:
                error = {"status": "failed", "stage": "normalized_opposition", "detail": "A/B STEP solids coincide"}
    else:
        error = {"status": "failed", "stage": "normalized_step", "detail": "one or both STEP witnesses missing"}
    return report_pair(pair, rows, error, separation)


def report_pair(pair, rows, error, separation=None):
    models = {}
    for method in METHODS:
        records = {entity: row["models"][method] for entity, row in rows.items()}
        failed = sorted({record["stage"] for record in records.values() if record["status"] != "verified"})
        if error:
            failed.append(error["stage"])
        models[method] = {"status": "failed" if failed else "verified",
                          "failure_reason": "+".join(dict.fromkeys(failed)) if failed else None,
                          "per_entity": records}
    return {"pair_id": pair["pair_id"], "family": pair["family"], "attribute": pair["attribute"],
            "kind": pair["kind"], "models": models, "common_geometry_error": error,
            "normalized_relative_symmetric_difference": separation}, rows


def run_family(task):
    out, checkout, items = task
    prior = mmcad_helpers(checkout)
    return [prepare_pair(out, prior, *case) for case in items]


def prepare(out, checkout, sources, workers):
    if workers == 1:
        return run_family((out, checkout, sources))
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    tasks = [(out, checkout, [case for case in sources if case[0]["family"] == family]) for family in families.FAMILIES]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return [value for group in pool.map(run_family, (task for task in tasks if task[2])) for value in group]


def write_outputs(out, prepared, axis, receipts, fixtures):
    from contextlib import ExitStack

    rows = {pair["pair_id"]: pair for pair, _ in prepared}
    method_groups = {}
    files = {"pairs": out / "pairs.jsonl", **{name: out / f"effective_geometry_{name}.jsonl" for name in METHODS}}
    with ExitStack() as stack:
        outputs = {name: stack.enter_context(path.open("xb")) for name, path in files.items()}
        for pair in axis:
            pid = pair["pair_id"]
            if pair["kind"] == "coordinate":
                parent = rows[pair["parent_pair_id"]]
                raw_ok = pair["coordinate_geometry_verified"]
                models = {method: {**result, "status": result["status"] if raw_ok else "failed",
                                   "failure_reason": result["failure_reason"] if raw_ok else "raw_coordinate_unverified"}
                          for method, result in parent["models"].items()}
                record = {"pair_id": pid, "parent_pair_id": parent["pair_id"], "family": pair["family"],
                          "attribute": pair["attribute"], "kind": "coordinate", "models": models,
                          "inherits_parent_assets": raw_ok}
            else:
                record = rows[pid]
            geo.jsonl_write(outputs["pairs"], record)
            for name, result in record["models"].items():
                counts = method_groups.setdefault((name, pair["family"], pair["attribute"], pair["kind"]), Counter())
                counts["denominator"] += 1
                counts[result["status"]] += 1
                if result["failure_reason"]:
                    counts["failure:" + result["failure_reason"]] += 1
                if pair["kind"] != "coordinate":
                    for item in result["per_entity"].values():
                        status = item.get("visibility", {}).get("status")
                        if status:
                            counts["visibility:" + status] += 1
                geo.jsonl_write(outputs[name], {
                    "pair_id": pid, "status": result["status"], "failure_reason": result["failure_reason"],
                    "evidence": f"pairs.jsonl/{pid}/{name}", "pair_preparation_sha256": geo.fingerprint(record),
                    "effective_coordinate_conflict": False,
                })
    cad_rows = [cad for _, by_entity in prepared for cad in by_entity.values()]
    texts = sorted({text for pair in axis if pair["kind"] != "coordinate" for view in pair["views"] for text in view["texts"]})
    inputs_path = out / "inputs.json"
    development.save_json(inputs_path, {"cad": cad_rows, "texts": texts,
                          "fixture": str(fixtures.resolve()), "fixture_sha256": receipts,
                          "loader_revision": development.E1_REV})
    files["inputs"] = inputs_path
    return method_groups, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--prepared-summary-sha256", required=True)
    parser.add_argument("--mmcad-checkout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", type=int, help="Cap at N parent pairs, preferring one per family")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (args.smoke is not None and args.smoke < 1) or not 1 <= args.workers <= len(families.FAMILIES):
        parser.error("use a positive smoke cap and 1–12 family workers")
    sources, axis, origin, receipts = load_sources(args.fixtures, args.prepared_summary_sha256, args.smoke)
    require(bool(sources), "no parent pairs selected")
    mmcad_helpers(args.mmcad_checkout)
    args.out.mkdir(parents=True, exist_ok=False)
    prepared = prepare(args.out, args.mmcad_checkout, sources, args.workers)
    require([row["pair_id"] for row, _ in prepared] == [pair["pair_id"] for pair, _, _, _ in sources],
            "external preparation reordered or omitted a frozen parent pair")
    groups, files = write_outputs(args.out, prepared, axis, receipts, args.fixtures)
    deps = {}
    for name in ("trimesh", "matplotlib", "numpy", "pillow"):
        try:
            deps[name] = version(name)
        except PackageNotFoundError:
            deps[name] = None
    import OCC
    from run_clip_g5_20261115 import SPEC as CLIP_SPEC

    deps["pythonocc_core"] = OCC.VERSION
    summary = {"schema": "idea01-formal-binding-finished-inputs-v1", "scope": "CPU preparation only",
               "mode": "smoke" if args.smoke is not None else origin["mode"], "parent_pairs": len(prepared),
               "coordinate_pairs": len(axis) - len(prepared), "mmcad_revision": development.E1_REV,
               "development_preprocessor_sha256": development.digest(Path(development.__file__)),
               "clip_iso_render_spec_sha256": development.digest(CLIP_SPEC),
               "surface_point_seed_base": SURFACE_POINT_SEED,
               "dependencies": deps, "files_sha256": {name: development.digest(path) for name, path in files.items()},
               "source_sha256": development.digest(Path(__file__)),
               "groups": [{"method": method, "family": family, "attribute": attribute, "kind": kind, **counts}
                          for (method, family, attribute, kind), counts in groups.items()]}
    development.save_json(args.out / "summary.json", summary)
    development.log(parent_pairs=len(prepared), coordinate_pairs=len(axis) - len(prepared),
                    model_pair_statuses={name: {status: sum(counts[status] for key, counts in groups.items()
                                                           if key[0] == name) for status in ("verified", "failed")}
                                         for name in METHODS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
