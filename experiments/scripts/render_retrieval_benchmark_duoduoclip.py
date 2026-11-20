"""Render the admitted DeepCAD STEP assets at eight fixed views on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_retrieval_benchmark_fill import read_jsonl, sha256, verify_freeze, verify_sha256, write_json, write_jsonl


EXPERIMENTS = Path(__file__).resolve().parents[1]
SPEC = EXPERIMENTS / "configs" / "retrieval_benchmark_duoduoclip.json"
sys.path.insert(0, str(EXPERIMENTS / "src"))


def rasterize(triangles, colors, size, background):
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    pixels[:] = background
    depth = np.full((size, size), -np.inf)
    for points, color in zip(triangles, colors, strict=True):
        origin = points[0]
        edge_a, edge_b = points[1:] - origin
        determinant = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
        if abs(determinant) < 1e-12:
            continue
        x0, y0 = np.maximum(0, np.ceil(points[:, :2].min(axis=0)).astype(int))
        x1, y1 = np.minimum(size - 1, np.floor(points[:, :2].max(axis=0)).astype(int))
        x = np.arange(x0, x1 + 1)[None, :] - origin[0]
        y = np.arange(y0, y1 + 1)[:, None] - origin[1]
        a = (x * edge_b[1] - y * edge_b[0]) / determinant
        b = (y * edge_a[0] - x * edge_a[1]) / determinant
        z = origin[2] + a * edge_a[2] + b * edge_b[2]
        depth_patch = depth[y0:y1 + 1, x0:x1 + 1]
        inside = (a >= -1e-10) & (b >= -1e-10) & (a + b <= 1 + 1e-10)
        visible = inside & (z > depth_patch)
        depth_patch[visible] = z[visible]
        pixel_patch = pixels[y0:y1 + 1, x0:x1 + 1]
        pixel_patch[visible] = color
    return pixels


def unit(vector):
    return vector / np.linalg.norm(vector)


def render_view(triangles, direction, spec):
    eye = unit(np.asarray(direction, dtype=float))
    right = unit(np.cross(spec["world_up"], eye))
    up = np.cross(eye, right)
    size = spec["size_px"]
    projected = triangles @ np.stack((right, -up, eye), axis=1)
    projected[:, :, :2] = (projected[:, :, :2] / spec["half_width"] + 1) * size / 2 - 0.5
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals = normals[valid] / lengths[valid, None]
    lamp = unit(np.asarray(spec["camera_relative_light_xyz"], dtype=float))
    light = lamp @ np.stack((right, up, eye))
    brightness = spec["ambient"] + spec["diffuse"] * np.maximum(normals @ light, 0)
    colors = np.rint(brightness[:, None] * np.asarray(spec["material_rgb"])).astype(np.uint8)
    image = rasterize(projected[valid], colors, size, spec["background_rgb"])
    if not np.any(image != spec["background_rgb"]):
        raise ValueError("empty render")
    return image


def triangles_from_step(path, spec):
    from idea01.text2cad_e1_geometry import ANGULAR_DEFLECTION, LINEAR_DEFLECTION, measures, mesh_triangles
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    if (spec["linear_deflection"], spec["angular_deflection_rad"]) != (LINEAR_DEFLECTION, ANGULAR_DEFLECTION):
        raise ValueError("renderer configuration differs from the E1 tessellator")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError("STEP import failed")
    reader.SetSystemLengthUnit(1.0)
    if reader.TransferRoots() != reader.NbRootsForTransfer():
        raise ValueError("incomplete STEP root transfer")
    shape = reader.OneShape()
    lower, upper = np.asarray(measures(shape)["bbox"]).reshape(2, 3)
    radius = np.linalg.norm(upper - lower) * 0.5
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("invalid STEP bounding box")
    center = (lower + upper) * 0.5
    transform = gp_Trsf()
    affine = np.column_stack((np.eye(3), -center)) / radius
    transform.SetValues(*affine.ravel())
    normalized = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    triangles = mesh_triangles(normalized)
    if not np.isfinite(triangles).all():
        raise ValueError("nonfinite STEP tessellation")
    return triangles


def render_candidate(path, destination, identity, spec):
    started = time.perf_counter()
    triangles = triangles_from_step(path, spec)
    view_records = []
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        views = Path(staging) / "views"
        views.mkdir()
        for index, direction in enumerate(spec["view_directions"]):
            image = render_view(triangles, direction, spec)
            name = f"{index:02}.png"
            target = views / name
            Image.fromarray(image).save(target, compress_level=spec["png_compress_level"])
            view_records.append({"filename": name, "sha256": sha256(target), "bytes": target.stat().st_size})
        receipt = identity | {"valid": True, "views": view_records, "triangles": len(triangles),
                              "seconds": time.perf_counter() - started}
        write_json(views / "receipt.json", receipt)
        views.rename(destination)
    return receipt


def verify_candidate(destination, identity, frame_count):
    receipt = json.loads((destination / "receipt.json").read_bytes())
    view_names = [f"{index:02}.png" for index in range(frame_count)]
    if (any(receipt.get(key) != value for key, value in identity.items()) or receipt.get("valid") is not True
            or [entry["filename"] for entry in receipt["views"]] != view_names):
        raise ValueError(f"render receipt does not match {identity['uid']}")
    for view in receipt["views"]:
        verify_sha256(destination / view["filename"], view["sha256"])
    return receipt


def run(frozen, steps, output, stride, limit):
    import OCC

    started = time.perf_counter()
    spec_sha = sha256(SPEC)
    spec = json.loads(SPEC.read_bytes())
    binding = spec["deepcad"]
    render = spec["render"]
    producer = {"render_code_sha256": sha256(Path(__file__)),
                "tessellator_code_sha256": sha256(EXPERIMENTS / "src" / "idea01" / "text2cad_e1_geometry.py"),
                "python": sys.version.split()[0], "occ": OCC.VERSION, "numpy": np.__version__,
                "pillow": importlib.metadata.version("Pillow")}
    if render["view_directions"] != [list(p) for p in product((-1, 1), repeat=3)]:
        raise ValueError("must render the eight fixed cube corners in canonical order")
    verify_sha256(frozen / "freeze.json", binding["freeze_sha256"])
    manifest = verify_freeze(frozen)
    gallery_sha = manifest["files"]["gallery.jsonl"]
    gallery = read_jsonl(frozen / "gallery.jsonl")
    if gallery_sha != binding["gallery_sha256"] or len(gallery) != binding["gallery_count"]:
        raise ValueError("incorrect DeepCAD gallery for the recorded STEP bytes")
    selected = gallery[::stride][:limit]
    full = len(selected) == len(gallery)
    source_hash = hashlib.sha256()
    sources = []
    for row in selected:
        key = row["uid"].replace("/", "__")
        path = steps / f"{key}.step"
        digest = sha256(path)
        sources.append((row, key, path, digest))
        source_hash.update(f"{row['uid']}\t{digest}\n".encode("utf-8"))
    source_sha = source_hash.hexdigest()
    if full and source_sha != binding["steps_manifest_sha256"]:
        raise ValueError("the source STEP manifest differs from the registered DeepCAD assets")

    views_root = output / "views"
    views_root.mkdir(parents=True, exist_ok=True)
    statuses = []
    for row, key, path, digest in sources:
        directory = views_root / key
        identity = {"cad_index": row["cad_index"], "uid": row["uid"], "source_sha256": digest,
                    "render_spec_sha256": spec_sha, "producer": producer}
        if directory.exists():
            status = verify_candidate(directory, identity, len(render["view_directions"]))
        else:
            try:
                status = render_candidate(path, directory, identity, render)
            except (ValueError, RuntimeError) as error:
                status = identity | {"valid": False, "reason": f"{type(error).__name__}: {error}"}
        statuses.append(status)
        if len(statuses) % 250 == 0:
            print(f"Rendered {len(statuses)}/{len(sources)}", flush=True)

    with tempfile.TemporaryDirectory(dir=output) as staging:
        stack = Path(staging)
        status_path = stack / "gallery_render_status.jsonl"
        write_jsonl(status_path, statuses)
        summary = {"gallery_sha256": gallery_sha, "full_gallery": full, "source_steps_sha256": source_sha,
                   "render_spec_sha256": spec_sha, "status_sha256": sha256(status_path), "rows": len(statuses),
                   "valid": sum(status["valid"] for status in statuses),
                   "png_bytes": sum(v["bytes"] for row in statuses for v in row.get("views", [])),
                   "wall_seconds": time.perf_counter() - started, "producer": producer}
        write_json(stack / "render_summary.json", summary)
        status_path.replace(output / status_path.name)
        (stack / "render_summary.json").replace(output / "render_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--stride and --limit must be positive")
    print(json.dumps(run(args.frozen, args.steps, args.output, args.stride, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
