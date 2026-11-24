"""Prepare blinded human-review inputs from independently verified solids."""
from __future__ import annotations

from hashlib import sha256
from html import escape
import json
from pathlib import Path

from . import formal_binding_families as families
from . import formal_binding_occ as occ
from . import in_domain_pilot_geometry as pilot


REVIEW_INSTRUCTIONS = """# Table 4 geometry and wording review

Open index.html to view the items. Use the image panels or open the STEP solids in a CAD viewer. The top and bottom panels provide plan views. The oblique panels show the ends, side openings, and raised or underside features.

For each caption, choose the part it describes, mark both if both fit, mark neither if neither fits, or mark unclear when the geometry or language is ambiguous. Confirm whether the anchor and both target holes are visible on each part. Where there are alternate wordings, check that texts describing the same part express the same hole property and relation. Record the view or English phrase that led to any uncertainty.

Use one copy of decisions_template.jsonl per reviewer. Fill reviewer_id, review date, and your observations. Return your decisions directly to the study coordinator before discussing them with the other reviewer. The coordinator records disagreements and any replacements.
"""


def _step(shape, path):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer

    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(path)) != IFSelect_RetDone:
        raise ValueError("STEP export failed")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() != 1:
        raise ValueError("STEP import did not recover one root")
    return reader.OneShape()


def _projected_edges(shape, direction):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCC.Core.GeomAbs import GeomAbs_Line
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCC.Extend.TopologyUtils import TopologyExplorer

    dx, dy, _ = direction
    across = (dy, -dx, 0.0) if dx or dy else (1.0, 0.0, 0.0)
    camera = gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(*direction), gp_Dir(*across))
    visibility = HLRBRep_Algo()
    visibility.Add(shape)
    visibility.Projector(HLRAlgo_Projector(camera))
    visibility.Update()
    visibility.Hide()
    views = HLRBRep_HLRToShape(visibility)
    edges = []
    for compound in (views.VCompound(), views.OutLineVCompound(), views.Rg1LineVCompound()):
        if compound is None or compound.IsNull():
            continue
        for edge in TopologyExplorer(compound).edges():
            curve = BRepAdaptor_Curve(edge)
            first, last = curve.FirstParameter(), curve.LastParameter()
            count = 2 if curve.GetType() == GeomAbs_Line else 65
            points = [curve.Value(first + (last - first) * i / (count - 1)) for i in range(count)]
            edges.append([(point.X(), point.Y()) for point in points])
    if not edges:
        raise ValueError("hidden-line projection returned no visible edges")
    return edges


def _preview(shape, path: Path, extent: float):
    from PIL import Image, ImageDraw

    width, height = 340, 260
    canvas = Image.new("RGB", (width * 3, height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    cameras = (
        ("Top", (0.0, 0.0, 1.0)), ("Oblique A", (1.0, 1.0, 0.8)), ("Oblique B", (-1.0, -1.0, 0.8)),
        ("Bottom", (0.0, 0.0, -1.0)), ("Low oblique A", (1.0, 1.0, -0.6)), ("Low oblique B", (-1.0, -1.0, -0.6)),
    )
    scale = 0.82 * min(width, height) / extent
    for index, (title, direction) in enumerate(cameras):
        edges = _projected_edges(shape, direction)
        points = [point for edge in edges for point in edge]
        cx, cy = ((min(p[i] for p in points) + max(p[i] for p in points)) / 2 for i in range(2))
        left, top = width * (index % 3), height * (index // 3)
        draw.rectangle((left, top, left + width - 1, top + height - 1), outline="#d5dbe0")
        draw.text((left + 13, top + 11), title, fill="#293441")
        for edge in edges:
            line = [(left + width / 2 + (x - cx) * scale, top + height / 2 - (y - cy) * scale) for x, y in edge]
            draw.line(line, fill="#223c55", width=2)
    canvas.save(path)


def _blind_order(options, *tags):
    return sorted(options, key=lambda item: families.seed_number("blind", *tags, item))


def item_sha256(item):
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def export(review_dir, design, pair, shapes, histories, use_previews=True):
    """Build neutral cases and return a separate private answer key."""
    wording_sha = pilot.fingerprint(pair["caption_sha256"])
    opaque = f"r{families.seed_number('review-id', design.pair_id, wording_sha):016x}"
    filenames = []
    part_rows = []
    private_parts = {}
    media_warnings = []
    try:
        for index, entity in enumerate(_blind_order(("A", "B"), opaque, "part"), start=1):
            label = f"P{index}"
            relative = Path("solids") / f"{opaque}-{label}"
            step = (review_dir / relative).with_suffix(".step")
            preview = (review_dir / relative).with_suffix(".png")
            filenames.append(step)
            member = entity + "0"
            shape = shapes[member]
            restored = _step(shape, step)
            evidence = occ.compare_same_body(shape, restored)
            measured = occ.measure(restored, design.family, design.slot.attribute, design.slot.kind,
                                   design.expected(entity))
            if evidence["status"] != "verified" or measured["status"] != "ok":
                raise ValueError(f"STEP roundtrip changed the measured solid for part {label}")
            rendered = None
            if use_previews:
                filenames.append(preview)
                try:
                    bounds = measured["bounds"]
                    extent = max(bounds[i + 3] - bounds[i] for i in range(3))
                    _preview(restored, preview, extent)
                    rendered = relative.with_suffix(".png").as_posix()
                except Exception as exc:
                    preview.unlink(missing_ok=True)
                    media_warnings.append({"part": label, "error": f"{type(exc).__name__}: {exc}"})
            part_rows.append({"part_id": label, "step": relative.with_suffix(".step").as_posix(), "image": rendered,
                              "step_sha256": sha256(step.read_bytes()).hexdigest(),
                              "image_sha256": sha256(preview.read_bytes()).hexdigest() if rendered else None})
            private_parts[label] = {"entity": entity, "source_history": member,
                                    "raw_program_sha256": histories[member]["raw_program_sha256"],
                                    "step_sha256": part_rows[-1]["step_sha256"],
                                    "image_sha256": part_rows[-1]["image_sha256"],
                                    "roundtrip_geometry": evidence}
    except Exception:
        for path in filenames:
            path.unlink(missing_ok=True)
        raise
    presentation = []
    instructions = []
    private_views = {}
    variants = tuple(pair["captions"]["A"]) if design.slot.kind == "binding" else (design.slot.view_variant,)
    for number, variant in enumerate(_blind_order(variants, opaque, "wording"), start=1):
        label = f"V{number}"
        choices = _blind_order(("A", "B"), opaque, variant)
        captions = [{"caption_id": f"C{i}", "text": pair["captions"][entity][variant]}
                    for i, entity in enumerate(choices, start=1)]
        presentation.append({"view_id": label, "captions": captions})
        instructions.append({"view_id": label, "caption_match": {item["caption_id"]: None for item in captions}})
        private_views[label] = {"variant": variant, "entities": {f"C{i}": entity for i, entity in enumerate(choices, start=1)}}
    item = {"review_id": opaque, "parts": part_rows, "wording_views": presentation}
    item_digest = item_sha256(item)
    blank = {"review_id": opaque, "review_item_sha256": item_digest,
             "reviewer_id": None, "review_date": None, "part_visibility": {
        part["part_id"]: {"anchor_visible": None, "both_target_holes_visible": None} for part in part_rows
    }, "wording_judgments": instructions, "equivalent_wording": None, "notes": None,
        "choices": ["P1", "P2", "both", "neither", "unclear"]}
    private = {"review_id": opaque, "review_item_sha256": item_digest,
               "pair_id": design.pair_id, "parts": private_parts, "views": private_views,
               "preview_warnings": media_warnings}
    return item, blank, private


def write_index(review_dir):
    """Keep the review display portable and sort cases by neutral ID."""
    data = (review_dir / "items.jsonl").read_text(encoding="utf-8")
    items = sorted((json.loads(line) for line in data.splitlines()), key=lambda item: item["review_id"])
    header = """<!DOCTYPE html>
<html lang="en"><meta charset="utf-8"><title>Table 4 blind review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:2rem auto;padding:0 1rem;max-width:78rem;font:16px/1.5 system-ui;color:#1b2a36}
article{margin:1rem 0;border:1px solid #c9d4dd;border-radius:6px;padding:1rem}
.parts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(24rem,100%),1fr));gap:1rem}
figure{margin:0}img{width:100%;height:auto}summary{cursor:pointer;font-weight:700}
h1,h2,p{margin:.4rem 0}li{margin:.35rem 0}a{color:#174b78}
</style>
<h1>Table 4 blind review</h1>
<p>Use the <a href="README.md">review instructions</a> and your own copy of the
<a href="decisions_template.jsonl">decision template</a>.</p>
"""
    page = review_dir / "index.html"
    with page.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(header)
        for item in items:
            case = escape(item["review_id"])
            stream.write(f'<article id="{case}"><details><summary>{case}</summary><div class="parts">\n')
            for part in item["parts"]:
                label, step = escape(part["part_id"]), escape(part["step"], quote=True)
                stream.write(f'<figure><figcaption>{label} · <a href="{step}">STEP solid</a></figcaption>\n')
                if part["image"]:
                    preview = escape(part["image"], quote=True)
                    stream.write(f'<img loading="lazy" src="{preview}" alt="Six CAD views of {label}">\n')
                stream.write("</figure>\n")
            stream.write("</div>\n")
            for view in item["wording_views"]:
                stream.write(f'<h2>{escape(view["view_id"])}</h2><ul>\n')
                for caption in view["captions"]:
                    stream.write(f'<li><strong>{escape(caption["caption_id"])}</strong> {escape(caption["text"])}</li>\n')
                stream.write("</ul>\n")
            stream.write("</details></article>\n")
        stream.write("</html>\n")
    return page
