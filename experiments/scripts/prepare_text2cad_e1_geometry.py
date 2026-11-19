"""E1a local CPU geometry preparation. Run development, then main, then fusion-alternate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import gzip
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VTK_SMP_MAX_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from idea01 import text2cad_e1_geometry as g

AUTODESK_URL = "https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip"
AUTODESK_BYTES = 2_103_449_157
SPLIT_SHA = "2de40371e99fb32c09ca11ba76995ab40fe465aa09609b4de6c81de1bc553ef4"
OFFICIAL_SHA = "9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa"
DEFAULT_DATA = Path("D:/projects/IdeaSpark/CADRetrievalIdea1")
DEFAULT_PREP = Path("D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/idea01_e1_fusion360_20261120/prep")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def implementation():
    return {p.name:digest(p) for p in [Path(__file__), Path(g.__file__)]}


def lines(path):
    with path.open(encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def put_json(path, data):
    tmp = path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False),encoding="utf8")
    tmp.replace(path)


def safe_uid(uid):
    if not re.fullmatch(r"(?:\d{4}/\d{8}|[a-zA-Z0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+)", uid):
        raise ValueError(f"invalid UID {uid!r}")
    return uid.replace("/", "_")


def save_brep(root, rel, shape):
    from OCC.Core.BRepTools import breptools_Write
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.brep")
    breptools_Write(shape,str(tmp))
    if not tmp.is_file() or not tmp.stat().st_size:
        raise OSError(f"BRep write failed: {rel}")
    tmp.replace(path)
    return rel.as_posix()


def official_arrays(path):
    import torch
    x = torch.load(io.BytesIO(gzip.decompress(path.read_bytes())),map_location="cpu",weights_only=True)
    if not np.array_equal(x["mask_cad_dict"]["attn_mask"].numpy(),g.causal_mask()):
        raise ValueError("official attention mask differs")
    a = {k:v.numpy() for k,v in x["vec"].items()}
    a["key_padding_mask"] = x["mask_cad_dict"]["key_padding_mask"].numpy()
    g.validate_arrays(a)
    return a


def array_diff(a,b):
    result = {}
    for k in g.ARRAY_KEYS:
        delta = np.argwhere(a[k] != b[k])
        result[k] = {"different":len(delta), "first":delta[0].tolist() if len(delta) else None}
    result["attn_mask"] = {"different":0, "first":None}
    return result


def prepare_references(prep,data,rows):
    deep = [r for r in rows if r["domain"]=="deepcad"]
    fusion = [r for r in rows if r["domain"]=="fusion360"]
    store = prep/"references"
    store.mkdir(exist_ok=True)
    if deep:
        source = data/".cache/text2cad_official/cad_seq.zip"
        if digest(source) != OFFICIAL_SHA:
            raise ValueError("official CAD ZIP mismatch")
        out = store/"official"
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(source) as z:
            for r in deep:
                uid = safe_uid(r["uid"])
                member = r["official_member"]
                info = z.getinfo(member)
                dest = out/(uid+".pth.gz")
                b = z.read(info) # zipfile verifies member CRC against verified archive
                if not dest.is_file() or b != gzip.decompress(dest.read_bytes()):
                    dest.write_bytes(gzip.compress(b,compresslevel=1,mtime=0))
    if fusion:
        final = store/"r1.0.1.zip"
        partial = store/"r1.0.1.zip.partial"
        source = final if final.is_file() else partial
        if source.stat().st_size != AUTODESK_BYTES:
            raise ValueError("Autodesk archive download incomplete")
        central = (data/".cache/crossdomain_survey_20261119/r1.0.1.directory.bin").read_bytes()
        with zipfile.ZipFile(source) as z:
            z.fp.seek(z.start_dir)
            if z.fp.read(len(central)) != central or hashlib.sha256(z.read("r1.0.1/train_test.json")).hexdigest() != SPLIT_SHA:
                raise ValueError("Autodesk central directory or split mismatch")
            out = store/"autodesk"
            out.mkdir(exist_ok=True)
            for r in fusion:
                uid = safe_uid(r["uid"])
                for ext in (".json",".step"):
                    name = uid+ext
                    b = z.read("r1.0.1/reconstruction/"+name)
                    dest = out/name
                    if not dest.is_file() or dest.read_bytes() != b:
                        dest.write_bytes(b)
        if source == partial:
            partial.replace(final)
        put_json(store/"autodesk_archive.json", {"url":AUTODESK_URL,"bytes":AUTODESK_BYTES,
                 "sha256":digest(final),"central_directory_sha256":hashlib.sha256(central).hexdigest(),
                 "split_sha256":SPLIT_SHA,"member_verification":"ZIP CRC32 on extraction"})


def encode_original(raw):
    # The r1.0.1 component file contains its actual ordered sequence and entities.
    used = [raw["entities"][q["entity"]] for q in raw["sequence"] if q["type"] == "ExtrudeFeature"]
    seq = g.sequence_class().from_dict(raw)
    if len(seq.extrude_seq) != len(used) or not used:
        raise ValueError("original feature dropped")
    for ext in used:
        if ext["extent_type"] == "TwoSidesFeatureExtentType" and ext.get("extent_two",{}).get("distance",{}).get("value",0) != 0:
            raise ValueError("unsupported two-sided original extrusion in pinned upstream")
        for p in ext["profiles"]:
            loops = raw["entities"][p["sketch"]]["profiles"][p["profile"]]["loops"]
            if any(c["type"] not in ("Circle3D","Line3D","Arc3D") for lp in loops for c in lp["profile_curves"]):
                raise ValueError("unsupported original profile curve")
    return g.padded_arrays(seq.normalize(bit=8).numericalize(bit=8))


def damage_results(seq, positive, uid):
    result = []
    indices = [i for i,e in enumerate(seq.extrude_seq) if e.metadata["boolean"] == 2]
    for label,idx in ([("delete_cut",indices[-1])] if indices else []) + [("halve_depth",0)]:
        alt = copy.deepcopy(seq)
        if label == "delete_cut":
            del alt.sketch_seq[idx], alt.extrude_seq[idx]
        else:
            for name in ("extent_one","extent_two"):
                alt.extrude_seq[idx].metadata[name] *= .5
        try:
            m = g.compare_shapes(g.strict_solid(alt),positive,uid,g.cut_probes(seq))
            result.append({"damage":label,"extrusion_index":idx,"comparison":m})
        except Exception as e:
            result.append({"damage":label,"extrusion_index":idx,"error":str(e)[:300]})
    return result


def one_job(args):
    r, prep, stage, train = args
    uid, domain = r["uid"], r["domain"]
    safe = safe_uid(uid)
    folder = {"development":"development","fusion-alternate":"fusion_original"}.get(stage,domain)
    record = {"uid":uid,"domain":domain,"geometry_pass":False,"reason":None,
              "n_valid":None,"n_extrusions":None,"cad_vec_sha256":None,
              "model_input_sha256":None,"vector_relpath":None,"geometry_relpath":None}
    phase = "encoding"
    try:
        g.sequence_class()
        native = prep/"references/autodesk"/safe
        raw = json.loads(native.with_suffix(".json").read_bytes()) if stage=="fusion-alternate" else None
        a = encode_original(raw) if stage=="fusion-alternate" else g.encode_minimal(r["minimal"])
        info = g.validate_arrays(a)
        record.update(g.fingerprints(a))
        record.update(n_valid=info.n_valid_total, n_extrusions=int(np.count_nonzero(a["cad_vec"][:,0]==6)))
        v = Path("vectors")/folder/(safe+".npz")
        (prep/v).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(prep/v, **a)
        record["vector_relpath"] = v.as_posix()
        if stage != "fusion-alternate":
            record["program_features"] = g.program_features(r["minimal"])
        if domain == "deepcad":
            phase = "official_vector"
            pth = prep/"references/official"/(safe+".pth.gz")
            ref = official_arrays(pth)
            record["reference_kind"] = "official Text2CAD 8-bit vector replay"
            record["reference_member"] = r["official_member"]
            record["reference_file_sha256"] = hashlib.sha256(gzip.decompress(pth.read_bytes())).hexdigest()
            record.update({"official_"+k:v for k,v in g.fingerprints(ref).items()})
            record["array_diff"] = array_diff(a,ref)
        if stage == "main" and (record["cad_vec_sha256"] in train or record.get("official_cad_vec_sha256") in train):
            record["reason"] = "source_train_quantized_exact"
            return record
        phase = "quantized_replay"
        seq = g.decode(a["cad_vec"])
        candidate = g.strict_solid(seq)
        probes = g.cut_probes(seq)
        phase = "reference_geometry"
        if domain == "deepcad":
            reference = g.strict_solid(g.decode(ref["cad_vec"]))
        else:
            raw_path = native.with_suffix(".json")
            step = native.with_suffix(".step")
            raw = json.loads(raw_path.read_bytes()) if raw is None else raw
            record.update(reference_kind="Autodesk r1.0.1 final STEP",reference_member="r1.0.1/reconstruction/"+step.name,
                          reference_file_sha256=digest(step),original_json_sha256=digest(raw_path))
            reference, record["reference_frame"] = g.read_reference(step,raw)
        phase = "geometry_comparison"
        record["geometry"] = g.compare_shapes(candidate,reference,uid,probes)
        if stage == "development":
            phase = "float_replay"
            float_seq = g.minimal_sequence(r["minimal"])
            floating = g.strict_solid(float_seq)
            record["float_to_official"] = g.compare_shapes(floating,reference,uid,g.cut_probes(float_seq))
            record["quantized_to_float"] = g.compare_shapes(candidate,floating,uid,probes)
            record["damage"] = damage_results(float_seq,floating,uid)
            record["float_relpath"] = save_brep(prep,Path("solids")/folder/(safe+"_float.brep"),floating)
            record["reference_relpath"] = save_brep(prep,Path("solids")/folder/(safe+"_official.brep"),reference)
        if record["geometry"]["pass"] or stage=="development":
            record["geometry_relpath"] = save_brep(prep,Path("solids")/folder/(safe+".brep"),candidate)
            if domain=="fusion360" and stage=="main":
                record["reference_relpath"] = save_brep(prep,Path("solids/fusion_step")/(safe+".brep"),reference)
        record["geometry_pass"] = record["geometry"]["pass"]
        if not record["geometry_pass"]:
            record["reason"] = "reference_geometry_threshold"
    except Exception as e:
        record.update(reason=("sequence_length_over_max" if "sequence_length_over_max" in str(e)
                              else phase+"_failed"), detail=f"{type(e).__name__}: {e}"[:700])
    return record


def worker_job(r):
    return one_job((r, _prep, _stage, _train))


def initialize(prep,stage,train):
    global _prep, _stage, _train
    _prep, _stage, _train = prep, stage, train
    import torch
    torch.set_num_threads(1)


def freeze_development(prep, rows, records):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox,BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Ax2,gp_Pnt,gp_Dir
    block = BRepPrimAPI_MakeBox(.75,.75,.30).Shape()
    axis = gp_Ax2(gp_Pnt(.35,.35,0),gp_Dir(0,0,1))
    def hole(depth):
        return BRepAlgoAPI_Cut(block,BRepPrimAPI_MakeCylinder(axis,.024,depth).Shape()).Shape()
    synthetic = g.compare_shapes(hole(.18),hole(.31),"blind-versus-through",[[.35,.35,.23]])
    positive = [r for r in records if r["geometry_pass"]]
    damage = Counter(d["damage"] for r in records for d in r.get("damage",[])
                     if d.get("comparison",{}).get("pass") is False)
    operations = Counter(e["extrusion"]["operation"] for r in rows for e in r["minimal"]["parts"].values())
    accepted = bool(len(records)==64 and positive and damage["delete_cut"] and damage["halve_depth"]
                    and not synthetic["pass"] and synthetic["material_disagreements"] == 1)
    result = {"development_accepted":accepted, "n":len(records),"geometry_passed":len(positive),
              "input_operations":operations,"rejections":Counter(r["reason"] for r in records if r["reason"]),
              "negative_controls_rejected":damage, "synthetic_blind_versus_through":synthetic,
              "rule":{"surfaces":"16384 deterministic area-weighted points per side; VTK point-to-triangle p99",
                      "uid_seed":"little-endian first 8 SHA256 bytes of UID; reference adds 1",
                      "linear_deflection":g.LINEAR_DEFLECTION,"angular_deflection":g.ANGULAR_DEFLECTION,
                      "rel_volume_max":.05,"bidirectional_p99_over_reference_diagonal_max":.01,
                      "solids":"equal, valid, all operations executed", "circle_cut_local_gate":
                      "17 equally spaced axis probes across each circular cut; reject material disagreement when both surfaces farther than 1% of reference diagonal"},
              "rationale":"Retained the prespecified global thresholds and added the prespecified local penetration test for circular cuts. Source controls establish sensitivity, per-part errors remain in the ledger.",
              "upstream_sha":g.UPSTREAM_SHA,"implementation":implementation(),"input_sha256":digest(prep/"development.jsonl"),
              "records_sha256":digest(prep/"geometry_development.jsonl")}
    put_json(prep/"geometry_freeze.json",result)
    if not accepted:
        raise ValueError("development gate failed; see geometry_freeze.json")


def fusion_alternate(prep, data, selection, workers=1):
    """Run the second, native-JSON path for the 32 already selected Fusion projects."""
    return run(Path(prep),Path(data),"fusion-alternate",workers,Path(selection))


def run(prep,data,stage,workers,selection=None):
    source = data/".cache/Text2CAD-official"
    if subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() != g.UPSTREAM_SHA:
        raise ValueError("unexpected upstream source SHA")
    os.environ["TEXT2CAD_SOURCE"] = str(source)
    inputs = prep/("development.jsonl" if stage=="development" else "inventory_seed.jsonl")
    keys = ("uid","domain","project","minimal","official_member","program_sha256")
    rows = [{k:r[k] for k in keys} for r in lines(inputs)
            if stage=="development" or r["first_exclusion"] is None]
    if stage=="development" and (len(rows)!=64 or len({r["uid"] for r in rows})!=64
            or any(r["domain"]!="deepcad" for r in rows)):
        raise ValueError("expected 64 unique DeepCAD validation models")
    if stage != "development":
        freeze = json.loads((prep/"geometry_freeze.json").read_text())
        if not freeze["development_accepted"] or freeze["implementation"] != implementation():
            raise ValueError("development freeze missing or implementation changed")
    if stage=="fusion-alternate":
        if selection.suffix == ".json":
            selection_data = json.loads(selection.read_text(encoding="utf8"))
            chosen = set(selection_data["fusion_encoding_uids"] if isinstance(selection_data,dict) else selection_data)
        else:
            chosen = {s.strip() for s in selection.read_text(encoding="utf8").splitlines() if s.strip()}
        rows = [r for r in rows if r["domain"]=="fusion360" and r["uid"] in chosen]
        if len(rows)!=32 or len({r["project"] for r in rows})!=32:
            raise ValueError("expected 32 selected projects")
    output = prep/{"development":"geometry_development.jsonl","main":"geometry.jsonl",
                  "fusion-alternate":"geometry_fusion_original.jsonl"}[stage]
    binding = {"stage":stage,"implementation":implementation(), "inputs_sha256":digest(inputs),
               "selection_sha256":digest(selection) if selection else None}
    binding_path = output.with_suffix(".binding.json")
    if output.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError("existing geometry rows have different bindings")
    else:
        put_json(binding_path,binding)
    previous = list(lines(output)) if output.exists() else []
    seen = {(r["domain"],r["uid"]) for r in previous}
    if len(seen)!=len(previous):
        raise ValueError("duplicate prior output")
    pending = [r for r in rows if (r["domain"],r["uid"]) not in seen]
    if pending:
        prepare_references(prep,data,pending)
        train = set()
        if stage=="main":
            train = {r["sha256"] for r in lines(prep/"train_vec_hashes.jsonl")}
        initialize(prep,stage,train)
        pool = ProcessPoolExecutor(workers,initializer=initialize,initargs=(prep,stage,train)) if workers>1 else None
        try:
            stream = pool.map(worker_job,pending,chunksize=4) if pool else map(worker_job,pending)
            with output.open("a",encoding="utf8") as f:
                for i,result in enumerate(stream,len(previous)+1):
                    f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
                    f.flush()
                    if i % 25 == 0:
                        print(f"{stage}: {i}/{len(rows)}",flush=True)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
    records = list(lines(output))
    if {(r["domain"],r["uid"]) for r in records} != {(r["domain"],r["uid"]) for r in rows}:
        raise ValueError("incomplete geometry ledger")
    if stage=="development":
        freeze_development(prep,rows,records)
    count = Counter((r["domain"],r["reason"] or "pass") for r in records)
    put_json(output.with_suffix(".summary.json"),{"records":len(records),"counts":{"/".join(k):v for k,v in count.items()},
             "sha256":digest(output),"implementation":implementation(),"exit_code":0})


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("development","main","fusion-alternate"))
    ap.add_argument("--prep",type=Path,default=DEFAULT_PREP)
    ap.add_argument("--data",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--workers",type=int,default=1)
    ap.add_argument("--selection",type=Path)
    opt = ap.parse_args()
    run(opt.prep,opt.data,opt.stage,opt.workers,opt.selection)
