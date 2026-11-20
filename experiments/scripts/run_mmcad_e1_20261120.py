"""Evaluate the frozen MM-CAD v2 checkpoint on the registered E1 galleries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from idea01 import text2cad_e1_eval as e1


RUN_ID = "idea01_e1_mmcad_20261120"
E1_RUN = "idea01_e1_fusion360_20261120"
FREEZE_SHA = "3df2a07cdd71317b9236c028f18725cd024f680cbf3f7337be0e5ed844e3f9e5"
GEOMETRY_SHA = "6c1e781ccba1505d1352220eebec85c571cab0b04905a31228bbaa5ba36cbcad"
SCORE_MANIFEST_SHA = "9ddcc49bea01a72c0e054d31f11c4950bc4964bd16ee1d6ead2bd90d6dc1f7d2"
CHECKPOINT_SHA = "009b23ce18f2baeca9b14bc8043e19147c244b495b3265500bba419eee5b2b84"
STATS_SHA = "8235fa5e505d714692b6f62cd69690be93a4c2b542ace9a0a06255f86b83ed13"
TOKENIZER_SHA = "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e"
INFERENCE_LF_SHA = "aa6128149324c9c7a4ecab2a1972439a99046203e563abec0eb1208580ace399"
DOMAINS = {"A": "deepcad", "B": "deepcad", "C": "fusion360"}
QUERY_COUNTS = {"A": 254, "B": 256, "C": 256}
MMCAD = "MM-CAD v2"
TEXT_PROMPT = "title: none | text: {q}"
TEXT_WINDOW = 512
EMBED_DIM = 768
FEATURES = ("face_features", "face_centroids", "edge_to_faces", "face_mask", "face_normals")
K = (1, 5, 20)
STEP_BBOX_TOL = 1e-5


def checked_file(path, expected):
    if e1.sha256(path) != expected:
        raise ValueError(f"fingerprint mismatch: {path}")
    return path


def file_manifest(root, output="SHA256SUMS"):
    payloads = {p.relative_to(root).as_posix(): e1.sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if output in payloads:
        raise ValueError("the output directory already has a checksum manifest")
    (root / output).write_text("".join(f"{sha}  {name}\n" for name, sha in payloads.items()),
                               encoding="ascii", newline="\n")
    return payloads


def read_json(path):
    return json.loads(path.read_bytes())


def log(**fields):
    print(json.dumps(fields, ensure_ascii=True, allow_nan=False), flush=True)


def frozen_axes(root):
    freeze = read_json(checked_file(root / "freeze.json", FREEZE_SHA))
    names = [f"gallery_{domain}.jsonl" for domain in set(DOMAINS.values())]
    names.extend(f"queries_{arm.lower()}.jsonl" for arm in DOMAINS)
    e1.verify_frozen_files(root, freeze, names)
    gallery = {domain: e1.frozen_gallery(root / f"gallery_{domain}.jsonl") for domain in set(DOMAINS.values())}
    queries = {a: e1.read_rows(root / f"queries_{a.lower()}.jsonl") for a in DOMAINS}
    for a, domain in DOMAINS.items():
        if len(queries[a]) != QUERY_COUNTS[a] or len(gallery[domain]) != 256:
            raise ValueError(f"unexpected frozen E1 shape for {a}")
        e1.validate_queries(queries[a], gallery[domain], complete=a != "A")
        if any(hashlib.sha256(q["text"].encode()).hexdigest() != q["text_sha256"] for q in queries[a]):
            raise ValueError(f"frozen text drift for {a}")
    return freeze, gallery, queries


def step_from_brep(source, step):
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCC.Core.TopoDS import TopoDS_Shape

    shape = TopoDS_Shape()
    if not breptools.Read(shape, str(source), BRep_Builder()) or shape.IsNull():
        raise ValueError("could not read the E1 B-Rep")
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(step)) != IFSelect_RetDone:
        raise ValueError("could not export the E1 solid to STEP")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step)) != IFSelect_RetDone:
        raise ValueError("could not reopen exported STEP")
    if reader.TransferRoots() != reader.NbRootsForTransfer():
        raise ValueError("incomplete STEP transfer")
    reopened = reader.OneShape()
    if reopened.IsNull():
        raise ValueError("STEP transfer returned a null shape")
    return shape, reopened


def topology(shape):
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    def count(kind):
        x = TopExp_Explorer(shape, kind)
        n = 0
        while x.More():
            n += 1
            x.Next()
        return n

    return {"faces": count(TopAbs_FACE), "edge_occurrences": count(TopAbs_EDGE)}


def checked_features(raw, shape_counts, max_faces, max_edges):
    if raw is None or raw["adjacency_error_count"]:
        raise ValueError(f"B-Rep input extraction failed: {None if raw is None else raw['adjacency_errors']}")
    n, m = int(raw["num_faces"]), int(raw["num_edges"])
    if n != min(max_faces, shape_counts["faces"]) or m != min(max_edges, shape_counts["edge_occurrences"]):
        raise ValueError("extractor and STEP topology disagree")
    result = {feature: raw[feature] for feature in FEATURES}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("nonfinite B-Rep features")
    edges = raw["edge_to_faces"][:m]
    attached = int(np.count_nonzero((edges >= 0).any(axis=1)))
    two_sided = int(np.count_nonzero((edges >= 0).all(axis=1)))
    if not n or not m or not two_sided or np.any((edges < -1) | (edges >= n)):
        raise ValueError("B-Rep adjacency is empty or out of range")
    if n == shape_counts["faces"] and m != attached:
        raise ValueError("unexplained missing edge adjacency")
    return result, {"retained_faces": n, "retained_edge_occurrences": m,
                    "attached_edges": attached, "two_face_entries": two_sided,
                    "truncated_faces": shape_counts["faces"] - n,
                    "truncated_edges": shape_counts["edge_occurrences"] - m}


def checked_step_extent(comparison):
    source = np.asarray(comparison["candidate"]["bbox"])
    step = np.asarray(comparison["reference"]["bbox"])
    delta = float(np.abs(source - step).max())
    if not np.isfinite(delta) or delta > STEP_BBOX_TOL:
        raise ValueError(f"STEP bounding-box drift {delta:.8g} exceeds {STEP_BBOX_TOL}")
    return delta


def prepare(a):
    from idea01 import text2cad_e1_geometry as geometry
    from step_to_brep_fixed_20261118 import MAX_EDGES, MAX_FACES, extract

    started = time.monotonic()
    frozen = a.e1 / "frozen_inputs"
    freeze, galleries, queries = frozen_axes(frozen)
    prep = a.e1 / "prep"
    verified_geometry = {(r["domain"], r["uid"]): r for r in e1.read_rows(
        checked_file(prep / "geometry.jsonl", GEOMETRY_SHA))}
    a.output.mkdir(parents=True)
    input_dir = a.output / "input"
    input_dir.mkdir()
    (input_dir / "references").mkdir()
    for source in sorted(a.references.glob("*.npz")):
        shutil.copyfile(source, input_dir / "references" / source.name)
    if len(list((input_dir / "references").glob("*.npz"))) != 3:
        raise ValueError("expected all three published MM-CAD reference inputs")
    for arm in queries:
        name = f"queries_{arm.lower()}.jsonl"
        shutil.copyfile(frozen / name, input_dir / name)
    coverage = {}
    with (input_dir / "geometry_evidence.jsonl").open("x", encoding="utf-8", newline="\n") as witness:
        for domain in dict.fromkeys(DOMAINS.values()):
            steps = a.output / "steps" / domain
            steps.mkdir(parents=True)
            available, arrays, failures = [], [], {}
            for index, item in enumerate(galleries[domain]):
                uid = item["uid"]
                prior = verified_geometry[domain, uid]
                vector = item["vector_relpath"]
                checked_file(frozen / vector, freeze["score_inputs_sha256"][vector])
                if (not prior["geometry_pass"] or prior["vector_relpath"] != vector or
                        prior["model_input_sha256"] != item["model_input_sha256"]):
                    raise ValueError(f"broken E1 geometry and input lineage: {uid}")
                source = prep / prior["geometry_relpath"]
                step = steps / source.with_suffix(".step").name
                record = {"uid": uid, "domain": domain, "source_brep": prior["geometry_relpath"],
                          "source_model_input_sha256": item["model_input_sha256"]}
                try:
                    record["brep_sha256"] = e1.sha256(source)
                    shape, reopened = step_from_brep(source, step)
                    record["step_sha256"] = e1.sha256(step)
                    comparison = geometry.compare_shapes(shape, reopened, uid)
                    original = prior["geometry"]["candidate"]
                    restored = comparison["candidate"]
                    if (restored["n_solids"] != original["n_solids"] or
                            not np.allclose(restored["bbox"], original["bbox"], rtol=0, atol=1e-6) or
                            not np.isclose(restored["volume"], original["volume"], rtol=1e-6, atol=0)):
                        raise ValueError("stored B-Rep measurements differ from the E1 replay receipt")
                    record["roundtrip"] = comparison
                    if not comparison["pass"]:
                        raise ValueError("STEP round-trip differs under the frozen E1 geometry gate")
                    record["bbox_delta_max"] = checked_step_extent(comparison)
                    record["topology_before"] = topology(shape)
                    record["topology_after"] = topology(reopened)
                    features, notes = checked_features(extract(str(step)), record["topology_after"], MAX_FACES, MAX_EDGES)
                    record["extractor"] = notes
                    available.append(uid)
                    arrays.append(features)
                    record["status"] = "ok"
                except (OSError, ValueError, RuntimeError) as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures[uid] = reason
                    record.update(status="failed", error=reason)
                witness.write(json.dumps(record, allow_nan=False, ensure_ascii=True) + "\n")
                if (index + 1) % 32 == 0:
                    witness.flush()
                    log(stage="prepare", domain=domain, checked=index + 1, succeeded=len(available),
                        failed=len(failures), seconds=round(time.monotonic() - started, 1))
            if not available:
                raise ValueError(f"no usable {domain} MM-CAD inputs")
            np.savez_compressed(input_dir / f"features_{domain}.npz", cad_uids=np.array(available),
                                **{name: np.stack([v[name] for v in arrays]) for name in FEATURES})
            coverage[domain] = {"full_gallery_uids": e1.uids_for(galleries[domain]),
                                "available_cad_uids": available, "failed": failures}
    e1.write_json(input_dir / "prepared.json", {"frozen_sha256": FREEZE_SHA, "geometry_sha256": GEOMETRY_SHA,
                  "pipeline": "exact E1 replay B-Rep -> STEP -> G5 v2 topology and surface extractor",
                  "extractor_sha256": e1.sha256(Path(__file__).with_name("step_to_brep_fixed_20261118.py")),
                  "coverage": coverage, "seconds": time.monotonic() - started})
    files = file_manifest(input_dir)
    log(stage="prepare_completed", files=len(files), sha256=e1.sha256(input_dir / "SHA256SUMS"))


def load_model(a):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(a.threads)
    code = (a.upstream / "inference.py").read_bytes()
    if hashlib.sha256(code.replace(b"\r\n", b"\n")).hexdigest() != INFERENCE_LF_SHA:
        raise ValueError("official MM-CAD inference.py differs from the pinned revision")
    checked_file(a.checkpoint, CHECKPOINT_SHA)
    checked_file(a.stats, STATS_SHA)
    checked_file(a.text_model / "tokenizer.json", TOKENIZER_SHA)
    sys.path.insert(0, str(a.upstream))
    import inference as mm

    if mm.Config.TEXT_MAX_LENGTH != TEXT_WINDOW or mm.Config.D_SHARED != EMBED_DIM:
        raise ValueError("MM-CAD window or embedding dimension drifted")

    class CPUTextEncoder(mm.EmbeddingGemmaEncoder):
        def __init__(self):
            torch.nn.Module.__init__(self)
            st = SentenceTransformer(str(a.text_model), device="cpu", local_files_only=True)
            self._pipeline = torch.nn.ModuleList(list(st))
            self.tokenizer = st.tokenizer

    model = mm.Baseline(CPUTextEncoder(), mm.Config).to(device="cpu", dtype=torch.float32)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    expected = set(model.state_dict())
    weights = {}
    remapped = 0
    for key, tensor in ck["model_state_dict"].items():
        fixed = key
        if key not in expected and key.startswith("text_encoder._pipeline.") and ".auto_model." in key:
            fixed = key.replace(".auto_model.", ".model.")
        if fixed in weights:
            raise ValueError(f"duplicate final MM-CAD weight key: {fixed}")
        weights[fixed] = tensor
        remapped += fixed != key
    missing, unexpected = model.load_state_dict(weights, strict=True)
    receipt = {"missing": len(missing), "unexpected": len(unexpected), "remapped": remapped,
               "epoch": int(ck["epoch"]) if "epoch" in ck else None, "tensor_count": len(weights)}
    model.eval().requires_grad_(False)
    return torch, mm, model, receipt


def normalized(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if matrix.shape[-1] != EMBED_DIM or not np.isfinite(matrix).all() or np.any(norms <= 0):
        raise ValueError("MM-CAD emitted an invalid embedding")
    return matrix / norms


def encode(a):
    started = time.monotonic()
    prepared = read_json(a.input / "prepared.json")
    if prepared["frozen_sha256"] != FREEZE_SHA or prepared["geometry_sha256"] != GEOMETRY_SHA:
        raise ValueError("prepared data belongs to a different E1 run")
    geometry = e1.read_rows(a.input / "geometry_evidence.jsonl")
    if len(geometry) != 512:
        raise ValueError("incomplete E1 geometry evidence")
    bbox_delta = max(checked_step_extent(row["roundtrip"]) for row in geometry if row["status"] == "ok")
    torch, mm, model, strict = load_model(a)
    stats = mm.load_brep_stats(a.stats)
    a.output.mkdir(parents=True)
    reference_rows = []
    with torch.inference_mode():
        def embed(raw):
            return normalized(mm.embed_brep(model, mm.prepare_brep(raw, *stats), dim=EMBED_DIM,
                                            device="cpu").numpy()[0])

        references = sorted((a.input / "references").glob("*.npz"))
        if len(references) != 3:
            raise ValueError("incomplete public reference bundle")
        for p in references:
            with np.load(p, allow_pickle=False) as z:
                raw = {name: z[f"raw_{name}"] for name in FEATURES}
                target = normalized(z["embedding"])
            observed = embed(raw)
            reference_rows.append({"file": p.name, "sha256": e1.sha256(p), "cosine": float(target @ observed)})
        e1.write_json(a.output / "reference_gate.json", {"strict_load": strict, "references": reference_rows})
        if not all(row["cosine"] > 0.999 for row in reference_rows):
            raise ValueError("MM-CAD published B-Rep reference replay failed")
        log(stage="published_reference_pass", **strict)

        gallery_inputs, first_embeds = {}, {}
        for domain in dict.fromkeys(DOMAINS.values()):
            path = a.input / f"features_{domain}.npz"
            with np.load(path, allow_pickle=False) as z:
                cad_uids = z["cad_uids"].tolist()
                features = {name: z[name] for name in FEATURES}
            if cad_uids != prepared["coverage"][domain]["available_cad_uids"]:
                raise ValueError(f"prepared features and candidate identities differ for {domain}")
            gallery_inputs[domain] = (cad_uids, features)
            first_embeds[domain] = embed({name: v[0] for name, v in features.items()})
        first_text = e1.read_rows(a.input / "queries_a.jsonl")[0]["text"]
        normalized(mm.embed_text(model, first_text, dim=EMBED_DIM, device="cpu").numpy())
        e1.write_json(a.output / "smoke_gate.json", {"passed": True, "dimension": EMBED_DIM,
                      "first_geometry_uids": {d: payload[0][0] for d, payload in gallery_inputs.items()},
                      "text_probe": "first frozen A query", "max_step_bbox_delta": bbox_delta})
        log(stage="sanity_completed", domains=list(gallery_inputs), device="cpu", max_step_bbox_delta=bbox_delta)

        data = {}
        for domain, (cad_uids, features) in gallery_inputs.items():
            embeddings, available, failed = [], [], {}
            for i, uid in enumerate(cad_uids):
                try:
                    raw = {name: v[i] for name, v in features.items()}
                    z = first_embeds[domain] if i == 0 else embed(raw)
                except (ValueError, RuntimeError) as exc:
                    failed[uid] = f"{type(exc).__name__}: {exc}"
                else:
                    embeddings.append(z)
                    available.append(uid)
                if not (i + 1) % 32 or i + 1 == len(cad_uids):
                    log(stage="geometry_embeddings", domain=domain, checked=i + 1, covered=len(available),
                        failed=len(failed), seconds=round(time.monotonic() - started, 1))
            matrix = np.stack(embeddings)
            np.savez(a.output / f"cad_{domain}.npz", cad_uids=np.array(available), embeddings=matrix)
            data[domain] = (available, matrix, failed)

        tokenizer = model.text_encoder.tokenizer
        text_stats = {}
        for arm, domain in DOMAINS.items():
            rows = e1.read_rows(a.input / f"queries_{arm.lower()}.jsonl")
            prompts = [TEXT_PROMPT.format(q=row["text"]) for row in rows]
            original = np.asarray([len(ids) for ids in tokenizer(prompts, truncation=False)["input_ids"]])
            used, text_vectors = [], []
            for start in range(0, len(rows), a.batch):
                tok = tokenizer(prompts[start:start + a.batch], padding=True, truncation=True,
                                max_length=TEXT_WINDOW, return_tensors="pt").to("cpu")
                used.extend(tok["attention_mask"].sum(dim=1).tolist())
                z = mm.truncate(model.text_encoder(tok["input_ids"], tok["attention_mask"]), EMBED_DIM)
                text_vectors.append(normalized(z.numpy()))
            kept = np.asarray(used)
            if not np.array_equal(kept, np.minimum(TEXT_WINDOW, original)):
                raise ValueError(f"text truncation receipt disagrees with actual tokenizer for {arm}")
            query = np.concatenate(text_vectors)
            cad_uids, cad_vectors, _ = data[domain]
            np.savez(a.output / f"scores_{arm.lower()}.npz", query_uids=np.array(e1.uids_for(rows)),
                     cad_uids=np.array(cad_uids), scores=query @ cad_vectors.T,
                     text_embeddings=query, original_tokens=original, encoded_tokens=kept)
            text_stats[arm] = {"queries": len(rows), "truncated": int((original > kept).sum()),
                               "max_prompt_tokens": int(original.max()), "text_window": TEXT_WINDOW}
            log(stage="text_embeddings", arm=arm, **text_stats[arm])
    e1.write_json(a.output / "encode_receipt.json", {"device": "cpu", "threads": a.threads, "text_batch": a.batch,
                  "python": platform.python_version(), "packages": {name: metadata.version(name) for name in (
                      "torch", "transformers", "sentence-transformers", "numpy")},
                  "strict_load": strict, "checkpoint_sha256": CHECKPOINT_SHA, "stats_sha256": STATS_SHA,
                  "tokenizer_sha256": TOKENIZER_SHA, "inference_lf_sha256": INFERENCE_LF_SHA,
                  "inference_sha256": e1.sha256(a.upstream / "inference.py"),
                  "input_manifest_sha256": e1.sha256(a.input / "SHA256SUMS"), "text_prompt": TEXT_PROMPT,
                  "text_truncation_side": tokenizer.truncation_side, "text": text_stats,
                  "embedding_failures": {domain: outcome[2] for domain, outcome in data.items()},
                  "seconds": time.monotonic() - started})
    file_manifest(a.output)


def matched_panel(query_uids, cad_uids, available, mm_scores, original, domain, complete=None):
    available_set = set(available)
    if available != [uid for uid in cad_uids if uid in available_set] or not available or len(available) != len(available_set):
        raise ValueError("MM-CAD candidate axis is not an ordered subset of the frozen gallery")
    if mm_scores.shape != (len(query_uids), len(available)):
        raise ValueError("MM-CAD score matrix shape disagrees with the axes")
    if complete is not None and np.asarray(complete).shape != (len(query_uids),):
        raise ValueError("complete-text mask disagrees with the query axis")
    row_indices = np.array([i for i, uid in enumerate(query_uids)
                            if uid in available_set and (complete is None or complete[i])], dtype=int)
    columns = np.array([available.index(query_uids[i]) for i in row_indices], dtype=int)
    subset_columns = [cad_uids.index(uid) for uid in available]
    ranks = {MMCAD: e1.rank_matrix(mm_scores[row_indices], available, domain)[np.arange(len(row_indices)), columns]}
    for method, matrix in original.items():
        if matrix.shape != (len(query_uids), len(cad_uids)):
            raise ValueError(f"invalid shape for {method}")
        aligned = matrix[np.ix_(row_indices, subset_columns)]
        ranks[method] = e1.rank_matrix(aligned, available, domain)[np.arange(len(row_indices)), columns]
    return row_indices, ranks


def paired_at_k(before, after, groups, domain, tag):
    dense, draws, seed = e1.cluster_draws(groups, domain, tag)
    n = len(draws[0])
    denominator = draws @ np.bincount(dense, minlength=n)
    result = {}
    for k in (1, 20):
        change = (after <= k).astype(np.int8) - (before <= k).astype(np.int8)
        samples = draws @ np.bincount(dense, weights=change, minlength=n) / denominator
        result[f"R@{k}"] = {"delta_pp": 100 * float(change.mean()), "added": int((change > 0).sum()),
                          "lost": int((change < 0).sum()), "ci95_pp": e1.percentile_interval(100 * samples),
                          "draws": len(draws), "seed": seed}
    return result


def original_scores(root, arm, query, gallery, pins):
    name = f"scores/{arm.lower()}.npz"
    qids, cad_ids = e1.uids_for(query), e1.uids_for(gallery)
    with np.load(checked_file(root / name, pins[name]), allow_pickle=False) as source:
        if source["query_uids"].tolist() != qids or source["cad_uids"].tolist() != cad_ids:
            raise ValueError(f"registered Text2CAD score axes disagree for {arm}")
        raw, bg, counts = (source[x] for x in ("scores", "background", "counts"))
    if counts.tolist() != [item["n_valid"] for item in gallery] or not np.isfinite(raw).all() or not np.isfinite(bg).all():
        raise ValueError(f"invalid registered Text2CAD scores for {arm}")
    return {"Text2CAD S0": raw, "Text2CAD field mean": raw / counts,
            "Text2CAD subtract mean background": raw - bg.mean(axis=0),
            "Text2CAD S48": e1.score_readouts(raw, bg)["S48"]}


def evaluate(a):
    frozen = a.e1 / "frozen_inputs"
    _, galleries, queries = frozen_axes(frozen)
    historical = a.e1 / "received" / "runs" / E1_RUN
    pins = {name: sha for sha, name in (line.split("  ", 1) for line in
            checked_file(historical / "SHA256SUMS", SCORE_MANIFEST_SHA).read_text(encoding="ascii").splitlines())}
    receipt = read_json(a.encoded / "encode_receipt.json")
    covered = read_json(a.input / "prepared.json")["coverage"]
    if receipt["input_manifest_sha256"] != e1.sha256(a.input / "SHA256SUMS"):
        raise ValueError("MM-CAD embeddings do not bind to the local prepared inputs")
    a.output.mkdir(parents=True)
    output = {"run_id": RUN_ID, "frozen_sha256": FREEZE_SHA, "reference_e1_result_manifest_sha256": SCORE_MANIFEST_SHA,
              "encoding_manifest_sha256": e1.sha256(a.encoded / "SHA256SUMS"), "arms": {}}
    table, rank_rows = [], []
    for arm, domain in DOMAINS.items():
        q, c = queries[arm], galleries[domain]
        qids, cids = e1.uids_for(q), e1.uids_for(c)
        base = original_scores(historical, arm, q, c, pins)
        with np.load(a.encoded / f"scores_{arm.lower()}.npz", allow_pickle=False) as z:
            mm_uids, mm_scores = z["cad_uids"].tolist(), z["scores"]
            before, after = z["original_tokens"], z["encoded_tokens"]
            if z["query_uids"].tolist() != qids or before.shape != after.shape or before.shape != (len(qids),):
                raise ValueError(f"unpaired MM-CAD queries or text coverage for {arm}")
        expected = covered[domain]["available_cad_uids"]
        failures = receipt["embedding_failures"][domain]
        if mm_uids != [uid for uid in expected if uid not in failures] or not np.isfinite(mm_scores).all():
            raise ValueError(f"unaccounted embedding coverage or invalid scores for {arm}")
        complete = before == after
        info = {"full_queries": len(qids), "full_gallery": len(cids), "mmcad_covered_gallery": len(mm_uids),
                "truncated_queries": int((~complete).sum()), "image_or_text_augmentation": False,
                "geometry_failed": covered[domain]["failed"], "embedding_failed": failures, "panels": {}}
        output["arms"][arm] = info
        for panel, mask in (("native_window", None), ("two_models_complete_read", complete)):
            keep, ranks = matched_panel(qids, cids, mm_uids, mm_scores, base, domain, complete=mask)
            groups = [q[i]["group_id"] for i in keep]
            stats = e1.summarize_methods(ranks, groups, len(mm_uids), domain, f"mmcad_{arm}_{panel}")
            pairs = {b: paired_at_k(ranks[b], ranks[MMCAD], groups, domain, f"mmcad_{arm}_{panel}_{b}")
                     for b in ("Text2CAD S0", "Text2CAD S48")} if len(keep) else {}
            info["panels"][panel] = {"n_queries": len(keep), "n_gallery": len(mm_uids), "methods": stats,
                                      "mmcad_minus_text2cad": pairs}
            for method, results in stats.items():
                table.append({"arm": arm, "panel": panel, "method": method, "n_queries": len(keep),
                              "n_gallery": len(mm_uids), **{f"R@{k}": results[f"R@{k}"] for k in K},
                              **{f"hits@{k}": results["hits"][f"R@{k}"] for k in K}, "MedR": results["MedR"]})
                rank_rows.extend({"arm": arm, "panel": panel, "method": method, "uid": qids[i],
                                  "group_id": q[i]["group_id"], "rank": int(rank),
                                  "mmcad_prompt_tokens": int(before[i]), "mmcad_encoded_tokens": int(after[i])}
                                 for i, rank in zip(keep, ranks[method]))
            table.append({"arm": arm, "panel": panel, "method": "Uniform random (expected)",
                          "n_queries": len(keep), "n_gallery": len(mm_uids),
                          **{f"R@{k}": min(k, len(mm_uids)) / len(mm_uids) for k in K}})
            if mask is None:
                # A missing CAD remains a miss in the original-query denominator.
                info["mmcad_full_denominator"] = {"n": len(qids), "unavailable_uid_queries": len(qids) - len(keep),
                    **{f"hits@{k}": int((ranks[MMCAD] <= k).sum()) for k in K}}
                keep_set = set(keep.tolist())
                rank_rows.extend({"arm": arm, "panel": "native_full_denominator_unavailable", "method": MMCAD,
                                  "uid": qids[i], "group_id": q[i]["group_id"], "rank": None,
                                  "mmcad_prompt_tokens": int(before[i]), "mmcad_encoded_tokens": int(after[i])}
                                 for i in range(len(qids)) if i not in keep_set)
    e1.write_json(a.output / "metrics.json", output)
    for filename, records in (("benchmark.csv", table), ("query_ranks.csv", rank_rows)):
        with (a.output / filename).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    file_manifest(a.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--e1", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("encode")
    for flag in ("input", "output", "upstream", "checkpoint", "stats", "text-model"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p = commands.add_parser("evaluate")
    for flag in ("e1", "input", "encoded", "output"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    commands = {"prepare": prepare, "encode": encode, "evaluate": evaluate}
    commands[args.command](args)


if __name__ == "__main__":
    main()
