#!/usr/bin/env python3
"""Prepare and audit the in-domain development CAD histories on a CPU with OpenCASCADE."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from idea01 import in_domain_pilot_geometry as geo


def unavailable(stage: str, detail: str) -> dict:
    return {"status": "unavailable", "stage": stage, "detail": detail}


def attempt(stage, execute, *args):
    try:
        return execute(*args), None
    except Exception as exc:
        detail = str(exc).strip()
        overlength = detail.startswith("cad_sequence_over_limit:") or (
            detail.startswith("HierCAD input ") and " tokens exceeds " in detail
        )
        return None, {"status": "overlength" if overlength else "failed", "stage": stage,
                      "detail": f"{type(exc).__name__}: {detail}"}


def activate(stage, location, loader):
    if location is None:
        return None, unavailable(stage, "no asset path supplied")
    loaded, failure = attempt(stage, loader, location)
    if failure:
        return None, failure
    implementation, provenance = loaded
    return implementation, {"status": "ok", "path": str(location.resolve()), **provenance}


def activate_hiercad(path):
    from idea01 import in_domain_pilot_native as native

    revision = native.use_hiercad_repo(path)
    return native, {"revision": revision,
                    "target_source": "geometry-derived author schema and released training-example renderer"}


def load_hiercad_tokenizer(path):
    from idea01 import in_domain_pilot_native as native

    tokenizer = native.load_hiercad_tokenizer(path)
    files = {name: sha256((path / name).read_bytes()).hexdigest()
             for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
             if (path / name).is_file()}
    return tokenizer, {"expected_base_revision": native.HIERCAD_BASE_REVISION, "context": native.HIERCAD_MAX_LENGTH,
                       "file_sha256": files}


def replay_with_truth(sequence, design, entity):
    shape, volumes = geo.strict_replay(sequence)
    measured = geo.measure_binding(shape, design, entity)
    measured["volume_after_each_feature"] = volumes
    return shape, measured


def encode_hiercad_panel(native, tokenizer, target, captions):
    """Use each caption with the same candidate to witness the effective query and labels."""
    prompts = {entity: {} for entity in captions}
    effective_target = None
    query_invariant = True
    errors = []
    for entity, variants in captions.items():
        for variant, text in variants.items():
            sample, failure = attempt("hiercad_tokenize", native.encode_hiercad, tokenizer, text, target)
            if failure:
                prompts[entity][variant] = failure
                errors.append({"entity": entity, "variant": variant, **failure})
                continue
            prompt_ids = sample.input_ids[:sample.target_start]
            target_ids = sample.target_ids
            target_mask = [label != -100 for label in sample.labels[sample.target_start:]]
            target_input = {"target_ids": target_ids, "target_mask": target_mask}
            record = {"status": "ok", "effective_input": prompt_ids, "input_sha256": geo.fingerprint(prompt_ids),
                      "full_input_sha256": geo.fingerprint(sample.input_ids),
                      "target_input_sha256": geo.fingerprint(target_input), "query_tokens": sample.target_start,
                      "total_tokens": len(sample.input_ids), "truncated": False}
            prompts[entity][variant] = record
            if effective_target is None:
                effective_target = {"effective_target": target_input, "input_sha256": record["target_input_sha256"],
                                    "target_tokens": len(target_ids), "segment_counts": sample.segment_counts,
                                    "truncated": False}
            else:
                query_invariant &= effective_target["input_sha256"] == record["target_input_sha256"]
    status = "ok"
    if errors:
        status = "overlength" if any(error["status"] == "overlength" for error in errors) else "failed"
    elif not query_invariant:
        status = "query_dependent_target"
    return {"status": status, "caption_inputs": prompts, "query_invariant_target": query_invariant and not errors,
            "tokenization_errors": errors, **(effective_target or {})}


def run_history(design, history, captions, assets, statuses):
    raw, order = geo.deepcad_history(design, history)
    sequence_type, spectra, native, hier_tokenizer = (assets[key] for key in (
        "text2cad_source", "spectrabrep_source", "hiercad_source", "hiercad_tokenizer"))
    measure = {}
    shapes = {}
    inputs = {}
    record = {"pair_id": design.pair_id, "history": history, "entity": history[0], "cut_order": order,
              "raw_program": raw, "raw_program_sha256": geo.fingerprint(raw), "geometry": measure, "inputs": inputs}

    def measure_sequence(name, stage, make_sequence):
        outcome, failure = attempt(stage, lambda: replay_with_truth(make_sequence(), design, history[0]))
        if failure:
            measure[name] = failure
        else:
            shapes[name], measure[name] = outcome

    if sequence_type is None:
        blocked = unavailable("text2cad_source", statuses["text2cad_source"].get("detail", "official converter unavailable"))
        measure.update({name: blocked for name in ("raw", "text2cad", "hiercad")})
        inputs.update(dict.fromkeys(geo.MODELS, blocked))
        return record, shapes

    def raw_sequence():
        sequence = sequence_type.from_dict(raw)
        for sketch in sequence.sketch_seq:
            # The OCC SWIG point constructor accepts real scalars, not np.int64.
            # Keep the authored program and the coordinate values unchanged.
            frame = sketch.coordsystem.metadata
            frame["origin"] = frame["origin"].astype("float64", copy=False)
        return sequence

    measure_sequence("raw", "deepcad_raw_replay", raw_sequence)
    encoded, failure = attempt("text2cad_program", geo.text2cad_program, sequence_type, raw)
    if failure:
        inputs["text2cad"] = failure
        blocked = unavailable("text2cad_program", "official quantized program unavailable")
        measure["text2cad"] = blocked
        inputs.update({name: blocked for name in ("command_dual_tower", "spectrabrep")})
    else:
        arrays, inputs["text2cad"] = encoded
        measure_sequence("text2cad", "text2cad_effective_replay",
                         lambda: sequence_type.from_vec(arrays["cad_vec"].numpy().copy(), bit=8, post_processing=False))
        command_result, command_failure = attempt("command_dual_tower_program", geo.dual_tower_program, arrays)
        inputs["command_dual_tower"] = command_failure or command_result
        if spectra is None:
            inputs["spectrabrep"] = unavailable("spectrabrep_source", statuses["spectrabrep_source"].get("detail", ""))
        else:
            descriptor, spectra_failure = attempt("spectrabrep_descriptor", geo.spectra_program, spectra, arrays)
            inputs["spectrabrep"] = spectra_failure or descriptor

    minimal, failure = attempt("text2cad_minimal_export", geo.normalized_minimal, sequence_type, raw)
    if failure:
        record["minimal_export"] = failure
        blocked = unavailable("text2cad_minimal_export", "geometry-only normalized program unavailable")
        measure["hiercad"] = inputs["hiercad"] = blocked
        return record, shapes
    record.update(normalized_minimal=minimal, normalized_minimal_sha256=geo.fingerprint(minimal))
    if native is None:
        blocked = unavailable("hiercad_source", statuses["hiercad_source"].get("detail", ""))
        measure["hiercad"] = inputs["hiercad"] = blocked
        return record, shapes

    target, failure = attempt("hiercad_serialize", native.hiercad_target, minimal)
    if failure:
        inputs["hiercad"] = failure
        measure["hiercad"] = unavailable("hiercad_serialize", "released target replay program unavailable")
        return record, shapes
    record["hiercad_target"] = {"response": target.response, "response_sha256": sha256(target.response.encode()).hexdigest(),
                                "replay_program": target.replay_program,
                                "replay_program_sha256": geo.fingerprint(target.replay_program)}
    measure_sequence("hiercad", "hiercad_effective_replay",
                     lambda: geo.minimal_replay_sequence(sequence_type, target.replay_program))
    if hier_tokenizer is None:
        inputs["hiercad"] = unavailable("hiercad_tokenizer", statuses["hiercad_tokenizer"].get("detail", ""))
    else:
        inputs["hiercad"] = encode_hiercad_panel(native, hier_tokenizer, target, captions)
    return record, shapes


def input_panel(links, geometry_status=None):
    counts = Counter(row["status"] for row in links)
    if counts["distinct"] == len(links):
        status = "pass" if geometry_status in (None, "verified") else "geometry_unverified"
    else:
        status = next(key for key in ("truncated", "overlength", "collision", "input_failed", "unavailable") if counts[key])
    return {"status": status, "geometry_status": geometry_status, "comparisons": links, "comparison_counts": dict(counts)}


def caption_panel(records, variants, candidates):
    links = []
    missing = unavailable("caption_input", "no effective caption tokens")
    for candidate in candidates:
        prompts = records[candidate]
        for variant in variants:
            a, b = (prompts.get(entity, {}).get(variant, missing) for entity in ("A", "B"))
            links.append({"candidate_history": candidate, "caption_variant": variant, **geo.input_link(a, b)})
    return input_panel(links)


def diagnose_pair(design, assets, statuses):
    captions = geo.opposite_captions(design)
    variants = tuple(captions["A"])
    rows = {}
    replays = {}
    for name in geo.HISTORIES:
        rows[name], replays[name] = run_history(design, name, captions, assets, statuses)
    if any(rows[left]["cut_order"] != rows[right]["cut_order"][::-1] for left, right in geo.HISTORY_LINKS):
        raise ValueError("fixture does not reverse the two cuts")
    geometry = {stage: geo.geometry_links({name: rows[name]["geometry"][stage] for name in rows},
                                          {name: replays[name].get(stage) for name in rows})
                for stage in ("raw", "text2cad", "hiercad")}
    text_inputs = None
    text_failure = unavailable("text2cad_tokenizer", statuses["text2cad_tokenizer"].get("detail", ""))
    if assets["text2cad_tokenizer"] is not None:
        text_inputs, text_failure = attempt("text2cad_caption_tokenize", geo.text2cad_captions,
                                            assets["text2cad_tokenizer"], captions)

    models = {}
    for model in geo.MODELS:
        replay_stage = "hiercad" if model == "hiercad" else "text2cad"
        stages = (geometry["raw"]["status"], geometry[replay_stage]["status"])
        geometry_status = "verified" if stages == ("verified", "verified") else "geometry_unverified"

        def compare_programs(links):
            return [{"histories": [left, right], **geo.input_link(rows[left]["inputs"][model], rows[right]["inputs"][model])}
                    for left, right in links]

        models[model] = {
            "same_entity_histories": input_panel(compare_programs(geo.HISTORY_LINKS), geometry_status),
            "opposite_entities": input_panel(compare_programs(geo.ENTITY_LINKS), geometry_status),
        }
    if text_failure:
        text_inputs = {entity: dict.fromkeys(variants, text_failure) for entity in captions}
    models["text2cad"]["opposite_captions"] = caption_panel({"shared": text_inputs}, variants, ("shared",))
    hier_inputs = {name: row["inputs"]["hiercad"].get("caption_inputs", {
        entity: dict.fromkeys(variants, row["inputs"]["hiercad"]) for entity in captions
    }) for name, row in rows.items()}
    models["hiercad"]["opposite_captions"] = caption_panel(hier_inputs, variants, geo.HISTORIES)
    pair = {"pair_id": design.pair_id, "design": asdict(design), "captions": captions,
            "caption_sha256": {entity: {variant: sha256(text.encode()).hexdigest() for variant, text in forms.items()}
                               for entity, forms in captions.items()},
            "text2cad_caption_inputs": text_inputs, "geometry": geometry, "models": models}
    field_counts = {name: row["inputs"]["text2cad"].get("valid_fields") for name, row in rows.items()}
    pair["text2cad_field_count_audit"] = {
        "by_history": field_counts,
        "matched": None if None in field_counts.values() else len(set(field_counts.values())) == 1,
    }
    pair["replay_caption_truth"] = {
        stage: {name: {entity: tuple(row["geometry"][stage]["measured_binding"]) == design.expected_words(entity)
                       for entity in captions} if "measured_binding" in row["geometry"][stage] else None
                for name, row in rows.items()}
        for stage in ("raw", "text2cad", "hiercad")
    }
    return pair, rows.values()


def summarize(pairs, histories):
    rows = []
    for family in geo.FAMILIES:
        for attribute in geo.ATTRIBUTES:
            pairs_here = [pair for pair in pairs if pair["design"]["family"] == family and pair["design"]["attribute"] == attribute]
            pair_ids = {pair["pair_id"] for pair in pairs_here}
            histories_here = [row for row in histories if row["pair_id"] in pair_ids]
            if len(pairs_here) != 10 or len(histories_here) != 40:
                raise ValueError(f"wrong pilot denominator for {family}/{attribute}")
            common_generators = [pair["pair_id"] for pair in pairs_here if all(
                pair["models"][model]["same_entity_histories"]["status"] == "pass" for model in ("text2cad", "hiercad"))]
            for model in geo.MODELS:
                stage = "hiercad" if model == "hiercad" else "text2cad"
                result = {"family": family, "attribute": attribute, "model": model, "planned_pairs": len(pairs_here),
                          "planned_histories": len(histories_here), "generator_common_pass_pairs": common_generators,
                          "raw_geometry_pair_statuses": dict(Counter(pair["geometry"]["raw"]["status"] for pair in pairs_here)),
                          "replay_pair_statuses": dict(Counter(pair["geometry"][stage]["status"] for pair in pairs_here)),
                          "raw_replay_history_statuses": dict(Counter(row["geometry"]["raw"]["status"] for row in histories_here)),
                          "replay_history_statuses": dict(Counter(row["geometry"][stage]["status"] for row in histories_here)),
                          "candidate_input_statuses": dict(Counter(row["inputs"][model]["status"] for row in histories_here))}
                for check in pairs_here[0]["models"][model]:
                    statuses = Counter(pair["models"][model][check]["status"] for pair in pairs_here)
                    examples = [pair["pair_id"] for pair in pairs_here if pair["models"][model][check]["status"] != "pass"]
                    comparison_counts = Counter()
                    for pair in pairs_here:
                        comparison_counts.update(pair["models"][model][check]["comparison_counts"])
                    result[check] = {"passed_pairs": statuses["pass"], "planned_pairs": len(pairs_here),
                                     "pair_statuses": dict(statuses), "comparison_counts": dict(comparison_counts),
                                     "other_pair_ids": examples}
                rows.append(result)
    return rows


def environment():
    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for name in ("numpy", "scipy", "torch", "transformers"):
        try:
            info[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[name] = None
    try:
        import OCC

        info["pythonocc_core"] = OCC.VERSION
    except ImportError:
        info["pythonocc_core"] = None
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new directory outside the checkout")
    parser.add_argument("--text2cad-source", required=True, type=Path, help="pinned upstream Text2CAD root")
    parser.add_argument("--text2cad-tokenizer", type=Path, help="local official BERT tokenizer-only assets")
    parser.add_argument("--spectrabrep-source", type=Path, help="pinned SpectraBRep release tree")
    parser.add_argument("--hiercad-repo", type=Path, help="pinned upstream HierCAD Git checkout")
    parser.add_argument("--hiercad-tokenizer", type=Path, help="local official HierCAD base tokenizer assets")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(ROOT):
        parser.error("--out must be outside the source checkout")
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    assets = {}
    statuses = {}
    for name, path, loader in (
        ("text2cad_source", args.text2cad_source, geo.activate_text2cad),
        ("spectrabrep_source", args.spectrabrep_source, geo.activate_spectrabrep),
        ("text2cad_tokenizer", args.text2cad_tokenizer, geo.text2cad_tokenizer),
        ("hiercad_source", args.hiercad_repo, activate_hiercad),
        ("hiercad_tokenizer", args.hiercad_tokenizer, load_hiercad_tokenizer),
    ):
        assets[name], statuses[name] = activate(name, path, loader)

    pairs, histories = [], []
    pair_file, history_file = (out / name for name in ("geometry_pairs.jsonl", "geometry_histories.jsonl"))
    with pair_file.open("wb") as pair_stream, history_file.open("wb") as history_stream:
        for design in geo.pilot_designs():
            pair, records = diagnose_pair(design, assets, statuses)
            geo.jsonl_write(pair_stream, pair)
            pairs.append(pair)
            for record in records:
                geo.jsonl_write(history_stream, record)
                histories.append(record)
            pair_stream.flush()
            history_stream.flush()
    if len(pairs) != 40 or len(histories) != 160:
        raise ValueError("wrong full pilot denominator")
    result_path = out / "geometry_summary.json"
    result = {
        "schema": "idea01_in_domain_pilot_geometry_v1", "scope": "development-only representation coverage",
        "planned_pairs": 40, "planned_program_histories": 160, "groups": summarize(pairs, histories),
        "assets": statuses, "environment": environment(), "wall_seconds": time.perf_counter() - start,
        "implementation_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "artifacts_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in (pair_file, history_file)},
        "equivalence_tolerances": {"same_entity_relative_symmetric_difference_max": 2e-6,
                                   "opposite_entity_relative_symmetric_difference_min": 1e-3},
        "geometry_for_retrievers": "replay of upstream shared Text2CAD program before the retrieval feature transform",
        "spectrabrep_feature_stage": "released float32 descriptor before training-set normalization",
    }
    result_path.write_bytes(geo.json_bytes(result) + b"\n")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in (pair_file, history_file, result_path)),
        encoding="ascii", newline="\n",
    )
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
