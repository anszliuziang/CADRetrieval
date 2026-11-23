#!/usr/bin/env python3
"""Validate native conditional CAD scores and measure HierCAD pilot costs."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/src"))

import torch

from idea01 import in_domain_pilot_native as native
from idea01.text2cad_pair_index import sha256_file
from idea01.text2cad_port.loader import load_frozen_text2cad


def read_axis(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def package_versions(names) -> dict:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def event(out: Path, kind: str, **facts) -> None:
    record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": kind, **facts}
    with (out / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def snapshot_facts(path: Path) -> dict:
    paths = sorted(child for child in path.iterdir() if child.is_file() and child.suffix in {
        ".json", ".safetensors", ".jinja", ".txt", ".model",
    })
    return {
        "path": str(path.resolve()), "directory_name": path.resolve().name,
        "files": {child.name: {"sha256": sha256_file(child), "bytes": child.stat().st_size} for child in paths},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("text2cad", "hiercad"))
    parser.add_argument("--stage", choices=("all", "prepare", "smoke", "readout", "timing", "training"), default="all")
    parser.add_argument("--source", choices=tuple(native.CONDITIONS))
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cad-zip", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bert-dir", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hiercad-repo", type=Path)
    parser.add_argument("--hiercad-base", type=Path)
    parser.add_argument("--hiercad-adapter", type=Path)
    parser.add_argument("--hiercad-validation", type=Path)
    parser.add_argument("--minimal-zip", type=Path)
    parser.add_argument("--hiercad-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--timing-candidates", type=int, default=8)
    parser.add_argument("--timing-queries", type=int, default=4)
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 4 if args.model == "text2cad" else 1
    if min(args.batch_size, args.timing_candidates, args.timing_queries, args.timing_repeats, args.training_steps) < 1:
        parser.error("batch, timing sample/repeat and training step sizes must be positive")
    if args.timing_warmups < 0:
        parser.error("timing warmup count must be nonnegative")
    if args.model == "text2cad":
        if args.stage in ("training", "timing", "smoke"):
            parser.error("the Text2CAD pilot implements its development readout only")
        if args.stage != "prepare" and (args.checkpoint is None or args.bert_dir is None):
            parser.error("Text2CAD readout requires --checkpoint and --bert-dir")
    else:
        if any(getattr(args, name) is None for name in ("hiercad_repo", "hiercad_base", "minimal_zip")):
            parser.error("HierCAD requires --hiercad-repo, --hiercad-base and --minimal-zip")
        if args.stage in ("all", "readout", "timing", "smoke") and args.hiercad_adapter is None:
            parser.error("HierCAD scoring requires --hiercad-adapter")
    return args


def setup_device(device_name: str, needs_forward: bool):
    if not needs_forward:
        return torch.device("cpu"), None
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(device_name)
    if device.type in ("npu", "cuda"):
        runtime = getattr(torch, device.type)
        runtime.set_device(device)
        return device, runtime.get_device_name(device)
    return device, "CPU"


def load_manifests(args, sources) -> tuple[dict, dict]:
    axes = {}
    hashes = {}
    for source in sources:
        paths = {"targets": "validation_targets", "gallery": "validation_gallery",
                 "background": "validation_background"}
        if args.model == "hiercad" and args.stage in ("all", "training"):
            paths["train"] = "pilot_train"
        axes[source] = {}
        hashes[source] = {}
        for key, name in paths.items():
            path = args.manifests / source / f"{name}.jsonl"
            axes[source][key] = read_axis(path)
            hashes[source][key] = sha256_file(path)
        src = axes[source]
        if len(src["targets"]) != 512 or len(src["gallery"]) != 2048:
            raise ValueError(f"{source}: validation counts changed from the frozen 512-by-2048 panel")
        if not {row["uid"] for row in src["targets"]} <= {row["uid"] for row in src["gallery"]}:
            raise ValueError(f"{source}: validation gallery is missing a frozen target")
        if len(src["background"]) != 48 or len({row["text"] for row in src["background"]}) != 48:
            raise ValueError(f"{source}: expected 48 different background texts")
    return axes, hashes


def prepare_candidates(source: str, selection: dict, args, backend, references: dict) -> dict:
    folder = args.out / source
    selected = selection["rows"]
    checks = []
    candidate_dir = folder / "candidate_targets"
    candidate_dir.mkdir()
    for i, row in enumerate(selected):
        record = {"uid": row["uid"], "member": row["cad_member"]}
        try:
            record.update(backend.target_info(row))
            if args.model == "hiercad":
                target, _ = backend.archive.load(row["minimal_member"])
                base = candidate_dir / f"candidate_{i}"
                native.write_json(base.with_suffix(".json"), target.replay_program)
                with base.with_suffix(".txt").open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(target.response)
                record.update(
                    minimal_member=row["minimal_member"],
                    released_validation_reference=native.compare_hiercad_reference(target, references.get(row["uid"])),
                    kernel_replay=native.replay_hiercad(target),
                )
                first = backend.prepare(row, row["texts"][native.CONDITIONS[source][0]])
                record.update(target_token_sha256=first["target_token_sha256"],
                              segment_counts=first["segment_counts"],
                              target_tokens=first["effective_units"])
            record["status"] = "prepared"
        except Exception as exc:
            record.update(native.error_record(exc))
        checks.append(record)
    native.write_json(folder / "selection.json", {
        **{key: value for key, value in selection.items() if key != "rows"}, "candidate_checks": checks,
    })
    event(args.out, "candidate_targets_prepared", source=source,
          selected=len(selected), preparation_failures=sum(row.get("status") == "failed" for row in checks))
    prepared = {
        "selected_candidates": len(selected),
        "targets_prepared": sum(row.get("status") == "prepared" for row in checks),
    }
    if args.model == "hiercad":
        prepared.update(
            kernel_replay_passes=sum(row.get("kernel_replay", {}).get("passed", False) for row in checks),
            official_validation_bytes_equal=sum(
                row.get("released_validation_reference", {}).get("response_bytes_equal", False)
                for row in checks
            ),
        )
    return prepared


def validate_readout(source: str, axes: dict, rows: list[dict], args, backend) -> dict:
    folder = args.out / source
    summaries = {}
    audits = {}
    for condition in native.CONDITIONS[source]:
        queries = [{"uid": row["uid"], "text": row["texts"][condition]} for row in rows]
        single = native.score_cases(backend, rows, queries, 1, audit=True)
        # The batch check has to use at least two actual pairs, even when
        # throughput uses one pair. Its failures stay visible beside singles.
        batched = native.score_cases(backend, rows, queries, max(2, args.batch_size), audit=True)
        audit = native.calibration_checks(single, batched, args.model)
        raw_path = folder / f"{condition}_raw_next_token_distribution.pt"
        try:
            prepared = [backend.prepare(row, queries[0]["text"]) for row in rows]
            causal = backend.causality(prepared, raw_path)
        except Exception as exc:
            causal = native.error_record(exc)
        record = {
            "source": source, "condition": condition, "model": args.model,
            "uid_axis": [row["uid"] for row in rows], "query_axis": queries,
            "single": single, "batched": batched, "calibration_audit": audit,
            "causal_prefix_probe": causal,
        }
        audits[condition] = record
        native.write_json(folder / f"{condition}_raw_readout.json", record)
        event(args.out, "native_matrix_audited", source=source, condition=condition,
              pairs=single["scored_pairs"], batch_pairs=batched["scored_pairs"],
              audit_passed=audit["passed"], causal_prefix_passed=causal.get("passed", False))
    native.reset_memory(backend.device)
    bg = native.score_cases(backend, rows, axes["background"], args.batch_size, audit=False, retry_single=True)
    priors = native.background_logmean(bg["pairs"], len(rows))
    native.write_json(folder / "background.json", {**bg, "candidate_logmeanexp": priors,
                                                   **native.memory_peaks(backend.device)})
    event(args.out, "background_scored", source=source, pairs=bg["scored_pairs"], planned=bg["planned_pairs"])
    for condition, record in audits.items():
        single = record["single"]
        batched = record["batched"]
        audit = record["calibration_audit"]
        causal = record["causal_prefix_probe"]
        corrected = native.calibrated_scores(single, bg, rows, priors)
        ok = (
            len(rows) == 4 and single["scored_pairs"] == 16 and batched["scored_pairs"] == 16
            and bg["scored_pairs"] == 192 and audit["passed"] and causal.get("passed", False)
            and corrected["fixed_cad_invariance"].get("passed", False)
            and all(entry["passed"] for entry in corrected["candidate_target_invariance"])
        )
        native.write_json(folder / f"{condition}_readout.json", {**record, **corrected, "passed": ok})
        summaries[condition] = {
            "passed": ok, "single_pairs": single["scored_pairs"], "batched_pairs": batched["scored_pairs"],
            "pair_audits_passed": audit["pairs_passed"], "causal_prefix_passed": causal.get("passed", False),
            "background_pairs": bg["scored_pairs"], "background_planned": bg["planned_pairs"],
        }
        event(args.out, "readout_finished", source=source, condition=condition, **summaries[condition])
    return summaries


def smoke_long_contexts(source: str, rows: list[dict], out: Path, backend) -> dict:
    eligible = []
    unsupported = []
    for query in rows:
        for level in native.CONDITIONS[source]:
            for row in rows:
                identity = {"query_uid": query["uid"], "candidate_uid": row["uid"], "level": level}
                try:
                    prepared = backend.prepare(row, query["texts"][level])
                except Exception as exc:
                    unsupported.append({**identity, **native.error_record(exc)})
                    continue
                eligible.append((len(prepared["encoded"].input_ids), identity, prepared))
    eligible.sort(key=lambda entry: entry[0], reverse=True)
    if len(eligible) < 2:
        native.write_json(out / "long_context_smoke.json", {"passed": False, "unsupported": unsupported})
        raise ValueError(f"{source}: fewer than two eligible contexts on the frozen panel")
    chosen = eligible[:2]
    native.reset_memory(backend.device)
    native.synchronize(backend.device)
    started = time.perf_counter()
    singles = [backend.score([example], audit=True) for _, _, example in chosen]
    batched, native_batch = backend.score([example for _, _, example in chosen], audit=True)
    native.synchronize(backend.device)
    results = []
    for (width, identity, _), ([single], native_single), pair_batch in zip(chosen, singles, batched, strict=True):
        label_fields = ("input_sha256", "target_token_sha256", "label_sha256", "effective_units")
        score_delta = abs(single["s0"] - pair_batch["s0"]) / single["effective_units"]
        native_delta = native_single["native_batch_ce_abs_delta"]
        passed = (
            score_delta <= native.H_SCORE_MEAN_ATOL
            and all(single[key] == pair_batch[key] for key in label_fields)
            and single["nll_abs_delta"] <= single["nll_atol"]
            and pair_batch["nll_abs_delta"] <= pair_batch["nll_atol"]
            and native_delta is not None and native_delta <= native_single["native_batch_ce_atol"]
        )
        results.append({**identity, "total_input_tokens": width, "single": single, "batch": pair_batch,
                        "native_single": native_single, "score_delta_per_unit": score_delta, "passed": passed})
    batch_delta = native_batch["native_batch_ce_abs_delta"]
    passed = (all(item["passed"] for item in results) and batch_delta is not None
              and batch_delta <= native_batch["native_batch_ce_atol"])
    result = {"passed": passed, "eligible_contexts": len(eligible), "unsupported": unsupported,
              "pairs": results, "native_batch": native_batch,
              "wall_seconds": time.perf_counter() - started, **native.memory_peaks(backend.device)}
    native.write_json(out / "long_context_smoke.json", result)
    return result


def timing_conditions(source: str, axes: dict, query_limit: int) -> dict:
    result = {}
    for level in native.CONDITIONS[source]:
        selected = []
        visited = set()
        for row in axes["targets"]:
            text = row["texts"][level]
            if text not in visited:
                selected.append({"uid": row["uid"], "text": text})
                visited.add(text)
            if len(selected) == query_limit:
                break
        result[level] = selected
    result["background48"] = axes["background"]
    return result


def measure_timing(source: str, axes: dict, args, backend) -> dict:
    folder = args.out / source
    tiers = native.hiercad_timing_bands(backend, axes["gallery"])
    native.write_json(folder / "timing_panel.json", {
        key: value for key, value in tiers.items() if key != "layers"
    })
    event(args.out, "hiercad_timing_panel_prepared", source=source, bands=tiers["bands"],
          candidates_failed=len(tiers["failures"]))
    queries_by_condition = timing_conditions(source, axes, args.timing_queries)
    probe_row = next((layer[0][2] for layer in tiers["layers"].values() if layer), None)
    cells = []
    for condition, queries in queries_by_condition.items():
        widths, failures = native.hiercad_query_widths(backend, queries, probe_row) if probe_row else ([], [])
        for band, layer in tiers["layers"].items():
            # Background uses all 48 frozen texts and one candidate from
            # each layer to keep that separately timed pass bounded.
            limit = 1 if condition == "background48" else args.timing_candidates
            rows, coverage = native.hiercad_timing_sample(layer, widths, limit, band)
            target_pairs = len(rows) * len(queries)
            windows = []
            if target_pairs:
                for i in range(args.timing_warmups + args.timing_repeats):
                    warmup = i < args.timing_warmups
                    window_queries = queries[:1] if warmup else queries
                    native.reset_memory(backend.device)
                    result = native.score_cases(backend, rows, window_queries, args.batch_size, audit=False)
                    windows.append({
                        "phase": "warmup" if warmup else "steady", "pass": i + 1,
                        "measurement": result, **native.memory_peaks(backend.device),
                    })
                    event(args.out, "hiercad_timing_window", source=source, condition=condition, band=band,
                          phase=windows[-1]["phase"], pass_index=i + 1, wall_seconds=result["wall_seconds"],
                          pairs=result["scored_pairs"], planned=result["planned_pairs"],
                          peak_allocated_bytes=windows[-1]["peak_allocated_bytes"])
            throughput = native.duration_range([
                window["measurement"] for window in windows if window["phase"] == "steady"
            ], target_pairs)
            cell = {
                "source": source, "condition": condition, "band": band, "steady_pairs_per_window": target_pairs,
                "query_uids": [row["uid"] for row in queries], "candidate_uids": [row["uid"] for row in rows],
                "batches_requested": args.batch_size, "windows": windows, "throughput": throughput,
                "query_prefix_widths": widths, "query_preparation_failures": failures,
                "target_length_coverage": coverage,
            }
            native.write_json(folder / f"timing_{condition}_{band}.json", cell)
            cells.append({
                key: val for key, val in cell.items()
                if key not in ("windows", "target_length_coverage")
            } | {
                "gallery_pairs_context_fits": coverage["fits"],
                "gallery_pairs_tokenizable": coverage["expected_tokenizable_pairs"],
                "gallery_pairs_expected": len(layer) * len(queries),
            })
    projection = native.project_hiercad(cells, tiers["bands"], source)
    result = {"cells": cells, "projection": projection}
    native.write_json(folder / "hiercad_timing.json", result)
    return result


def run_hiercad_training(source: str, axes: dict, args, backend, base) -> dict:
    import yaml

    root = args.hiercad_repo / "hiercad/config"
    tokenize_config = yaml.safe_load((root / "tokenize.yaml").read_text(encoding="utf-8"))
    train_config = yaml.safe_load((root / "train.yaml").read_text(encoding="utf-8"))
    if tokenize_config["max_length"] != native.HIERCAD_MAX_LENGTH:
        raise ValueError("release trainer context differs from readout window")
    prepared = native.prepare_hiercad_training(
        backend, axes["train"], native.CONDITIONS[source], tokenize_config, args.training_steps,
    )
    folder = args.out / source
    native.write_json(folder / "training_inputs.json", {
        "usable": [{key: val for key, val in row.items() if key != "item"} for row in prepared["examples"]],
        "failures": prepared["preparation_failures"], "source": source,
    })
    result = native.train_hiercad_probe(
        base, prepared, backend.device, train_config,
        lambda record: event(args.out, "hiercad_training_step", source=source, **record),
    )
    native.write_json(folder / "hiercad_training.json", result)
    return result


def scientific_status(result: dict, args) -> None:
    missing = []
    for source, details in result["sources"].items():
        for name, passed in details["stage_checks"].items():
            if not passed:
                missing.append({"source": source, "check": name})
    result.update(status="complete_evidence" if not missing else "partial_evidence", incomplete_checks=missing)
    event(args.out, "scientific_status", status=result["status"], incomplete_checks=missing)


def run(args) -> int:
    sources = (args.source,) if args.source else tuple(native.CONDITIONS)
    readout = args.stage in ("all", "readout")
    smoke = args.stage == "smoke"
    timing = args.model == "hiercad" and args.stage in ("all", "timing")
    training = args.model == "hiercad" and args.stage in ("all", "training")
    preparing = args.stage in ("all", "prepare", "readout")
    forwarding = readout or smoke or timing or training
    device, device_name = setup_device(args.device, forwarding)
    axes, manifest_hashes = load_manifests(args, sources)
    for source in sources:
        (args.out / source).mkdir()
    runtime = {
        "device": str(device), "physical_type": device_name, "devices": int(forwarding),
        "sharding": None, "python": platform.python_version(), "torch": torch.__version__,
        "package_versions": package_versions(("torch-npu", "transformers", "peft", "datasets", "accelerate")),
        "execution_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_hashes": manifest_hashes,
        "scope": "Training and validation only. Released generation weights are development references",
    }
    native.write_json(args.out / "run_config.json", runtime)
    selection = {}
    store = None
    member_index = {}
    if preparing or smoke:
        members = list(dict.fromkeys(row["cad_member"] for src in axes.values() for row in src["targets"]))
        store = native.ZipCandidateStore(args.cad_zip, members)
        member_index = {name: i for i, name in enumerate(members)}
        preload = store.preload()
        native.write_json(args.out / "text2cad_vector_store.json", {"load": preload, "archive": store.zip_report})
        for source in sources:
            selection[source] = native.select_readout_targets(axes[source]["targets"], store, member_index)
            event(args.out, "score_panel_selected", source=source,
                  selected=selection[source]["selected"],
                  length_scan_failures=len(selection[source]["failures"]))
    archive = None
    base_model = None
    model = None
    references = {}
    start = time.perf_counter()
    if args.model == "text2cad":
        backend = native.Text2CadReadout(None, store, member_index, device)
        if readout:
            model, load = load_frozen_text2cad(
                args.checkpoint, args.bert_dir, device,
                expected_keys_manifest=ROOT / "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json",
            )
            backend = native.Text2CadReadout(model, store, member_index, device)
            native.write_json(args.out / "model_load.json", {
                "wall_seconds": time.perf_counter() - start, "load": load,
                "weight_protocol": "published Text2CAD v1.0 checkpoint, development only. "
                                   "Exact joint four-level training exposure and selection unverified; "
                                   "CT supervision is outside this checkpoint's target training.",
                "reference_model_version": "Text2CAD v1.0, pinned upstream port",
                "training_loss": "mean of separately masked heads with label smoothing 0.1",
                "score": "teacher-forced raw factorized log probabilities; full 272-row padded input",
            })
    else:
        revision = native.use_hiercad_repo(args.hiercad_repo)
        tokenizer = native.load_hiercad_tokenizer(args.hiercad_base)
        archive = native.MinimalArchive(args.minimal_zip)
        if preparing:
            references = native.candidate_reference(args.hiercad_validation, {
                row["uid"] for selected in selection.values() for row in selected["rows"]
            })
        dtype = getattr(torch, args.hiercad_dtype)
        if readout or timing or smoke:
            model = native.hiercad_scoring_model(args.hiercad_base, args.hiercad_adapter, device, dtype)
        elif training:
            base_model = native.hiercad_base_model(args.hiercad_base, device, torch.bfloat16)
        load_seconds = time.perf_counter() - start
        backend = native.HierCadReadout(model, tokenizer, archive, device)
        native.write_json(args.out / "model_load.json", {
            "wall_seconds": load_seconds, "hiercad_source_sha": revision,
            "base_repo": native.HIERCAD_BASE, "base_expected_snapshot": native.HIERCAD_BASE_REVISION,
            "base": snapshot_facts(args.hiercad_base),
            "adapter_repo": native.HIERCAD_ADAPTER, "adapter_expected_snapshot": native.HIERCAD_ADAPTER_REVISION,
            "adapter": snapshot_facts(args.hiercad_adapter) if args.hiercad_adapter is not None else None,
            "native_reference_data_revision": native.HIERCAD_DATA_REVISION,
            "native_reference_sha256": sha256_file(args.hiercad_validation) if args.hiercad_validation else None,
            "minimal_archive_sha256": sha256_file(args.minimal_zip),
            "dtype": str(dtype), "training_dtype": str(torch.bfloat16), "context_limit": native.HIERCAD_MAX_LENGTH,
            "weight_protocol": "Published HierCAD LoRA uses CADmium reannotations. Development only for T and CT. "
                               "Target benchmark retraining is still required.",
            "trajectory_protocol": "Local geometry-only mapping to published hierarchical_struct_focus renderer, "
                                   "native parser and Qwen assistant mask. Reference bytes are audited separately.",
        })
    event(args.out, "setup_finished", model=args.model, device=str(device),
          seconds=time.perf_counter() - start)
    result = {"model": args.model, "stage": args.stage, "sources": {}, "status": "interrupted"}
    try:
        for source in sources:
            result["sources"][source] = {}
            summary = result["sources"][source]
            checks = summary["stage_checks"] = {}
            if preparing:
                prepared = prepare_candidates(source, selection[source], args, backend, references)
                summary["preparation"] = prepared
                checks["candidate_targets"] = prepared["targets_prepared"] == 4
                if args.model == "hiercad":
                    checks["hiercad_kernel_replay"] = prepared["kernel_replay_passes"] == 4
                    checks["hiercad_released_validation_trajectory"] = prepared["official_validation_bytes_equal"] == 4
            if readout:
                summary["readout"] = validate_readout(source, axes[source], selection[source]["rows"], args, backend)
                checks["native_readout"] = all(row["passed"] for row in summary["readout"].values())
            if smoke:
                tested = smoke_long_contexts(source, selection[source]["rows"], args.out / source, backend)
                checks["long_context_native_ce"] = tested["passed"]
                event(args.out, "long_context_smoke_finished", source=source, passed=tested["passed"])
            if timing:
                stats = measure_timing(source, axes[source], args, backend)
                summary["timing_cells"] = len(stats["cells"])
                summary["timing_cells_measured"] = sum(row["throughput"] is not None for row in stats["cells"])
                summary["timing_projections"] = stats["projection"]
                checks["timing_cells"] = summary["timing_cells_measured"] == summary["timing_cells"]

        if training:
            if base_model is None:
                base_model = model.unload()
                backend.model = None
                del model
            if next(base_model.parameters()).dtype != torch.bfloat16:
                base_model.to(dtype=torch.bfloat16)
            for source in sources:
                try:
                    stats = run_hiercad_training(source, axes[source], args, backend, base_model)
                    result["sources"][source]["train_updates"] = {
                        stage["variant"]: stage.get("steps_updated", 0) for stage in stats["stages"]
                    }
                    result["sources"][source]["stage_checks"]["real_optimizer_updates"] = all(
                        count == args.training_steps for count in result["sources"][source]["train_updates"].values()
                    )

                except Exception as exc:
                    failure = native.error_record(exc)
                    result["sources"][source]["stage_checks"]["real_optimizer_updates"] = False
                    result["sources"][source]["training_failure"] = failure
                    native.write_json(args.out / source / "hiercad_training_failure.json", failure)
                    event(args.out, "hiercad_training_failed", source=source, **failure)
        scientific_status(result, args)
    finally:
        if archive:
            archive.close()
        native.write_json(args.out / "summary.json", result)
    return int(smoke and result["status"] != "complete_evidence")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        return run(args)
    except Exception as exc:
        failure = {**native.error_record(exc), "traceback": traceback.format_exc()}
        native.write_json(args.out / "setup_or_runtime_failure.json", failure)
        event(args.out, "run_failed", **native.error_record(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
