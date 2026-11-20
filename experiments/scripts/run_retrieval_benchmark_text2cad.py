"""Exact native Text2CAD likelihood on the frozen DeepCAD benchmark axes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from idea01.text2cad_e1_scoring import observe_devices, write_json
from run_text2cad_accuracy_night import (
    ensure_outside_checkout, git, initialize_npu, require_clean_detached, sha256_file,
)
from run_text2cad_e1_score import ledger_snapshot, pending_time, stop_owned


AUTH_REF = "refs/remotes/origin/codex/retrieval-benchmark-fill"
AUTH_PATH = "experiments/configs/idea01_retrieval_benchmark_fill_run_authorization.json"
MIRROR = "https://github.com/anszliuziang/CADRetrieval"
RUN_ID = "idea01_retrieval_benchmark_fill_20261120"
RUN_ROOT = Path("/home/ma-user/work/CADRetrieval_runtime/runs") / RUN_ID
GALLERY_SIZE = 8032
BATCH_SIZE = 16
TEXT_COUNTS = {"l0": 0, "ct_deepcad": 2029, "l2": 7604, "l1": 6300, "l3": 7614}
PROBE_WIDTH = 128
REPLAY_ROWS = 4
PROBES_PER_CONDITION = 3
REPLAY_TEXTS_SHA = "c59b4aef8137b1e71c1581291a4c1d2bcbafe36f22d81d01431a5cc02dad7a21"
G4_MANIFEST_SHA = "97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733"


def read_json(path: Path):
    return json.loads(path.read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_row_path(root: Path, condition: str, text_index: int) -> Path:
    return root / "scores" / condition / f"t{text_index:05d}.npy"


def row_receipt(path: Path, row: dict, binding: str) -> dict:
    return {"text_index": row["text_index"], "text_sha256": row["text_sha256"],
            "binding": binding, "scores_sha256": sha256_file(path)}


def publish_file_once(temporary: Path, path: Path) -> None:
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if sha256_file(temporary) != sha256_file(path):
                raise ValueError(f"conflicting completed output: {path}")
    finally:
        temporary.unlink()


def publish_score_row(path: Path, row: dict, binding: str, scores: np.ndarray) -> None:
    if scores.ndim != 1 or scores.dtype != np.dtype("<f8") or not np.isfinite(scores).all():
        raise ValueError("invalid score row")
    temporary = path.with_suffix(f".npy.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.save(stream, scores, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    publish_file_once(temporary, path)
    receipt = path.with_suffix(".json")
    temporary = receipt.with_suffix(f".json.{os.getpid()}.tmp")
    write_json(temporary, row_receipt(path, row, binding))
    publish_file_once(temporary, receipt)


def load_score_row(path: Path, row: dict, binding: str, width: int) -> np.ndarray:
    if read_json(path.with_suffix(".json")) != row_receipt(path, row, binding):
        raise ValueError(f"score row receipt mismatch: {path}")
    scores = np.load(path, allow_pickle=False)
    if scores.shape != (width,) or scores.dtype != np.dtype("<f8") or not np.isfinite(scores).all():
        raise ValueError(f"invalid score row {path}")
    return scores


def assert_equivalent(scores, counts, reference, reference_counts) -> float:
    scores, reference = np.asarray(scores), np.asarray(reference)
    if not np.array_equal(counts, reference_counts):
        raise ValueError("field counts differ from the audited reference")
    if scores.shape != reference.shape:
        raise ValueError("score shape mismatch against the audited reference")
    delta = np.abs(scores - reference)
    if not np.isfinite(delta).all() or np.any(
        delta > np.maximum(1e-3, 1e-6 * np.abs(reference))
    ):
        raise ValueError("score mismatch against the audited reference")
    return float(delta.max())


def prepare_replay(args) -> None:
    """Select historical real B16 batches present in the common gallery."""
    ensure_outside_checkout(args.output, ROOT, "historical replay witness")
    if args.output.exists():
        raise FileExistsError(f"historical replay witness already exists: {args.output}")
    gallery = read_jsonl(args.input / "gallery.jsonl")
    common = {row["cad_index"]: i for i, row in enumerate(gallery)}
    texts_path = args.g4_worker / "texts.jsonl"
    sums = args.g4 / "SHA256SUMS"
    if sha256_file(texts_path) != REPLAY_TEXTS_SHA or sha256_file(sums) != G4_MANIFEST_SHA:
        raise ValueError("historical G4 reference inputs differ from approved pins")
    historical = sorted(read_jsonl(texts_path), key=lambda row: (row["n_chars"], row["text_index"]))
    if len(historical) != 3644:
        raise ValueError("incomplete historical text axis")
    complete_batches = [
        list(range(start, start + BATCH_SIZE))
        for start in range(0, 8046 - BATCH_SIZE + 1, BATCH_SIZE)
        if all(i in common for i in range(start, start + BATCH_SIZE))
    ]
    selected = np.linspace(0, len(complete_batches) - 1, PROBE_WIDTH // BATCH_SIZE, dtype=int)
    old_indices = [i for index in selected for i in complete_batches[index]]
    new_indices = [common[i] for i in old_indices]
    files = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        files[name.removeprefix("./")] = digest
    cases = []
    for offset in np.linspace(0, len(historical) - 1, REPLAY_ROWS, dtype=int):
        text = historical[offset]
        i = text["text_index"]
        card = 0 if i < 1822 else 1
        relative = f"card{card}/chunk_s{card:02d}_t{i:05d}_{i+1:05d}.jsonl"
        path = args.g4 / relative
        digest = sha256_file(path)
        if digest != files[relative] or hash_text(text["text"]) != text["text_sha256"]:
            raise ValueError(f"replay witness drift for historical row {i}")
        records = read_jsonl(path)
        if (
            len(records) != 8047 or records[0]["text_range"] != [i, i + 1]
            or records[0]["n_rejected"] or records[0]["n_scored"] != 8046
        ):
            raise ValueError(f"invalid G4 chunk {relative}")
        values = [records[j + 1] for j in old_indices]
        if any(rec["cad_index"] != j or rec["text_index"] != i or rec["status"] != "scored"
               for rec, j in zip(values, old_indices)):
            raise ValueError(f"inconsistent G4 index in {relative}")
        cases.append({"text": text["text"], "text_sha256": text["text_sha256"],
                      "chunk": relative, "chunk_sha256": digest,
                      "scores": [rec["score_total"] for rec in values],
                      "counts": [rec["n_valid_total"] for rec in values]})
    staged = args.output.with_suffix(f".json.{os.getpid()}.tmp")
    write_json(staged, {"g4_manifest_sha256": G4_MANIFEST_SHA, "cases": cases,
                       "dense_indices": new_indices, "historical_indices": old_indices,
                       "gallery_sha256": sha256_file(args.input / "gallery.jsonl")})
    publish_file_once(staged, args.output)


def frozen_axes(args, auth: dict):
    manifest_path = args.input / "freeze.json"
    if sha256_file(manifest_path) != auth["freeze_sha256"]:
        raise ValueError("input freeze changed")
    files = read_json(manifest_path)["files"]
    wanted = ["gallery.jsonl", *(f"texts_{arm}.jsonl" for arm in auth["conditions"])]
    for name in wanted:
        if files[name] != sha256_file(args.input / name):
            raise ValueError(f"frozen scoring input changed: {name}")
    gallery = read_jsonl(args.input / "gallery.jsonl")
    indexes = [row["cad_index"] for row in gallery]
    if (len(gallery) != GALLERY_SIZE or indexes != sorted(set(indexes))
            or len({row["member"] for row in gallery}) != GALLERY_SIZE):
        raise ValueError("gallery is not the full frozen historical axis")
    texts = {}
    for arm in auth["conditions"]:
        rows = read_jsonl(args.input / f"texts_{arm}.jsonl")
        if (len(rows) != TEXT_COUNTS[arm] or len({r["text"] for r in rows}) != len(rows)
                or any(row["text_index"] != i or not row["text"]
                       or row["text_sha256"] != hash_text(row["text"]) for i, row in enumerate(rows))):
            raise ValueError(f"invalid frozen text records: {arm}")
        texts[arm] = rows
    return gallery, texts


def authorize(args) -> dict:
    git_state = require_clean_detached(ROOT)
    remote = git(ROOT, "remote", "get-url", "origin").removesuffix(".git")
    if (remote != MIRROR or not re.fullmatch(r"[0-9a-f]{40}", args.auth_sha)
            or git(ROOT, "rev-parse", AUTH_REF) != args.auth_sha):
        raise ValueError("unverified project mirror or authorization reference")
    auth = json.loads(git(ROOT, "show", f"{args.auth_sha}:{AUTH_PATH}"))
    if (auth["status"] != "pilot_authorized" or auth["mode"] != args.mode
            or auth["approved_sha"] != git_state["head"]):
        raise ValueError("authorization does not match the detached execution checkout and mode")
    runtime = RUN_ROOT.resolve()
    if (auth["run_id"] != RUN_ID or not re.fullmatch(r"(probe|full)(_[a-z0-9_]+)?", auth["output_directory"])
            or args.input.resolve() != runtime / "frozen"
            or args.out.resolve() != runtime / auth["output_directory"]
            or args.replay.resolve() != runtime / "control/historical_replay.json"):
        raise ValueError("authorization applies to only one bound share4 input and ledger path")
    if not re.fullmatch(r"[0-9a-f]{64}", auth["field_counts_sha256"]):
        raise ValueError("invalid valid-field count pin")
    arms = auth["conditions"]
    if not arms or len(set(arms)) != len(arms) or any(a not in TEXT_COUNTS or a == "l0" for a in arms):
        raise ValueError("invalid condition authorization")
    intended = (2 * PROBE_WIDTH * (REPLAY_ROWS + len(arms) * PROBES_PER_CONDITION)
                if args.mode == "probe" else sum(TEXT_COUNTS[a] * GALLERY_SIZE for a in arms))
    recovery_limit = 0 if args.mode == "probe" else 4 * GALLERY_SIZE
    if type(auth["max_pairs"]) is not int or not intended <= auth["max_pairs"] <= intended + recovery_limit:
        raise ValueError("pair ceiling differs from the planned matrix and bounded row recovery")
    if (not math.isfinite(auth["max_card_seconds"]) or not 0 < auth["max_card_seconds"]
            or not math.isfinite(auth["deadline_epoch"]) or auth["deadline_epoch"] <= time.time()
            or type(auth["max_devices"]) is not int or not 1 <= auth["max_devices"] <= 4):
        raise ValueError("invalid device, card-time or deadline authorization")
    for path in (args.input, args.out, args.replay):
        ensure_outside_checkout(path, ROOT, "benchmark runtime")
    if sha256_file(args.replay) != auth["replay_sha256"]:
        raise ValueError("historical replay file changed")
    if args.mode == "score":
        if args.probe_result is None or sha256_file(args.probe_result) != auth["probe_result_sha256"]:
            raise ValueError("full scoring requires the authorized successful replay receipt")
        ensure_outside_checkout(args.probe_result, ROOT, "probe result")
        probe = read_json(args.probe_result)
        if (probe["passed"] is not True or probe["implementation_sha"] != auth["approved_sha"]
                or probe["freeze_sha256"] != auth["freeze_sha256"]
                or probe["replay_sha256"] != auth["replay_sha256"]
                or set(arms) - set(probe["conditions"])):
            raise ValueError("probe did not certify these score conditions")
        capacity = project_capacity(probe, arms, auth["max_devices"])
        if capacity["card_seconds"] > auth["max_card_seconds"]:
            raise ValueError("card-second cap is below pilot projection")
    return auth


def token_records(tokenizer, texts: list[dict], condition: str) -> list[dict]:
    limit = 512
    result = []
    for row in texts:
        ids = tokenizer(row["text"], truncation=False)["input_ids"]
        if len(ids) > limit and condition != "l3":
            raise ValueError(f"unexpected overlength input in {condition}")
        actual = tokenizer(row["text"], truncation=True, max_length=limit)["input_ids"]
        expected = ids if len(ids) <= limit else ids[: limit - 1] + ids[-1:]
        if actual != expected:
            raise ValueError("BERT input does not follow the frozen prefix/SEP rule")
        result.append({"text_index": row["text_index"], "text_sha256": row["text_sha256"],
                       "token_count": len(ids), "effective_token_count": len(actual),
                       "truncated": len(ids) > limit, "ids_sha256": hash_text(json.dumps(actual))})
    return result


def set_explicit_benchmark_window(embedder):
    original = embedder.tokenizer

    def tokenize(*args, **kwargs):
        # Validated externally for every input, including special tokens.
        kwargs["truncation"] = True
        kwargs["max_length"] = 512
        return original(*args, **kwargs)

    embedder.tokenizer = tokenize
    return original


def install_same_batch_cache(model):
    """Cache the original BERT output for the identical full batch and prompt."""
    import torch

    original = model.base_text_embedder

    class SameBatchCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = original
            self.key = None
            self.embedding = None
            self.hits = 0

        def get_embedding(self, texts):
            key = tuple(texts)
            if key != self.key:
                self.embedding = self.embedder.get_embedding(texts)
                self.key = key
            else:
                self.hits += 1
            return self.embedding

    return original, SameBatchCache().eval()


def score_columns(model, store, text, columns, device, expected_counts, deadline, batch_seconds=None):
    from idea01.text2cad_frozen_runner import forward_and_score_batch

    scores = []
    for start in range(0, len(columns), BATCH_SIZE):
        if time.time() >= deadline:
            raise TimeoutError("benchmark wall-clock deadline reached")
        batch = columns[start : start + BATCH_SIZE]
        started = time.monotonic()
        result = forward_and_score_batch(model, store, [text] * len(batch), batch, device, trim_mode="none")
        if len(result) != len(batch) or any(
            item["n_valid_total"] != expected_counts[i] for item, i in zip(result, batch)
        ):
            raise ValueError("scorer returned an incomplete batch or changed valid fields")
        scores.extend(item["score_total"] for item in result)
        if batch_seconds is not None:
            batch_seconds.append(time.monotonic() - started)
    output = np.array(scores, dtype="<f8")
    if not np.isfinite(output).all():
        raise ValueError("nonfinite Text2CAD score")
    return output


def project_capacity(probe, arms, devices):
    """Project from the slowest measured representative of each text condition."""
    suffix = "cached" if probe["cache_passed"] else "uncached"
    startup = probe["asset_preparation_seconds"]
    work = {}
    for arm in arms:
        measurements = [m for m in probe["measurements"] if m["arm"] == arm]
        if len(measurements) != PROBES_PER_CONDITION:
            raise ValueError(f"incomplete throughput witness for {arm}")
        row_estimate = max(
            times[0] + float(np.mean(times[1:])) * (GALLERY_SIZE // BATCH_SIZE - 1)
            for m in measurements for times in [m[f"{suffix}_batch_seconds"]]
        )
        work[arm] = row_estimate * TEXT_COUNTS[arm]
    return {"devices": devices, "cache_enabled": probe["cache_passed"], "scoring_card_seconds": work,
            "card_seconds": sum(work.values()) + startup * devices * len(arms),
            "wall_seconds": sum(work.values()) / devices + startup * len(arms),
            "new_score_bytes": (GALLERY_SIZE * 8 + 128) * sum(TEXT_COUNTS[a] for a in arms),
            "peak_memory_allocated_bytes": probe["peak_memory_allocated_bytes"],
            "peak_memory_reserved_bytes": probe["peak_memory_reserved_bytes"]}


def run_probe(args, auth, texts, model, store, counts, device, ledger, invocation, asset_seconds):
    import torch

    witness = read_json(args.replay)
    if (witness["g4_manifest_sha256"] != G4_MANIFEST_SHA
            or witness["gallery_sha256"] != sha256_file(args.input / "gallery.jsonl")
            or len(witness["cases"]) != REPLAY_ROWS):
        raise ValueError("replay identity mismatch")
    columns = witness["dense_indices"]
    if len(columns) != PROBE_WIDTH or len(set(columns)) != PROBE_WIDTH or any(not 0 <= j < GALLERY_SIZE for j in columns):
        raise ValueError("invalid probe candidate indices")
    original, cache = install_same_batch_cache(model)
    tokenizer = original.tokenizer
    measurements = {}
    for arm, rows in texts.items():
        meta = token_records(tokenizer, rows, arm)
        ordered = sorted(range(len(rows)), key=lambda i: (meta[i]["token_count"], i))
        targets = np.linspace(0, len(ordered) - 1, PROBES_PER_CONDITION, dtype=int)
        measurements[arm] = [(rows[ordered[i]], meta[ordered[i]]) for i in targets]
    all_cases = [("l0_history", row, None) for row in witness["cases"]] + [
        (arm, row, meta) for arm, selected in measurements.items() for row, meta in selected
    ]
    results = []
    for arm, row, meta in all_cases:
        if arm == "l3":
            set_explicit_benchmark_window(original)
        scores = {}
        timing = {}
        batches = {False: [], True: []}
        for cached in (False, True):
            ledger.heartbeat(invocation)
            ledger.charge_pairs(invocation, len(columns))
            model.base_text_embedder = cache if cached else original
            start = time.monotonic()
            scores[cached] = score_columns(model, store, row["text"], columns, device, counts,
                                           auth["deadline_epoch"], batches[cached])
            timing[cached] = time.monotonic() - start
        original.tokenizer = tokenizer
        max_historical_delta = (assert_equivalent(scores[False], [counts[j] for j in columns],
                                                   row["scores"], row["counts"])
                                if arm == "l0_history" else None)
        results.append({"arm": arm, "text_sha256": row["text_sha256"], "text_tokens": meta,
                        "pairs_per_pass": len(columns), "historical_max_abs_delta": max_historical_delta,
                        "cache_bitwise_equal": bool(np.array_equal(scores[False], scores[True])),
                        "cache_max_abs_delta": float(np.abs(scores[False] - scores[True]).max()),
                        "uncached_seconds": timing[False], "cached_seconds": timing[True],
                        "uncached_batch_seconds": batches[False], "cached_batch_seconds": batches[True]})
    result = {"passed": True, "conditions": list(texts), "measurements": results, "cache_hits": cache.hits,
              "cache_passed": cache.hits > 0 and all(r["cache_bitwise_equal"] for r in results),
              "freeze_sha256": auth["freeze_sha256"], "replay_sha256": auth["replay_sha256"],
              "implementation_sha": auth["approved_sha"], "pairs_charged": auth["max_pairs"],
              "peak_memory_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
              "peak_memory_reserved_bytes": int(torch.npu.max_memory_reserved(device)),
              "asset_preparation_seconds": asset_seconds}
    result["projections"] = [project_capacity(result, texts, n) for n in range(1, 5)]
    return result


def condition_binding(auth: dict, arm: str) -> str:
    return hash_text("\n".join((auth["approved_sha"], auth["freeze_sha256"], arm,
                                auth["probe_result_sha256"], "fp32/full272/b16")))


def run_condition(args, auth, rows, model, store, counts, device, ledger, invocation):
    arm = args.condition
    original, cache = install_same_batch_cache(model)
    meta = token_records(original.tokenizer, rows, arm)
    out = args.out / "scores" / arm
    if arm == "l3":
        set_explicit_benchmark_window(original)
    probe = read_json(args.probe_result)
    model.base_text_embedder = cache if probe["cache_passed"] else original
    binding = condition_binding(auth, arm)
    if args.shard == 0:
        write_json(out / "text_tokens.json", meta)
    written = 0
    resumed = 0
    start = time.monotonic()
    columns = list(range(len(counts)))
    last_log = start
    for row in rows[args.shard :: args.shards]:
        path = score_row_path(args.out, arm, row["text_index"])
        if path.with_suffix(".json").exists():
            load_score_row(path, row, binding, len(columns))
            resumed += 1
            continue
        ledger.heartbeat(invocation)
        ledger.charge_pairs(invocation, len(columns))
        scores = score_columns(model, store, row["text"], columns, device, counts, auth["deadline_epoch"])
        publish_score_row(path, row, binding, scores)
        written += 1
        if written == 1 or time.monotonic() - last_log > 60:
            print(json.dumps({"arm": arm, "shard": args.shard, "rows": written,
                              "elapsed_seconds": round(time.monotonic() - start, 3)}), flush=True)
            last_log = time.monotonic()
    return {"condition": arm, "rows_written": written, "verified_prior_rows": resumed,
            "binding": binding, "cache_hits": cache.hits}


def run_worker(args, auth, gallery, texts):
    import torch
    import torch_npu  # noqa: F401
    from idea01.text2cad_frozen_runner import (
        RunnerConfig, SharedBudgetLedger, check_revision_paths, load_gated_model_and_store, new_invocation_id,
    )
    from idea01.text2cad_vector_scorer import validate_cad_vec

    available, _, _ = observe_devices()
    if args.device not in available or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.device):
        raise ValueError("assigned Ascend device is busy or incorrectly visible")
    torch.set_num_threads(1)
    if torch.npu.device_count() != 1:
        raise ValueError("expected one assigned Ascend device")
    ledger = SharedBudgetLedger(args.out / "ledger.json", approved_sha=auth["approved_sha"],
                                max_pairs=auth["max_pairs"], max_card_seconds=auth["max_card_seconds"])
    invocation = new_invocation_id()
    ledger.begin_invocation(invocation, args.device, 1)
    verdict = 2
    started = time.monotonic()

    def terminate(_sig, _frame):
        raise TimeoutError("benchmark controller terminated this owned worker")

    signal.signal(signal.SIGTERM, terminate)
    try:
        device = torch.device("npu:0")
        environment = initialize_npu(str(device))
        keys = ROOT / "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"
        cfg = RunnerConfig(checkpoint_path=str(args.checkpoint), bert_dir=str(args.bert_dir),
                           cad_seq_zip=str(args.cad_zip), pair_manifest_dir=str(args.input),
                           expected_keys_manifest=str(keys), device=str(device), batch_size=BATCH_SIZE)
        check_revision_paths(cfg)
        model, load, store, checked = load_gated_model_and_store(cfg, [row["member"] for row in gallery])
        if checked["n_loaded"] != GALLERY_SIZE or checked["n_rejected"]:
            raise ValueError(f"frozen gallery candidate rejected: {checked}")
        counts = []
        for i in range(GALLERY_SIZE):
            rec = store.get(i)
            counts.append(validate_cad_vec(rec["cad_vec"], rec["key_padding_mask"],
                                           rec["flag_vec"], rec["index_vec"]).n_valid_total)
        if hashlib.sha256(np.asarray(counts, dtype="<i4").tobytes()).hexdigest() != auth["field_counts_sha256"]:
            raise ValueError("official candidate field counts differ from the frozen gallery")
        ledger.heartbeat(invocation)
        details = (run_probe(args, auth, texts, model, store, counts, device, ledger, invocation,
                             time.monotonic() - started)
                   if args.mode == "probe" else
                   run_condition(args, auth, texts[args.condition], model, store, counts, device, ledger, invocation))
        ledger.heartbeat(invocation)
        if time.time() >= auth["deadline_epoch"]:
            raise TimeoutError("worker ran past the benchmark deadline")
        record = {"invocation": invocation, "device": args.device, "environment": environment,
                  "load": load, **details}
        write_json(args.out / f"{args.condition}_worker{args.shard}_{invocation}.json", record)
        verdict = 0
    finally:
        ledger.finalize_invocation(invocation, exit_code=verdict, note=f"benchmark_{args.condition}")


def run_controller(args, auth, texts):
    args.out.mkdir(parents=True, exist_ok=True)
    binding_path = args.out / "run_authorization.json"
    if binding_path.exists():
        if args.mode != "score" or read_json(binding_path) != auth:
            raise ValueError("refusing to reuse a probe or a differently authorized run directory")
    else:
        write_json(binding_path, auth)
    if any(not inv["finalized"] for inv in ledger_snapshot(args.out / "ledger.json")["invocations"]):
        raise ValueError("run has a pending scoring invocation")
    if (args.out / "controller_exit.json").exists():
        raise ValueError("completed benchmark controller must not be relaunched")
    axis_path = args.out / "gallery.jsonl"
    staged = axis_path.with_suffix(f".jsonl.{os.getpid()}.tmp")
    shutil.copyfile(args.input / "gallery.jsonl", staged)
    publish_file_once(staged, axis_path)
    conditions = ["probe"] if args.mode == "probe" else auth["conditions"]
    if args.mode == "score":
        remaining = [arm for arm in conditions if not (args.out / f"{arm}_complete.json").exists()]
        capacity = project_capacity(read_json(args.probe_result), remaining, auth["max_devices"])
        if time.time() + capacity["wall_seconds"] >= auth["deadline_epoch"]:
            raise ValueError("benchmark wall limit is shorter than the pilot projection")
        if shutil.disk_usage(args.out).free < capacity["new_score_bytes"] * 1.1:
            raise ValueError("insufficient disk for the projected score matrix")
        write_json(args.out / "capacity_projection.json", capacity)
    for arm in conditions:
        if args.mode == "score":
            marker = args.out / f"{arm}_complete.json"
            if marker.exists():
                for row in texts[arm]:
                    load_score_row(score_row_path(args.out, arm, row["text_index"]), row,
                                   condition_binding(auth, arm), GALLERY_SIZE)
                continue
            (args.out / "scores" / arm).mkdir(parents=True, exist_ok=True)
        while True:
            if time.time() >= auth["deadline_epoch"] - 15:
                raise TimeoutError("no free device before the benchmark deadline")
            free, inventory, mapping = observe_devices()
            write_json(args.out / "last_availability.json", {"arm": arm, "utc_epoch": time.time(),
                       "free": free, "inventory": inventory, "mapping": mapping})
            if free:
                break
            time.sleep(60)
        devices = free[:1 if args.mode == "probe" else auth["max_devices"]]
        processes = []
        files = []
        try:
            for shard, device in enumerate(devices):
                suffix = f"{arm}_shard{shard}_{time.time_ns()}"
                stdout = (args.out / f"{suffix}.stdout.log").open("xb")
                stderr = (args.out / f"{suffix}.stderr.log").open("xb")
                files.extend((stdout, stderr))
                command = [
                    sys.executable, "-B", "-u", str(Path(__file__).resolve()), "worker",
                    "--mode", args.mode, "--auth-sha", args.auth_sha, "--condition", arm,
                    "--device", str(device), "--shard", str(shard), "--shards", str(len(devices)),
                ]
                for name in ("input", "out", "replay", "probe_result", "checkpoint", "bert_dir", "cad_zip"):
                    value = getattr(args, name)
                    if value is not None:
                        command += [f"--{name.replace('_', '-')}", str(value)]
                worker_env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device),
                                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                           env=worker_env, start_new_session=True)
                processes.append(process)
                write_json(args.out / f"{suffix}.launch.json", {"pid": process.pid, "device": device,
                           "argv": command, "started_epoch": time.time()})
            while True:
                statuses = [process.poll() for process in processes]
                if any(status not in (None, 0) for status in statuses):
                    raise RuntimeError(f"scoring worker on {arm} failed: {statuses}")
                if all(status == 0 for status in statuses):
                    break
                snapshot = ledger_snapshot(args.out / "ledger.json")
                if (pending_time(snapshot, time.time()) + 12 * len(devices) >= auth["max_card_seconds"]
                        or time.time() + 12 >= auth["deadline_epoch"]):
                    raise TimeoutError("stopping owned workers at the reserved budget cutoff")
                time.sleep(2)
        finally:
            try:
                stop_owned(processes)
            finally:
                for stream in files:
                    stream.close()
                exits = [process.poll() for process in processes]
                write_json(args.out / f"{arm}_exit_{time.time_ns()}.json", {"devices": devices, "exits": exits})
        snapshot = ledger_snapshot(args.out / "ledger.json")
        if (any(not inv["finalized"] for inv in snapshot["invocations"])
                or snapshot["card_seconds_spent"] > auth["max_card_seconds"]
                or time.time() > auth["deadline_epoch"]):
            raise ValueError("budget or settlement check failed")
        if args.mode == "probe":
            records = list(args.out.glob("probe_worker0_*.json"))
            if len(records) != 1 or snapshot["pairs_charged"] != auth["max_pairs"]:
                raise ValueError("probe did not finish all planned pairs")
            write_json(args.out / "probe_result.json", read_json(records[0]))
        else:
            for row in texts[arm]:
                load_score_row(score_row_path(args.out, arm, row["text_index"]), row,
                               condition_binding(auth, arm), GALLERY_SIZE)
            write_json(marker, {"condition": arm, "complete_rows": len(texts[arm]),
                       "gallery": GALLERY_SIZE, "devices": devices, "pairs_charged_total": snapshot["pairs_charged"],
                       "cumulative_card_seconds": snapshot["card_seconds_spent"],
                       "binding": condition_binding(auth, arm)})
    write_json(args.out / "controller_exit.json", {"exit_code": 0, "conditions": conditions,
               "implementation_sha": auth["approved_sha"], "authorization_sha": args.auth_sha})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("prepare-replay")
    for name in ("input", "g4", "g4-worker", "output"):
        replay.add_argument(f"--{name}", type=Path, required=True)
    for command in ("controller", "worker"):
        p = sub.add_parser(command)
        p.add_argument("--mode", required=True, choices=("probe", "score"))
        p.add_argument("--auth-sha", required=True)
        for name in ("input", "out", "replay", "checkpoint", "bert-dir", "cad-zip"):
            p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument("--probe-result", type=Path)
        if command == "worker":
            p.add_argument("--condition", required=True)
            p.add_argument("--device", type=int, required=True)
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare-replay":
        prepare_replay(args)
        return
    auth = authorize(args)
    gallery, texts = frozen_axes(args, auth)
    if args.command == "controller":
        import fcntl

        def request_stop(_sig, _frame):
            raise InterruptedError("benchmark controller received a stop request")

        signal.signal(signal.SIGTERM, request_stop)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / ".controller.lock").open("a+b") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("a benchmark controller already owns this run") from exc
            run_controller(args, auth, texts)
    else:
        if not 0 <= args.shard < args.shards <= auth["max_devices"]:
            raise ValueError("invalid shard assignment")
        if (args.mode == "probe" and args.condition != "probe"
                or args.mode == "score" and args.condition not in texts):
            raise ValueError("worker condition is outside the authorization")
        run_worker(args, auth, gallery, texts)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        traceback.print_exception(error)
        sys.exit(2)
