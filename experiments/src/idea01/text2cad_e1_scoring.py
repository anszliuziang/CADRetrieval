"""Reuse the audited Text2CAD scorer on frozen E1 query and gallery axes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import numpy as np


BATCH_SIZE = 16
STAGES = {
    "probe": ("replay", "probe"),
    "source": ("b", "b_official", "a"),
    "target": ("c", "c_alt", "c_rewrite"),
}
GALLERIES = {
    "a": "deepcad",
    "b": "deepcad",
    "b_official": "deepcad",
    "c": "fusion360",
    "c_alt": "fusion_original",
    "c_rewrite": "fusion360",
    "replay": "replay",
    "probe": "probe",
}
BG_ARMS = {"b", "b_official", "c", "c_alt"}
OFFICIAL_ARMS = {"b_official", "replay"}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def input_path(root: Path, manifest: dict, relpath: str) -> Path:
    path = (root / relpath).resolve()
    if root.resolve() not in path.parents or relpath not in manifest:
        raise ValueError(f"unregistered frozen input {relpath!r}")
    if sha256_file(path) != manifest[relpath]:
        raise ValueError(f"frozen digest mismatch: {relpath}")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


@dataclass(frozen=True)
class WorkArm:
    name: str
    queries: list[dict]
    gallery: list[dict]
    background: list[dict]

    @property
    def row_ids(self) -> list[str]:
        return [r["uid"] for r in self.queries] + [
            f"background:{row['index']}" for row in self.background
        ]

    @property
    def column_ids(self) -> list[str]:
        return [r["uid"] for r in self.gallery]

    @property
    def pairs(self) -> int:
        return (len(self.queries) + len(self.background)) * len(self.gallery)


def load_freeze(root: Path, expected_sha: str) -> dict:
    path = root / "freeze.json"
    if sha256_file(path) != expected_sha:
        raise ValueError("freeze.json digest mismatch")
    freeze = json.loads(path.read_bytes())
    if freeze["gallery_size"] not in (256, 512):
        raise ValueError("E1 gallery size must be a registered size")
    return freeze


def get_arms(root: Path, freeze: dict, stage: str) -> list[WorkArm]:
    manifest = freeze["score_inputs_sha256"]
    bgs = read_jsonl(input_path(root, manifest, "background.jsonl"))
    if len(bgs) != 48 or [r["index"] for r in bgs] != list(range(48)):
        raise ValueError("expected the frozen ordered 48 background texts")
    arms = []
    for name in STAGES[stage]:
        if name not in freeze["scored_arms"]:
            continue
        q = read_jsonl(input_path(root, manifest, f"queries_{name}.jsonl"))
        if name == "c_rewrite":
            q = [r for r in q if r["status"] in ("changed_valid", "no_edit")]
        gallery = read_jsonl(input_path(root, manifest, f"gallery_{GALLERIES[name]}.jsonl"))
        columns = [r["uid"] for r in gallery]
        keys = [(r["uid"], r.get("variant")) for r in q]
        if not q or not columns or len(set(keys)) != len(keys) or len(set(columns)) != len(columns):
            raise ValueError(f"empty or duplicate frozen score axis for {name}")
        for row in q + (bgs if name in BG_ARMS else []):
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"invalid text in {name}")
            if "text_sha256" in row and row["text_sha256"] != sha256_text(row["text"]):
                raise ValueError(f"text digest mismatch in {name}")
        arms.append(WorkArm(name, q, gallery, bgs if name in BG_ARMS else []))
    if {a.name for a in arms} & {"a"} and not any(a.name == "b" for a in arms):
        raise ValueError("A requires the same B gallery and background")
    mandatory = {"probe": {"replay"}, "source": {"b", "b_official"}, "target": {"c"}}
    if not mandatory[stage] <= {arm.name for arm in arms}:
        raise ValueError(f"missing required scoring arm in {stage}")
    n = freeze["gallery_size"]
    for arm in arms:
        if arm.name in ("a", "b", "b_official", "c", "c_rewrite") and len(arm.gallery) != n:
            raise ValueError(f"{arm.name}: gallery size differs from freeze")
        if arm.name in ("b", "c", "c_alt") and len(arm.queries) != n:
            raise ValueError(f"{arm.name}: expected one original query per main CAD")
        if arm.name == "b_official" and len(arm.queries) != 256:
            raise ValueError("B encoding check must contain 256 queries")
        if arm.name == "c_rewrite" and len(arm.queries) > 96:
            raise ValueError("C rewrite exceeds the frozen 48 by two budget")
        if arm.name == "c_alt" and not 1 <= len(arm.gallery) <= 32:
            raise ValueError("C alternate arm requires one to 32 replacement columns")
        if arm.name == "replay" and (len(arm.queries), len(arm.gallery)) != (16, 16):
            raise ValueError("expected the 16 by 16 historical replay")
        if arm.name == "probe" and (len(arm.queries), len(arm.gallery)) != (2, 16):
            raise ValueError("expected the two representatives with sixteen test inputs")
    return arms


def compare_replay(observed, counts, reference, reference_counts) -> dict:
    if observed.shape != reference.shape or counts.shape != reference_counts.shape:
        raise ValueError("replay shapes differ")
    if observed.ndim != 2 or counts.shape != (observed.shape[1],):
        raise ValueError("invalid replay matrix or counts")
    if not (np.isfinite(observed).all() and np.isfinite(reference).all()):
        raise ValueError("nonfinite replay scores")
    delta = np.abs(observed - reference)
    bound = np.maximum(1e-3, 1e-6 * np.abs(reference))
    wrong = int(np.count_nonzero(delta > bound))
    counts_wrong = int(np.count_nonzero(counts != reference_counts))
    left, right = np.triu_indices(observed.shape[1], 1)
    margin = reference[:, left] - reference[:, right]
    replay_margin = observed[:, left] - observed[:, right]
    confident = np.abs(margin) > bound[:, left] + bound[:, right]
    changed = np.sign(margin) != np.sign(replay_margin)
    ordering_wrong = int(np.count_nonzero(confident & changed))
    return {
        "passed": wrong == 0 and counts_wrong == 0 and ordering_wrong == 0,
        "pairs": int(observed.size),
        "max_abs_delta": float(delta.max()),
        "out_of_tolerance_pairs": wrong,
        "candidate_counts_mismatched": counts_wrong,
        "distinct_order_violations": ordering_wrong,
        "indistinguishable_comparisons": int(np.count_nonzero(~confident)),
        "indistinguishable_order_changes": int(np.count_nonzero(~confident & changed)),
    }


def merge_shards(paths, query_uids, cad_uids) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.empty((len(query_uids), len(cad_uids)), dtype=np.float64)
    seen = np.zeros(len(query_uids), dtype=bool)
    counts = None
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            rows = file["rows"]
            if (
                rows.ndim != 1 or rows.dtype.kind not in "iu"
                or np.any(rows < 0) or np.any(rows >= len(query_uids))
            ):
                raise ValueError("out-of-range score shard row indices")
            if np.unique(rows).size != rows.size or seen[rows].any():
                raise ValueError("duplicate score shard rows")
            if (
                file["cad_uids"].tolist() != cad_uids
                or file["query_uids"].tolist() != [query_uids[i] for i in rows]
            ):
                raise ValueError("score shard UID axis mismatch")
            scores, observed_counts = file["scores"], file["counts"]
            if (
                scores.shape != (len(rows), len(cad_uids))
                or scores.dtype != np.float64 or not np.isfinite(scores).all()
            ):
                raise ValueError("invalid or non-finite score shard matrix")
            if (
                observed_counts.shape != (len(cad_uids),) or observed_counts.dtype.kind not in "iu"
                or np.any(observed_counts <= 0)
            ):
                raise ValueError("invalid score shard field counts")
            if counts is None:
                counts = observed_counts
            elif not np.array_equal(counts, observed_counts):
                raise ValueError("shards disagree on valid field counts")
            matrix[rows] = scores
            seen[rows] = True
    if counts is None or not seen.all():
        raise ValueError("incomplete score shard row coverage")
    return matrix, counts


class FrozenVectorStore:
    """In-memory read-only candidate store for the audited batch assembler."""

    def __init__(self, root: Path, manifest: dict, rows: list[dict]):
        import torch

        from idea01.text2cad_frozen_runner import generate_attention_mask
        from idea01.text2cad_vector_scorer import validate_cad_vec

        self.canonical_attn_mask = generate_attention_mask(271)
        self.records = []
        self.counts = []
        fields = {
            "cad_vec": (np.dtype("<i4"), (272, 2)),
            "flag_vec": (np.dtype("<i4"), (272,)),
            "index_vec": (np.dtype("<i4"), (272,)),
            "key_padding_mask": (np.bool_, (272, 2)),
        }
        for row in rows:
            path = input_path(root, manifest, row["vector_relpath"])
            with np.load(path, allow_pickle=False) as archive:
                tensors = {}
                for key, (dtype, shape) in fields.items():
                    arr = archive[key]
                    if arr.dtype != dtype or arr.shape != shape:
                        raise ValueError(f"invalid {key} for {row['uid']}")
                    tensors[key] = torch.from_numpy(np.ascontiguousarray(arr))
            diag = validate_cad_vec(
                tensors["cad_vec"], tensors["key_padding_mask"],
                tensors["flag_vec"], tensors["index_vec"],
            )
            if "n_valid" in row and diag.n_valid_total != row["n_valid"]:
                raise ValueError(f"candidate field-count mismatch for {row['uid']}")
            tensors["n_real_tokens"] = diag.n_real_tokens
            self.counts.append(diag.n_valid_total)
            self.records.append(tensors)

    def get(self, index: int) -> dict:
        return self.records[index]


def arm_stores(args, arms, freeze) -> dict:
    from transformers import BertTokenizer

    from idea01.text2cad_frozen_runner import ZipCandidateStore
    from idea01.text2cad_vector_scorer import validate_cad_vec

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_dir), local_files_only=True)
    manifest = freeze["score_inputs_sha256"]
    stores = {}
    for arm in arms:
        for row in arm.queries + arm.background:
            ids = tokenizer.encode(row["text"], truncation=False)
            length = len(ids)
            if length > 512 or ("token_count" in row and row["token_count"] != length):
                raise ValueError(f"actual tokenizer length violates frozen input for {arm.name}")
            digest = sha256_text(json.dumps(ids, separators=(",", ":")))
            if row.get("token_ids_sha256", digest) != digest:
                raise ValueError(f"actual token IDs differ from the frozen input for {arm.name}")
        if arm.name in OFFICIAL_ARMS:
            store = ZipCandidateStore(args.cad_zip, [row["official_member"] for row in arm.gallery])
            report = store.preload()
            if report["n_rejected"]:
                raise ValueError(f"official gallery rejected in {arm.name}: {report}")
            counts = [
                validate_cad_vec(
                    r["cad_vec"], r["key_padding_mask"], r["flag_vec"], r["index_vec"],
                ).n_valid_total
                for r in (store.get(i) for i in range(len(arm.gallery)))
            ]
        else:
            store = FrozenVectorStore(args.input, manifest, arm.gallery)
            counts = store.counts
        stores[arm.name] = store, np.asarray(counts, dtype=np.int64)
    return stores


def preflight(args, arms, freeze) -> dict:
    return {arm: counts for arm, (_, counts) in arm_stores(args, arms, freeze).items()}


def worker(args, arms, freeze, auth) -> dict:
    import torch
    import torch_npu

    from idea01.text2cad_frozen_runner import (
        SharedBudgetLedger, forward_and_score_batch,
        load_frozen_text2cad, new_invocation_id,
    )

    torch.set_num_threads(1)
    stores = arm_stores(args, arms, freeze)
    available, _, _ = observe_devices()
    if args.device not in available:
        raise RuntimeError("the chosen NPU became busy during CPU preflight")

    ledger = SharedBudgetLedger(
        args.out / "ledger.json", approved_sha=auth["approved_sha"],
        max_pairs=auth["max_pairs"], max_card_seconds=auth["max_card_seconds"],
    )
    invocation = new_invocation_id()
    ledger.begin_invocation(invocation, shard_id=args.device, cards_used=1)
    result = {"invocation_id": invocation, "stages": {}, "started_epoch": time.time()}
    verdict = 2
    completed_pairs = 0

    def terminate(_signum, _frame):
        raise TimeoutError("worker stopped by its experiment controller")

    try:
        signal.signal(signal.SIGTERM, terminate)
        device = torch.device("npu:0")
        from run_text2cad_accuracy_night import initialize_npu

        if torch.npu.device_count() != 1 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.device):
            raise RuntimeError("worker does not see exactly its assigned Ascend device")
        result["environment"] = initialize_npu(str(device))
        result["environment"]["container_logic_id"] = args.device
        result["environment"]["torch_npu"] = str(torch_npu.__version__)
        model, load = load_frozen_text2cad(
            args.checkpoint, args.bert_dir, str(device), expected_keys_manifest=args.keys_manifest,
        )
        result["strict_load"] = load["strict_load"]
        for arm in arms:
            arm_started = time.monotonic()
            all_rows = arm.queries + arm.background
            assigned = np.arange(args.shard, len(all_rows), args.shards, dtype=np.int64)
            ncols = len(arm.gallery)
            uid_axis = arm.column_ids
            store, expected_counts = stores[arm.name]
            matrix = np.empty((len(assigned), ncols), dtype=np.float64)
            representative_times = []
            for ordinal, row in enumerate(assigned):
                row_started = time.monotonic()
                ledger.heartbeat(invocation)
                ledger.charge_pairs(invocation, ncols)
                for first in range(0, ncols, BATCH_SIZE):
                    if time.time() >= auth["deadline_epoch"]:
                        raise TimeoutError("E1 deadline expired")
                    indices = list(range(first, min(first + BATCH_SIZE, ncols)))
                    records = forward_and_score_batch(
                        model, store, [all_rows[row]["text"]] * len(indices), indices, device,
                        trim_mode="none",
                    )
                    if len(records) != len(indices):
                        raise ValueError("scorer returned an incomplete batch")
                    for idx, rec in zip(indices, records):
                        if rec["n_valid_total"] != expected_counts[idx]:
                            raise ValueError(f"scorer field count drift for {uid_axis[idx]}")
                        matrix[ordinal, idx] = rec["score_total"]
                    completed_pairs += len(indices)
                if arm.name == "probe":
                    representative_times.append({
                        "uid": all_rows[row]["uid"], "token_count": all_rows[row]["token_count"],
                        "pairs": ncols, "seconds": time.monotonic() - row_started,
                    })
                if ordinal == 0 or (ordinal + 1) % 16 == 0 or ordinal + 1 == len(assigned):
                    print(json.dumps({
                        "arm": arm.name, "shard": args.shard,
                        "rows_done": ordinal + 1, "rows_total": len(assigned),
                        "completed_pairs": completed_pairs,
                    }), flush=True)
            if not np.isfinite(matrix).all():
                raise ValueError(f"nonfinite matrix for {arm.name}")
            output = args.out / args.stage / f"{arm.name}_shard{args.shard}.npz"
            write_npz(
                output, rows=assigned,
                query_uids=np.array([arm.row_ids[i] for i in assigned], dtype=str),
                cad_uids=np.array(arm.column_ids, dtype=str), scores=matrix,
                counts=expected_counts,
            )
            result["stages"][arm.name] = {
                "pairs": int(matrix.size), "sha256": sha256_file(output),
                "seconds": time.monotonic() - arm_started,
            }
            if representative_times:
                result["stages"][arm.name]["representatives"] = representative_times
        ledger.heartbeat(invocation)
        if time.time() >= auth["deadline_epoch"]:
            raise TimeoutError("E1 deadline expired at worker settlement")
        result["ended_epoch"] = time.time()
        result["completed_pairs"] = completed_pairs
        write_json(args.out / args.stage / f"worker{args.shard}_record.json", result)
        verdict = 0
        return result
    finally:
        ledger.finalize_invocation(invocation, exit_code=verdict, note=f"e1_{args.stage}")
        write_json(args.out / args.stage / f"worker{args.shard}_settlement.json", {
            "invocation_id": invocation, "exit_code": verdict, "completed_pairs": completed_pairs,
        })


def idle_from_smi(inventory: str, mapping: str) -> list[int]:
    """Use the driver's logic-ID mapping rather than assuming printed order."""
    keys = {
        (int(npu), int(chip)): int(logic)
        for npu, chip, logic in re.findall(
            r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+Ascend910\b", mapping, re.M,
        )
    }
    if len(keys) != 4 or set(keys.values()) != {0, 1, 2, 3} or "Process id" not in inventory:
        raise ValueError("unexpected share4 NPU inventory or chip mapping")
    descriptions, process_table = inventory.split("Process id", 1)
    active = {
        (int(npu), int(chip))
        for npu, chip in re.findall(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*\d+\s*\|", process_table, re.M)
    }
    if not active <= keys.keys():
        raise ValueError("unknown NPU process row")
    free = []
    current = None
    healthy = False
    recognized = set()
    for line in descriptions.splitlines():
        header = re.match(r"^\|\s*(\d+)\s+Ascend910\s*\|\s*(\S+)", line)
        if header:
            current, healthy = int(header[1]), header[2] == "OK"
        chip = re.match(
            r"^\|\s*(\d+)\s+\d+\s*\|\s*[0-9a-fA-F:.]+\s*\|\s*(\d+)\s+"
            r"\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|", line,
        )
        if chip:
            key = (current, int(chip[1]))
            if key not in keys or key in recognized:
                raise ValueError("unexpected NPU chip row")
            recognized.add(key)
            if healthy and int(chip[2]) == 0 and int(chip[3]) < 8192 and key not in active:
                free.append(keys[key])
    if recognized != keys.keys():
        raise ValueError("NPU health table incomplete")
    return sorted(free)


def observe_devices() -> tuple[list[int], str, str]:
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True, timeout=20)
    smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    return idle_from_smi(smi, mapping), smi, mapping
