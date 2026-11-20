"""Preflight the Fusion score view and supply an externally authorized row worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval_retrieval_benchmark_fusion import CONDITION, DOMAIN, MAX_VALID_FIELDS
from prepare_retrieval_benchmark_fill import (
    BACKGROUND_ORDER_SHA256, load_json, read_jsonl, text_digest, verify_sha256,
)
from prepare_retrieval_benchmark_fusion_score_inputs import N_BACKGROUND, TEXT_AXIS
from run_retrieval_benchmark_text2cad import (
    ROOT, load_score_row, publish_file_once, publish_score_row, score_columns, score_row_path,
    set_explicit_benchmark_window, token_records,
)

from idea01.text2cad_e1_scoring import FrozenVectorStore


ROLE_AXES = {"query": TEXT_AXIS, "background": "background.jsonl"}
AXIS_FILES = frozenset({*ROLE_AXES.values(), "gallery.jsonl", "score_rows.jsonl"})
SCORE_POLICY = "fp32/full272/b16/no-trim/native-bert"


@dataclass(frozen=True)
class FusionScoreView:
    root: Path
    freeze_sha256: str
    payload_pins: dict[str, str]
    gallery: list[dict]
    rows: list[dict]

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(row["n_valid"] for row in self.gallery)


def read_score_view(root: Path, freeze_sha256: str) -> FusionScoreView:
    root = root.resolve()
    verify_sha256(root / "freeze.json", freeze_sha256)
    manifest = load_json(root / "freeze.json")
    if (manifest.get("schema_version"), manifest.get("kind"), manifest.get("domain"),
        manifest.get("condition"), manifest.get("role_axes")) != (1, "fusion_score_only_inputs", DOMAIN,
                                                                CONDITION, ROLE_AXES):
        raise ValueError("Unexpected Fusion score-only input schema")
    pins = manifest["score_inputs_sha256"]
    for name in AXIS_FILES:
        path = root / name
        if name not in pins or not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing score axis coverage for {name}")
        verify_sha256(path, pins[name])

    gallery = read_jsonl(root / "gallery.jsonl")
    if not gallery:
        raise ValueError("The Fusion scoring gallery is empty")
    seen = []
    for index, row in enumerate(gallery):
        uid = row.get("uid")
        if (set(row) != {"uid", "cad_index", "vector_relpath", "n_valid"}
                or not isinstance(uid, str) or not re.fullmatch(r"[0-9]{4}/[0-9]{8}", uid)
                or type(row["cad_index"]) is not int or row["cad_index"] != index
                or row["vector_relpath"] != f"objects/{uid.replace('/', '_')}/program.npz"
                or type(row["n_valid"]) is not int or not 1 <= row["n_valid"] <= MAX_VALID_FIELDS):
            raise ValueError("Invalid identity, index or field count in the scoring gallery")
        seen.append(uid)
    if seen != sorted(set(seen)):
        raise ValueError("Scoring gallery UIDs must be unique and ordered")
    members = {row["vector_relpath"] for row in gallery}
    entries = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]
    if (set(pins) != AXIS_FILES | members or any(path.is_symlink() for path in entries)
            or {path.relative_to(root).as_posix() for path in entries} != set(pins) | {"freeze.json"}):
        raise ValueError("Unexpected portable Fusion payload coverage")
    for name in sorted(members):
        verify_sha256(root / name, pins[name])
    digest = hashlib.sha256(np.asarray([row["n_valid"] for row in gallery], dtype="<i4").tobytes()).hexdigest()
    if digest != manifest["field_counts_sha256"]:
        raise ValueError("Fusion scoring field count digest differs from its gallery")

    rows = []
    digests = {}
    for role, filename, index_key in (("query", TEXT_AXIS, "text_index"),
                                      ("background", "background.jsonl", "index")):
        axis = read_jsonl(root / filename)
        digests[role] = []
        for index, row in enumerate(axis):
            text = row.get("text")
            if (set(row) != {index_key, "text", "text_sha256"} or type(row[index_key]) is not int
                    or row[index_key] != index or not isinstance(text, str) or not text or text.strip() != text
                    or row["text_sha256"] != text_digest(text)):
                raise ValueError(f"Invalid ordered original text on the Fusion {role} axis")
            digests[role].append(row["text_sha256"])
            rows.append({"row_index": len(rows), "role": role, "axis_index": index,
                         "text_sha256": row["text_sha256"], "text": text})
    role_rows = read_jsonl(root / "score_rows.jsonl")
    expected_rows = [{key: value for key, value in row.items() if key != "text"} for row in rows]
    if (role_rows != expected_rows
            or any(type(row["row_index"]) is not int or type(row["axis_index"]) is not int for row in role_rows)):
        raise ValueError("Frozen score role rows differ from the ordered text axes")
    query, background = digests["query"], digests["background"]
    if not query or query != sorted(set(query)) or len(background) != N_BACKGROUND:
        raise ValueError("Unexpected Fusion unique-query or background role coverage")
    if (manifest["source_pins"]["text_axis_sha256"] != pins[TEXT_AXIS]
            or manifest["source_pins"]["background_ordered_digest"] != BACKGROUND_ORDER_SHA256
            or text_digest("\n".join(background)) != BACKGROUND_ORDER_SHA256):
        raise ValueError("Fusion original text or fixed background pins differ")
    width, height = len(gallery), len(rows)
    if ((manifest["gallery_size"], manifest["unique_query_texts"], manifest["background_count"]) !=
            (width, len(query), N_BACKGROUND) or manifest["score_shape"] != [height, width]
            or manifest["planned_pairs"] != height * width
            or manifest["pairs_by_role"] != {"query": len(query) * width, "background": N_BACKGROUND * width}):
        raise ValueError("Fusion scoring budget or dimensions differ from the two role axes")
    return FusionScoreView(root, freeze_sha256, pins, gallery, rows)


def token_receipts(tokenizer, view: FusionScoreView) -> list[dict]:
    # The audited L3 admission path applies the native 512-token prefix/SEP rule.
    indexed = [dict(row, text_index=row["row_index"]) for row in view.rows]
    audited = token_records(tokenizer, indexed, "l3")
    return [dict(meta, row_index=row["row_index"], role=row["role"], axis_index=row["axis_index"])
            for meta, row in zip(audited, view.rows, strict=True)]


def load_score_store(view: FusionScoreView) -> FrozenVectorStore:
    store = FrozenVectorStore(view.root, view.payload_pins, view.gallery)
    if tuple(store.counts) != view.counts:
        raise ValueError("Frozen store field counts differ from the portable Fusion gallery")
    return store


def cpu_preflight(view: FusionScoreView, bert_dir: Path) -> dict:
    from transformers import BertTokenizer

    from idea01.text2cad_port.loader import PINNED_BERT_FILE_SHA256, verify_bert_asset_shas

    tokenizer_pins = {name: digest for name, digest in PINNED_BERT_FILE_SHA256.items()
                      if name != "model.safetensors"}
    verify_bert_asset_shas(bert_dir, tokenizer_pins)
    tokenizer = BertTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    load_score_store(view)
    tokens = token_receipts(tokenizer, view)
    return {
        "freeze_sha256": view.freeze_sha256, "score_policy": SCORE_POLICY, "zero_forward": True,
        "score_shape": [len(view.rows), len(view.gallery)], "planned_pairs": len(view.rows) * len(view.gallery),
        "tokenizer_sha256": tokenizer_pins, "tokens": tokens,
        "truncation_by_role": dict(Counter(row["role"] for row in tokens if row["truncated"])),
    }


def load_fusion_worker_assets(view: FusionScoreView, checkpoint: Path, bert_dir: Path, device,
                              preflight_receipt: dict):
    """Prepare the pinned model and CPU store for the separately supervised worker."""
    from idea01.text2cad_frozen_runner import EXPECTED_KEYS_MANIFEST_REL, load_frozen_text2cad

    if (preflight_receipt.get("freeze_sha256"), preflight_receipt.get("score_policy"),
        preflight_receipt.get("zero_forward")) != (view.freeze_sha256, SCORE_POLICY, True):
        raise ValueError("Worker preflight does not certify this Fusion score view")
    store = load_score_store(view)
    model, report = load_frozen_text2cad(checkpoint, bert_dir, device,
                                        expected_keys_manifest=ROOT / EXPECTED_KEYS_MANIFEST_REL)
    embedder = model.base_text_embedder
    actual_tokens = token_receipts(embedder.tokenizer, view)
    if actual_tokens != preflight_receipt["tokens"]:
        raise ValueError("Loaded BERT does not reproduce the frozen text admission records")
    if any(row["truncated"] for row in actual_tokens):
        set_explicit_benchmark_window(embedder)
    return model, store, report


@dataclass(frozen=True)
class WorkerBinding:
    """Identifiers supplied by the measured, approved run supervisor."""

    run_id: str
    freeze_sha256: str
    implementation_sha: str
    authorization_sha: str
    measured_probe_sha256: str
    deadline_epoch: float
    shards: int
    max_pairs: int
    max_card_seconds: float

    def scope(self, view: FusionScoreView, run: Path) -> dict:
        if (self.freeze_sha256 != view.freeze_sha256 or self.run_id != run.name
                or any(not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in
                       (self.implementation_sha, self.authorization_sha))
                or not re.fullmatch(r"[0-9a-f]{64}", self.measured_probe_sha256)
                or type(self.shards) is not int or not 1 <= self.shards <= len(view.rows)
                or type(self.max_pairs) is not int or self.max_pairs != len(view.gallery) * len(view.rows)
                or not math.isfinite(self.deadline_epoch) or not math.isfinite(self.max_card_seconds)
                or self.max_card_seconds <= 0):
            raise ValueError("Invalid measured run binding for the frozen Fusion matrix")
        return {**asdict(self), "score_policy": SCORE_POLICY}


def publish_json_once(path: Path, record: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_file_once(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def row_identity(row: dict, scope: dict) -> tuple[dict, str]:
    identity = {"run": scope, **{key: value for key, value in row.items() if key != "text"}}
    return identity, text_digest(json.dumps(identity, sort_keys=True, allow_nan=False))


def completed_row(path: Path, row: dict, scope: dict, width: int) -> bool:
    claim, receipt = path.with_suffix(".claim.json"), path.with_suffix(".json")
    files = (claim, path, receipt)
    if not any(file.exists() or file.is_symlink() for file in files):
        return False
    if any(not file.is_file() or file.is_symlink() for file in files):
        raise ValueError(f"Incomplete Fusion row claim or publication: {path}")
    identity, row_binding = row_identity(row, scope)
    if load_json(claim) != identity:
        raise ValueError(f"Fusion row claim disagrees with the frozen role and run: {path}")
    load_score_row(path, dict(row, text_index=row["row_index"]), row_binding, width)
    return True


def score_fusion_shard(view: FusionScoreView, run: Path, binding: WorkerBinding, shard: int,
                       model, store, device, ledger, invocation: str) -> dict:
    """Score whole columns per role row under the caller's supervised ledger invocation."""
    scope = binding.scope(view, run)
    if time.time() >= binding.deadline_epoch:
        raise TimeoutError("Fusion score deadline reached")
    if type(shard) is not int or not 0 <= shard < binding.shards:
        raise ValueError("Invalid Fusion worker shard")
    if run.resolve().is_relative_to(view.root) or (Path(ledger.path).resolve(), ledger.approved_sha,
            ledger.max_pairs, ledger.max_card_seconds) != ((run / "ledger.json").resolve(),
            binding.implementation_sha, binding.max_pairs, binding.max_card_seconds):
        raise ValueError("Fusion worker ledger must be bound to the approved output run")
    counts = view.counts
    if tuple(store.counts) != counts:
        raise ValueError("Fusion scoring store field counts differ from the reviewed axis")
    run.mkdir(parents=True, exist_ok=True)
    publish_json_once(run / "run_binding.json", scope)
    rows = view.rows[shard::binding.shards]
    paths = [score_row_path(run, CONDITION, row["row_index"]) for row in rows]
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    columns = list(range(len(counts)))
    budget_path = run / "ledger.json"
    if budget_path.exists():
        # Read charges first. Each concurrent worker publishes a claim before adding its charge.
        charged = load_json(budget_path)["pairs_charged"]
        claims = sum(score_row_path(run, CONDITION, row["row_index"]).with_suffix(".claim.json").is_file()
                     for row in view.rows)
        if type(charged) is not int or not 0 <= charged <= claims * len(columns) or charged % len(columns):
            raise ValueError("Charged Fusion pairs do not match durable row claims")
    pending = [(path, row) for path, row in zip(paths, rows, strict=True)
               if not completed_row(path, row, scope, len(columns))]
    for path, row in pending:
        if time.time() >= binding.deadline_epoch:
            raise TimeoutError("Fusion score deadline reached before claiming a row")
        ledger.heartbeat(invocation)
        identity, row_binding = row_identity(row, scope)
        # A durable exclusive claim precedes the charge and the first model call.
        with path.with_suffix(".claim.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(identity, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        ledger.charge_pairs(invocation, len(columns))
        scores = score_columns(model, store, row["text"], columns, device, counts, binding.deadline_epoch)
        if time.time() >= binding.deadline_epoch:
            raise TimeoutError("Fusion score deadline reached before publishing a row")
        ledger.heartbeat(invocation)
        publish_score_row(path, dict(row, text_index=row["row_index"]), row_binding, scores)
    return {"written": len(pending), "verified_previous": len(rows) - len(pending), "shard": shard}


def verify_complete_rows(view: FusionScoreView, run: Path, binding: WorkerBinding) -> dict[str, int]:
    scope = binding.scope(view, run)
    if load_json(run / "run_binding.json") != scope:
        raise ValueError("Fusion result directory disagrees with the approved run binding")
    covered = set()
    for row in view.rows:
        path = score_row_path(run, CONDITION, row["row_index"])
        if not completed_row(path, row, scope, len(view.gallery)):
            raise ValueError(f"Missing frozen Fusion score row: {path}")
        covered.update((path, path.with_suffix(".json"), path.with_suffix(".claim.json")))
    if set((run / "scores" / CONDITION).iterdir()) != covered:
        raise ValueError("Unexpected files on the full Fusion result axis")
    return dict(Counter(row["role"] for row in view.rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="Validate input and token admission without loading a model")
    preflight.add_argument("--input", type=Path, required=True)
    preflight.add_argument("--freeze-sha256", required=True)
    preflight.add_argument("--bert-dir", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    view = read_score_view(args.input, args.freeze_sha256)
    if args.output.resolve().is_relative_to(view.root):
        raise ValueError("Keep the CPU preflight report outside the scoring input view")
    report = cpu_preflight(view, args.bert_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_once(args.output, report)
    print(f"Verified Fusion score shape {report['score_shape']} without model forward")


if __name__ == "__main__":
    main()
