"""Native conditional scoring and training probes for the validation-only pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from .text2cad_frozen_runner import ZipCandidateStore, assemble_batch
from .text2cad_port.model import build_teacher_forcing_inputs, teacher_forced_logits
from .text2cad_vector_scorer import two_head_factorized_score

HIERCAD_COMMIT = "c10c4dce9fb3674932431905879ec597b65b9122"
HIERCAD_BASE = "Qwen/Qwen2.5-Coder-14B-Instruct"
HIERCAD_BASE_REVISION = "aedcc2d42b622764e023cf882b6652e646b95671"
HIERCAD_ADAPTER = "xujimin/HierCAD-14B"
HIERCAD_ADAPTER_REVISION = "6e14c276c9347d2fbdbbe1159b21efdf08ae0a95"
HIERCAD_DATA_REVISION = "0482b88ecfb8bbd723057aac2bf64279a2e2265a"
HIERCAD_MAX_LENGTH = 8192
BANDS = ("short", "middle", "long")
CONDITIONS = {"t2cad-v1.0": ("L0", "L1", "L2", "L3"), "ct-deepcad": ("CT",)}
T2_NLL_ATOL = 2e-5
H_NLL_ATOL = 2e-5
T2_SCORE_ATOL, T2_SCORE_RTOL = 1e-3, 1e-6
H_SCORE_MEAN_ATOL = 0.05
T2_CAUSAL_ATOL, H_CAUSAL_ATOL = 3e-5, 0.05
# Output closings and intervening whitespace belong to the adjacent block.
_SECTION_RE = re.compile(
    r"<(GLOBAL_PLAN|PART_PLAN|PART_JSON)(?: part_id=part_\d+)?>"
)
_CURVE_FIELDS = {
    "line": (("Start Point", 2), ("End Point", 2)),
    "arc": (("Start Point", 2), ("Mid Point", 2), ("End Point", 2)),
    "circle": (("Center", 2), ("Radius", 0)),
}
_EXTRUSION_NUMBERS = (
    "extrude_depth_towards_normal", "extrude_depth_opposite_normal", "sketch_scale"
)
_OPERATIONS = {
    "NewBodyFeatureOperation", "JoinFeatureOperation",
    "CutFeatureOperation", "IntersectFeatureOperation",
}


def digest(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def digest_json(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def digest_tensor(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        return digest(cpu.view(torch.int16).numpy().tobytes())
    return digest(cpu.numpy().tobytes())


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def error_record(exc: Exception) -> dict:
    return {
        "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        "reason": getattr(exc, "reason", None),
    }


def use_hiercad_repo(path: Path) -> str:
    path = path.resolve(strict=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != HIERCAD_COMMIT:
        raise ValueError(f"HierCAD source revision {actual} differs from {HIERCAD_COMMIT}")
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("HierCAD source checkout has tracked changes")
    sys.path.insert(0, str(path))
    return actual


def load_hiercad_tokenizer(base: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base.resolve(strict=True)), local_files_only=True, trust_remote_code=False,
        padding_side="right", use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("HierCAD token boundary audit requires the official fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _geometry_number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("CAD geometry field is not a finite number")
    return round(value, 4) if isinstance(value, float) else value


def _geometry_fields(record, fields):
    clean = {}
    for key, width in fields:
        value = record[key]
        if width:
            if not isinstance(value, (list, tuple)) or len(value) != width:
                raise ValueError(f"wrong dimension for geometry field {key}")
            clean[key] = [_geometry_number(number) for number in value]
        else:
            clean[key] = _geometry_number(value)
    return clean


@dataclass(frozen=True)
class HierCadTarget:
    response: str
    replay_program: dict
    example: dict


def hiercad_target(minimal: dict) -> HierCadTarget:
    """Derive a geometry-only HierCAD record, then use the released renderer/parser.

    The pre-rendering conversion is local. The author release starts from
    prebuilt hierarchical records and provides the rendering and parser.
    Sequence order follows the minimal program. Indices are ordinal and
    regenerated. The source's names, annotations and descriptive fields are
    excluded by the geometry field whitelist.
    """
    from hiercad.src.utils.hierarchical_alignment import build_training_example
    from hiercad.src.utils.hierarchical_parse import merge_part_jsons, parse_hierarchical_response

    plan = []
    part_plans = {}
    parts = {}
    for part_no, part in enumerate(minimal["parts"].values(), 1):
        part_id = f"part_{part_no}"
        operation = part["extrusion"]["operation"]
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported extrusion operation {operation!r}")
        extrusion = _geometry_fields(part["extrusion"], [(key, 0) for key in _EXTRUSION_NUMBERS])
        extrusion["operation"] = operation
        sketch = {}
        topology = {}
        for face_no, face in enumerate(part["sketch"].values(), 1):
            face_id = f"face_{face_no}"
            sketch[face_id] = {}
            topology[face_id] = {}
            for loop_no, loop in enumerate(face.values(), 1):
                loop_id = f"loop_{loop_no}"
                signature = []
                typed_counts = Counter()
                curves = {}
                for key, curve in loop.items():
                    match = re.fullmatch(r"(line|arc|circle)_\d+", key)
                    if match is None:
                        raise ValueError(f"unsupported primitive key {key!r}")
                    kind = match[1]
                    typed_counts[kind] += 1
                    signature.append(kind)
                    curves[f"{kind}_{typed_counts[kind]}"] = _geometry_fields(curve, _CURVE_FIELDS[kind])
                if not curves:
                    raise ValueError("empty sketch loop")
                sketch[face_id][loop_id] = curves
                topology[face_id][loop_id] = "|".join(signature)
            if not sketch[face_id]:
                raise ValueError("empty sketch face")
        if not sketch:
            raise ValueError("empty sketch")
        parts[part_id] = {
            "coordinate_system": _geometry_fields(
                part["coordinate_system"], (("Euler Angles", 3), ("Translation Vector", 3))
            ),
            "sketch": sketch,
            "extrusion": extrusion,
        }
        plan.append({"part_id": part_id, "operation": operation})
        part_plans[part_id] = {"topology_hint": topology}
    if not parts:
        raise ValueError("empty CAD program")
    example = build_training_example(
        {
            "uid": "", "text_input": "", "global_plan": {"plan_sequence": plan},
            "part_plans": part_plans, "part_jsons": parts,
        },
        prompt_mode="hierarchical_struct_focus",
    )
    response = example["response"]
    parsed = parse_hierarchical_response(response)
    replay = merge_part_jsons(parsed)
    if (parsed["parse_errors"] or parsed["global_plan"]["plan_sequence"] != plan
            or digest_json(replay["parts"]) != digest_json(parts)):
        raise ValueError("native HierCAD parser did not round-trip the geometry and operation order")
    return HierCadTarget(response, replay, example)


@dataclass(frozen=True)
class HierCadInput:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_start: int
    segment_counts: dict[str, int]

    @property
    def target_ids(self) -> tuple[int, ...]:
        return self.labels[self.target_start:]


def encode_hiercad(tokenizer, text: str, target: HierCadTarget, *, max_length=HIERCAD_MAX_LENGTH) -> HierCadInput:
    """Check author label masking against both token and char prompt boundaries."""
    from hiercad.src.tokenize_hier_consistency import build_messages
    from hiercad.src.tokenize_parameter_grounding import build_masked_labels

    messages = build_messages([text], [target.response])[0]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    target_start_char = len(prefix)
    target_stop_char = target_start_char + len(target.response)
    if not full.startswith(prefix + target.response):
        raise ValueError("HierCAD chat does not preserve the assistant trajectory bytes at its boundary")
    if full[target_stop_char:] != tokenizer.eos_token + "\n":
        raise ValueError("HierCAD chat template changed its assistant end marker or trailer")
    enc = tokenizer(full, truncation=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_length:
        raise ValueError(f"HierCAD input {len(ids)} tokens exceeds {max_length}")
    prefix_ids = tokenizer(prefix, truncation=False)["input_ids"]
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("HierCAD prompt tokens merge across the assistant boundary")
    marker_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    labels = build_masked_labels(ids, enc["attention_mask"], marker_ids)
    start = len(prefix_ids)
    if start < 1 or labels[:start] != [-100] * start or labels[start:] != ids[start:]:
        raise ValueError("HierCAD native label mask disagrees with the generation prompt boundary")
    if labels[start:].count(tokenizer.eos_token_id) != 1:
        raise ValueError("HierCAD target must contain exactly one supervised assistant EOS")
    sections = list(_SECTION_RE.finditer(target.response))
    if not sections or sections[0].start() != 0:
        raise ValueError("HierCAD target has no starting global plan")
    counts = dict.fromkeys(("global_plan", "part_plan", "part_json", "termination"), 0)
    section = 0
    for offset_start, offset_end in enc["offset_mapping"][start:]:
        if offset_start < target_start_char or offset_end <= offset_start:
            raise ValueError("HierCAD supervised token has an invalid char span")
        if offset_start >= target_stop_char:
            counts["termination"] += 1
            continue
        relative = offset_start - target_start_char
        while section + 1 < len(sections) and relative >= sections[section + 1].start():
            section += 1
        counts[sections[section][1].lower()] += 1
    if any(counts[key] == 0 for key in counts):
        raise ValueError("HierCAD target has an empty supervised segment")
    return HierCadInput(tuple(ids), tuple(labels), start, counts)


class MinimalArchive:
    def __init__(self, path: Path):
        self.archive = ZipFile(path)
        self.members = Counter(self.archive.namelist())

    @lru_cache(maxsize=4096)
    def load(self, member: str) -> tuple[HierCadTarget, str]:
        if self.members[member] != 1:
            raise ValueError(f"expected one minimal JSON member at {member}")
        data = self.archive.read(member)
        return hiercad_target(json.loads(data)), digest(data)

    def close(self):
        self.archive.close()
        self.load.cache_clear()


def _tensor_candidate_hashes(candidate: dict, attention: torch.Tensor) -> dict:
    result = {
        name: digest_tensor(candidate[name])
        for name in ("cad_vec", "flag_vec", "index_vec", "key_padding_mask")
    }
    result["attn_mask"] = digest_tensor(attention)
    result["target_sha256"] = digest_tensor(candidate["cad_vec"][1:])
    return result


def check_text2cad_flags(candidate: dict) -> None:
    """Independently recover official training-time flag and index channels."""
    n_real = candidate["n_real_tokens"]
    vec = candidate["cad_vec"]
    labels = vec[:, 0].tolist()
    sketch_stops = [i for i in range(1, n_real - 1) if labels[i] == 2]
    extrusion_stops = [i for i in range(1, n_real - 1) if labels[i] == 6]
    if not sketch_stops or len(sketch_stops) != len(extrusion_stops):
        raise ValueError("Text2CAD sketch/extrusion delimiters do not pair")
    flags = [0] * n_real + [11] * (len(labels) - n_real)
    indices = [0] * n_real + [len(sketch_stops)] * (len(labels) - n_real)
    first = 1
    for block, (sketch_stop, extrusion_stop) in enumerate(zip(sketch_stops, extrusion_stops)):
        if sketch_stop <= first or extrusion_stop != sketch_stop + 11:
            raise ValueError("Text2CAD extrusion does not contain the official 11 rows")
        flags[sketch_stop + 1:extrusion_stop + 1] = [1, *range(1, 11)]
        indices[first:extrusion_stop + 1] = [block] * (extrusion_stop - first + 1)
        first = extrusion_stop + 1
    if first != n_real - 1:
        raise ValueError("Text2CAD last extrusion does not precede the terminal token")
    indices[n_real - 1] = len(sketch_stops) - 1
    if candidate["flag_vec"].tolist() != flags or candidate["index_vec"].tolist() != indices:
        raise ValueError("Text2CAD flags or indices disagree with official training serialization")


def select_readout_targets(rows: list[dict], store: ZipCandidateStore, member_index: dict) -> dict:
    """Three length layers, stable validation order within each extreme layer."""
    eligible = []
    failures = []
    for row in rows:
        member = row.get("cad_member")
        try:
            candidate = store.get(member_index[member])
            count = int((~candidate["key_padding_mask"][1:]).sum())
            eligible.append((row, count))
        except (KeyError, ValueError) as exc:
            failures.append({"uid": row["uid"], "member": member, **error_record(exc)})
    eligible.sort(key=lambda item: item[1])
    cuts = (0, len(eligible) // 3, 2 * len(eligible) // 3, len(eligible))
    original_order = {row["uid"]: i for i, row in enumerate(rows)}
    extremes = []
    bands = {}
    for i, name in enumerate(BANDS):
        items = eligible[cuts[i]:cuts[i + 1]]
        bands[name] = {
            "count": len(items),
            "min_fields": items[0][1] if items else None,
            "max_fields": items[-1][1] if items else None,
        }
        if name != "middle":
            extremes.extend(sorted(items, key=lambda pair: original_order[pair[0]["uid"]])[:2])
    return {
        "rows": [row for row, _ in extremes],
        "selected": [
            {"uid": row["uid"], "member": row["cad_member"], "effective_fields": count}
            for row, count in extremes
        ],
        "length_bands": bands, "expected_targets": len(rows), "failures": failures,
    }



class Text2CadReadout:
    def __init__(self, model, store, member_index, device):
        self.model = model
        self.store = store
        self.member_index = member_index
        self.device = device

    @lru_cache(maxsize=4096)
    def text_tokens(self, text: str) -> tuple[int, ...]:
        embedder = self.model.base_text_embedder
        ids = embedder.tokenizer(text, truncation=False)["input_ids"]
        if len(ids) > embedder.max_seq_len:
            raise ValueError(f"Text2CAD text has {len(ids)} BERT tokens, exceeding {embedder.max_seq_len}")
        return tuple(ids)

    @lru_cache(maxsize=4096)
    def candidate(self, member: str) -> dict:
        candidate = self.store.get(self.member_index[member])
        check_text2cad_flags(candidate)
        return _tensor_candidate_hashes(candidate, self.store.canonical_attn_mask)

    def target_info(self, row: dict) -> dict:
        return self.candidate(row["cad_member"])

    def prepare(self, row: dict, text: str) -> dict:
        hashes = self.target_info(row)
        tokens = self.text_tokens(text)
        return {
            "text": text, "index": self.member_index[row["cad_member"]],
            "input_sha256": digest_json([hashes, tokens]), "query_tokens": len(tokens),
            "target_sha256": hashes["target_sha256"],
        }

    @torch.inference_mode()
    def score(self, pairs: list[dict], *, audit=False) -> tuple[list[dict], dict]:
        vecs, masks, seq_len = assemble_batch(
            self.store, [pair["index"] for pair in pairs], self.device
        )
        outputs, shifted, shifted_mask = teacher_forced_logits(
            self.model, [pair["text"] for pair in pairs], vecs, masks, cad_seq_len=seq_len,
        )
        dtype = str(outputs.dtype)
        logits = outputs.float().cpu()
        cad_vec, kpm = vecs["cad_vec"].cpu(), masks["key_padding_mask"].cpu()
        result = two_head_factorized_score(logits, cad_vec, kpm)
        if audit:
            labels = shifted.cpu().long()
            valid = ~shifted_mask.cpu()
            if not torch.equal(labels, cad_vec[:, 1:].long()) or not torch.equal(valid, labels.ne(0)):
                raise ValueError("Text2CAD native target shift/mask disagrees with independent labels")
            inputs, input_masks, _, _ = build_teacher_forcing_inputs(vecs, masks)
            if (any(not torch.equal(inputs[key], vecs[key][:, :-1]) for key in vecs)
                    or not torch.equal(input_masks["key_padding_mask"], masks["key_padding_mask"][:, :-1])
                    or not torch.equal(input_masks["attn_mask"], masks["attn_mask"])):
                raise ValueError("Text2CAD CAD, flag, index or attention input slicing changed")
            del inputs, input_masks
            # Upstream CELoss uses separate masked head means and 0.1 smoothing.
            smooth_losses = [
                F.cross_entropy(
                    logits[:, :, head].flatten(0, 1), labels[:, :, head].reshape(-1),
                    reduction="none", label_smoothing=0.1,
                ).reshape(labels.shape[:2])
                for head in (0, 1)
            ]
            active = valid.reshape(len(pairs), -1)
            for i, score in enumerate(result):
                count = int(active[i].sum())
                if count != score["n_valid_total"]:
                    raise ValueError("Text2CAD independent valid-field count disagrees with scorer")
                raw_nll = float(F.cross_entropy(
                    logits[i].flatten(0, 1)[active[i]], labels[i].reshape(-1)[active[i]],
                    reduction="mean", label_smoothing=0.0,
                ))
                native_smoothed_heads = [
                    float(smooth_losses[head][i][valid[i, :, head]].mean())
                    if bool(valid[i, :, head].any()) else None
                    for head in (0, 1)
                ]
                score.update(
                    raw_nll=raw_nll,
                    nll_abs_delta=abs(raw_nll + score["score_total"] / count),
                    nll_atol=T2_NLL_ATOL,
                    label_sha256=digest_tensor(labels[i]),
                    label_mask_sha256=digest_tensor(valid[i]),
                    native_smoothed_head_means=native_smoothed_heads,
                    native_smoothed_head_average=(
                        math.fsum(native_smoothed_heads) / 2
                        if all(value is not None for value in native_smoothed_heads) else None
                    ),
                    valid_logits_sha256=digest_tensor(logits[i][valid[i]]),
                    terminal_label=labels[i, score["n_real_tokens"] - 2].tolist(),
                )
        for score, pair in zip(result, pairs):
            score.update(s0=score["score_total"], effective_units=score["n_valid_total"],
                         logit_dtype=dtype, input_sha256=pair["input_sha256"],
                         target_sha256=pair["target_sha256"], query_tokens=pair["query_tokens"])
        return result, {"logit_dtype": dtype}

    @torch.inference_mode()
    def causality(self, pairs: list[dict], raw_path: Path) -> dict:
        """Keep the full native prefix, vary one legal-class numeric suffix field."""
        pair = pairs[-1]
        vecs, masks, seq_len = assemble_batch(self.store, [pair["index"]] * 2, self.device)
        base = vecs["cad_vec"][0]
        choices = (base[:seq_len, 0] >= 11).nonzero(as_tuple=True)[0].tolist()
        choices = [i for i in choices if i > 1]
        if not choices:
            raise ValueError("no eligible numerical suffix field")
        changed_at = choices[len(choices) // 2]
        old_label = int(base[changed_at, 0])
        new_label = old_label + 1 if old_label < 266 else old_label - 1
        vecs["cad_vec"][1, changed_at, 0] = new_label
        inputs, input_masks, target, target_mask = build_teacher_forcing_inputs(vecs, masks)
        equal_prefix = {
            key: bool(torch.equal(tensor[0, :changed_at], tensor[1, :changed_at]))
            for key, tensor in inputs.items()
        }
        equal_prefix["key_padding_mask"] = bool(torch.equal(
            input_masks["key_padding_mask"][0, :changed_at],
            input_masks["key_padding_mask"][1, :changed_at],
        ))
        causal_row = input_masks["attn_mask"][changed_at - 1]
        masks_future = bool(torch.isneginf(causal_row[changed_at:]).all())
        for i in range(2):
            check_text2cad_flags({
                **{name: tensor[i].cpu() for name, tensor in vecs.items()},
                "n_real_tokens": self.store.get(pair["index"])["n_real_tokens"],
            })
        shift_correct = bool(
            target[0, changed_at - 1, 0].item() == old_label
            and target[1, changed_at - 1, 0].item() == new_label
            and not target_mask[:, changed_at - 1, 0].any()
        )
        if not all(equal_prefix.values()) or not masks_future or not shift_correct:
            raise ValueError("Text2CAD suffix probe does not preserve the effective causal prefix")
        logits, _, _ = teacher_forced_logits(
            self.model, [pair["text"]] * 2, vecs, masks, cad_seq_len=seq_len
        )
        next_logits = logits[:, changed_at - 1].float().cpu()
        return {
            **save_distribution_comparison(next_logits, raw_path, T2_CAUSAL_ATOL),
            "first_differing_cad_row": changed_at,
            "predicted_from_decoder_row": changed_at - 1,
            "different_target_label": [old_label, new_label],
            "prefix_channels_equal": equal_prefix,
            "causal_mask_blocks_all_future_columns": masks_future,
            "causal_mask_sha256": digest_tensor(input_masks["attn_mask"]),
            "flag_and_index_reconstructed": True, "shift_verified": shift_correct,
            "candidate_prefix_input_sha256": {
                key: digest_tensor(tensor[0, :changed_at]) for key, tensor in inputs.items()
            },
            "candidate_full_input_sha256": [
                digest_tensor(tensor) for tensor in vecs["cad_vec"]
            ],
            "query_text_sha256": digest(pair["text"]),
            "probe_type": "change one future numeric x label, retain full native program shape",
        }


def save_distribution_comparison(logits: torch.Tensor, path: Path, atol: float) -> dict:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    if not bool(torch.isfinite(log_probs).all()):
        raise ValueError("non-finite next-step distribution")
    delta = (log_probs[0] - log_probs[1]).abs()
    probabilities = log_probs.exp()
    with path.open("xb") as stream:
        torch.save({"logits_fp32": logits, "log_probabilities_fp32": log_probs}, stream)
    return {
        "raw_tensors": path.name, "shape_per_variant": list(log_probs[0].shape),
        "whole_log_distribution_atol": atol,
        "max_log_probability_abs_delta": float(delta.max()),
        "max_total_variation": float((probabilities[0] - probabilities[1]).abs().sum(-1).max() / 2),
        "n_over_tolerance": int((delta > atol).sum()),
        "distribution_sha256": [digest_tensor(row) for row in log_probs],
        "passed": bool((delta <= atol).all()),
    }


def hiercad_base_model(base: Path, device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(base.resolve(strict=True)), local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=True, torch_dtype=dtype,
    )
    if model.config.model_type != "qwen2" or model.config.num_hidden_layers != 48:
        raise ValueError("expected the published Qwen2.5-Coder 14B architecture")
    model.config.use_cache = False
    return model.to(device)


def hiercad_scoring_model(base: Path, adapter: Path, device, dtype: torch.dtype):
    from peft import PeftModel

    adapter = adapter.resolve(strict=True)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if config["base_model_name_or_path"] != HIERCAD_BASE:
        raise ValueError("HierCAD adapter does not name the published Qwen2.5-Coder 14B base")
    model = hiercad_base_model(base, device, dtype)
    return PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()


class HierCadReadout:
    def __init__(self, model, tokenizer, archive: MinimalArchive, device):
        self.model = model
        self.tokenizer = tokenizer
        self.archive = archive
        self.device = device

    def target_info(self, row: dict) -> dict:
        target, source_hash = self.archive.load(row["minimal_member"])
        return {
            "source_sha256": source_hash, "target_sha256": digest(target.response),
            "replay_program_sha256": digest_json(target.replay_program),
            "n_parts": len(target.replay_program["parts"]),
        }

    def prepare(self, row: dict, text: str) -> dict:
        target, _ = self.archive.load(row["minimal_member"])
        encoded = encode_hiercad(self.tokenizer, text, target)
        return {
            "text": text, "encoded": encoded, "input_sha256": digest_json(encoded.input_ids),
            "target_token_sha256": digest_json(encoded.target_ids),
            "target_sha256": digest(target.response), "query_tokens": encoded.target_start,
            "effective_units": len(encoded.target_ids), "segment_counts": encoded.segment_counts,
        }

    def batch(self, encodings: list[HierCadInput]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        width = max(len(example.input_ids) for example in encodings)
        return {
            key: torch.tensor(values, dtype=torch.long, device=self.device)
            for key, values in {
                "input_ids": [
                    list(example.input_ids) + [pad_id] * (width - len(example.input_ids))
                    for example in encodings
                ],
                "attention_mask": [
                    [1] * len(example.input_ids) + [0] * (width - len(example.input_ids))
                    for example in encodings
                ],
                "labels": [
                    list(example.labels) + [-100] * (width - len(example.input_ids))
                    for example in encodings
                ],
            }.items()
        }

    @torch.inference_mode()
    def score(self, pairs: list[dict], *, audit=False) -> tuple[list[dict], dict]:
        encoded = [pair["encoded"] for pair in pairs]
        batch = self.batch(encoded)
        labels = batch.pop("labels")
        outputs = self.model(**batch, **({"labels": labels} if audit else {}), use_cache=False)
        logits = outputs.logits
        result = []
        batch_total = 0.0
        batch_count = 0
        for i, (pair, example) in enumerate(zip(pairs, encoded)):
            start, stop = example.target_start, len(example.input_ids)
            actual = labels[i, start:stop]
            target_logits = logits[i, start - 1:stop - 1]
            gathered_chunks = []
            for j in range(0, len(actual), 128):
                chunk = target_logits[j:j + 128].float()
                ids = actual[j:j + 128]
                gathered_chunks.append(
                    chunk.gather(-1, ids[:, None]).squeeze(-1) - chunk.logsumexp(-1)
                )
            gathered = torch.cat(gathered_chunks).cpu().double()
            if not bool(torch.isfinite(gathered).all()):
                raise ValueError("HierCAD produced non-finite target scores")
            total = float(gathered.sum())
            count = len(gathered)
            batch_total += total
            batch_count += count
            score = {
                key: pair[key] for key in (
                    "input_sha256", "target_sha256", "target_token_sha256", "query_tokens",
                    "effective_units", "segment_counts",
                )
            }
            score.update(s0=total, logit_dtype=str(logits.dtype),
                         target_logprob_sha256=digest_tensor(gathered),
                         total_input_tokens=len(example.input_ids), padded_context_tokens=logits.shape[1])
            if audit:
                score["label_sha256"] = digest_json(example.labels)
                # Feed the native logits and independent labels to a separate CE kernel.
                chunks = [
                    F.cross_entropy(
                        target_logits[j:j + 128].float(), actual[j:j + 128],
                        reduction="none", label_smoothing=0.0,
                    ).cpu()
                    for j in range(0, count, 128)
                ]
                nll = float(torch.cat(chunks).double().mean())
                score.update(raw_nll=nll, nll_abs_delta=abs(nll + total / count), nll_atol=H_NLL_ATOL)
            result.append(score)
        audit_report = {"logit_dtype": str(logits.dtype), "context": logits.shape[1]}
        if audit:
            native_loss = float(outputs.loss)
            audit_report.update(
                native_batch_ce=native_loss if math.isfinite(native_loss) else None,
                gathered_batch_nll=-batch_total / batch_count,
                native_batch_ce_abs_delta=(
                    abs(native_loss + batch_total / batch_count) if math.isfinite(native_loss) else None
                ),
                native_batch_ce_atol=H_NLL_ATOL,
            )
        return result, audit_report

    @torch.inference_mode()
    def causality(self, pairs: list[dict], raw_path: Path) -> dict:
        """Choose the longest shared assistant token prefix among real CAD pairs."""
        matches = []
        for i, first in enumerate(pairs):
            for j, second in enumerate(pairs[i + 1:], i + 1):
                left, right = first["encoded"], second["encoded"]
                if first["text"] != second["text"] or left.target_start != right.target_start:
                    continue
                for pos, (a, b) in enumerate(zip(left.input_ids, right.input_ids)):
                    if a != b:
                        if pos > left.target_start and left.labels[pos] >= 0 and right.labels[pos] >= 0:
                            matches.append((pos, i, j))
                        break
        if not matches:
            raise ValueError("no two native HierCAD trajectories share a nonempty assistant prefix and differ later")
        pos, i, j = max(matches)
        left, right = pairs[i]["encoded"], pairs[j]["encoded"]
        if left.labels[:pos] != right.labels[:pos] or left.input_ids[:pos] != right.input_ids[:pos]:
            raise ValueError("HierCAD effective assistant prefixes differ before the comparison point")
        if (left.segment_counts["termination"] < 1 or right.segment_counts["termination"] < 1
                or left.input_ids[pos] == right.input_ids[pos]):
            raise ValueError("HierCAD suffix probe lacks a supervised suffix contrast")
        # Keep full, distinct suffixes in both forwards.
        batch = self.batch([left, right])
        batch.pop("labels")
        if not bool(batch["attention_mask"][:, :pos + 1].all()):
            raise ValueError("HierCAD suffix probe has unexpected prefix padding")
        prediction = self.model(**batch, use_cache=False).logits[:, pos - 1].float().cpu()
        return {
            **save_distribution_comparison(prediction, raw_path, H_CAUSAL_ATOL),
            "pair_indices": [i, j], "first_differing_assistant_target_index": pos - left.target_start,
            "assistant_target_start": left.target_start,
            "first_differing_full_token_index": pos, "predicted_from_full_token_index": pos - 1,
            "different_target_token_ids": [left.input_ids[pos], right.input_ids[pos]],
            "query_text_sha256": digest(pairs[i]["text"]),
            "matched_query_and_chat_prefix_sha256": digest_json(left.input_ids[:left.target_start]),
            "matched_effective_program_prefix_sha256": digest_json(left.input_ids[left.target_start:pos]),
            "full_input_sha256": [pairs[i]["input_sha256"], pairs[j]["input_sha256"]],
            "attention_prefix_active": True,
            "attention_implementation": self.model.config._attn_implementation,
            "probe_type": "distinct real candidate suffixes under the same chat and assistant prefix",
        }



def synchronize(device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_memory(device) -> None:
    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_peaks(device) -> dict:
    if device.type not in ("npu", "cuda"):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    runtime = getattr(torch, device.type)
    return {
        "peak_allocated_bytes": int(runtime.max_memory_allocated(device)),
        "peak_reserved_bytes": int(runtime.max_memory_reserved(device)),
    }


def score_cases(backend, rows: list[dict], query_rows: list[dict], batch_size: int, *,
                audit: bool, retry_single: bool = False) -> dict:
    """Preserve every planned matrix position even when preparation or forwarding fails."""
    started = time.perf_counter()
    cases = []
    outputs = [None] * (len(query_rows) * len(rows))
    attempted_batches = []
    for i, query in enumerate(query_rows):
        for j, row in enumerate(rows):
            slot = i * len(rows) + j
            ids = {
                "query_i": i, "query_uid": query["uid"], "candidate_i": j,
                "candidate_uid": row["uid"], "query_text_sha256": digest(query["text"]),
            }
            try:
                cases.append((slot, ids, backend.prepare(row, query["text"])))
            except Exception as exc:
                outputs[slot] = {**ids, "stage": "prepare_input", **error_record(exc)}
    preparation_seconds = time.perf_counter() - started

    def call(part):
        start = time.perf_counter()
        batch_slots = [slot for slot, _, _ in part]
        try:
            synchronize(backend.device)
            values, report = backend.score([case for _, _, case in part], audit=audit)
            synchronize(backend.device)
            for (slot, ids, _), value in zip(part, values, strict=True):
                outputs[slot] = {**ids, **value, "status": "scored"}
            attempted_batches.append({
                "slots": batch_slots, "seconds": time.perf_counter() - start,
                "status": "scored", **report,
            })
            return True
        except Exception as exc:
            failure = error_record(exc)
            attempted_batches.append({
                "slots": batch_slots, "seconds": time.perf_counter() - start, **failure,
            })
            for slot, ids, _ in part:
                outputs[slot] = {**ids, "stage": "forward_or_score", **failure}
            return False

    for begin in range(0, len(cases), batch_size):
        batch = cases[begin:begin + batch_size]
        ok = call(batch)
        if retry_single and not ok and len(batch) > 1:
            for item in batch:
                call([item])
    if any(output is None for output in outputs):
        raise RuntimeError("score matrix contains an unaccounted position")
    return {
        "pairs": outputs, "attempted_batches": attempted_batches,
        "preparation_seconds": preparation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "planned_pairs": len(outputs),
        "scored_pairs": sum(row["status"] == "scored" for row in outputs),
    }


def calibration_checks(single: dict, batch: dict, model_name: str) -> dict:
    comparisons = []
    for left, right in zip(single["pairs"], batch["pairs"], strict=True):
        match = {key: left[key] for key in ("query_i", "candidate_i", "query_uid", "candidate_uid")}
        if left["status"] != "scored" or right["status"] != "scored":
            comparisons.append({**match, "status": "missing_score",
                                "single_status": left["status"], "batch_status": right["status"]})
            continue
        fields = left["effective_units"]
        same_labels = all(left.get(key) == right.get(key) for key in (
            "effective_units", "input_sha256", "target_sha256", "target_token_sha256",
            "segment_counts", "label_sha256",
        ))
        delta = abs(left["s0"] - right["s0"])
        tolerance = (
            max(T2_SCORE_ATOL, T2_SCORE_RTOL * abs(left["s0"]))
            if model_name == "text2cad" else H_SCORE_MEAN_ATOL * fields
        )
        nll_passed = left["nll_abs_delta"] <= left["nll_atol"] and right["nll_abs_delta"] <= right["nll_atol"]
        comparisons.append({
            **match, "status": "compared", "same_labels": same_labels,
            "score_abs_delta": delta, "score_abs_delta_per_effective_unit": delta / fields,
            "sum_tolerance": tolerance, "raw_nll_verified": nll_passed,
            "passed": same_labels and nll_passed and delta <= tolerance,
        })
    native_losses = [
        b for r in (single, batch) for b in r["attempted_batches"] if "native_batch_ce" in b
    ]
    native_failures = [
        b for b in native_losses if b["native_batch_ce_abs_delta"] is None
        or b["native_batch_ce_abs_delta"] > b["native_batch_ce_atol"]
    ]
    expected_native = len(single["attempted_batches"]) + len(batch["attempted_batches"])
    return {
        "pair_checks": comparisons,
        "pairs_compared": sum(item["status"] == "compared" for item in comparisons),
        "pairs_passed": sum(item.get("passed", False) for item in comparisons),
        "native_loss_checks": len(native_losses), "native_loss_failures": native_failures,
        "passed": bool(comparisons) and all(item.get("passed") for item in comparisons)
        and not native_failures and (model_name == "text2cad" or len(native_losses) == expected_native),
    }


def background_logmean(rows: list[dict], candidate_count: int, count: int = 48) -> list:
    priors = []
    for i in range(candidate_count):
        column = [row for row in rows if row["candidate_i"] == i]
        if (len(column) != count or len({row["query_text_sha256"] for row in column}) != count
                or any(row["status"] != "scored" for row in column)):
            priors.append(None)
            continue
        scores = [row["s0"] for row in column]
        top = max(scores)
        priors.append(top + math.log(math.fsum(math.exp(score - top) for score in scores)) - math.log(count))
    return priors


def calibrated_scores(single: dict, background: dict, candidates: list[dict], priors: list) -> dict:
    vectors = []
    hashes = [set() for _ in candidates]
    for pair in [*single["pairs"], *background["pairs"]]:
        if pair["status"] != "scored":
            continue
        identity = (pair["target_sha256"], pair.get("target_token_sha256"), pair["effective_units"])
        hashes[pair["candidate_i"]].add(identity)
    for pair in single["pairs"]:
        prior = priors[pair["candidate_i"]]
        vectors.append(
            {**pair, "background_logmeanexp": prior, "s48": pair["s0"] - prior}
            if pair["status"] == "scored" and prior is not None else {**pair, "s48": None}
        )
    invariant = {"status": "unavailable", "reason": "no fixed CAD with two distinct scored texts and complete background"}
    for i in range(len(candidates)):
        column = [pair for pair in vectors if pair["candidate_i"] == i and pair["s48"] is not None]
        if not column:
            continue
        left = column[0]
        right = next((pair for pair in column[1:] if pair["query_text_sha256"] != left["query_text_sha256"]), None)
        if right is None:
            continue
        raw_delta = left["s0"] - right["s0"]
        corrected_delta = left["s48"] - right["s48"]
        tolerance = 1e-9
        invariant = {
            "status": "checked", "candidate_uid": candidates[i]["uid"],
            "query_indices": [left["query_i"], right["query_i"]],
            "delta_s0": raw_delta, "delta_s48": corrected_delta,
            "abs_delta": abs(raw_delta - corrected_delta), "atol": tolerance,
            "passed": abs(raw_delta - corrected_delta) <= tolerance,
        }
        break
    return {
        "pairs": vectors, "fixed_cad_invariance": invariant,
        "candidate_target_invariance": [
            {"candidate_uid": row["uid"], "observed_target_identities": sorted(values),
             "passed": len(values) == 1}
            for row, values in zip(candidates, hashes)
        ],
    }


def candidate_reference(path: Path | None, uids: set[str]) -> dict:
    if path is None:
        return {}
    import pyarrow.parquet as pq

    matches = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=["uid", "response"]):
        for row in batch.to_pylist():
            if row["uid"] in uids:
                if row["uid"] in matches:
                    raise ValueError(f"duplicate HierCAD validation response for {row['uid']}")
                matches[row["uid"]] = row["response"]
    return matches


def compare_hiercad_reference(target: HierCadTarget, reference: str | None) -> dict:
    from hiercad.src.utils.hierarchical_parse import merge_part_jsons, parse_hierarchical_response

    if reference is None:
        return {"status": "unavailable", "reason": "no published validation response available for UID"}
    parsed = parse_hierarchical_response(reference)
    replay = merge_part_jsons(parsed)
    return {
        "status": "compared", "reference_response_sha256": digest(reference),
        "reference_program_sha256": digest_json(replay),
        "response_bytes_equal": target.response == reference,
        "ordered_geometry_bytes_equal": digest_json(replay) == digest_json(target.replay_program),
        "geometry_values_equal": replay == target.replay_program,
        "reference_parse_error": parsed["parse_errors"] or None,
    }


def replay_hiercad(target: HierCadTarget) -> dict:
    try:
        from hiercad.src.utils.cadseqproc_loader import load_cadsequence_class

        sequence = load_cadsequence_class().from_minimal_dict(target.replay_program)
        solid = sequence.create_cad_model().cad_model
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        valid = not solid.IsNull() and BRepCheck_Analyzer(solid).IsValid()
        properties = GProp_GProps()
        brepgprop.VolumeProperties(solid, properties)
        volume = float(properties.Mass())
        return {"status": "replayed", "valid_solid": bool(valid), "volume": volume,
                "passed": bool(valid and math.isfinite(volume) and volume > 0)}
    except Exception as exc:
        return error_record(exc)


def hiercad_timing_bands(backend: HierCadReadout, gallery: list[dict]) -> dict:
    """Build target-token tertiles on the frozen validation gallery."""
    started = time.perf_counter()
    catalog = []
    failures = []
    for index, row in enumerate(gallery):
        try:
            target, _ = backend.archive.load(row["minimal_member"])
            # This fixed query determines only the assistant token length.
            # Each real query's prefix width is included before sampling.
            encoded = encode_hiercad(
                backend.tokenizer, "CAD", target, max_length=backend.tokenizer.model_max_length,
            )
            catalog.append((len(encoded.target_ids), index, row))
        except Exception as exc:
            failures.append({"uid": row["uid"], **error_record(exc)})
    catalog.sort(key=lambda value: (value[0], value[1]))
    cuts = (0, len(catalog) // 3, 2 * len(catalog) // 3, len(catalog))
    layers = {}
    facts = {}
    population = []
    for i, band in enumerate(BANDS):
        layer = catalog[cuts[i]:cuts[i + 1]]
        layers[band] = layer
        facts[band] = {
            "size": len(layer),
            "min_target_tokens": layer[0][0] if layer else None,
            "max_target_tokens": layer[-1][0] if layer else None,
        }
        population.extend(
            {"uid": row["uid"], "target_tokens": length, "band": band}
            for length, _, row in layer
        )
    return {"layers": layers, "bands": facts, "failures": failures, "population": population,
            "expected_candidates": len(gallery), "preparation_seconds": time.perf_counter() - started}


def hiercad_query_widths(backend: HierCadReadout, queries: list[dict], probe_row: dict) -> tuple[list, list]:
    widths = []
    failures = []
    for i, query in enumerate(queries):
        try:
            target, _ = backend.archive.load(probe_row["minimal_member"])
            encoded = encode_hiercad(
                backend.tokenizer, query["text"], target, max_length=backend.tokenizer.model_max_length,
            )
            widths.append({"query_i": i, "uid": query["uid"], "prefix_tokens": encoded.target_start})
        except Exception as exc:
            failures.append({"query_i": i, "uid": query["uid"], **error_record(exc)})
    return widths, failures


def hiercad_timing_sample(layer: list, widths: list, limit: int, band: str) -> tuple[list[dict], dict]:
    """Sample contexts that fit all real query prefixes; retain all coverage."""
    query_widths = [row["prefix_tokens"] for row in widths]
    coverage = {"expected_tokenizable_pairs": len(layer) * len(query_widths)}
    if not query_widths:
        return [], {**coverage, "fits": 0, "overlength_pairs": [], "selected": []}
    eligible = []
    overlength = []
    for units, _, row in layer:
        totals_over_window = [
            {"query_i": query["query_i"], "query_uid": query["uid"],
             "context_tokens": units + query["prefix_tokens"]}
            for query in widths if units + query["prefix_tokens"] > HIERCAD_MAX_LENGTH
        ]
        if totals_over_window:
            overlength.append({"candidate_uid": row["uid"], "pairs": totals_over_window})
        else:
            eligible.append((units, row))
    size = min(limit, len(eligible))
    if size == 1:
        indices = [len(eligible) - 1 if band == "long" else len(eligible) // 2]
    elif size > 1:
        indices = [j * (len(eligible) - 1) // (size - 1) for j in range(size)]
    else:
        indices = []
    selected = [eligible[j] for j in indices]
    return [row for _, row in selected], {
        **coverage,
        "fits": coverage["expected_tokenizable_pairs"] - sum(len(item["pairs"]) for item in overlength),
        "overlength_pairs": overlength,
        "candidates_fitting_every_query": len(eligible),
        "selected": [
            {"uid": row["uid"], "target_tokens": units,
             "min_input_tokens": units + min(query_widths),
             "max_input_tokens": units + max(query_widths)}
            for units, row in selected
        ],
    }


def duration_range(samples, expected_pairs: int) -> dict | None:
    valid = [item for item in samples if item["planned_pairs"] == expected_pairs
             and item["scored_pairs"] == expected_pairs and item["wall_seconds"] > 0]
    if not valid:
        return None
    times = [item["wall_seconds"] for item in valid]
    return {
        "n_complete_steady_runs": len(valid), "min_seconds": min(times), "max_seconds": max(times),
        "pairs_per_second_low": expected_pairs / max(times),
        "pairs_per_second_high": expected_pairs / min(times),
    }


def project_hiercad(timing: list[dict], band_facts: dict, source: str) -> dict:
    by_cell = {(record["condition"], record["band"]): record for record in timing}
    available = sum(info["size"] for info in band_facts.values())
    conditions = CONDITIONS[source]
    if not available:
        return {"status": "unavailable", "reason": "no measured validation strata"}

    def estimate(counts: dict):
        low = high = supported_low = supported_high = 0.0
        expected_outside_context = 0.0
        for name in (*conditions, "background48"):
            for band in BANDS:
                cell = by_cell.get((name, band), {})
                rates = cell.get("throughput")
                if rates is None:
                    return {"status": "unavailable", "missing_condition": name, "missing_band": band}
                pairs = counts[name] * band_facts[band]["size"] / available
                fraction_fitting = cell["gallery_pairs_context_fits"] / cell["gallery_pairs_expected"]
                low += pairs / rates["pairs_per_second_high"]
                high += pairs / rates["pairs_per_second_low"]
                supported_low += pairs * fraction_fitting / rates["pairs_per_second_high"]
                supported_high += pairs * fraction_fitting / rates["pairs_per_second_low"]
                expected_outside_context += pairs * (1.0 - fraction_fitting)
        return {
            "status": "estimated", "pairs_by_condition": counts,
            "wall_seconds": [low, high], "wall_hours": [low / 3600, high / 3600],
            "device_hours_one_measured_device": [low / 3600, high / 3600],
            "context_fitting_wall_seconds": [supported_low, supported_high],
            "projected_pairs_outside_supported_context": expected_outside_context,
        }

    full_count = 61_366_842 if source == "t2cad-v1.0" else 16_349_472
    controlled_condition = "L0" if source == "t2cad-v1.0" else "CT"
    # Two queries against two candidates, eight presentations, 240 pairs per
    # attribute, for two attributes. The measured mid-band rate is a scenario.
    rates = by_cell.get((controlled_condition, "middle"), {}).get("throughput")
    controlled = {"status": "unavailable"}
    if rates:
        pairs = 2 * 240 * 8 * 4
        low, high = pairs / rates["pairs_per_second_high"], pairs / rates["pairs_per_second_low"]
        controlled = {"status": "estimated", "condition_proxy": controlled_condition, "pairs": pairs,
                      "wall_seconds": [low, high], "device_hours_one_measured_device": [low / 3600, high / 3600]}
    return {
        "basis": "Validation candidate target-token tertile fractions and complete steady runs. "
                 "One device with no sharding. Forward scoring and input preparation included. "
                 "Training, startup, archive, replay, geometry-family coverage and failures are extra. "
                 "The all-pairs scenario applies measured rates to the full nominal pair count. "
                 "The context-fitting scenario additionally uses estimated validation coverage.",
        "full_test_gallery": estimate({**dict.fromkeys(conditions, full_count), "background48": 48 * 8046}),
        "common_test_512_by_1024_upper_bound": estimate(
            {**dict.fromkeys(conditions, 512 * 1024), "background48": 48 * 1024},
        ),
        "controlled_recognition_mid_length_scenario": controlled,
    }


def prepare_hiercad_training(backend: HierCadReadout, rows: list[dict], conditions: tuple[str, ...],
                             tokenize_config: dict, n_steps: int) -> dict:
    """Select deterministic pilot-train examples with a genuine author-generated negative."""
    from hiercad.src.tokenize_hier_dataset import tokenize_batch
    from hiercad.src.utils.hierarchical_parse import merge_part_jsons, parse_hierarchical_response

    examples = []
    failures = []
    for row in rows:
        level = conditions[len(examples) % len(conditions)]
        try:
            text = row["texts"][level]
            target, _ = backend.archive.load(row["minimal_member"])
            encoded = encode_hiercad(backend.tokenizer, text, target)
            example = {**target.example, "uid": row["uid"], "prompt": text}
            values = tokenize_batch(
                {key: [val] for key, val in example.items()}, tokenizer=backend.tokenizer,
                max_length=tokenize_config["max_length"], split="train",
                seed=tokenize_config["seed"],
                p_text_numeric_dropout=tokenize_config["p_text_numeric_dropout"],
                numeric_span_drop_ratio=tokenize_config["numeric_span_drop_ratio"],
                max_drop_spans=tokenize_config["max_drop_spans"],
                prompt_mode=tokenize_config["prompt_mode"],
            )
            item = {key: value[0] for key, value in values.items()}
            n_clean = sum(item["attention_mask"])
            if tuple(item["input_ids"][:n_clean]) != encoded.input_ids or tuple(item["labels"][:n_clean]) != encoded.labels:
                raise ValueError("released trainer tokens differ from verified untruncated clean readout tokens")
            n_corrupt = sum(item["corrupted_attention_mask"])
            negative_labels = [label for label in item["corrupted_labels"][:n_corrupt] if label != -100]
            if not (len(negative_labels) and negative_labels.count(backend.tokenizer.eos_token_id) == 1):
                raise ValueError("author-generated perturbed response is truncated or missing assistant EOS")
            decoded = backend.tokenizer.decode(negative_labels, clean_up_tokenization_spaces=False)
            if not decoded.endswith(backend.tokenizer.eos_token + "\n"):
                raise ValueError("author-generated negative tail differs from the released chat template")
            negative_target = decoded[:-(len(backend.tokenizer.eos_token) + 1)]
            before = parse_hierarchical_response(target.response)
            after = parse_hierarchical_response(negative_target)
            if after["parse_errors"] or any(before[key] != after[key] for key in ("global_plan", "part_plans")):
                raise ValueError("author-generated negative does not preserve the hierarchical plan")
            negative_program = merge_part_jsons(after)
            if negative_program == target.replay_program:
                raise ValueError("author-generated negative did not change any program values")
            clean_mask = sum(m and label != -100 for m, label in zip(
                item["param_token_mask"], item["labels"], strict=True
            ))
            corrupt_mask = sum(m and label != -100 for m, label in zip(
                item["corrupted_param_token_mask"], item["corrupted_labels"], strict=True
            ))
            if not clean_mask or not corrupt_mask or not item["corruption_tags"]:
                raise ValueError("author-generated negative has no scoring evidence on parameters")
            examples.append({
                "item": item, "condition": level, "candidate_uid": row["uid"],
                "query_sha256": digest(text), "clean_target_sha256": digest(target.response),
                "perturbed_target_sha256": digest(negative_target),
                "clean_program_sha256": digest_json(target.replay_program),
                "perturbed_program_sha256": digest_json(negative_program),
                "clean_tokens": n_clean, "perturbed_tokens": n_corrupt,
                "clean_parameter_units": clean_mask, "perturbed_parameter_units": corrupt_mask,
                "tags": item["corruption_tags"],
            })
        except Exception as exc:
            failures.append({"uid": row["uid"], "condition": level, **error_record(exc)})
        if len(examples) == n_steps:
            break
    return {"examples": examples, "preparation_failures": failures}


def train_hiercad_probe(base_model, prepared: dict, device, train_config: dict, emit, *, seed=17) -> dict:
    """Measure fresh-adapter optimizer updates using the released SAPG objective."""
    from hiercad.src.sapg_trainer import SAPGDataCollator, SAPGTrainer
    from peft import LoraConfig, get_peft_model

    config = train_config["training"]
    weights = train_config["sapg"]
    lora = LoraConfig(**train_config["model"]["lora"])
    if config["gradient_checkpointing"]:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    stages = []
    for variant in ("ce_only", "ce_structure", "ce_structure_parameter"):
        if not prepared["examples"]:
            stages.append({"variant": variant, "steps": [], "status": "no eligible train example"})
            continue
        torch.manual_seed(seed)
        model = get_peft_model(base_model, lora).train()
        params = [param for param in model.parameters() if param.requires_grad]
        named_probe = next((name, param) for name, param in model.named_parameters()
                           if param.requires_grad and "lora_B" in name)
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
        # This pinned method does not use Trainer's distributed loop or
        # accelerator when given an already placed batch and model. Calling
        # it directly preserves the actual author loss and detached negative.
        objective = object.__new__(SAPGTrainer)
        for name, value in weights.items():
            setattr(objective, name, value)
        step_rows = []
        for i, example in enumerate(prepared["examples"]):
            facts = {key: value for key, value in example.items() if key != "item"}
            start = time.perf_counter()
            try:
                raw = SAPGDataCollator()([example["item"]])
                if variant != "ce_structure_parameter":
                    del raw["corrupted_input_ids"]
                if variant == "ce_only":
                    del raw["alignment_meta_tokenized_json"]
                features = {
                    key: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
                    for key, tensor in raw.items()
                }
                param = named_probe[1]
                before = digest_tensor(param)
                sync_start = time.perf_counter()
                synchronize(device)
                reset_memory(device)
                forward_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, raw_outputs, breakdown = objective._forward_components(model, features)
                del raw_outputs
                synchronize(device)
                backward_start = time.perf_counter()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite training loss")
                loss.backward()
                synchronize(device)
                update_start = time.perf_counter()
                norm = torch.nn.utils.clip_grad_norm_(params, max_norm=config["max_grad_norm"])
                if not bool(torch.isfinite(norm)):
                    raise ValueError("non-finite training gradient")
                optimizer.step()
                synchronize(device)
                finished = time.perf_counter()
                peaks = memory_peaks(device)
                after = digest_tensor(param)
                changed = before != after
                if variant != "ce_only":
                    if not breakdown["valid_part_units"] or not breakdown["valid_loop_units"]:
                        raise ValueError("upstream alignment did not process both part and loop units")
                if variant == "ce_structure_parameter":
                    if not all(math.isfinite(breakdown[key]) for key in (
                        "parameter_ce_loss", "corrupted_parameter_ce_loss", "ranking_loss"
                    )):
                        raise ValueError("upstream clean/negative parameter objectives were non-finite")
                step = {
                    "status": "updated" if changed else "parameter_probe_unchanged",
                    "variant": variant, "step": i + 1, **facts, **peaks,
                    "loss_components": breakdown, "unclipped_grad_norm": float(norm),
                    "forward_seconds": backward_start - forward_start,
                    "backward_seconds": update_start - backward_start,
                    "clip_and_optimizer_seconds": finished - update_start,
                    "train_step_seconds": finished - forward_start,
                    "collate_transfer_and_probe_seconds": sync_start - start,
                    "step_with_preparation_seconds": finished - start,
                    "optimizer_step_called": True, "optimizer_step_observed": int(optimizer.state.get(param, {}).get("step", 0)),
                    "changed_parameter_probe": named_probe[0], "parameter_sha256_before": before,
                    "parameter_sha256_after": after, "parameter_changed": changed,
                }
                del loss, features, raw, norm
            except Exception as exc:
                step = {"variant": variant, "step": i + 1, **facts, **error_record(exc),
                        "seconds_until_failure": time.perf_counter() - start, **memory_peaks(device)}
                optimizer.zero_grad(set_to_none=True)
            emit(step)
            step_rows.append(step)
            if step["status"] == "failed":
                break
        stages.append({
            "variant": variant, "steps": step_rows, "n_planned": len(prepared["examples"]),
            "steps_updated": sum(row["status"] == "updated" for row in step_rows),
            "trainable_parameters": sum(param.numel() for param in params),
        })
        del optimizer, params, named_probe
        base_model = model.unload()
        del model
        if device.type in ("npu", "cuda"):
            getattr(torch, device.type).empty_cache()
    return {
        "stages": stages,
        "training_settings": {"config": config, "sapg": weights, "seed": seed,
                              "training_context": HIERCAD_MAX_LENGTH, "lora": train_config["model"]["lora"]},
        "preparation_failures": prepared["preparation_failures"],
        "scope": (
            "Single-device microbatch of one, freshly initialized LoRA in each arm; optimizer "
            "uses the published release peak learning rate. Timings omit distributed "
            "communication, scheduler, weighted sampling and checkpoint write. The negative "
            "forward is detached as in the released SAPG implementation. This measures "
            "the public source configuration; the paper gives different SAPG weights and a staged schedule."
        ),
    }
