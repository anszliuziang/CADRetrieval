"""Program representations and objectives for the in-domain neural pilots."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
import io
import multiprocessing
from pathlib import Path
import pickle
import sys
from types import ModuleType
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch import nn

from .text2cad_vector_scorer import (
    FLAG_SIZE,
    INDEX_SIZE,
    MAX_CAD_SEQUENCE_LENGTH,
    ONE_HOT_SIZE,
    validate_cad_vec,
)

SPECTRA_COMMIT = "5254a331d69f6eab08a66369c184f2effb117623"
SPECTRA_TREE = "fa7cf0a4c13f63bba05374fe5652e8b40fd6c79e"
# Same 27-blob digest and verification procedure as preparation commit 14a3757.
SPECTRA_SOURCE_SHA256 = "fd82c2552373d179cf981cdcda8e3dd8aa81892088ebdd82654935b08187e9a5"
SPECTRA_MODULES = ("config", "geometry", "model", "runtime")
SPECTRA_DESCRIPTOR_DIM = 280
OUTPUT_DIM = 512


def direct_registration(pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Index exact training text strings by their directly registered UIDs."""
    registration = defaultdict(set)
    for text, uid in pairs:
        registration[text].add(uid)
    return dict(registration)


def registered_positive_mask(
    texts: Sequence[str], candidate_uids: Sequence[str], registration: Mapping[str, set[str]]
) -> torch.Tensor:
    """Return the text-by-CAD edge matrix, in the given global batch order."""
    linked = [registration[text] for text in texts]
    return torch.tensor([[uid in uids for uid in candidate_uids] for uids in linked], dtype=torch.bool)


def symmetric_positive_set_loss(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Negative log probability of the directly registered positive set, both ways."""
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    text_to_cad = logits.logsumexp(dim=1) - positive_logits.logsumexp(dim=1)
    cad_to_text = logits.logsumexp(dim=0) - positive_logits.logsumexp(dim=0)
    return 0.5 * (text_to_cad.mean() + cad_to_text.mean())


def known_positive_counts(
    mask: torch.Tensor, texts: Sequence[str], uids: Sequence[str]
) -> dict[str, int]:
    """Count known off-diagonal cells once in the text-by-CAD matrix."""
    off_diagonal = mask.clone()
    off_diagonal.fill_diagonal_(False)
    edges = off_diagonal.nonzero().tolist()
    return {
        "off_diagonal_cells": len(texts) * (len(uids) - 1),
        "known_off_diagonal_positives": len(edges),
        "known_off_diagonal_same_uid": sum(uids[i] == uids[j] for i, j in edges),
        "known_off_diagonal_same_text": sum(texts[i] == texts[j] for i, j in edges),
        "known_off_diagonal_other_registration": sum(
            uids[i] != uids[j] and texts[i] != texts[j] for i, j in edges
        ),
        "text_anchors_with_known_off_diagonal_positive": len({i for i, _ in edges}),
        "cad_anchors_with_known_off_diagonal_positive": len({j for _, j in edges}),
    }


def effective_command_input(payload: Mapping[str, Any]) -> torch.Tensor:
    """Return ordered int32 x/y/flag/index rows, including both STARTs and no padding."""
    cad, flag, index, mask = (
        torch.as_tensor(payload[key], device="cpu")
        for key in ("cad_vec", "flag_vec", "index_vec", "key_padding_mask")
    )
    length = validate_cad_vec(cad, mask, flag, index).n_real_tokens
    return torch.column_stack((cad[:length], flag[:length], index[:length])).to(torch.int32)


class CommandTower(nn.Module):
    """Position-aware residual convolution over the official quantized sequence."""

    def __init__(self, width: int = 128, depth: int = 3):
        super().__init__()
        self.fields = nn.ModuleList(
            nn.Embedding(size, width) for size in (ONE_HOT_SIZE, ONE_HOT_SIZE, FLAG_SIZE, INDEX_SIZE)
        )
        self.position = nn.Embedding(MAX_CAD_SEQUENCE_LENGTH, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(), nn.Dropout(0.1))
            for _ in range(depth)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, OUTPUT_DIM))

    def forward(self, commands: torch.Tensor) -> torch.Tensor:
        valid = commands[:, :, 0] != 0
        position = torch.arange(commands.shape[1], device=commands.device)
        features = self.position(position)
        for index, field in enumerate(self.fields):
            features = features + field(commands[:, :, index].contiguous())
        mask = valid.unsqueeze(1)
        features = features.transpose(1, 2) * mask
        for block in self.blocks:
            features = (features + block(features)) * mask
        return self.output(features.sum(dim=2) / valid.sum(dim=1, keepdim=True).clamp_min(1))


def verify_spectra_source(root: Path) -> dict[str, bytes]:
    """Authenticate the clean author snapshot, then return the modules used here."""
    files = {}
    digest = sha256()
    for path in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Symlink in the SpectraBRep release tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_size > 25_000 or len(files) == 27:
            raise ValueError(f"Unexpected file in the SpectraBRep release tree: {path}")
        name = path.relative_to(root).as_posix()
        files[name] = data = path.read_bytes()
        digest.update(f"{name}\0{sha256(data).hexdigest()}\n".encode("utf-8"))
    if len(files) != 27 or digest.hexdigest() != SPECTRA_SOURCE_SHA256:
        raise ValueError(f"SpectraBRep source differs from {SPECTRA_COMMIT}: {root}")
    return {name: files[f"src/cadret/{name}.py"] for name in SPECTRA_MODULES}


def load_spectra_modules(
    source: Mapping[str, bytes], names: Sequence[str], *, attention: str = "eager"
) -> dict[str, ModuleType]:
    """Load verified release bytes directly, with a local-model API adjustment."""
    if attention not in {"eager", "sdpa"}:
        raise ValueError(f"Unsupported attention implementation: {attention}")
    package_name = "_spectrabrep_indomain_pilot"
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    modules = {}
    for name in names:
        module = ModuleType(f"{package_name}.{name}")
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        data = source[name]
        if name == "model":
            # Transformers 4.53 accepts torch_dtype, as used by the Ascend runtime.
            original = b"dtype=dtype,\n            trust_remote_code=True,"
            if data.count(original) != 1:
                raise ValueError("Unexpected upstream AutoModel call")
            data = data.replace(original, (
                "torch_dtype=dtype,\n"
                "            trust_remote_code=False,\n"
                "            local_files_only=True,\n"
                f"            attn_implementation={attention!r},"
            ).encode("ascii"), 1)
        exec(compile(data, f"SpectraBRep/{SPECTRA_COMMIT}/src/cadret/{name}.py", "exec"), module.__dict__)
        modules[name] = module
    return modules


_archive = None
_members = None
_geometry = None
_solver_errors = None


def _initialize_cad_worker(archive: Path, wanted: set[str], spectra: Mapping[str, bytes] | None) -> None:
    global _archive, _members, _geometry, _solver_errors
    _archive = ZipFile(archive)
    _members = Counter(info.filename for info in _archive.infolist() if info.filename in wanted)
    _geometry = None
    _solver_errors = []
    if spectra is not None:
        _geometry = load_spectra_modules(spectra, ("config", "geometry"))["geometry"]
        eigsh = _geometry.eigsh

        def audit_eigsh(*args, **kwargs):
            try:
                return eigsh(*args, **kwargs)
            except Exception as exc:
                _solver_errors.append(type(exc).__name__)
                raise

        _geometry.eigsh = audit_eigsh


def _program_features(item: tuple[str, str]) -> tuple[dict, np.ndarray | None]:
    uid, member = item
    row = {"uid": uid, "cad_member": member, "status": "ok"}
    _solver_errors.clear()
    if _members[member] != 1:
        return dict(row, status="missing_member" if _members[member] == 0 else "ambiguous_member"), None
    try:
        blob = _archive.read(member)
        row["member_sha256"] = sha256(blob).hexdigest()
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        cad = payload["vec"]["cad_vec"]
        if cad.shape != (MAX_CAD_SEQUENCE_LENGTH, 2):
            raise ValueError("Expected official padded cad_vec[272,2]")
        if _geometry is None:
            commands = effective_command_input({
                **payload["vec"], "key_padding_mask": payload["mask_cad_dict"]["key_padding_mask"],
            }).numpy()
            features = np.zeros((MAX_CAD_SEQUENCE_LENGTH, 4), dtype=np.int32)
            features[:len(commands)] = commands
            row["effective_command_rows"] = len(commands)
        else:
            cad = cad.numpy()
            index = payload["vec"]["index_vec"].numpy()
            if index.shape != (MAX_CAD_SEQUENCE_LENGTH,) or index.dtype.kind not in "iu" or cad.dtype.kind not in "iu":
                raise ValueError("Expected integer cad_vec[272,2] and index_vec[272]")
            features = _geometry.descriptor_vector(_geometry.describe_sequence(cad, index))
            row["solver_fallbacks"] = dict(Counter(_solver_errors))
            if features.shape != (SPECTRA_DESCRIPTOR_DIM,) or features.dtype != np.float32:
                raise ValueError("Expected a float32 SpectraBRep descriptor of width 280")
            if not np.isfinite(features).all():
                return dict(row, status="nonfinite_descriptor"), None
            if not np.any(features):
                return dict(row, status="empty_geometry"), None
        row["feature_sha256"] = sha256(features.tobytes()).hexdigest()
        return row, features
    except (OSError, BadZipFile, ValueError, TypeError, KeyError, EOFError, RuntimeError,
            ArithmeticError, AttributeError, IndexError, pickle.UnpicklingError) as exc:
        return dict(row, status="representation_failure", detail=f"{type(exc).__name__}: {exc}"[:500]), None


def prepare_cad_features(
    rows: Sequence[Mapping[str, Any]], archive: Path, *, workers: int,
    spectra: Mapping[str, bytes] | None, report_progress: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Build features on the immutable input axis, keeping failed entries in place."""
    members = [(row["uid"], row["cad_member"]) for row in rows]
    wanted = {member for _, member in members}
    shape = (len(rows), MAX_CAD_SEQUENCE_LENGTH, 4) if spectra is None else (len(rows), SPECTRA_DESCRIPTOR_DIM)
    features = np.zeros(shape, dtype=np.int32 if spectra is None else np.float32)
    coverage = []
    with ExitStack() as stack:
        if workers == 1:
            _initialize_cad_worker(archive, wanted, spectra)
            stack.callback(_archive.close)
            results = map(_program_features, members)
        else:
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_cad_worker,
                initargs=(archive, wanted, spectra),
            ))
            results = pool.map(_program_features, members, chunksize=4)
        for index, (row, value) in enumerate(results):
            expected_sha256 = rows[index].get("member_sha256")
            if expected_sha256 and "member_sha256" in row and row["member_sha256"] != expected_sha256:
                row = dict(row, status="member_hash_mismatch", expected_member_sha256=expected_sha256)
                value = None
            coverage.append(row)
            if value is not None:
                features[index] = value
            if report_progress and ((index + 1) % 512 == 0 or index + 1 == len(rows)):
                print(f"CAD feature coverage processed {index + 1}/{len(rows)}", flush=True)
    return features, coverage
