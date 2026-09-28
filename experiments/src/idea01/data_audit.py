"""Source mapping and leakage audit: JSONL <-> text2cad_v1.1.csv <-> CadQuery.zip.

Join rule (per plan): JSONL `input` == CSV `intermediate` (verbatim string) to
recover the source `uid`; ZIP member `CQ/{uid}.py` is the reference program to
compare against JSONL `output`.

The audit layer is the ONLY place allowed to see uids/CSV captions; the scorer
never receives them (input isolation contract).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_csv_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_field_limit)
        break
    except OverflowError:
        # Windows passes a 32-bit C long to csv even with 64-bit Python.
        _csv_field_limit //= 10

CSV_INTERMEDIATE_COL = "intermediate"
CSV_UID_COL = "uid"


def text_digest(text: str) -> str:
    """sha256 hexdigest of the utf-8 text; digest equality implies byte-identical
    text up to collision resistance (any hit is re-verified byte-exact by callers)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_intermediate_index(csv_path: str | Path) -> tuple[dict[str, list[str]], int]:
    """Map CSV `intermediate` text -> list of uids (verbatim join key).

    Returns (index, n_csv_rows). Memory: ~176k rows of text; acceptable.
    """
    index: dict[str, list[str]] = {}
    n = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if CSV_INTERMEDIATE_COL not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing column {CSV_INTERMEDIATE_COL!r}; header={reader.fieldnames}")
        if CSV_UID_COL not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing column {CSV_UID_COL!r}; header={reader.fieldnames}")
        for row in reader:
            n += 1
            index.setdefault(row[CSV_INTERMEDIATE_COL], []).append(row[CSV_UID_COL])
    return index, n


@dataclass
class SplitAudit:
    split: str
    n_rows: int
    n_unique_map: int = 0
    n_ambiguous_map: int = 0
    n_missing_map: int = 0
    # uid -> "identical" | "different" | "missing_in_zip" | None (unmapped rows)
    row_uids: list[str | None] = field(default_factory=list)
    program_status: Counter = field(default_factory=Counter)
    program_different_uids: list[str] = field(default_factory=list)
    exact_duplicate_inputs: int = 0  # rows beyond first occurrence
    exact_duplicate_outputs: int = 0
    unique_uids: int = 0
    # sha256 digest sets of raw text, for cross-split exact-duplicate checks
    input_hashes: set[str] = field(default_factory=set)
    output_hashes: set[str] = field(default_factory=set)


def audit_split(
    split: str,
    rows: list[dict],
    index: dict[str, list[str]],
    zf: zipfile.ZipFile | None,
) -> SplitAudit:
    """Audit one split against the CSV index and (optionally) the program ZIP."""
    out = SplitAudit(split=split, n_rows=len(rows))
    in_ctr = Counter(r["input"] for r in rows)
    out_ctr = Counter(r["output"] for r in rows)
    out.exact_duplicate_inputs = sum(c - 1 for c in in_ctr.values() if c > 1)
    out.exact_duplicate_outputs = sum(c - 1 for c in out_ctr.values() if c > 1)
    out.input_hashes = {text_digest(t) for t in in_ctr}
    out.output_hashes = {text_digest(t) for t in out_ctr}

    names: set[str] | None = None
    if zf is not None:
        names = set(zf.namelist())

    for r in rows:
        hits = index.get(r["input"], [])
        if len(hits) == 1:
            out.n_unique_map += 1
            uid = hits[0]
        elif len(hits) > 1:
            out.n_ambiguous_map += 1
            uid = None
        else:
            out.n_missing_map += 1
            uid = None
        out.row_uids.append(uid)

        if zf is None:
            continue
        if uid is None:
            out.program_status["no_uid"] += 1
            continue
        member = f"CQ/{uid}.py"
        if names is not None and member not in names:
            out.program_status["missing_in_zip"] += 1
            continue
        ref = zf.read(member).decode("utf-8")
        if ref == r["output"]:
            out.program_status["identical"] += 1
        else:
            out.program_status["different"] += 1
            out.program_different_uids.append(uid)

    out.unique_uids = len({u for u in out.row_uids if u is not None})
    return out


def load_text_digest_sets(jsonl_path: str | Path) -> tuple[set[str], set[str]]:
    """Streaming digest sets of `input` / `output` text for one JSONL split."""
    inputs: set[str] = set()
    outputs: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            inputs.add(text_digest(r["input"]))
            outputs.add(text_digest(r["output"]))
    return inputs, outputs


def cross_split_overlap(audits: dict[str, SplitAudit]) -> dict:
    """UID / exact-input / exact-output intersections between split pairs.

    Input/output intersections use sha256 digest sets (see text_digest); a zero
    count means no byte-identical text is shared (up to collision resistance).
    """
    res = {}
    splits = sorted(audits)
    for i, sa in enumerate(splits):
        for sb in splits[i + 1 :]:
            ua = {u for u in audits[sa].row_uids if u}
            ub = {u for u in audits[sb].row_uids if u}
            res[f"{sa}__{sb}"] = {
                "uid_intersection": len(ua & ub),
                "input_intersection": len(audits[sa].input_hashes & audits[sb].input_hashes),
                "output_intersection": len(audits[sa].output_hashes & audits[sb].output_hashes),
            }
    return res


def audit_to_dict(a: SplitAudit) -> dict:
    return {
        "split": a.split,
        "n_rows": a.n_rows,
        "n_unique_map": a.n_unique_map,
        "n_ambiguous_map": a.n_ambiguous_map,
        "n_missing_map": a.n_missing_map,
        "unique_uids": a.unique_uids,
        "exact_duplicate_inputs": a.exact_duplicate_inputs,
        "exact_duplicate_outputs": a.exact_duplicate_outputs,
        "program_status": dict(a.program_status),
        "program_different_uids": list(a.program_different_uids),
    }
