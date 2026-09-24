"""Token coverage / boundary / boilerplate audit.

Tokenizer access is injected as an `encode(text) -> list[int]` callable so the
statistics are testable without transformers. The remote runner builds that
callable from the checkpoint tokenizer with use_fast=False and
add_special_tokens=False.

Measurements per row:
  prefix_ids  = encode(prefix)
  full_ids    = encode(prefix + output)          (EOS appended at id level)
  - prompt / response / full+EOS token lengths
  - full+EOS <= max_len coverage
  - boundary_ok: prefix_ids is a strict token prefix of full_ids
  - eos_in_output: EOS id already present inside tokenized output (boundary
    separability check; expected 0)
  - boilerplate: comment-line ratio, cq.exporters / export presence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def comment_line_ratio(program: str) -> float:
    """Fraction of non-empty lines whose first non-space char is '#'."""
    lines = [ln for ln in program.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln.lstrip().startswith("#")) / len(lines)


def has_export_boilerplate(program: str) -> bool:
    return "cq.exporters" in program or ".export(" in program or "exporters.export" in program


@dataclass
class RowCoverage:
    row_index: int
    n_prefix_tokens: int
    n_response_tokens: int  # includes EOS
    n_full_tokens: int  # prefix + response (includes EOS)
    fits_max_len: bool
    boundary_ok: bool
    eos_in_output: bool
    comment_line_ratio: float
    has_export_boilerplate: bool


def measure_row(
    row_index: int,
    query: str,
    output: str,
    encode: Callable[[str], list[int]],
    eos_id: int,
    max_len: int,
    prefix_builder: Callable[[str], str],
) -> RowCoverage:
    prefix = prefix_builder(query)
    prefix_ids = encode(prefix)
    full_ids = encode(prefix + output)
    n = len(prefix_ids)
    boundary_ok = n < len(full_ids) and full_ids[:n] == prefix_ids
    # Response ids as sliced from the full encoding when the boundary holds;
    # otherwise re-encode output alone so lengths are still reported (row is
    # marked and excluded from scoring either way).
    if boundary_ok:
        resp_ids = full_ids[n:]
    else:
        resp_ids = encode(output)
    eos_in_output = eos_id in resp_ids
    n_full = len(full_ids) + 1  # + EOS appended at id level
    return RowCoverage(
        row_index=row_index,
        n_prefix_tokens=len(prefix_ids),
        n_response_tokens=len(resp_ids) + 1,
        n_full_tokens=n_full,
        fits_max_len=n_full <= max_len,
        boundary_ok=boundary_ok,
        eos_in_output=eos_in_output,
        comment_line_ratio=comment_line_ratio(output),
        has_export_boilerplate=has_export_boilerplate(output),
    )


def summarize(rows: list[RowCoverage], max_len: int) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_rows": 0}
    full_lens = sorted(r.n_full_tokens for r in rows)

    def pct(p: float) -> float:
        k = min(n - 1, max(0, int(round(p * (n - 1)))))
        return full_lens[k]

    return {
        "n_rows": n,
        "max_len": max_len,
        "n_fit_max_len": sum(r.fits_max_len for r in rows),
        "coverage_max_len": sum(r.fits_max_len for r in rows) / n,
        "n_boundary_mismatch": sum(not r.boundary_ok for r in rows),
        "n_eos_in_output": sum(r.eos_in_output for r in rows),
        "full_len_min": full_lens[0],
        "full_len_p50": pct(0.50),
        "full_len_p90": pct(0.90),
        "full_len_max": full_lens[-1],
        "mean_prefix_tokens": sum(r.n_prefix_tokens for r in rows) / n,
        "mean_response_tokens": sum(r.n_response_tokens for r in rows) / n,
        "frac_has_export_boilerplate": sum(r.has_export_boilerplate for r in rows) / n,
        "mean_comment_line_ratio": sum(r.comment_line_ratio for r in rows) / n,
    }
