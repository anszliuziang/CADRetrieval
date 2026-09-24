"""P1 matrix-runner pure helpers: pair-id parsing, statuses, stop policy.

Kept torch-free and unit-testable; the heavy runner (run_p1_matrix.py) imports
these. Differs from P0 in one deliberate way (plan sec.4): an overlong planned
pair is an ANOMALY that stops the run, because the CPU gate pre-verified every
planned pair at <= max_seq_len — an overlong pair at scoring time means the
tokenizer/environment drifted, not an ordinary coverage marker.
"""

from __future__ import annotations

MAX_PAIRS = 416  # 256 conditional + <=160 empty (plan sec.2: total forwards <= 416 < 512)

EXIT_OK = 0
EXIT_ANOMALY = 2
EXIT_BUDGET = 3

STATUS_OK = "ok"
STATUS_BOUNDARY = "boundary_error"
STATUS_NONFINITE = "nonfinite"
STATUS_ERROR = "error"
STATUS_OVERLONG = "overlong_sequence"

STOP_STATUSES = frozenset({STATUS_BOUNDARY, STATUS_NONFINITE, STATUS_ERROR, STATUS_OVERLONG})

DEFAULT_MAX_WALLCLOCK_S = 2700.0  # 45 min (plan sec.4: 416 forwards / 1 card-hour / 45 min wall, first hit stops)
DEFAULT_MAX_CARD_HOURS = 1.0


def should_stop(status: str) -> bool:
    """True => record the failed pair, then abort the run with EXIT_ANOMALY."""
    return status in STOP_STATUSES


def parse_pair_ids(spec: str, n_pairs: int, max_pairs: int = MAX_PAIRS) -> list[int]:
    """Parse '0', '1-59', 'all', '0,3,5-7'. Deduplicates (order preserved) and
    enforces the task pair cap. Raises ValueError on out-of-range or >max_pairs."""
    if n_pairs > max_pairs:
        raise ValueError(f"pair plan has {n_pairs} pairs, exceeds hard cap {max_pairs}")
    if spec == "all":
        ids = list(range(n_pairs))
    else:
        ids = []
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                ids.extend(range(int(a), int(b) + 1))
            else:
                ids.append(int(part))
    seen: set[int] = set()
    deduped: list[int] = []
    for i in ids:
        if not (0 <= i < n_pairs):
            raise ValueError(f"pair index {i} out of range [0, {n_pairs})")
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    if len(deduped) > max_pairs:
        raise ValueError(f"{len(deduped)} unique pairs requested, exceeds cap {max_pairs}")
    return deduped
