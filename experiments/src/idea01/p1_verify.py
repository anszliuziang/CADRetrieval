"""P1 raw-run verification (Codex review 2026-09-25, R2).

Every discrepancy is a counted failure and produces a non-zero exit — the
previous version measured |token_sum - pair_sum| but never failed on it, so a
tampered record passed. Per pair we now check, against BOTH the pair record
and its per-token evidence file:

  - record status ok and all_finite true
  - token file exists, non-empty, length == n_target_tokens
  - target positions are exactly n_prefix_tokens .. n_prefix_tokens+n_target-1
    (contiguous, first == prefix end, last == input length-1)
  - pred_row == position-1 for every token (shift convention)
  - token_id == input_ids[position] for every token (needs full input ids,
    which the post-P0-review scorer records)
  - input_ids: present, length == n_input_tokens == n_prefix + n_target,
    last id == EOS
  - recomputed sum of per-token logprobs == sum_logprob within tolerance
  - mean_logprob == sum/n within tolerance
  - eos_logprob == last token logprob within tolerance
  - the run exactly covers the manifest's pair ids: no duplicates, none
    missing, none extra (a success-only subset is a failure)
"""

from __future__ import annotations

EOS_ID = 151645  # <|im_end|>, pinned checkpoint tokenizer eos
REL_TOL = 1e-6  # relative tolerance for float recomputation checks
EOS_ABS_TOL = 1e-9


def _close(a: float, b: float, rel: float = REL_TOL) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def verify_pair_record(rec: dict, toks: list[dict] | None, eos_id: int = EOS_ID) -> list[str]:
    """Verify one pair record against its per-token evidence. Returns errors
    (empty list = pass). `toks` is None when the token file is missing."""
    errs: list[str] = []
    pid = rec.get("pair_id", "?")
    if rec.get("status") != "ok":
        errs.append(f"{pid}: status {rec.get('status')!r} != 'ok'")
        return errs  # nothing else is meaningful
    if not rec.get("all_finite"):
        errs.append(f"{pid}: all_finite is false")

    ids = rec.get("input_ids")
    if not isinstance(ids, list) or not ids:
        errs.append(f"{pid}: input_ids missing/empty")
        ids = None
    else:
        if len(ids) != rec.get("n_input_tokens"):
            errs.append(f"{pid}: len(input_ids) {len(ids)} != n_input_tokens {rec.get('n_input_tokens')}")
        if ids[-1] != eos_id:
            errs.append(f"{pid}: last input id {ids[-1]} != EOS {eos_id}")
        if rec.get("n_prefix_tokens", -1) + rec.get("n_target_tokens", -1) != len(ids):
            errs.append(
                f"{pid}: n_prefix({rec.get('n_prefix_tokens')}) + n_target({rec.get('n_target_tokens')}) != len(input_ids) {len(ids)}"
            )

    if toks is None:
        errs.append(f"{pid}: missing token evidence file")
        return errs
    if not toks:
        errs.append(f"{pid}: empty token evidence list")
        return errs
    if len(toks) != rec.get("n_target_tokens"):
        errs.append(f"{pid}: token count {len(toks)} != n_target_tokens {rec.get('n_target_tokens')}")

    n_prefix = rec.get("n_prefix_tokens")
    for k, t in enumerate(toks):
        pos_expected = n_prefix + k
        if t.get("position") != pos_expected:
            errs.append(f"{pid}: target[{k}] position {t.get('position')} != expected {pos_expected} (not contiguous)")
            break
        if t.get("pred_row") != pos_expected - 1:
            errs.append(f"{pid}: target[{k}] pred_row {t.get('pred_row')} != position-1 {pos_expected - 1}")
            break
        if ids is not None and pos_expected < len(ids) and t.get("token_id") != ids[pos_expected]:
            errs.append(f"{pid}: target[{k}] token_id {t.get('token_id')} != input_ids[{pos_expected}] {ids[pos_expected]}")
            break

    if toks[-1].get("token_id") != eos_id:
        errs.append(f"{pid}: last target token {toks[-1].get('token_id')} != EOS {eos_id}")

    try:
        recomputed = sum(float(t["logprob"]) for t in toks)
        if not _close(recomputed, float(rec["sum_logprob"])):
            errs.append(f"{pid}: token-sum {recomputed!r} != record sum_logprob {rec['sum_logprob']!r}")
        n = int(rec["n_target_tokens"])
        if n and not _close(float(rec["sum_logprob"]) / n, float(rec["mean_logprob"])):
            errs.append(f"{pid}: mean_logprob {rec['mean_logprob']!r} != sum/n {float(rec['sum_logprob'])/n!r}")
        if rec.get("eos_logprob") is not None and abs(float(toks[-1]["logprob"]) - float(rec["eos_logprob"])) > EOS_ABS_TOL:
            errs.append(f"{pid}: eos_logprob {rec['eos_logprob']!r} != last token logprob {toks[-1]['logprob']!r}")
    except (KeyError, TypeError, ValueError) as e:
        errs.append(f"{pid}: unparseable numeric evidence: {e}")
    return errs


def verify_run(
    records: list[dict],
    tokens_by_pair: dict[str, list[dict] | None],
    expected_pair_ids: list[str],
    eos_id: int = EOS_ID,
) -> dict:
    """Verify a whole run dir against the fixed manifest pair ids.

    records: parsed pairs.jsonl rows; tokens_by_pair: pair_id -> token rows or
    None if the evidence file is missing. Returns {"ok", "failures", ...}."""
    failures: list[str] = []
    ids = [str(r.get("pair_id")) for r in records]
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            failures.append(f"duplicate pair_id {i!r} in pairs.jsonl")
        seen.add(i)
    expected = set(expected_pair_ids)
    missing = [p for p in expected_pair_ids if p not in seen]
    extra = [p for p in ids if p not in expected]
    if missing:
        failures.append(f"pairs missing from raw records: {missing[:5]} ({len(missing)} total)")
    if extra:
        failures.append(f"unexpected pair ids in raw records: {extra[:5]} ({len(extra)} total)")

    by_id: dict[str, dict] = {}
    for r in records:
        by_id.setdefault(str(r.get("pair_id")), r)
    for pid in expected_pair_ids:
        rec = by_id.get(pid)
        if rec is None:
            continue  # already counted as missing
        failures.extend(verify_pair_record(rec, tokens_by_pair.get(pid), eos_id))

    return {
        "ok": not failures,
        "n_records": len(records),
        "n_expected": len(expected_pair_ids),
        "n_failures": len(failures),
        "failures": failures,
        "eos_id": eos_id,
        "rel_tol": REL_TOL,
    }
