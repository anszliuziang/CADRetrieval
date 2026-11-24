"""Prepare the frozen batch assignments for the November 24 eight-card handoff.

This is a CPU scheduling tool. Its five inputs contain lengths, identities and
previous timing measurements. It needs neither model weights nor relevance labels.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from hashlib import sha256
from itertools import accumulate
import json
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile


STAGES = Path("idea01_four_tables_20261124")
STABLE = STAGES / "stable_verified_20261124"
INPUTS = {
    "effective": (
        STABLE / "idea01_four_tables_axes_cache_20261124/identity_visible_cpu/candidates_effective.jsonl",
        "8a29647863d975a37debc8eb5afee3a94bfe9ddfd8af8336dbac71be9997e9a3",
    ),
    "candidate_inputs": (
        STAGES / "m0_verified/runs/idea01_four_tables_m0_replay_20261124/deepcad/candidate_inputs.jsonl",
        "ce692cc7a6d087f9bd120d208d23e38ae8c5ce64c56915d98ad61e33a6845d96",
    ),
    "lengths": (Path("pair_manifest_v4_worker/cad_length_index.jsonl"),
                "daac66a102e34fe45c9cb474922076b0d62345ab0c9796f917a0f8085c6ea0c3"),
    "members": (Path("pair_manifest_v4_worker/cad_store.jsonl"),
                "2f441110f013bb614b0bd2fdb7a3b49569a231bbf3f8eaa2c03ac1d725cb1380"),
    "bert_cost": (
        STABLE / "idea01_four_tables_ct_seed17_20261124/T_bert_cost/summary.json",
        "05d18584bf715bcd3ac13e1a35c710935de5da38f09d3fe686d4074fd7dfeac7",
    ),
}
CAD_CAPS = (32, 64, 96, 128, 160, 192, 224, 256, 272)


def load_inputs(runtime):
    data = {}
    for name, (relative, expected) in INPUTS.items():
        raw = (runtime / relative).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Input differs from the planning receipt: {relative}")
        data[name] = ([json.loads(line) for line in raw.splitlines()]
                      if relative.suffix == ".jsonl" else json.loads(raw))
    return data


def cad_work(data):
    bins = defaultdict(list)
    representatives = {}
    duplicates = {}
    rows = zip(data["effective"], data["candidate_inputs"], data["lengths"], data["members"], strict=True)
    for i, (effective, feature, length, member) in enumerate(rows):
        if not all(row["cad_index"] == i for row in (effective, feature, length, member)):
            raise ValueError(f"Candidate order diverges at {i}")
        if (feature["cad_member"] != member["member"]
                or feature["n_fields"] != length["n_valid_x"] + length["n_valid_y"]
                or feature["quantized_sha256"] != effective["fields"]["cad_vec"]):
            raise ValueError(f"Candidate identity or field counts diverge at {i}")
        identity = effective["effective_sha256"]
        fields = json.dumps(effective["fields"], sort_keys=True, separators=(",", ":")).encode()
        if sha256(fields).hexdigest() != identity:
            raise ValueError(f"Incomplete effective input identity at {i}")
        raw_width = length["n_real_tokens"]
        if identity in representatives:
            first, prior_width = representatives[identity]
            if raw_width != prior_width:
                raise ValueError(f"Identical input has different real length at {i}")
            duplicates[str(i)] = first
            continue
        representatives[identity] = (i, raw_width)
        width = CAD_CAPS[bisect_left(CAD_CAPS, raw_width)]
        bins[width].append(i)

    groups = [{"cad_rows_including_target": width, "representative_cad_indices": indices}
              for width, indices in sorted(bins.items())]
    count = len(representatives)
    batches = sum(ceil(len(indices) / 16) for indices in bins.values())
    if (len(data["effective"]), count, batches, batches * 16 - count) != (8046, 7016, 441, 40):
        raise ValueError("Frozen candidate work totals diverge")
    return {"batch_size": 16, "gallery_size": len(data["effective"]), "distinct_candidates": count,
            "batches_per_text": batches, "extra_lanes_if_fill_tail": batches * 16 - count,
            "buckets": groups, "duplicate_index_to_representative": duplicates}


def estimate_width(width, measured):
    if width in measured:
        return measured[width]
    left = max(anchor for anchor in measured if anchor < width)
    right = min(anchor for anchor in measured if anchor > width)
    return measured[left] + (width - left) * (measured[right] - measured[left]) / (right - left)


def bert_work(cost):
    counts = sorted((int(k), v) for k, v in cost["batches_per_padded_width"].items())
    measured = {int(k): v for k, v in cost["steady_batch_seconds"].items()}
    widths = [width for width, count in counts for _ in range(count)]
    original_texts = cost["unique_training_texts"]
    if len(widths) != ceil(original_texts / 32) or original_texts != 446461:
        raise ValueError("Frozen command-tower training axis diverges")
    estimates = {width: estimate_width(width, measured) for width, _ in counts}
    seconds = list(accumulate((estimates[w] for w in widths), initial=0.0))
    partitions = {}
    for workers in (1, 2, 4, 8):
        cuts = [0]
        for rank in range(1, workers):
            target = seconds[-1] * rank / workers
            upper = bisect_left(seconds, target)
            cuts.append(min((upper - 1, upper), key=lambda x: abs(seconds[x] - target)))
        cuts.append(len(widths))
        if len(set(cuts)) != len(cuts):
            raise ValueError("A frozen BERT shard is empty")
        partitions[str(workers)] = [
            {"first_batch": first, "stop_batch_exclusive": stop,
             "padded_widths": sorted(set(widths[first:stop])),
             "proxy_worker_hours": round((seconds[stop] - seconds[first]) / 3600, 4)}
            for first, stop in zip(cuts, cuts[1:])
        ]
    return {"source": cost["source"], "split": "train", "train_manifest_sha256": cost["manifest_sha256"]["train.jsonl"],
            "unique_texts": original_texts, "batch_size": 32, "num_batches": len(widths),
            "real_texts_in_last_batch": (original_texts - 1) % 32 + 1,
            "batches_per_padded_width": dict(counts), "measured_seconds_per_batch": measured,
            "scheduling_seconds_per_batch": estimates,
            "timing_model": "Piecewise linear between measured widths; warm BERT computation only",
            "proxy_total_card_hours": round(seconds[-1] / 3600, 4),
            "partitions_by_worker_count": partitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = load_inputs(args.runtime)
    plan = {
        "schema": "cadretrieval-eight-910a-worksets-v1",
        "evidence_sha256": {name: checksum for name, (_, checksum) in INPUTS.items()},
        "text2cad_deepcad_candidate_work": cad_work(data),
        "t_dual_tower_train_bert_work": bert_work(data["bert_cost"]),
    }
    payload = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = NamedTemporaryFile(dir=args.output.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(payload)
        temporary_path.replace(args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"{sha256(payload).hexdigest()}  {args.output.name}")


if __name__ == "__main__":
    main()
