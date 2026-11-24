#!/usr/bin/env python3
"""Prepare an input-only CAD feature store shared by the two DeepCAD tables."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from experiments.scripts.prepare_formal_deepcad_tables import verify_prior
from experiments.src.idea01 import in_domain_pilot as pilot
from experiments.src.idea01.in_domain_pilot_retrievers import prepare_cad_features, verify_spectra_source


def make_axis(prior, receipt):
    sources = {"test_gallery.jsonl": "shared_test_gallery"}
    for source in pilot.SOURCE_LEVELS:
        sources[f"{source}/train.jsonl"] = f"{source}_train"
        sources[f"{source}/validation_gallery.jsonl"] = f"{source}_validation_gallery"

    by_uid, roles = {}, {}
    for filename, role in sources.items():
        path = prior / filename
        if pilot.sha256_file(path) != receipt["file_sha256"][filename]:
            raise ValueError(f"Original frozen CAD axis changed: {filename}")
        members = []
        for row in pilot.read_jsonl(path):
            uid, member = row["uid"], row["cad_member"]
            identity = {"uid": uid, "cad_member": member, "member_sha256": row["member_sha256"]}
            if by_uid.setdefault(uid, identity) != identity:
                raise ValueError(f"Source-specific axes disagree for {uid}")
            members.append(uid)
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate UID in {filename}")
        roles[role] = members
    return sorted(by_uid.values(), key=lambda row: row["uid"]), roles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prior-pilot", "cad-zip", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--spectrabrep-source", type=Path,
                   help="Use the pinned author's descriptor instead of command input tensors")
    a = p.parse_args()
    if a.workers < 1:
        p.error("At least one CPU feature worker is required")

    started = time.monotonic()
    prior, receipt = verify_prior(a.prior_pilot)
    if pilot.sha256_file(a.cad_zip) != pilot.INPUT_HASHES["official_zip"]:
        raise ValueError("Official CAD archive SHA changed")
    axes, roles = make_axis(prior, receipt)
    names = [row["uid"] for row in axes]
    locations = {uid: i for i, uid in enumerate(names)}
    implementation = verify_spectra_source(a.spectrabrep_source) if a.spectrabrep_source else None
    a.out.mkdir()
    pilot.write_jsonl(a.out / "feature_axis.jsonl", iter(axes))
    print(f"Preparing {len(axes)} distinct CAD inputs for {len(roles)} registered axes", flush=True)
    features, coverage = prepare_cad_features(axes, a.cad_zip, workers=a.workers, spectra=implementation,
                                              report_progress=True)
    if len(coverage) != len(names) or [row["uid"] for row in coverage] != names:
        raise ValueError("Feature writer changed UID order or count")
    np.save(a.out / "cad_features.npy", features, allow_pickle=False)
    pilot.write_jsonl(a.out / "coverage.jsonl", iter(coverage))
    role_summary = {}
    for role, uids in roles.items():
        ordered = [locations[uid] for uid in uids]
        pilot.write_jsonl(a.out / f"{role}_indices.jsonl", ({"feature_index": i, "uid": names[i]} for i in ordered))
        role_summary[role] = dict(Counter(coverage[i]["status"] for i in ordered))
    pilot.write_json(a.out / "summary.json", {
        "scope": "CPU-only candidate features; no text embeddings, model scores or query labels",
        "representation": "spectrabrep" if implementation is not None else "command-dual-tower",
        "axis_count": len(axes), "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
        "coverage": dict(Counter(row["status"] for row in coverage)), "role_coverage": role_summary,
        "cad_zip_sha256": pilot.INPUT_HASHES["official_zip"],
        "files_sha256": {path.name: pilot.sha256_file(path) for path in sorted(a.out.iterdir())},
        "seconds_before_summary": time.monotonic() - started,
    })
    print(f"Completed aligned feature coverage for {len(axes)} CAD UIDs", flush=True)


if __name__ == "__main__":
    main()
