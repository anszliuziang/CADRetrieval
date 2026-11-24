#!/usr/bin/env python3
"""Rebuild the formal command tower's validation ranks and full exposure ledger."""
from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

from experiments.scripts.audit_in_domain_pilot_scores import (
    fingerprint, read, require, rows, verify_ranking, verify_training,
)
from experiments.scripts.prepare_formal_deepcad_tables import RECEIPT_SHA
from experiments.src.idea01.in_domain_pilot import COHORTS, SOURCE_LEVELS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest-root", "exit-record", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--source", choices=tuple(SOURCE_LEVELS), required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--seed", choices=(17, 29, 43), type=int, default=17)
    a = p.parse_args()
    require(a.exit_record.read_text().strip() == "0", "The actual training process did not exit 0")
    success = read(a.run / "run.json")
    config = read(a.run / "config.json")
    chosen = read(a.run / "selection.json")
    checkpoint = a.run / "retrieval.pt"
    require(success["status"] == "completed" and config["git_sha"] == a.expected_source_sha,
            "Unexpected source or process status")
    for record in (success, config, chosen):
        require(record["source"] == a.source and record["method"] == "dual-tower" and record["seed"] == a.seed
                and record["run_scope"] == "formal_full_train_one_epoch", "Wrong frozen recipe")
        for key in ("test_queries_opened", "test_qrels_opened", "test_scoring_executed"):
            require(record[key] is False, "Formal validation artefact contains test use")
    require(config["backbone_precision"] == "fp32" and config["npu_precision_mode"] == "must_keep_origin_dtype"
            and config["candidate_epochs"] == chosen["candidate_epochs"] == [1] == [chosen["selected_epoch"]]
            and config["world_size"] == 1 and config["global_pair_batch"] == 32,
            "Training precision, batch or checkpoint selection differs from the registered recipe")
    require(chosen["selected_checkpoint"] == "retrieval.pt" and fingerprint(checkpoint) == chosen["checkpoint_sha256"],
            "Selected checkpoint did not bind the same trained bytes")
    expected_files = set(success["files"])
    require({f.relative_to(a.run).as_posix() for f in a.run.rglob("*") if f.is_file()} == expected_files | {"run.json"},
            "Training artefact coverage differs from the receipt")
    require(all(fingerprint(a.run / f) == success["files"][f] for f in expected_files), "Training artefact SHA differs")

    receipt = a.manifest_root / "manifest_receipt.json"
    require(fingerprint(receipt) == RECEIPT_SHA == config["manifest_receipt_sha256"], "Unregistered input receipt")
    pins = read(receipt)["file_sha256"]
    names = ("train.jsonl", "validation_targets.jsonl", "validation_gallery.jsonl")
    for f in names:
        require(fingerprint(a.manifest_root / a.source / f) == pins[f"{a.source}/{f}"] == config["manifest_sha256"][f],
                f"Frozen source manifest changed: {f}")
    train, targets, gallery = (rows(a.manifest_root / a.source / f) for f in names)
    train_uids, target_uids, gallery_uids = ({r["uid"] for r in d} for d in (train, targets, gallery))
    n = COHORTS[a.source]["train"][0]
    require(len(train) == len(train_uids) == n and len(targets) == len(target_uids) == 512
            and len(gallery) == len(gallery_uids) == 2048 and target_uids <= gallery_uids and not train_uids & gallery_uids,
            "Full-source training and validation identity check failed")
    levels = SOURCE_LEVELS[a.source]
    queries = [{"uid": r["uid"], "level": level, "text": r["texts"][level]} for level in levels for r in targets]
    counts = verify_training(a.run, train, levels, "dual-tower", expected_uid_count=n)
    require(counts["updates"] == ceil(counts["pairs"] / 32), "Wrong full epoch optimizer update count")
    validate = verify_ranking(a.run, queries, gallery, False)
    macro = sum(validate["metrics"][level]["R@20"] for level in levels) / len(levels)
    require(macro == chosen["validation_R@20_equal_level_macro"], "Selection changed the validation ranking rule")
    result = {"scope": "formal full train and original validation panel, no test scoring",
              "source": a.source, "seed": a.seed, "source_sha": a.expected_source_sha,
              "training": counts, "validation": validate, "checkpoint_sha256": chosen["checkpoint_sha256"],
              "runner_summary_sha256": fingerprint(a.run / "run.json"), "manifest_receipt_sha256": RECEIPT_SHA,
              "sha256_verified_files": len(expected_files), "all_passed": True}
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
