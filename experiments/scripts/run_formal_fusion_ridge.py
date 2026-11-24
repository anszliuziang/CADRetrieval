"""Prepare common Fusion training eligibility, then fit ridge on the frozen validation panel."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic

from convert_formal_fusion360 import (
    ARCHIVE_SHA256, HIERCAD_REVISION, MODELS, ROUTES, TEXT2CAD_REVISION, count_features, select_routes,
)
from prepare_formal_fusion360 import (
    CADMIUM_REVISION, EXPECTED_SPLITS, PARQUET_SHA256, SOURCE, SPLIT_SHA256, hash_order, json_bytes,
    read_jsonl, require_sha, sha256_bytes, sha256_file, write_jsonl,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PANEL_SIZE = 512
FEATURE_WIDTH = 29
PREPARED_MANIFEST = "eligibility_manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_manifest(root, basename, sha):
    path = root / basename
    require_sha(path, sha)
    return json.loads(path.read_bytes())


def pinned_file(root, manifest, basename):
    path = root / basename
    require_sha(path, manifest["file_sha256"][basename])
    return path


def indexed(rows, fields):
    records = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in records, f"Duplicate axis key {key}")
        records[key] = row
    return records


def text_observation(row):
    text = row["text"]
    require(text is None or isinstance(text, str), "Expected a released annotation or a missing value")
    digest = sha256_bytes(text.encode("utf-8")) if text is not None else None
    status = "ok" if text and text.strip() else "missing"
    require((row["text_sha256"], row["status"]) == (digest, status), "Annotation bytes or availability drift")
    return digest, status


def catalog_axes(root, sha):
    receipt = pinned_manifest(root, "manifest.json", sha)
    identity = (receipt["source"], receipt["CADmium_revision"], receipt["published_parquet_sha256"],
                receipt["autodesk_split_sha256"], receipt["frozen_counts"])
    require(identity == (SOURCE, CADMIUM_REVISION, PARQUET_SHA256, SPLIT_SHA256, EXPECTED_SPLITS),
            "Fusion catalogue provenance or registered denominators changed")
    inventory = list(read_jsonl(pinned_file(root, receipt, "source_inventory.jsonl")))
    uids = [row["uid"] for row in inventory]
    require(uids == sorted(set(uids)) and Counter(row["split"] for row in inventory) == EXPECTED_SPLITS,
            "Fusion inventory lost its frozen UID and split coverage")
    val_uids = [row["uid"] for row in inventory if row["split"] == "validation"]
    panel = hash_order(val_uids, f"cadretrieval-val-q-v1|{SOURCE}|")[:PANEL_SIZE]
    by_uid = {row["uid"]: row for row in inventory}

    training = list(read_jsonl(pinned_file(root, receipt, "train_pairs.jsonl")))
    require([row["uid"] for row in training] == [row["uid"] for row in inventory if row["split"] == "train"],
            "Training pairs do not match the original 6011-UID axis")
    for row in training:
        original = by_uid[row["uid"]]
        require(text_observation(row) == (original["text_sha256"], original["annotation_status"]),
                f"Training annotation differs at {row['uid']}")

    queries = list(read_jsonl(pinned_file(root, receipt, "validation_queries.jsonl")))
    gallery = list(read_jsonl(pinned_file(root, receipt, "validation_gallery.jsonl")))
    require(gallery == [{"cad_index": index, "uid": uid} for index, uid in enumerate(val_uids)],
            "Validation gallery differs from the original 635-UID axis")
    require(len(queries) == receipt["validation_queries"] == PANEL_SIZE
            and receipt["validation_gallery"] == len(gallery), "Validation panel dimensions changed")
    for index, (query, uid) in enumerate(zip(queries, panel, strict=True)):
        original = by_uid[uid]
        require(query["query_id"] == f"fusion-validation-{index:04d}" and query["query_index"] == index
                and query["level"] == "annotation" and "uid" not in query
                and text_observation(query) == (original["text_sha256"], original["annotation_status"]),
                f"Frozen validation query differs at row {index}")
    return receipt, inventory, training, queries, gallery


def conversion_inputs(root, sha, catalog_sha, inventory):
    from idea01.in_domain_pilot import FEATURE_NAMES

    receipt = pinned_manifest(root, "conversion_manifest.json", sha)
    identity = (receipt["scope"], receipt["source"], receipt["catalog_manifest_sha256"],
                receipt["autodesk_archive_sha256"], receipt["text2cad_revision"], receipt["hiercad_revision"])
    require(identity == ("full", SOURCE, catalog_sha, ARCHIVE_SHA256, TEXT2CAD_REVISION, HIERCAD_REVISION),
            "Expected the complete approved conversion of this catalogue")
    for split, count in EXPECTED_SPLITS.items():
        facts = receipt["statistics"][split]
        require(facts["attempted"] == facts["frozen_denominator"] == count, f"Incomplete {split} conversion receipt")

    def rows(basename):
        return read_jsonl(pinned_file(root, receipt, basename))

    coverage = list(rows("geometry_coverage.jsonl"))
    require([(row["uid"], row["split"]) for row in coverage] == [(row["uid"], row["split"]) for row in inventory],
            "Conversion coverage changed the original UID or split axes")
    axes = list(rows("candidate_inputs.jsonl"))
    require([row["uid"] for row in axes] == [row["uid"] for row in inventory], "Conversion candidate axis differs")
    by_uid = {row["uid"]: row for row in coverage if row["split"] in {"train", "validation"}}
    axis = {row["uid"]: row for row in axes if row["uid"] in by_uid}
    representations = indexed((row for row in rows("candidate_representations.jsonl") if row["uid"] in by_uid),
                              ("uid", "route", "model"))
    count_sets = ("command_count_features.jsonl", "command_count_alternative_features.jsonl")
    features = indexed((row for name in count_sets for row in rows(name) if row["uid"] in by_uid), ("uid", "route"))
    names = json.loads(pinned_file(root, receipt, "command_count_feature_names.json").read_bytes())
    require(names == list(FEATURE_NAMES) and len(names) == FEATURE_WIDTH, "Unordered feature schema differs")
    targets = indexed(({**row, "file": name}
                       for name in ("hiercad_targets.jsonl", "hiercad_alternative_targets.jsonl")
                       for row in rows(name) if row["uid"] in by_uid and by_uid[row["uid"]]["split"] == "train"),
                      ("uid", "route"))
    geometric = {(row["uid"],): row for row in by_uid.values()
                 if row["split"] == "train" and row["joint_aligned_route"] and row["text_present"]}
    reported = indexed(rows("geometry_verified_train_intersection.jsonl"), ("uid",))
    require({key: row["joint_aligned_route"] for key, row in geometric.items()}
            == {key: row["route"] for key, row in reported.items()}, "Joint training geometry receipt drift")
    archive = pinned_file(root, receipt, "text2cad_candidates.zip")
    return by_uid, axis, representations, features, targets, archive


def count_vector(row):
    values = row["features"]
    require(len(values) == FEATURE_WIDTH and all(type(value) is int and value >= 0 for value in values),
            f"Invalid unordered command histogram for {row['uid']}")
    return values


def build_axes(training, gallery, conversion):
    coverage, axes, representations, features, targets, _ = conversion
    initial, stages = [], []
    for annotation in training:
        uid = annotation["uid"]
        row = coverage[uid]
        aligned, chosen, joint = select_routes(row["source_routes"])
        require((aligned, chosen, joint) == (row["selected_routes"], row["selected_input_routes"],
                                            row["joint_aligned_route"])
                and row["text_present"] == (annotation["status"] == "ok"), f"Invalid training route evidence for {uid}")
        reason = "missing_annotation" if not row["text_present"] else "no_joint_aligned_route" if not joint else None
        stage = {"uid": uid, "joint_route": joint, "reason": reason}
        stages.append(stage)
        if reason:
            continue

        models = row["source_routes"][joint]["models"]
        refs = {name: representations[(uid, joint, name)] for name in MODELS}
        require(all(refs[name]["geometry_verified"] and refs[name]["geometry_status"] == "aligned"
                    and models[name]["input_status"] == "ok" and models[name]["geometry"]["comparison"]["pass"]
                    for name in MODELS), f"Joint route lacks independently aligned model representations for {uid}")
        feature = features[(uid, joint)]
        require(feature["geometry_verified"], f"Joint route lacks its command histogram for {uid}")
        target = targets[(uid, joint)]
        target_sha = sha256_bytes(target["response"].encode("utf-8"))
        require(target["file"] == refs["hiercad"]["file"] and target["geometry_verified"]
                and target_sha == models["hiercad"]["response_sha256"]
                and sha256_bytes(json_bytes(target["program"])) == models["hiercad"]["replay_program_sha256"],
                f"Joint HierCAD target drift for {uid}")
        initial.append({"uid": uid, "route": joint, "cad_member": refs["text2cad"]["member"],
                        "text2cad_model_input_sha256": models["text2cad"]["model_input_sha256"],
                        "hiercad_target_file": target["file"], "hiercad_response_sha256": target_sha,
                        "features": count_vector(feature)})

    validation = []
    for entry in gallery:
        uid = entry["uid"]
        axis = axes[uid]
        chosen = axis["text2cad_route"]
        item = None
        if chosen is not None:
            require(chosen == coverage[uid]["selected_input_routes"]["text2cad"],
                    f"Validation selection differs for {uid}")
            item = features[(uid, chosen)]
            ref = representations[(uid, chosen, "text2cad")]
            require(ref["selected"] and item["selected"] and ref["member"] == axis["text2cad_member"]
                    and item["geometry_verified"] == axis["text2cad_geometry_verified"],
                    f"Validation feature provenance drift for {uid}")
        else:
            require(axis["text2cad_member"] is None, f"Missing route for validation member {uid}")
        validation.append({**entry, "route": chosen, "feature_status": "ok" if item is not None else "missing_cad_input",
                           "geometry_verified": axis["text2cad_geometry_verified"],
                           "geometry_status": axis["text2cad_geometry_status"], "cad_member": axis["text2cad_member"],
                           "features": count_vector(item) if item is not None else None})
    return initial, stages, validation


def confirmed_training(initial, stages, archive, spectra_source, workers):
    import numpy as np

    from idea01.in_domain_pilot_retrievers import prepare_cad_features, verify_spectra_source

    source = verify_spectra_source(spectra_source.resolve(strict=True))
    commands, command_coverage = prepare_cad_features(initial, archive, workers=workers, spectra=None, report_progress=True)
    require([row["uid"] for row in command_coverage] == [row["uid"] for row in initial], "Command input axis shifted")
    stage_by_uid = {row["uid"]: row for row in stages}
    eligible_commands = []
    for index, (row, command) in enumerate(zip(initial, command_coverage, strict=True)):
        state = stage_by_uid[row["uid"]]
        state["command_status"] = command["status"]
        if command["status"] != "ok":
            state["reason"] = "command_input_failure"
            continue
        require(count_features({"cad_vec": commands[index, :, :2]}) == row["features"],
                f"Live joint-route histogram disagrees with converter receipt for {row['uid']}")
        eligible_commands.append({**row, "member_sha256": command["member_sha256"],
                                  "effective_command_sha256": command["feature_sha256"]})
    descriptors, descriptor_coverage = prepare_cad_features(
        eligible_commands, archive, workers=workers, spectra=source, report_progress=True)
    require([row["uid"] for row in descriptor_coverage] == [row["uid"] for row in eligible_commands],
            "SpectraBRep descriptor axis shifted")
    accepted, successful = [], []
    for index, (row, descriptor) in enumerate(zip(eligible_commands, descriptor_coverage, strict=True)):
        state = stage_by_uid[row["uid"]]
        state["spectrabrep_status"] = descriptor["status"]
        state["spectrabrep_solver_fallbacks"] = descriptor.get("solver_fallbacks", {})
        if descriptor["status"] != "ok":
            state["reason"] = "spectrabrep_descriptor_failure"
            continue
        accepted.append({**row, "spectrabrep_feature_sha256": descriptor["feature_sha256"],
                         "feature_index": len(accepted)})
        successful.append(index)
    return accepted, descriptors[np.asarray(successful, dtype=np.intp)]


def prepare(args, out):
    import numpy as np

    from idea01.in_domain_pilot_retrievers import SPECTRA_COMMIT, SPECTRA_SOURCE_SHA256

    catalog, inventory, training, queries, gallery = catalog_axes(args.catalog, args.catalog_sha256)
    conversion = conversion_inputs(args.conversion, args.conversion_sha256, args.catalog_sha256, inventory)
    initial, stages, validation = build_axes(training, gallery, conversion)
    require(bool(initial), "No geometry-qualified common training candidates")
    print(f"Auditing {len(initial)}/{len(training)} joint-route training inputs", flush=True)
    accepted, spectra = confirmed_training(initial, stages, conversion[-1], args.spectrabrep_source, args.workers)
    require(bool(accepted), "No common training candidates with actual command and SpectraBRep inputs")
    train_by_uid = {row["uid"]: row for row in training}
    members = [{key: value for key, value in row.items() if key != "features"} for row in accepted]
    training_features = np.asarray([row["features"] for row in accepted], dtype=np.float64)
    validation_features = np.full((len(validation), FEATURE_WIDTH), np.nan, dtype=np.float64)
    for index, row in enumerate(validation):
        if row["feature_status"] == "ok":
            validation_features[index] = row["features"]
    np.save(out / "ridge_train_features.npy", training_features, allow_pickle=False)
    np.save(out / "ridge_validation_features.npy", validation_features, allow_pickle=False)
    np.save(out / "spectrabrep_train_features.npy", spectra, allow_pickle=False)
    write_jsonl(out / "shared_training_candidate.jsonl", members)
    write_jsonl(out / "training_eligibility.jsonl", stages)
    write_jsonl(out / "train_pairs.jsonl", (train_by_uid[row["uid"]] for row in members))
    write_jsonl(out / "validation_queries.jsonl", queries)
    write_jsonl(out / "validation_gallery.jsonl", (
        {key: value for key, value in row.items() if key != "features"} for row in validation))
    result = {
        "source": SOURCE, "status": "training_eligibility_candidate", "scope": "preparation_without_training",
        "original_denominators": {**EXPECTED_SPLITS, "validation_queries": PANEL_SIZE},
        "catalog_manifest_sha256": args.catalog_sha256, "conversion_manifest_sha256": args.conversion_sha256,
        "validation_qrels_sha256": catalog["file_sha256"]["validation_qrels.jsonl"],
        "training_geometry_candidates": len(initial), "training_eligible": len(members),
        "training_eligibility": dict(Counter(row["reason"] or "eligible" for row in stages)),
        "training_route_counts": dict(Counter(row["route"] for row in members)),
        "validation_feature_coverage": dict(Counter(row["feature_status"] for row in validation)),
        "validation_geometry_verified": sum(row["geometry_verified"] for row in validation),
        "spectrabrep_revision": SPECTRA_COMMIT, "spectrabrep_tree_sha256": SPECTRA_SOURCE_SHA256,
        "file_sha256": {path.name: sha256_file(path) for path in sorted(out.iterdir())},
    }
    (out / PREPARED_MANIFEST).write_bytes(json_bytes(result) + b"\n")
    return {**result, "eligibility_manifest_sha256": sha256_file(out / PREPARED_MANIFEST)}


def fit_validation(args, out):
    import joblib
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, normalize

    from idea01.in_domain_pilot import evaluate_rankings
    from idea01.in_domain_pilot_ridge import REGULARIZATION, tokenize_unigrams, train_ridge_columns

    receipt = pinned_manifest(args.prepared, PREPARED_MANIFEST, args.prepared_sha256)
    require(receipt["source"] == SOURCE and receipt["status"] == "training_eligibility_candidate"
            and receipt["scope"] == "preparation_without_training"
            and receipt["original_denominators"] == {**EXPECTED_SPLITS, "validation_queries": PANEL_SIZE},
            "Unexpected eligibility receipt")

    def load_rows(name):
        return list(read_jsonl(pinned_file(args.prepared, receipt, name)))

    members = load_rows("shared_training_candidate.jsonl")
    training = load_rows("train_pairs.jsonl")
    query_rows = load_rows("validation_queries.jsonl")
    gallery = load_rows("validation_gallery.jsonl")
    raw_training = np.load(pinned_file(args.prepared, receipt, "ridge_train_features.npy"), allow_pickle=False)
    raw_gallery = np.load(pinned_file(args.prepared, receipt, "ridge_validation_features.npy"), allow_pickle=False)
    train_uids = [row["uid"] for row in members]
    require(train_uids == sorted(set(train_uids)) == [row["uid"] for row in training]
            and [row["feature_index"] for row in members] == list(range(len(members)))
            and 0 < len(members) == receipt["training_eligible"] <= EXPECTED_SPLITS["train"],
            "Training fit does not match the reviewed common candidate UID axis")
    require(len(gallery) == EXPECTED_SPLITS["validation"] and len(query_rows) == PANEL_SIZE,
            "Validation fit would change the original denominator")
    require([row["cad_index"] for row in gallery] == list(range(len(gallery)))
            and [row["uid"] for row in gallery] == sorted({row["uid"] for row in gallery}),
            "Validation gallery index changed")
    valid_queries = np.asarray([text_observation(row)[1] == "ok" for row in query_rows])
    valid_gallery = np.asarray([row["feature_status"] == "ok" for row in gallery])
    require(raw_training.shape == (len(training), FEATURE_WIDTH) and np.isfinite(raw_training).all()
            and raw_gallery.shape == (len(gallery), FEATURE_WIDTH) and np.isfinite(raw_gallery[valid_gallery]).all()
            and np.isnan(raw_gallery[~valid_gallery]).all(), "Ridge features do not match the frozen validity axes")
    require(all(text_observation(row)[1] == "ok" for row in training), "Missing training annotations after eligibility")

    require_sha(args.validation_qrels, receipt["validation_qrels_sha256"])
    qrels = list(read_jsonl(args.validation_qrels))
    expected_targets = hash_order((row["uid"] for row in gallery), f"cadretrieval-val-q-v1|{SOURCE}|")[:PANEL_SIZE]
    require([row["query_id"] for row in qrels] == [row["query_id"] for row in query_rows]
            and [row["uid"] for row in qrels] == expected_targets, "Validation qrels do not match the frozen panel")

    vectorizer = TfidfVectorizer(tokenizer=tokenize_unigrams, token_pattern=None, dtype=np.float64)
    train_matrix = vectorizer.fit_transform(row["text"] for row in training)
    val_matrix = vectorizer.transform(row["text"] or "" for row in query_rows)
    scaler = StandardScaler().fit(raw_training)
    train_targets = scaler.transform(raw_training)
    gallery_norm = normalize(scaler.transform(raw_gallery[valid_gallery])) if valid_gallery.any() else raw_gallery[:0]
    evaluation_queries = [{**query, "uid": rel["uid"], "text": query["text"] or ""}
                          for query, rel in zip(query_rows, qrels, strict=True)]
    tuning = {}
    best = None
    for alpha in REGULARIZATION:
        weights, bias, iterations = train_ridge_columns(train_matrix, train_targets, alpha, args.workers)
        predicted = normalize(val_matrix @ weights.T + bias)
        scores = np.full((len(query_rows), len(gallery)), np.nan, dtype=np.float64)
        scores[np.ix_(valid_queries, valid_gallery)] = predicted[valid_queries] @ gallery_norm.T
        ranks, metrics = evaluate_rankings(scores, evaluation_queries, gallery)
        ranks = [{**rank, "query_id": query["query_id"], "text_sha256": query["text_sha256"]}
                 for query, rank in zip(query_rows, ranks, strict=True)]
        write_jsonl(out / f"ranks_alpha_{alpha:g}.jsonl", ranks)
        r20 = metrics["annotation"]["R@20"]
        tuning[str(alpha)] = {"metrics": metrics, "lsqr_iterations": iterations}
        print(f"Fusion validation alpha {alpha:g}, R@20 {r20:.6f}", flush=True)
        if best is None or (r20, alpha) > best[0]:
            best = (r20, alpha), weights, bias, scores, ranks, metrics
    (_, alpha), weights, bias, scores, ranks, metrics = best
    np.save(out / "prediction_scores.npy", scores, allow_pickle=False)
    write_jsonl(out / "query_ranks.jsonl", ranks)
    write_jsonl(out / "prediction_queries.jsonl", query_rows)
    write_jsonl(out / "prediction_gallery.jsonl", gallery)
    joblib.dump({"tfidf": vectorizer, "feature_scaler": scaler, "coefficients": weights,
                 "bias": bias, "alpha": alpha}, out / "fitted_ridge.joblib", compress=3)
    report = {"source": SOURCE, "scope": "validation_only", "eligibility_manifest_sha256": args.prepared_sha256,
              "validation_qrels_sha256": receipt["validation_qrels_sha256"], "selected_alpha": alpha,
              "alpha_rule": "validation R@20, then larger alpha", "validation_metrics": metrics,
              "regularizations": tuning, "training_uid_count": len(training),
              "original_denominators": receipt["original_denominators"],
              "valid_annotation_queries": int(valid_queries.sum()), "valid_cad_candidates": int(valid_gallery.sum()),
              "feature_width": FEATURE_WIDTH, "vocabulary": len(vectorizer.vocabulary_), "sklearn": sklearn.__version__,
              "uniform_random_R@20": 100.0 * 20 / len(gallery),
              "file_sha256": {path.name: sha256_file(path) for path in sorted(out.iterdir())}}
    (out / "summary.json").write_bytes(json_bytes(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="stage", required=True)
    inputs = phases.add_parser("prepare")
    for option in ("catalog", "conversion", "spectrabrep-source"):
        inputs.add_argument(f"--{option}", type=Path, required=True)
    for option in ("catalog-sha256", "conversion-sha256"):
        inputs.add_argument(f"--{option}", required=True)
    scoring = phases.add_parser("fit-validation")
    scoring.add_argument("--prepared", type=Path, required=True)
    scoring.add_argument("--prepared-sha256", required=True)
    scoring.add_argument("--validation-qrels", type=Path, required=True)
    for phase in (inputs, scoring):
        phase.add_argument("--output", type=Path, required=True)
        phase.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("at least one CPU worker is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    for thread_limit in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(thread_limit, "1")

    start = monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fusion-ridge-", dir=args.output.parent) as stage:
        payload = Path(stage) / "output"
        payload.mkdir()
        action = prepare if args.stage == "prepare" else fit_validation
        result = action(args, payload)
        payload.rename(args.output)
    print(json.dumps({"stage": args.stage, "seconds": monotonic() - start, **result}), flush=True)


if __name__ == "__main__":
    main()
