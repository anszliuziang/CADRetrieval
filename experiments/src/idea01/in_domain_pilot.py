"""Data registration and exact rank evaluation for the in-domain CAD pilot."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

import numpy as np

LEVEL_FIELDS = {"L0": "abstract", "L1": "beginner", "L2": "intermediate", "L3": "expert"}
SOURCE_LEVELS = {"t2cad-v1.0": tuple(LEVEL_FIELDS), "ct-deepcad": ("CT",)}
SPLIT_COUNTS = {"train": 159049, "validation": 8925, "test": 8046}
EMPTY = {"", "nan", "n/a", "null", "<na>"}
PANEL_LENGTH = 512
GALLERY_LENGTH = 2048
TRAIN_LENGTH = 4096
BACKGROUND_LENGTH = 48
CUTOFFS = (1, 5, 10, 20)
ORIGINAL_UID = re.compile(r"[0-9]{4}/[0-9]{8}")
CT_PREFIX = re.compile(r"^/?([0-9]{4}/[0-9]{8})/")
CT_VECTOR = re.compile(r"dataset/deepcad_txt/cad_vec_original/([0-9]{4}/[0-9]{8})\.h5")
INPUT_HASHES = {
    "annotations": "32427ca7169f8b93411b64a025c387cdb0b55d9b9ce0e9cf9755ac8a3528f4a3",
    "split": "4ea6db9027b4cebee8f37903b6601b73b810adacd808a4874dbe847c7c508e7b",
    "official_zip": "9f3a8bb85f9ceb078c19bc43f750fe596d3323631f2e846a7f541e47af7e7eaa",
    "ct_zip": "0c6c57a2266f6519bb43bc85c16d51450db38429237889faf4683043b1089170",
}
COHORTS = {
    "t2cad-v1.0": {
        "train": (151454, "c6a24e92d1e2df106ef2f4ae15389f2d55f18af6eb46ad5cc33f1347b95a623f"),
        "validation": (8509, "d9b88795d638034567df6a31ab4301cf995cf871db21cf8d11fb95d459f2de51"),
        "test": (7627, "74e104ce4a9b90d03f27888c2d921509262698f997bd9912d0142dfec7d049c5"),
    },
    "ct-deepcad": {
        "train": (148046, "dea6c8ee30b8965c0459697bf9cc47c4cdadbf541c27f59b65a1a7d01a5fa0bd"),
        "validation": (8221, "a104753b890723c1d645a2eeef75c84b1737042569facf486ac7bdecba0e40ae"),
        "test": (7526, "4c6337f4d01655f26e479d3cafe35834d7dcdd35d8594b8ccd283f14ec77b89d"),
    },
}
PANEL_HASH_PREFIXES = {
    "t2cad-v1.0": ("a29a48566f5e", "c5ea7bbb42b4", "8c688341733b"),
    "ct-deepcad": ("7f35d2f3543e", "109fd68e4451", "d6457715e3da"),
}
# Program types are the native structural and Boolean end codes. Coordinates use
# their published 8-bit value bins, combined without row, axis or history order.
FEATURE_NAMES = (
    "end_sketch", "end_face", "end_loop", "end_curve", "end_extrusion",
    "new_body", "join", "cut", "intersect", "line", "arc", "circle", "other_curve",
    *(f"quantized_scalar_bin_{index:02d}" for index in range(16)),
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_lines(values):
    return sha256_bytes("".join(f"{value}\n" for value in values).encode("utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def hash_order(uids, prefix, source):
    return sorted(uids, key=lambda uid: (sha256_bytes(f"{prefix}|{source}|{uid}".encode("utf-8")), uid))


def select_panel(validation, source):
    targets = hash_order(validation, "cadretrieval-val-q-v1", source)[:PANEL_LENGTH]
    extras = hash_order(set(validation) - set(targets), "cadretrieval-val-c-v1", source)
    gallery = sorted(targets + extras[:GALLERY_LENGTH - len(targets)])
    return targets, gallery


def select_background(validation, source):
    level = SOURCE_LEVELS[source][0]
    original = defaultdict(list)
    for uid, captions in validation.items():
        original[captions[level]].append(uid)
    texts = sorted(original, key=lambda text: (sha256_bytes(("cadretrieval-bg-v1|" + text).encode("utf-8")), text))
    return [{"text": text, "uid": min(original[text]), "exact_count": len(original[text])}
            for text in texts[:BACKGROUND_LENGTH]]


def evaluate_rankings(scores, queries, gallery):
    """Rank the one registered UID per row, retaining repeated texts and failures."""
    matrix = np.asarray(scores)
    if matrix.shape != (len(queries), len(gallery)) or not len(gallery):
        raise ValueError("score matrix does not match query and CAD axes")
    index_by_uid = {row["uid"]: index for index, row in enumerate(gallery)}
    cad_indices = np.asarray([row["cad_index"] for row in gallery], dtype=np.int64)
    if len(index_by_uid) != len(gallery) or len(set(cad_indices)) != len(gallery):
        raise ValueError("CAD axis contains duplicate identifiers")
    identifiers = {(row["level"], row["uid"]) for row in queries}
    if len(identifiers) != len(queries):
        raise ValueError("duplicate level and UID on query axis")
    rank_rows = []
    by_level = defaultdict(list)
    for query, row in zip(queries, matrix, strict=True):
        column = index_by_uid.get(query["uid"])
        finite = np.isfinite(row)
        if column is None or not finite[column]:
            rank = len(gallery) + 1
        else:
            precedes = (row > row[column]) | ((row == row[column]) & (cad_indices < cad_indices[column]))
            rank = 1 + int(np.count_nonzero(finite & precedes))
        by_level[query["level"]].append(rank)
        rank_rows.append({"uid": query["uid"], "level": query["level"],
                          "text_sha256": sha256_bytes(query["text"].encode("utf-8")),
                          "rank": rank, "valid_candidates": int(finite.sum()),
                          "failed": rank == len(gallery) + 1})
    metrics = {}
    for level, ranks in by_level.items():
        hits = {str(k): sum(rank <= min(k, len(gallery)) for rank in ranks) for k in CUTOFFS}
        metrics[level] = {"queries": len(ranks), "candidates": len(gallery), "hits": hits,
                          **{f"R@{k}": 100.0 * hits[str(k)] / len(ranks) for k in CUTOFFS},
                          "MedR": float(np.median(ranks)),
                          "failures": sum(rank == len(gallery) + 1 for rank in ranks)}
    return rank_rows, metrics


@dataclass(frozen=True)
class Assets:
    annotations: Path
    split: Path
    official_zip: Path
    minimal_zip: Path
    ct_zip: Path
    old_test_gallery: Path

    def fingerprints(self):
        result = {}
        for key, path in vars(self).items():
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if key in INPUT_HASHES and actual != INPUT_HASHES[key]:
                raise ValueError(f"release fingerprint mismatch for {key}")
            result[key] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
        return result


def cad_member(uid):
    return f"cad_seq/{uid}/seq/{uid.rsplit('/', 1)[1]}.pth"


def minimal_member(uid):
    return f"{uid}/minimal_json/{uid.rsplit('/', 1)[1]}.json"


def official_splits(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if {key: len(values) for key, values in raw.items()} != SPLIT_COUNTS:
        raise ValueError("unexpected official split sizes")
    if len(set().union(*(set(values) for values in raw.values()))) != sum(SPLIT_COUNTS.values()):
        raise ValueError("official splits contain duplicate UIDs")
    if any(not ORIGINAL_UID.fullmatch(uid) for values in raw.values() for uid in values):
        raise ValueError("unexpected official UID namespace")
    return {key: set(values) for key, values in raw.items()}


def read_native_captions(path, uids):
    csv.field_size_limit(32 * 1024 * 1024)
    found = {}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            uid = row["uid"]
            if uid not in uids:
                continue
            if uid in seen:
                raise ValueError(f"repeated Text2CAD caption UID {uid}")
            seen.add(uid)
            captions = {level: row[field].strip() for level, field in LEVEL_FIELDS.items()}
            if all(value.casefold() not in EMPTY for value in captions.values()):
                found[uid] = captions
    return found


def read_ct_release(path):
    captions = {}
    with ZipFile(path) as archive:
        names = archive.namelist()
        text_member = "dataset/deepcad_txt/text.txt"
        if names.count(text_member) != 1:
            raise ValueError("ambiguous CadTranslator caption member")
        with io.TextIOWrapper(archive.open(text_member), encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                prefix = CT_PREFIX.match(line)
                if prefix is None:
                    raise ValueError(f"invalid CadTranslator UID at source line {number}")
                uid = prefix[1]
                text = line[prefix.end():].strip()
                if uid in captions:
                    raise ValueError(f"repeated CadTranslator caption UID {uid}")
                captions[uid] = text
        vectors = {}
        for member in names:
            match = CT_VECTOR.fullmatch(member)
            if match:
                uid = match[1]
                if uid in vectors:
                    raise ValueError(f"ambiguous CadTranslator CAD member for {uid}")
                vectors[uid] = member
    return {uid: {"CT": text} for uid, text in captions.items() if text}, vectors


_VECTOR_ARCHIVE = None


def _worker_setup(path):
    global _VECTOR_ARCHIVE
    _VECTOR_ARCHIVE = ZipFile(path)


def primitive_counts(x):
    """Count the primitive types defined by the author's per-loop parser."""
    counts = [0, 0, 0, 0]
    curve_lengths = []
    previous_end = -1
    for position in np.flatnonzero(x < 11):
        code = x[position]
        if code == 5:
            curve_lengths.append(int(position - previous_end - 1))
        elif code == 4:
            if len(curve_lengths) == 1:
                counts[2] += 1
            else:
                for length in curve_lengths:
                    counts[0 if length == 1 else 1 if length == 2 else 3] += 1
            curve_lengths.clear()
        previous_end = position
    return counts



def _features_for_uid(entry):
    uid, compute_features = entry
    import torch

    raw = _VECTOR_ARCHIVE.read(cad_member(uid))
    record = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    data = record["vec"]["cad_vec"].detach().numpy()
    if data.shape != (272, 2) or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{uid}: invalid quantized program shape or dtype")
    if np.any((data < 0) | (data > 266)):
        raise ValueError(f"{uid}: label outside native vocabulary")
    n_fields = int(np.count_nonzero(data[1:]))
    if not n_fields:
        raise ValueError(f"{uid}: no effective output fields")
    fingerprints = {"member_sha256": sha256_bytes(raw),
                    "quantized_sha256": sha256_bytes(data.astype("<i4", copy=False).tobytes()),
                    "n_fields": n_fields}
    if not compute_features:
        return uid, fingerprints, None
    labels = data[1:].ravel()
    structural = np.bincount(data[1:, 0], minlength=267)[2:11]
    numeric = labels[labels >= 11].astype(np.int16)
    bins = np.bincount((numeric - 11) // 16, minlength=16)
    return uid, fingerprints, np.concatenate((structural, primitive_counts(data[:, 0]), bins)).astype(np.int16)


def scan_official_programs(archive, uids, feature_uids, workers):
    ordered = sorted(uids)
    feature_axis = sorted(feature_uids)
    positions = {uid: index for index, uid in enumerate(feature_axis)}
    features = np.empty((len(feature_axis), len(FEATURE_NAMES)), dtype=np.int16)
    fingerprints = {}
    # fork shares the already imported CPU Torch runtime. Workers each reopen the
    # archive to avoid sharing a ZIP seek pointer.
    import torch

    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
                             initializer=_worker_setup, initargs=(archive,)) as pool:
        requested = ((uid, uid in feature_uids) for uid in ordered)
        for uid, record, feature in pool.map(_features_for_uid, requested, chunksize=64):
            fingerprints[uid] = record
            if feature is not None:
                features[positions[uid]] = feature
    return feature_axis, fingerprints, features


def cohort_summary(by_split, fingerprints, panel_queries, levels):
    train_hashes = {fingerprints[uid]["quantized_sha256"] for uid in by_split["train"]}
    train_texts = {text for captions in by_split["train"].values() for text in captions.values()}
    result = {}
    for level in levels:
        test_counts = Counter(captions[level] for captions in by_split["test"].values())
        result[level] = {
            "unique_panel_texts": len({by_split["validation"][uid][level] for uid in panel_queries}),
            "test_text_groups": len(test_counts),
            "test_singletons": sum(count == 1 for count in test_counts.values()),
            "test_maximum_text_group": max(test_counts.values(), default=0),
            "test_exact_uid_ceiling": {str(k): 100.0 * sum(min(k, n) for n in test_counts.values())
                                        / len(by_split["test"]) for k in CUTOFFS},
            "panel_queries_with_exact_training_text": sum(
                by_split["validation"][uid][level] in train_texts for uid in panel_queries),
            "panel_queries_with_exact_training_program": sum(
                fingerprints[uid]["quantized_sha256"] in train_hashes for uid in panel_queries),
            "test_queries_with_exact_training_program": sum(
                fingerprints[uid]["quantized_sha256"] in train_hashes for uid in by_split["test"]),
        }
    return result


def build_manifests(assets, output, workers):
    if output.exists():
        raise FileExistsError(output)
    pinned_assets = assets.fingerprints()
    splits = official_splits(assets.split)
    uids = set().union(*splits.values())
    test_axis = [row["cad_id"] for row in read_jsonl(assets.old_test_gallery)]
    if len(test_axis) != SPLIT_COUNTS["test"] or set(test_axis) != splits["test"]:
        raise ValueError("legacy official test CAD index does not cover the current gallery")
    with ZipFile(assets.official_zip) as archive:
        official_members = Counter(item.filename for item in archive.infolist())
    with ZipFile(assets.minimal_zip) as archive:
        minimal_members = Counter(item.filename for item in archive.infolist())
    if any(official_members[cad_member(uid)] != 1 for uid in uids):
        raise ValueError("missing or duplicate official quantized program")
    native = read_native_captions(assets.annotations, uids)
    ct, ct_programs = read_ct_release(assets.ct_zip)
    available = {"t2cad-v1.0": native, "ct-deepcad": ct}
    absent_minimal = {uid for uid in uids if minimal_members[minimal_member(uid)] != 1}
    print("Source inventories aligned; computing program fingerprints and count features", flush=True)
    uid_order, fingerprints, features = scan_official_programs(assets.official_zip, uids, uids - splits["test"], workers)
    result = {"assets": pinned_assets, "official_split_counts": SPLIT_COUNTS,
              "official_catalog_size": len(fingerprints), "missing_minimal_json": sorted(absent_minimal),
              "test_cad_axis_sha256": sha256_lines(test_axis),
              "test_evaluation": "inventory and program fingerprints only", "sources": {}}

    def row(uid, captions, index=None, *, use_ct=False):
        values = {"uid": uid, "cad_member": cad_member(uid),
                  "minimal_member": minimal_member(uid) if uid not in absent_minimal else None,
                  **fingerprints[uid], "texts": captions}
        if use_ct:
            values["ct_vector_member"] = ct_programs.get(uid)
        if index is not None:
            values["cad_index"] = index
        return values

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".manifest-") as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        np.save(staged / "program_count_features.npy", features, allow_pickle=False)
        write_jsonl(staged / "feature_axis.jsonl", (
            {"uid": uid, **fingerprints[uid]} for uid in uid_order))
        write_json(staged / "feature_names.json", FEATURE_NAMES)
        write_jsonl(staged / "test_gallery.jsonl", (row(uid, {}, index) for index, uid in enumerate(test_axis)))
        for source, captions in available.items():
            ct_source = source == "ct-deepcad"
            by_split = {
                split: {uid: captions[uid] for uid in sorted(members & captions.keys())
                        if minimal_members[minimal_member(uid)] == 1
                        and (not ct_source or uid in ct_programs)}
                for split, members in splits.items()
            }
            for split, registered in by_split.items():
                actual = len(registered), sha256_lines(registered)
                if actual != COHORTS[source][split]:
                    raise ValueError(f"{source} {split} cohort differs from registered contract: {actual}")
            source_dir = staged / source
            source_dir.mkdir()
            query_uids, gallery_uids = select_panel(by_split["validation"], source)
            background = select_background(by_split["validation"], source)
            anchors = (sha256_lines(sorted(query_uids)), sha256_lines(gallery_uids),
                       sha256_lines(item["text"] for item in background))
            if tuple(value[:12] for value in anchors) != PANEL_HASH_PREFIXES[source]:
                raise ValueError(f"{source} validation panel or background differs: {anchors}")
            train_uids = hash_order(by_split["train"], "cadretrieval-pilot-train-v1", source)[:TRAIN_LENGTH]
            write_jsonl(source_dir / "pilot_train.jsonl",
                        (row(uid, by_split["train"][uid], use_ct=ct_source) for uid in train_uids))
            for split, records in by_split.items():
                filename = "test_queries.jsonl" if split == "test" else f"{split}.jsonl"
                write_jsonl(source_dir / filename,
                            (row(uid, texts, use_ct=ct_source) for uid, texts in records.items()))
            write_jsonl(source_dir / "validation_targets.jsonl",
                        (row(uid, by_split["validation"][uid], use_ct=ct_source) for uid in query_uids))
            write_jsonl(source_dir / "validation_gallery.jsonl",
                        (row(uid, {}, index, use_ct=ct_source) for index, uid in enumerate(gallery_uids)))
            write_jsonl(source_dir / "validation_background.jsonl", background)
            result["sources"][source] = {
                "split_sizes": {split: len(records) for split, records in by_split.items()},
                "split_uid_sha256": {split: sha256_lines(records) for split, records in by_split.items()},
                "validation_target_set_sha256": anchors[0], "validation_gallery_sha256": anchors[1],
                "validation_background_sha256": anchors[2], "pilot_train_uids": len(train_uids),
                "pilot_train_pairs": len(train_uids) * len(SOURCE_LEVELS[source]),
                "pilot_train_order_sha256": sha256_lines(train_uids),
                "validation_queries": len(query_uids), "validation_candidates": len(gallery_uids),
                "by_level": cohort_summary(by_split, fingerprints, query_uids, SOURCE_LEVELS[source]),
            }
        result["file_sha256"] = {path.relative_to(staged).as_posix(): sha256_file(path)
                                 for path in sorted(staged.rglob("*")) if path.is_file()}
        write_json(staged / "manifest_receipt.json", result)
        staged.rename(output)
    return result
