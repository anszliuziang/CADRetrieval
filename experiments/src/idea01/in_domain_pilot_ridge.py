"""Fit the registered vocabulary-to-unordered-CAD baseline on share4."""

from concurrent.futures import ThreadPoolExecutor
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize

from .in_domain_pilot import SOURCE_LEVELS, evaluate_rankings, read_jsonl, write_json, write_jsonl

_TOKEN_PATTERN = re.compile(
    r"(?u)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?|[^\W\d_]+(?:['’][^\W\d_]+)?",
    re.IGNORECASE,
)
REGULARIZATION = (0.1, 1.0, 10.0)


def tokenize_unigrams(text):
    return _TOKEN_PATTERN.findall(text)


def raw_pairs(rows, levels):
    return [{"uid": row["uid"], "level": level, "text": row["texts"][level]}
            for level in levels for row in rows]


def train_ridge_columns(train_features, targets, alpha, number_of_workers):
    def fit_column(column):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
        model.fit(train_features, targets[:, column])
        return model.coef_, model.intercept_, int(model.n_iter_[0])

    with ThreadPoolExecutor(max_workers=number_of_workers) as pool:
        fitted = list(pool.map(fit_column, range(targets.shape[1])))
    coefficients, bias, iterations = zip(*fitted, strict=True)
    return np.stack(coefficients), np.asarray(bias), list(iterations)


def fit_baseline(manifests, source, subset, output, workers):
    import joblib

    if output.exists():
        raise FileExistsError(output)
    source_dir = manifests / source
    path = "pilot_train.jsonl" if subset == "pilot" else "train.jsonl"
    training = read_jsonl(source_dir / path)
    targets = read_jsonl(source_dir / "validation_targets.jsonl")
    gallery = read_jsonl(source_dir / "validation_gallery.jsonl")
    levels = SOURCE_LEVELS[source]
    entries = raw_pairs(training, levels)
    queries = raw_pairs(targets, levels)
    indices = {row["uid"]: i for i, row in enumerate(read_jsonl(manifests / "feature_axis.jsonl"))}
    all_features = np.load(manifests / "program_count_features.npy", mmap_mode="r", allow_pickle=False)
    def features(records):
        positions = [row['uid'] for row in records]
        return np.asarray(all_features[[indices[uid] for uid in positions]], dtype=np.float64)

    vectorizer = TfidfVectorizer(tokenizer=tokenize_unigrams, token_pattern=None, dtype=np.float64)
    text_matrix = vectorizer.fit_transform(row["text"] for row in entries)
    val_matrix = vectorizer.transform(row["text"] for row in queries)
    scale = StandardScaler().fit(features(training))
    target_array = scale.transform(features(entries))
    gallery_array = normalize(scale.transform(features(gallery)), norm="l2")
    result = {}
    best = None
    output.mkdir(parents=True)
    print(f"{source} {subset}: {len(entries)} training pairs, {len(vectorizer.vocabulary_)} terms", flush=True)
    for alpha in REGULARIZATION:
        coefficients, bias, iterations = train_ridge_columns(text_matrix, target_array, alpha, workers)
        predicted = normalize(val_matrix @ coefficients.T + bias, norm="l2")
        scores = predicted @ gallery_array.T
        ranks, metrics = evaluate_rankings(scores, queries, gallery)
        mean_r20 = sum(group["R@20"] for group in metrics.values()) / len(levels)
        selection_score = mean_r20, alpha
        result[str(alpha)] = {"metrics": metrics, "macro_R@20": mean_r20,
                              "lsqr_iterations": iterations}
        write_jsonl(output / f"ranks_alpha_{alpha:g}.jsonl", ranks)
        print(f"{source} {subset} alpha {alpha:g}: macro R@20 {mean_r20:.6f}", flush=True)
        if best is None or selection_score > best[0]:
            best = selection_score, coefficients, bias, scores
    (_, selected_alpha), coefficients, bias, scores = best
    np.save(output / "prediction_scores.npy", scores, allow_pickle=False)
    write_jsonl(output / "prediction_queries.jsonl", queries)
    write_jsonl(output / "prediction_gallery.jsonl", gallery)
    joblib.dump({"tfidf": vectorizer, "feature_scaler": scale, "coefficients": coefficients,
                 "bias": bias, "alpha": selected_alpha}, output / "fitted_ridge.joblib", compress=3)
    ranks, metrics = evaluate_rankings(scores, queries, gallery)
    write_jsonl(output / "query_ranks.jsonl", ranks)
    report = {"source": source, "training_subset": subset, "train_uids": len(training),
              "train_pairs": len(entries), "vocabulary": len(vectorizer.vocabulary_),
              "selected_alpha": selected_alpha, "validation_metrics": metrics,
              "uniform_random_R@20": 100.0 * 20 / len(gallery), "regularizations": result}
    write_json(output / "summary.json", report)
    return report
