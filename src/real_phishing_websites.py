"""Verified actual UCI Phishing Websites features -> supervised site-risk benchmark.

Real independently collected historical phishing/legitimate sites, UCI #327,
Mohammad & McCluskey 2012. Not live URL scanning, intrusion detection,
reinforcement learning, BERT, real action rewards, or current threat defense.

Near-duplicate safety: identical observed 30-field feature vectors are GROUPED,
not split between development and holdout. Labels are NOT fabricated.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UCI_ID = 327
CATALOG = "https://archive.ics.uci.edu/dataset/327/phishing+websites"
DOI = "10.24432/C51W2X"
DATASET_SIZE = 11055
POSITIVE_PHISHING = 4898
NEGATIVE_LEGITIMATE = 6157


def validate_published_phishing_data(features, targets):
    if features is None or targets is None:
        raise ValueError("Genuine original UCI Phishing Websites features and labels are required")
    X = pd.DataFrame(features).copy()
    y_original = pd.DataFrame(targets).copy()
    if X.shape != (DATASET_SIZE, 30) or y_original.shape != (DATASET_SIZE, 1):
        raise ValueError(f"Publisher row or feature contract unexpectedly changed: {X.shape}, {y_original.shape}")
    if X.columns.duplicated().any() or X.isna().any().any() or y_original.isna().any().any():
        raise ValueError("Original data is missing fields, contains duplicate names or nulls")
    X = X.apply(pd.to_numeric, errors="raise").astype("int8")
    if not X.isin([-1, 0, 1]).all().all():
        raise ValueError("Unexpected values outside documented {-1,0,1} UCI feature coding")
    y_raw = pd.to_numeric(y_original.iloc[:, 0], errors="raise").astype("int8")
    if set(y_raw.unique()) != {-1, 1}:
        raise ValueError("UCI Result must encode -1 phishing and +1 legitimate")
    if (y_raw == -1).sum() != POSITIVE_PHISHING or (y_raw == 1).sum() != NEGATIVE_LEGITIMATE:
        raise ValueError("Original publisher phishing/legitimate class counts failed verification")
    X.reset_index(drop=True, inplace=True)
    y = (y_raw.reset_index(drop=True) == -1).astype("int8")
    digest = sha256(
        X.to_numpy(dtype="<i1", copy=True).tobytes() +
        y_raw.to_numpy(dtype="<i1").tobytes()
    ).hexdigest()
    # Fingerprint refers to decoded ordered tabular values, NOT network bytes.
    return X, y, digest


def download_uci_phishing_data():
    from ucimlrepo import fetch_ucirepo
    original = fetch_ucirepo(id=UCI_ID)
    X, y, digest = validate_published_phishing_data(
        original.data.features, original.data.targets)
    return X, y, digest


def split_unique_feature_patterns(X, y, seed=42):
    """Keep identical observed website feature fingerprints in ONE partition."""
    if len(X) != len(y):
        raise ValueError("Features/targets length mismatch")
    source_ids = np.arange(len(X))
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    dev_idx, test_idx = next(first.split(X, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    train_position, val_position = next(
        second.split(X.iloc[dev_idx], y.iloc[dev_idx], groups[dev_idx]))
    train_idx, val_idx = dev_idx[train_position], dev_idx[val_position]
    used_groups = [set(groups[indices]) for indices in (train_idx, val_idx, test_idx)]
    if any(used_groups[i] & used_groups[j] for i, j in ((0,1),(0,2),(1,2))):
        raise AssertionError("Identical feature patterns leaked between original partitions")
    if len(np.unique(np.concatenate([train_idx,val_idx,test_idx]))) != len(X):
        raise AssertionError("Original-source row was omitted or reused")
    if any(len(np.unique(y.iloc[idx])) != 2 for idx in (train_idx, val_idx, test_idx)):
        raise ValueError("Every partition must include real examples of both observed classes")
    if min(len(train_idx), len(val_idx), len(test_idx)) < 900:
        raise ValueError("Original train/validation/test partitions implausibly small")
    return train_idx, val_idx, test_idx, groups


def summarize_real_labels(y, probs):
    predictions = (probs >= 0.5).astype("int8")
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_phishing": float(precision_score(y, predictions, zero_division=0)),
        "recall_phishing": float(recall_score(y, predictions, zero_division=0)),
        "f1_phishing": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "confusion_matrix_actual_0_1": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def run(output="results/real_uci_phishing"):
    X, y, data_hash = download_uci_phishing_data()
    train_idx, val_idx, test_idx, groups = split_unique_feature_patterns(X, y)
    estimators = {
        "original_prevalence_baseline": DummyClassifier(strategy="prior"),
        "scaled_logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)),
        "random_forest": RandomForestClassifier(
            n_estimators=180, max_depth=14, min_samples_leaf=2,
            n_jobs=1, random_state=42),
    }
    validated = {}
    for name, model in estimators.items():
        print(f"Fit actual UCI website feature risk classifier: {name}", flush=True)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        validation_probs = model.predict_proba(X.iloc[val_idx])[:, list(model.classes_).index(1)]
        if not np.isfinite(validation_probs).all():
            raise ValueError("Invalid model probabilities on original validation source rows")
        validated[name] = {
            "model": model,
            "validation_average_precision": float(
                average_precision_score(y.iloc[val_idx], validation_probs)),
        }
    # Model choice uses only original train and validation, never untouched test.
    choices = ("scaled_logistic_regression", "random_forest")
    selected = max(choices, key=lambda name: validated[name]["validation_average_precision"])
    metrics = {}
    heldout_predictions = pd.DataFrame({
        "original_publisher_row_position": test_idx,
        "actual_original_phishing_label": y.iloc[test_idx].to_numpy(),
        "original_feature_vector_group_hash": groups[test_idx].astype(str),
    })
    for name, record in validated.items():
        p = record["model"].predict_proba(X.iloc[test_idx])[:, list(record["model"].classes_).index(1)]
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Unexpected learned phishing scoring values")
        heldout_predictions[f"phishing_probability_{name}"] = p
        metrics[name] = {
            "model_selection_validation_average_precision": record["validation_average_precision"],
            "untouched_test_metrics": summarize_real_labels(y.iloc[test_idx], p),
        }
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    heldout_predictions.sort_values("original_publisher_row_position").to_csv(
        dest / "observed_uci_website_heldout_scores.csv", index=False)
    joblib.dump(validated[selected]["model"], dest / "selected_uci_phishing_model.joblib")
    (dest / "ordered_original_feature_columns.json").write_text(
        json.dumps(list(X.columns), indent=2) + "\n", encoding="utf-8")
    evaluation = {
        "source_catalog": CATALOG,
        "publisher_dataset_id": UCI_ID,
        "original_publication": "Mohammad and McCluskey (2012), collected mainly from PhishTank, MillerSmiles and Google search",
        "original_dataset_doi": DOI,
        "license": "CC BY 4.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_decoded_ordered_feature_and_label_values_sha256": data_hash,
        "note_on_digest": "Computed on 30 source feature values and original -1/+1 labels in publisher order; not original HTTP bytes",
        "original_rows": DATASET_SIZE,
        "feature_columns": list(X.columns),
        "feature_count": 30,
        "original_phishing_sites": int(y.sum()),
        "original_legitimate_sites": int((1-y).sum()),
        "source_has_raw_url_strings": False,
        "train_validation_test": {
            "method": "two fixed-seed GroupShuffleSplits on original 30-value feature fingerprints; not time ordered",
            "seed": 42,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "test_phishing_sites": int(y.iloc[test_idx].sum()),
            "distinct_feature_vector_patterns": int(len(np.unique(groups))),
            "feature_pattern_overlap_across_partitions": 0,
        },
        "selected_by_validation_only": selected,
        "classification_decision": "probability >= 0.5 is flagged as phishing in this study, not automatic site blocking",
        "model_evaluations": metrics,
        "limitations": (
            "Historical annotated phishing-website *engineered numeric features*; "
            "not actual raw page HTML or URL scanner, not current phishing attacks "
            "or production intrusion prevention. Source features include webpage, "
            "registration, link, index and traffic observations potentially "
            "collected after URL discovery. Original publication has no trustworthy "
            "chronological train/test timestamps, so the group-disjoint random "
            "split cannot estimate current/future performance or operational "
            "false alarms. No RL model, action logging or defense effect established."
        ),
    }
    (dest / "verified_results.json").write_text(
        json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "original_rows": evaluation["original_rows"],
        "training": evaluation["train_validation_test"],
        "selected_model": selected,
        "observed_model_results": {
            name: item["untouched_test_metrics"] for name, item in metrics.items()
        },
        "scope": "Original UCI 2012 tabular phishing website feature benchmark only",
    }, indent=2), flush=True)
    return evaluation


def score_preextracted_website_features(input_json, folder="results/real_uci_phishing"):
    """Classify one already-measured 30-field UCI feature vector.

    This does not visit arbitrary URLs, fetch web pages, validate a domain,
    or execute code from a website. Only use joblib files YOU have trained
    or independently trust, since loading joblib has pickle semantics.
    """
    destination = Path(folder)
    required = json.loads(
        (destination / "ordered_original_feature_columns.json").read_text(encoding="utf-8"))
    provided = json.loads(Path(input_json).read_text(encoding="utf-8"))
    if not isinstance(provided, dict) or set(provided) != set(required):
        raise ValueError("Expected exactly 30 original UCI website attribute names")
    frame = pd.DataFrame([provided], columns=required)
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All 30 required website features must be numeric") from exc
    if numeric.isna().any().any() or not numeric.isin([-1, 0, 1]).all().all():
        raise ValueError("Every published UCI feature must be exactly -1, 0, or 1")
    model = joblib.load(destination / "selected_uci_phishing_model.joblib")
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Unknown model label mapping; 1 must represent phishing")
    probability = float(model.predict_proba(numeric)[:, classes.index(1)][0])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The loaded classifier returned invalid probabilities")
    return {
        "phishing_probability": probability,
        "historical_dataset_study_flag_at_0_5": bool(probability >= 0.5),
        "input": "preextracted UCI Phishing Websites 30 integer-coded site features",
        "not_a_live_website_scanner": True,
        "not_an_intrusion_blocker": True,
        "warning": "Historical 2012 classifier; no independent current phishing-site validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/real_uci_phishing")
    parser.add_argument("--predict-features-json",
                        help="Score only an already-extracted JSON object with all 30 original UCI website features")
    args = parser.parse_args()
    if args.predict_features_json:
        print(json.dumps(score_preextracted_website_features(
            args.predict_features_json, args.output), indent=2))
    else:
        run(args.output)


if __name__ == "__main__":
    main()
