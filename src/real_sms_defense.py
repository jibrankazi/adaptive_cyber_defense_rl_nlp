"""Real UCI SMS spam messages -> NLP -> validation-selected defensive triage.

A runnable offline spam message *review recommender*, not intrusion defense,
malware detection, phishing detection, or trained reinforcement learning.
Real labels are used to calculate historical, ASSUMED review/miss costs.
No real-world interventions or attack outcomes are present in this dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SOURCE = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
CATALOG = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
CITATION = "Almeida and Hidalgo, UCI SMS Spam Collection, DOI 10.24432/C5CC84"
# Explicit ASSUMPTIONS, not observed financial costs or real interventions.
FALSE_REVIEW_COST_UNITS = 2
MISSED_SPAM_COST_UNITS = 10
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def parse_uci_bytes(raw: bytes) -> pd.DataFrame:
    if len(raw) < 100_000:
        raise ValueError("UCI data download missing or unexpectedly small")
    try:
        with ZipFile(BytesIO(raw)) as archive:
            names = [name for name in archive.namelist()
                     if name.rsplit("/", 1)[-1].lower() == "smsspamcollection"]
            if len(names) != 1:
                raise ValueError("UCI archive missing uniquely named SMS collection")
            original = archive.read(names[0]).decode("latin1")
    except BadZipFile as exc:
        raise ValueError("UCI publisher response was not a ZIP archive") from exc
    rows = []
    for line in original.splitlines():
        if not line.strip():
            continue
        label, sep, message = line.partition("\t")
        if not sep or label not in {"ham", "spam"} or not message.strip():
            raise ValueError("Original UCI message or label corrupted")
        rows.append((label, message))
    df = pd.DataFrame(rows, columns=["label", "message"])
    if len(df) < 5500 or len(df) > 5700:
        raise ValueError(f"Unexpected source corpus size: {len(df)}")
    if not 650 <= (df["label"] == "spam").sum() <= 850:
        raise ValueError("Unexpected original publisher spam prevalence")
    conflicts = df.groupby("message")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Same public SMS text occurs under different target labels")
    df = df.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    if len(df) < 5000:
        raise ValueError("Too few original unique SMS messages")
    return df


def fetch_original_uci(session=None):
    client = session or requests
    response = client.get(SOURCE, timeout=70, headers={"Accept-Encoding": "identity"})
    response.raise_for_status()
    raw = response.content
    return parse_uci_bytes(raw), sha256(raw).hexdigest()


def partition_original_messages(df):
    # Original text duplicates were removed BEFORE any stratified split.
    indexes = np.arange(len(df))
    labels = df["label"].eq("spam").astype(int).to_numpy()
    train_valid, test = train_test_split(
        indexes, test_size=0.20, stratify=labels, random_state=42)
    train, valid = train_test_split(
        train_valid, test_size=0.25, stratify=labels[train_valid], random_state=42)
    if len(set(train) | set(valid) | set(test)) != len(df):
        raise AssertionError("A row appeared in more than one split")
    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise AssertionError("Source texts overlap partitions")
    return train, valid, test, labels


def build_classifier():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20000,
            strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1600, random_state=42)),
    ])


def spam_probabilities(model, messages):
    classes = model.named_steps["classifier"].classes_
    if set(classes) != {0, 1}:
        raise ValueError("Classifier must preserve actual 0=ham and 1=spam labels")
    positive_idx = int(np.flatnonzero(classes == 1)[0])
    probability = model.predict_proba(messages)[:, positive_idx]
    if not np.isfinite(probability).all() or not np.logical_and(
        probability >= 0, probability <= 1).all():
        raise ValueError("Invalid predicted spam probabilities")
    return probability


def confusion_cost(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Actual targets and scored probabilities must be aligned")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Expected original binary spam labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Bad probability input")
    predicts_review = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predicts_review, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "true_ham_allow": int(tn),
        "false_ham_review": int(fp),
        "missed_spam_allow": int(fn),
        "true_spam_review": int(tp),
        "assumed_cost_units": int(fp * FALSE_REVIEW_COST_UNITS +
                                  fn * MISSED_SPAM_COST_UNITS),
        "spam_precision": float(precision_score(labels, predicts_review, zero_division=0)),
        "spam_recall": float(recall_score(labels, predicts_review, zero_division=0)),
        "spam_f1": float(f1_score(labels, predicts_review, zero_division=0)),
    }


def choose_review_threshold(actual_labels, probabilities):
    # Validation-only: choose minimum ASSUMED total missed-spam/false-review cost.
    options = [confusion_cost(actual_labels, probabilities, t) for t in THRESHOLDS]
    return min(options, key=lambda item: (
        item["assumed_cost_units"],
        item["false_ham_review"],
        abs(item["threshold"] - 0.5),
    ))


def run(output_dir="results/real_uci_sms_triage"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, source_hash = fetch_original_uci()
    train, valid, test, labels = partition_original_messages(df)
    model = build_classifier()
    model.fit(df.iloc[train]["message"], labels[train])
    valid_prob = spam_probabilities(model, df.iloc[valid]["message"])
    selected = choose_review_threshold(labels[valid], valid_prob)
    test_prob = spam_probabilities(model, df.iloc[test]["message"])
    threshold = selected["threshold"]
    scenarios = {
        "allow_all": confusion_cost(labels[test], test_prob, 1.01),
        "review_all": confusion_cost(labels[test], test_prob, 0.0),
        "fixed_probability_0_5": confusion_cost(labels[test], test_prob, 0.5),
        "selected_on_validation_only": confusion_cost(labels[test], test_prob, threshold),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_frame = pd.DataFrame({
        "deduplicated_original_source_row_index": test,
        "original_uci_label_spam_1": labels[test],
        "trained_model_spam_probability": test_prob,
        "validation_selected_review": (test_prob >= threshold).astype(int),
    }).sort_values("deduplicated_original_source_row_index")
    test_frame.to_csv(output / "heldout_observed_sms_scores.csv", index=False)
    precision, recall, _ = precision_recall_curve(labels[test], test_prob)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, label="Original UCI SMS heldout PR curve")
    ax.axhline(float(labels[test].mean()), linestyle="--",
               label="Original observed spam label prevalence")
    ax.set(title="UCI 2011 SMS: heldout spam precision-recall",
           xlabel="Spam recall", ylabel="Spam precision",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "heldout_precision_recall.png", dpi=150)
    plt.close(fig)
    policy = {
        "task": "Historical UCI SMS message spam triage; not cyber intrusion defense",
        "data_source": SOURCE,
        "data_catalog": CATALOG,
        "citation": CITATION,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "publisher_zip_bytes_sha256": source_hash,
        "deduplicated_original_message_count": int(len(df)),
        "deduplicated_original_spam_count": int(labels.sum()),
        "duplicate_messages_removed_before_split": None,
        "split": {
            "seed": 42, "strategy": "stratified 60/20/20 on unique original messages",
            "train": int(len(train)), "validation": int(len(valid)), "test": int(len(test)),
            "test_spam": int(labels[test].sum()),
            "text_duplicates_crossing_splits": 0,
        },
        "model": "Training-only word/bigram TF-IDF + balanced logistic regression",
        "score_and_action": "Probability >= chosen threshold means review; not actual message quarantine",
        "cost_assumptions": {
            "false_review_of_real_ham_units": FALSE_REVIEW_COST_UNITS,
            "missed_original_spam_units": MISSED_SPAM_COST_UNITS,
            "units": "arbitrary scenario points, NOT observed real incident or financial costs",
        },
        "threshold_choice": {
            "data_used": "validation only, threshold grid from 0.01 through 0.99",
            "selected": selected,
        },
        "untouched_test": {
            "average_precision": float(average_precision_score(labels[test], test_prob)),
            "roc_auc": float(roc_auc_score(labels[test], test_prob)),
            "original_spam_prevalence": float(labels[test].mean()),
            "policy_comparisons": scenarios,
        },
        "risk_boundaries": (
            "Genuine historically collected SMS spam and ham labels, NOT current cyber "
            "attack logs. Review/miss costs are manually assumed and are not "
            "economic effects. The model is not evaluated for phishing, malware, "
            "intrusion, real-world blocking, adversarial evasions or generalization "
            "to other years. No RL or sequential defense trajectories were trained."
        ),
    }
    # Never package public SMS bodies as model-inference evidence.
    policy["duplicate_messages_removed_before_split"] = None
    (output / "verified_results.json").write_text(
        json.dumps(policy, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump(model, output / "uci_sms_review_model.joblib")
    (output / "review_threshold.json").write_text(
        json.dumps({
            "threshold": threshold,
            "positive_class": "spam",
            "model": "uci_sms_review_model.joblib",
            "warning": "Only load local model files from trusted sources. Offline SMS review advice only.",
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in policy.items()
                      if key not in ("risk_boundaries",)}, indent=2), flush=True)
    return policy


def predict_review(text: str, model_file: str, results_file: str):
    if not text.strip():
        raise ValueError("A nonempty message is needed for local review")
    # Model-file loading uses pickle semantics: only use artifacts YOU trust.
    model = joblib.load(model_file)
    result = json.loads(Path(results_file).read_text(encoding="utf-8"))
    if result["task"] != "Historical UCI SMS message spam triage; not cyber intrusion defense":
        raise ValueError("Unexpected evaluation/provenance file; refusing inference")
    threshold = float(result["threshold_choice"]["selected"]["threshold"])
    probability = float(spam_probabilities(model, [text])[0])
    return {
        "spam_probability": probability,
        "recommendation": "review" if probability >= threshold else "allow",
        "validation_selected_threshold": threshold,
        "not_real_network_blocking": True,
        "scope": "Original UCI SMS spam/ham lexical pattern research; not cyber intrusion",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/real_uci_sms_triage")
    parser.add_argument("--predict-text",
                        help="If specified, load locally saved model and score this one SMS")
    parser.add_argument("--model-file", help="Trusted existing local joblib model")
    parser.add_argument("--metrics-file", help="Matching source/evaluation JSON")
    args = parser.parse_args()
    if args.predict_text is None:
        if args.model_file or args.metrics_file:
            parser.error("--model-file and --metrics-file are for --predict-text")
        run(args.output_dir)
    else:
        directory = Path(args.output_dir)
        scored = predict_review(
            args.predict_text,
            args.model_file or str(directory / "uci_sms_review_model.joblib"),
            args.metrics_file or str(directory / "verified_results.json"),
        )
        print(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()
