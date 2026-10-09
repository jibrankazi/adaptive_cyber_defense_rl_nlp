"""Unit/software-contract fixtures only; never pretend these are actual UCI cases."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.real_sms_defense import (
    build_classifier, choose_review_threshold, confusion_cost, parse_uci_bytes,
    partition_original_messages, spam_probabilities,
)


def test_reject_missing_or_corrupt_uci_bytes():
    with pytest.raises(ValueError):
        parse_uci_bytes(b"fake")
    with pytest.raises(ValueError):
        parse_uci_bytes(b"not a zip" * 20000)


def test_no_duplicate_original_text_leaks_into_holdout():
    # These are program test fixtures, NOT user/real-source observations.
    df = pd.DataFrame({
        "message": [f"Unique piece of software fixture number {i}" for i in range(110)],
        "label": ["spam" if i % 6 == 0 else "ham" for i in range(110)],
    })
    train, valid, test, y = partition_original_messages(df)
    assert len(train) + len(valid) + len(test) == len(df)
    assert not set(train) & set(valid)
    assert not set(train) & set(test)
    assert not set(valid) & set(test)
    assert y.sum() > 0


def test_assumed_review_cost_counts_actual_label_outcomes():
    # 2 ham and 2 spam fixtures, threshold .5 yields TP/TN/FP/FN each 1.
    actual = np.array([0, 0, 1, 1])
    probs = np.array([.1, .8, .2, .9])
    result = confusion_cost(actual, probs, .5)
    assert result["true_ham_allow"] == 1
    assert result["false_ham_review"] == 1
    assert result["missed_spam_allow"] == 1
    assert result["true_spam_review"] == 1
    assert result["assumed_cost_units"] == 12  # 2 false review + 10 missed spam
    assert result["spam_precision"] == .5
    assert result["spam_recall"] == .5
    with pytest.raises(ValueError):
        confusion_cost(actual, np.array([.1, .8]), .5)
    with pytest.raises(ValueError):
        confusion_cost(actual, np.array([.1, .8, -1, .4]), .5)


def test_review_threshold_selected_only_from_provided_validation_labels():
    actual = np.array([0, 0, 0, 1, 1])
    prob = np.array([.05, .09, .15, .79, .95])
    selected = choose_review_threshold(actual, prob)
    assert 0.01 <= selected["threshold"] <= .99
    assert selected["assumed_cost_units"] == 0
    assert selected["missed_spam_allow"] == 0


def test_actual_pipeline_scoring_class_contract():
    # Minimal fake text is solely to check code paths.
    text = [
        "my lovely cat was sleeping on the pillow",
        "hello friend see you at noon",
        "buy prizes now and win cash bonus",
        "claim your prize get money today",
        "i am home can you come over",
        "go to that free bonus cash deal now",
        "would you like to have dinner tonight",
        "exclusive winnings money money bonus",
    ]
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    pipeline = build_classifier()
    pipeline.fit(text, labels)
    prob = spam_probabilities(pipeline, ["my cat", "cash bonus money"])
    assert prob.shape == (2,)
    assert np.isfinite(prob).all()
    assert (prob >= 0).all() and (prob <= 1).all()
    assert isinstance(pipeline, Pipeline)
