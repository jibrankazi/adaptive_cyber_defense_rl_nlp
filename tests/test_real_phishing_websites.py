"""Unit fixtures check code contracts; numbers here are not empirical phishing sites."""
import numpy as np
import pandas as pd
import pytest

from src.real_phishing_websites import (
    split_unique_feature_patterns, summarize_real_labels,
    validate_published_phishing_data,
)


def test_refuse_partial_missing_or_fabricated_dataset_contract():
    with pytest.raises(ValueError, match="Publisher row"):
        validate_published_phishing_data(
            pd.DataFrame(np.zeros((10, 30))),
            pd.DataFrame({"Result": [-1] * 10}),
        )
    with pytest.raises(ValueError):
        validate_published_phishing_data(None, None)


def test_observed_feature_patterns_grouped_across_splits():
    rng = np.random.default_rng(42)
    # Software-only fixture, not a claimed real phishing corpus.
    synthetic = pd.DataFrame(
        rng.integers(-1, 2, size=(6000, 30)),
        columns=[f"fixture_feature_{i}" for i in range(30)],
    )
    synthetic = pd.concat([synthetic, synthetic.iloc[:90].copy()], ignore_index=True)
    y = pd.Series(rng.integers(0, 2, size=6000), dtype="int8")
    y = pd.concat([y, y.iloc[:90].copy()], ignore_index=True)
    train, valid, test, groups = split_unique_feature_patterns(synthetic, y)
    assert len(train) + len(valid) + len(test) == len(synthetic)
    assert len(set(groups[train]) & set(groups[valid])) == 0
    assert len(set(groups[train]) & set(groups[test])) == 0
    assert len(set(groups[valid]) & set(groups[test])) == 0
    assert len(train) > 1000
    assert len(valid) > 900 and len(test) > 900


def test_confusion_and_precision_recall_are_calculated_from_actual_inputs():
    true = np.array([1, 1, 0, 0, 1, 0])
    prob = np.array([.9, .6, .3, .02, .2, .8])
    result = summarize_real_labels(true, prob)
    assert result["confusion_matrix_actual_0_1"] == [[2, 1], [1, 2]]
    assert result["precision_phishing"] == pytest.approx(2/3)
    assert result["recall_phishing"] == pytest.approx(2/3)
    assert 0 < result["average_precision"] < 1
    assert 0 < result["roc_auc"] < 1
