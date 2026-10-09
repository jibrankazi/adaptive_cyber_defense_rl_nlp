"""Unit fixtures check code contracts; numbers here are not empirical phishing sites."""
import numpy as np
import pandas as pd
import pytest

from src.real_phishing_websites import (
    split_unique_feature_patterns, summarize_real_labels,
    validate_published_phishing_data, score_preextracted_website_features,
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


def test_local_phishing_model_inference_accepts_only_original_30_attributes(tmp_path):
    """Fixture model tests loader and input contract only; no real-site claim."""
    import json
    import joblib
    from sklearn.dummy import DummyClassifier
    cols = [f"published_feature_{i}" for i in range(30)]
    (tmp_path / "ordered_original_feature_columns.json").write_text(json.dumps(cols))
    train = pd.DataFrame(np.zeros((4,30)), columns=cols)
    model = DummyClassifier(strategy="prior")
    model.fit(train, [0,1,0,1])
    joblib.dump(model, tmp_path / "selected_uci_phishing_model.joblib")
    valid = tmp_path / "original_attributes.json"
    valid.write_text(json.dumps({name: -1 if i % 2 == 0 else 1 for i,name in enumerate(cols)}))
    result = score_preextracted_website_features(str(valid), str(tmp_path))
    assert result["phishing_probability"] == pytest.approx(.5)
    assert result["historical_dataset_study_flag_at_0_5"] is True
    assert result["not_a_live_website_scanner"]
    valid.write_text(json.dumps({cols[0]: 1}))
    with pytest.raises(ValueError, match="exactly 30"):
        score_preextracted_website_features(str(valid), str(tmp_path))
    valid.write_text(json.dumps({name: (4 if i == 2 else 1) for i,name in enumerate(cols)}))
    with pytest.raises(ValueError, match="exactly -1, 0, or 1"):
        score_preextracted_website_features(str(valid), str(tmp_path))
