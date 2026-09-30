"""Unit tests for Semi-Supervised Pseudo-Labeling Engine."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.pseudo_labeler import PseudoLabeler


def test_pseudo_labeler_harvests_high_confidence_samples():
    """Verify pseudo-labeler harvests samples with confidence >= threshold and retrains."""
    X, y = make_classification(n_samples=300, n_features=6, random_state=42)
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:200], y[150:200]
    X_unlabelled = X[200:]

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)

    labeler = PseudoLabeler(confidence_threshold=0.90, random_state=42)
    result, refined_model = labeler.refine_with_pseudo_labels(
        model=clf,
        X_train=X_train,
        y_train=y_train,
        X_unlabelled=X_unlabelled,
        X_val=X_val,
        y_val=y_val,
    )

    assert result["harvested_count"] > 0
    assert 0.0 < result["harvested_ratio"] <= 1.0
    assert "baseline_score" in result
    assert "refined_score" in result
    assert "score_lift" in result
    assert hasattr(refined_model, "predict")
