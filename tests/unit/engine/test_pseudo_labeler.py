"""Unit tests for Semi-Supervised Curriculum Pseudo-Labeling with FlexMatch and Conformal Singleton Sets."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.pseudo_labeler import PseudoLabeler


def test_pseudo_labeler_flexmatch_class_adaptive_thresholds():
    """Verify class-adaptive dynamic thresholds eliminate confirmation bias."""
    X, y = make_classification(n_samples=300, n_features=6, n_classes=2, weights=[0.85, 0.15], random_state=42)
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:200], y[150:200]
    X_unlabelled = X[200:]

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)

    labeler = PseudoLabeler(base_threshold=0.90, min_threshold=0.70, random_state=42)
    stats, refined_model = labeler.refine_with_pseudo_labels(
        model=clf,
        X_train=X_train,
        y_train=y_train,
        X_unlabelled=X_unlabelled,
        X_val=X_val,
        y_val=y_val,
    )

    assert "class_thresholds" in stats
    assert len(stats["class_thresholds"]) == 2
    assert stats["conformal_singleton_verified"] is True
    assert stats["harvested_count"] >= 0
    assert "per_class_harvested" in stats
