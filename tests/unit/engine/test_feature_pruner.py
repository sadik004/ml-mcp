"""Unit tests for OpenFE-inspired GradientFeatureSelector."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from ml_mcp.engine.feature_pruner import GradientFeatureSelector
from ml_mcp.schemas.feature import FeaturePruningReportDTO


def test_gradient_feature_selector_classification_pruning():
    assert issubclass(GradientFeatureSelector, (BaseEstimator, TransformerMixin))

    np.random.seed(42)
    n = 100
    informative = np.linspace(0, 10, n)
    noise_1 = np.random.normal(0, 1, n)
    noise_2 = np.random.normal(0, 1, n)
    noise_3 = np.random.normal(0, 1, n)

    X = pd.DataFrame({
        "informative": informative,
        "noise_1": noise_1,
        "noise_2": noise_2,
        "noise_3": noise_3,
    })
    y = (informative > 5.0).astype(int)

    selector = GradientFeatureSelector(
        top_k=2,
        task_type="classification",
        random_state=42,
    )
    X_pruned = selector.fit_transform(X, y)

    assert X_pruned.shape[1] == 2
    assert "informative" in selector.selected_features_
    assert len(selector.dropped_features_) == 2

    report = selector.get_report()
    assert isinstance(report, FeaturePruningReportDTO)
    assert report.original_feature_count == 4
    assert report.pruned_feature_count == 2
    assert "informative" in report.selected_features


def test_gradient_feature_selector_regression_threshold():
    np.random.seed(42)
    n = 80
    signal = np.linspace(-5, 5, n)
    noise = np.random.normal(0, 1, n)

    X = pd.DataFrame({
        "signal": signal,
        "noise": noise,
    })
    y = 3.5 * signal + np.random.normal(0, 0.1, n)

    selector = GradientFeatureSelector(
        importance_threshold=0.1,
        task_type="regression",
        random_state=42,
    )
    X_pruned = selector.fit_transform(X, y)

    assert "signal" in selector.selected_features_
    # At least the top feature is always retained
    assert X_pruned.shape[1] >= 1
