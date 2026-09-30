"""Unit tests for Target Leakage Detector with task-aware mutual information."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.leakage import TargetLeakageDetector


def test_target_leakage_classification_synthetic():
    np.random.seed(42)
    n = 200

    y = np.random.binomial(1, 0.4, size=n)
    # Legitimate features
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.exponential(1.5, size=n)
    # Target leakage feature: almost identical to target with tiny noise
    x_leaked = y + np.random.normal(0, 0.01, size=n)

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "suspicious_post_event": x_leaked,
        "churn": y,
    })

    detector = TargetLeakageDetector(threshold_correlation=0.95, threshold_mi=0.85)
    report = detector.detect_leakage(df, target_column="churn", task_type="classification")

    assert report.has_critical_leakage is True
    assert "suspicious_post_event" in report.leaked_features
    assert "feature_1" not in report.leaked_features
    assert report.correlation_matrix["suspicious_post_event"] > 0.95


def test_target_leakage_regression_synthetic():
    np.random.seed(42)
    n = 200

    # Continuous target (e.g. house price)
    y = np.random.uniform(100000, 500000, size=n)
    x1 = np.random.normal(50, 10, size=n)
    # Leaked feature (e.g. exact price with 1% tax)
    x_leaked = y * 1.01

    df = pd.DataFrame({
        "sqft": x1,
        "price_with_tax": x_leaked,
        "price": y,
    })

    detector = TargetLeakageDetector(threshold_correlation=0.95)
    report = detector.detect_leakage(df, target_column="price", task_type="regression")

    assert report.has_critical_leakage is True
    assert "price_with_tax" in report.leaked_features
    assert "sqft" not in report.leaked_features
