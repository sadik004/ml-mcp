"""Unit tests for Target Leakage Detector with Chatterjee xi, Cramer's V, and PPS tree."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.leakage import TargetLeakageDetector


def test_target_leakage_classification_synthetic():
    np.random.seed(42)
    n = 200

    y = np.random.binomial(1, 0.4, size=n)
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.exponential(1.5, size=n)
    x_leaked = y + np.random.normal(0, 0.01, size=n)

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "suspicious_post_event": x_leaked,
        "churn": y,
    })

    detector = TargetLeakageDetector(threshold_correlation=0.95)
    report = detector.detect_leakage(df, target_column="churn", task_type="classification")

    assert report.has_critical_leakage is True
    assert "suspicious_post_event" in report.leaked_features
    assert "feature_1" not in report.leaked_features


def test_target_leakage_chatterjee_non_linear():
    """Verify Chatterjee xi catches perfect non-linear leakage where Pearson r is near 0.0."""
    np.random.seed(42)
    n = 300

    x = np.linspace(-5, 5, n)
    # Target is deterministic non-linear quadratic function of feature: Y = X^2
    y = x ** 2

    df = pd.DataFrame({
        "non_linear_leaker": x,
        "target": y,
    })

    detector = TargetLeakageDetector()
    xi = detector.calculate_chatterjee_correlation(df["non_linear_leaker"], df["target"])
    r = float(np.abs(np.corrcoef(x, y)[0, 1]))

    # Pearson correlation fails to detect quadratic dependence (r < 0.1)
    assert r < 0.1
    # Chatterjee xi reliably detects non-linear functional dependence (xi > 0.85)
    assert xi >= 0.85

    report = detector.detect_leakage(df, target_column="target", task_type="regression")
    assert "non_linear_leaker" in report.leaked_features
    assert report.has_critical_leakage is True


def test_target_leakage_cramers_v_categorical():
    df = pd.DataFrame({
        "status": ["Approved", "Rejected"] * 100,
        "outcome": ["Approved", "Rejected"] * 100,
    })
    detector = TargetLeakageDetector()
    v = detector.calculate_bias_corrected_cramers_v(df["status"], df["outcome"])
    assert v >= 0.95
