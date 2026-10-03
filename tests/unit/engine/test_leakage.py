"""Unit tests for Target Leakage Detector with task-aware mutual information and Bias-Corrected Cramer's V."""
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


def test_target_leakage_categorical_bias_corrected_cramers_v():
    np.random.seed(42)
    n = 250

    # Categorical target: churn or retained
    y = np.random.choice(["churned", "retained"], size=n, p=[0.3, 0.7])
    
    # Benign categorical predictor
    region = np.random.choice(["north", "south", "east", "west"], size=n)
    
    # Critical post-event categorical leakage: status reflects target directly
    leaked_status = []
    for val in y:
        if val == "churned":
            leaked_status.append(np.random.choice(["account_terminated", "refunded"], p=[0.95, 0.05]))
        else:
            leaked_status.append(np.random.choice(["active_standing", "premium"], p=[0.95, 0.05]))

    df = pd.DataFrame({
        "region": region,
        "account_closure_status": leaked_status,
        "target": y,
    })

    detector = TargetLeakageDetector(threshold_cramers_v=0.90)
    report = detector.detect_leakage(df, target_column="target", task_type="classification")

    assert report.has_critical_leakage is True
    assert "account_closure_status" in report.leaked_features
    assert "region" not in report.leaked_features
    assert report.cramers_v_scores["account_closure_status"] >= 0.85


def test_target_leakage_speed_guard_on_large_dataset():
    np.random.seed(42)
    n = 3500

    y = np.random.binomial(1, 0.5, size=n)
    x1 = np.random.normal(0, 1, size=n)
    x_leaked = y.astype(float)

    df = pd.DataFrame({"x1": x1, "x_leaked": x_leaked, "y": y})
    detector = TargetLeakageDetector(threshold_correlation=0.95)

    # Subsampling guard ensures execution finishes in < 2 seconds
    report = detector.detect_leakage(df, target_column="y", task_type="classification")
    assert report.has_critical_leakage is True
    assert "x_leaked" in report.leaked_features
