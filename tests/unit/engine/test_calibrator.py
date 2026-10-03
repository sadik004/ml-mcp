"""Unit tests for Probability Calibrator with Beta Calibration and Adaptive ECE."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from ml_mcp.engine.calibrator import ProbabilityCalibrator, BetaCalibrator, calculate_adaptive_ece


def test_beta_calibrator_asymmetric_fit():
    np.random.seed(42)
    n = 200
    p = np.random.beta(0.3, 0.8, size=n)
    y = (p > 0.4).astype(int)

    cal = BetaCalibrator()
    cal.fit(p, y)
    cal_probs = cal.predict_proba(p)

    assert cal_probs.shape == (n, 2)
    assert not np.isnan(cal_probs).any()
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)


def test_adaptive_ece_elimination_of_sample_bias():
    np.random.seed(42)
    n = 300
    probs = np.random.uniform(0.0, 1.0, size=(n, 2))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    y = np.random.choice([0, 1], size=n)

    aece = calculate_adaptive_ece(y, probs, n_bins=10)
    assert 0.0 <= aece <= 1.0


def test_probability_calibrator_with_beta_method():
    np.random.seed(42)
    n = 250
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    base_model = LogisticRegression()
    calibrator = ProbabilityCalibrator()
    report, cal_model = calibrator.calibrate(base_model, X, y, method="beta", cv=3)

    assert report.method == "beta"
    assert report.adaptive_ece is not None
    assert 0.0 <= report.post_brier_score <= 1.0
