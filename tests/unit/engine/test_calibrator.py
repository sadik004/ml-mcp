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


def test_regression_calibration_uses_out_of_fold_preventing_insample_leakage():
    """Verify that calibration does not evaluate or fit on overfitted in-sample predictions."""
    from sklearn.tree import DecisionTreeClassifier

    np.random.seed(42)
    n = 200
    # Pure random noise features - unpruned decision tree will achieve 100% in-sample accuracy
    X = np.random.randn(n, 10)
    y = np.random.choice([0, 1], size=n)

    overfitting_tree = DecisionTreeClassifier(random_state=42)
    overfitting_tree.fit(X, y)
    insample_brier = np.mean((y - overfitting_tree.predict_proba(X)[:, 1]) ** 2)
    assert insample_brier == 0.0, "Decision tree must have 0.0 in-sample brier score"

    calibrator = ProbabilityCalibrator(random_state=42)
    report, cal_model = calibrator.calibrate(DecisionTreeClassifier(random_state=42), X, y, method="beta", cv=3)

    # If evaluated out-of-fold, pre_brier_score on random noise will be ~0.3-0.5, NOT 0.0
    assert report.pre_brier_score > 0.20, f"Expected out-of-fold brier > 0.20, got {report.pre_brier_score}"
    assert report.post_brier_score > 0.10, f"Expected honest post brier > 0.10, got {report.post_brier_score}"


def test_multiclass_dirichlet_calibration_oof():
    """Verify multiclass Dirichlet calibration uses out-of-fold probas."""
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    n = 150
    X = np.random.randn(n, 4)
    y = np.random.choice([0, 1, 2], size=n)

    calibrator = ProbabilityCalibrator(random_state=42)
    report, cal_model = calibrator.calibrate(RandomForestClassifier(n_estimators=10, random_state=42), X, y, method="dirichlet", cv=3)

    assert report.method == "dirichlet"
    assert report.pre_brier_score > 0.0
    assert report.post_brier_score > 0.0
    assert cal_model is not None


def test_probability_calibrator_isotonic_and_sigmoid():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = (X[:, 0] > 0).astype(int)

    base = LogisticRegression()
    calibrator = ProbabilityCalibrator(random_state=42)
    rep_iso, m_iso = calibrator.calibrate(base, X, y, method="isotonic", cv=3)
    assert rep_iso.method == "isotonic"
    assert rep_iso.post_brier_score is not None

    rep_sig, m_sig = calibrator.calibrate(base, X, y, method="sigmoid", cv=3)
    assert rep_sig.method == "sigmoid"
    assert rep_sig.post_brier_score is not None


def test_probability_calibrator_temperature_scaler():
    np.random.seed(42)
    n = 150
    X = np.random.randn(n, 4)
    y = np.random.choice([0, 1, 2], size=n)

    base = LogisticRegression(max_iter=200)
    calibrator = ProbabilityCalibrator(random_state=42)
    rep_temp, m_temp = calibrator.calibrate(base, X, y, method="temperature", cv=3)
    assert rep_temp.method == "temperature"
    assert rep_temp.temperature is not None


def test_probability_calibrator_skips_regression_task():
    calibrator = ProbabilityCalibrator()
    base = LogisticRegression()
    rep, m = calibrator.calibrate(base, np.zeros((10, 2)), np.zeros(10), task_type="regression")
    assert rep.status == "skipped_regression_task"
    assert rep.method == "none"


