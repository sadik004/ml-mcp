"""RED Verification Suite: Probability Calibration & Conformal Reliability (C3, C4, H3).

Enforces Rule T3 (independent sklearn/numpy oracles) and Rule T4 (no answer-faking mocks).
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ml_mcp.engine.calibrator import (
    BetaCalibrator,
    ProbabilityCalibrator,
    TemperatureScaler,
    calculate_adaptive_ece,
    calculate_ece,
)


def _reference_ece(y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """Exact independent mathematical oracle for Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probas >= low) & (probas < high) if i < n_bins - 1 else (probas >= low) & (probas <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probas[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def test_ece_matches_reference_binning() -> None:
    """Guard test: calculate_ece must match independent true equal-width binning formula."""
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.65, 0.85, 0.95])
    computed = calculate_ece(y, p, n_bins=5)
    expected = _reference_ece(y, p, n_bins=5)
    assert abs(computed - expected) < 1e-12


def test_holdout_branch_lengths_consistent() -> None:
    """C3: When n < 10 (e.g. n=9), holdout fallback branch must not crash with inconsistent sample shapes."""
    X = np.random.randn(9, 4)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    base_clf = LogisticRegression()
    base_clf.fit(X, y)

    cal = ProbabilityCalibrator()
    # Currently crashes with ValueError: Found input variables with inconsistent numbers of samples: [9, 5]
    report, calibrated_model = cal.calibrate(
        base_clf,
        X,
        y,
        method="temperature",
    )
    assert report is not None
    assert len(report.y_eval) == len(report.post_probas)


def test_labels_one_two_n400() -> None:
    """C4: Labels {1, 2} at n=400 must not trigger min=0 in bincount and crash."""
    X, y_raw = make_classification(n_samples=400, n_features=5, random_state=42)
    y = y_raw + 1  # classes are {1, 2}
    base_clf = LogisticRegression()
    base_clf.fit(X, y)

    cal = ProbabilityCalibrator()
    # Currently crashes with ValueError: Found input variables with inconsistent numbers of samples: [400, 200]
    report, calibrated_model = cal.calibrate(
        base_clf,
        X,
        y,
        method="temperature",
    )
    assert report is not None
    preds = calibrated_model.predict(X)
    assert set(preds).issubset({1, 2})


def test_string_labels() -> None:
    """C4: String labels e.g. {'no', 'yes'} must not cause TypeError in np.bincount."""
    X, y_raw = make_classification(n_samples=200, n_features=5, random_state=42)
    y = np.where(y_raw == 1, "yes", "no")
    base_clf = RandomForestClassifier(n_estimators=5, random_state=42)
    base_clf.fit(X, y)

    cal = ProbabilityCalibrator()
    # Currently raises TypeError: Cannot cast array data from dtype('<U3') to dtype('int64')
    report, calibrated_model = cal.calibrate(
        base_clf,
        X,
        y,
        method="temperature",
    )
    assert report is not None
    preds = calibrated_model.predict(X)
    assert set(preds).issubset({"no", "yes"})
    assert list(calibrated_model.classes_) == ["no", "yes"]


def test_fallback_mode_is_not_out_of_fold() -> None:
    """H3: When cross_val_predict fails and calibrator uses in-sample probas, evaluation_mode must NOT claim 'out_of_fold'."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    base_clf = LogisticRegression()
    base_clf.fit(X, y)

    cal = ProbabilityCalibrator()
    with patch("ml_mcp.engine.calibrator.cross_val_predict", side_effect=RuntimeError("CV Failed")):
        report, _ = cal.calibrate(
            base_clf,
            X,
            y,
            method="temperature",
        )
        assert report.evaluation_mode != "out_of_fold", "Hardcoded out_of_fold mode returned during in-sample fallback!"
        assert any("in_sample" in w.lower() or "fallback" in w.lower() for w in report.warnings)


def test_calibrators_unfitted_before_fit() -> None:
    """Hygiene: TemperatureScaler & BetaCalibrator must not set fitted attributes in __init__."""
    scaler = TemperatureScaler(base_estimator=LogisticRegression())
    with pytest.raises(NotFittedError) as exc_info_s:
        check_is_fitted(scaler, attributes=["temperature_"])
    assert "not fitted" in str(exc_info_s.value).lower()

    beta = BetaCalibrator(base_estimator=LogisticRegression())
    with pytest.raises(NotFittedError) as exc_info_b:
        check_is_fitted(beta, attributes=["lr_"])
    assert "not fitted" in str(exc_info_b.value).lower()




def test_adaptive_ece_docstring_no_debiased_claim() -> None:
    """Doc integrity: calculate_adaptive_ece docstring must not falsely claim 'Debiased (Roelofs 2022)'."""
    doc = inspect.getdoc(calculate_adaptive_ece) or ""
    assert "debiased" not in doc.lower(), "Docstring claims debiased calibration without debiasing implementation"
