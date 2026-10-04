"""RED Verification Suite: Conformal Prediction & Risk Control Rigor (H5, H6, H7, H8, Quantile).

Enforces Rule T3 (independent statistical oracles) and Rule T6 (declared mathematical bounds).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.calibrator import ProbabilityCalibrator
from ml_mcp.engine.conformal_risk_control import ConformalRiskControlEngine


def test_non_monotone_asymmetric_cost_rejected() -> None:
    """H5: Asymmetric cost function where expanding sets increases loss violates CRC monotonicity; must be rejected."""
    engine = ConformalRiskControlEngine()
    # Cost matrix where singleton correct = 0, but set size > 1 has cost 0.2
    # Expanding set increases loss, violating monotonicity L(C) <= L(C') for C' subset of C.
    with pytest.raises(ValueError) as exc_info:
        engine.evaluate_loss(
            prediction_sets=[[0, 1]],
            y_true=np.array([0]),
            loss_type="asymmetric_cost",
        )
    assert "monotone" in str(exc_info.value).lower()



def test_monotone_loss_property() -> None:
    """H5: For valid monotone losses, increasing lambda must non-increase sample loss across the grid."""
    engine = ConformalRiskControlEngine()
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 1])

    # Check loss across lambda grid for misclassification loss
    grid = np.linspace(0.0, 1.0, 50)
    loss_history = []
    for lam in grid:
        psets = engine.get_prediction_sets(probs, float(lam))
        l_vals = engine.evaluate_loss(psets, y, loss_type="misclassification")
        loss_history.append(l_vals)

    L = np.array(loss_history)  # shape (50, 2)
    # Increasing lambda (larger sets) must have diff <= 0 for every sample
    assert np.all(np.diff(L, axis=0) <= 1e-12), "Loss is not monotone non-increasing in lambda!"


def test_infeasible_mondrian_flagged_with_warning() -> None:
    """H7: When n_c < 1/alpha - 1, finite-sample guarantee is mathematically impossible; must report INFEASIBLE."""
    engine = ConformalRiskControlEngine()
    # Target alpha = 0.05 requires n_c >= 19. If n_c = 10, it is mathematically infeasible.
    probs = np.tile([0.9, 0.1], (10, 1))
    y = np.zeros(10, dtype=int)

    # Currently silently returns lambda = 1.0 without raising or warning
    res, emp_risk = engine.calibrate(
        probs_cal=probs,
        y_cal=y,
        loss_type="misclassification",
        target_risk=0.05,
        mondrian=True,
    )
    assert hasattr(engine, "warnings_") or hasattr(engine, "status_"), "Engine must track feasibility status/warnings"
    assert getattr(engine, "status_", "") == "infeasible" or any("INFEASIBLE" in w for w in getattr(engine, "warnings_", []))


def test_mondrian_lambda_is_none_or_explicit() -> None:
    """H8: For Mondrian calibration, single calibrated_lambda is invalid and must be None or per-class dict."""
    engine = ConformalRiskControlEngine()
    probs = np.array([[0.9, 0.1]] * 30 + [[0.1, 0.9]] * 30)
    y = np.array([0] * 30 + [1] * 30)

    lambdas, emp_risk = engine.calibrate(
        probs_cal=probs,
        y_cal=y,
        loss_type="misclassification",
        target_risk=0.1,
        mondrian=True,
    )
    assert isinstance(lambdas, dict), "Mondrian calibration must return per-class dictionary of lambdas"
    assert len(lambdas) == 2


def test_split_conformal_quantile_higher() -> None:
    """Quantile: Split-conformal prediction quantile must use method='higher' for exact coverage guarantee."""
    X, y = make_classification(n_samples=99, n_features=4, random_state=42)
    base = LogisticRegression().fit(X, y)
    cal = ProbabilityCalibrator()
    report, _ = cal.calibrate(base, X, y, method="temperature")

    # In split conformal: level = ceil((n + 1) * (1 - alpha)) / n
    # np.quantile must use method="higher", not default linear interpolation
    assert hasattr(report, "quantile_method_"), "Calibration report must specify quantile_method_ used"
    assert report.quantile_method_ == "higher"


def test_quantile_level_above_one_is_inf() -> None:
    """Quantile: When n is too small such that ceil((n+1)(1-alpha))/n > 1, q_hat must be inf with warning."""
    X = np.random.randn(5, 4)
    y = np.array([0, 1, 0, 1, 0])
    base = LogisticRegression().fit(X, y)
    cal = ProbabilityCalibrator()
    # At n=5, alpha=0.05: (6 * 0.95)/5 = 5.7/5 = 1.14 > 1.0!
    report, _ = cal.calibrate(base, X, y, method="temperature", conformal_alpha=0.05)

    assert report.q_hat == float("inf"), f"At n=5 and alpha=0.05 coverage cannot be guaranteed, q_hat must be inf, got {report.q_hat}"
    assert any("cannot be guaranteed" in w.lower() or "inf" in w.lower() for w in report.warnings)


def test_raps_named_honestly() -> None:
    """Hygiene: Non-randomized adaptive prediction sets without u randomization must be documented as 'raps_like'."""
    engine = ConformalRiskControlEngine()
    import inspect
    doc = inspect.getdoc(engine.get_prediction_sets) or ""
    # Docstring or method should acknowledge non-randomized / RAPS-like variant
    assert "raps-like" in doc.lower() or hasattr(engine, "get_raps_like_sets"), "Must designate non-randomized sets as RAPS-like"
