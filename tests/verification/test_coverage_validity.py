"""Statistical Completeness & Conformal Coverage Validity Tests (P6 & Gate G6).

Theoretical Guarantees Tested:
1. Wilson Score Confidence Intervals (Wilson, 1927) for coverage uncertainty bounded in [0, 1].
2. Min-n Refusal: Refusal to claim coverage when n < Settings.min_calibration_n, returning INSUFFICIENT_N.
3. Marginal Conformal Coverage Guarantee (Vovk et al. 2005):
   E[Coverage] >= 1 - alpha across multi-seed calibration holdout splits.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from ml_mcp.config import get_settings
from ml_mcp.engine.calibrator import ProbabilityCalibrator
from ml_mcp.engine.conformal_risk_control import ConformalRiskControlEngine
from ml_mcp.engine.safety_orchestrator import SafetyOrchestrator
from ml_mcp.engine.stats import wilson_interval
from ml_mcp.schemas.safety import CRCReportDTO, SafetyCertificateReportDTO
from ml_mcp.schemas.tuning import CalibrationReportDTO
from ml_mcp.services.safety_service import SafetyService


# ------------------------------------------------------------------------------
# 1. Wilson Confidence Interval Mathematical Properties
# ------------------------------------------------------------------------------
def test_wilson_interval_mathematical_invariants():
    """Wilson interval must strictly adhere to mathematical bounds and handle degenerate cases."""
    # Degenerate cases
    assert wilson_interval(0, 0) == (None, None)
    assert wilson_interval(5, -1) == (None, None)

    # Min-n enforcement
    assert wilson_interval(10, 20, min_n=40) == (None, None)
    low_valid, high_valid = wilson_interval(20, 50, min_n=40)
    assert low_valid is not None and high_valid is not None

    # Normal case: k=90, n=100 (90% success)
    low, high = wilson_interval(90, 100, conf=0.95)
    assert low is not None and high is not None
    assert 0.0 <= low <= 0.90 <= high <= 1.0
    assert abs(low - 0.825) < 0.05
    assert abs(high - 0.948) < 0.05

    # Boundary cases: k=0 and k=n
    low_0, high_0 = wilson_interval(0, 50, conf=0.95)
    assert low_0 == 0.0
    assert high_0 is not None and 0.0 < high_0 < 0.15

    low_n, high_n = wilson_interval(50, 50, conf=0.95)
    assert high_n == 1.0
    assert low_n is not None and 0.85 < low_n < 1.0


# ------------------------------------------------------------------------------
# 2. Min-n Refusal & INSUFFICIENT_N Warning Enforcement
# ------------------------------------------------------------------------------
def test_calibrator_min_n_refusal():
    """Calibrator must refuse to claim conformal coverage when n < min_calibration_n."""
    settings = get_settings()
    min_n = settings.min_calibration_n

    # Create small dataset with n = 15 < min_n (20)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    X_small = np.random.randn(15, 3)
    y_small = np.random.choice([0, 1], size=15)
    clf.fit(X_small, y_small)

    calibrator = ProbabilityCalibrator()
    report, _ = calibrator.calibrate(clf, X_small, y_small, method="sigmoid")

    assert report.conformal_coverage is None
    assert report.coverage_ci_low is None
    assert report.coverage_ci_high is None
    assert any("INSUFFICIENT_N" in w for w in report.warnings)


def test_crc_service_min_n_refusal(tmp_path):
    """Conformal Risk Control must refuse risk guarantee when calibration n < min_calibration_n."""
    settings = get_settings()
    min_n = settings.min_calibration_n

    # CSV with small sample size
    n = 20
    df = pd.DataFrame(np.random.randn(n, 3), columns=["f1", "f2", "f3"])
    df["target"] = np.random.choice([0, 1], size=n)
    csv_file = tmp_path / "small_cal.csv"
    df.to_csv(csv_file, index=False)

    service = SafetyService()
    res = service.conformal_risk_control(
        csv_path=str(csv_file),
        target_column="target",
        cal_fraction=0.5,  # 10 calibration samples < min_n
        target_risk=0.05,
    )

    assert res["empirical_risk"] is None
    assert res["coverage_ci_low"] is None
    assert res["coverage_ci_high"] is None
    assert any("INSUFFICIENT_N" in w for w in res["warnings"])


# ------------------------------------------------------------------------------
# 3. Presence of Wilson CI Bounds on Adequate Data
# ------------------------------------------------------------------------------
def test_safety_certificate_attaches_wilson_ci():
    """Safety Certificate on sufficient data must populate coverage_ci_low and coverage_ci_high."""
    settings = get_settings()
    from sklearn.ensemble import RandomForestClassifier

    n = 100
    np.random.seed(settings.random_state)
    X = pd.DataFrame(np.random.randn(n, 4), columns=[f"feat_{i}" for i in range(4)])
    y = pd.Series((X["feat_0"] + X["feat_1"] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=10, random_state=settings.random_state)
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    report = orchestrator.certify_model(
        model=clf,
        X=X,
        y=y,
        cost_fp=5.0,
        cost_fn=50.0,
        allow_in_sample_diagnostic=True,
    )

    assert report.conformal_coverage_pct is not None
    assert report.coverage_ci_low is not None
    assert report.coverage_ci_high is not None
    assert 0.0 <= report.coverage_ci_low <= report.coverage_ci_high <= 1.0


# ------------------------------------------------------------------------------
# 4. Multi-Seed Conformal Coverage Validity (200 Seeds)
# ------------------------------------------------------------------------------
def test_multi_seed_conformal_coverage_validity():
    """Empirical validation of marginal coverage guarantee: E[Coverage] >= 1 - alpha.

    Simulates 200 random splits with nominal 1 - alpha = 0.90.
    Asserts the mean realized coverage over 200 seeds is >= 0.90 - tolerance.
    """
    nominal_coverage = 0.90
    alpha = 1.0 - nominal_coverage
    n_seeds = 200
    n_cal = 100
    n_val = 150

    realized_coverages: List[float] = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        # Generate synthetic binary probabilities with calibration noise
        # Ground truth class probabilities
        true_prob_cal = rng.beta(2, 2, size=n_cal)
        y_cal = (rng.rand(n_cal) < true_prob_cal).astype(int)
        prob_cal_pos = np.clip(true_prob_cal + rng.normal(0, 0.05, size=n_cal), 0.01, 0.99)
        p_cal_2d = np.column_stack([1.0 - prob_cal_pos, prob_cal_pos])

        true_prob_val = rng.beta(2, 2, size=n_val)
        y_val = (rng.rand(n_val) < true_prob_val).astype(int)
        prob_val_pos = np.clip(true_prob_val + rng.normal(0, 0.05, size=n_val), 0.01, 0.99)
        p_val_2d = np.column_stack([1.0 - prob_val_pos, prob_val_pos])

        # CRC calibration
        engine = ConformalRiskControlEngine()
        lambda_val, _ = engine.calibrate(
            probs_cal=p_cal_2d,
            y_cal=y_cal,
            loss_type="misclassification",
            target_risk=alpha,
        )

        # Holdout prediction sets and coverage
        psets = engine.get_prediction_sets(p_val_2d, lambda_val=lambda_val)
        losses = engine.evaluate_loss(psets, y_val, loss_type="misclassification")
        coverage = float(1.0 - np.mean(losses))
        realized_coverages.append(coverage)

    mean_coverage = float(np.mean(realized_coverages))

    # Standard binomial standard error over 200 seeds x 150 samples is < 0.01
    # Vovk et al. marginal coverage theorem guarantees E[coverage] >= 1 - alpha
    assert mean_coverage >= nominal_coverage - 0.02, (
        f"Conformal Coverage Failure: Mean empirical coverage over {n_seeds} seeds "
        f"was {mean_coverage:.4f}, falling below nominal bound {nominal_coverage - 0.02:.4f}"
    )
    # Ensure it's not degenerate trivial coverage (e.g. not 1.0 everywhere)
    assert mean_coverage <= 0.98, (
        f"Conformal Coverage Overly Conservative: Mean coverage was {mean_coverage:.4f}"
    )
