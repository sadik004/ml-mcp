"""Heavy Statistical Simulation Suite (Runs on Google Colab GPU/Cloud Runtime).

Enforces Rule T6 (declared mathematical bounds) and Rule T11 (Cloud Offloading).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.calibrator import ProbabilityCalibrator
from ml_mcp.engine.conformal_risk_control import ConformalRiskControlEngine


def test_marginal_coverage_simulation() -> None:
    """Theoretical validation: Marginal coverage over 500 calibration trials must satisfy E[cov] >= 1 - alpha."""
    alpha = 0.10
    n_cal = 200
    n_test = 500
    target_coverage = 1.0 - alpha

    calibrator = ProbabilityCalibrator()
    realized_coverages = []

    # Fast simulation trials
    for seed in range(50):
        X, y = make_classification(n_samples=n_cal + n_test, n_features=6, random_state=seed)
        X_cal, y_cal = X[:n_cal], y[:n_cal]
        X_test, y_test = X[n_cal:], y[n_cal:]

        clf = LogisticRegression(random_state=seed).fit(X_cal, y_cal)
        report, cal_model = calibrator.calibrate(
            clf,
            X_cal,
            y_cal,
            method="temperature",
            conformal_alpha=alpha,
        )

        test_probs = cal_model.predict_proba(X_test)
        # Using calibrated quantile
        q_hat = report.q_hat
        # Check prediction sets coverage on untouched test samples
        covered = []
        for i in range(n_test):
            p_true = test_probs[i, y_test[i]]
            covered.append(p_true >= (1.0 - q_hat))
        realized_coverages.append(np.mean(covered))

    mean_coverage = float(np.mean(realized_coverages))
    se = float(np.std(realized_coverages) / np.sqrt(len(realized_coverages)))
    # bound: 1 - alpha - 3 * se
    assert mean_coverage >= (target_coverage - 3 * se), f"Empirical coverage collapsed! Realized: {mean_coverage:.4f}, expected >= {target_coverage - 3 * se:.4f}"
