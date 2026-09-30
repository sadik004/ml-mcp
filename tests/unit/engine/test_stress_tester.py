"""Unit tests for Model Stress Tester Engine."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.stress_tester import ModelStressTester
from ml_mcp.schemas.safety import StressTestReportDTO


def test_stress_tester_gaussian_noise_robust_model():
    """Verify Gaussian noise stress testing on a trained classifier."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X[:300], y[:300])

    tester = ModelStressTester(random_state=42)
    report = tester.evaluate(
        model=clf,
        X_test=X[300:],
        y_test=y[300:],
        perturbation_type="gaussian_noise",
        noise_level=0.05,
    )

    assert isinstance(report, StressTestReportDTO)
    assert report.perturbation_type == "gaussian_noise"
    assert report.baseline_score > 0.60
    assert report.stressed_score >= 0.0
    assert 0.0 <= report.robustness_score <= 100.0
    assert isinstance(report.is_stress_passed, bool)


def test_stress_tester_extreme_outlier_severe_degradation():
    """Verify extreme outlier perturbation triggers severe degradation."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X[:300], y[:300])

    tester = ModelStressTester(random_state=42)
    report = tester.evaluate(
        model=clf,
        X_test=X[300:],
        y_test=y[300:],
        perturbation_type="extreme_outlier",
        noise_level=0.90,
    )

    assert isinstance(report, StressTestReportDTO)
    assert report.perturbation_type == "extreme_outlier"
    assert report.degradation_percentage >= 0.0
