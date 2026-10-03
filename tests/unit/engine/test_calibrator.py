"""Unit tests for Probability Calibrator engine."""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_mcp.engine.calibrator import ProbabilityCalibrator
from ml_mcp.schemas.tuning import CalibrationReportDTO


def test_calibrator_regression_task_skipped():
    """Verify regression tasks are safely skipped without crashing."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    calibrator = ProbabilityCalibrator()
    report, calibrated_model = calibrator.calibrate(
        model=model,
        X=X,
        y=y,
        task_type="regression",
    )

    assert isinstance(report, CalibrationReportDTO)
    assert report.status == "skipped_regression_task"
    assert calibrated_model is model


def test_calibrator_temperature_scaling_and_adaptive_ece():
    """Verify temperature scaling and debiased adaptive ECE calculation."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    base_model = RandomForestClassifier(n_estimators=15, random_state=42)

    calibrator = ProbabilityCalibrator()
    report, calibrated_model = calibrator.calibrate(
        model=base_model,
        X=X,
        y=y,
        task_type="classification",
        method="temperature",
        cv=3,
    )

    assert isinstance(report, CalibrationReportDTO)
    assert report.method == "temperature"
    assert report.temperature is not None
    assert report.adaptive_ece is not None
    assert report.conformal_coverage is not None
    assert 0.80 <= report.conformal_coverage <= 1.0
    assert hasattr(calibrated_model, "predict_proba")


def test_calibrator_multiclass_isotonic_and_simplex_normalization():
    """Verify Isotonic calibration strictly satisfies sum-to-one simplex probability."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_classes=3,
        n_informative=6,
        random_state=42,
    )
    base_model = RandomForestClassifier(n_estimators=15, random_state=42)

    calibrator = ProbabilityCalibrator()
    report, calibrated_model = calibrator.calibrate(
        model=base_model,
        X=X,
        y=y,
        task_type="classification",
        method="isotonic",
        cv=3,
    )

    assert isinstance(report, CalibrationReportDTO)
    assert report.method == "isotonic"
    assert report.adaptive_ece is not None

    # Verify probability simplex constraint
    probs = calibrated_model.predict_proba(X[:20])
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)
