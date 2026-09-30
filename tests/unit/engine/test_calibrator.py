"""Unit tests for Probability Calibrator engine."""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

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


def test_calibrator_binary_classification_platt_and_ece():
    """Verify Platt scaling (sigmoid) on binary classification and ECE calculation."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    # Uncalibrated baseline
    base_model = RandomForestClassifier(n_estimators=15, random_state=42)

    calibrator = ProbabilityCalibrator()
    report, calibrated_model = calibrator.calibrate(
        model=base_model,
        X=X,
        y=y,
        task_type="classification",
        method="sigmoid",
        cv=3,
    )

    assert isinstance(report, CalibrationReportDTO)
    assert report.method == "sigmoid"
    assert report.pre_brier_score >= 0.0
    assert report.post_brier_score >= 0.0
    assert report.pre_ece is not None
    assert report.post_ece is not None
    assert hasattr(calibrated_model, "predict_proba")


def test_calibrator_multiclass_isotonic():
    """Verify Isotonic calibration on multi-class and macro ECE calculation."""
    X, y = make_classification(
        n_samples=500,
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
    assert report.pre_ece is not None
    assert report.post_ece is not None
    assert hasattr(calibrated_model, "predict_proba")
