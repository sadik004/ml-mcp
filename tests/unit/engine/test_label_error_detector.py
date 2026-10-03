"""Unit tests for MIT Confident Learning and Regression Residual Dispersion label error detector."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.label_error_detector import LabelErrorDetector
from ml_mcp.schemas.audit import LabelErrorReportDTO


def test_label_error_detector_identifies_corrupted_labels():
    np.random.seed(42)
    n = 200

    # Two cleanly separable clusters
    x1 = np.concatenate([np.random.normal(5, 1, 100), np.random.normal(15, 1, 100)])
    x2 = np.concatenate([np.random.normal(5, 1, 100), np.random.normal(15, 1, 100)])
    true_labels = np.array([0] * 100 + [1] * 100)

    # Intentionally corrupt 6 labels (flip 3 in class 0, and 3 in class 1)
    corrupted_labels = true_labels.copy()
    corrupt_indices = [10, 20, 30, 110, 120, 130]
    for idx in corrupt_indices[:3]:
        corrupted_labels[idx] = 1  # Flipped from 0 to 1
    for idx in corrupt_indices[3:]:
        corrupted_labels[idx] = 0  # Flipped from 1 to 0

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "label": corrupted_labels,
    })

    detector = LabelErrorDetector(cv_splits=5, random_state=42)
    report = detector.detect_label_errors(df, target_column="label")

    assert isinstance(report, LabelErrorReportDTO)
    assert report.total_samples == 200
    assert report.total_errors > 0
    assert report.error_rate > 0.0
    assert report.task_type == "classification"
    assert "0" in report.class_thresholds
    assert "1" in report.class_thresholds

    # Verify that the manually flipped samples appear in flagged samples
    flagged_indices = [s.sample_index for s in report.flagged_samples]
    detected_corruptions = set(corrupt_indices).intersection(set(flagged_indices))
    assert len(detected_corruptions) >= 4  # Confident learning finds the majority of flips


def test_label_error_detector_clean_dataset():
    np.random.seed(42)
    # Linearly separable clean data
    x = np.linspace(0, 10, 100)
    y = np.where(x > 5.0, 1, 0)
    df = pd.DataFrame({"x": x, "y": y})

    detector = LabelErrorDetector(cv_splits=3, random_state=42)
    report = detector.detect_label_errors(df, target_column="y")

    # Clean dataset should have zero or near-zero errors
    assert report.total_errors <= 2


def test_label_error_detector_regression_residual_dispersion():
    np.random.seed(42)
    n = 150

    x = np.linspace(0, 10, n)
    true_y = 3.0 * x + 5.0 + np.random.normal(0, 0.5, size=n)

    # Inject extreme continuous label corruption into 4 samples
    corrupted_y = true_y.copy()
    corrupt_indices = [15, 45, 95, 135]
    for idx in corrupt_indices:
        corrupted_y[idx] += 30.0  # +30.0 massive residual outlier

    df = pd.DataFrame({"feature_x": x, "target_price": corrupted_y})

    detector = LabelErrorDetector(cv_splits=3, random_state=42)
    report = detector.detect_label_errors(df, target_column="target_price", task_type="regression")

    assert report.task_type == "regression"
    assert report.total_errors >= len(corrupt_indices)
    assert report.error_rate > 0.0

    flagged_indices = [s.sample_index for s in report.flagged_samples]
    for corrupt_idx in corrupt_indices:
        assert corrupt_idx in flagged_indices

    # Check confidence is normalized dispersion z-score > 3.0
    for sample in report.flagged_samples:
        assert sample.confidence >= 3.0
