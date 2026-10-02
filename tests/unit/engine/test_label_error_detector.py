"""Unit tests for MIT Confident Learning label error detector."""
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
