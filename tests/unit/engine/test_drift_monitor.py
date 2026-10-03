"""Unit tests for Wasserstein-1 Earth Mover's Distance and Permutation Drift Testing."""
import numpy as np
import pytest
from ml_mcp.engine.drift_monitor import DataDriftMonitor


def test_drift_monitor_wasserstein_no_drift():
    np.random.seed(42)
    ref = np.random.normal(0.0, 1.0, size=(500, 3))
    curr = np.random.normal(0.0, 1.0, size=(500, 3))

    monitor = DataDriftMonitor()
    report = monitor.detect_drift(ref, curr)

    assert report.drift_status == "no_drift"
    assert report.retraining_recommended is False
    assert report.wasserstein_distance is not None
    assert report.wasserstein_distance < 0.25


def test_drift_monitor_wasserstein_severe_drift():
    np.random.seed(42)
    ref = np.random.normal(0.0, 1.0, size=(500, 3))
    # Heavy tail shift on feature 0
    curr = np.random.normal(5.0, 3.0, size=(500, 3))

    monitor = DataDriftMonitor()
    report = monitor.detect_drift(ref, curr)

    assert report.drift_status in ("moderate_drift", "severe_drift")
    assert report.wasserstein_distance > 0.50
    assert len(report.drifted_features) > 0
