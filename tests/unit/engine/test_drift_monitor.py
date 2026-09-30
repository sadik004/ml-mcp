"""Unit tests for Population Stability Index (PSI) and Drift Monitor."""
import numpy as np
import pytest

from ml_mcp.engine.drift_monitor import DataDriftMonitor
from ml_mcp.schemas.serving import DataDriftReportDTO


def test_drift_monitor_no_drift():
    """Verify monitor reports no_drift when datasets come from the same distribution."""
    np.random.seed(42)
    reference = np.random.normal(loc=10.0, scale=2.0, size=(500, 3))
    current = np.random.normal(loc=10.0, scale=2.0, size=(500, 3))

    monitor = DataDriftMonitor()
    report = monitor.detect_drift(
        reference_data=reference,
        current_data=current,
        feature_names=["f1", "f2", "f3"],
    )

    assert isinstance(report, DataDriftReportDTO)
    assert report.psi_score < 0.10
    assert report.drift_status == "no_drift"
    assert report.retraining_recommended is False


def test_drift_monitor_severe_drift():
    """Verify monitor catches severe distribution shift and recommends retraining."""
    np.random.seed(42)
    reference = np.random.normal(loc=10.0, scale=2.0, size=(500, 3))
    # Substantial distribution shift
    current = np.random.normal(loc=25.0, scale=8.0, size=(500, 3))

    monitor = DataDriftMonitor()
    report = monitor.detect_drift(
        reference_data=reference,
        current_data=current,
        feature_names=["f1", "f2", "f3"],
    )

    assert isinstance(report, DataDriftReportDTO)
    assert report.psi_score > 0.25
    assert report.drift_status == "severe_drift"
    assert report.retraining_recommended is True
    assert len(report.drifted_features) > 0
