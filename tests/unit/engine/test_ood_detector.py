"""Unit tests for Out-of-Distribution Detection (Energy & Mahalanobis)."""
import numpy as np
import pytest
from ml_mcp.engine.ood_detector import OODDetector, compute_free_energy


def test_compute_free_energy_scaling():
    logits_in = np.array([[10.0, 1.0], [8.0, 2.0]])
    logits_out = np.array([[0.1, 0.1], [-2.0, -1.5]])

    e_in = compute_free_energy(logits_in, temperature=1.0)
    e_out = compute_free_energy(logits_out, temperature=1.0)

    assert np.mean(e_in) < np.mean(e_out)


def test_compute_free_energy_extreme_no_overflow():
    """Verify that extreme values (e.g. 1e300) do not overflow to NaN or inf."""
    extreme_logits = np.array([[1e300, 10.0], [50.0, 1e300]])
    e = compute_free_energy(extreme_logits, temperature=0.5)
    assert not np.isnan(e).any()
    assert not np.isneginf(e).any()
    assert not np.isposinf(e).any()


def test_ood_detector_mahalanobis_scale_invariance():
    """Verify that Mahalanobis distance standardizes features so small-scale features are not ignored."""
    train_data = np.array([
        [100000.0, 20.0],
        [105000.0, 25.0],
        [95000.0, 22.0],
        [102000.0, 28.0],
        [98000.0, 24.0],
        [101000.0, 26.0],
    ] * 5)

    detector = OODDetector(method="mahalanobis", contamination=0.10)
    detector.fit(train_data)

    # In-distribution sample
    in_sample = np.array([[100000.0, 24.0]])
    rep_in = detector.detect(in_sample)

    # Out-of-distribution in age (e.g. Age 90 instead of ~25), while income is normal
    age_anomaly = np.array([[100000.0, 95.0]])
    rep_anomaly = detector.detect(age_anomaly)

    assert rep_anomaly.ood_detected_count == 1


def test_ood_detector_energy_based_detection():
    np.random.seed(42)
    # In-distribution: high magnitude logits
    logits_train = np.random.normal(5.0, 1.0, size=(200, 4))
    # OOD: low entropy noise logits
    logits_test_ood = np.random.normal(0.0, 0.5, size=(50, 4))

    detector = OODDetector(method="energy", contamination=0.10)
    detector.fit(logits_train)
    report = detector.detect(logits_test_ood)

    assert report.detector_name == "EnergyBasedOOD"
    assert report.ood_detected_count > 0
    assert report.ood_ratio > 0.50
