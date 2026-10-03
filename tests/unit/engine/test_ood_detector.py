"""Unit tests for Energy-Based Out-of-Distribution Detection (Liu et al. NeurIPS 2020)."""
import numpy as np
import pytest
from ml_mcp.engine.ood_detector import OODDetector, compute_free_energy


def test_compute_free_energy_scaling():
    logits_in = np.array([[10.0, 1.0], [8.0, 2.0]])
    logits_out = np.array([[0.1, 0.1], [-2.0, -1.5]])

    e_in = compute_free_energy(logits_in, temperature=1.0)
    e_out = compute_free_energy(logits_out, temperature=1.0)

    # In-distribution logits have lower energy, OOD low-confidence logits have higher energy
    assert np.mean(e_in) < np.mean(e_out)


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
