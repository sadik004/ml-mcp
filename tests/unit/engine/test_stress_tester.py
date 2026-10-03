"""Unit tests for Covariance-Aware Manifold Stress Testing."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from ml_mcp.engine.stress_tester import ModelStressTester


def test_stress_tester_covariance_noise_manifold():
    np.random.seed(42)
    n = 200
    # Correlated features
    x1 = np.random.randn(n)
    x2 = 0.9 * x1 + 0.1 * np.random.randn(n)
    X = np.column_stack([x1, x2])
    y = (x1 + x2 > 0).astype(int)

    model = LogisticRegression().fit(X, y)
    tester = ModelStressTester()
    report = tester.evaluate(model, X, y, perturbation_type="covariance_noise", noise_level=0.10)

    assert report.perturbation_type == "covariance_noise"
    assert 0.0 <= report.robustness_score <= 100.0
    assert report.degradation_percentage >= 0.0
