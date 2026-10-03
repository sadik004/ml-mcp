"""Unit tests for Decision Threshold Optimizer and Decision Curve Analysis (DCA)."""
import numpy as np
import pytest
from ml_mcp.engine.threshold import DecisionThresholdOptimizer, calculate_decision_curve_analysis


def test_decision_curve_analysis_net_benefit():
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.3, size=n)
    p = np.clip(y * 0.7 + np.random.uniform(0.0, 0.3, size=n), 0.01, 0.99)

    dca_res = calculate_decision_curve_analysis(y, p)

    assert "net_benefit" in dca_res
    assert "treat_all" in dca_res
    assert "treat_none" in dca_res
    assert "optimal_zone" in dca_res
    assert len(dca_res["net_benefit"]) == len(dca_res["thresholds"])


def test_threshold_optimizer_with_dca_integration():
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.25, size=n)
    p = np.random.uniform(0.05, 0.95, size=n)

    optimizer = DecisionThresholdOptimizer()
    report = optimizer.optimize(y, p, cost_fp=1.0, cost_fn=10.0, criterion="cost_loss")

    assert report.optimal_threshold > 0.0
    assert report.dca_net_benefit is not None
    assert report.total_cost_optimal <= report.total_cost_default
