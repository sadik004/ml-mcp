"""Unit tests for Conformal Risk Control (CRC) Engine with Mondrian and RAPS."""
import numpy as np
import pytest
from ml_mcp.engine.conformal_risk_control import ConformalRiskControlEngine


@pytest.fixture
def synthetic_multiclass_data():
    np.random.seed(42)
    n = 300
    K = 4
    logits = np.random.randn(n, K)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    y = np.random.choice(K, size=n, p=[0.5, 0.25, 0.15, 0.1])
    return probs, y


@pytest.fixture
def synthetic_binary_data():
    np.random.seed(42)
    n = 200
    p1 = np.random.beta(0.5, 0.5, size=n)
    probs = np.column_stack([1.0 - p1, p1])
    y = np.random.binomial(1, p1)
    return probs, y


def test_misclassification_risk_control(synthetic_multiclass_data):
    probs, y = synthetic_multiclass_data
    engine = ConformalRiskControlEngine()
    alpha = 0.10
    lam, emp_risk = engine.calibrate(probs, y, loss_type="misclassification", target_risk=alpha)

    assert isinstance(lam, float)
    assert 0.0 <= lam <= 1.0
    assert emp_risk <= alpha + 0.02


def test_mondrian_group_conditional_risk_control(synthetic_multiclass_data):
    probs, y = synthetic_multiclass_data
    engine = ConformalRiskControlEngine()
    alpha = 0.10
    per_class_lam, emp_risk = engine.calibrate(
        probs, y, loss_type="misclassification", target_risk=alpha, mondrian=True
    )

    assert isinstance(per_class_lam, dict)
    assert len(per_class_lam) == len(np.unique(y))
    for c, lam in per_class_lam.items():
        assert 0.0 <= lam <= 1.0

    # Verify per-class coverage guarantee P(Y in C(X) | Y = k) >= 1 - alpha
    psets = engine.get_prediction_sets(probs, per_class_lam)
    for c in np.unique(y):
        mask = (y == c)
        class_losses = [1.0 if c not in psets[i] else 0.0 for i in np.where(mask)[0]]
        class_error = np.mean(class_losses)
        assert class_error <= alpha + 0.05


def test_raps_regularized_adaptive_prediction_sets(synthetic_multiclass_data):
    probs, y = synthetic_multiclass_data
    engine = ConformalRiskControlEngine()
    alpha = 0.15
    lam, emp_risk = engine.calibrate(probs, y, loss_type="misclassification", target_risk=alpha, use_raps=True)
    psets = engine.get_prediction_sets(probs, lam, use_raps=True)

    assert len(psets) == len(y)
    set_sizes = [len(s) for s in psets]
    assert np.mean(set_sizes) >= 1.0
    assert max(set_sizes) <= 4


def test_false_negative_rate_control(synthetic_binary_data):
    probs, y = synthetic_binary_data
    engine = ConformalRiskControlEngine()
    alpha = 0.05
    lam, emp_risk = engine.calibrate(probs, y, loss_type="fnr", target_risk=alpha)

    assert isinstance(lam, float)
    assert emp_risk <= alpha + 0.02


def test_human_triage_escalation(synthetic_binary_data):
    probs, y = synthetic_binary_data
    engine = ConformalRiskControlEngine()
    psets, records = engine.predict_and_triage(probs, lambda_val=0.5)

    assert len(records) == len(y)
    assert "needs_human_review" in records[0]
