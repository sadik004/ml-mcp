"""Unit tests for Conformal Risk Control (CRC) Engine."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_mcp.engine.conformal_risk_control import ConformalRiskControlEngine


@pytest.fixture
def synthetic_binary_data():
    """Generates synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=1200,
        n_features=10,
        n_informative=6,
        n_classes=2,
        weights=[0.8, 0.2],  # imbalanced positive class
        random_state=42,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)

    probs_cal = clf.predict_proba(X_cal)
    probs_test = clf.predict_proba(X_test)

    return probs_cal, y_cal, probs_test, y_test


@pytest.fixture
def synthetic_multiclass_data():
    """Generates synthetic 3-class classification dataset."""
    X, y = make_classification(
        n_samples=1500,
        n_features=12,
        n_informative=8,
        n_classes=3,
        random_state=42,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)

    probs_cal = clf.predict_proba(X_cal)
    probs_test = clf.predict_proba(X_test)

    return probs_cal, y_cal, probs_test, y_test


def test_misclassification_risk_control(synthetic_multiclass_data):
    """Verify that empirical 0-1 test risk satisfies E[loss] <= alpha."""
    probs_cal, y_cal, probs_test, y_test = synthetic_multiclass_data
    alpha = 0.10  # 90% coverage guarantee

    crc = ConformalRiskControlEngine()
    calibrated_lambda, cal_risk = crc.calibrate(
        probs_cal=probs_cal,
        y_cal=y_cal,
        loss_type="misclassification",
        target_risk=alpha,
    )

    psets_test = crc.get_prediction_sets(probs_test, calibrated_lambda)
    test_losses = crc.evaluate_loss(psets_test, y_test, loss_type="misclassification")
    empirical_test_risk = float(np.mean(test_losses))

    # Empirical risk should satisfy alpha + small finite sample variance bound
    assert empirical_test_risk <= alpha + 0.03
    # Prediction sets should not be trivially degenerate
    avg_size = np.mean([len(s) for s in psets_test])
    assert 1.0 <= avg_size <= 3.0


def test_false_negative_rate_control(synthetic_binary_data):
    """Verify that False Negative Rate (FNR) on positive class satisfies E[FNR] <= alpha."""
    probs_cal, y_cal, probs_test, y_test = synthetic_binary_data
    alpha_fnr = 0.05  # Max 5% missed positive instances

    crc = ConformalRiskControlEngine()
    calibrated_lambda, _ = crc.calibrate(
        probs_cal=probs_cal,
        y_cal=y_cal,
        loss_type="fnr",
        target_risk=alpha_fnr,
    )

    psets_test = crc.get_prediction_sets(probs_test, calibrated_lambda)
    test_losses = crc.evaluate_loss(psets_test, y_test, loss_type="fnr")

    pos_mask = (y_test == 1)
    empirical_fnr = float(np.mean(test_losses[pos_mask]))

    assert empirical_fnr <= alpha_fnr + 0.03


def test_asymmetric_financial_cost_control(synthetic_binary_data):
    """Verify that custom asymmetric cost is strictly bounded by target risk."""
    probs_cal, y_cal, probs_test, y_test = synthetic_binary_data
    target_risk = 0.15

    crc = ConformalRiskControlEngine()
    calibrated_lambda, _ = crc.calibrate(
        probs_cal=probs_cal,
        y_cal=y_cal,
        loss_type="asymmetric_cost",
        target_risk=target_risk,
    )

    psets_test = crc.get_prediction_sets(probs_test, calibrated_lambda)
    test_losses = crc.evaluate_loss(psets_test, y_test, loss_type="asymmetric_cost")
    empirical_cost = float(np.mean(test_losses))

    assert empirical_cost <= target_risk + 0.04


def test_mondrian_group_conditional_risk_control(synthetic_multiclass_data):
    """Verify that Mondrian CRC satisfies risk bound per individual class."""
    probs_cal, y_cal, probs_test, y_test = synthetic_multiclass_data
    alpha = 0.12

    crc = ConformalRiskControlEngine()
    per_class_lambdas, _ = crc.calibrate(
        probs_cal=probs_cal,
        y_cal=y_cal,
        loss_type="misclassification",
        target_risk=alpha,
        mondrian=True,
    )

    assert isinstance(per_class_lambdas, dict)
    assert len(per_class_lambdas) == 3

    psets_test = crc.get_prediction_sets(probs_test, per_class_lambdas)
    test_losses = crc.evaluate_loss(psets_test, y_test, loss_type="misclassification")

    # Evaluate conditional risk for each class
    for c in [0, 1, 2]:
        mask_c = (y_test == c)
        class_risk = float(np.mean(test_losses[mask_c]))
        assert class_risk <= alpha + 0.05


def test_human_triage_escalation(synthetic_binary_data):
    """Verify that ambiguous or empty sets trigger needs_human_review."""
    probs_cal, y_cal, probs_test, _ = synthetic_binary_data

    crc = ConformalRiskControlEngine()
    calibrated_lambda, _ = crc.calibrate(probs_cal, y_cal, target_risk=0.02)

    psets, records = crc.predict_and_triage(probs_test, calibrated_lambda)

    assert len(records) == len(probs_test)
    for r in records:
        if r["is_ambiguous"] or r["is_empty"]:
            assert r["needs_human_review"] is True
        else:
            assert r["needs_human_review"] is False
