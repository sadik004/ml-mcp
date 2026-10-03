"""Unit tests for Decision Threshold Optimizer and Cost Forensics."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.threshold import DecisionThresholdOptimizer
from ml_mcp.schemas.tuning import ThresholdReportDTO


def test_threshold_optimizer_f1_balanced():
    """Verify threshold optimization improves F1 on imbalanced data vs default 0.50."""
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        weights=[0.85, 0.15],
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X[:400], y[:400])
    probas = clf.predict_proba(X[400:])[:, 1]
    y_test = y[400:]

    optimizer = DecisionThresholdOptimizer()
    report = optimizer.optimize(y_true=y_test, y_probas=probas, beta=1.0)

    assert isinstance(report, ThresholdReportDTO)
    assert 0.001 <= report.optimal_threshold <= 0.999
    assert report.f_beta_score >= 0.0
    assert report.precision >= 0.0
    assert report.recall >= 0.0

    # Validate confusion matrix integrity
    cm = report.confusion_matrix
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_test)
    assert report.false_positive_count == cm["fp"]
    assert report.false_negative_count == cm["fn"]


def test_threshold_cost_matrix_optimization():
    """Verify cost-loss minimization produces positive cost savings over default 0.50."""
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        weights=[0.90, 0.10],
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X[:350], y[:350])
    probas = clf.predict_proba(X[350:])[:, 1]
    y_test = y[350:]

    optimizer = DecisionThresholdOptimizer()
    report = optimizer.optimize(
        y_true=y_test,
        y_probas=probas,
        criterion="cost_loss",
        cost_fp=1.0,
        cost_fn=10.0,
    )

    assert isinstance(report, ThresholdReportDTO)
    assert report.total_cost_optimal is not None
    assert report.total_cost_default is not None
    assert report.cost_savings is not None
    assert report.cost_savings >= 0.0  # optimal must be <= default cost
    assert report.analytical_cost_threshold is not None
    assert 0.01 <= report.analytical_cost_threshold <= 0.99


def test_threshold_optimizer_edge_cases():
    """Verify division by zero protection with all zeros or all ones."""
    optimizer = DecisionThresholdOptimizer()

    y_zeros = np.zeros(50, dtype=int)
    p_zeros = np.zeros(50, dtype=float)
    report_zeros = optimizer.optimize(y_true=y_zeros, y_probas=p_zeros, beta=1.0)
    assert isinstance(report_zeros, ThresholdReportDTO)
    assert report_zeros.f_beta_score == 0.0

    y_ones = np.ones(50, dtype=int)
    p_ones = np.ones(50, dtype=float)
    report_ones = optimizer.optimize(y_true=y_ones, y_probas=p_ones, beta=1.0)
    assert isinstance(report_ones, ThresholdReportDTO)
    assert report_ones.f_beta_score == 1.0
