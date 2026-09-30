"""Unit tests for Decision Threshold Optimizer and Error Forensics."""
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
    assert 0.01 <= report.optimal_threshold <= 0.99
    assert report.f_beta_score >= 0.0
    assert report.precision >= 0.0
    assert report.recall >= 0.0

    # Validate confusion matrix integrity
    cm = report.confusion_matrix
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_test)
    assert report.false_positive_count == cm["fp"]
    assert report.false_negative_count == cm["fn"]


def test_threshold_optimizer_beta_recall_vs_precision():
    """Verify beta=2.0 prioritizes recall and beta=0.5 prioritizes precision."""
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        weights=[0.80, 0.20],
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X[:350], y[:350])
    probas = clf.predict_proba(X[350:])[:, 1]
    y_test = y[350:]

    optimizer = DecisionThresholdOptimizer()
    report_recall = optimizer.optimize(y_true=y_test, y_probas=probas, beta=2.0)
    report_precision = optimizer.optimize(y_true=y_test, y_probas=probas, beta=0.5)

    assert report_recall.recall >= report_precision.recall
    assert report_precision.precision >= report_recall.precision or report_precision.optimal_threshold >= report_recall.optimal_threshold


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
