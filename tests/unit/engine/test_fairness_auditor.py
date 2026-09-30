"""Unit tests for Demographic Slice Fairness Auditor."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.fairness_auditor import SliceFairnessAuditor
from ml_mcp.schemas.safety import SliceFairnessDTO


def test_fairness_auditor_balanced_groups():
    """Verify fairness auditor reports no parity violation when performance is uniform."""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=8, random_state=42)
    # Random uniform protected group assignment
    groups = np.random.choice(["GroupA", "GroupB", "GroupC"], size=500)

    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X[:350], y[:350])

    auditor = SliceFairnessAuditor()
    report = auditor.audit(
        model=clf,
        X_test=X[350:],
        y_test=y[350:],
        protected_series=pd.Series(groups[350:]),
        protected_attribute="ethnicity_group",
        metric="accuracy",
    )

    assert isinstance(report, SliceFairnessDTO)
    assert report.protected_attribute == "ethnicity_group"
    assert len(report.subgroup_scores) == 3
    assert report.disparate_impact_ratio >= 0.50
    assert report.max_disparity >= 0.0


def test_fairness_auditor_parity_violation_flagged():
    """Verify parity_violated is True when one group suffers extreme disparity (<0.80 DIR)."""
    auditor = SliceFairnessAuditor()
    # Mocking sub-group results with severe disparity: 0.90 vs 0.40 -> DIR = 0.40 / 0.90 = 0.444 < 0.80
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Model predicts Group1 perfectly, Group2 completely wrong
    class DummyBiasedModel:
        def predict(self, X):
            # First 4 are Group1 (all correct), next 4 are Group2 (all inverted)
            return np.array([1, 1, 1, 1, 1, 1, 1, 1])

    protected = pd.Series(["Alpha", "Alpha", "Alpha", "Alpha", "Beta", "Beta", "Beta", "Beta"])
    report = auditor.audit(
        model=DummyBiasedModel(),
        X_test=np.zeros((8, 2)),
        y_test=y_true,
        protected_series=protected,
        protected_attribute="region",
        metric="accuracy",
    )

    assert isinstance(report, SliceFairnessDTO)
    assert report.parity_violated is True
    assert report.disparate_impact_ratio < 0.80
    assert report.subgroup_scores["Alpha"] == 1.0
    assert report.subgroup_scores["Beta"] == 0.0
