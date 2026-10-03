"""Unit tests for Intersectional Subgroup Fairness (Kearns et al. ICML 2018)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from ml_mcp.engine.fairness_auditor import SliceFairnessAuditor


def test_fairness_auditor_intersectional_slices():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = np.random.binomial(1, 0.5, size=n)

    # Two protected attributes creating 4 intersectional subgroups
    protected_df = pd.DataFrame({
        "gender": np.random.choice(["M", "F"], size=n),
        "age_tier": np.random.choice(["Young", "Senior"], size=n),
    })

    model = LogisticRegression().fit(X, y)
    auditor = SliceFairnessAuditor()
    report = auditor.audit(model, X, y, protected_series=protected_df)

    assert "gender+age_tier" in report.protected_attribute
    # Must audit intersectional subgroups (e.g. M_x_Young, F_x_Senior, etc.)
    assert len(report.subgroup_scores) >= 2
    assert 0.0 <= report.disparate_impact_ratio <= 1.0
    assert 0.0 <= report.max_disparity <= 1.0
