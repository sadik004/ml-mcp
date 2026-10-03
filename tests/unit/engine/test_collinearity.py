"""Unit tests for Collinearity filter with SVD Spectral Conditioning, Ridge-VIF, and Belsley diagnostics."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.collinearity import CollinearityFilter


def test_collinearity_filter_drops_inferior_twin():
    np.random.seed(42)
    n = 200

    y = np.random.binomial(1, 0.5, size=n)
    feat_a = y * 2.0 + np.random.normal(0, 0.5, size=n)
    feat_b = feat_a + np.random.normal(0, 0.05, size=n)
    feat_c = np.random.normal(0, 1, size=n)

    df = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
        "target": y,
    })

    col_filter = CollinearityFilter(vif_threshold=10.0, condition_number_threshold=30.0)
    report = col_filter.filter_collinearity(df, target_column="target")

    assert len(report.dropped_features) >= 1
    assert "feat_c" not in report.dropped_features
    assert report.remaining_features_count < 3


def test_ridge_vif_no_singularity_crash():
    """Exact identical duplicate columns must not crash with singular matrix error."""
    n = 100
    x = np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "x1": x,
        "x2": x,  # Exact duplicate
        "x3": np.random.normal(0, 1, n),
    })

    col_filter = CollinearityFilter(ridge_alpha=1e-4)
    vifs = col_filter.calculate_vif(df)

    assert "x1" in vifs
    assert "x2" in vifs
    assert vifs["x1"] > 10.0
    assert vifs["x2"] > 10.0


def test_variance_decomposition_proportions():
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.01, n)
    x3 = np.random.normal(0, 1, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    col_filter = CollinearityFilter()
    decomp = col_filter.calculate_variance_decomposition_proportions(df)

    assert "condition_indices" in decomp
    assert "collinear_groups" in decomp
    assert len(decomp["condition_indices"]) == 3
