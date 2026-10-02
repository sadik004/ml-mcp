"""Unit tests for Collinearity and VIF filter with Competitive Drop Rule."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.collinearity import CollinearityFilter


def test_collinearity_filter_drops_inferior_twin():
    np.random.seed(42)
    n = 200

    y = np.random.binomial(1, 0.5, size=n)
    # Feature A has high correlation with target
    feat_a = y * 2.0 + np.random.normal(0, 0.5, size=n)
    # Feature B is a twin of Feature A (r > 0.95), but with more noise relative to target
    feat_b = feat_a + np.random.normal(0, 0.1, size=n)
    # Independent feature C
    feat_c = np.random.normal(0, 1, size=n)

    df = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
        "target": y,
    })

    collin_filter = CollinearityFilter(threshold_corr=0.90)
    filtered_df, report = collin_filter.filter_collinearity(
        df, target_column="target", task_type="classification"
    )

    # One of the collinear twins must be dropped
    dropped_cols = report["dropped_features"]
    assert len(dropped_cols) == 1
    # feat_a has stronger correlation with target, so feat_b should be dropped
    assert "feat_b" in dropped_cols
    assert "feat_a" not in dropped_cols
    assert "feat_c" not in dropped_cols
    # Target column must remain untouched
    assert "target" in filtered_df.columns
    assert "feat_a" in filtered_df.columns
    assert "feat_b" not in filtered_df.columns
    # Check that VIF scores are present in the report
    assert "vif_scores" in report
    assert "high_vif_features" in report


def test_collinearity_filter_handles_singular_matrix():
    # Exactly identical duplicate columns (matrix singularity)
    df = pd.DataFrame({
        "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col2": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfect duplicate
        "target": [0, 1, 0, 1, 0],
    })

    collin_filter = CollinearityFilter(threshold_corr=0.95)
    filtered_df, report = collin_filter.filter_collinearity(df, target_column="target")

    assert len(report["dropped_features"]) == 1
    assert "col2" in report["dropped_features"] or "col1" in report["dropped_features"]
    assert "target" in filtered_df.columns


def test_vif_calculation_pure_numpy():
    np.random.seed(42)
    n = 300

    # Two orthogonal features: VIF should be close to 1.0
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)

    # Collinear feature: x3 = 0.95 * x1 + noise
    x3 = x1 * 0.95 + np.random.normal(0, 0.1, size=n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    collin_filter = CollinearityFilter(threshold_corr=0.90, vif_threshold=10.0)
    vif_dict = collin_filter.calculate_vif(df)

    assert "x1" in vif_dict
    assert "x2" in vif_dict
    assert "x3" in vif_dict

    # Orthogonal x2 should have low VIF near 1.0
    assert vif_dict["x2"] < 2.0
    # Highly correlated x1 and x3 should have high VIF (close to or above 10.0)
    assert vif_dict["x1"] > 5.0
    assert vif_dict["x3"] > 5.0
