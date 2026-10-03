"""Unit tests for Collinearity and VIF filter with SVD Spectral Conditioning and Competitive Drop Rule."""
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
    # Check that VIF scores and condition number are present in the report
    assert "vif_scores" in report
    assert "spectral_condition_number" in report
    assert report["spectral_condition_number"] is not None


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
    # Highly correlated x1 and x3 should have high VIF
    assert vif_dict["x1"] > 5.0
    assert vif_dict["x3"] > 5.0


def test_collinearity_svd_spectral_and_iterative_pruning_on_linear_combination():
    np.random.seed(42)
    n = 400

    # Multi-column dependency: X3 is an exact linear combination of X1 and X2 (X3 = X1 + X2)
    # Pairwise correlations between (X1, X3) and (X2, X3) are around 0.707 (well below 0.90!)
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    x3 = x1 + x2 + np.random.normal(0, 0.001, size=n)  # exact collinearity
    x_independent = np.random.normal(10, 2, size=n)

    y = 2.0 * x1 + np.random.normal(0, 0.5, size=n)

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x_ind": x_independent,
        "target": y,
    })

    # Prior pairwise filter with threshold 0.90 would miss this because pairwise r ~ 0.70!
    # SVD condition number and iterative VIF will catch it!
    initial_cond = CollinearityFilter.calculate_spectral_condition_number(df, ["x1", "x2", "x3", "x_ind"])
    assert initial_cond > 30.0  # Catches multi-column collinearity!

    collin_filter = CollinearityFilter(threshold_corr=0.90, vif_threshold=10.0, condition_number_threshold=30.0)
    filtered_df, report = collin_filter.filter_collinearity(df, target_column="target")

    # One of the collinear linear combination features must be pruned
    dropped = report["dropped_features"]
    assert len(dropped) >= 1
    assert any(feat in dropped for feat in ["x1", "x2", "x3"])
    assert "x_ind" not in dropped
    assert "target" in filtered_df.columns
