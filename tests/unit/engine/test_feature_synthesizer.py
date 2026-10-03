"""Unit tests for Scikit-Learn compliant Feature Synthesizer with OpenFE Cross-Numeric synthesis."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from ml_mcp.engine.feature_synthesizer import (
    CyclicalFeatureTransformer,
    RatioFeatureTransformer,
    CrossNumericTransformer,
    GroupByAggregationTransformer,
)


def test_cyclical_transformer_is_sklearn_compliant():
    assert issubclass(CyclicalFeatureTransformer, (BaseEstimator, TransformerMixin))

    df = pd.DataFrame({
        "hour_of_day": [0, 6, 12, 18, 23],
    })

    transformer = CyclicalFeatureTransformer(time_periods={"hour_of_day": 24.0})
    transformed = transformer.fit_transform(df)

    assert "hour_of_day_sin" in transformed.columns
    assert "hour_of_day_cos" in transformed.columns

    assert pytest.approx(transformed.loc[0, "hour_of_day_sin"], abs=1e-5) == 0.0
    assert pytest.approx(transformed.loc[0, "hour_of_day_cos"], abs=1e-5) == 1.0
    assert pytest.approx(transformed.loc[1, "hour_of_day_sin"], abs=1e-5) == 1.0
    assert pytest.approx(transformed.loc[1, "hour_of_day_cos"], abs=1e-5) == 0.0


def test_ratio_transformer_division_by_zero_protection():
    assert issubclass(RatioFeatureTransformer, (BaseEstimator, TransformerMixin))

    df = pd.DataFrame({
        "total_revenue": [100.0, 200.0, 50.0],
        "zero_denominator": [0.0, 5.0, 0.0],
    })

    transformer = RatioFeatureTransformer(
        ratio_pairs=[("total_revenue", "zero_denominator", "rev_per_unit")],
        epsilon=1e-6,
    )
    transformed = transformer.fit_transform(df)

    assert "rev_per_unit" in transformed.columns
    assert not np.isinf(transformed["rev_per_unit"]).any()
    assert not np.isnan(transformed["rev_per_unit"]).any()
    assert transformed.loc[0, "rev_per_unit"] == pytest.approx(100.0 / 1e-6)


def test_cross_numeric_transformer_openfe():
    assert issubclass(CrossNumericTransformer, (BaseEstimator, TransformerMixin))

    # Create dataset with 4 numerical features having distinct variances
    np.random.seed(42)
    df = pd.DataFrame({
        "high_var": np.random.normal(100.0, 50.0, size=50),
        "med_var": np.random.normal(10.0, 15.0, size=50),
        "low_var": np.random.normal(1.0, 0.5, size=50),
        "zero_den": [0.0] * 25 + [2.0] * 25,
        "cat_str": ["A"] * 25 + ["B"] * 25,
    })

    transformer = CrossNumericTransformer(top_k=3, max_pairs=3, epsilon=1e-6)
    transformed = transformer.fit_transform(df)

    # Must select top variance pairs and create sub and ratio
    assert len(transformer.selected_pairs_) <= 3
    for col_a, col_b in transformer.selected_pairs_:
        diff_col = f"{col_a}_sub_{col_b}"
        ratio_col = f"{col_a}_ratio_{col_b}"
        assert diff_col in transformed.columns
        assert ratio_col in transformed.columns
        # Zero-division and NaN protection
        assert not transformed[diff_col].isna().any()
        assert not transformed[ratio_col].isna().any()
        assert not np.isinf(transformed[ratio_col]).any()


def test_groupby_aggregation_transformer_explorekit():
    assert issubclass(GroupByAggregationTransformer, (BaseEstimator, TransformerMixin))

    df_train = pd.DataFrame({
        "city": ["Mirpur", "Mirpur", "Gulshan", "Gulshan", "Uttara"],
        "rent": [20000.0, 30000.0, 80000.0, 100000.0, 50000.0],
    })

    transformer = GroupByAggregationTransformer(group_specs=[{
        "cat_col": "city",
        "num_col": "rent",
        "aggregations": ["mean", "std"],
        "create_relative_diff": True,
        "create_relative_ratio": True,
        "create_zscore": True,
    }])

    transformed_train = transformer.fit_transform(df_train)

    assert "rent_mean_by_city" in transformed_train.columns
    assert "rent_std_by_city" in transformed_train.columns
    assert "rent_diff_from_city_mean" in transformed_train.columns
    assert "rent_ratio_to_city_mean" in transformed_train.columns
    assert "rent_zscore_in_city" in transformed_train.columns

    assert transformed_train.loc[0, "rent_mean_by_city"] == 25000.0
    assert transformed_train.loc[0, "rent_diff_from_city_mean"] == -5000.0
    assert pytest.approx(transformed_train.loc[0, "rent_ratio_to_city_mean"], abs=1e-3) == 0.8

    df_test = pd.DataFrame({
        "city": ["Dhanmondi"],
        "rent": [60000.0],
    })
    transformed_test = transformer.transform(df_test)
    assert not transformed_test.isna().any().any()
    assert not np.isinf(transformed_test["rent_ratio_to_city_mean"]).any()


def test_groupby_catboost_smoothing_and_cardinality_guard():
    N = 100
    df = pd.DataFrame({
        "uuid_col": [f"id_{i}" for i in range(N)],
        "cat_col": ["A"] * 90 + ["B"] * 10,
        "val": [10.0] * 90 + [100.0] * 10,
    })

    transformer = GroupByAggregationTransformer(
        group_specs=[
            {"cat_col": "uuid_col", "num_col": "val", "aggregations": ["mean"]},
            {"cat_col": "cat_col", "num_col": "val", "aggregations": ["mean"]},
        ],
        max_cardinality=50,
        max_cardinality_ratio=0.20,
        smoothing=10.0,
    )

    transformed = transformer.fit_transform(df)

    assert "val_mean_by_uuid_col" not in transformed.columns
    assert len(transformer.skipped_specs_) == 1
    assert transformer.skipped_specs_[0]["cat_col"] == "uuid_col"

    assert "val_mean_by_cat_col" in transformed.columns
    assert "val_diff_from_cat_col_mean" in transformed.columns

    cat_b_row = transformed[transformed["cat_col"] == "B"].iloc[0]
    expected_smooth_mean = (10 * 100.0 + 10.0 * 19.0) / 20.0
    diff_val = cat_b_row["val_diff_from_cat_col_mean"]
    assert abs((cat_b_row["val"] - expected_smooth_mean) - diff_val) < 1e-4
