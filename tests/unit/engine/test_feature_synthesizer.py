"""Unit tests for Scikit-Learn compliant Feature Synthesizer."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from ml_mcp.engine.feature_synthesizer import (
    CyclicalFeatureTransformer,
    RatioFeatureTransformer,
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

    # Check trigonometric accuracy: hour 0: sin(0)=0, cos(0)=1
    assert pytest.approx(transformed.loc[0, "hour_of_day_sin"], abs=1e-5) == 0.0
    assert pytest.approx(transformed.loc[0, "hour_of_day_cos"], abs=1e-5) == 1.0

    # Hour 6: 6/24 = 1/4 cycle (90 deg) -> sin=1, cos=0
    assert pytest.approx(transformed.loc[1, "hour_of_day_sin"], abs=1e-5) == 1.0
    assert pytest.approx(transformed.loc[1, "hour_of_day_cos"], abs=1e-5) == 0.0


def test_ratio_transformer_division_by_zero_protection():
    assert issubclass(RatioFeatureTransformer, (BaseEstimator, TransformerMixin))

    df = pd.DataFrame({
        "total_revenue": [100.0, 200.0, 50.0],
        "zero_denominator": [0.0, 5.0, 0.0],  # Division by zero risk!
    })

    transformer = RatioFeatureTransformer(
        ratio_pairs=[("total_revenue", "zero_denominator", "rev_per_unit")],
        epsilon=1e-6,
    )
    transformed = transformer.fit_transform(df)

    assert "rev_per_unit" in transformed.columns
    # Must NOT produce Inf or NaN
    assert not np.isinf(transformed["rev_per_unit"]).any()
    assert not np.isnan(transformed["rev_per_unit"]).any()
    # At row 0: 100.0 / 1e-6 = 1e8
    assert transformed.loc[0, "rev_per_unit"] == pytest.approx(100.0 / 1e-6)
