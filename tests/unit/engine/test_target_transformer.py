"""Unit tests for Skewed Target Transformer with negative scale protection."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor

from ml_mcp.engine.target_transformer import SkewedTargetTransformer


def test_target_transformer_positive_skewed():
    np.random.seed(42)
    # Heavy right-skewed positive target (e.g. log-normal distribution, incomes)
    y_pos = np.random.lognormal(mean=10.0, sigma=1.8, size=200)
    assert (y_pos >= 0).all()

    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    regressor = transformer.wrap_estimator(Ridge(), y_pos)

    # Must be wrapped in TransformedTargetRegressor with log1p
    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.func == np.log1p


def test_target_transformer_negative_skewed():
    np.random.seed(42)
    # Skewed target containing negative numbers (e.g. net profit/loss: -$50k to +$500k)
    y_neg = np.random.lognormal(mean=5.0, sigma=2.0, size=200) - 200.0
    assert (y_neg < 0).any()

    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    regressor = transformer.wrap_estimator(Ridge(), y_neg)

    # Because negative values exist, it MUST NOT use log1p! It must use Yeo-Johnson transformer
    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.transformer is not None


def test_target_transformer_unskewed():
    np.random.seed(42)
    # Standard normal distribution (skewness near 0)
    y_normal = np.random.normal(100.0, 15.0, size=200)

    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    regressor = transformer.wrap_estimator(Ridge(), y_normal)

    # Normal target should NOT be wrapped, returns base estimator
    assert isinstance(regressor, Ridge)
