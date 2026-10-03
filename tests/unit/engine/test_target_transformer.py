"""Unit tests for Residual-Driven Skewed Target Transformer (Tarasiuk 2021/2023)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor

from ml_mcp.engine.target_transformer import SkewedTargetTransformer


def test_target_transformer_positive_skewed():
    np.random.seed(42)
    y_pos = np.random.lognormal(mean=10.0, sigma=1.8, size=200)
    assert (y_pos >= 0).all()

    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    regressor = transformer.wrap_estimator(Ridge(), y_pos)

    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.func == np.log1p


def test_target_transformer_negative_skewed():
    np.random.seed(42)
    y_neg = np.random.lognormal(mean=5.0, sigma=2.0, size=200) - 200.0
    assert (y_neg < 0).any()

    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    regressor = transformer.wrap_estimator(Ridge(), y_neg)

    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.transformer is not None


def test_target_transformer_unskewed():
    np.random.seed(42)
    y_normal = np.random.normal(100.0, 15.0, size=200)

    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    regressor = transformer.wrap_estimator(Ridge(), y_normal)

    assert isinstance(regressor, Ridge)


def test_target_transformer_direct_transform():
    np.random.seed(42)
    y_pos = np.random.lognormal(mean=10.0, sigma=1.8, size=200)
    transformer = SkewedTargetTransformer(skew_threshold=1.5)
    res_pos = transformer.transform_target(y_pos)
    assert res_pos["is_skewed"] is True
    assert res_pos["method"] == "log1p"
    assert len(res_pos["transformed_values"]) == 200

    y_neg = np.random.lognormal(mean=5.0, sigma=2.0, size=200) - 200.0
    res_neg = transformer.transform_target(y_neg)
    assert res_neg["is_skewed"] is True
    assert res_neg["method"] == "yeo-johnson"


def test_target_transformer_residual_driven_validation():
    # Verify Tarasiuk (2021/2023) residual-driven transform
    np.random.seed(42)
    X = np.random.randn(150, 3)
    # Target has heavy linear relation + lognormal residuals
    linear_signal = 5.0 * X[:, 0] + 2.0 * X[:, 1]
    skewed_noise = np.random.lognormal(mean=2.0, sigma=1.5, size=150)
    y = linear_signal + skewed_noise

    transformer = SkewedTargetTransformer(skew_threshold=1.2)
    res = transformer.transform_target(y, X=X)
    assert "residual_skewness" in res
    assert res["is_skewed"] is True
