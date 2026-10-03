"""Unit tests for Cost-Sensitive Class Balancer and Empirical Resampler (Wallace et al. 2021)."""
import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.balancer import ClassBalancer


def test_balancer_compute_weights():
    # 80 class 0, 20 class 1 -> N=100, K=2
    # w0 = 100 / (2 * 80) = 0.625
    # w1 = 100 / (2 * 20) = 2.5
    y = np.array([0] * 80 + [1] * 20)
    balancer = ClassBalancer()
    weights = balancer.compute_weights(y)

    assert len(weights) == 100
    assert pytest.approx(weights[0], abs=1e-5) == 0.625
    assert pytest.approx(weights[80], abs=1e-5) == 2.5


def test_balancer_compute_class_weights():
    y = np.array([0] * 75 + [1] * 25)
    balancer = ClassBalancer()
    class_weights = balancer.compute_class_weights(y)

    assert 0 in class_weights
    assert 1 in class_weights
    assert class_weights[1] > class_weights[0]
    # w0 = 100 / (2 * 75) = 2/3 ≈ 0.6667
    # w1 = 100 / (2 * 25) = 2.0
    assert pytest.approx(class_weights[0], abs=1e-3) == 0.6667
    assert pytest.approx(class_weights[1], abs=1e-3) == 2.0


def test_balancer_under_sampler_default():
    # 85 class 0, 15 class 1: Standard imbalance
    y = np.array([0] * 85 + [1] * 15)
    balancer = ClassBalancer()
    resampler = balancer.get_resampler(y)

    # In tabular ML, SMOTE is avoided (Wallace et al. 2021); default to RandomUnderSampler
    assert isinstance(resampler, RandomUnderSampler)


def test_balancer_small_minority_random_fallback():
    # Ultra-rare minority: only 2 minority samples
    y = np.array([0] * 98 + [1] * 2)

    balancer = ClassBalancer()
    resampler = balancer.get_resampler(y)

    # For tiny minority, undersampling would discard too much; fallback to RandomOverSampler
    assert isinstance(resampler, RandomOverSampler)


def test_balancer_smote_prohibited():
    # SMOTE must raise ValueError on tabular data (Wallace et al. 2021)
    y = np.array([0] * 80 + [1] * 20)
    balancer = ClassBalancer()
    with pytest.raises(ValueError, match="SMOTE is explicitly prohibited"):
        balancer.get_resampler(y, strategy="smote")


def test_balancer_wraps_in_pipeline_zero_leakage():
    y = np.array([0] * 80 + [1] * 20)
    balancer = ClassBalancer()
    pipe = balancer.wrap_pipeline(resampler_or_name="undersample", estimator=LogisticRegression(), y=y)

    assert isinstance(pipe, ImbPipeline)
    assert "sampler" in pipe.named_steps
    assert "estimator" in pipe.named_steps
    assert isinstance(pipe.named_steps["sampler"], RandomUnderSampler)
