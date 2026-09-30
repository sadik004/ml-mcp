"""Unit tests for Class Balancer with dynamic k-neighbors and fallback guards."""
import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

from ml_mcp.engine.balancer import ClassBalancer


def test_balancer_standard_smote():
    # 100 samples total, 15 minority samples (minority > 5)
    y = np.array([0] * 85 + [1] * 15)

    balancer = ClassBalancer()
    resampler = balancer.get_resampler(y)

    assert isinstance(resampler, SMOTE)
    assert resampler.k_neighbors <= 5


def test_balancer_small_minority_dynamic_k_neighbors():
    # Only 4 minority samples (less than standard k=5)
    y = np.array([0] * 96 + [1] * 4)

    balancer = ClassBalancer()
    resampler = balancer.get_resampler(y)

    # Standard SMOTE with k=5 would crash with ValueError!
    # Our dynamic balancer must cap k_neighbors <= 3!
    assert isinstance(resampler, SMOTE)
    assert resampler.k_neighbors <= 3


def test_balancer_ultra_small_minority_random_fallback():
    # Ultra-rare minority: only 2 minority samples
    y = np.array([0] * 98 + [1] * 2)

    balancer = ClassBalancer()
    resampler = balancer.get_resampler(y)

    # With only 2 samples, SMOTE is unsafe; must fallback to RandomOverSampler
    assert isinstance(resampler, RandomOverSampler)


def test_balancer_wraps_in_pipeline_zero_leakage():
    from sklearn.linear_model import LogisticRegression

    y = np.array([0] * 80 + [1] * 20)
    balancer = ClassBalancer()
    pipe = balancer.wrap_pipeline(resampler_or_name="smote", estimator=LogisticRegression(), y=y)

    assert isinstance(pipe, ImbPipeline)
    assert "sampler" in pipe.named_steps
    assert "estimator" in pipe.named_steps
