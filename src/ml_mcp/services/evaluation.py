"""Shared Out-Of-Sample Evaluation Engine and Cross-Validation Helpers (Phase P3)."""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ml_mcp.config import get_settings
from ml_mcp.engine.pipeline_builder import (
    build_sealed_pipeline,
    prepare_estimator,
)

logger = logging.getLogger(__name__)

__all__ = ["build_sealed_pipeline", "prepare_estimator", "oof_predict_proba", "oof_predict", "fit_final"]


def oof_predict_proba(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Run honest out-of-fold probability estimation using sealed cross-validation pipelines.

    Guarantees:
    - Preprocessing (scaling, imputation, encoding) is refit strictly per training fold.
    - Test folds are never observed during feature transformation or model training.
    """
    settings = get_settings()
    seed = random_state if random_state is not None else settings.random_state

    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    unique_classes, counts = np.unique(y_arr, return_counts=True)
    min_class_count = int(np.min(counts)) if len(counts) > 0 else 0

    effective_cv = min(cv, min_class_count) if min_class_count >= 2 else min(cv, n_samples)
    effective_cv = max(2, min(effective_cv, n_samples // 2)) if n_samples >= 4 else 2

    try:
        splitter = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=seed)
        next(splitter.split(X, y_arr))
    except Exception:
        splitter = KFold(n_splits=effective_cv, shuffle=True, random_state=seed)

    pipeline = build_sealed_pipeline(clone(estimator), X)

    oof_probas = cross_val_predict(
        pipeline,
        X,
        y_arr,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )
    return np.asarray(oof_probas)


def oof_predict(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Run honest out-of-fold prediction estimation using sealed cross-validation pipelines."""
    settings = get_settings()
    seed = random_state if random_state is not None else settings.random_state

    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    effective_cv = max(2, min(cv, n_samples // 2)) if n_samples >= 4 else 2
    splitter = KFold(n_splits=effective_cv, shuffle=True, random_state=seed)

    pipeline = build_sealed_pipeline(clone(estimator), X)
    preds = cross_val_predict(
        pipeline,
        X,
        y_arr,
        cv=splitter,
        n_jobs=1,
    )
    return np.asarray(preds)


def fit_final(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    target_column: Optional[str] = None,
) -> Pipeline:
    """Fit full dataset for artifact serialization, explicitly marked trained_on='full_data'."""
    pipeline = build_sealed_pipeline(clone(estimator), X, target_column=target_column)
    pipeline.fit(X, y)
    setattr(pipeline, "trained_on", "full_data")
    return pipeline
