"""Unified Scoring Engine & Metric Registry.

Provides mathematically verified scoring functions on Out-Of-Fold (OOF) predictions.
Enforces Cardinal Sin #2 prevention: strictly distinguishes probabilities from argmax labels.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import sklearn.metrics

logger = logging.getLogger(__name__)


def score_roc_auc(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    """Calculates ROC AUC strictly on probabilities."""
    arr = np.asarray(y_pred_or_proba)
    if arr.ndim > 1 and arr.shape[1] > 1:
        if arr.shape[1] == 2:
            return float(sklearn.metrics.roc_auc_score(y_true, arr[:, 1]))
        return float(sklearn.metrics.roc_auc_score(y_true, arr, multi_class="ovr"))
    return float(sklearn.metrics.roc_auc_score(y_true, arr))


def score_average_precision(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    """Calculates PR-AUC (Average Precision) strictly on probabilities."""
    arr = np.asarray(y_pred_or_proba)
    if arr.ndim > 1 and arr.shape[1] > 1:
        if arr.shape[1] == 2:
            return float(sklearn.metrics.average_precision_score(y_true, arr[:, 1]))
        return float(sklearn.metrics.average_precision_score(y_true, arr))
    return float(sklearn.metrics.average_precision_score(y_true, arr))


def score_neg_log_loss(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    """Calculates negative Log-Loss strictly on probability matrices."""
    return -float(sklearn.metrics.log_loss(y_true, y_pred_or_proba))


def score_accuracy(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    """Calculates Accuracy on discrete label predictions."""
    arr = np.asarray(y_pred_or_proba)
    if arr.ndim > 1 and arr.shape[1] > 1:
        preds = np.argmax(arr, axis=1)
    elif arr.dtype.kind == "f" and set(np.unique(y_true)).issubset({0, 1}):
        preds = (arr >= 0.5).astype(int)
    else:
        preds = arr
    return float(sklearn.metrics.accuracy_score(y_true, preds))


def score_f1_weighted(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    arr = np.asarray(y_pred_or_proba)
    preds = np.argmax(arr, axis=1) if arr.ndim > 1 and arr.shape[1] > 1 else arr
    return float(sklearn.metrics.f1_score(y_true, preds, average="weighted", zero_division=0))


def score_f1_macro(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    arr = np.asarray(y_pred_or_proba)
    preds = np.argmax(arr, axis=1) if arr.ndim > 1 and arr.shape[1] > 1 else arr
    return float(sklearn.metrics.f1_score(y_true, preds, average="macro", zero_division=0))


def score_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(sklearn.metrics.r2_score(y_true, y_pred))


def score_neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(sklearn.metrics.mean_squared_error(y_true, y_pred))


def score_neg_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(sklearn.metrics.root_mean_squared_error(y_true, y_pred))


def score_neg_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(sklearn.metrics.mean_absolute_error(y_true, y_pred))


# Registry: maps canonical metric names to scoring function & probability requirement
SCORER_REGISTRY: Dict[str, Tuple[Callable[[np.ndarray, np.ndarray], float], bool]] = {
    "roc_auc": (score_roc_auc, True),
    "average_precision": (score_average_precision, True),
    "pr_auc": (score_average_precision, True),
    "neg_log_loss": (score_neg_log_loss, True),
    "log_loss": (score_neg_log_loss, True),
    "accuracy": (score_accuracy, False),
    "f1": (score_f1_weighted, False),
    "f1_weighted": (score_f1_weighted, False),
    "f1_macro": (score_f1_macro, False),
    "r2": (score_r2, False),
    "neg_mean_squared_error": (score_neg_mse, False),
    "mse": (score_neg_mse, False),
    "neg_root_mean_squared_error": (score_neg_rmse, False),
    "rmse": (score_neg_rmse, False),
    "neg_mean_absolute_error": (score_neg_mae, False),
    "mae": (score_neg_mae, False),
}


def resolve_scorer(name: str) -> Tuple[Callable[[np.ndarray, np.ndarray], float], bool]:
    """Resolves metric name to scoring callable and probability flag.

    Raises:
        ValueError if metric name is unrecognized (anti-silent fallback).
    """
    clean_name = name.strip().lower()
    if clean_name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unsupported scoring metric '{name}'. Allowed metrics: {sorted(SCORER_REGISTRY.keys())}"
        )
    return SCORER_REGISTRY[clean_name]
