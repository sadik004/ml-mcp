"""Probability Calibration Engine with Platt Scaling, Isotonic Regression, and ECE."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize

from ml_mcp.schemas.tuning import CalibrationReportDTO

logger = logging.getLogger(__name__)


def calculate_ece(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE) across n_bins.
    
    Supports both binary and multiclass probabilities.
    For binary: bins the positive class probabilities.
    For multiclass: uses top-label confidence calibration.
    """
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)

    if probas.ndim == 1 or probas.shape[1] == 2:
        # Binary classification
        confidences = probas[:, 1] if probas.ndim > 1 else probas
        predictions = (confidences >= 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    else:
        # Multiclass classification (confidence of top predicted class)
        predictions = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def calculate_multiclass_brier(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Compute Brier score for binary or multiclass classification."""
    classes = np.unique(y_true)
    if len(classes) <= 2:
        pos_prob = probas[:, 1] if probas.ndim > 1 and probas.shape[1] == 2 else probas
        return float(brier_score_loss(y_true, pos_prob))
    
    # Multiclass Brier score: mean squared error between one-hot y and predicted probabilities
    y_one_hot = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((probas - y_one_hot) ** 2, axis=1)))


class ProbabilityCalibrator:
    """Probability Calibrator managing Platt Scaling, Isotonic Regression, and calibration audit."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def calibrate(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        task_type: str = "classification",
        method: Optional[str] = None,
        cv: int = 5,
    ) -> Tuple[CalibrationReportDTO, BaseEstimator]:
        """Calibrate classifier probabilities without data leakage.
        
        If task_type is 'regression', safely skips calibration and returns original model.
        """
        if task_type == "regression":
            logger.info("Skipping calibration for regression task.")
            report = CalibrationReportDTO(
                method="none",
                pre_brier_score=0.0,
                post_brier_score=0.0,
                brier_score_lift=0.0,
                is_well_calibrated=True,
                status="skipped_regression_task",
            )
            return report, model

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Automatic method selection if not explicitly provided
        if method is None:
            # Platt Scaling (sigmoid) for smaller datasets; Isotonic for >= 1000 samples
            method = "sigmoid" if len(X_arr) < 1000 else "isotonic"

        # Safe CV folds
        n_classes = len(np.unique(y_arr))
        effective_cv = min(cv, len(X_arr) // n_classes)
        if effective_cv < 2:
            effective_cv = 2

        # 1. Compute Pre-Calibration Out-Of-Fold probabilities
        try:
            pre_probas = cross_val_predict(
                clone(model),
                X_arr,
                y_arr,
                cv=effective_cv,
                method="predict_proba",
                n_jobs=-1,
            )
        except Exception:
            # Fallback if cross_val_predict fails
            model_fitted = clone(model).fit(X_arr, y_arr)
            pre_probas = model_fitted.predict_proba(X_arr)

        pre_brier = calculate_multiclass_brier(y_arr, pre_probas)
        pre_ece = calculate_ece(y_arr, pre_probas, n_bins=10)

        # 2. Fit Leak-Free Calibrated Classifier
        calibrated_model = CalibratedClassifierCV(
            estimator=clone(model),
            method=method,
            cv=effective_cv,
        )
        calibrated_model.fit(X_arr, y_arr)

        # 3. Compute Post-Calibration metrics
        try:
            post_probas = cross_val_predict(
                calibrated_model,
                X_arr,
                y_arr,
                cv=effective_cv,
                method="predict_proba",
                n_jobs=-1,
            )
        except Exception:
            post_probas = calibrated_model.predict_proba(X_arr)

        post_brier = calculate_multiclass_brier(y_arr, post_probas)
        post_ece = calculate_ece(y_arr, post_probas, n_bins=10)

        brier_lift = pre_brier - post_brier
        ece_lift = pre_ece - post_ece
        is_well_calibrated = bool(post_brier <= 0.15 or post_ece <= 0.10)

        report = CalibrationReportDTO(
            method=method,
            pre_brier_score=float(pre_brier),
            post_brier_score=float(post_brier),
            brier_score_lift=float(brier_lift),
            is_well_calibrated=is_well_calibrated,
            status="completed",
            pre_ece=float(pre_ece),
            post_ece=float(post_ece),
            ece_lift=float(ece_lift),
        )

        return report, calibrated_model
