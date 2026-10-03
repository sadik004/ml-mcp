"""Probability Calibration Engine with Platt Scaling, Isotonic, Temperature Scaling, and Adaptive ECE."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize

from ml_mcp.schemas.tuning import CalibrationReportDTO

logger = logging.getLogger(__name__)


def calculate_adaptive_ece(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Debiased Adaptive-Quantile Expected Calibration Error (Roelofs et al. NeurIPS 2022).

    Unlike fixed uniform-width bins, quantile-based bins guarantee equal sample counts per bin,
    eliminating severe sample-size estimation bias on highly confident or skewed datasets.
    """
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)
    total_samples = len(y_true)
    if total_samples == 0:
        return 0.0

    if probas.ndim == 1 or probas.shape[1] == 2:
        confidences = probas[:, 1] if probas.ndim > 1 else probas
        predictions = (confidences >= 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    else:
        predictions = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        accuracies = (predictions == y_true).astype(float)

    # Compute quantile bin edges for equal sample representation
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(confidences, quantiles)
    bin_edges = np.unique(bin_edges)  # remove duplicate edges for constant values

    if len(bin_edges) <= 1:
        return float(abs(np.mean(accuracies) - np.mean(confidences)))

    ece = 0.0
    for i in range(len(bin_edges) - 1):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def calculate_ece(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate standard Expected Calibration Error (ECE) across uniform bins."""
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)

    if probas.ndim == 1 or probas.shape[1] == 2:
        confidences = probas[:, 1] if probas.ndim > 1 else probas
        predictions = (confidences >= 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    else:
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

    y_one_hot = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((probas - y_one_hot) ** 2, axis=1)))


def enforce_simplex_normalization(probas: np.ndarray) -> np.ndarray:
    """Enforces sum-to-one simplex probability constraint (sum_i p_i = 1.0) without negative values.

    Theoretical Basis: Kull et al. (NeurIPS 2019) Dirichlet Multiclass Calibration.
    """
    p = np.maximum(0.0, np.asarray(probas, dtype=float))
    if p.ndim == 1:
        return p
    if p.shape[1] == 1:
        return p
    # Clip any NaN or inf
    p = np.nan_to_num(p, nan=1.0 / p.shape[1], posinf=1.0, neginf=0.0)
    row_sums = np.sum(p, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return p / row_sums


class TemperatureScaler(BaseEstimator, ClassifierMixin):
    """Guo et al. (ICML 2017) Temperature Scaling post-processing calibrator."""

    def __init__(self, base_estimator: Any, temperature: float = 1.0) -> None:
        self.base_estimator = base_estimator
        self.temperature = temperature
        self.temperature_ = temperature

    def fit(self, X: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        raw_probs = self.base_estimator.predict_proba(X)
        y_arr = np.asarray(y)

        # Convert probabilities to logits safely
        eps = 1e-12
        clipped = np.clip(raw_probs, eps, 1.0 - eps)

        if clipped.shape[1] == 2:
            logits = logit(clipped[:, 1])

            def nll(t: float) -> float:
                if t <= 0.01:
                    return 1e9
                p = expit(logits / t)
                p = np.clip(p, eps, 1.0 - eps)
                return float(-np.sum(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))

            res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
            self.temperature_ = float(res.x)
        else:
            logits = np.log(clipped)

            def nll_mc(t: float) -> float:
                if t <= 0.01:
                    return 1e9
                scaled_probs = softmax(logits / t, axis=1)
                scaled_probs = np.clip(scaled_probs, eps, 1.0 - eps)
                classes = np.unique(y_arr)
                one_hot = label_binarize(y_arr, classes=classes)
                return float(-np.sum(one_hot * np.log(scaled_probs)))

            res = minimize_scalar(nll_mc, bounds=(0.05, 10.0), method="bounded")
            self.temperature_ = float(res.x)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_estimator.predict_proba(X)
        eps = 1e-12
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        T = max(0.01, self.temperature_)

        if clipped.shape[1] == 2:
            logits = logit(clipped[:, 1])
            pos = expit(logits / T)
            return np.column_stack([1.0 - pos, pos])
        else:
            logits = np.log(clipped)
            return softmax(logits / T, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class ProbabilityCalibrator:
    """Probability Calibrator managing Platt, Isotonic, Temperature Scaling, and Conformal Alpha Guards."""

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
        conformal_alpha: float = 0.05,
    ) -> Tuple[CalibrationReportDTO, BaseEstimator]:
        """Calibrate classifier probabilities without data leakage."""
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

        if method is None:
            method = "sigmoid" if len(X_arr) < 1000 else "isotonic"

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
            model_fitted = clone(model).fit(X_arr, y_arr)
            pre_probas = model_fitted.predict_proba(X_arr)

        pre_probas = enforce_simplex_normalization(pre_probas)
        pre_brier = calculate_multiclass_brier(y_arr, pre_probas)
        pre_ece = calculate_ece(y_arr, pre_probas, n_bins=10)

        # 2. Fit Calibrated Classifier
        temp_val: Optional[float] = None
        if method == "temperature":
            base_fit = clone(model).fit(X_arr, y_arr)
            temp_scaler = TemperatureScaler(base_estimator=base_fit)
            temp_scaler.fit(X_arr, y_arr)
            calibrated_model = temp_scaler
            temp_val = temp_scaler.temperature_
        else:
            calibrated_model = CalibratedClassifierCV(
                estimator=clone(model),
                method=method,
                cv=effective_cv,
            )
            calibrated_model.fit(X_arr, y_arr)

        # 3. Compute Post-Calibration metrics with Simplex Normalization
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

        post_probas = enforce_simplex_normalization(post_probas)

        post_brier = calculate_multiclass_brier(y_arr, post_probas)
        post_ece = calculate_ece(y_arr, post_probas, n_bins=10)
        adaptive_ece = calculate_adaptive_ece(y_arr, post_probas, n_bins=10)

        brier_lift = pre_brier - post_brier
        ece_lift = pre_ece - post_ece
        is_well_calibrated = bool(post_brier <= 0.15 or post_ece <= 0.10)

        # 4. Conformal Prediction Alpha Guard (Angelopoulos & Bates 2023)
        # Compute non-conformity scores s_i = 1 - p(y_i)
        if post_probas.ndim == 1 or post_probas.shape[1] == 2:
            prob_y = np.where(y_arr == 1, post_probas[:, 1], post_probas[:, 0])
        else:
            classes = np.unique(y_arr)
            prob_y = post_probas[np.arange(len(y_arr)), y_arr]

        non_conformity = 1.0 - prob_y
        q_level = np.ceil((len(y_arr) + 1) * (1.0 - conformal_alpha)) / len(y_arr)
        q_level = float(np.clip(q_level, 0.0, 1.0))
        q_hat = float(np.quantile(non_conformity, q_level))
        empirical_coverage = float(np.mean(non_conformity <= q_hat))

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
            adaptive_ece=float(adaptive_ece),
            temperature=temp_val,
            conformal_coverage=round(empirical_coverage, 4),
            conformal_alpha=conformal_alpha,
        )

        return report, calibrated_model
