"""Probability Calibration Engine with Beta Calibration (Kull et al. AISTATS) and Adaptive ECE."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, label_binarize

from ml_mcp.config import get_settings
from ml_mcp.schemas.tuning import CalibrationReportDTO

logger = logging.getLogger(__name__)


def calculate_ece(y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """Calculate standard Expected Calibration Error (ECE) with equal-width binning."""
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)
    if probas.ndim == 2:
        confidences = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)
    else:
        confidences = np.where(probas >= 0.5, probas, 1.0 - probas)
        predictions = (probas >= 0.5).astype(int)

    accuracies = predictions == y_true
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def calculate_adaptive_ece(y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Equal-Mass Adaptive-Quantile Expected Calibration Error."""
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)
    if probas.ndim == 2:
        confidences = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)
    else:
        confidences = np.where(probas >= 0.5, probas, 1.0 - probas)
        predictions = (probas >= 0.5).astype(int)

    accuracies = predictions == y_true
    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0

    # Equal-frequency quantile binning eliminates sample-size bias
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_boundaries = np.percentile(confidences, percentiles)
    bin_boundaries[0] -= 1e-6
    bin_boundaries[-1] += 1e-6

    adaptive_ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = np.sum(in_bin)
        if count > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            adaptive_ece += (count / n_samples) * np.abs(conf_in_bin - acc_in_bin)

    return float(adaptive_ece)


def calculate_multiclass_brier(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Calculate Brier score supporting binary and multiclass probabilities."""
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    if len(classes) <= 2:
        pos_prob = probas[:, 1] if probas.ndim == 2 and probas.shape[1] == 2 else probas.flatten()
        return float(brier_score_loss(y_true, pos_prob))

    y_one_hot = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((probas - y_one_hot) ** 2, axis=1)))


def enforce_simplex_normalization(probas: np.ndarray) -> np.ndarray:
    """Enforces sum-to-one simplex probability constraint without negative values."""
    p = np.maximum(0.0, np.asarray(probas, dtype=float))
    if p.ndim == 1:
        return p
    if p.shape[1] == 1:
        return p
    p = np.nan_to_num(p, nan=1.0 / p.shape[1], posinf=1.0, neginf=0.0)
    row_sums = np.sum(p, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return p / row_sums


class BetaCalibrator(BaseEstimator, ClassifierMixin):
    """Beta Calibration for asymmetric tabular confidence calibration (Kull et al. AISTATS / EJS).

    Formula:
        p_cal = 1 / (1 + 1/exp(c) * (1-p)^b / p^a)
    Equivalent to Logistic Regression on [ln(p), -ln(1-p)].
    """

    def __init__(self, base_estimator: Any = None) -> None:
        self.base_estimator = base_estimator
        self.eps_ = 1e-7

    def _extract_features(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps_, 1.0 - self.eps_)
        x1 = np.log(p)
        x2 = -np.log(1.0 - p)
        return np.column_stack([x1, x2])

    def fit(self, X: Any, y: np.ndarray, cal_probas: Optional[np.ndarray] = None) -> "BetaCalibrator":
        if cal_probas is not None:
            raw_probs = np.asarray(cal_probas, dtype=float)
        elif self.base_estimator is not None and X is not None:
            raw_probs = self.base_estimator.predict_proba(X)
        elif X is not None:
            raw_probs = np.asarray(X, dtype=float)
        else:
            raise ValueError("Either X or cal_probas must be provided to fit.")

        if raw_probs.ndim == 2 and raw_probs.shape[1] == 2:
            p = raw_probs[:, 1]
        else:
            p = raw_probs.flatten()

        feats = self._extract_features(p)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(feats, y)
        self.lr_ = lr
        self.classes_ = getattr(lr, "classes_", np.unique(y))
        return self

    def calibrate_probas(self, raw_probs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "lr_") or self.lr_ is None:
            raise NotFittedError("BetaCalibrator is not fitted.")
        raw_probs = np.asarray(raw_probs, dtype=float)
        if raw_probs.ndim == 2 and raw_probs.shape[1] == 2:
            p = raw_probs[:, 1]
        else:
            p = raw_probs.flatten()
        feats = self._extract_features(p)
        return self.lr_.predict_proba(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.base_estimator is not None:
            raw_probs = self.base_estimator.predict_proba(X)
        else:
            raw_probs = X
        return self.calibrate_probas(raw_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if hasattr(self, "classes_") and self.classes_ is not None:
            return np.asarray(self.classes_)[idx]
        return idx


class TemperatureScaler(BaseEstimator, ClassifierMixin):
    """Guo et al. (ICML 2017) Temperature Scaling post-processing calibrator."""

    def __init__(self, base_estimator: Any, temperature: float = 1.0) -> None:
        self.base_estimator = base_estimator
        self.temperature = temperature

    def fit(self, X: Any, y: np.ndarray, cal_probas: Optional[np.ndarray] = None) -> "TemperatureScaler":
        if cal_probas is not None:
            raw_probs = np.asarray(cal_probas, dtype=float)
        elif self.base_estimator is not None and X is not None:
            raw_probs = self.base_estimator.predict_proba(X)
        else:
            raise ValueError("Either X or cal_probas must be provided to fit.")
        y_arr = np.asarray(y)

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

        if self.base_estimator is not None and hasattr(self.base_estimator, "classes_"):
            self.classes_ = self.base_estimator.classes_
        else:
            self.classes_ = np.unique(y_arr)
        return self

    def calibrate_probas(self, raw_probs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "temperature_") or self.temperature_ is None:
            raise NotFittedError("TemperatureScaler is not fitted.")
        raw_probs = np.asarray(raw_probs, dtype=float)
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_estimator.predict_proba(X)
        return self.calibrate_probas(raw_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if hasattr(self, "classes_") and self.classes_ is not None:
            return np.asarray(self.classes_)[idx]
        return idx


class DirichletCalibrator(BaseEstimator, ClassifierMixin):
    """Kull et al. (NeurIPS 2019) Dirichlet Calibration with L2 regularization for multiclass probability simplex."""

    def __init__(self, base_estimator: Any = None, l2_reg: float = 1.0) -> None:
        self.base_estimator = base_estimator
        self.l2_reg = l2_reg
        self.eps = 1e-12

    def fit(self, X: Any, y: np.ndarray, cal_probas: Optional[np.ndarray] = None) -> "DirichletCalibrator":
        if cal_probas is not None:
            raw_probs = np.asarray(cal_probas, dtype=float)
        elif self.base_estimator is not None and X is not None:
            raw_probs = self.base_estimator.predict_proba(X)
        elif X is not None:
            raw_probs = np.asarray(X, dtype=float)
        else:
            raise ValueError("Either X or cal_probas must be provided to fit.")

        clipped = np.clip(raw_probs, self.eps, 1.0 - self.eps)
        log_probs = np.log(clipped)

        # Multinomial Logistic Regression over log-probabilities
        lr = LogisticRegression(
            C=1.0 / max(1e-5, self.l2_reg),
            solver="lbfgs",
            max_iter=1000,
        )
        lr.fit(log_probs, y)
        self.lr_ = lr
        self.classes_ = getattr(lr, "classes_", np.unique(y))
        return self

    def calibrate_probas(self, raw_probs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "lr_") or self.lr_ is None:
            raise NotFittedError("DirichletCalibrator is not fitted.")
        raw_probs = np.asarray(raw_probs, dtype=float)
        clipped = np.clip(raw_probs, self.eps, 1.0 - self.eps)
        log_probs = np.log(clipped)
        return self.lr_.predict_proba(log_probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_estimator.predict_proba(X) if self.base_estimator is not None else X
        return self.calibrate_probas(raw_probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if hasattr(self, "classes_") and self.classes_ is not None:
            return np.asarray(self.classes_)[idx]
        return idx


class ProbabilityCalibrator:
    """Probability Calibrator managing Beta, Platt, Isotonic, and Temperature Scaling."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state if random_state is not None else get_settings().random_state

    def _create_calibrator_instance(self, method: str, base_fit: Any, n_classes: int) -> Tuple[Any, str]:
        if method in ("beta", "dirichlet"):
            if n_classes > 2:
                return DirichletCalibrator(base_estimator=base_fit), "dirichlet"
            return BetaCalibrator(base_estimator=base_fit), "beta"
        elif method == "temperature":
            return TemperatureScaler(base_estimator=base_fit), "temperature"
        return None, method

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
                pre_brier_score=None,
                post_brier_score=None,
                brier_score_lift=None,
                is_well_calibrated=None,
                status="skipped_regression_task",
            )
            return report, model

        X_data = X
        y_arr = np.asarray(y)
        warnings: list[str] = []

        # Label encoding at entry to prevent indexing and type errors (Defect C4 fix)
        le = LabelEncoder()
        y_enc = le.fit_transform(y_arr)
        classes_ = le.classes_
        n_classes = len(classes_)

        n_samples = len(X_data)
        if method is None:
            method = "beta" if n_samples < 2000 else "isotonic"

        effective_cv = min(cv, n_samples // n_classes) if n_classes > 0 else cv
        if effective_cv < 2:
            effective_cv = 2

        # 1. Pre-calibration probabilities via out-of-fold cross-validation
        y_eval = y_enc
        evaluation_mode = "out_of_fold"
        try:
            pre_probas = cross_val_predict(
                clone(model),
                X_data,
                y_enc,
                cv=effective_cv,
                method="predict_proba",
                n_jobs=1,
            )
            cal_oof_probas = pre_probas
            cal_fit_X = X_data
            cal_fit_y = y_enc
        except Exception as exc:
            # Defect H3: Honest provenance when cross_val_predict fails
            evaluation_mode = "in_sample"
            warnings.append(
                f"cross_val_predict failed on base model ({exc}); falling back to in_sample split for calibration."
            )
            from sklearn.model_selection import KFold, StratifiedShuffleSplit
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=self.random_state)
                tr_idx, cal_idx = next(sss.split(X_data, y_enc))
            except Exception:
                kf = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
                tr_idx, cal_idx = next(kf.split(X_data))
            X_tr = X_data.iloc[tr_idx] if hasattr(X_data, "iloc") else X_data[tr_idx]
            X_cal_split = X_data.iloc[cal_idx] if hasattr(X_data, "iloc") else X_data[cal_idx]
            m_tr = clone(model).fit(X_tr, y_enc[tr_idx])
            cal_oof_probas = m_tr.predict_proba(X_cal_split)
            cal_fit_X = X_cal_split
            cal_fit_y = y_enc[cal_idx]
            pre_probas = cal_oof_probas
            y_eval = cal_fit_y

        pre_probas = enforce_simplex_normalization(pre_probas)
        pre_brier = calculate_multiclass_brier(y_eval, pre_probas)
        pre_ece = calculate_ece(y_eval, pre_probas, n_bins=10)

        # 2. Out-of-fold calibration evaluation across cal_oof_probas to prevent calibration in-sample leakage
        temp_val: Optional[float] = None
        base_fit = clone(model).fit(X_data, y_enc)

        if method in ("beta", "dirichlet", "temperature"):
            min_class_count = int(np.min(np.bincount(y_eval))) if len(y_eval) > 0 else 0
            cal_eval_cv = min(effective_cv, min_class_count) if len(y_eval) >= 10 else 0
            if cal_eval_cv >= 2:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=cal_eval_cv, shuffle=True, random_state=self.random_state)
                oof_post_probas = np.zeros_like(pre_probas)
                for fit_idx, eval_idx in skf.split(cal_oof_probas, y_eval):
                    fold_cal, _ = self._create_calibrator_instance(method, None, n_classes)
                    fold_cal.fit(None, y_eval[fit_idx], cal_probas=cal_oof_probas[fit_idx])
                    oof_post_probas[eval_idx] = fold_cal.calibrate_probas(cal_oof_probas[eval_idx])
                post_probas = oof_post_probas
            else:
                from sklearn.model_selection import train_test_split
                cal_idx, eval_idx = train_test_split(
                    np.arange(len(y_eval)), test_size=0.5, random_state=self.random_state
                )
                fold_cal, _ = self._create_calibrator_instance(method, None, n_classes)
                fold_cal.fit(None, y_eval[cal_idx], cal_probas=cal_oof_probas[cal_idx])
                eval_post = fold_cal.calibrate_probas(cal_oof_probas[eval_idx])
                # Defect C3 Fix: post_probas must match eval_idx length
                post_probas = eval_post
                y_eval = y_eval[eval_idx]
                pre_probas = pre_probas[eval_idx]
                warnings.append("Sample size too small for k-fold calibration; evaluated on 50% holdout split.")

            # Fit final deployed model on all calibration data
            calibrated_model, method = self._create_calibrator_instance(method, base_fit, n_classes)
            calibrated_model.fit(cal_fit_X, cal_fit_y, cal_probas=cal_oof_probas)
            temp_val = getattr(calibrated_model, "temperature_", None)
        else:
            calibrated_model = CalibratedClassifierCV(
                estimator=clone(model),
                method="sigmoid" if method == "sigmoid" else "isotonic",
                cv=effective_cv,
            )
            calibrated_model.fit(X_data, y_enc)
            try:
                post_probas = cross_val_predict(
                    calibrated_model,
                    X_data,
                    y_enc,
                    cv=effective_cv,
                    method="predict_proba",
                    n_jobs=1,
                )
            except Exception:
                post_probas = calibrated_model.predict_proba(cal_fit_X)
                warnings.append("cross_val_predict failed for CalibratedClassifierCV; using in_sample predict_proba.")

        # Ensure classes_ are stored on deployed calibrated model with original label types (Defect C4 fix)
        calibrated_model.classes_ = classes_

        post_probas = enforce_simplex_normalization(post_probas)
        post_brier = calculate_multiclass_brier(y_eval, post_probas)
        post_ece = calculate_ece(y_eval, post_probas, n_bins=10)
        adaptive_ece = calculate_adaptive_ece(y_eval, post_probas, n_bins=10)

        brier_lift = pre_brier - post_brier
        ece_lift = pre_ece - post_ece
        is_well_calibrated = bool(post_ece <= get_settings().ece_tolerance)

        # 4. Split-Conformal Non-Conformity coverage evaluated on an untouched holdout
        empirical_coverage: Optional[float] = None
        cov_ci_low: Optional[float] = None
        cov_ci_high: Optional[float] = None
        min_n = get_settings().min_calibration_n

        n_cal = len(y_eval)
        q_level = float(np.ceil((n_cal + 1) * (1.0 - conformal_alpha)) / n_cal) if n_cal > 0 else float("inf")
        quantile_method_ = "higher"

        if q_level > 1.0:
            q_hat = float("inf")
            warnings.append(
                f"INSUFFICIENT_N: Quantile level {q_level:.3f} > 1.0; exact coverage cannot be guaranteed with n={n_cal}, alpha={conformal_alpha}. q_hat set to inf."
            )
        else:
            if post_probas.ndim == 1 or post_probas.shape[1] == 2:
                prob_cal_all = np.where(y_eval == 1, post_probas[:, 1], post_probas[:, 0])
            else:
                prob_cal_all = post_probas[np.arange(len(y_eval)), y_eval]
            non_conf_all = 1.0 - prob_cal_all
            q_hat = float(np.quantile(non_conf_all, q_level, method="higher"))

        if len(y_eval) >= min_n:
            from sklearn.model_selection import train_test_split
            cal_idx, val_idx = train_test_split(
                np.arange(len(y_eval)),
                test_size=0.50,
                random_state=self.random_state,
                stratify=y_eval if len(np.unique(y_eval)) == 2 else None,
            )
            if post_probas.ndim == 1 or post_probas.shape[1] == 2:
                prob_cal = np.where(y_eval[cal_idx] == 1, post_probas[cal_idx, 1], post_probas[cal_idx, 0])
                prob_val = np.where(y_eval[val_idx] == 1, post_probas[val_idx, 1], post_probas[val_idx, 0])
            else:
                prob_cal = post_probas[cal_idx, y_eval[cal_idx]]
                prob_val = post_probas[val_idx, y_eval[val_idx]]

            non_conf_cal = 1.0 - prob_cal
            non_conf_val = 1.0 - prob_val
            sub_q_level = float(np.ceil((len(cal_idx) + 1) * (1.0 - conformal_alpha)) / len(cal_idx))
            if sub_q_level <= 1.0:
                sub_q_hat = float(np.quantile(non_conf_cal, sub_q_level, method="higher"))
            else:
                sub_q_hat = float("inf")
                warnings.append(
                    f"INSUFFICIENT_N: Quantile level {sub_q_level:.3f} > 1.0 for holdout calibration; evaluated with universal prediction sets."
                )
            covered_mask = non_conf_val <= sub_q_hat
            empirical_coverage = float(np.mean(covered_mask))
            from ml_mcp.engine.stats import wilson_interval
            cov_ci_low, cov_ci_high = wilson_interval(int(np.sum(covered_mask)), len(val_idx), conf=0.95, min_n=min_n // 2)
        else:
            warnings.append(f"INSUFFICIENT_N: sample size ({len(y_eval)}) below minimum calibration threshold ({min_n}).")

        coverage_valid = (evaluation_mode == "out_of_fold")

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
            conformal_coverage=round(empirical_coverage, 4) if empirical_coverage is not None else None,
            conformal_alpha=conformal_alpha,
            coverage_ci_low=cov_ci_low,
            coverage_ci_high=cov_ci_high,
            oof_brier=float(post_brier),
            oof_ece=float(post_ece),
            evaluation_mode=evaluation_mode,
            coverage_valid=coverage_valid,
            y_eval=list(y_eval),
            post_probas=post_probas,
            q_hat=q_hat,
            quantile_method_=quantile_method_,
            warnings=warnings,
        )

        return report, calibrated_model
