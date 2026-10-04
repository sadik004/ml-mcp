"""Semi-Supervised Curriculum Pseudo-Labeling with FlexMatch Dynamic Thresholds and Conformal Singleton Sets."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class PseudoLabeler:
    """Harvests high-confidence unlabelled predictions using FlexMatch class-adaptive curriculum thresholds
    and peer-reviewed Conformal Prediction Singleton uncertainty set filters.

    Theoretical foundations:
        - FlexMatch Curriculum Learning: Zhang et al. (NeurIPS 2021)
        - Self-Adaptive Thresholding: Wang et al. (ICLR 2023, FreeMatch)
        - Conformal Prediction Singleton Sets: Angelopoulos & Bates (2023)
    """

    def __init__(
        self,
        base_threshold: float = 0.95,
        confidence_threshold: Optional[float] = None,
        min_threshold: float = 0.70,
        alpha: float = 0.10,
        random_state: Optional[int] = None,
    ) -> None:
        from ml_mcp.config import get_settings
        self.base_threshold = confidence_threshold if confidence_threshold is not None else base_threshold
        self.confidence_threshold = self.base_threshold
        self.min_threshold = min_threshold
        self.alpha = alpha
        self.random_state = random_state if random_state is not None else get_settings().random_state

    def compute_class_adaptive_thresholds(
        self, probas: np.ndarray, num_classes: int
    ) -> Dict[int, float]:
        """Calculates dynamic class thresholds tau_c(t) based on curriculum learning status sigma_c(t).

        Formula (Zhang et al. NeurIPS 2021):
            sigma_c = sum(I(argmax(p_i) == c and max(p_i) >= tau_base))
            tau_c = tau_base * max(min_ratio, (sigma_c + 1) / (max_{c'} sigma_{c'} + 1))
        """
        max_conf = np.max(probas, axis=1)
        argmax_c = np.argmax(probas, axis=1)

        learning_status = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            learning_status[c] = float(np.sum((argmax_c == c) & (max_conf >= self.base_threshold)))

        max_status = max(float(np.max(learning_status)), 1.0)
        thresholds: Dict[int, float] = {}

        for c in range(num_classes):
            ratio = (learning_status[c] + 1.0) / (max_status + 1.0)
            tau_c = self.base_threshold * max(self.min_threshold / self.base_threshold, ratio)
            thresholds[c] = round(float(np.clip(tau_c, self.min_threshold, self.base_threshold)), 4)

        return thresholds

    def compute_conformal_quantile(
        self,
        model: Any,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> float:
        """Computes empirical conformal nonconformity quantile q_hat at 1 - alpha statistical guarantee.

        Formulation (Angelopoulos & Bates 2023):
            s_i = 1 - P_hat(Y = y_i | X_i)
            q_hat = Quantile({s_i}, ceil((n + 1)(1 - alpha)) / n)
        """
        cal_probas = model.predict_proba(X_cal)
        n_cal = len(y_cal)
        scores = np.zeros(n_cal, dtype=np.float64)

        classes_ = getattr(model, "classes_", np.arange(cal_probas.shape[1]))
        class_to_idx = {c: idx for idx, c in enumerate(classes_)}

        for i in range(n_cal):
            true_label = y_cal[i]
            idx = class_to_idx.get(true_label, None)
            if idx is not None and idx < cal_probas.shape[1]:
                scores[i] = 1.0 - cal_probas[i, idx]
            else:
                scores[i] = 1.0

        p_level = min(1.0, float(np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal))
        q_hat = float(np.quantile(scores, p_level, method="higher" if hasattr(np, "quantile") else "linear"))
        return q_hat

    def refine_with_pseudo_labels(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_unlabelled: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """Harvests class-adaptive pseudo-labels satisfying conformal singleton uncertainty and retrains model."""
        X_tr = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        y_tr = np.asarray(y_train)
        X_unlab = X_unlabelled.to_numpy() if isinstance(X_unlabelled, pd.DataFrame) else np.asarray(X_unlabelled)

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must support predict_proba to compute confidence thresholds.")

        baseline_score = 0.0
        if X_val is not None and y_val is not None:
            val_preds = model.predict(X_val)
            baseline_score = float(accuracy_score(y_val, val_preds))

        # 1. Derive Calibration Quantile q_hat (Angelopoulos & Bates 2023)
        if X_val is not None and y_val is not None:
            X_cal = X_val.to_numpy() if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
            y_cal = np.asarray(y_val)
            q_hat = self.compute_conformal_quantile(model, X_cal, y_cal)
        else:
            # Out-of-fold calibration on training set if validation set not provided
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            oof_scores: List[float] = []
            for tr_idx, cal_idx in skf.split(X_tr, y_tr):
                fold_model = clone(model)
                fold_model.fit(X_tr[tr_idx], y_tr[tr_idx])
                fold_probas = fold_model.predict_proba(X_tr[cal_idx])
                for i, true_l in enumerate(y_tr[cal_idx]):
                    c_idx = np.where(fold_model.classes_ == true_l)[0]
                    p = fold_probas[i, c_idx[0]] if len(c_idx) > 0 else 0.0
                    oof_scores.append(1.0 - p)
            n_cal = len(oof_scores)
            p_level = min(1.0, float(np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal))
            q_hat = float(np.quantile(oof_scores, p_level))

        # Predict probabilities on unlabelled pool
        probas = model.predict_proba(X_unlab)
        num_classes = probas.shape[1]
        max_conf = np.max(probas, axis=1)
        argmax_c = np.argmax(probas, axis=1)
        pseudo_labels = model.predict(X_unlab)

        # 2. Compute FlexMatch Class-Adaptive Dynamic Thresholds tau_c(t)
        class_thresholds = self.compute_class_adaptive_thresholds(probas, num_classes)

        # 3. Mathematical Conformal Prediction Singleton Filter (|C(x)| == 1):
        # C(x) = { c : 1 - P_hat(Y = c | x) <= q_hat }  <=>  P_hat(Y = c | x) >= 1 - q_hat
        # Singleton (|C(x)| == 1) strictly requires:
        #   (a) Top-1 probability >= 1 - q_hat
        #   (b) Top-2 probability < 1 - q_hat
        p_cutoff = max(0.0, 1.0 - q_hat)
        sorted_probas = np.sort(probas, axis=1)
        second_conf = sorted_probas[:, -2] if num_classes > 1 else np.zeros(len(probas))

        harvest_mask = np.zeros(len(X_unlab), dtype=bool)
        per_class_harvested: Dict[str, int] = {str(c): 0 for c in range(num_classes)}

        for i in range(len(X_unlab)):
            c = argmax_c[i]
            tau_c = class_thresholds.get(c, self.base_threshold)

            # Mathematical singleton check:
            # Top-1 is in prediction set (>= p_cutoff) AND Top-2 is NOT in prediction set (< p_cutoff)
            is_singleton = bool((max_conf[i] >= p_cutoff) and (second_conf[i] < p_cutoff))

            # Conformal Singleton + FlexMatch Gate:
            if max_conf[i] >= tau_c and is_singleton:
                harvest_mask[i] = True
                per_class_harvested[str(c)] += 1

        harvested_count = int(np.sum(harvest_mask))
        total_unlabelled = len(X_unlab)
        harvested_ratio = float(harvested_count / total_unlabelled) if total_unlabelled > 0 else 0.0

        refined_model = clone(model)

        if harvested_count > 0:
            X_harvested = X_unlab[harvest_mask]
            y_harvested = pseudo_labels[harvest_mask]

            X_combined = np.vstack([X_tr, X_harvested])
            y_combined = np.concatenate([y_tr, y_harvested])

            refined_model.fit(X_combined, y_combined)
        else:
            refined_model = model

        refined_score = baseline_score
        if X_val is not None and y_val is not None:
            ref_preds = refined_model.predict(X_val)
            refined_score = float(accuracy_score(y_val, ref_preds))

        score_lift = refined_score - baseline_score

        stats = {
            "harvested_count": harvested_count,
            "harvested_ratio": round(harvested_ratio, 4),
            "confidence_threshold": self.base_threshold,
            "class_thresholds": class_thresholds,
            "conformal_quantile_q_hat": round(q_hat, 4),
            "conformal_singleton_cutoff": round(p_cutoff, 4),
            "per_class_harvested": per_class_harvested,
            "baseline_score": round(baseline_score, 4),
            "refined_score": round(refined_score, 4),
            "score_lift": round(score_lift, 4),
            "conformal_singleton_verified": True,
        }

        return stats, refined_model
