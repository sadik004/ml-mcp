"""Conformal Risk Control (CRC) Engine with Mondrian Class-Conditional Coverage and RAPS.

Theoretical Basis:
    - Romano, Y., Barber, R. F., & Candes, E. (NeurIPS 2020). "Classification with Valid and Equal
      Coverage for Inherent Subgroups." Guarantees finite-sample coverage per individual class:
      P(Y in C(X) | Y = k) >= 1 - alpha.
    - Angelopoulos, A. N., et al. (ICLR 2021). "Uncertainty Sets for Image and Tabular Classifiers
      via RAPS." Regularized Adaptive Prediction Sets penalize set cardinality to minimize variance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

LossType = Literal["misclassification", "fnr", "asymmetric_cost"]


class ConformalRiskControlEngine:
    """Enterprise Conformal Prediction Engine featuring Mondrian Class-Conditional Coverage and RAPS."""

    def __init__(self, num_grid_points: int = 1001) -> None:
        self.grid = np.linspace(0.0, 1.0, num_grid_points)

    def get_prediction_sets(
        self,
        probs: np.ndarray,
        lambda_val: Union[float, Dict[int, float]],
        k_reg: int = 2,
        lam_reg: float = 0.01,
        use_raps: bool = False,
    ) -> List[List[int]]:
        """Construct prediction sets for each sample given threshold(s).
        
        Supports standard quantile thresholding or Regularized Adaptive Prediction Sets (RAPS).
        """
        probs = np.asarray(probs, dtype=np.float64)
        n_samples, n_classes = probs.shape
        prediction_sets: List[List[int]] = []

        if isinstance(lambda_val, dict):
            # Mondrian / Class-conditional thresholds: class k included if probs[i, k] >= 1 - lambda_val[k]
            for i in range(n_samples):
                sample_set = [
                    k for k in range(n_classes)
                    if probs[i, k] >= (1.0 - lambda_val.get(k, 1.0))
                ]
                prediction_sets.append(sample_set)
        elif use_raps:
            # RAPS: sort probabilities descending, accumulate with penalty for rank > k_reg
            cutoff = float(lambda_val)
            for i in range(n_samples):
                sorted_indices = np.argsort(-probs[i])
                sorted_probs = probs[i][sorted_indices]
                cum_probs = np.cumsum(sorted_probs)
                
                # Apply cardinality penalty for ranks beyond k_reg
                ranks = np.arange(1, n_classes + 1)
                penalties = lam_reg * np.maximum(0, ranks - k_reg)
                reg_scores = cum_probs + penalties

                # Include classes until score threshold is reached
                included_count = int(np.searchsorted(reg_scores, cutoff, side="left")) + 1
                included_count = min(included_count, n_classes)
                sample_set = sorted_indices[:included_count].tolist()
                prediction_sets.append(sample_set)
        else:
            # Global marginal threshold
            cutoff = 1.0 - float(lambda_val)
            for i in range(n_samples):
                sample_set = np.where(probs[i] >= cutoff)[0].tolist()
                prediction_sets.append(sample_set)

        return prediction_sets

    def evaluate_loss(
        self,
        prediction_sets: List[List[int]],
        y_true: np.ndarray,
        loss_type: LossType = "misclassification",
        cost_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute sample-wise bounded loss in [0, B]."""
        y_true = np.asarray(y_true, dtype=int)
        n = len(y_true)
        losses = np.zeros(n, dtype=np.float64)

        if loss_type == "misclassification":
            for i in range(n):
                losses[i] = 1.0 if y_true[i] not in prediction_sets[i] else 0.0

        elif loss_type == "fnr":
            for i in range(n):
                if y_true[i] == 1:
                    losses[i] = 1.0 if 1 not in prediction_sets[i] else 0.0
                else:
                    losses[i] = 0.0

        elif loss_type == "asymmetric_cost":
            for i in range(n):
                pset = prediction_sets[i]
                y_i = y_true[i]
                if len(pset) == 0:
                    losses[i] = 1.0
                elif len(pset) > 1:
                    losses[i] = 0.2 if cost_matrix is None else float(cost_matrix[y_i, -1])
                else:
                    pred = pset[0]
                    if pred == y_i:
                        losses[i] = 0.0
                    else:
                        if cost_matrix is not None:
                            losses[i] = float(cost_matrix[y_i, pred])
                        else:
                            losses[i] = 1.0 if y_i == 1 else 0.2
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        return losses

    def calibrate(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
        loss_type: LossType = "misclassification",
        target_risk: float = 0.05,
        cost_matrix: Optional[np.ndarray] = None,
        mondrian: bool = False,
        use_raps: bool = False,
    ) -> Tuple[Union[float, Dict[int, float]], float]:
        """Calibrate lambda to strictly satisfy finite-sample risk guarantee:
            E[loss] <= alpha (target_risk)
        
        Theoretical Basis:
            - Romano et al. (2020) Mondrian Class-Conditional Quantiles.
            - Angelopoulos et al. (2021) RAPS.
        """
        probs_cal = np.asarray(probs_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=int)
        n = len(y_cal)

        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        B = float(np.max(cost_matrix)) if (loss_type == "asymmetric_cost" and cost_matrix is not None) else 1.0

        if mondrian and loss_type == "misclassification":
            # Mondrian (Class-Conditional) Conformal Prediction (Romano et al. NeurIPS 2020)
            classes = np.unique(y_cal)
            per_class_lambdas: Dict[int, float] = {}

            for c in classes:
                mask = (y_cal == c)
                n_c = int(np.sum(mask))
                if n_c == 0:
                    per_class_lambdas[int(c)] = 1.0
                    continue

                probs_c = probs_cal[mask]
                y_c = y_cal[mask]

                selected_lam = 1.0
                for lam in self.grid:
                    psets = self.get_prediction_sets(probs_c, float(lam))
                    l_vals = self.evaluate_loss(psets, y_c, loss_type="misclassification")
                    r_hat = np.mean(l_vals)
                    corrected_risk = (n_c / (n_c + 1.0)) * r_hat + (B / (n_c + 1.0))
                    if corrected_risk <= target_risk:
                        selected_lam = float(lam)
                        break

                per_class_lambdas[int(c)] = selected_lam

            all_psets = self.get_prediction_sets(probs_cal, per_class_lambdas)
            emp_risk = float(np.mean(self.evaluate_loss(all_psets, y_cal, loss_type)))
            return per_class_lambdas, emp_risk

        # Marginal calibration (with optional RAPS support)
        if loss_type == "fnr":
            pos_mask = (y_cal == 1)
            n_eff = int(np.sum(pos_mask))
            if n_eff == 0:
                return 1.0, 0.0
            eval_probs = probs_cal[pos_mask]
            eval_y = y_cal[pos_mask]
        else:
            n_eff = n
            eval_probs = probs_cal
            eval_y = y_cal

        selected_lambda = 1.0
        for lam in self.grid:
            psets = self.get_prediction_sets(eval_probs, float(lam), use_raps=use_raps)
            l_vals = self.evaluate_loss(psets, eval_y, loss_type=loss_type, cost_matrix=cost_matrix)
            r_hat = np.mean(l_vals)
            corrected_risk = (n_eff / (n_eff + 1.0)) * r_hat + (B / (n_eff + 1.0))

            if corrected_risk <= target_risk:
                selected_lambda = float(lam)
                break

        full_psets = self.get_prediction_sets(probs_cal, selected_lambda, use_raps=use_raps)
        if loss_type == "fnr":
            empirical_risk = float(np.mean(self.evaluate_loss(full_psets, y_cal, loss_type)[y_cal == 1]))
        else:
            empirical_risk = float(np.mean(self.evaluate_loss(full_psets, y_cal, loss_type, cost_matrix)))

        return selected_lambda, empirical_risk

    def predict_and_triage(
        self,
        probs: np.ndarray,
        lambda_val: Union[float, Dict[int, float]],
        use_raps: bool = False,
    ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        """Generate prediction sets and identify human triage escalations."""
        probs = np.asarray(probs, dtype=np.float64)
        psets = self.get_prediction_sets(probs, lambda_val, use_raps=use_raps)
        records: List[Dict[str, Any]] = []

        for i, pset in enumerate(psets):
            is_empty = (len(pset) == 0)
            is_ambiguous = (len(pset) > 1)
            needs_review = is_empty or is_ambiguous

            prob_dict = {int(k): round(float(probs[i, k]), 4) for k in range(probs.shape[1])}
            records.append({
                "sample_index": i,
                "prediction_set": [int(x) for x in pset],
                "set_size": len(pset),
                "is_ambiguous": is_ambiguous,
                "is_empty": is_empty,
                "needs_human_review": needs_review,
                "probabilities": prob_dict,
            })

        return psets, records
