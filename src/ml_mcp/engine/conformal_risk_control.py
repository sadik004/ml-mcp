"""Conformal Risk Control (CRC) Engine with pure NumPy/SciPy implementation.

Controls arbitrary bounded loss functions:
    E[loss(C_lambda(X), Y)] <= alpha

Supports:
1. 0-1 Misclassification Loss (Standard Conformal Prediction: E[1(Y not in C(X))] <= alpha)
2. False Negative Rate (FNR) Control for high-risk / medical / fraud domains: E[1(1 not in C(X)) | Y=1] <= alpha
3. Custom Bounded Asymmetric Cost Matrix / Financial Loss Control: E[Cost(C(X), Y)] <= alpha
4. Mondrian (Class-Conditional) Conformal Risk Control
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

LossType = Literal["misclassification", "fnr", "asymmetric_cost"]


class ConformalRiskControlEngine:
    """Zero-dependency, pure NumPy/SciPy Conformal Risk Control engine."""

    def __init__(self, num_grid_points: int = 1001) -> None:
        self.grid = np.linspace(0.0, 1.0, num_grid_points)

    def get_prediction_sets(
        self,
        probs: np.ndarray,
        lambda_val: Union[float, Dict[int, float]],
    ) -> List[List[int]]:
        """Construct prediction sets for each sample given lambda threshold(s).
        
        A class k is included in C(x) if:
            probs[i, k] >= 1.0 - lambda_val (or lambda_val[k] for Mondrian)
        """
        probs = np.asarray(probs, dtype=np.float64)
        n_samples, n_classes = probs.shape
        prediction_sets: List[List[int]] = []

        if isinstance(lambda_val, dict):
            # Mondrian / Class-conditional thresholds
            for i in range(n_samples):
                sample_set = [
                    k for k in range(n_classes)
                    if probs[i, k] >= (1.0 - lambda_val.get(k, 1.0))
                ]
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
        """Compute sample-wise bounded loss in [0, B].
        
        Returns:
            1D array of loss values for each sample.
        """
        y_true = np.asarray(y_true, dtype=int)
        n = len(y_true)
        losses = np.zeros(n, dtype=np.float64)

        if loss_type == "misclassification":
            for i in range(n):
                # 0-1 loss: 1 if true class not in prediction set, 0 otherwise
                losses[i] = 1.0 if y_true[i] not in prediction_sets[i] else 0.0

        elif loss_type == "fnr":
            # False Negative Rate: penalizes if positive class (1) is omitted
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
                    losses[i] = 1.0  # Empty set penalty
                elif len(pset) > 1:
                    # Ambiguous set penalty (human escalation triage cost)
                    losses[i] = 0.2 if cost_matrix is None else float(cost_matrix[y_i, -1])
                else:
                    pred = pset[0]
                    if pred == y_i:
                        losses[i] = 0.0
                    else:
                        if cost_matrix is not None:
                            losses[i] = float(cost_matrix[y_i, pred])
                        else:
                            # Default asymmetric cost: FN (missed fraud) is 1.0, FP is 0.2
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
    ) -> Tuple[Union[float, Dict[int, float]], float]:
        """Calibrate lambda to strictly satisfy finite-sample risk guarantee:
            E[loss] <= alpha (target_risk)
        
        Using CRC theorem (Angelopoulos et al. 2022):
            R_hat_n(lambda) * n / (n + 1) + B / (n + 1) <= alpha
            where B = max loss bound.
        """
        probs_cal = np.asarray(probs_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=int)
        n = len(y_cal)

        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Determine loss upper bound B
        if loss_type == "asymmetric_cost" and cost_matrix is not None:
            B = float(np.max(cost_matrix))
        else:
            B = 1.0

        if mondrian and loss_type == "misclassification":
            # Class-conditional calibration (Mondrian CRC)
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

        # Marginal calibration
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
            psets = self.get_prediction_sets(eval_probs, float(lam))
            l_vals = self.evaluate_loss(psets, eval_y, loss_type=loss_type, cost_matrix=cost_matrix)
            r_hat = np.mean(l_vals)
            corrected_risk = (n_eff / (n_eff + 1.0)) * r_hat + (B / (n_eff + 1.0))

            if corrected_risk <= target_risk:
                selected_lambda = float(lam)
                break

        full_psets = self.get_prediction_sets(probs_cal, selected_lambda)
        if loss_type == "fnr":
            empirical_risk = float(np.mean(self.evaluate_loss(full_psets, y_cal, loss_type)[y_cal == 1]))
        else:
            empirical_risk = float(np.mean(self.evaluate_loss(full_psets, y_cal, loss_type, cost_matrix)))

        return selected_lambda, empirical_risk

    def predict_and_triage(
        self,
        probs: np.ndarray,
        lambda_val: Union[float, Dict[int, float]],
    ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        """Generate prediction sets and identify human triage escalations.
        
        Returns:
            Tuple of (prediction_sets, triage_records)
        """
        probs = np.asarray(probs, dtype=np.float64)
        psets = self.get_prediction_sets(probs, lambda_val)
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
