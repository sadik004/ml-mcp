"""Decision Threshold Optimizer, Cost-Sensitive Matrix, and Error Forensics Engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import precision_recall_curve

from ml_mcp.schemas.tuning import ThresholdReportDTO

logger = logging.getLogger(__name__)


class DecisionThresholdOptimizer:
    """Optimizes classification decision thresholds using cost-loss matrices and PR-curve cutoffs.

    Theoretical Basis:
        - Sheng, V. S., & Ling, C. X. (2014). Thresholding for making better decisions in cost-sensitive learning.
          IEEE Transactions on Knowledge and Data Engineering.
        - Hernández-Orallo, J., Flach, P., & Ferri, C. (2013). Brier curves and cost curves.
          Machine Learning / JMLR.
        - Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
    """

    def __init__(self, default_threshold: float = 0.50) -> None:
        self.default_threshold = default_threshold

    def calculate_analytical_threshold(
        self,
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
    ) -> float:
        """Computes closed-form theoretical optimal threshold (Sheng & Ling 2014).

        Formula:
            theta* = (C_FP - B_TN) / ((C_FP - B_TN) + (C_FN - B_TP))
        """
        numerator = cost_fp - benefit_tn
        denominator = (cost_fp - benefit_tn) + (cost_fn - benefit_tp)
        if denominator <= 0:
            return 0.50
        theta_star = numerator / denominator
        return float(np.clip(theta_star, 0.001, 0.999))

    def calculate_financial_loss(
        self,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
    ) -> float:
        """Calculates total net financial loss given confusion counts."""
        return float((fp * cost_fp) + (fn * cost_fn) - (tp * benefit_tp) - (tn * benefit_tn))

    def optimize(
        self,
        y_true: Union[np.ndarray, list],
        y_probas: Union[np.ndarray, list],
        beta: float = 1.0,
        criterion: Literal["f_beta", "cost_loss"] = "f_beta",
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
    ) -> ThresholdReportDTO:
        """Find optimal probability threshold using PR-curve exact cutoffs and cost-sensitive loss.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_probas: Predicted probabilities for the positive class (1).
            beta: Relative weight of recall to precision in F-beta (default 1.0).
            criterion: 'f_beta' (default) or 'cost_loss' (minimizes financial loss).
            cost_fp: Cost of a False Positive (user annoyance / review overhead).
            cost_fn: Cost of a False Negative (missed fraud / false negative disease).
            benefit_tp: Economic gain of a True Positive.
            benefit_tn: Economic gain of a True Negative.

        Returns:
            ThresholdReportDTO with optimal threshold, savings, and full confusion matrix.
        """
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(y_probas, dtype=float)

        if len(y) != len(p):
            raise ValueError(f"Length mismatch: len(y_true)={len(y)} != len(y_probas)={len(p)}")

        total_positives = int(np.sum(y == 1))
        total_negatives = int(np.sum(y == 0))

        # Analytical baseline threshold
        analytical_theta = self.calculate_analytical_threshold(
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            benefit_tp=benefit_tp,
            benefit_tn=benefit_tn,
        )

        # Edge cases: all zeros or all ones in ground truth
        if total_positives == 0:
            cm = {"tn": total_negatives, "fp": 0, "fn": 0, "tp": 0}
            return ThresholdReportDTO(
                default_threshold=self.default_threshold,
                optimal_threshold=self.default_threshold,
                f_beta_score=0.0,
                precision=0.0,
                recall=0.0,
                confusion_matrix=cm,
                false_positive_count=0,
                false_negative_count=0,
                total_cost_optimal=0.0,
                total_cost_default=0.0,
                cost_savings=0.0,
                analytical_cost_threshold=analytical_theta,
            )

        if total_negatives == 0:
            cm = {"tn": 0, "fp": 0, "fn": 0, "tp": total_positives}
            return ThresholdReportDTO(
                default_threshold=self.default_threshold,
                optimal_threshold=self.default_threshold,
                f_beta_score=1.0,
                precision=1.0,
                recall=1.0,
                confusion_matrix=cm,
                false_positive_count=0,
                false_negative_count=0,
                total_cost_optimal=0.0,
                total_cost_default=0.0,
                cost_savings=0.0,
                analytical_cost_threshold=analytical_theta,
            )

        # 1. Eliminate 100-step linear grid: derive exact cutoffs from precision_recall_curve
        _, _, pr_thresholds = precision_recall_curve(y, p)

        # Combine PR-curve cutoffs + analytical threshold + fine tail percentiles + default 0.50
        candidate_pool = set(pr_thresholds.tolist())
        candidate_pool.add(analytical_theta)
        candidate_pool.add(self.default_threshold)
        # Add fine-grained tail percentiles to ensure extreme imbalanced cutoffs are evaluated
        tail_percentiles = np.percentile(p, np.linspace(0.5, 99.5, 50)).tolist()
        for t_p in tail_percentiles:
            if 0.001 < t_p < 0.999:
                candidate_pool.add(t_p)

        thresholds = np.array(sorted([t for t in candidate_pool if 0.0 < t < 1.0]))
        if len(thresholds) == 0:
            thresholds = np.array([self.default_threshold])

        # 2. Evaluate performance and financial loss across all candidate thresholds
        beta_sq = beta ** 2
        best_f_beta = -1.0
        min_total_cost = float("inf")
        best_threshold_fbeta = self.default_threshold
        best_threshold_cost = self.default_threshold

        best_prec_fbeta = 0.0
        best_rec_fbeta = 0.0
        best_prec_cost = 0.0
        best_rec_cost = 0.0

        for t in thresholds:
            y_pred = (p >= t).astype(int)
            tp = int(np.sum((y == 1) & (y_pred == 1)))
            fp = int(np.sum((y == 0) & (y_pred == 1)))
            fn = int(np.sum((y == 1) & (y_pred == 0)))
            tn = int(np.sum((y == 0) & (y_pred == 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f_beta = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
            else:
                f_beta = 0.0

            cost = self.calculate_financial_loss(
                tp=tp, fp=fp, fn=fn, tn=tn,
                cost_fp=cost_fp, cost_fn=cost_fn,
                benefit_tp=benefit_tp, benefit_tn=benefit_tn,
            )

            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_threshold_fbeta = float(t)
                best_prec_fbeta = float(precision)
                best_rec_fbeta = float(recall)

            if cost < min_total_cost:
                min_total_cost = cost
                best_threshold_cost = float(t)
                best_prec_cost = float(precision)
                best_rec_cost = float(recall)

        # 3. Select final optimal threshold according to criterion
        if criterion == "cost_loss":
            final_optimal_threshold = best_threshold_cost
            final_precision = best_prec_cost
            final_recall = best_rec_cost
        else:
            final_optimal_threshold = best_threshold_fbeta
            final_precision = best_prec_fbeta
            final_recall = best_rec_fbeta

        # 4. Compute baseline financial loss at default 0.50 threshold
        default_pred = (p >= self.default_threshold).astype(int)
        def_tp = int(np.sum((y == 1) & (default_pred == 1)))
        def_fp = int(np.sum((y == 0) & (default_pred == 1)))
        def_fn = int(np.sum((y == 1) & (default_pred == 0)))
        def_tn = int(np.sum((y == 0) & (default_pred == 0)))
        default_cost = self.calculate_financial_loss(
            tp=def_tp, fp=def_fp, fn=def_fn, tn=def_tn,
            cost_fp=cost_fp, cost_fn=cost_fn,
            benefit_tp=benefit_tp, benefit_tn=benefit_tn,
        )

        # Compute optimal confusion counts & net loss
        opt_pred = (p >= final_optimal_threshold).astype(int)
        opt_tp = int(np.sum((y == 1) & (opt_pred == 1)))
        opt_fp = int(np.sum((y == 0) & (opt_pred == 1)))
        opt_fn = int(np.sum((y == 1) & (opt_pred == 0)))
        opt_tn = int(np.sum((y == 0) & (opt_pred == 0)))

        optimal_cost = self.calculate_financial_loss(
            tp=opt_tp, fp=opt_fp, fn=opt_fn, tn=opt_tn,
            cost_fp=cost_fp, cost_fn=cost_fn,
            benefit_tp=benefit_tp, benefit_tn=benefit_tn,
        )
        savings = float(default_cost - optimal_cost)

        # Compute F-beta at optimal threshold if criterion was cost_loss
        if criterion == "cost_loss":
            if final_precision + final_recall > 0:
                opt_f_beta = (1.0 + beta_sq) * (final_precision * final_recall) / ((beta_sq * final_precision) + final_recall)
            else:
                opt_f_beta = 0.0
        else:
            opt_f_beta = max(0.0, best_f_beta)

        cm = {"tn": opt_tn, "fp": opt_fp, "fn": opt_fn, "tp": opt_tp}

        return ThresholdReportDTO(
            default_threshold=self.default_threshold,
            optimal_threshold=round(final_optimal_threshold, 4),
            f_beta_score=round(float(opt_f_beta), 4),
            precision=round(final_precision, 4),
            recall=round(final_recall, 4),
            confusion_matrix=cm,
            false_positive_count=opt_fp,
            false_negative_count=opt_fn,
            total_cost_optimal=round(optimal_cost, 2),
            total_cost_default=round(default_cost, 2),
            cost_savings=round(savings, 2),
            analytical_cost_threshold=round(analytical_theta, 4),
        )
