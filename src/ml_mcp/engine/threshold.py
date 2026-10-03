"""Decision Threshold Optimizer with Decision Curve Analysis (DCA - Vickers & Elkin)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import precision_recall_curve

from ml_mcp.schemas.tuning import ThresholdReportDTO

logger = logging.getLogger(__name__)


def calculate_decision_curve_analysis(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes Decision Curve Analysis (Net Benefit Curve - Vickers & Elkin BMJ/Lancet).
    
    Net Benefit(pt) = TP/N - FP/N * (pt / (1 - pt))
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_probas, dtype=float)
    N = len(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    net_benefits: List[float] = []
    treat_all: List[float] = []
    treat_none: List[float] = [0.0] * len(thresholds)

    for pt in thresholds:
        weight = pt / (1.0 - pt)
        y_pred = (p >= pt).astype(int)
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))

        nb_model = (tp / N) - (fp / N) * weight
        nb_all = (n_pos / N) - (n_neg / N) * weight

        net_benefits.append(float(nb_model))
        treat_all.append(float(nb_all))

    # Identify optimal operating zone: thresholds where model exceeds both Treat All and Treat None
    superior_indices = [
        i for i, (nb, na) in enumerate(zip(net_benefits, treat_all))
        if nb > max(na, 0.0)
    ]
    if superior_indices:
        optimal_zone = (float(thresholds[superior_indices[0]]), float(thresholds[superior_indices[-1]]))
    else:
        optimal_zone = (0.0, 0.0)

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit": net_benefits,
        "treat_all": treat_all,
        "treat_none": treat_none,
        "optimal_zone": optimal_zone,
    }


class DecisionThresholdOptimizer:
    """Optimizes classification boundaries balancing cost, F-beta, and Decision Curve Analysis."""

    def __init__(self, default_threshold: float = 0.50) -> None:
        self.default_threshold = default_threshold

    @staticmethod
    def calculate_analytical_threshold(
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
    ) -> float:
        """Computes Sheng & Ling (2014) closed-form cost-matrix threshold."""
        numerator = cost_fp - benefit_tn
        denominator = (cost_fp - benefit_tn) + (cost_fn - benefit_tp)
        if denominator <= 0:
            return 0.50
        theta = numerator / denominator
        return float(np.clip(theta, 0.01, 0.99))

    @staticmethod
    def calculate_financial_loss(
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
    ) -> float:
        """Calculates total net financial cost from confusion matrix."""
        return float(fp * cost_fp + fn * cost_fn - tp * benefit_tp - tn * benefit_tn)

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
        """Optimizes decision threshold across precision-recall cutoffs and DCA Net Benefit."""
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(y_probas, dtype=float)

        if len(y) != len(p):
            raise ValueError(f"Length mismatch: len(y_true)={len(y)} != len(y_probas)={len(p)}")

        total_positives = int(np.sum(y == 1))
        total_negatives = int(np.sum(y == 0))

        analytical_theta = self.calculate_analytical_threshold(
            cost_fp=cost_fp, cost_fn=cost_fn, benefit_tp=benefit_tp, benefit_tn=benefit_tn
        )

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

        _, _, pr_thresholds = precision_recall_curve(y, p)
        candidate_pool = set(pr_thresholds.tolist())
        candidate_pool.add(analytical_theta)
        candidate_pool.add(self.default_threshold)
        tail_percentiles = np.percentile(p, np.linspace(0.5, 99.5, 50)).tolist()
        for t_p in tail_percentiles:
            if 0.001 < t_p < 0.999:
                candidate_pool.add(t_p)

        thresholds = np.array(sorted([t for t in candidate_pool if 0.0 < t < 1.0]))
        if len(thresholds) == 0:
            thresholds = np.array([self.default_threshold])

        beta_sq = beta ** 2
        best_f_beta = -1.0
        min_total_cost = float("inf")
        best_threshold_fbeta = self.default_threshold
        best_threshold_cost = self.default_threshold
        best_prec_fbeta, best_rec_fbeta = 0.0, 0.0
        best_prec_cost, best_rec_cost = 0.0, 0.0

        for t in thresholds:
            y_pred = (p >= t).astype(int)
            tp = int(np.sum((y == 1) & (y_pred == 1)))
            fp = int(np.sum((y == 0) & (y_pred == 1)))
            fn = int(np.sum((y == 1) & (y_pred == 0)))
            tn = int(np.sum((y == 0) & (y_pred == 0)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f_b = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
            else:
                f_b = 0.0

            cost = self.calculate_financial_loss(
                tp=tp, fp=fp, fn=fn, tn=tn,
                cost_fp=cost_fp, cost_fn=cost_fn,
                benefit_tp=benefit_tp, benefit_tn=benefit_tn,
            )

            if f_b > best_f_beta:
                best_f_beta = f_b
                best_threshold_fbeta = float(t)
                best_prec_fbeta = float(precision)
                best_rec_fbeta = float(recall)

            if cost < min_total_cost:
                min_total_cost = cost
                best_threshold_cost = float(t)
                best_prec_cost = float(precision)
                best_rec_cost = float(recall)

        if criterion == "cost_loss":
            final_optimal_threshold = best_threshold_cost
            final_precision = best_prec_cost
            final_recall = best_rec_cost
            opt_f_beta = (1.0 + beta_sq) * (final_precision * final_recall) / ((beta_sq * final_precision) + final_recall) if (final_precision + final_recall > 0) else 0.0
        else:
            final_optimal_threshold = best_threshold_fbeta
            final_precision = best_prec_fbeta
            final_recall = best_rec_fbeta
            opt_f_beta = max(0.0, best_f_beta)

        default_pred = (p >= self.default_threshold).astype(int)
        def_tp = int(np.sum((y == 1) & (default_pred == 1)))
        def_fp = int(np.sum((y == 0) & (default_pred == 1)))
        def_fn = int(np.sum((y == 1) & (default_pred == 0)))
        def_tn = int(np.sum((y == 0) & (default_pred == 0)))
        default_cost = self.calculate_financial_loss(def_tp, def_fp, def_fn, def_tn, cost_fp, cost_fn, benefit_tp, benefit_tn)

        opt_pred = (p >= final_optimal_threshold).astype(int)
        opt_tp = int(np.sum((y == 1) & (opt_pred == 1)))
        opt_fp = int(np.sum((y == 0) & (opt_pred == 1)))
        opt_fn = int(np.sum((y == 1) & (opt_pred == 0)))
        opt_tn = int(np.sum((y == 0) & (opt_pred == 0)))
        optimal_cost = self.calculate_financial_loss(opt_tp, opt_fp, opt_fn, opt_tn, cost_fp, cost_fn, benefit_tp, benefit_tn)

        # DCA Net Benefit calculation at optimal threshold
        pt = final_optimal_threshold
        weight = pt / (1.0 - pt) if pt < 1.0 else 1.0
        dca_nb = float((opt_tp / len(y)) - (opt_fp / len(y)) * weight)
        dca_all = float((total_positives / len(y)) - (total_negatives / len(y)) * weight)

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
            cost_savings=round(float(default_cost - optimal_cost), 2),
            analytical_cost_threshold=round(analytical_theta, 4),
            dca_net_benefit=round(dca_nb, 4),
            dca_treat_all_net_benefit=round(dca_all, 4),
        )
