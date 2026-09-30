"""Decision Threshold Optimizer and Error Forensics Engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ml_mcp.schemas.tuning import ThresholdReportDTO

logger = logging.getLogger(__name__)


class DecisionThresholdOptimizer:
    """Optimizes classification decision thresholds using F-beta and performs error forensics."""

    def __init__(self, default_threshold: float = 0.50) -> None:
        self.default_threshold = default_threshold

    def optimize(
        self,
        y_true: Union[np.ndarray, list],
        y_probas: Union[np.ndarray, list],
        beta: float = 1.0,
        grid_steps: int = 100,
    ) -> ThresholdReportDTO:
        """Find optimal probability threshold maximizing F-beta score.
        
        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_probas: Predicted probabilities for the positive class (1).
            beta: Weight of recall relative to precision in F-beta metric (beta=2 prioritizes recall).
            grid_steps: Number of candidate thresholds evaluated between 0.01 and 0.99.
            
        Returns:
            ThresholdReportDTO with optimal threshold, precision, recall, and error forensics.
        """
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(y_probas, dtype=float)

        if len(y) != len(p):
            raise ValueError(f"Length mismatch: len(y_true)={len(y)} != len(y_probas)={len(p)}")

        # Handle edge cases: all zeros or all ones in ground truth
        total_positives = int(np.sum(y == 1))
        total_negatives = int(np.sum(y == 0))

        if total_positives == 0:
            # No positive examples exist
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
            )

        if total_negatives == 0:
            # All examples are positive
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
            )

        thresholds = np.linspace(0.01, 0.99, grid_steps)
        best_f_beta = -1.0
        best_threshold = self.default_threshold
        best_precision = 0.0
        best_recall = 0.0

        beta_sq = beta ** 2

        for t in thresholds:
            y_pred = (p >= t).astype(int)
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f_beta = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
            else:
                f_beta = 0.0

            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_threshold = float(t)
                best_precision = float(precision)
                best_recall = float(recall)

        # Fallback if no threshold gave positive score
        if best_f_beta < 0.0:
            best_threshold = self.default_threshold
            best_f_beta = 0.0

        # Calculate final confusion matrix and forensics at optimal threshold
        final_pred = (p >= best_threshold).astype(int)
        tn = int(np.sum((y == 0) & (final_pred == 0)))
        fp = int(np.sum((y == 0) & (final_pred == 1)))
        fn = int(np.sum((y == 1) & (final_pred == 0)))
        tp = int(np.sum((y == 1) & (final_pred == 1)))

        cm = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        return ThresholdReportDTO(
            default_threshold=self.default_threshold,
            optimal_threshold=round(best_threshold, 4),
            f_beta_score=round(best_f_beta, 4),
            precision=round(best_precision, 4),
            recall=round(best_recall, 4),
            confusion_matrix=cm,
            false_positive_count=fp,
            false_negative_count=fn,
        )
