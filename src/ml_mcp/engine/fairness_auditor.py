"""Demographic Slice Fairness and Disparate Impact Auditor."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from ml_mcp.schemas.safety import SliceFairnessDTO

logger = logging.getLogger(__name__)


class SliceFairnessAuditor:
    """Audits model performance across demographic subgroups to detect disparate impact and bias."""

    def __init__(self, min_slice_samples: int = 2) -> None:
        self.min_slice_samples = min_slice_samples

    def _calculate_slice_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate performance score for a subgroup slice."""
        if len(y_true) == 0:
            return 0.0

        metric_lower = metric.lower()
        if "f1" in metric_lower:
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif "r2" in metric_lower:
            return float(r2_score(y_true, y_pred))
        else:
            # Default to accuracy
            return float(accuracy_score(y_true, y_pred))

    def audit(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        protected_series: Union[pd.Series, np.ndarray, list],
        protected_attribute: str = "protected_attribute",
        metric: str = "accuracy",
    ) -> SliceFairnessDTO:
        """Execute demographic slice fairness audit and enforce US EEOC 80% rule."""
        X_arr = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
        y_arr = np.asarray(y_test)
        groups = np.asarray(protected_series)

        if len(X_arr) != len(groups):
            raise ValueError(f"Length mismatch: X_test ({len(X_arr)}) and protected_series ({len(groups)})")

        preds = model.predict(X_arr)
        unique_groups = np.unique(groups)

        subgroup_scores: Dict[str, float] = {}
        for grp in unique_groups:
            mask = groups == grp
            if np.sum(mask) >= self.min_slice_samples:
                score = self._calculate_slice_metric(y_arr[mask], preds[mask], metric)
                subgroup_scores[str(grp)] = round(float(score), 4)

        if not subgroup_scores:
            # Fallback if no subgroups met min_samples
            overall_score = self._calculate_slice_metric(y_arr, preds, metric)
            subgroup_scores["all"] = round(float(overall_score), 4)

        scores_list = list(subgroup_scores.values())
        min_score = min(scores_list)
        max_score = max(scores_list)
        max_disparity = float(max_score - min_score)

        # Disparate impact ratio: min_score / max_score (Four-Fifths 80% Rule)
        if max_score > 1e-6:
            dir_ratio = float(min_score / max_score)
        else:
            dir_ratio = 1.0 if min_score == max_score else 0.0

        # Parity violated if ratio is below 80% threshold (0.80)
        parity_violated = bool(dir_ratio < 0.80)

        return SliceFairnessDTO(
            protected_attribute=protected_attribute,
            subgroup_scores=subgroup_scores,
            max_disparity=round(max_disparity, 4),
            disparate_impact_ratio=round(dir_ratio, 4),
            parity_violated=parity_violated,
        )
