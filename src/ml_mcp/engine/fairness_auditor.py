"""Intersectional Subgroup Fairness and Disparate Impact Auditor (Kearns et al. ICML 2018)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from ml_mcp.schemas.safety import SliceFairnessDTO

logger = logging.getLogger(__name__)


class SliceFairnessAuditor:
    """Audits model performance across demographic subgroups and intersectional combinations."""

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
            return float(accuracy_score(y_true, y_pred))

    def audit(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        protected_series: Union[pd.Series, pd.DataFrame, np.ndarray, list, Dict[str, Any]],
        protected_attribute: str = "protected_attribute",
        metric: str = "accuracy",
    ) -> SliceFairnessDTO:
        """Execute intersectional demographic subgroup fairness audit enforcing US EEOC 80% rule."""
        X_arr = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
        y_arr = np.asarray(y_test)

        # Handle multiple protected attributes for intersectional subgroup analysis
        if isinstance(protected_series, pd.DataFrame):
            # Cartesian product combination: "col1_val_col2_val"
            composite_groups = protected_series.astype(str).agg("_x_".join, axis=1).to_numpy()
            attr_name = "+".join(protected_series.columns)
        elif isinstance(protected_series, dict):
            df_temp = pd.DataFrame(protected_series)
            composite_groups = df_temp.astype(str).agg("_x_".join, axis=1).to_numpy()
            attr_name = "+".join(df_temp.columns)
        elif isinstance(protected_series, np.ndarray) and protected_series.ndim > 1:
            df_temp = pd.DataFrame(protected_series)
            composite_groups = df_temp.astype(str).agg("_x_".join, axis=1).to_numpy()
            attr_name = protected_attribute
        else:
            composite_groups = np.asarray(protected_series, dtype=str)
            attr_name = protected_attribute

        if len(X_arr) != len(composite_groups):
            raise ValueError(f"Length mismatch: X_test ({len(X_arr)}) and protected_series ({len(composite_groups)})")

        preds = model.predict(X_test)
        unique_groups = np.unique(composite_groups)

        subgroup_scores: Dict[str, float] = {}
        for grp in unique_groups:
            mask = composite_groups == grp
            if np.sum(mask) >= self.min_slice_samples:
                score = self._calculate_slice_metric(y_arr[mask], preds[mask], metric)
                subgroup_scores[str(grp)] = round(float(score), 4)

        if not subgroup_scores:
            overall_score = self._calculate_slice_metric(y_arr, preds, metric)
            subgroup_scores["all"] = round(float(overall_score), 4)

        scores_list = list(subgroup_scores.values())
        min_score = min(scores_list)
        max_score = max(scores_list)
        max_disparity = float(max_score - min_score)

        # Disparate impact ratio: min_score / max_score
        if max_score > 1e-6:
            dir_ratio = float(min_score / max_score)
        else:
            dir_ratio = 1.0 if min_score == max_score else 0.0

        parity_violated = bool(dir_ratio < 0.80)

        return SliceFairnessDTO(
            protected_attribute=attr_name,
            subgroup_scores=subgroup_scores,
            max_disparity=round(max_disparity, 4),
            disparate_impact_ratio=round(dir_ratio, 4),
            parity_violated=parity_violated,
        )
