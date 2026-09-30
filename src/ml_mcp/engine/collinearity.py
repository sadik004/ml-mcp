"""Collinearity and VIF filter implementing the Competitive Drop Rule."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class CollinearityFilter:
    """Detects and prunes highly collinear twin features using target-aware competitive drop."""

    def __init__(self, threshold_corr: float = 0.90) -> None:
        self.threshold_corr = threshold_corr

    def filter_collinearity(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: str = "classification",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Identifies collinear clusters and drops inferior twins based on target correlation.

        Returns:
            Tuple of (pruned_dataframe, collinearity_report_dict)
        """
        # Separate target if specified
        y = None
        feature_df = df.copy()
        if target_column and target_column in feature_df.columns:
            y = feature_df[target_column]
            feature_df = feature_df.drop(columns=[target_column])

        num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return df, {"collinear_pairs": [], "dropped_features": []}

        # Compute pairwise correlation matrix safely
        corr_matrix = feature_df[num_cols].corr().abs()

        # Compute correlation with target for competitive drop decision
        target_corrs: Dict[str, float] = {}
        if y is not None and pd.api.types.is_numeric_dtype(y):
            for col in num_cols:
                valid = feature_df[col].notna() & y.notna()
                if valid.sum() > 5:
                    r = float(np.abs(np.corrcoef(feature_df.loc[valid, col], y[valid])[0, 1]))
                    target_corrs[col] = 0.0 if np.isnan(r) else r
                else:
                    target_corrs[col] = 0.0
        else:
            # Default to 0.0 if no numeric target available
            target_corrs = {col: 0.0 for col in num_cols}

        collinear_pairs: List[Dict[str, Any]] = []
        dropped_set: Set[str] = set()

        # Inspect upper triangle of correlation matrix
        for i in range(len(num_cols)):
            col_a = num_cols[i]
            if col_a in dropped_set:
                continue

            for j in range(i + 1, len(num_cols)):
                col_b = num_cols[j]
                if col_b in dropped_set:
                    continue

                r_val = float(corr_matrix.loc[col_a, col_b])
                if np.isnan(r_val):
                    r_val = 1.0  # Handle zero variance or identical columns

                if r_val >= self.threshold_corr:
                    # Competitive Drop Rule:
                    # Drop the feature that has weaker predictive correlation with target
                    corr_a = target_corrs.get(col_a, 0.0)
                    corr_b = target_corrs.get(col_b, 0.0)

                    if corr_a >= corr_b:
                        victim = col_b
                        winner = col_a
                    else:
                        victim = col_a
                        winner = col_b

                    dropped_set.add(victim)
                    collinear_pairs.append({
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(r_val, 4),
                        "kept": winner,
                        "dropped": victim,
                    })

                    if victim == col_a:
                        break  # col_a was dropped, move to next outer column

        # Build output dataframe
        remaining_cols = [c for c in feature_df.columns if c not in dropped_set]
        pruned_df = feature_df[remaining_cols].copy()

        # Restore target if separated
        if y is not None and target_column:
            pruned_df[target_column] = y

        report = {
            "threshold": self.threshold_corr,
            "collinear_pairs": collinear_pairs,
            "dropped_features": list(dropped_set),
            "remaining_features_count": len(remaining_cols),
        }

        return pruned_df, report
