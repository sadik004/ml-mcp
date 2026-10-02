"""Collinearity and VIF filter implementing the Competitive Drop Rule."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class CollinearityFilter:
    """Detects and prunes highly collinear twin features using target-aware competitive drop."""

    def __init__(self, threshold_corr: float = 0.90, vif_threshold: float = 10.0) -> None:
        self.threshold_corr = threshold_corr
        self.vif_threshold = vif_threshold

    def calculate_vif(
        self,
        df: pd.DataFrame,
        num_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Calculates Variance Inflation Factor (VIF) for numeric features using pure NumPy.

        Uses the relationship:
            VIF_i = 1 / (1 - R_i^2) = (R^-1)_ii
        where R is the correlation matrix of the features and (R^-1)_ii is the i-th
        diagonal entry of the inverted correlation matrix.

        Returns:
            Dict mapping feature name to its calculated VIF value.
        """
        if num_cols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            return {col: 1.0 for col in num_cols}

        # Compute pairwise correlation matrix safely
        corr_df = df[num_cols].corr()
        corr_matrix = corr_df.values

        # Clean NaNs in correlation matrix (e.g. constant columns)
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        # Invert correlation matrix using pseudo-inverse if singular
        try:
            cond = np.linalg.cond(corr_matrix)
            if np.isinf(cond) or cond > 1e12 or np.isnan(cond):
                inv_corr = np.linalg.pinv(corr_matrix)
            else:
                inv_corr = np.linalg.inv(corr_matrix)
        except np.linalg.LinAlgError:
            inv_corr = np.linalg.pinv(corr_matrix)

        # The diagonal elements of R^-1 are VIF_i = 1 / (1 - R_i^2)
        vif_diagonal = np.diag(inv_corr)

        # Floor at 1.0 since theoretical minimum VIF is 1.0 (zero correlation)
        vif_diagonal = np.where(vif_diagonal < 1.0, 1.0, vif_diagonal)

        vif_dict: Dict[str, float] = {}
        for idx, col in enumerate(num_cols):
            vif_val = float(vif_diagonal[idx])
            if np.isnan(vif_val):
                vif_val = float("inf")
            vif_dict[col] = round(vif_val, 2)

        return vif_dict

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
            return df, {
                "threshold_corr": self.threshold_corr,
                "vif_threshold": self.vif_threshold,
                "vif_scores": {col: 1.0 for col in num_cols},
                "high_vif_features": [],
                "collinear_pairs": [],
                "dropped_features": [],
                "remaining_features_count": len(feature_df.columns),
            }

        # Calculate pure NumPy VIF scores for all numeric features
        vif_scores = self.calculate_vif(feature_df, num_cols)
        high_vif_features = [
            col for col, score in vif_scores.items() if score >= self.vif_threshold
        ]

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
            "threshold_corr": self.threshold_corr,
            "vif_threshold": self.vif_threshold,
            "vif_scores": vif_scores,
            "high_vif_features": high_vif_features,
            "collinear_pairs": collinear_pairs,
            "dropped_features": list(dropped_set),
            "remaining_features_count": len(remaining_cols),
        }

        return pruned_df, report
