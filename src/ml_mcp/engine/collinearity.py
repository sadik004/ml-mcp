"""Collinearity and VIF filter implementing the Competitive Drop Rule."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from ml_mcp.schemas.audit import CollinearPairDTO


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
        task_type: str = "auto",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Identifies collinear clusters and drops inferior twins based on target signal.

        Competitive Drop Rule:
        - Continuous Target: Pearson correlation |r(A, y)| vs |r(B, y)|
        - Categorical Target: ANOVA F-value (F_A vs F_B) via sklearn.feature_selection.f_classif
        - No Target Provided: Variance * Non-Null Ratio (N_valid / N_total)

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

        # Determine target modality (continuous, categorical, or none)
        is_continuous_target = False
        is_categorical_target = False

        if y is not None:
            valid_y = y.dropna()
            if len(valid_y) > 5:
                unique_y = valid_y.nunique()
                if task_type == "regression" or (task_type == "auto" and pd.api.types.is_numeric_dtype(y) and unique_y > 15):
                    is_continuous_target = True
                else:
                    is_categorical_target = True

        # Precompute predictive relevance scores
        feature_signal_scores: Dict[str, float] = {}
        selection_metric = "variance_non_null"

        if is_continuous_target and y is not None:
            selection_metric = "pearson"
            for col in num_cols:
                valid = feature_df[col].notna() & y.notna()
                if valid.sum() > 5:
                    r = float(np.abs(np.corrcoef(feature_df.loc[valid, col], y[valid])[0, 1]))
                    feature_signal_scores[col] = 0.0 if np.isnan(r) else r
                else:
                    feature_signal_scores[col] = 0.0

        elif is_categorical_target and y is not None:
            selection_metric = "anova_f"
            # Encode categorical target if needed
            y_codes = pd.Categorical(y).codes
            valid_mask = (y_codes >= 0)
            
            for col in num_cols:
                col_valid = valid_mask & feature_df[col].notna()
                if col_valid.sum() > 5 and len(np.unique(y_codes[col_valid])) > 1:
                    try:
                        X_sub = feature_df.loc[col_valid, [col]].to_numpy()
                        y_sub = y_codes[col_valid]
                        f_scores, _ = f_classif(X_sub, y_sub)
                        f_val = float(f_scores[0]) if len(f_scores) > 0 and not np.isnan(f_scores[0]) else 0.0
                        feature_signal_scores[col] = max(0.0, f_val)
                    except Exception:
                        feature_signal_scores[col] = 0.0
                else:
                    feature_signal_scores[col] = 0.0

        else:
            # No target provided: Retain feature with higher variance * non-null ratio
            selection_metric = "variance_non_null"
            for col in num_cols:
                series = feature_df[col].dropna()
                if len(series) > 1:
                    var = float(series.var())
                    non_null_ratio = float(len(series) / len(feature_df))
                    feature_signal_scores[col] = var * non_null_ratio
                else:
                    feature_signal_scores[col] = 0.0

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
                    score_a = feature_signal_scores.get(col_a, 0.0)
                    score_b = feature_signal_scores.get(col_b, 0.0)

                    if score_a >= score_b:
                        victim = col_b
                        winner = col_a
                    else:
                        victim = col_a
                        winner = col_b

                    dropped_set.add(victim)
                    pair_dto = CollinearPairDTO(
                        feature_a=col_a,
                        feature_b=col_b,
                        correlation=round(r_val, 4),
                        kept=winner,
                        dropped=victim,
                        selection_metric=selection_metric,
                    )
                    collinear_pairs.append(pair_dto.model_dump())

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
            "selection_metric": selection_metric,
        }

        return pruned_df, report
