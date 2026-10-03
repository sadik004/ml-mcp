"""High-performance collinearity filter using Spectral SVD conditioning and iterative VIF pruning."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from ml_mcp.schemas.audit import CollinearPairDTO, CollinearityReportDTO


class CollinearityFilter:
    """Eliminates severe multicollinearity using SVD Spectral Condition Number and Iterative VIF."""

    def __init__(
        self,
        threshold_corr: float = 0.90,
        vif_threshold: float = 10.0,
        condition_number_threshold: float = 30.0,
    ) -> None:
        self.threshold_corr = threshold_corr
        self.vif_threshold = vif_threshold
        self.condition_number_threshold = condition_number_threshold

    @staticmethod
    def calculate_spectral_condition_number(
        df: pd.DataFrame, num_cols: Optional[List[str]] = None
    ) -> float:
        """Calculates Spectral Condition Number kappa(X) = sigma_max / sigma_min via SVD (Lafon et al. 2023).

        A condition number kappa > 30 indicates moderate-to-severe collinearity;
        kappa > 100 indicates catastrophic matrix ill-conditioning.
        """
        cols = num_cols if num_cols is not None else df.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) < 2:
            return 1.0

        sub_df = df[cols].dropna()
        if len(sub_df) < 3:
            return 1.0

        X = sub_df.to_numpy(dtype=np.float64)
        # Standardize features (zero mean, unit variance) to avoid scale-induced ill-conditioning
        std = np.std(X, axis=0)
        std[std == 0.0] = 1.0
        X_scaled = (X - np.mean(X, axis=0)) / std

        try:
            s = np.linalg.svd(X_scaled, compute_uv=False)
            sigma_max = float(s[0])
            sigma_min = float(s[-1])
            if sigma_min <= 1e-12:
                return float("inf")
            cond = sigma_max / sigma_min
            return round(cond, 2)
        except Exception:
            return float("inf")

    def calculate_vif(
        self, df: pd.DataFrame, num_cols: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculates Variance Inflation Factor (VIF) using pure NumPy correlation matrix inversion.

        Mathematical foundation:
            VIF_i = diag(R^-1)_i
        """
        cols = num_cols if num_cols is not None else df.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) < 2:
            return {col: 1.0 for col in cols}

        # Filter out NaN rows for correlation matrix
        sub_df = df[cols].dropna()
        if len(sub_df) < 3:
            return {col: 1.0 for col in cols}

        corr_matrix = sub_df.corr().to_numpy(dtype=np.float64)

        # Handle zero-variance features causing NaN in correlation
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        # Invert correlation matrix using pseudo-inverse if ill-conditioned
        try:
            cond = np.linalg.cond(corr_matrix)
            if np.isinf(cond) or cond > 1e12 or np.isnan(cond):
                inv_corr = np.linalg.pinv(corr_matrix)
            else:
                inv_corr = np.linalg.inv(corr_matrix)
        except np.linalg.LinAlgError:
            inv_corr = np.linalg.pinv(corr_matrix)

        vif_diagonal = np.diag(inv_corr)
        vif_diagonal = np.where(vif_diagonal < 1.0, 1.0, vif_diagonal)

        vif_dict: Dict[str, float] = {}
        for idx, col in enumerate(cols):
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
        """Prunes collinear clusters and multi-column linear dependencies using SVD and Iterative VIF.

        Competitive Drop Rule:
        - Continuous Target: Pearson correlation |r(A, y)| vs |r(B, y)|
        - Categorical Target: ANOVA F-value (F_A vs F_B) via sklearn.feature_selection.f_classif
        - No Target Provided: Variance * Non-Null Ratio (N_valid / N_total)

        Returns:
            Tuple of (pruned_dataframe, collinearity_report_dict)
        """
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
                "spectral_condition_number": 1.0,
                "vif_scores": {col: 1.0 for col in num_cols},
                "high_vif_features": [],
                "collinear_pairs": [],
                "dropped_features": [],
                "remaining_features_count": len(feature_df.columns),
                "selection_metric": "none",
            }

        # Determine target modality
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

        # Precompute predictive relevance scores for the Competitive Drop Rule
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

        # Phase 1: Pairwise Correlation Pruning (drops exact/high twins)
        corr_matrix = feature_df[num_cols].corr().abs()
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
                    r_val = 1.0

                if r_val >= self.threshold_corr:
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
                        break

        # Phase 2: Iterative VIF & SVD Spectral Pruning (Belsley et al. / Lafon et al. 2023)
        # Catches 3+ column linear combinations (e.g. X3 = X1 + X2) that pairwise scans miss
        active_num_cols = [c for c in num_cols if c not in dropped_set]

        while len(active_num_cols) > 2:
            current_vifs = self.calculate_vif(feature_df, active_num_cols)
            current_cond = self.calculate_spectral_condition_number(feature_df, active_num_cols)

            high_vif_candidates = [
                col for col, score in current_vifs.items() if score > self.vif_threshold
            ]

            if not high_vif_candidates and current_cond <= self.condition_number_threshold:
                # All collinearity resolved
                break

            if high_vif_candidates:
                # Among candidates with VIF > threshold, drop the one with lowest predictive signal
                victim = min(
                    high_vif_candidates,
                    key=lambda c: (feature_signal_scores.get(c, 0.0), -current_vifs.get(c, 0.0)),
                )
            elif current_cond > self.condition_number_threshold:
                # Severe SVD condition number without individual VIF > 10; drop candidate with highest VIF
                victim = max(active_num_cols, key=lambda c: current_vifs.get(c, 0.0))
            else:
                break

            dropped_set.add(victim)
            active_num_cols.remove(victim)

        # Final audit metrics on remaining features
        final_vif_scores = self.calculate_vif(feature_df, active_num_cols)
        final_cond = self.calculate_spectral_condition_number(feature_df, active_num_cols)
        high_vif_features = [
            col for col, score in final_vif_scores.items() if score >= self.vif_threshold
        ]

        remaining_cols = [c for c in feature_df.columns if c not in dropped_set]
        pruned_df = feature_df[remaining_cols].copy()

        if y is not None and target_column:
            pruned_df[target_column] = y

        report = {
            "threshold_corr": self.threshold_corr,
            "vif_threshold": self.vif_threshold,
            "spectral_condition_number": final_cond,
            "vif_scores": final_vif_scores,
            "high_vif_features": high_vif_features,
            "collinear_pairs": collinear_pairs,
            "dropped_features": list(dropped_set),
            "remaining_features_count": len(remaining_cols),
            "selection_metric": selection_metric,
        }

        return pruned_df, report
