"""Target Leakage Detector using task-aware mutual information, correlation, and Bias-Corrected Cramer's V."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

from ml_mcp.schemas.audit import TargetLeakageReportDTO


class TargetLeakageDetector:
    """Detects suspicious post-event features exhibiting anomalous correlation, mutual info, or categorical association."""

    def __init__(
        self,
        threshold_correlation: float = 0.95,
        threshold_mi: float = 0.85,
        threshold_cramers_v: float = 0.90,
    ) -> None:
        self.threshold_correlation = threshold_correlation
        self.threshold_mi = threshold_mi
        self.threshold_cramers_v = threshold_cramers_v

    @staticmethod
    def calculate_bias_corrected_cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Computes Bias-Corrected Cramer's V for categorical association (Greenacre 2021/2023).

        Formula:
            phi_tilde_sq = max(0, phi_sq - ((r - 1) * (k - 1)) / (n - 1))
            r_tilde = r - (r - 1)^2 / (n - 1)
            k_tilde = k - (k - 1)^2 / (n - 1)
            V_tilde = sqrt(phi_tilde_sq / min(r_tilde - 1, k_tilde - 1))
        """
        valid_mask = x.notna() & y.notna()
        x_clean = x[valid_mask].astype(str)
        y_clean = y[valid_mask].astype(str)

        n = len(x_clean)
        if n < 10:
            return 0.0

        contingency = pd.crosstab(x_clean, y_clean)
        r, k = contingency.shape
        if r < 2 or k < 2:
            return 0.0

        try:
            chi2, _, _, _ = chi2_contingency(contingency, correction=False)
        except Exception:
            return 0.0

        phi2 = chi2 / n
        phi2_corr = max(0.0, phi2 - ((r - 1) * (k - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(r_corr - 1, k_corr - 1)

        if denom <= 1e-12:
            return 0.0

        v_tilde = float(np.sqrt(phi2_corr / denom))
        return min(1.0, max(0.0, v_tilde))

    def detect_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
    ) -> TargetLeakageReportDTO:
        """Analyzes all predictor columns (numeric and categorical) against the target to detect data leakage.

        Args:
            df: Input dataset.
            target_column: Name of target feature.
            task_type: classification or regression.

        Returns:
            TargetLeakageReportDTO containing flagged features, Pearson correlations, MI, and Cramer's V scores.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        y = df[target_column]
        feature_df = df.drop(columns=[target_column])

        correlations: Dict[str, float] = {}
        cramers_v_scores: Dict[str, float] = {}
        leaked_features: List[str] = []

        # 1. Pearson Correlation Analysis (for Numeric Predictors)
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

        if pd.api.types.is_numeric_dtype(y):
            for col in numeric_cols:
                series = feature_df[col]
                valid_mask = series.notna() & y.notna()
                if valid_mask.sum() > 5:
                    r = float(np.abs(np.corrcoef(series[valid_mask], y[valid_mask])[0, 1]))
                    if np.isnan(r):
                        r = 0.0
                    correlations[col] = round(r, 4)
                    if r >= self.threshold_correlation:
                        if col not in leaked_features:
                            leaked_features.append(col)

        # 2. Bias-Corrected Cramer's V Analysis (for Categorical Predictors)
        # Evaluated when target is categorical or for discrete classification targets
        is_discrete_target = (
            task_type == "classification"
            or not pd.api.types.is_numeric_dtype(y)
            or y.nunique() <= 20
        )

        for col in categorical_cols:
            if is_discrete_target:
                v_score = self.calculate_bias_corrected_cramers_v(feature_df[col], y)
                cramers_v_scores[col] = round(v_score, 4)
                if v_score >= self.threshold_cramers_v:
                    if col not in leaked_features:
                        leaked_features.append(col)
            else:
                # If target is continuous regression, check ordinal correlation with target
                valid_mask = feature_df[col].notna() & y.notna()
                if valid_mask.sum() > 5:
                    try:
                        codes = pd.Categorical(feature_df.loc[valid_mask, col]).codes
                        r = float(np.abs(np.corrcoef(codes, y[valid_mask])[0, 1]))
                        if not np.isnan(r):
                            correlations[col] = round(r, 4)
                            if r >= self.threshold_correlation:
                                if col not in leaked_features:
                                    leaked_features.append(col)
                    except Exception:
                        pass

        # 3. Task-Aware Scalable Mutual Information Analysis (Chaudhuri et al. 2021)
        # Subsample to 2,000 samples to prevent O(N log N) k-NN compute bottleneck on large datasets
        mi_scores: Dict[str, float] = {}
        all_eval_cols = numeric_cols + categorical_cols

        if all_eval_cols and len(df) > 10:
            eval_df = feature_df[all_eval_cols].copy()
            y_eval = y.copy()

            # Speed guard: subsample if len(df) > 2000
            if len(eval_df) > 2000:
                rng = np.random.RandomState(42)
                sub_indices = rng.choice(len(eval_df), size=2000, replace=False)
                eval_df = eval_df.iloc[sub_indices].copy()
                y_eval = y_eval.iloc[sub_indices].copy()

            # Prepare X: Impute numeric NaNs with median, encode categoricals
            X_prepared = pd.DataFrame(index=eval_df.index)
            for col in numeric_cols:
                median_val = eval_df[col].median()
                X_prepared[col] = eval_df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

            for col in categorical_cols:
                col_str = eval_df[[col]].astype(str)
                ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_prepared[col] = ord_enc.fit_transform(col_str)

            # Target handling for MI
            if pd.api.types.is_numeric_dtype(y_eval):
                if y_eval.isna().any():
                    y_eval = y_eval.fillna(y_eval.median())
            else:
                y_eval = y_eval.astype("category").cat.codes

            try:
                if task_type == "classification" or is_discrete_target:
                    raw_mi = mutual_info_classif(
                        X_prepared, y_eval, random_state=42, n_neighbors=3
                    )
                else:
                    raw_mi = mutual_info_regression(
                        X_prepared, y_eval, random_state=42, n_neighbors=3
                    )

                for col, score in zip(all_eval_cols, raw_mi):
                    mi_scores[col] = round(float(score), 4)
                    if score >= self.threshold_mi:
                        if col not in leaked_features:
                            leaked_features.append(col)
            except Exception:
                pass

        return TargetLeakageReportDTO(
            target_column=target_column,
            leaked_features=leaked_features,
            correlation_matrix=correlations,
            cramers_v_scores=cramers_v_scores,
            mutual_info_scores=mi_scores,
            has_critical_leakage=len(leaked_features) > 0,
        )
