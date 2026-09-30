"""Target Leakage Detector using task-aware mutual information and correlation analysis."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ml_mcp.schemas.audit import TargetLeakageReportDTO


class TargetLeakageDetector:
    """Detects suspicious post-event features exhibiting anomalous correlation or mutual info."""

    def __init__(
        self, threshold_correlation: float = 0.95, threshold_mi: float = 0.85
    ) -> None:
        self.threshold_correlation = threshold_correlation
        self.threshold_mi = threshold_mi

    def detect_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
    ) -> TargetLeakageReportDTO:
        """Analyzes all predictor columns against the target to detect data leakage.

        Args:
            df: Input dataset.
            target_column: Name of target feature.
            task_type: classification or regression.

        Returns:
            TargetLeakageReportDTO containing flagged features and scores.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        y = df[target_column]
        feature_df = df.drop(columns=[target_column])

        # Numerical features for correlation analysis
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        correlations: Dict[str, float] = {}
        leaked_features: List[str] = []

        # 1. Pearson Correlation Analysis
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

        # 2. Task-Aware Mutual Information Analysis
        mi_scores: Dict[str, float] = {}
        if numeric_cols and len(df) > 10:
            # Impute NaNs with median for MI computation
            X_num = feature_df[numeric_cols].copy()
            for col in numeric_cols:
                if X_num[col].isna().any():
                    median_val = X_num[col].median()
                    X_num[col] = X_num[col].fillna(median_val if not np.isnan(median_val) else 0.0)

            # Target handling for MI
            y_clean = y.copy()
            if pd.api.types.is_numeric_dtype(y_clean):
                if y_clean.isna().any():
                    y_clean = y_clean.fillna(y_clean.median())
            else:
                y_clean = y_clean.astype("category").cat.codes

            try:
                if task_type == "classification":
                    raw_mi = mutual_info_classif(
                        X_num, y_clean, random_state=42, n_neighbors=3
                    )
                else:
                    raw_mi = mutual_info_regression(
                        X_num, y_clean, random_state=42, n_neighbors=3
                    )

                for col, score in zip(numeric_cols, raw_mi):
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
            mutual_info_scores=mi_scores,
            has_critical_leakage=len(leaked_features) > 0,
        )
