"""Comprehensive dataset hygiene auditor and pre-flight health scanner."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew

from ml_mcp.engine.sentinel_hunter import SentinelHunter
from ml_mcp.schemas.audit import AuditReportDTO


class DatasetAuditor:
    """Pre-flight statistical and integrity scan for tabular datasets."""

    GROUP_ID_REGEX = re.compile(
        r".*(id|group|patient|user|store|site|unit|subject|hospital|account|device|session).*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.sentinel_hunter = SentinelHunter()

    @staticmethod
    def calculate_shannon_entropy(series: pd.Series) -> float:
        """Calculates Shannon entropy H(X) = -sum(p * log2(p)) in bits."""
        clean_s = series.dropna()
        if clean_s.empty:
            return 0.0
        counts = clean_s.value_counts(normalize=True).to_numpy(dtype=np.float64)
        counts = counts[counts > 0.0]
        entropy = -float(np.sum(counts * np.log2(counts)))
        return max(0.0, entropy)

    def audit_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Literal["classification", "regression"] = "classification",
    ) -> AuditReportDTO:
        """Executes a defensive audit of the input dataset before pipeline construction.

        Returns:
            AuditReportDTO containing all hygiene indicators, group candidates, and metrics.
        """
        row_count = len(df)
        column_count = len(df.columns)

        if row_count == 0 or column_count == 0:
            raise ValueError("Input dataset cannot be empty.")

        # 1. Sentinel Hunting
        cleaned_df, sentinel_report = self.sentinel_hunter.mask_sentinels(df)
        sentinel_count = int(sentinel_report["total_sentinels_masked"])

        # 2. Missing Cells and Infinities
        missing_cells = int(cleaned_df.isna().sum().sum())
        missing_ratio = float(missing_cells / (row_count * column_count))

        infinite_count = 0
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            inf_mask = np.isinf(cleaned_df[col])
            infinite_count += int(inf_mask.sum())

        # 3. Duplicate Rows
        duplicate_rows = int(df.duplicated().sum())

        # 4. Target Analysis & Accuracy Paradox Guard
        class_imbalance_ratio: Optional[float] = None
        target_skewness: Optional[float] = None
        recommended_metric = "accuracy"

        if target_column and target_column in df.columns:
            y = df[target_column].dropna()
            if task_type == "classification" and not y.empty:
                val_counts = y.value_counts(normalize=True)
                majority_ratio = float(val_counts.iloc[0])
                class_imbalance_ratio = majority_ratio

                # Accuracy Paradox Guard:
                # If majority class > 85%, accuracy is strictly banned!
                if majority_ratio > 0.85:
                    num_classes = len(val_counts)
                    recommended_metric = "pr_auc" if num_classes == 2 else "f1_weighted"
                elif majority_ratio > 0.65:
                    recommended_metric = "f1_weighted"
                else:
                    recommended_metric = "accuracy"
            elif task_type == "regression" and not y.empty:
                recommended_metric = "rmse"
                if pd.api.types.is_numeric_dtype(y) and len(y) > 3:
                    try:
                        skew_val = float(skew(y.to_numpy(dtype=np.float64), nan_policy="omit"))
                        if not np.isnan(skew_val):
                            target_skewness = round(skew_val, 4)
                    except Exception:
                        pass

        # 5. Group Column Candidate Detection (Group Leakage Guard)
        group_candidate: Optional[str] = None
        for col in df.columns:
            if col == target_column:
                continue
            if self.GROUP_ID_REGEX.match(col):
                unique_ratio = df[col].nunique() / row_count
                if 0.001 < unique_ratio < 0.90:
                    group_candidate = col
                    break

        # 6. Temporal Ordering Detection (TimeSeriesSplit Guard)
        has_temporal_order = False
        date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        if not date_cols:
            for col in df.select_dtypes(include=["object"]).columns:
                if "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
                    try:
                        parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                        if parsed.notna().all():
                            date_cols.append(col)
                    except Exception:
                        pass

        for dcol in date_cols:
            parsed_series = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if len(parsed_series) > 10 and parsed_series.is_monotonic_increasing:
                has_temporal_order = True
                break

        # 7. Entropy & High-Cardinality ID Memorization Guard (Mazumder et al. NeurIPS 2023)
        # Prevents tree-based models from memorizing arbitrary indices, GUIDs, or row numbers
        id_memorization_columns: List[str] = []
        log2_n = np.log2(row_count) if row_count > 1 else 1.0

        for col in df.columns:
            if col == target_column:
                continue
            series = df[col].dropna()
            n_valid = len(series)
            if n_valid < 20:
                continue

            unique_count = series.nunique()
            unique_ratio = unique_count / n_valid

            # Exact unique identifier (100% distinct)
            if unique_ratio >= 0.99:
                entropy = self.calculate_shannon_entropy(series)
                # Max possible entropy is log2(N); if entropy is near maximal (> 90%), it is an ID column
                if entropy >= 0.90 * log2_n or unique_ratio == 1.0:
                    id_memorization_columns.append(col)

        # 8. Recommended CV Splitting Strategy
        if group_candidate:
            recommended_split_strategy = "group_kfold"
        elif has_temporal_order:
            recommended_split_strategy = "time_series_split"
        elif task_type == "classification":
            recommended_split_strategy = "stratified_kfold"
        else:
            recommended_split_strategy = "kfold"

        # 9. Detailed Column Summaries
        column_details: Dict[str, Any] = {}
        for col in df.columns:
            column_details[col] = {
                "dtype": str(df[col].dtype),
                "missing": int(cleaned_df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
                "is_id_candidate": col in id_memorization_columns,
            }

        return AuditReportDTO(
            row_count=row_count,
            column_count=column_count,
            missing_cells=missing_cells,
            missing_ratio=round(missing_ratio, 6),
            infinite_count=infinite_count,
            sentinel_count=sentinel_count,
            duplicate_rows=duplicate_rows,
            class_imbalance_ratio=round(class_imbalance_ratio, 4) if class_imbalance_ratio is not None else None,
            recommended_metric=recommended_metric,
            group_column_candidate=group_candidate,
            has_temporal_order=has_temporal_order,
            id_memorization_columns=id_memorization_columns,
            target_skewness=target_skewness,
            recommended_split_strategy=recommended_split_strategy,
            column_details=column_details,
        )
