"""Central Pre-Flight Data Auditor orchestrating data hygiene and metric guards."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd

from ml_mcp.engine.sentinel_hunter import SentinelHunter
from ml_mcp.schemas.audit import AuditReportDTO


class DatasetAuditor:
    """Pre-flight statistical auditor enforcing Accuracy Paradox and Group Leakage guards."""

    GROUP_ID_REGEX = re.compile(r".*(_id|id|_key|user|customer|patient|account)$", re.IGNORECASE)

    def __init__(self) -> None:
        self.sentinel_hunter = SentinelHunter()

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
                else:
                    recommended_metric = "accuracy"
            elif task_type == "regression":
                recommended_metric = "rmse"

        # 5. Group Column Candidate Detection (Group Leakage Guard)
        group_candidate: Optional[str] = None
        for col in df.columns:
            if col == target_column:
                continue
            if self.GROUP_ID_REGEX.match(col):
                # Verify that values repeat (entity visits)
                unique_ratio = df[col].nunique() / row_count
                if 0.001 < unique_ratio < 0.90:
                    group_candidate = col
                    break

        # 6. Temporal Ordering Detection (TimeSeriesSplit Guard)
        has_temporal_order = False
        date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        if not date_cols:
            # Check string columns parseable as dates
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

        # 7. Detailed Column Summaries
        column_details: Dict[str, Any] = {}
        for col in df.columns:
            column_details[col] = {
                "dtype": str(df[col].dtype),
                "missing": int(cleaned_df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
            }

        return AuditReportDTO(
            row_count=row_count,
            column_count=column_count,
            missing_cells=missing_cells,
            missing_ratio=round(missing_ratio, 6),
            infinite_count=infinite_count,
            sentinel_count=sentinel_count,
            duplicate_rows=duplicate_rows,
            class_imbalance_ratio=round(class_imbalance_ratio, 4) if class_imbalance_ratio else None,
            recommended_metric=recommended_metric,
            group_column_candidate=group_candidate,
            has_temporal_order=has_temporal_order,
            column_details=column_details,
        )
