"""Comprehensive dataset hygiene auditor and pre-flight health scanner with missingness mechanism tests and Benford's law."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, skew

from ml_mcp.engine.sentinel_hunter import SentinelHunter
from ml_mcp.schemas.audit import AuditReportDTO

logger = logging.getLogger(__name__)


class DatasetAuditor:
    """Pre-flight statistical and integrity scan for tabular datasets.

    Theoretical foundations:
        - Missingness Mechanism: Jamshidian, Jalal, & Jansen (2020) & Jaeger et al. (NeurIPS 2023)
        - DataPerf Memorization: Mazumder et al. (NeurIPS 2023 DataPerf Benchmark)
        - Benford's Law Audit: Nigrini (2021) First-digit chi-square distribution test
    """

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

    @staticmethod
    def classify_missingness_mechanism(df: pd.DataFrame, col: str, target_column: Optional[str] = None) -> str:
        """Tests whether missingness is completely at random (MCAR) or depends on observed data/target (MNAR).

        Theoretical basis: Jamshidian & Jalal (2020); Jaeger et al. (NeurIPS 2023).
        """
        missing_mask = df[col].isna().astype(int)
        if missing_mask.sum() == 0 or missing_mask.sum() == len(df):
            return "COMPLETE"

        # Check correlation with other numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for other in num_cols:
            if other == col:
                continue
            series = df[other].dropna()
            if len(series) > 10:
                common_idx = df[other].notna()
                if common_idx.sum() > 10:
                    r = np.abs(np.corrcoef(missing_mask[common_idx], df.loc[common_idx, other])[0, 1])
                    if not np.isnan(r) and r > 0.15:
                        return "MNAR"

        # Check target dependency if provided
        if target_column and target_column in df.columns and target_column != col:
            y = df[target_column]
            if pd.api.types.is_numeric_dtype(y):
                valid = y.notna()
                if valid.sum() > 10:
                    r = np.abs(np.corrcoef(missing_mask[valid], y[valid])[0, 1])
                    if not np.isnan(r) and r > 0.15:
                        return "MNAR"
            else:
                contingency = pd.crosstab(missing_mask, y)
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    try:
                        _, p_val, _, _ = chi2_contingency(contingency)
                        if p_val < 0.05:
                            return "MNAR"
                    except Exception as e:
                        logger.debug(f"Chi2 contingency test failed on missingness check: {e}")

        return "MCAR"

    @staticmethod
    def check_benfords_law(series: pd.Series) -> Tuple[bool, float]:
        """Tests first-digit compliance with Benford's Law P(d) = log10(1 + 1/d) (Nigrini 2021).

        Applies to positive continuous variables spanning >= 3 orders of magnitude (N >= 200).
        Returns:
            Tuple of (is_anomalous, chi_square_stat) where is_anomalous is True if p < 0.01 (chi2 > 20.09, df=8).
        """
        clean = series.dropna()
        pos_vals = clean[clean > 0.0].to_numpy(dtype=np.float64)
        if len(pos_vals) < 200:
            return False, 0.0

        min_val, max_val = float(np.min(pos_vals)), float(np.max(pos_vals))
        if min_val <= 0.0 or (max_val / min_val) < 1000.0:
            # Does not span 3 orders of magnitude, Benford not applicable
            return False, 0.0

        # Extract first non-zero digit
        digits = []
        for v in pos_vals[:5000]:  # Cap at 5000 for sub-10ms performance
            s = f"{v:.10e}"
            for ch in s:
                if ch in "123456789":
                    digits.append(int(ch))
                    break

        if len(digits) < 200:
            return False, 0.0

        observed = np.bincount(digits, minlength=10)[1:10]
        n_obs = len(digits)
        expected = np.array([np.log10(1.0 + 1.0 / d) for d in range(1, 10)]) * n_obs

        # Chi-square test
        chi2 = float(np.sum(((observed - expected) ** 2) / expected))
        # df = 8; critical value for p < 0.01 is 20.09
        is_anomalous = chi2 > 20.09
        return is_anomalous, round(chi2, 2)

    def audit_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Literal["classification", "regression"] = "classification",
    ) -> AuditReportDTO:
        """Executes defensive pre-flight audit of the dataset."""
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

        # 3. Missingness Mechanisms (MCAR vs MNAR)
        missingness_mechanisms: Dict[str, str] = {}
        for col in df.columns:
            if cleaned_df[col].isna().any():
                mech = self.classify_missingness_mechanism(cleaned_df, col, target_column=target_column)
                missingness_mechanisms[col] = mech

        # 4. Benford's Law Checks
        benford_anomalies: List[str] = []
        for col in num_cols:
            is_anom, _ = self.check_benfords_law(cleaned_df[col])
            if is_anom:
                benford_anomalies.append(col)

        # 5. Duplicate Rows
        duplicate_rows = int(df.duplicated().sum())

        # 6. Target Analysis & Accuracy Paradox Guard
        class_imbalance_ratio: Optional[float] = None
        target_skewness: Optional[float] = None
        recommended_metric = "accuracy"

        if target_column and target_column in df.columns:
            y = df[target_column].dropna()
            if task_type == "classification" and not y.empty:
                val_counts = y.value_counts(normalize=True)
                majority_ratio = float(val_counts.iloc[0])
                class_imbalance_ratio = majority_ratio

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
                    except Exception as e:
                        logger.debug(f"Target skew calculation failed: {e}")

        # 7. Group Column Candidate Detection
        group_candidate: Optional[str] = None
        for col in df.columns:
            if col == target_column:
                continue
            if self.GROUP_ID_REGEX.match(col):
                unique_ratio = df[col].nunique() / row_count
                if 0.001 < unique_ratio < 0.90:
                    group_candidate = col
                    break

        # 8. Temporal Ordering Detection
        has_temporal_order = False
        date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        if not date_cols:
            for col in df.select_dtypes(include=["object"]).columns:
                if "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
                    try:
                        parsed = pd.to_datetime(df[col].dropna().head(20), errors="coerce")
                        if parsed.notna().all():
                            date_cols.append(col)
                    except Exception as e:
                        logger.debug(f"Date inference failed for column '{col}': {e}")

        for dcol in date_cols:
            parsed_series = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if len(parsed_series) > 10 and parsed_series.is_monotonic_increasing:
                has_temporal_order = True
                break

        # 9. DataPerf Index & High-Cardinality Memorization Guard (Mazumder et al. NeurIPS 2023)
        id_memorization_columns: List[str] = []
        log2_n = np.log2(row_count) if row_count > 1 else 1.0

        for col in df.columns:
            if col == target_column:
                continue
            series = df[col].dropna()
            n_valid = len(series)
            if n_valid < 20:
                continue

            # Continuous floats are real-valued measurements, not index keys (Mazumder et al. 2023)
            if pd.api.types.is_float_dtype(series):
                continue

            unique_count = series.nunique()
            unique_ratio = unique_count / n_valid

            if unique_ratio >= 0.99:
                entropy = self.calculate_shannon_entropy(series)
                if entropy >= 0.90 * log2_n or unique_ratio == 1.0:
                    id_memorization_columns.append(col)

        # 10. Recommended CV Splitting Strategy
        if group_candidate:
            recommended_split_strategy = "group_kfold"
        elif has_temporal_order:
            recommended_split_strategy = "time_series_split"
        elif task_type == "classification":
            recommended_split_strategy = "stratified_kfold"
        else:
            recommended_split_strategy = "kfold"

        # 11. Column Details
        column_details: Dict[str, Any] = {}
        for col in df.columns:
            column_details[col] = {
                "dtype": str(df[col].dtype),
                "missing": int(cleaned_df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
                "is_id_candidate": col in id_memorization_columns,
                "missingness_mechanism": missingness_mechanisms.get(col, "COMPLETE"),
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
            missingness_mechanisms=missingness_mechanisms,
            benford_anomalies=benford_anomalies,
            target_skewness=target_skewness,
            recommended_split_strategy=recommended_split_strategy,
            column_details=column_details,
        )
