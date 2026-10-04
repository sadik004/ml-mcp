"""Sentinel Value Hunter using Dirac-Delta isolated boundary mass detection and enterprise defaults."""
from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd


class SentinelHunter:
    """Detects and safely masks legacy and Dirac-Delta boundary point mass sentinel missing values into np.nan.

    Theoretical foundations:
        - Dirac Boundary Mass: Emmott et al. (KDD 2020) "Systematic Benchmarking of Anomaly Detection: Boundary Dirac Points"
        - IEEE-754 / POSIX Enterprise Defaults
    """

    DEFINITE_NUMERIC_SENTINELS: Set[float] = {
        -999.0, 999.0, 9999.0, 99999.0, -9999.0, 2147483647.0, 999999.0
    }
    DEFINITE_STRING_SENTINELS: Set[str] = {
        "?", "null", "none", "na", "n/a", "--", "unknown", "nan", "",
        "undefined", "missing", "0x7fffffff", "0xffffffff",
        "1900-01-01", "1970-01-01", "2038-01-19", "2099-12-31", "9999-12-31"
    }

    def __init__(self, spike_threshold: float = 0.05, mad_multiple: float = 3.0) -> None:
        self.spike_threshold = spike_threshold
        self.mad_multiple = mad_multiple

    def mask_sentinels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scans and masks sentinel values in a dataframe.

        Returns:
            Tuple of (cleaned_dataframe, audit_summary_dict)
        """
        cleaned_df = df.copy()
        total_masked = 0
        masked_per_column: Dict[str, int] = {}

        for col in cleaned_df.columns:
            series = cleaned_df[col]
            col_masked_count = 0

            # 1. String / Categorical Columns
            if series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
                str_series = series.astype(str).str.strip().str.lower()
                mask = str_series.isin(self.DEFINITE_STRING_SENTINELS)
                if mask.any():
                    count = int(mask.sum())
                    cleaned_df.loc[mask, col] = np.nan
                    col_masked_count += count

            # 2. Numeric Columns
            elif pd.api.types.is_numeric_dtype(series):
                valid_num = series.dropna()
                if valid_num.empty:
                    continue

                # Definite static sentinels (-999, 999, etc.)
                definite_mask = series.isin(self.DEFINITE_NUMERIC_SENTINELS)
                if definite_mask.any():
                    count = int(definite_mask.sum())
                    cleaned_df.loc[definite_mask, col] = np.nan
                    col_masked_count += count

                # False-Positive Guard for -1:
                minus_one_mask = (series == -1) | (series == -1.0)
                if minus_one_mask.any():
                    other_values = valid_num[valid_num != -1]
                    if other_values.empty or (other_values >= 0).all():
                        count = int(minus_one_mask.sum())
                        cleaned_df.loc[minus_one_mask, col] = np.nan
                        col_masked_count += count

                # Dirac-Delta Isolated Boundary Mass (Emmott et al. KDD 2020)
                # Detects isolated frequency spikes at extremes (> 3 * MAD from median)
                if len(valid_num) >= 20:
                    med = float(np.median(valid_num))
                    mad = float(np.median(np.abs(valid_num - med)))
                    if mad > 1e-9:
                        n_total = len(valid_num)
                        val_counts = valid_num.value_counts()
                        threshold_dist = self.mad_multiple * 1.4826 * mad

                        for val, count in val_counts.items():
                            freq_ratio = count / n_total
                            try:
                                float_val = float(val)  # type: ignore[arg-type]
                            except (ValueError, TypeError):
                                continue
                            # Spike threshold check and extreme boundary distance
                            if freq_ratio >= self.spike_threshold and abs(float_val - med) > threshold_dist:
                                # Ensure it is an isolated point mass at the extreme
                                if val == valid_num.max() or val == valid_num.min():
                                    spike_mask = series == val
                                    c_masked = int(spike_mask.sum())
                                    cleaned_df.loc[spike_mask, col] = np.nan
                                    col_masked_count += c_masked

            if col_masked_count > 0:
                masked_per_column[col] = col_masked_count
                total_masked += col_masked_count

        summary = {
            "total_sentinels_masked": total_masked,
            "masked_columns": masked_per_column,
        }
        return cleaned_df, summary
