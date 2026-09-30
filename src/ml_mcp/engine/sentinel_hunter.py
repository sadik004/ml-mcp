"""Sentinel Value Hunter with domain-aware false-positive guards."""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd


class SentinelHunter:
    """Detects and safely masks legacy sentinel missing values into true np.nan."""

    DEFINITE_NUMERIC_SENTINELS: Set[float] = {-999.0, 999.0, 9999.0, 99999.0, -9999.0}
    DEFINITE_STRING_SENTINELS: Set[str] = {
        "?", "null", "none", "na", "n/a", "--", "unknown", "nan", ""
    }

    def __init__(self, spike_threshold: float = 0.05) -> None:
        self.spike_threshold = spike_threshold

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
                # Clean strings
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

                # Definite numeric sentinels (-999, 999, etc.)
                definite_mask = series.isin(self.DEFINITE_NUMERIC_SENTINELS)
                if definite_mask.any():
                    count = int(definite_mask.sum())
                    cleaned_df.loc[definite_mask, col] = np.nan
                    col_masked_count += count

                # False-Positive Guard for -1:
                # Only mask -1 if:
                # a) All OTHER numbers are >= 0 (e.g. age, salary, tenure)
                # OR b) There is an abnormal frequency spike (>5%) while other values are non-negative
                minus_one_mask = (series == -1) | (series == -1.0)
                if minus_one_mask.any():
                    other_values = valid_num[valid_num != -1]
                    if other_values.empty or (other_values >= 0).all():
                        # The column domain is non-negative, so -1 is a sentinel code!
                        count = int(minus_one_mask.sum())
                        cleaned_df.loc[minus_one_mask, col] = np.nan
                        col_masked_count += count
                    else:
                        # Genuine negative scale exists (e.g. temperature, profit/loss), preserve -1!
                        pass

            if col_masked_count > 0:
                masked_per_column[col] = col_masked_count
                total_masked += col_masked_count

        summary = {
            "total_sentinels_masked": total_masked,
            "masked_columns": masked_per_column,
        }
        return cleaned_df, summary
