"""Scikit-Learn compliant Feature Synthesizer for cyclical time projections and safe ratios."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Projects periodic temporal features into continuous circular space via sin and cos."""

    def __init__(
        self,
        time_periods: Optional[Dict[str, float]] = None,
        period: Optional[float] = None,
    ) -> None:
        # e.g. {"hour": 24.0, "dayofweek": 7.0, "month": 12.0}
        self.time_periods = time_periods or {}
        self.period = period

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CyclicalFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.time_periods:
            for col, period in self.time_periods.items():
                if col in X_out.columns:
                    series = pd.to_numeric(X_out[col], errors="coerce").fillna(0.0)
                    radians = 2.0 * np.pi * series / float(period)
                    X_out[f"{col}_sin"] = np.sin(radians)
                    X_out[f"{col}_cos"] = np.cos(radians)
        elif self.period is not None:
            for col in X_out.columns:
                series = pd.to_numeric(X_out[col], errors="coerce").fillna(0.0)
                radians = 2.0 * np.pi * series / float(self.period)
                X_out[f"{col}_sin"] = np.sin(radians)
                X_out[f"{col}_cos"] = np.cos(radians)
        return X_out


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Computes interaction ratios with epsilon protection preventing division by zero."""

    def __init__(
        self,
        ratio_pairs: Optional[List[Tuple[str, str, str]]] = None,
        epsilon: float = 1e-6,
    ) -> None:
        # List of (numerator_col, denominator_col, output_col_name)
        self.ratio_pairs = ratio_pairs or []
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RatioFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for num_col, den_col, out_col in self.ratio_pairs:
            if num_col in X_out.columns and den_col in X_out.columns:
                num = pd.to_numeric(X_out[num_col], errors="coerce").fillna(0.0)
                den = pd.to_numeric(X_out[den_col], errors="coerce").fillna(0.0)

                # Apply epsilon to denominator avoiding zero division
                safe_den = np.where(np.abs(den) < self.epsilon, np.sign(den) * self.epsilon + (den == 0) * self.epsilon, den)
                ratio_val = num / safe_den
                # Clip infinite results if any
                ratio_val = np.nan_to_num(ratio_val, nan=0.0, posinf=1e8, neginf=-1e8)
                X_out[out_col] = ratio_val
        return X_out
