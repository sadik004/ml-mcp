"""Scikit-Learn compliant Feature Synthesizer for cyclical time projections, safe ratios, and ExploreKit group aggregations."""
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

                safe_den = np.where(np.abs(den) < self.epsilon, np.sign(den) * self.epsilon + (den == 0) * self.epsilon, den)
                ratio_val = num / safe_den
                ratio_val = np.nan_to_num(ratio_val, nan=0.0, posinf=1e8, neginf=-1e8)
                X_out[out_col] = ratio_val
        return X_out


class GroupByAggregationTransformer(BaseEstimator, TransformerMixin):
    """Computes group-level aggregations and relative features based on ExploreKit architecture (Katz et al., IEEE ICDM 2016)."""

    def __init__(
        self,
        group_specs: Optional[List[Dict[str, Any]]] = None,
        epsilon: float = 1e-6,
    ) -> None:
        self.group_specs = group_specs or []
        self.epsilon = epsilon
        self.group_stats_: Dict[str, pd.DataFrame] = {}
        self.global_stats_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GroupByAggregationTransformer":
        self.group_stats_ = {}
        self.global_stats_ = {}

        for i, spec in enumerate(self.group_specs):
            cat_col = spec.get("cat_col")
            num_col = spec.get("num_col")
            aggs = spec.get("aggregations", ["mean", "std"])

            if not cat_col or not num_col or cat_col not in X.columns or num_col not in X.columns:
                continue

            spec_key = f"{cat_col}__{num_col}__{i}"
            num_series = pd.to_numeric(X[num_col], errors="coerce")
            temp_df = pd.DataFrame({cat_col: X[cat_col], num_col: num_series})

            # Calculate group stats strictly on training fold
            grp = temp_df.groupby(cat_col, observed=False)[num_col].agg(aggs)
            self.group_stats_[spec_key] = grp

            # Calculate global fallback statistics for unseen categories during transform
            g_mean = float(num_series.mean()) if not np.isnan(num_series.mean()) else 0.0
            g_std = float(num_series.std(ddof=0)) if not np.isnan(num_series.std(ddof=0)) and num_series.std(ddof=0) > 0 else 1.0
            
            fallback_dict: Dict[str, float] = {
                "mean": g_mean,
                "std": g_std,
            }
            for agg in aggs:
                if agg not in fallback_dict:
                    if agg == "median":
                        fallback_dict["median"] = float(num_series.median()) if not np.isnan(num_series.median()) else 0.0
                    elif agg == "min":
                        fallback_dict["min"] = float(num_series.min()) if not np.isnan(num_series.min()) else 0.0
                    elif agg == "max":
                        fallback_dict["max"] = float(num_series.max()) if not np.isnan(num_series.max()) else 0.0
                    else:
                        fallback_dict[agg] = g_mean
            self.global_stats_[spec_key] = fallback_dict

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for i, spec in enumerate(self.group_specs):
            cat_col = spec.get("cat_col")
            num_col = spec.get("num_col")
            aggs = spec.get("aggregations", ["mean", "std"])
            create_diff = spec.get("create_relative_diff", True)
            create_ratio = spec.get("create_relative_ratio", True)
            create_zscore = spec.get("create_zscore", True)

            spec_key = f"{cat_col}__{num_col}__{i}"
            if spec_key not in self.group_stats_ or cat_col not in X_out.columns or num_col not in X_out.columns:
                continue

            grp = self.group_stats_[spec_key]
            g_stats = self.global_stats_[spec_key]
            num_series = pd.to_numeric(X_out[num_col], errors="coerce").fillna(g_stats["mean"])

            agg_series_dict: Dict[str, pd.Series] = {}
            for agg in aggs:
                if agg in grp.columns:
                    mapped = X_out[cat_col].map(grp[agg])
                    fallback_val = g_stats.get(agg, g_stats["mean"])
                    mapped = mapped.fillna(fallback_val)
                else:
                    mapped = pd.Series(g_stats.get(agg, g_stats["mean"]), index=X_out.index)
                
                col_name = f"{num_col}_{agg}_by_{cat_col}"
                X_out[col_name] = mapped
                agg_series_dict[agg] = mapped

            group_mean = agg_series_dict.get("mean", pd.Series(g_stats["mean"], index=X_out.index))
            group_std = agg_series_dict.get("std", pd.Series(g_stats["std"], index=X_out.index))

            if create_diff:
                diff_col = f"{num_col}_diff_from_{cat_col}_mean"
                X_out[diff_col] = num_series - group_mean

            if create_ratio:
                ratio_col = f"{num_col}_ratio_to_{cat_col}_mean"
                safe_mean = np.where(
                    np.abs(group_mean) < self.epsilon,
                    np.sign(group_mean) * self.epsilon + (group_mean == 0) * self.epsilon,
                    group_mean
                )
                ratio_val = num_series / safe_mean
                ratio_val = np.nan_to_num(ratio_val, nan=1.0, posinf=1e6, neginf=-1e6)
                X_out[ratio_col] = ratio_val

            if create_zscore:
                z_col = f"{num_col}_zscore_in_{cat_col}"
                safe_std = np.where(np.abs(group_std) < self.epsilon, self.epsilon, group_std)
                z_val = (num_series - group_mean) / safe_std
                z_val = np.nan_to_num(z_val, nan=0.0, posinf=10.0, neginf=-10.0)
                X_out[z_col] = z_val

        return X_out
