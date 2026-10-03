"""Feature synthesis module with cyclical, ratio, OpenFE cross-numeric, and CatBoost/ExploreKit aggregations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Trigonometric encoding for cyclical temporal features (hours, days, months)."""

    def __init__(self, time_periods: Optional[Dict[str, float]] = None) -> None:
        self.time_periods = time_periods or {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CyclicalFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, period in self.time_periods.items():
            if col in X_out.columns:
                series = pd.to_numeric(X_out[col], errors="coerce").fillna(0.0)
                theta = (2.0 * np.pi * series) / float(period)
                X_out[f"{col}_sin"] = np.sin(theta)
                X_out[f"{col}_cos"] = np.cos(theta)
        return X_out


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Generates guarded pairwise interaction ratios with epsilon denominator protection."""

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
        for num_col, den_col, target_name in self.ratio_pairs:
            if num_col in X_out.columns and den_col in X_out.columns:
                num_series = pd.to_numeric(X_out[num_col], errors="coerce").fillna(0.0)
                den_series = pd.to_numeric(X_out[den_col], errors="coerce").fillna(0.0)

                safe_denom = np.where(
                    np.abs(den_series) < self.epsilon,
                    np.sign(den_series) * self.epsilon + (den_series == 0) * self.epsilon,
                    den_series,
                )
                ratio_val = num_series / safe_denom
                ratio_val = np.nan_to_num(ratio_val, nan=0.0, posinf=1e6, neginf=-1e6)
                X_out[target_name] = ratio_val

        return X_out


class CrossNumericTransformer(BaseEstimator, TransformerMixin):
    """Pairwise cross-numeric generator for top-variance features (Zhang et al. ICML 2023; Grinsztajn et al. NeurIPS 2022).

    Generates:
        1. Safe Ratio: A / (B + eps)
        2. Linear Difference: A - B
    Restricts candidate pairs to the top K highest-variance numerical columns (max 5 pairs)
    to prevent O(P^2) combinatorial explosion.
    """

    def __init__(
        self,
        top_k: int = 4,
        max_pairs: int = 5,
        epsilon: float = 1e-6,
    ) -> None:
        self.top_k = top_k
        self.max_pairs = max_pairs
        self.epsilon = epsilon
        self.selected_pairs_: List[Tuple[str, str]] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CrossNumericTransformer":
        self.selected_pairs_ = []
        if not isinstance(X, pd.DataFrame):
            return self

        # 1. Identify valid numerical features
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if len(num_cols) < 2:
            return self

        # 2. Compute sample variances to select top_k most informative candidate features
        variances = {}
        for c in num_cols:
            var = float(X[c].var(ddof=0))
            if not np.isnan(var) and var > 0:
                variances[c] = var

        if len(variances) < 2:
            return self

        # Sort descending by variance
        sorted_cols = sorted(variances.keys(), key=lambda c: variances[c], reverse=True)[:self.top_k]

        # 3. Generate candidate pairs bounded by max_pairs
        pairs: List[Tuple[str, str]] = []
        for i in range(len(sorted_cols)):
            for j in range(i + 1, len(sorted_cols)):
                pairs.append((sorted_cols[i], sorted_cols[j]))
                if len(pairs) >= self.max_pairs:
                    break
            if len(pairs) >= self.max_pairs:
                break

        self.selected_pairs_ = pairs
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col_a, col_b in self.selected_pairs_:
            if col_a in X_out.columns and col_b in X_out.columns:
                s_a = pd.to_numeric(X_out[col_a], errors="coerce").fillna(0.0)
                s_b = pd.to_numeric(X_out[col_b], errors="coerce").fillna(0.0)

                # Linear Difference: A - B
                diff_col = f"{col_a}_sub_{col_b}"
                X_out[diff_col] = s_a - s_b

                # Safe Ratio: A / (B + eps) with zero/sign protection
                safe_b = np.where(
                    np.abs(s_b) < self.epsilon,
                    np.sign(s_b) * self.epsilon + (s_b == 0) * self.epsilon,
                    s_b,
                )
                ratio_col = f"{col_a}_ratio_{col_b}"
                ratio_val = s_a / safe_b
                X_out[ratio_col] = np.nan_to_num(ratio_val, nan=0.0, posinf=1e6, neginf=-1e6)

        return X_out


class GroupByAggregationTransformer(BaseEstimator, TransformerMixin):
    """
    Computes group-level aggregations and relative features based on:
    1. ExploreKit architecture (Katz et al., IEEE ICDM 2016): Candidate generation & cardinality filter.
    2. CatBoost architecture (Prokhorenkova et al., NeurIPS 2018): Empirical Bayes m-estimate smoothing.
    """

    def __init__(
        self,
        group_specs: Optional[List[Dict[str, Any]]] = None,
        max_cardinality: int = 1000,
        max_cardinality_ratio: float = 0.20,
        smoothing: float = 0.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.group_specs = group_specs or []
        self.max_cardinality = max_cardinality
        self.max_cardinality_ratio = max_cardinality_ratio
        self.smoothing = smoothing
        self.epsilon = epsilon

        self.group_stats_: Dict[str, pd.DataFrame] = {}
        self.global_stats_: Dict[str, Dict[str, float]] = {}
        self.skipped_specs_: List[Dict[str, Any]] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GroupByAggregationTransformer":
        self.group_stats_ = {}
        self.global_stats_ = {}
        self.skipped_specs_ = []
        n_rows = len(X)

        for i, spec in enumerate(self.group_specs):
            cat_col = spec.get("cat_col")
            num_col = spec.get("num_col")
            aggs = spec.get("aggregations", ["mean", "std"])

            if not cat_col or not num_col or cat_col not in X.columns or num_col not in X.columns:
                continue

            # Cardinality Guard (ExploreKit ICDM 2016; CatBoost 2018)
            cat_series = X[cat_col].dropna()
            n_unique = int(cat_series.nunique())
            ratio = n_unique / n_rows if n_rows > 0 else 1.0

            is_high_ratio = bool(n_rows >= 50 and ratio >= self.max_cardinality_ratio)
            if n_unique <= 1 or n_unique > self.max_cardinality or is_high_ratio:
                self.skipped_specs_.append({
                    "cat_col": cat_col,
                    "num_col": num_col,
                    "n_unique": n_unique,
                    "unique_ratio": round(ratio, 4),
                    "reason": "exceeds_max_cardinality" if n_unique > self.max_cardinality else (
                        "high_cardinality_ratio" if is_high_ratio else "constant_column"
                    ),
                })
                continue

            spec_key = f"{cat_col}__{num_col}__{i}"
            num_series = pd.to_numeric(X[num_col], errors="coerce")
            temp_df = pd.DataFrame({cat_col: X[cat_col], num_col: num_series})

            # Calculate global fallback statistics strictly on training fold
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

            # Compute standard group stats
            requested_aggs = list(set(aggs + ["count", "mean"]))
            grp = temp_df.groupby(cat_col, observed=False)[num_col].agg(requested_aggs)
            if "std" in grp.columns:
                grp["std"] = grp["std"].fillna(0.0)

            # CatBoost-style Empirical Bayes m-estimate smoothing (Prokhorenkova et al. NeurIPS 2018):
            m = float(self.smoothing)
            if m > 0.0:
                grp["smoothed_mean"] = (grp["count"] * grp["mean"] + m * g_mean) / (grp["count"] + m)
            else:
                grp["smoothed_mean"] = grp["mean"]

            self.group_stats_[spec_key] = grp

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

            # Use smoothed mean for relative calculations to eliminate small-group variance
            smoothed_mean = X_out[cat_col].map(grp["smoothed_mean"]).fillna(g_stats["mean"])
            group_std = agg_series_dict.get("std", pd.Series(g_stats["std"], index=X_out.index))

            if create_diff:
                diff_col = f"{num_col}_diff_from_{cat_col}_mean"
                X_out[diff_col] = num_series - smoothed_mean

            if create_ratio:
                ratio_col = f"{num_col}_ratio_to_{cat_col}_mean"
                safe_mean = np.where(
                    np.abs(smoothed_mean) < self.epsilon,
                    np.sign(smoothed_mean) * self.epsilon + (smoothed_mean == 0) * self.epsilon,
                    smoothed_mean,
                )
                ratio_val = num_series / safe_mean
                ratio_val = np.nan_to_num(ratio_val, nan=1.0, posinf=1e6, neginf=-1e6)
                X_out[ratio_col] = ratio_val

            if create_zscore:
                z_col = f"{num_col}_zscore_in_{cat_col}"
                safe_std = np.where(np.abs(group_std) < self.epsilon, self.epsilon, group_std)
                z_val = (num_series - smoothed_mean) / safe_std
                z_val = np.nan_to_num(z_val, nan=0.0, posinf=10.0, neginf=-10.0)
                X_out[z_col] = z_val

        return X_out


class LatentManifoldOutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Computes geometrical latent space manifold distance and outlier energy for anonymous/PCA feature blocks.
    
    Theoretical Basis:
        - Euclidean L2 Norm: ||V||_2 = sqrt(sum(V_i^2))
        - Mahalanobis / Variance-weighted normalized anomaly score
    """

    def __init__(
        self,
        feature_prefixes: Optional[List[str]] = None,
        min_block_size: int = 4,
        epsilon: float = 1e-6,
    ) -> None:
        self.feature_prefixes = feature_prefixes or ["V", "pca", "comp", "feat_"]
        self.min_block_size = min_block_size
        self.epsilon = epsilon
        self.detected_blocks_: Dict[str, List[str]] = {}
        self.block_stats_: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LatentManifoldOutlierTransformer":
        self.detected_blocks_ = {}
        self.block_stats_ = {}
        if not isinstance(X, pd.DataFrame):
            return self

        # Detect candidate feature blocks
        cols = list(X.columns)
        for prefix in self.feature_prefixes:
            matching = [c for c in cols if c.startswith(prefix) and pd.api.types.is_numeric_dtype(X[c])]
            if len(matching) >= self.min_block_size:
                clean_name = prefix.rstrip("_")
                self.detected_blocks_[clean_name] = matching
                # Store training mean and std for variance-weighted Mahalanobis proxy
                arr = X[matching].to_numpy(dtype=np.float64)
                means = np.nanmean(arr, axis=0)
                stds = np.nanstd(arr, axis=0)
                stds = np.where(stds < self.epsilon, 1.0, stds)
                self.block_stats_[clean_name] = {"mean": means, "std": stds}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for block_name, cols in self.detected_blocks_.items():
            if all(c in X_out.columns for c in cols):
                arr = X_out[cols].to_numpy(dtype=np.float64)
                arr = np.nan_to_num(arr, nan=0.0)

                # 1. Euclidean L2 Norm
                l2_norm = np.linalg.norm(arr, axis=1)
                X_out[f"{block_name}_l2_norm"] = l2_norm

                # 2. Normalized Anomaly / Mahalanobis Proxy
                if block_name in self.block_stats_:
                    means = self.block_stats_[block_name]["mean"]
                    stds = self.block_stats_[block_name]["std"]
                    z_scores = (arr - means) / stds
                    mahalanobis_proxy = np.sqrt(np.sum(z_scores ** 2, axis=1))
                    X_out[f"{block_name}_mahalanobis_proxy"] = mahalanobis_proxy

        return X_out


class AutomaticTemporalTransformer(BaseEstimator, TransformerMixin):
    """
    Automatically detects elapsed time / continuous seconds columns and synthesizes:
    1. Hour of Day: (Time // 3600) % 24
    2. Cyclical Harmonics: Hour_Sin & Hour_Cos
    """

    def __init__(
        self,
        time_col_names: Optional[List[str]] = None,
        min_seconds_range: float = 3600.0,
    ) -> None:
        self.time_col_names = time_col_names or ["time", "timestamp", "seconds", "sec", "elapsed_time"]
        self.min_seconds_range = min_seconds_range
        self.detected_time_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AutomaticTemporalTransformer":
        self.detected_time_cols_ = []
        if not isinstance(X, pd.DataFrame):
            return self

        for col in X.columns:
            col_lower = col.lower()
            if any(cand == col_lower or cand in col_lower for cand in self.time_col_names):
                if pd.api.types.is_numeric_dtype(X[col]):
                    col_range = float(X[col].max() - X[col].min()) if len(X[col]) > 0 else 0.0
                    if col_range >= self.min_seconds_range:
                        self.detected_time_cols_.append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.detected_time_cols_:
            if col in X_out.columns:
                series = pd.to_numeric(X_out[col], errors="coerce").fillna(0.0)
                hour = (series // 3600.0) % 24.0
                theta = (2.0 * np.pi * hour) / 24.0

                X_out[f"{col}_hour"] = hour
                X_out[f"{col}_hour_sin"] = np.sin(theta)
                X_out[f"{col}_hour_cos"] = np.cos(theta)

        return X_out
