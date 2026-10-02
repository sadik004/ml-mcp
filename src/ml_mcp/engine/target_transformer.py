"""Skewed Target Transformer with negative scale protection."""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer


class SkewedTargetTransformer:
    """Wraps regression estimators to normalize right-skewed targets with negative value guards."""

    def __init__(self, skew_threshold: float = 1.5) -> None:
        self.skew_threshold = skew_threshold

    def wrap_estimator(self, estimator: BaseEstimator, y: Any) -> BaseEstimator:
        """Evaluates target skewness and applies log1p or Yeo-Johnson power transform if needed.

        Args:
            estimator: Scikit-learn compliant regression model.
            y: Target values array or Series.

        Returns:
            Wrapped TransformedTargetRegressor or original estimator.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        y_valid = y_arr[~np.isnan(y_arr)]

        if len(y_valid) < 10:
            return estimator

        # Compute sample skewness
        skewness = float(skew(y_valid))

        # Check if skewness exceeds threshold
        if abs(skewness) > self.skew_threshold:
            min_val = float(np.min(y_valid))

            if min_val >= 0.0:
                # Positive skewed: safe to apply log(1+y)
                return TransformedTargetRegressor(
                    regressor=estimator,
                    func=np.log1p,
                    inverse_func=np.expm1,
                )
            else:
                # Negative values present: log1p is invalid! Use Yeo-Johnson power transform
                return TransformedTargetRegressor(
                    regressor=estimator,
                    transformer=PowerTransformer(method="yeo-johnson"),
                )

        return estimator

    def transform_target(self, y: Any, method: str = "auto") -> Dict[str, Any]:
        """Evaluates skewness and transforms target values directly using log1p or Yeo-Johnson.

        Args:
            y: Target values array or Series.
            method: "auto", "log1p", or "yeo-johnson" / "box-cox".

        Returns:
            Dict containing transformed values, skewness, method used, and summary stats.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        y_valid = y_arr[~np.isnan(y_arr)]

        if len(y_valid) < 3:
            return {
                "method": "none",
                "skewness": 0.0,
                "is_skewed": False,
                "min_value": float(np.min(y_arr)) if len(y_arr) > 0 else 0.0,
                "max_value": float(np.max(y_arr)) if len(y_arr) > 0 else 0.0,
                "transformed_values": y_arr.tolist(),
            }

        skewness = float(skew(y_valid))
        is_skewed = abs(skewness) > self.skew_threshold
        min_val = float(np.min(y_valid))

        if method == "auto":
            if is_skewed:
                if min_val >= 0.0:
                    chosen_method = "log1p"
                    transformed = np.log1p(y_arr)
                else:
                    chosen_method = "yeo-johnson"
                    pt = PowerTransformer(method="yeo-johnson")
                    transformed = pt.fit_transform(y_arr.reshape(-1, 1)).flatten()
            else:
                chosen_method = "none"
                transformed = y_arr
        elif method == "log1p":
            chosen_method = "log1p"
            transformed = np.log1p(np.maximum(y_arr, 0.0))
        elif method in ("yeo-johnson", "box-cox"):
            chosen_method = "yeo-johnson"
            pt = PowerTransformer(method="yeo-johnson")
            transformed = pt.fit_transform(y_arr.reshape(-1, 1)).flatten()
        else:
            chosen_method = "none"
            transformed = y_arr

        return {
            "method": chosen_method,
            "skewness": round(skewness, 4),
            "is_skewed": is_skewed,
            "min_value": round(min_val, 4),
            "max_value": round(float(np.max(y_valid)), 4),
            "count": int(len(y_valid)),
            "transformed_values": transformed.tolist() if len(transformed) <= 1000 else transformed[:1000].tolist(),
        }
