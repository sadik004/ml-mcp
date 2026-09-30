"""Skewed Target Transformer with negative scale protection."""
from __future__ import annotations

from typing import Any
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
