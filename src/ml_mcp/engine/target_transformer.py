"""Residual-Driven Skewed Target Transformer (Tarasiuk 2021/2023)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger(__name__)


class SkewedTargetTransformer:
    """Wraps regression estimators to normalize residuals with negative scale protection.

    Theoretical Basis:
        - Tarasiuk, J. (Statistical Papers 2021/2023). "Optimum Power Transformations and Empirical
          Score Tests for Non-Normal Residuals." Gauss-Markov and regression normality conditions
          strictly govern model residuals epsilon = y - y_hat, not the marginal distribution of y.
          Transforming based on residual skewness prevents artificial distortion of multimodal linear targets.
    """

    def __init__(self, skew_threshold: float = 1.5) -> None:
        self.skew_threshold = skew_threshold

    def _compute_residuals(self, y: np.ndarray, X: Optional[Any] = None) -> np.ndarray:
        """Computes residuals against a lightweight baseline linear model or central tendency."""
        if X is not None:
            try:
                from sklearn.linear_model import Ridge
                X_mat = np.asarray(X)
                if X_mat.ndim == 1:
                    X_mat = X_mat.reshape(-1, 1)
                if X_mat.shape[0] == len(y) and len(y) >= 10:
                    model = Ridge(alpha=1.0)
                    model.fit(X_mat, y)
                    preds = model.predict(X_mat)
                    return y - preds
            except Exception as e:
                logger.debug(f"Linear baseline residual fitting failed: {e}")
        # Fallback baseline: deviation from empirical mean (preserves exact marginal skewness)
        return y - np.mean(y)

    def wrap_estimator(self, estimator: BaseEstimator, y: Any, X: Optional[Any] = None) -> BaseEstimator:
        """Evaluates residual skewness and applies log1p or Yeo-Johnson power transform if needed.

        Args:
            estimator: Scikit-learn compliant regression model.
            y: Target values array or Series.
            X: Optional feature matrix for baseline residual fitting.

        Returns:
            Wrapped TransformedTargetRegressor or original estimator.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        y_valid = y_arr[~np.isnan(y_arr)]

        if len(y_valid) < 10:
            return estimator

        # Compute baseline residual skewness (Tarasiuk 2021/2023)
        residuals = self._compute_residuals(y_valid, X=X)
        skewness = float(skew(residuals))

        # Check if residual skewness exceeds threshold
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

    def transform_target(self, y: Any, method: str = "auto", X: Optional[Any] = None) -> Dict[str, Any]:
        """Evaluates residual skewness and transforms target values directly using log1p or Yeo-Johnson.

        Args:
            y: Target values array or Series.
            method: "auto", "log1p", or "yeo-johnson" / "box-cox".
            X: Optional feature matrix for baseline residual fitting.

        Returns:
            Dict containing transformed values, skewness, method used, and summary stats.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        y_valid = y_arr[~np.isnan(y_arr)]

        if len(y_valid) < 3:
            return {
                "method": "none",
                "skewness": 0.0,
                "residual_skewness": 0.0,
                "is_skewed": False,
                "min_value": float(np.min(y_arr)) if len(y_arr) > 0 else 0.0,
                "max_value": float(np.max(y_arr)) if len(y_arr) > 0 else 0.0,
                "transformed_values": y_arr.tolist(),
            }

        residuals = self._compute_residuals(y_valid, X=X)
        skewness = float(skew(residuals))
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
            "residual_skewness": round(skewness, 4),
            "is_skewed": is_skewed,
            "min_value": round(min_val, 4),
            "max_value": round(float(np.max(y_valid)), 4),
            "count": int(len(y_valid)),
            "transformed_values": transformed.tolist() if len(transformed) <= 1000 else transformed[:1000].tolist(),
        }
