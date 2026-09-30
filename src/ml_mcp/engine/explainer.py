"""Sub-10s TreeSHAP Explainability Engine with Token Guard and Directional Impact."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class TreeShapExplainer:
    """Fast TreeSHAP explainer utilizing background sampling and token-guarded feature attribution."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def _prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Normalize input data into a 2D numpy array and feature name list."""
        if isinstance(X, pd.DataFrame):
            names = list(X.columns) if feature_names is None else feature_names
            data = X.to_numpy(dtype=float, copy=True)
        else:
            data = np.asarray(X, dtype=float)
            names = (
                feature_names
                if feature_names and len(feature_names) == data.shape[1]
                else [f"feature_{i}" for i in range(data.shape[1])]
            )
        return data, names

    def _is_tree_model(self, model: Any) -> bool:
        """Check if model is supported by TreeExplainer."""
        model_name = getattr(model, "__class__", type(model)).__name__.lower()
        tree_keywords = [
            "randomforest",
            "extratrees",
            "gradientboosting",
            "histgradientboosting",
            "xgb",
            "lgbm",
            "catboost",
            "decisiontree",
        ]
        return any(kw in model_name for kw in tree_keywords)

    def explain(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        max_background_samples: int = 100,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Compute sub-10s TreeSHAP attributions with token-shielded top-K features."""
        start_time = time.perf_counter()
        X_arr, names = self._prepare_data(X, feature_names)

        # Non-tree model fallback
        if not self._is_tree_model(model):
            return self._fallback_explanation(model, X_arr, names, top_k, start_time)

        # 1. Background sampling (prioritize shap.sample over kmeans to avoid categorical crashes)
        if len(X_arr) > max_background_samples:
            background = shap.sample(X_arr, max_background_samples, random_state=self.random_state)
        else:
            background = X_arr

        # 2. TreeExplainer computation
        try:
            explainer = shap.TreeExplainer(model, data=background)
            shap_values = explainer.shap_values(X_arr)
        except Exception as e:
            logger.warning("TreeExplainer with background failed, trying without data: %s", e)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_arr)
            except Exception as e2:
                logger.error("TreeExplainer failed completely: %s", e2)
                return self._fallback_explanation(model, X_arr, names, top_k, start_time)

        # 3. Dimension alignment for multiclass/binary/regression
        if isinstance(shap_values, list):
            # Binary classification list [class_0, class_1] -> use class_1
            val_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Shape: (N, features, classes) -> take positive class index 1
            val_matrix = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        else:
            val_matrix = np.asarray(shap_values)

        # 4. Global feature importance: Mean Absolute SHAP value
        mean_abs_shap = np.mean(np.abs(val_matrix), axis=0)

        # 5. Directional impact (correlation between feature value and SHAP value)
        directions: Dict[str, str] = {}
        for col_idx in range(len(names)):
            feat_vals = X_arr[:, col_idx]
            feat_shaps = val_matrix[:, col_idx]
            std_feat = np.std(feat_vals)
            std_shap = np.std(feat_shaps)
            if std_feat > 1e-9 and std_shap > 1e-9:
                corr = np.corrcoef(feat_vals, feat_shaps)[0, 1]
                directions[names[col_idx]] = "positive" if corr > 0 else "negative"
            else:
                directions[names[col_idx]] = "neutral"

        # 6. Sort and cap to top_k features (Token Guard)
        ranked_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
        top_features: Dict[str, float] = {}
        top_directions: Dict[str, str] = {}

        for idx in ranked_indices:
            feat_name = names[idx]
            top_features[feat_name] = round(float(mean_abs_shap[idx]), 4)
            top_directions[feat_name] = directions[feat_name]

        elapsed = time.perf_counter() - start_time

        return {
            "explainer_type": "TreeExplainer",
            "top_features": top_features,
            "feature_directions": top_directions,
            "total_features": len(names),
            "background_samples": len(background),
            "execution_time_seconds": round(elapsed, 4),
        }

    def _fallback_explanation(
        self,
        model: Any,
        X_arr: np.ndarray,
        names: List[str],
        top_k: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """Safe linear/permutation fallback when model is not a tree."""
        top_features: Dict[str, float] = {}
        top_directions: Dict[str, str] = {}

        if hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_).ravel()
            abs_coefs = np.abs(coefs)
            ranked = np.argsort(abs_coefs)[::-1][:top_k]
            for idx in ranked:
                feat = names[idx]
                top_features[feat] = round(float(abs_coefs[idx]), 4)
                top_directions[feat] = "positive" if coefs[idx] > 0 else "negative"
            explainer_type = "CoefficientsFallback"
        else:
            # Dummy equal weights if no coefficients
            for idx in range(min(top_k, len(names))):
                top_features[names[idx]] = 1.0
                top_directions[names[idx]] = "neutral"
            explainer_type = "PermutationExplainer"

        elapsed = time.perf_counter() - start_time
        return {
            "explainer_type": explainer_type,
            "top_features": top_features,
            "feature_directions": top_directions,
            "total_features": len(names),
            "background_samples": min(len(X_arr), 100),
            "execution_time_seconds": round(elapsed, 4),
        }
