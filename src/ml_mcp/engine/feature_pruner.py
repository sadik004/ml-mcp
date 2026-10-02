"""OpenFE-inspired Gradient & Tree Importance Feature Pruner (Zhang et al., ICML 2023)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder

from ml_mcp.schemas.feature import FeaturePruningReportDTO


class GradientFeatureSelector(BaseEstimator, TransformerMixin):
    """Gradient and tree importance feature selector following OpenFE architecture."""

    def __init__(
        self,
        top_k: Optional[int] = None,
        importance_threshold: float = 0.005,
        task_type: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.top_k = top_k
        self.importance_threshold = importance_threshold
        self.task_type = task_type
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.original_feature_count_: int = 0
        self.task_type_: str = "classification"

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any) -> "GradientFeatureSelector":
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            X_df = X.copy()
        else:
            X_arr = np.asarray(X)
            cols = [f"feat_{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=cols)

        self.original_feature_count_ = len(cols)
        y_arr = np.asarray(y)

        # 1. Infer task type
        if self.task_type == "auto":
            valid_y = y_arr[~pd.isna(y_arr)] if pd.isna(y_arr).any() else y_arr
            unique_vals = np.unique(valid_y)
            if y_arr.dtype == object or y_arr.dtype == bool or len(unique_vals) <= 15:
                self.task_type_ = "classification"
            else:
                self.task_type_ = "regression"
        else:
            self.task_type_ = self.task_type.lower()

        # 2. Encode non-numeric features for HistGradientBoosting
        X_encoded = pd.DataFrame(index=X_df.index)
        for col in cols:
            series = X_df[col]
            if pd.api.types.is_numeric_dtype(series):
                X_encoded[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)
            else:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                reshaped = series.astype(str).to_numpy().reshape(-1, 1)
                X_encoded[col] = enc.fit_transform(reshaped).ravel()

        # 3. Fit fast baseline HistGradientBooster
        if self.task_type_ == "classification":
            booster = HistGradientBoostingClassifier(
                max_iter=30,
                max_depth=5,
                random_state=self.random_state,
            )
        else:
            booster = HistGradientBoostingRegressor(
                max_iter=30,
                max_depth=5,
                random_state=self.random_state,
            )

        booster.fit(X_encoded.to_numpy(), y_arr)

        # 4. Extract feature importances via permutation importance
        perm = permutation_importance(
            booster,
            X_encoded.to_numpy(),
            y_arr,
            n_repeats=3,
            random_state=self.random_state,
            n_jobs=1,
        )
        raw_importances = np.maximum(0.0, perm.importances_mean)
        total_imp = float(np.sum(raw_importances))

        if total_imp > 0:
            norm_importances = raw_importances / total_imp
        else:
            norm_importances = np.ones_like(raw_importances) / len(raw_importances)

        self.feature_importances_ = {
            col: round(float(norm_importances[i]), 5) for i, col in enumerate(cols)
        }

        # 5. Selection and Pruning Logic
        sorted_features = sorted(cols, key=lambda c: self.feature_importances_[c], reverse=True)

        if self.top_k is not None and self.top_k > 0:
            selected = sorted_features[: self.top_k]
        else:
            selected = [c for c in sorted_features if self.feature_importances_[c] >= self.importance_threshold]

        # Safety: At least retain the single best feature to avoid empty feature space
        if not selected and sorted_features:
            selected = [sorted_features[0]]

        self.selected_features_ = selected
        self.dropped_features_ = [c for c in cols if c not in self.selected_features_]

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].copy()
        else:
            X_arr = np.asarray(X)
            cols = [f"feat_{i}" for i in range(X_arr.shape[1])]
            indices = [cols.index(c) for c in self.selected_features_ if c in cols]
            return X_arr[:, indices]

    def get_report(self) -> FeaturePruningReportDTO:
        return FeaturePruningReportDTO(
            original_feature_count=self.original_feature_count_,
            pruned_feature_count=len(self.dropped_features_),
            selected_features=self.selected_features_,
            dropped_features=self.dropped_features_,
            feature_importances=self.feature_importances_,
        )
