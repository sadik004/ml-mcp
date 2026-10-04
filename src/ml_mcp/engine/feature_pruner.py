"""OpenFE-inspired Gradient & Tree Importance Feature Pruner with OOF Permutation (Zhang et al. ICML 2023; Breiman 2001; Molnar 2020)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from ml_mcp.schemas.feature import FeaturePruningReportDTO


class GradientFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Gradient and tree importance feature selector following OpenFE architecture.
    Evaluates permutation importance strictly Out-Of-Fold (OOF) to prevent memorization trap.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        importance_threshold: float = 0.005,
        task_type: str = "auto",
        cv: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        from ml_mcp.config import get_settings
        self.top_k = top_k
        self.importance_threshold = importance_threshold
        self.task_type = task_type
        self.cv = cv
        self.random_state = random_state if random_state is not None else get_settings().random_state

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

        X_mat = X_encoded.to_numpy()
        n_samples = len(X_mat)

        # 3. Determine if Out-Of-Fold (OOF) cross-validation is feasible (Breiman 2001; Molnar 2020)
        can_cv = self.cv > 1 and n_samples >= max(15, self.cv * 3)
        if can_cv and self.task_type_ == "classification":
            _, class_counts = np.unique(y_arr, return_counts=True)
            if np.min(class_counts) < self.cv:
                can_cv = False

        if can_cv:
            if self.task_type_ == "classification":
                splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

            fold_importances: List[np.ndarray] = []
            for train_idx, val_idx in splitter.split(X_mat, y_arr):
                X_tr, y_tr = X_mat[train_idx], y_arr[train_idx]
                X_va, y_va = X_mat[val_idx], y_arr[val_idx]

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
                booster.fit(X_tr, y_tr)

                # Evaluate permutation importance strictly on unseen validation fold
                perm = permutation_importance(
                    booster,
                    X_va,
                    y_va,
                    n_repeats=3,
                    random_state=self.random_state,
                    n_jobs=1,
                )
                fold_importances.append(perm.importances_mean)

            raw_importances = np.maximum(0.0, np.mean(fold_importances, axis=0))
        else:
            # Fallback for small datasets or unbalanced singletons
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
            booster.fit(X_mat, y_arr)

            perm = permutation_importance(
                booster,
                X_mat,
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

        # 4. Selection and Pruning Logic
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
