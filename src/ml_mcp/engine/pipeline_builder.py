"""Defensive Zero-Leakage Pipeline Builder with Omnipresent Imputers, MICE, and Adaptive Encoders."""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder


class DefensivePipelineBuilder:
    """Builds sealed zero-leakage ColumnTransformer pipelines resilient to unexpected test NaNs and preserving covariance."""

    def __init__(
        self,
        high_cardinality_threshold: int = 50,
        imputation_strategy: str = "median",
    ) -> None:
        self.high_cardinality_threshold = high_cardinality_threshold
        self.imputation_strategy = imputation_strategy

    def build_pipeline(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        estimator: Optional[BaseEstimator] = None,
        high_cardinality_threshold: Optional[int] = None,
        imputation_strategy: Optional[str] = None,
    ) -> Pipeline:
        """Constructs an omnipresent imputer pipeline tailored to input feature distributions.

        Supports:
            - 'median': Univariate robust median imputation
            - 'mean': Univariate mean imputation
            - 'iterative': Multivariate Imputation by Chained Equations (MICE / BayesianRidge)

        Returns:
            Fittable Scikit-Learn Pipeline object.
        """
        threshold = high_cardinality_threshold or self.high_cardinality_threshold
        imp_strat = (imputation_strategy or self.imputation_strategy).lower()
        feature_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df

        num_cols: List[str] = []
        low_card_cats: List[str] = []
        high_card_cats: List[str] = []

        for col in feature_df.columns:
            series = feature_df[col]
            if pd.api.types.is_numeric_dtype(series):
                num_cols.append(col)
            elif series.dtype == "object" or isinstance(series.dtype, (pd.CategoricalDtype, pd.StringDtype)):
                n_unique = series.nunique()
                if n_unique > threshold:
                    high_card_cats.append(col)
                else:
                    low_card_cats.append(col)

        transformers = []

        # 1. Numeric pipeline: Imputer (Median / Mean / MICE Iterative) + RobustScaler
        if num_cols:
            if imp_strat == "iterative":
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                from sklearn.linear_model import BayesianRidge

                num_imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=10,
                    random_state=42,
                    sample_posterior=False,
                )
            elif imp_strat == "mean":
                num_imputer = SimpleImputer(strategy="mean")
            else:
                num_imputer = SimpleImputer(strategy="median")

            num_pipe = Pipeline([
                ("imputer", num_imputer),
                ("scaler", RobustScaler()),
            ])
            transformers.append(("num", num_pipe, num_cols))

        # 2. Low-cardinality categorical: Missing Imputer + OneHotEncoder (ignoring unseen categories)
        if low_card_cats:
            low_cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("low_cat", low_cat_pipe, low_card_cats))

        # 3. High-cardinality categorical: Missing Imputer + Out-of-Fold TargetEncoder
        if high_card_cats:
            high_cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", TargetEncoder(cv=5, smooth="auto")),
            ])
            transformers.append(("high_cat", high_cat_pipe, high_card_cats))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        steps = [("preprocessor", preprocessor)]
        if estimator is not None:
            steps.append(("estimator", estimator))

        return Pipeline(steps)
