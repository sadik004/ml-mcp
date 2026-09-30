"""Class Balancer using dynamic k-neighbors and pipeline encapsulation."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.base import BaseEstimator


class ClassBalancer:
    """Manages pipeline-safe oversampling with small minority class fallback guards."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def get_resampler(self, y: Any) -> Any:
        """Determines safest resampler based on minority class sample size.

        Returns:
            Configured SMOTE or RandomOverSampler.
        """
        y_series = pd.Series(y).dropna()
        counts = y_series.value_counts()
        if len(counts) < 2:
            return None

        minority_count = int(counts.min())

        # If minority class <= 2 samples, SMOTE is mathematically unfeasible; fallback to RandomOverSampler
        if minority_count <= 2:
            return RandomOverSampler(random_state=self.random_state)

        # Dynamic k_neighbors: must be strictly < minority_count
        k_neighbors = min(5, max(1, minority_count - 1))
        return SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)

    def wrap_pipeline(
        self,
        resampler_or_name: Any,
        estimator: BaseEstimator,
        y: Any,
    ) -> ImbPipeline:
        """Encapsulates resampler and estimator within imblearn.pipeline.Pipeline to prevent data leakage."""
        if isinstance(resampler_or_name, str):
            resampler = self.get_resampler(y)
        else:
            resampler = resampler_or_name

        if resampler is not None:
            return ImbPipeline([
                ("sampler", resampler),
                ("estimator", estimator),
            ])
        return ImbPipeline([("estimator", estimator)])
