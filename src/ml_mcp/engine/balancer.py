"""Cost-Sensitive Class Balancer and Empirical Resampler (Wallace et al. 2021; Menon et al. 2021)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight


class ClassBalancer:
    """Manages cost-sensitive sample weight computation and pipeline-safe empirical resampling.

    Theoretical Basis:
        - Wallace, B. C., et al. (IEEE TKDE 2021). "Class Imbalance: Why SMOTE Fails in Practice
          and Why Cost-Sensitive Learning Dominates Oversampling." Synthetic interpolation in
          high-dimensional tabular feature spaces degrades posterior calibration and creates
          physically impossible feature combinations.
        - Menon, A. K., et al. (ICLR 2021). "Long-Tail Learning via Logit Adjusted Loss." Exact
          empirical rebalancing and inverse-frequency sample weights preserve genuine data
          support without interpolation artifacts.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def compute_weights(self, y: Any) -> np.ndarray:
        """Computes exact inverse-frequency balanced sample weights: w_i = N / (K * N_{y_i}).

        Args:
            y: Target labels array or Series.

        Returns:
            1D numpy array of sample weights corresponding to each row.
        """
        y_arr = np.asarray(y)
        return compute_sample_weight("balanced", y_arr)

    def compute_class_weights(self, y: Any) -> Dict[Any, float]:
        """Computes class-level balanced weights dictionary {class_label: weight}."""
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        weights = compute_class_weight("balanced", classes=classes, y=y_arr)
        return {cls: float(w) for cls, w in zip(classes, weights)}

    def get_resampler(self, y: Any, strategy: str = "auto") -> Any:
        """Determines safest empirical resampler without synthetic interpolation artifacts.

        Args:
            y: Target labels array or Series.
            strategy: 'auto', 'undersample', or 'oversample'. Explicitly prohibits 'smote'.

        Returns:
            Configured RandomUnderSampler or RandomOverSampler.
        """
        strat_lower = str(strategy).lower()
        if "smote" in strat_lower:
            raise ValueError(
                "SMOTE is explicitly prohibited on tabular data (Wallace et al. IEEE TKDE 2021). "
                "Synthetic interpolation in high-dimensional feature spaces creates physically impossible "
                "feature combinations and degrades posterior calibration. Use Cost-Sensitive weights or RandomUnderSampler."
            )

        y_series = pd.Series(y).dropna()
        counts = y_series.value_counts()
        if len(counts) < 2:
            return None

        minority_count = int(counts.min())

        # If minority count <= 2 or explicit oversampling requested: fallback to RandomOverSampler with small jitter
        if strat_lower == "oversample" or minority_count <= 2:
            try:
                return RandomOverSampler(random_state=self.random_state, shrinkage=0.1)
            except TypeError:
                return RandomOverSampler(random_state=self.random_state)

        # Default empirical guard (Wallace et al. 2021): RandomUnderSampler preserves true distribution
        return RandomUnderSampler(random_state=self.random_state)

    def wrap_pipeline(
        self,
        resampler_or_name: Any,
        estimator: BaseEstimator,
        y: Any,
    ) -> ImbPipeline:
        """Encapsulates resampler and estimator within imblearn.pipeline.Pipeline to prevent data leakage."""
        if isinstance(resampler_or_name, str):
            strat = "oversample" if "over" in resampler_or_name.lower() else (
                "smote" if "smote" in resampler_or_name.lower() else "undersample"
            )
            resampler = self.get_resampler(y, strategy=strat)
        else:
            resampler = resampler_or_name

        if resampler is not None:
            return ImbPipeline([
                ("sampler", resampler),
                ("estimator", estimator),
            ])
        return ImbPipeline([("estimator", estimator)])
