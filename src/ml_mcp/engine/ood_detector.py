"""Out-of-Distribution (OOD) Anomaly Detector Engine."""
from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml_mcp.schemas.safety import OODReportDTO

logger = logging.getLogger(__name__)


class OODDetector:
    """Detects Out-of-Distribution / unseen anomalous samples using Isolation Forest or Mahalanobis Distance."""

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.method = method.lower()
        self.contamination = contamination
        self.random_state = random_state
        self.fitted = False

        # Internal state
        self._iso_forest: Optional[IsolationForest] = None
        self._mean_vector: Optional[np.ndarray] = None
        self._inv_cov_matrix: Optional[np.ndarray] = None
        self._anomaly_threshold: float = 0.0

    def _to_numpy(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def fit(self, X: Any) -> "OODDetector":
        """Fit OOD detector on in-distribution training data."""
        X_arr = self._to_numpy(X)

        if "mahalanobis" in self.method:
            self._mean_vector = np.mean(X_arr, axis=0)
            cov = np.cov(X_arr, rowvar=False)

            # Ensure 2D covariance matrix even for 1D feature
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = cov.reshape(-1, 1)

            # Regularize diagonal to guard against singular/collinear matrices
            epsilon = 1e-6
            cov_reg = cov + np.eye(cov.shape[0]) * epsilon
            self._inv_cov_matrix = np.linalg.pinv(cov_reg)

            # Compute training Mahalanobis distances to set percentile threshold
            diff = X_arr - self._mean_vector
            # Vectorized Mahalanobis distance computation
            train_dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            train_dist = np.sqrt(np.maximum(0.0, train_dist_sq))

            percentile_cutoff = (1.0 - self.contamination) * 100.0
            self._anomaly_threshold = float(np.percentile(train_dist, percentile_cutoff))
        else:
            # Default Isolation Forest
            self._iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._iso_forest.fit(X_arr)
            # Offset_ in IsolationForest is the decision threshold
            self._anomaly_threshold = float(self._iso_forest.offset_)

        self.fitted = True
        return self

    def detect(self, X: Any) -> OODReportDTO:
        """Scan input data and generate OOD report."""
        if not self.fitted:
            raise RuntimeError("OODDetector must be fit() on training data before detect().")

        X_arr = self._to_numpy(X)
        total_samples = len(X_arr)
        if total_samples == 0:
            raise ValueError("Input dataset X cannot be empty.")

        if "mahalanobis" in self.method:
            detector_name = "MahalanobisDistance"
            diff = X_arr - self._mean_vector
            dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            dists = np.sqrt(np.maximum(0.0, dist_sq))
            is_ood = dists > self._anomaly_threshold
            ood_count = int(np.sum(is_ood))
        else:
            detector_name = "IsolationForest"
            preds = self._iso_forest.predict(X_arr)
            # IsolationForest flags -1 as anomaly/outlier
            ood_count = int(np.sum(preds == -1))

        ood_ratio = float(ood_count / total_samples)

        return OODReportDTO(
            detector_name=detector_name,
            total_samples=total_samples,
            ood_detected_count=ood_count,
            ood_ratio=round(ood_ratio, 4),
            anomaly_threshold=round(self._anomaly_threshold, 4),
        )
