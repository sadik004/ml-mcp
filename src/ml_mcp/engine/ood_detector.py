"""Out-of-Distribution (OOD) Anomaly Detector Engine with Energy-Based Detection (Liu et al. NeurIPS 2020)."""
from __future__ import annotations

import logging
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import IsolationForest

from ml_mcp.schemas.safety import OODReportDTO

logger = logging.getLogger(__name__)


def compute_free_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes Helmholtz Free Energy E(x; f) = -T * logsumexp(logits / T).
    
    Theoretical Basis: Liu et al. (NeurIPS 2020). Energy-Based OOD Detection.
    Free energy is provably proportional to negative log marginal density -log p(x).
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        # Binary log-odds: [0, logits]
        logits = np.column_stack([np.zeros_like(logits), logits])
    scaled = logits / temperature
    return -temperature * logsumexp(scaled, axis=1)


class OODDetector:
    """Detects Out-of-Distribution anomalous samples using Energy Scoring, Mahalanobis Distance, or Isolation Forest."""

    def __init__(
        self,
        method: str = "energy",
        contamination: float = 0.05,
        temperature: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.method = method.lower()
        self.contamination = contamination
        self.temperature = temperature
        self.random_state = random_state
        self.fitted = False

        self._iso_forest: Optional[IsolationForest] = None
        self._mean_vector: Optional[np.ndarray] = None
        self._inv_cov_matrix: Optional[np.ndarray] = None
        self._anomaly_threshold: float = 0.0

    def _to_numpy(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def fit(self, X: Any) -> "OODDetector":
        """Fit OOD detector on in-distribution training data or training logits."""
        X_arr = self._to_numpy(X)

        if "energy" in self.method:
            # Energy-based OOD detection: X_arr is treated as logits/representations
            energies = compute_free_energy(X_arr, temperature=self.temperature)
            percentile_cutoff = (1.0 - self.contamination) * 100.0
            self._anomaly_threshold = float(np.percentile(energies, percentile_cutoff))

        elif "mahalanobis" in self.method:
            self._mean_vector = np.mean(X_arr, axis=0)
            cov = np.cov(X_arr, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = cov.reshape(-1, 1)

            epsilon = 1e-6
            cov_reg = cov + np.eye(cov.shape[0]) * epsilon
            self._inv_cov_matrix = np.linalg.pinv(cov_reg)

            diff = X_arr - self._mean_vector
            train_dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            train_dist = np.sqrt(np.maximum(0.0, train_dist_sq))

            percentile_cutoff = (1.0 - self.contamination) * 100.0
            self._anomaly_threshold = float(np.percentile(train_dist, percentile_cutoff))

        else:
            # Isolation Forest fallback
            self._iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._iso_forest.fit(X_arr)
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

        if "energy" in self.method:
            detector_name = "EnergyBasedOOD"
            energies = compute_free_energy(X_arr, temperature=self.temperature)
            is_ood = energies > self._anomaly_threshold
            ood_count = int(np.sum(is_ood))

        elif "mahalanobis" in self.method:
            detector_name = "MahalanobisDistance"
            diff = X_arr - self._mean_vector
            dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            dists = np.sqrt(np.maximum(0.0, dist_sq))
            is_ood = dists > self._anomaly_threshold
            ood_count = int(np.sum(is_ood))

        else:
            detector_name = "IsolationForest"
            preds = self._iso_forest.predict(X_arr)
            ood_count = int(np.sum(preds == -1))

        ood_ratio = float(ood_count / total_samples)

        return OODReportDTO(
            detector_name=detector_name,
            total_samples=total_samples,
            ood_detected_count=ood_count,
            ood_ratio=round(ood_ratio, 4),
            anomaly_threshold=round(self._anomaly_threshold, 4),
        )
