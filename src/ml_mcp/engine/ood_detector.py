"""Out-of-Distribution (OOD) Anomaly Detector Engine with Energy-Based Detection (Liu et al. NeurIPS 2020)
and Robust Mahalanobis Distance for Tabular Representations.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_mcp.schemas.safety import OODReportDTO

logger = logging.getLogger(__name__)


def compute_free_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes Helmholtz Free Energy E(x; f) = -T * logsumexp(logits / T).

    Theoretical Basis: Liu et al. (NeurIPS 2020). Energy-Based OOD Detection.
    Free energy is provably proportional to negative log marginal density -log p(x).
    Numerically stabilized via shifted log-sum-exp identity to prevent overflow on extreme representations.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        # Binary log-odds: [0, logits]
        logits = np.column_stack([np.zeros_like(logits), logits])

    temp = max(1e-6, float(temperature))
    max_val = np.max(logits, axis=1)
    scaled_diff = (logits - max_val[:, np.newaxis]) / temp
    # Clip extreme underflow to avoid precision artifacts
    scaled_diff = np.clip(scaled_diff, -700.0, 0.0)
    lse = logsumexp(scaled_diff, axis=1)
    return -max_val - temp * lse


class OODDetector:
    """Detects Out-of-Distribution anomalous samples using Mahalanobis Distance, Energy Scoring, or Isolation Forest."""

    def __init__(
        self,
        method: str = "mahalanobis",
        contamination: float = 0.05,
        temperature: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        from ml_mcp.config import get_settings
        self.method = method.lower()
        self.contamination = contamination
        self.temperature = temperature
        self.random_state = random_state if random_state is not None else get_settings().random_state
        self.fitted = False

        self._scaler = StandardScaler()
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
        if len(X_arr) == 0:
            raise ValueError("Input dataset X cannot be empty.")

        # Disjoint calibration split to avoid circular in-sample threshold calibration
        if len(X_arr) >= 20:
            tr_idx, cal_idx = train_test_split(
                np.arange(len(X_arr)), test_size=0.20, random_state=self.random_state
            )
            X_tr, X_cal = X_arr[tr_idx], X_arr[cal_idx]
        else:
            X_tr, X_cal = X_arr, X_arr

        if "energy" in self.method:
            # Energy-based OOD detection: expects logits/representations
            energies_cal = compute_free_energy(X_cal, temperature=self.temperature)
            percentile_cutoff = (1.0 - self.contamination) * 100.0
            self._anomaly_threshold = float(np.percentile(energies_cal, percentile_cutoff))

        elif "mahalanobis" in self.method:
            # Fit scaler strictly on in-distribution training split
            X_tr_scaled = self._scaler.fit_transform(X_tr)
            self._mean_vector = np.mean(X_tr_scaled, axis=0)
            cov = np.cov(X_tr_scaled, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = cov.reshape(-1, 1)

            epsilon = 1e-4
            cov_reg = cov + np.eye(cov.shape[0]) * epsilon
            self._inv_cov_matrix = np.linalg.pinv(cov_reg)

            # Evaluate distance on held-out calibration samples to set threshold
            X_cal_scaled = self._scaler.transform(X_cal)
            diff = X_cal_scaled - self._mean_vector
            cal_dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            cal_dist = np.sqrt(np.maximum(0.0, cal_dist_sq))

            percentile_cutoff = (1.0 - self.contamination) * 100.0
            self._anomaly_threshold = float(np.percentile(cal_dist, percentile_cutoff))

        else:
            # Isolation Forest
            self._iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._iso_forest.fit(X_tr)
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
            assert self._scaler is not None and self._mean_vector is not None and self._inv_cov_matrix is not None
            X_scaled = self._scaler.transform(X_arr)
            diff = X_scaled - self._mean_vector
            dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            dists = np.sqrt(np.maximum(0.0, dist_sq))
            is_ood = dists > self._anomaly_threshold
            ood_count = int(np.sum(is_ood))

        else:
            detector_name = "IsolationForest"
            assert self._iso_forest is not None
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

    def score_samples(self, X: Any) -> np.ndarray:
        """Computes continuous anomaly/OOD scores (higher = more anomalous)."""
        if not self.fitted:
            raise RuntimeError("OODDetector must be fit() on training data before scoring.")
        X_arr = self._to_numpy(X)
        if "energy" in self.method:
            return compute_free_energy(X_arr, temperature=self.temperature)
        elif "mahalanobis" in self.method:
            assert self._scaler is not None and self._mean_vector is not None and self._inv_cov_matrix is not None
            X_scaled = self._scaler.transform(X_arr)
            diff = X_scaled - self._mean_vector
            dist_sq = np.sum(np.dot(diff, self._inv_cov_matrix) * diff, axis=1)
            return np.sqrt(np.maximum(0.0, dist_sq))
        else:
            assert self._iso_forest is not None
            return -self._iso_forest.decision_function(X_arr)

    def score(self, X: Any) -> np.ndarray:
        """Alias for score_samples."""
        return self.score_samples(X)
