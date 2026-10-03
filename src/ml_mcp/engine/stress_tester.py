"""Model Adversarial Noise and Covariance-Aware Manifold Stress Tester Engine (Hendrycks & Dietterich ICLR)."""
from __future__ import annotations

import logging
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from ml_mcp.schemas.safety import StressTestReportDTO

logger = logging.getLogger(__name__)


class ModelStressTester:
    """Stress tests ML models using Covariance-Aware Manifold Perturbations and Outlier Injections."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def _inject_perturbation(
        self,
        X_arr: np.ndarray,
        perturbation_type: str,
        noise_level: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Inject specified perturbation into feature matrix."""
        X_pert = X_arr.copy()
        n_rows, n_cols = X_pert.shape

        if perturbation_type in ("covariance_noise", "manifold_noise"):
            # Covariance-Preserving Manifold Perturbation: delta ~ N(0, eps^2 * Cov(X))
            cov = np.cov(X_pert, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = cov.reshape(-1, 1)

            # Regularize diagonal to avoid singular matrices
            cov_reg = cov + np.eye(n_cols) * 1e-6
            try:
                L = np.linalg.cholesky(cov_reg)
                z = rng.normal(size=(n_rows, n_cols))
                correlated_noise = noise_level * np.dot(z, L.T)
                X_pert = X_pert + correlated_noise
            except np.linalg.LinAlgError:
                # SVD fallback if Cholesky fails on degenerate matrix
                u, s, _ = np.linalg.svd(cov_reg)
                transform = np.dot(u, np.diag(np.sqrt(s)))
                z = rng.normal(size=(n_rows, n_cols))
                X_pert = X_pert + noise_level * np.dot(z, transform.T)

        elif perturbation_type == "gaussian_noise":
            stds = np.std(X_pert, axis=0)
            stds[stds == 0] = 1.0
            noise = rng.normal(loc=0.0, scale=noise_level * stds, size=X_pert.shape)
            X_pert = X_pert + noise

        elif perturbation_type == "feature_swap":
            if n_cols > 1:
                col_indices = np.arange(n_cols)
                shuffled_cols = rng.permutation(col_indices)
                swap_mask = rng.rand(n_rows, 1) < noise_level
                X_pert = np.where(swap_mask, X_pert[:, shuffled_cols], X_pert)

        elif perturbation_type == "extreme_outlier":
            q25 = np.percentile(X_pert, 25, axis=0)
            q75 = np.percentile(X_pert, 75, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0

            outlier_mask = rng.rand(*X_pert.shape) < noise_level
            outlier_direction = rng.choice([-1.0, 1.0], size=X_pert.shape)
            outlier_shift = outlier_direction * (5.0 * iqr)
            X_pert = np.where(outlier_mask, X_pert + outlier_shift, X_pert)

        return X_pert

    def evaluate(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        perturbation_type: str = "covariance_noise",
        noise_level: float = 0.10,
        task_type: Optional[str] = None,
    ) -> StressTestReportDTO:
        """Evaluate model degradation under feature perturbation."""
        rng = np.random.RandomState(self.random_state)
        X_arr = X_test.to_numpy(dtype=float) if isinstance(X_test, pd.DataFrame) else np.asarray(X_test, dtype=float)
        y_arr = np.asarray(y_test)

        baseline_preds = model.predict(X_arr)
        is_classification = task_type == "classification" or len(np.unique(y_arr)) <= 10

        if is_classification:
            baseline_score = float(accuracy_score(y_arr, baseline_preds))
        else:
            baseline_score = float(r2_score(y_arr, baseline_preds))

        X_stressed = self._inject_perturbation(X_arr, perturbation_type, noise_level, rng)
        stressed_preds = model.predict(X_stressed)

        if is_classification:
            stressed_score = float(accuracy_score(y_arr, stressed_preds))
        else:
            stressed_score = float(r2_score(y_arr, stressed_preds))

        denominator = abs(baseline_score) if abs(baseline_score) > 1e-6 else 1.0
        raw_degradation = ((baseline_score - stressed_score) / denominator) * 100.0
        degradation_pct = float(max(0.0, raw_degradation))

        robustness_score = float(max(0.0, min(100.0, 100.0 - degradation_pct)))
        is_stress_passed = bool(degradation_pct <= 10.0)

        return StressTestReportDTO(
            perturbation_type=perturbation_type,
            noise_level=round(noise_level, 4),
            baseline_score=round(baseline_score, 4),
            stressed_score=round(stressed_score, 4),
            degradation_percentage=round(degradation_pct, 2),
            robustness_score=round(robustness_score, 2),
            is_stress_passed=is_stress_passed,
        )
