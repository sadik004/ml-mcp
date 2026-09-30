"""Population Stability Index (PSI) and Kolmogorov-Smirnov Data Drift Monitor."""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ml_mcp.schemas.serving import DataDriftReportDTO

logger = logging.getLogger(__name__)


def calculate_feature_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a single numerical feature."""
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference data
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    # Ensure strictly increasing edges to avoid duplicate bin boundaries
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Expand outer bounds slightly
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Fractions with epsilon smoothing to guard against division by zero
    eps = 1e-4
    ref_fractions = ref_counts / len(ref_clean)
    curr_fractions = curr_counts / len(curr_clean)

    ref_fractions = np.where(ref_fractions == 0, eps, ref_fractions)
    curr_fractions = np.where(curr_fractions == 0, eps, curr_fractions)

    psi_values = (ref_fractions - curr_fractions) * np.log(ref_fractions / curr_fractions)
    return float(np.sum(psi_values))


class DataDriftMonitor:
    """Monitors dataset distribution shifts using PSI and KS-tests for automated MLOps retraining."""

    def __init__(self, psi_threshold: float = 0.25, ks_alpha: float = 0.01) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def detect_drift(
        self,
        reference_data: Any,
        current_data: Any,
        feature_names: Optional[List[str]] = None,
    ) -> DataDriftReportDTO:
        """Scan feature distributions and report drift severity."""
        ref_arr = (
            reference_data.to_numpy(dtype=float)
            if isinstance(reference_data, pd.DataFrame)
            else np.asarray(reference_data, dtype=float)
        )
        curr_arr = (
            current_data.to_numpy(dtype=float)
            if isinstance(current_data, pd.DataFrame)
            else np.asarray(current_data, dtype=float)
        )

        n_features = ref_arr.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        feature_psis: List[float] = []
        ks_p_values: List[float] = []
        drifted_features: List[str] = []

        for i in range(n_features):
            ref_col = ref_arr[:, i]
            curr_col = curr_arr[:, i]

            psi = calculate_feature_psi(ref_col, curr_col)
            feature_psis.append(psi)

            # Two-sample Kolmogorov-Smirnov test
            try:
                ks_res = ks_2samp(ref_col, curr_col)
                p_val = float(ks_res.pvalue)
            except Exception:
                p_val = 1.0
            ks_p_values.append(p_val)

            if psi > 0.10 or p_val < self.ks_alpha:
                drifted_features.append(names[i])

        overall_psi = float(np.mean(feature_psis)) if feature_psis else 0.0
        min_ks_p = float(np.min(ks_p_values)) if ks_p_values else 1.0

        if overall_psi < 0.10:
            drift_status = "no_drift"
            retraining_recommended = False
        elif overall_psi <= self.psi_threshold:
            drift_status = "moderate_drift"
            retraining_recommended = False
        else:
            drift_status = "severe_drift"
            retraining_recommended = True

        # If individual features underwent extreme shift
        if min_ks_p < 1e-4 and len(drifted_features) > n_features / 2:
            retraining_recommended = True

        return DataDriftReportDTO(
            psi_score=round(overall_psi, 4),
            drift_status=drift_status,
            ks_p_value=round(min_ks_p, 4),
            retraining_recommended=retraining_recommended,
            drifted_features=drifted_features,
        )
