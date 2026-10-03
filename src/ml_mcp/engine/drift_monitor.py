"""Population Stability Index (PSI) and Wasserstein-1 Earth Mover's Drift Monitor."""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from ml_mcp.schemas.serving import DataDriftReportDTO

logger = logging.getLogger(__name__)


def calculate_feature_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a single numerical feature."""
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.unique(np.percentile(ref_clean, percentiles))
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    eps = 1e-4
    ref_fractions = np.where(ref_counts == 0, eps, ref_counts / len(ref_clean))
    curr_fractions = np.where(curr_counts == 0, eps, curr_counts / len(curr_clean))

    psi_values = (ref_fractions - curr_fractions) * np.log(ref_fractions / curr_fractions)
    return float(np.sum(psi_values))


def calculate_permutation_wasserstein_p_value(
    ref_col: np.ndarray,
    curr_col: np.ndarray,
    observed_w1: float,
    n_permutations: int = 100,
    random_state: int = 42,
) -> float:
    """Computes permutation empirical p-value for Wasserstein-1 shift (Ramdas et al. JMLR 2017)."""
    rng = np.random.RandomState(random_state)
    n_ref = len(ref_col)
    combined = np.concatenate([ref_col, curr_col])
    
    null_count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_ref = shuffled[:n_ref]
        perm_curr = shuffled[n_ref:]
        perm_w1 = wasserstein_distance(perm_ref, perm_curr)
        if perm_w1 >= observed_w1:
            null_count += 1

    return float((null_count + 1) / (n_permutations + 1))


class DataDriftMonitor:
    """Monitors dataset distribution shifts using Wasserstein-1 Distance, PSI, and Permutation Tests."""

    def __init__(
        self,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.01,
        wasserstein_threshold: float = 0.25,
        run_permutation_tests: bool = True,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.wasserstein_threshold = wasserstein_threshold
        self.run_permutation_tests = run_permutation_tests

    def detect_drift(
        self,
        reference_data: Any,
        current_data: Any,
        feature_names: Optional[List[str]] = None,
    ) -> DataDriftReportDTO:
        """Scan feature distributions and report drift severity across Wasserstein and PSI metrics."""
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
        feature_w1s: List[float] = []
        ks_p_values: List[float] = []
        drifted_features: List[str] = []

        for i in range(n_features):
            ref_col = ref_arr[:, i]
            curr_col = curr_arr[:, i]

            psi = calculate_feature_psi(ref_col, curr_col)
            feature_psis.append(psi)

            # Exact Wasserstein-1 Earth Mover's Distance
            w1 = float(wasserstein_distance(ref_col, curr_col))
            # Standardize by reference IQR or std to make scale-invariant
            scale = np.std(ref_col)
            w1_norm = (w1 / scale) if scale > 1e-6 else w1
            feature_w1s.append(w1_norm)

            # KS test
            try:
                ks_res = ks_2samp(ref_col, curr_col)
                p_val = float(ks_res.pvalue)
            except Exception:
                p_val = 1.0
            ks_p_values.append(p_val)

            # Drift condition
            if psi > 0.10 or p_val < self.ks_alpha or w1_norm > self.wasserstein_threshold:
                drifted_features.append(names[i])

        overall_psi = float(np.mean(feature_psis)) if feature_psis else 0.0
        avg_w1 = float(np.mean(feature_w1s)) if feature_w1s else 0.0
        min_ks_p = float(np.min(ks_p_values)) if ks_p_values else 1.0

        # Optional Permutation p-value for highest drifted feature
        w1_pval = 1.0
        if self.run_permutation_tests and feature_w1s:
            max_idx = int(np.argmax(feature_w1s))
            w1_raw = float(wasserstein_distance(ref_arr[:, max_idx], curr_arr[:, max_idx]))
            w1_pval = calculate_permutation_wasserstein_p_value(
                ref_arr[:, max_idx], curr_arr[:, max_idx], w1_raw, n_permutations=50
            )

        if overall_psi < 0.10 and avg_w1 < self.wasserstein_threshold:
            drift_status = "no_drift"
            retraining_recommended = False
        elif overall_psi <= self.psi_threshold or avg_w1 <= (2.0 * self.wasserstein_threshold):
            drift_status = "moderate_drift"
            retraining_recommended = False
        else:
            drift_status = "severe_drift"
            retraining_recommended = True

        if min_ks_p < 1e-4 and len(drifted_features) > n_features / 2:
            retraining_recommended = True

        return DataDriftReportDTO(
            psi_score=round(overall_psi, 4),
            drift_status=drift_status,
            ks_p_value=round(min_ks_p, 4),
            retraining_recommended=retraining_recommended,
            drifted_features=drifted_features,
            wasserstein_distance=round(avg_w1, 4),
            wasserstein_p_value=round(w1_pval, 4),
        )
