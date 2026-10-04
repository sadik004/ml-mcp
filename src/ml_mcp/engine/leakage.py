"""Target Leakage Detector using Chatterjee non-parametric rank correlation, Predictive Power Score (PPS), and Bias-Corrected Cramer's V."""
from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, rankdata
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_mcp.config import get_settings
from ml_mcp.schemas.audit import TargetLeakageReportDTO

logger = logging.getLogger(__name__)


class TargetLeakageDetector:
    """Detects suspicious post-event features exhibiting anomalous linear, non-linear, or categorical association.

    Theoretical foundations:
        - Non-linear correlation: Chatterjee (JASA 2021) "A New Coefficient of Correlation"
        - Categorical association: Greenacre (2021/2023) Bias-Corrected Cramer's V
        - Predictive safety net: Wetschoreck et al. (2020/2022) Predictive Power Score (PPS)
    """

    def __init__(
        self,
        threshold_correlation: float = 0.95,
        threshold_chatterjee: float = 0.85,
        threshold_cramers_v: float = 0.90,
        threshold_pps: float = 0.98,
        threshold_mi: float = 0.85,
    ) -> None:
        self.threshold_correlation = threshold_correlation
        self.threshold_chatterjee = threshold_chatterjee
        self.threshold_cramers_v = threshold_cramers_v
        self.threshold_pps = threshold_pps
        self.threshold_mi = threshold_mi

    @staticmethod
    def calculate_chatterjee_correlation(x: pd.Series, y: pd.Series) -> float:
        """Computes Chatterjee's non-parametric rank correlation xi_n(X, Y) in O(N log N) (Chatterjee, JASA 2021).

        Unlike Pearson or Spearman, xi_n detects arbitrary non-linear functional relationships (Y = f(X)).
        xi_n -> 1 iff Y is a measurable function of X; xi_n -> 0 iff X and Y are independent.
        """
        valid_mask = x.notna() & y.notna()
        x_clean = x[valid_mask].to_numpy(dtype=np.float64)
        y_clean = y[valid_mask].to_numpy(dtype=np.float64)

        n = len(x_clean)
        if n < 10:
            return 0.0

        # Sort pairs by X (stable sort preserves order on ties)
        order = np.argsort(x_clean, kind="stable")
        y_sorted = y_clean[order]

        # Ranks of Y
        r = rankdata(y_sorted, method="max")
        l_rank = n - rankdata(y_sorted, method="min") + 1
        denom = 2.0 * float(np.sum(l_rank * (n - l_rank)))
        if denom <= 1e-12:
            denom = (n * (n**2 - 1)) / 3.0

        diff_sum = float(np.sum(np.abs(np.diff(r))))
        xi = 1.0 - (n * diff_sum) / denom
        return round(float(np.clip(xi, 0.0, 1.0)), 4)

    @staticmethod
    def calculate_bias_corrected_cramers_v(x: pd.Series, y: pd.Series) -> Optional[float]:
        """Computes Bias-Corrected Cramer's V for categorical association (Greenacre 2021/2023).

        Formula:
            phi_tilde_sq = max(0, phi_sq - ((r - 1) * (k - 1)) / (n - 1))
            r_tilde = r - (r - 1)^2 / (n - 1)
            k_tilde = k - (k - 1)^2 / (n - 1)
            V_tilde = sqrt(phi_tilde_sq / min(r_tilde - 1, k_tilde - 1))
        """
        valid_mask = x.notna() & y.notna()
        x_clean = x[valid_mask].astype(str)
        y_clean = y[valid_mask].astype(str)

        n = len(x_clean)
        if n < 10:
            return 0.0

        contingency = pd.crosstab(x_clean, y_clean)
        r, k = contingency.shape
        if r < 2 or k < 2:
            return 0.0

        try:
            chi2, _, _, _ = chi2_contingency(contingency, correction=False)
        except Exception as e:
            logger.debug(f"chi2_contingency calculation failed: {e}")
            return None

        phi2 = chi2 / n
        phi2_corr = max(0.0, phi2 - ((r - 1) * (k - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(r_corr - 1, k_corr - 1)

        if denom <= 1e-12:
            return 0.0

        v_tilde = float(np.sqrt(phi2_corr / denom))
        return round(float(np.clip(v_tilde, 0.0, 1.0)), 4)

    @staticmethod
    def calculate_single_feature_pps(
        x: pd.Series, y: pd.Series, is_classification: bool = True
    ) -> Optional[float]:
        """Evaluates single-feature Predictive Power Score (PPS) via a 1-split Decision Tree (Wetschoreck et al. 2020)."""
        valid_mask = x.notna() & y.notna()
        if valid_mask.sum() < 20:
            return 0.0

        x_sub = x[valid_mask]
        y_sub = y[valid_mask]

        if pd.api.types.is_numeric_dtype(x_sub):
            X_mat = x_sub.to_numpy(dtype=np.float64).reshape(-1, 1)
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_mat = enc.fit_transform(x_sub.to_frame())

        if len(X_mat) > 1000:
            rng = np.random.RandomState(get_settings().random_state)
            idx = rng.choice(len(X_mat), 1000, replace=False)
            X_mat = X_mat[idx]
            y_sub = y_sub.iloc[idx]

        try:
            if is_classification:
                y_cat = pd.Categorical(y_sub).codes
                clf = DecisionTreeClassifier(max_depth=1, random_state=get_settings().random_state)
                scores = cross_val_score(clf, X_mat, y_cat, cv=3, scoring="accuracy")
                model_score = float(np.mean(scores))
                # Wetschoreck et al. PPS: normalized against majority class baseline
                _, counts = np.unique(y_cat, return_counts=True)
                baseline_acc = float(np.max(counts) / len(y_cat)) if len(y_cat) > 0 else 0.5
                if baseline_acc >= 1.0 - 1e-6:
                    return 0.0
                norm_pps = (model_score - baseline_acc) / (1.0 - baseline_acc)
                return round(float(np.clip(norm_pps, 0.0, 1.0)), 4)
            else:
                y_num = y_sub.to_numpy(dtype=np.float64)
                reg = DecisionTreeRegressor(max_depth=1, random_state=get_settings().random_state)
                scores = cross_val_score(reg, X_mat, y_num, cv=3, scoring="r2")
                return round(float(np.clip(np.mean(scores), 0.0, 1.0)), 4)
        except Exception as e:
            logger.debug(f"calculate_single_feature_pps failed: {e}")
            return None

    def detect_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
    ) -> TargetLeakageReportDTO:
        """Analyzes all predictor columns against the target to detect linear, non-linear, and categorical leakage."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Out-of-core reservoir sampling guard for massive datasets (>50k rows) to prevent RAM explosion
        if len(df) > 50000:
            df = df.sample(50000, random_state=get_settings().random_state)

        y = df[target_column]
        feature_df = df.drop(columns=[target_column])

        correlations: Dict[str, Optional[float]] = {}
        chatterjee_scores: Dict[str, Optional[float]] = {}
        cramers_v_scores: Dict[str, Optional[float]] = {}
        pps_scores: Dict[str, Optional[float]] = {}
        mi_scores: Dict[str, Optional[float]] = {}
        leaked_features: List[str] = []
        warnings: List[str] = []

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

        is_discrete_target = (
            task_type == "classification"
            or not pd.api.types.is_numeric_dtype(y)
            or y.nunique() <= 20
        )

        # 1. Numeric Feature Analysis: Pearson and Chatterjee xi
        for col in numeric_cols:
            series = feature_df[col]
            valid_mask = series.notna() & y.notna()
            if valid_mask.sum() > 5:
                # Pearson Linear Correlation
                if pd.api.types.is_numeric_dtype(y):
                    try:
                        r = float(np.abs(np.corrcoef(series[valid_mask], y[valid_mask])[0, 1]))
                        if not np.isnan(r):
                            correlations[col] = round(r, 4)
                            if r >= self.threshold_correlation and col not in leaked_features:
                                leaked_features.append(col)
                    except Exception as corr_err:
                        logger.warning(f"Correlation calculation failed for '{col}': {corr_err}")
                        warnings.append(f"Correlation calculation failed for '{col}': {corr_err}")
                        correlations[col] = None

                # Chatterjee Rank Correlation (Non-linear Leakage)
                if pd.api.types.is_numeric_dtype(y):
                    try:
                        xi_val = self.calculate_chatterjee_correlation(series, y)
                        chatterjee_scores[col] = xi_val
                        if xi_val >= self.threshold_chatterjee and col not in leaked_features:
                            leaked_features.append(col)
                    except Exception as chat_err:
                        logger.warning(f"Chatterjee calculation failed for '{col}': {chat_err}")
                        warnings.append(f"Chatterjee calculation failed for '{col}': {chat_err}")
                        chatterjee_scores[col] = None

        # 2. Categorical Feature Analysis: Bias-Corrected Cramer's V
        for col in categorical_cols:
            if is_discrete_target:
                v_score = self.calculate_bias_corrected_cramers_v(feature_df[col], y)
                if v_score is None:
                    warnings.append(f"Failed to calculate Cramer's V for column '{col}'.")
                    cramers_v_scores[col] = None
                else:
                    cramers_v_scores[col] = v_score
                    if v_score >= self.threshold_cramers_v and col not in leaked_features:
                        leaked_features.append(col)
            else:
                # Continuous target vs categorical feature: evaluate ordinal codes
                valid_mask = feature_df[col].notna() & y.notna()
                if valid_mask.sum() > 5:
                    try:
                        codes = pd.Series(pd.Categorical(feature_df.loc[valid_mask, col]).codes, index=feature_df.index[valid_mask])
                        xi_val = self.calculate_chatterjee_correlation(codes, y)
                        chatterjee_scores[col] = xi_val
                        if xi_val >= self.threshold_chatterjee and col not in leaked_features:
                            leaked_features.append(col)
                    except Exception as cat_err:
                        logger.warning(f"Chatterjee categorical calculation failed for '{col}': {cat_err}")
                        warnings.append(f"Chatterjee categorical calculation failed for '{col}': {cat_err}")
                        chatterjee_scores[col] = None

        # 3. PPS Tree Safety Net on High-Signal or Borderline Predictors
        all_cols = numeric_cols + categorical_cols
        for col in all_cols:
            corr_val = correlations.get(col) or 0.0
            chat_val = chatterjee_scores.get(col) or 0.0
            cv_val = cramers_v_scores.get(col) or 0.0
            is_suspicious = (
                corr_val >= 0.80
                or chat_val >= 0.70
                or cv_val >= 0.80
            )
            if is_suspicious or len(all_cols) <= 10:
                pps = self.calculate_single_feature_pps(
                    feature_df[col], y, is_classification=is_discrete_target
                )
                if pps is None:
                    warnings.append(f"Failed to calculate PPS for column '{col}'.")
                    pps_scores[col] = None
                else:
                    pps_scores[col] = pps
                    if pps >= self.threshold_pps and col not in leaked_features:
                        leaked_features.append(col)

        # 4. Scalable Mutual Information
        if all_cols and len(df) > 10:
            eval_df = feature_df[all_cols].copy()
            y_eval = y.copy()
            if len(eval_df) > 2000:
                rng = np.random.RandomState(get_settings().random_state)
                sub_indices = rng.choice(len(eval_df), size=2000, replace=False)
                eval_df = eval_df.iloc[sub_indices].copy()
                y_eval = y_eval.iloc[sub_indices].copy()

            X_prepared = pd.DataFrame(index=eval_df.index)
            for col in numeric_cols:
                median_val = eval_df[col].median()
                X_prepared[col] = eval_df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
            for col in categorical_cols:
                col_str = eval_df[[col]].astype(str)
                ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_prepared[col] = ord_enc.fit_transform(col_str)

            if pd.api.types.is_numeric_dtype(y_eval):
                if y_eval.isna().any():
                    y_eval = y_eval.fillna(y_eval.median())
            else:
                y_eval = y_eval.astype("category").cat.codes

            try:
                if is_discrete_target:
                    raw_mi = mutual_info_classif(X_prepared, y_eval, random_state=get_settings().random_state, n_neighbors=3)
                else:
                    raw_mi = mutual_info_regression(X_prepared, y_eval, random_state=get_settings().random_state, n_neighbors=3)
                for col, score in zip(all_cols, raw_mi):
                    mi_scores[col] = round(float(score), 4)
                    if score >= self.threshold_mi and col not in leaked_features:
                        leaked_features.append(col)
            except Exception as mi_err:
                logger.warning(f"Mutual information calculation failed: {mi_err}")
                warnings.append(f"Mutual information calculation failed: {mi_err}")

        return TargetLeakageReportDTO(
            target_column=target_column,
            leaked_features=leaked_features,
            correlation_matrix=correlations,
            chatterjee_scores=chatterjee_scores,
            cramers_v_scores=cramers_v_scores,
            pps_scores=pps_scores,
            mutual_info_scores=mi_scores,
            has_critical_leakage=len(leaked_features) > 0,
            warnings=warnings,
        )
