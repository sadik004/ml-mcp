"""MIT Confident Learning and Out-Of-Fold Residual Dispersion for tabular label noise detection."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from ml_mcp.schemas.audit import LabelErrorReportDTO, LabelErrorSampleDTO


class LabelErrorDetector:
    """Estimates label uncertainty and detects corrupt training labels via Confident Learning and Residual Dispersion."""

    def __init__(self, cv_splits: int = 5, random_state: Optional[int] = None) -> None:
        from ml_mcp.config import get_settings
        self.cv_splits = cv_splits
        self.random_state = random_state if random_state is not None else get_settings().random_state

    def detect_label_errors(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["auto", "classification", "regression"] = "auto",
    ) -> LabelErrorReportDTO:
        """Executes Out-Of-Fold CV to find corrupted ground truth labels across classification or regression.

        Theoretical foundations:
            - Classification: Northcutt, Jiang, & Chuang (JAIR 2021) MIT Confident Learning
            - Regression: Papanikolaou et al. (NeurIPS 2023) Normalized Residual Dispersion

        Args:
            df: Input dataset pandas DataFrame.
            target_column: Name of the target column.
            task_type: "auto", "classification", or "regression".

        Returns:
            LabelErrorReportDTO detailing detected noisy labels, error rates, and confidence.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        clean_df = df.dropna(subset=[target_column]).copy()
        n_samples = len(clean_df)
        if n_samples < 10:
            raise ValueError("Dataset requires at least 10 valid labeled samples for Label Error Detection.")

        y_raw = clean_df[target_column]

        # Determine task type if auto
        if task_type == "auto":
            if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 15:
                resolved_task = "regression"
            else:
                resolved_task = "classification"
        else:
            resolved_task = task_type

        # Feature matrix preparation
        feature_df = clean_df.drop(columns=[target_column])
        X_proc = pd.DataFrame(index=clean_df.index)

        for col in feature_df.columns:
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                X_proc[col] = feature_df[col].fillna(
                    feature_df[col].median() if not feature_df[col].isna().all() else 0.0
                )
            else:
                ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                col_vals = feature_df[[col]].astype(str)
                X_proc[col] = ord_enc.fit_transform(col_vals)

        # -------------------------------------------------------------------------
        # Branch 1: Continuous Regression (Papanikolaou et al. NeurIPS 2023)
        # -------------------------------------------------------------------------
        if resolved_task == "regression":
            y_vals = pd.to_numeric(y_raw, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
            oof_preds = np.zeros(n_samples, dtype=np.float64)

            for train_idx, val_idx in kf.split(X_proc):
                reg = HistGradientBoostingRegressor(
                    max_iter=60,
                    random_state=self.random_state,
                    min_samples_leaf=max(2, min(20, len(train_idx) // 10)),
                )
                reg.fit(X_proc.iloc[train_idx], y_vals[train_idx])
                oof_preds[val_idx] = reg.predict(X_proc.iloc[val_idx])

            residuals = y_vals - oof_preds
            abs_residuals = np.abs(residuals)

            # Robust scale dispersion using IQR / 1.349
            q75, q25 = np.percentile(abs_residuals, [75, 25])
            iqr = q75 - q25
            scale = iqr / 1.349 if iqr > 1e-6 else max(float(np.std(abs_residuals)), 1e-6)

            z_scores = abs_residuals / scale

            # Flag samples with extreme residual dispersion (z > 3.0)
            flagged_samples: List[LabelErrorSampleDTO] = []
            flagged_indices = np.where(z_scores > 3.0)[0]

            for idx in flagged_indices:
                orig_idx = int(clean_df.index[idx])
                flagged_samples.append(
                    LabelErrorSampleDTO(
                        sample_index=orig_idx,
                        given_label=round(float(y_vals[idx]), 4),
                        suggested_label=round(float(oof_preds[idx]), 4),
                        confidence=round(float(z_scores[idx]), 4),
                    )
                )

            total_errors = len(flagged_samples)
            error_rate = float(total_errors / n_samples)

            return LabelErrorReportDTO(
                total_samples=n_samples,
                total_errors=total_errors,
                error_rate=round(error_rate, 4),
                task_type="regression",
                class_thresholds={
                    "residual_dispersion_iqr": round(float(iqr), 4),
                    "dispersion_scale": round(float(scale), 4),
                },
                flagged_samples=flagged_samples,
            )

        # -------------------------------------------------------------------------
        # Branch 2: Classification via MIT Confident Learning (Northcutt et al. 2021)
        # -------------------------------------------------------------------------
        label_enc = LabelEncoder()
        y_encoded = label_enc.fit_transform(y_raw.values)
        classes = label_enc.classes_
        k_classes = len(classes)

        if k_classes < 2:
            raise ValueError(f"Label error detection requires >= 2 unique classes, found {k_classes}.")

        min_class_count = int(pd.Series(y_encoded).value_counts().min())
        n_splits = max(2, min(self.cv_splits, min_class_count))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros((n_samples, k_classes), dtype=np.float64)

        for train_idx, val_idx in skf.split(X_proc, y_encoded):
            clf = HistGradientBoostingClassifier(
                max_iter=60,
                random_state=self.random_state,
                min_samples_leaf=max(2, min(20, len(train_idx) // 10)),
            )
            clf.fit(X_proc.iloc[train_idx], y_encoded[train_idx])
            probs = clf.predict_proba(X_proc.iloc[val_idx])

            if probs.shape[1] == k_classes:
                oof_probs[val_idx] = probs
            else:
                for idx_in_fold, c in enumerate(clf.classes_):
                    oof_probs[val_idx, c] = probs[:, idx_in_fold]

        # Compute class-specific self-confidence thresholds:
        # t_j = 1/|X_j| * sum_{x in X_j} P(y=j | x)
        class_thresholds: Dict[str, float] = {}
        t = np.zeros(k_classes, dtype=np.float64)
        for j in range(k_classes):
            mask_j = (y_encoded == j)
            if np.any(mask_j):
                t[j] = float(np.mean(oof_probs[mask_j, j]))
            else:
                t[j] = float(1.0 / k_classes)
            class_thresholds[str(classes[j])] = round(float(t[j]), 4)

        # Flag samples: y_given == j, but P(y=k|x) >= t_k for k != j and P(y=k|x) > P(y=j|x)
        flagged_samples = []
        for i in range(n_samples):
            j = y_encoded[i]
            probs_i = oof_probs[i]
            class_prob = probs_i[j]

            # Best alternative candidate
            alt_candidates = [k for k in range(k_classes) if k != j]
            best_alt = max(alt_candidates, key=lambda k: probs_i[k])
            best_alt_prob = probs_i[best_alt]

            if best_alt_prob >= t[best_alt] and best_alt_prob > class_prob:
                orig_idx = int(clean_df.index[i])
                orig_given_label = classes[j]
                orig_sugg_label = classes[best_alt]
                flagged_samples.append(
                    LabelErrorSampleDTO(
                        sample_index=orig_idx,
                        given_label=orig_given_label,
                        suggested_label=orig_sugg_label,
                        confidence=round(float(best_alt_prob), 4),
                    )
                )

        total_errors = len(flagged_samples)
        error_rate = float(total_errors / n_samples) if n_samples > 0 else 0.0

        return LabelErrorReportDTO(
            total_samples=n_samples,
            total_errors=total_errors,
            error_rate=round(error_rate, 4),
            task_type="classification",
            class_thresholds=class_thresholds,
            flagged_samples=flagged_samples,
        )
