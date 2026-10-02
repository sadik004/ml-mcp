"""MIT Confident Learning for label error and noise detection in tabular datasets."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from ml_mcp.schemas.audit import LabelErrorReportDTO, LabelErrorSampleDTO


class LabelErrorDetector:
    """Estimates label uncertainty and detects corrupt training labels using Confident Learning."""

    def __init__(self, cv_splits: int = 5, random_state: int = 42) -> None:
        self.cv_splits = cv_splits
        self.random_state = random_state

    def detect_label_errors(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> LabelErrorReportDTO:
        """Executes Out-Of-Fold Stratified CV to find corrupted ground truth labels.

        Args:
            df: Input dataset pandas DataFrame.
            target_column: Name of the categorical/class label column.

        Returns:
            LabelErrorReportDTO detailing detected noisy labels, error rates, and confidence.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        clean_df = df.dropna(subset=[target_column]).copy()
        n_samples = len(clean_df)
        if n_samples < 10:
            raise ValueError("Dataset requires at least 10 valid labeled samples for Confident Learning.")

        y_raw = clean_df[target_column].values
        label_enc = LabelEncoder()
        y_encoded = label_enc.fit_transform(y_raw)
        classes = label_enc.classes_
        k_classes = len(classes)

        if k_classes < 2:
            raise ValueError(f"Label error detection requires >= 2 unique classes, found {k_classes}.")

        # Feature matrix preparation
        feature_df = clean_df.drop(columns=[target_column])
        X_proc = pd.DataFrame(index=clean_df.index)

        # Process numeric and categorical columns
        for col in feature_df.columns:
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                X_proc[col] = feature_df[col].fillna(feature_df[col].median() if not feature_df[col].isna().all() else 0.0)
            else:
                ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                col_vals = feature_df[[col]].astype(str)
                X_proc[col] = ord_enc.fit_transform(col_vals)

        # Dynamic fold adaptation for rare classes
        min_class_count = int(pd.Series(y_encoded).value_counts().min())
        n_splits = max(2, min(self.cv_splits, min_class_count))

        # 1. Stratified K-Fold Out-Of-Fold probability generation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros((n_samples, k_classes), dtype=np.float64)

        for train_idx, val_idx in skf.split(X_proc, y_encoded):
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                random_state=self.random_state,
                min_samples_leaf=max(2, min(20, len(train_idx) // 10)),
            )
            X_train, y_train = X_proc.iloc[train_idx], y_encoded[train_idx]
            X_val = X_proc.iloc[val_idx]

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)

            # Map predicted probabilities to full class set if a fold has missing classes
            if probs.shape[1] == k_classes:
                oof_probs[val_idx] = probs
            else:
                for idx_in_fold, c in enumerate(clf.classes_):
                    oof_probs[val_idx, c] = probs[:, idx_in_fold]

        # 2. Compute class-specific self-confidence thresholds:
        # t_j = 1/|X_j| * sum_{x in X_j} P(y=j | x)
        class_thresholds: Dict[str, float] = {}
        t = np.zeros(k_classes, dtype=np.float64)
        for j in range(k_classes):
            mask_j = (y_encoded == j)
            if np.any(mask_j):
                t[j] = float(np.mean(oof_probs[mask_j, j]))
            else:
                t[j] = 1.0 / k_classes
            # Bound threshold away from 0 and 1
            t[j] = float(np.clip(t[j], 1e-4, 0.9999))
            class_thresholds[str(classes[j])] = round(t[j], 4)

        # 3. Detect Label Errors (Confident Joint filtering):
        # Sample with given label i is corrupted if exists j != i such that:
        # P_hat_{i, j} >= t_j AND P_hat_{i, j} > P_hat_{i, i}
        flagged_samples: List[LabelErrorSampleDTO] = []
        original_indices = clean_df.index.tolist()

        for idx in range(n_samples):
            given_c = y_encoded[idx]
            given_prob = oof_probs[idx, given_c]

            # Evaluate alternative classes
            best_alt_class = None
            best_alt_prob = -1.0

            for j in range(k_classes):
                if j == given_c:
                    continue
                p_j = oof_probs[idx, j]
                if p_j >= t[j] and p_j > given_prob:
                    if p_j > best_alt_prob:
                        best_alt_prob = p_j
                        best_alt_class = j

            if best_alt_class is not None:
                orig_given_label = classes[given_c]
                orig_sugg_label = classes[best_alt_class]
                # Convert numpy types to python natives
                if hasattr(orig_given_label, "item"):
                    orig_given_label = orig_given_label.item()
                if hasattr(orig_sugg_label, "item"):
                    orig_sugg_label = orig_sugg_label.item()

                flagged_samples.append(
                    LabelErrorSampleDTO(
                        sample_index=int(original_indices[idx]),
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
            class_thresholds=class_thresholds,
            flagged_samples=flagged_samples,
        )
