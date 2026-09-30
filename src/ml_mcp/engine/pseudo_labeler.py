"""Semi-Supervised High-Confidence Pseudo-Labeling Engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class PseudoLabeler:
    """Harvests >=98% high-confidence unlabelled predictions as pseudo-labels for model refinement."""

    def __init__(self, confidence_threshold: float = 0.98, random_state: int = 42) -> None:
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

    def refine_with_pseudo_labels(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_unlabelled: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """Harvest high-confidence unlabelled predictions and retrain model."""
        X_tr = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        y_tr = np.asarray(y_train)
        X_unlab = X_unlabelled.to_numpy() if isinstance(X_unlabelled, pd.DataFrame) else np.asarray(X_unlabelled)

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must support predict_proba to compute confidence thresholds.")

        # Compute baseline validation score
        baseline_score = 0.0
        if X_val is not None and y_val is not None:
            val_preds = model.predict(X_val)
            baseline_score = float(accuracy_score(y_val, val_preds))

        # Predict probabilities on unlabelled test data
        probas = model.predict_proba(X_unlab)
        max_conf = np.max(probas, axis=1)
        pseudo_labels = model.predict(X_unlab)

        # Harvest mask for confidence >= threshold
        harvest_mask = max_conf >= self.confidence_threshold
        harvested_count = int(np.sum(harvest_mask))
        total_unlabelled = len(X_unlab)
        harvested_ratio = float(harvested_count / total_unlabelled) if total_unlabelled > 0 else 0.0

        refined_model = clone(model)

        if harvested_count > 0:
            X_harvested = X_unlab[harvest_mask]
            y_harvested = pseudo_labels[harvest_mask]

            X_combined = np.vstack([X_tr, X_harvested])
            y_combined = np.concatenate([y_tr, y_harvested])

            refined_model.fit(X_combined, y_combined)
        else:
            # No samples reached confidence threshold; retain original trained model
            refined_model = model

        # Compute refined validation score
        refined_score = baseline_score
        if X_val is not None and y_val is not None:
            ref_preds = refined_model.predict(X_val)
            refined_score = float(accuracy_score(y_val, ref_preds))

        score_lift = refined_score - baseline_score

        stats = {
            "harvested_count": harvested_count,
            "harvested_ratio": round(harvested_ratio, 4),
            "confidence_threshold": self.confidence_threshold,
            "baseline_score": round(baseline_score, 4),
            "refined_score": round(refined_score, 4),
            "score_lift": round(score_lift, 4),
        }

        return stats, refined_model
