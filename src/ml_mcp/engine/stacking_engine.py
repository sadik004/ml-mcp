"""Leak-Free Stacking Ensemble Engine using Out-of-Fold cross-validation meta-features."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from ml_mcp.config import get_settings
from ml_mcp.engine.scoring import resolve_scorer

logger = logging.getLogger(__name__)


class StackingEngine:
    """Blends top tournament performers using leak-free out-of-fold cross-validation predictions.

    Theoretical Basis:
        - Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.
        - van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super Learner.
          Statistical Applications in Genetics and Molecular Biology, 6(1), Article 25.
    """

    def __init__(self) -> None:
        self.oof_predictions_: Dict[str, np.ndarray] = {}
        self.meta_features_: Optional[np.ndarray] = None
        self.oof_meta_proba_: Optional[np.ndarray] = None
        self.warnings_: List[str] = []

    def build_stacking_ensemble(
        self,
        base_models: List[Tuple[str, Any]],
        X: Any,
        y: Any,
        task_type: Literal["classification", "regression"] = "classification",
        cv_splits: int = 5,
        scoring: str = "roc_auc",
    ) -> Tuple[Any, Optional[float]]:
        """Constructs and trains an end-to-end stacking ensemble pipeline without nested CV explosion.

        Returns:
            Tuple of (fitted_stacking_pipeline, oof_score)
        """
        y_arr = np.asarray(y)
        self.oof_predictions_.clear()
        self.oof_meta_proba_ = None
        self.warnings_.clear()

        # Strict metric validation: unknown metric raises ValueError (Defect H9)
        scorer_fn, needs_proba = resolve_scorer(scoring)

        # 1. Resolve Super Learner Bounded Meta-Learner (van der Laan et al. 2007)
        if task_type == "classification":
            # Regularized bounded LogisticRegression to prevent meta-overfitting
            final_estimator = LogisticRegression(
                C=0.5, solver="lbfgs", max_iter=500, random_state=get_settings().random_state
            )
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=final_estimator,
                cv=cv_splits,
                passthrough=False,
                n_jobs=1,
            )
            cv_splitter = StratifiedKFold(
                n_splits=cv_splits, shuffle=True, random_state=get_settings().random_state
            )
        else:
            # Enforce Non-Negative Least Squares (positive=True) to prevent extreme multicollinearity
            final_estimator = Ridge(alpha=1.0, positive=True, random_state=get_settings().random_state)
            stacking_model = StackingRegressor(
                estimators=base_models,
                final_estimator=final_estimator,
                cv=cv_splits,
                passthrough=False,
                n_jobs=1,
            )
            cv_splitter = KFold(
                n_splits=cv_splits, shuffle=True, random_state=get_settings().random_state
            )

        # 2. Fit full stacking model on full training data (1 pass)
        stacking_model.fit(X, y)

        # 3. Generate 1-pass Out-Of-Fold (OOF) predictions for base models (Wolpert 1992 / Super Learner)
        # Avoids nested cross_val_score(stacking_model) which causes 25x redundant refits
        oof_meta_cols: List[np.ndarray] = []
        oof_score: Optional[float] = None
        try:
            for name, base_est in base_models:
                cloned_est = clone(base_est)
                if task_type == "classification":
                    if hasattr(cloned_est, "predict_proba"):
                        pred = cross_val_predict(
                            cloned_est, X, y, cv=cv_splitter, method="predict_proba", n_jobs=1
                        )
                        # Extract positive class prob for binary, or keep full probs
                        col_to_stack = pred[:, 1] if pred.ndim > 1 and pred.shape[1] == 2 else pred
                    elif hasattr(cloned_est, "decision_function"):
                        col_to_stack = cross_val_predict(
                            cloned_est, X, y, cv=cv_splitter, method="decision_function", n_jobs=1
                        )
                    else:
                        col_to_stack = cross_val_predict(cloned_est, X, y, cv=cv_splitter, n_jobs=1)
                else:
                    col_to_stack = cross_val_predict(cloned_est, X, y, cv=cv_splitter, n_jobs=1)

                self.oof_predictions_[name] = col_to_stack
                if col_to_stack.ndim == 1:
                    oof_meta_cols.append(col_to_stack)
                else:
                    for c_idx in range(col_to_stack.shape[1]):
                        oof_meta_cols.append(col_to_stack[:, c_idx])

            if oof_meta_cols:
                self.meta_features_ = np.column_stack(oof_meta_cols)
                # Compute fast OOF ensemble prediction using meta-learner
                meta_clf = clone(final_estimator)

                if task_type == "classification":
                    if hasattr(meta_clf, "predict_proba"):
                        self.oof_meta_proba_ = cross_val_predict(
                            meta_clf, self.meta_features_, y, cv=cv_splitter, method="predict_proba", n_jobs=1
                        )
                    if hasattr(meta_clf, "predict"):
                        meta_preds = cross_val_predict(
                            meta_clf, self.meta_features_, y, cv=cv_splitter, method="predict", n_jobs=1
                        )
                    else:
                        meta_preds = np.argmax(self.oof_meta_proba_, axis=1) if self.oof_meta_proba_ is not None else None
                else:
                    meta_preds = cross_val_predict(
                        meta_clf, self.meta_features_, y, cv=cv_splitter, n_jobs=1
                    )

                if needs_proba:
                    if self.oof_meta_proba_ is None:
                        raise ValueError(
                            f"Metric '{scoring}' requires probability predictions, but meta-learner does not support predict_proba."
                        )
                    oof_score = float(scorer_fn(y_arr, self.oof_meta_proba_))
                else:
                    if meta_preds is None:
                        raise ValueError(f"Meta-learner failed to generate discrete predictions for metric '{scoring}'.")
                    oof_score = float(scorer_fn(y_arr, meta_preds))
        except Exception as exc:
            msg = f"OOF prediction failed: {exc}"
            logger.warning(msg)
            self.warnings_.append(msg)
            # Defect C2 Fix: Never substitute in-sample model.score(X,y) on OOF failure.
            oof_score = None

        return stacking_model, oof_score

