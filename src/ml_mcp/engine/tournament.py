"""8-Model Competitive Arena with Cross-Validation and Diversity-Guarded Stacking."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from ml_mcp.engine.gpu_manager import GPUManager
from ml_mcp.engine.stacking_engine import StackingEngine
from ml_mcp.schemas.tournament import ModelEvaluationDTO, TournamentLeaderboardDTO


class TournamentArena:
    """Orchestrates an 8-model competitive tournament with automated scoring alignment and diversity guard."""

    def __init__(
        self,
        cv_splits: int = 5,
        fast_mode: bool = False,
        force_cpu: bool = False,
        gpu_manager: Optional[GPUManager] = None,
    ) -> None:
        self.cv_splits = cv_splits
        self.fast_mode = fast_mode
        self.gpu_manager = gpu_manager or GPUManager(force_cpu=force_cpu)
        self.stacking_engine = StackingEngine()
        self.champion_estimator: Optional[Any] = None

    def _get_roster(self, task_type: str) -> List[Tuple[str, Any]]:
        """Constructs the competitive roster prioritizing GPU acceleration when available."""
        models: List[Tuple[str, Any]] = []
        n_trees = 30 if self.fast_mode else 100

        if task_type == "classification":
            models.append(("LogisticRegression", LogisticRegression(max_iter=500, random_state=42)))
            models.append(("RidgeClassifier", RidgeClassifier(random_state=42)))
            models.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)))
            models.append(("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)))
            models.append(("HistGradientBoostingClassifier", HistGradientBoostingClassifier(random_state=42)))

            # LightGBM
            lgb_p = self.gpu_manager.get_lgb_params()
            try:
                from lightgbm import LGBMClassifier
                models.append(("LGBMClassifier", LGBMClassifier(n_estimators=n_trees, random_state=42, verbose=-1, **lgb_p)))
            except Exception:
                pass

            # XGBoost
            xgb_p = self.gpu_manager.get_xgb_params()
            try:
                from xgboost import XGBClassifier
                models.append(("XGBClassifier", XGBClassifier(n_estimators=n_trees, random_state=42, eval_metric="logloss", **xgb_p)))
            except Exception:
                pass

            # CatBoost
            cb_p = self.gpu_manager.get_catboost_params()
            try:
                from catboost import CatBoostClassifier
                models.append(("CatBoostClassifier", CatBoostClassifier(iterations=n_trees, random_seed=42, verbose=0, **cb_p)))
            except Exception:
                pass

        else:
            # Regression Roster
            models.append(("Ridge", Ridge(random_state=42)))
            models.append(("RandomForestRegressor", RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)))
            models.append(("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)))
            models.append(("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=42)))

            lgb_p = self.gpu_manager.get_lgb_params()
            try:
                from lightgbm import LGBMRegressor
                models.append(("LGBMRegressor", LGBMRegressor(n_estimators=n_trees, random_state=42, verbose=-1, **lgb_p)))
            except Exception:
                pass

            xgb_p = self.gpu_manager.get_xgb_params()
            try:
                from xgboost import XGBRegressor
                models.append(("XGBRegressor", XGBRegressor(n_estimators=n_trees, random_state=42, **xgb_p)))
            except Exception:
                pass

            cb_p = self.gpu_manager.get_catboost_params()
            try:
                from catboost import CatBoostRegressor
                models.append(("CatBoostRegressor", CatBoostRegressor(iterations=n_trees, random_seed=42, verbose=0, **cb_p)))
            except Exception:
                pass

        return models

    def _extract_feature_importances(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extracts and formats raw feature importances."""
        importances: Dict[str, float] = {}
        try:
            if hasattr(model, "feature_importances_"):
                raw = model.feature_importances_
                for name, imp in zip(feature_names, raw):
                    importances[name] = float(imp)
            elif hasattr(model, "coef_"):
                coef = model.coef_
                raw = np.abs(coef).ravel() if hasattr(coef, "ravel") else np.abs(coef)
                for name, imp in zip(feature_names, raw[:len(feature_names)]):
                    importances[name] = float(imp)
        except Exception:
            pass
        return importances

    def _select_diverse_candidates(
        self,
        sorted_evaluations: List[ModelEvaluationDTO],
        oof_dict: Dict[str, np.ndarray],
        max_candidates: int = 3,
        correlation_threshold: float = 0.95,
    ) -> List[str]:
        """Selects diverse stacking candidates based on pairwise Pearson correlation.
        
        Theoretical Basis:
            - Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles.
            - Caruana, R. et al. (ICML 2004). Ensemble selection from libraries of models.
        """
        if not sorted_evaluations:
            return []

        sorted_names = [m.model_name for m in sorted_evaluations]
        selected: List[str] = [sorted_names[0]]

        for candidate in sorted_names[1:]:
            cand_oof = oof_dict.get(candidate)
            if cand_oof is None:
                continue

            is_redundant = False
            for chosen in selected:
                chosen_oof = oof_dict.get(chosen)
                if chosen_oof is not None:
                    # Guard against zero-variance constants
                    if np.std(cand_oof) > 1e-9 and np.std(chosen_oof) > 1e-9:
                        corr = np.corrcoef(cand_oof, chosen_oof)[0, 1]
                        if not np.isnan(corr) and abs(corr) >= correlation_threshold:
                            is_redundant = True
                            break

            if not is_redundant:
                selected.append(candidate)

            if len(selected) >= max_candidates:
                break

        # Fallback to top candidates if diversity filter is too strict
        if len(selected) < 2 and len(sorted_names) >= 2:
            return sorted_names[:max_candidates]

        return selected

    def run_tournament(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        group_column: Optional[str] = None,
        scoring: Optional[str] = None,
    ) -> TournamentLeaderboardDTO:
        """Executes the competitive cross-validation tournament across all candidate models."""
        # 1. Resolve target metric alignment
        default_metric = "roc_auc" if task_type == "classification" else "r2"
        metric_str = scoring or default_metric
        try:
            scorer = get_scorer(metric_str)
        except Exception:
            metric_str = default_metric
            scorer = get_scorer(metric_str)

        X = df.drop(columns=[target_column])
        if group_column and group_column in X.columns:
            groups = X[group_column]
            X = X.drop(columns=[group_column])
        else:
            groups = None

        y = df[target_column]

        # Defensive auto-preprocessing if non-numeric/null features exist
        has_non_numeric = any(
            X[col].dtype == "object" or isinstance(X[col].dtype, pd.StringDtype) or X[col].isnull().any()
            for col in X.columns
        )
        if has_non_numeric:
            from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
            builder = DefensivePipelineBuilder()
            pipe = builder.build_pipeline(df, target_column=target_column)
            X_trans = pipe.fit_transform(X, y)
            feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]
            X = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
        else:
            feature_names = [str(c) for c in X.columns]

        # Configure cross-validation splitter
        if group_column and groups is not None:
            cv = StratifiedGroupKFold(n_splits=self.cv_splits)
            split_gen = cv.split(X, y, groups)
        elif task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
            split_gen = cv.split(X, y)
        else:
            cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
            split_gen = cv.split(X, y)

        splits_list = list(split_gen)
        roster = self._get_roster(task_type)
        evaluations: List[ModelEvaluationDTO] = []
        fitted_models: List[Tuple[str, Any]] = []
        oof_predictions: Dict[str, np.ndarray] = {}

        n_samples = len(X)

        for name, base_model in roster:
            fold_val_scores: List[float] = []
            fold_train_scores: List[float] = []
            oof_vector = np.zeros(n_samples, dtype=float)
            start_time = time.perf_counter()

            # Execute cross-validation folds using clone to prevent state leakage
            for train_idx, val_idx in splits_list:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                fold_model = clone(base_model)
                try:
                    fold_model.fit(X_train, y_train)
                except Exception:
                    # Fallback to CPU if GPU driver failure
                    cpu_params = self.gpu_manager.get_fallback_cpu_params(name)
                    if hasattr(fold_model, "set_params"):
                        try:
                            fold_model.set_params(**cpu_params)
                            fold_model.fit(X_train, y_train)
                        except Exception:
                            continue

                # Evaluate using official Scikit-learn Scorer (eliminates model.score trap)
                try:
                    train_score = float(scorer(fold_model, X_train, y_train))
                    val_score = float(scorer(fold_model, X_val, y_val))
                except Exception:
                    # Fallback to raw accuracy/r2 if scorer fails on edge-case fold
                    train_score = float(fold_model.score(X_train, y_train))
                    val_score = float(fold_model.score(X_val, y_val))

                fold_train_scores.append(train_score)
                fold_val_scores.append(val_score)

                # Collect OOF predictions for Diversity Guard
                if task_type == "classification":
                    if hasattr(fold_model, "predict_proba"):
                        probs = fold_model.predict_proba(X_val)
                        oof_vector[val_idx] = probs[:, 1] if probs.ndim > 1 and probs.shape[1] == 2 else np.argmax(probs, axis=1)
                    elif hasattr(fold_model, "decision_function"):
                        oof_vector[val_idx] = fold_model.decision_function(X_val)
                    else:
                        oof_vector[val_idx] = fold_model.predict(X_val)
                else:
                    oof_vector[val_idx] = fold_model.predict(X_val)

            if not fold_val_scores:
                continue

            # Fit final model on complete dataset
            final_fitted = clone(base_model)
            try:
                final_fitted.fit(X, y)
            except Exception:
                final_fitted = fold_model

            fit_time = time.perf_counter() - start_time
            oof_predictions[name] = oof_vector

            # Compute inference latency (ms per sample)
            lat_start = time.perf_counter()
            test_slice = X.head(min(50, len(X)))
            _ = final_fitted.predict(test_slice)
            latency_ms = (time.perf_counter() - lat_start) * 1000.0 / len(test_slice)

            mean_val = float(np.mean(fold_val_scores))
            std_val = float(np.std(fold_val_scores))
            mean_train = float(np.mean(fold_train_scores))
            overfit_gap = float(abs(mean_train - mean_val))

            feat_importances = self._extract_feature_importances(final_fitted, feature_names)

            eval_dto = ModelEvaluationDTO(
                model_name=name,
                metric_name=metric_str,
                mean_cv_score=round(mean_val, 4),
                std_cv_score=round(std_val, 4),
                fit_time_seconds=round(fit_time, 2),
                inference_latency_ms=round(latency_ms, 3),
                overfit_gap=round(overfit_gap, 4),
                feature_importances=feat_importances,
            )
            evaluations.append(eval_dto)
            fitted_models.append((name, final_fitted))

        # Sort leaderboard descending by validation score
        evaluations.sort(key=lambda m: m.mean_cv_score, reverse=True)

        # Select Stacking candidates using Pairwise OOF Prediction Diversity Guard (Kuncheva & Whitaker 2003)
        top_candidates = self._select_diverse_candidates(
            sorted_evaluations=evaluations,
            oof_dict=oof_predictions,
            max_candidates=3,
            correlation_threshold=0.95,
        )

        stacking_base = [(name, model) for name, model in fitted_models if name in top_candidates]

        if len(stacking_base) >= 2:
            try:
                stack_start = time.perf_counter()
                stacking_model, stacking_score = self.stacking_engine.build_stacking_ensemble(
                    base_models=stacking_base,
                    X=X,
                    y=y,
                    task_type=task_type,
                    cv_splits=self.cv_splits,
                    scoring=metric_str,
                )
                stack_fit_time = time.perf_counter() - stack_start

                # Measure actual latency for Stacking Ensemble
                lat_start = time.perf_counter()
                test_slice = X.head(min(50, len(X)))
                _ = stacking_model.predict(test_slice)
                stack_latency_ms = (time.perf_counter() - lat_start) * 1000.0 / len(test_slice)

                # Measure actual overfit gap
                try:
                    stack_train_score = float(scorer(stacking_model, X, y))
                    stack_overfit_gap = float(abs(stack_train_score - stacking_score))
                except Exception:
                    stack_overfit_gap = 0.0

                stacking_eval = ModelEvaluationDTO(
                    model_name="StackingEnsemble",
                    metric_name=metric_str,
                    mean_cv_score=round(stacking_score, 4),
                    std_cv_score=0.0,
                    fit_time_seconds=round(stack_fit_time, 2),
                    inference_latency_ms=round(stack_latency_ms, 3),
                    overfit_gap=round(stack_overfit_gap, 4),
                    feature_importances={},
                )
                evaluations.append(stacking_eval)
                evaluations.sort(key=lambda m: m.mean_cv_score, reverse=True)
                fitted_models.append(("StackingEnsemble", stacking_model))
            except Exception:
                pass

        champion = evaluations[0] if evaluations else None
        self.champion_estimator = None
        if champion:
            for name, m in fitted_models:
                if name == champion.model_name:
                    self.champion_estimator = m
                    break
        if self.champion_estimator is None and fitted_models:
            self.champion_estimator = fitted_models[0][1]

        return TournamentLeaderboardDTO(
            task_type=task_type,
            primary_metric=metric_str,
            champion_model=champion.model_name if champion else "None",
            champion_score=champion.mean_cv_score if champion else 0.0,
            stacking_candidates=top_candidates,
            leaderboard=evaluations,
        )
