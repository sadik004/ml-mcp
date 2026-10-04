"""8-Model Competitive Arena with Cross-Validation and Diversity-Guarded Stacking."""
from __future__ import annotations

import logging
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
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml_mcp.config import get_settings
from ml_mcp.engine.gpu_manager import GPUManager
from ml_mcp.engine.pipeline_builder import build_sealed_pipeline
from ml_mcp.engine.scoring import resolve_scorer
from ml_mcp.engine.stacking_engine import StackingEngine
from ml_mcp.schemas.tournament import ModelEvaluationDTO, TournamentLeaderboardDTO

logger = logging.getLogger(__name__)


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
        self.champion_pipeline_: Optional[Pipeline] = None

    def _get_roster(self, task_type: str, warnings: Optional[List[str]] = None) -> List[Tuple[str, Any]]:
        """Constructs the competitive roster prioritizing GPU acceleration when available."""
        models: List[Tuple[str, Any]] = []
        n_trees = 30 if self.fast_mode else 100
        seed = get_settings().random_state

        if task_type == "classification":
            models.append(("LogisticRegression", LogisticRegression(max_iter=500, random_state=seed)))
            models.append(("RidgeClassifier", RidgeClassifier(random_state=seed)))
            models.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)))
            models.append(("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)))
            models.append(("HistGradientBoostingClassifier", HistGradientBoostingClassifier(random_state=seed)))

            # LightGBM
            lgb_p = dict(self.gpu_manager.get_lgb_params())
            lgb_p.setdefault("verbose", -1)
            try:
                from lightgbm import LGBMClassifier
                models.append(("LGBMClassifier", LGBMClassifier(n_estimators=n_trees, random_state=seed, **lgb_p)))
            except Exception as e:
                msg = f"LGBMClassifier excluded from tournament: {e}"
                logger.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

            # XGBoost
            xgb_p = dict(self.gpu_manager.get_xgb_params())
            try:
                from xgboost import XGBClassifier
                models.append(("XGBClassifier", XGBClassifier(n_estimators=n_trees, random_state=seed, eval_metric="logloss", **xgb_p)))
            except Exception as e:
                msg = f"XGBClassifier excluded from tournament: {e}"
                logger.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

            # CatBoost
            cb_p = dict(self.gpu_manager.get_catboost_params())
            cb_p.setdefault("verbose", 0)
            try:
                from catboost import CatBoostClassifier
                models.append(("CatBoostClassifier", CatBoostClassifier(iterations=n_trees, random_seed=seed, **cb_p)))
            except Exception as e:
                msg = f"CatBoostClassifier excluded from tournament: {e}"
                logger.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

        else:
            # Regression Roster
            models.append(("Ridge", Ridge(random_state=seed)))
            models.append(("RandomForestRegressor", RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)))
            models.append(("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)))
            models.append(("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=seed)))

            lgb_p = dict(self.gpu_manager.get_lgb_params())
            lgb_p.setdefault("verbose", -1)
            try:
                from lightgbm import LGBMRegressor
                models.append(("LGBMRegressor", LGBMRegressor(n_estimators=n_trees, random_state=seed, **lgb_p)))
            except Exception as e:
                msg = f"LGBMRegressor excluded from tournament: {e}"
                logger.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

            xgb_p = dict(self.gpu_manager.get_xgb_params())
            try:
                from xgboost import XGBRegressor
                models.append(("XGBRegressor", XGBRegressor(n_estimators=n_trees, random_state=seed, **xgb_p)))
            except Exception as e:
                msg = f"XGBRegressor excluded from tournament: {e}"
                logger.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

            cb_p = dict(self.gpu_manager.get_catboost_params())
            cb_p.setdefault("verbose", 0)
            try:
                from catboost import CatBoostRegressor
                models.append(("CatBoostRegressor", CatBoostRegressor(iterations=n_trees, random_seed=seed, **cb_p)))
            except Exception as e:
                msg = f"CatBoostRegressor excluded from tournament: {e}"
                logger.warning(msg)
                if warnings is not None:
                    warnings.append(msg)

        return models

    def _extract_feature_importances(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extracts and formats raw feature importances."""
        importances: Dict[str, float] = {}
        try:
            est = model.named_steps.get("model", model[-1]) if isinstance(model, Pipeline) else model
            if hasattr(est, "feature_importances_"):
                raw = est.feature_importances_
                for name, imp in zip(feature_names, raw):
                    importances[name] = float(imp)
            elif hasattr(est, "coef_"):
                coef = est.coef_
                raw = np.abs(coef).ravel() if hasattr(coef, "ravel") else np.abs(coef)
                for name, imp in zip(feature_names, raw[:len(feature_names)]):
                    importances[name] = float(imp)
        except Exception as e:
            logger.debug(f"Feature importance extraction not supported or failed for {model}: {e}")
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
        warnings: List[str] = []
        default_metric = "roc_auc" if task_type == "classification" else "r2"
        metric_str = scoring or default_metric
        try:
            scorer_fn, needs_proba = resolve_scorer(metric_str)
        except Exception as sc_err:
            msg = f"Requested scorer '{metric_str}' is invalid ({sc_err}). Defaulting to '{default_metric}'."
            logger.warning(msg)
            warnings.append(msg)
            metric_str = default_metric
            scorer_fn, needs_proba = resolve_scorer(metric_str)

        settings = get_settings()
        min_n = settings.min_holdout_n
        holdout_fraction = settings.holdout_fraction

        if len(df) >= min_n:
            from sklearn.model_selection import train_test_split
            strat = df[target_column] if (task_type == "classification" and len(np.unique(df[target_column])) <= 10) else None
            try:
                sel_indices, hold_indices = train_test_split(
                    np.arange(len(df)),
                    test_size=holdout_fraction,
                    random_state=settings.random_state,
                    stratify=strat,
                )
            except Exception:
                sel_indices, hold_indices = train_test_split(
                    np.arange(len(df)),
                    test_size=holdout_fraction,
                    random_state=settings.random_state,
                )
            selection_df = df.iloc[sel_indices].copy()
            holdout_df = df.iloc[hold_indices].copy()
            champion_score_source = "outer_holdout"
        else:
            sel_indices = np.arange(len(df))
            hold_indices = np.array([], dtype=int)
            selection_df = df
            holdout_df = None
            champion_score_source = "unavailable"
            warnings.append("Sample size too small for outer holdout; champion holdout score unavailable.")

        X = selection_df.drop(columns=[target_column])
        if group_column and group_column in X.columns:
            groups = X[group_column]
            X = X.drop(columns=[group_column])
        else:
            groups = None

        y = selection_df[target_column]

        # Defensive auto-preprocessing flag if non-numeric/null features exist
        has_non_numeric = any(
            X[col].dtype == "object" or isinstance(X[col].dtype, pd.StringDtype) or X[col].isnull().any()
            for col in X.columns
        )
        feature_names = [str(c) for c in X.columns]

        # Configure cross-validation splitter
        if group_column and groups is not None:
            if task_type == "classification":
                cv = StratifiedGroupKFold(n_splits=self.cv_splits)
                splitter_name = "StratifiedGroupKFold"
            else:
                cv = GroupKFold(n_splits=self.cv_splits)
                splitter_name = "GroupKFold"
            split_gen = cv.split(X, y, groups)
        elif task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=get_settings().random_state)
            splitter_name = "StratifiedKFold"
            split_gen = cv.split(X, y)
        else:
            cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=get_settings().random_state)
            splitter_name = "KFold"
            split_gen = cv.split(X, y)

        splits_list = list(split_gen)
        roster = self._get_roster(task_type, warnings=warnings)
        evaluations: List[ModelEvaluationDTO] = []
        fitted_models: List[Tuple[str, Any]] = []
        oof_predictions: Dict[str, np.ndarray] = {}

        n_samples = len(X)

        for name, base_model in roster:
            fold_val_scores: List[float] = []
            fold_train_scores: List[float] = []
            model_warnings: List[str] = []
            has_scorer_failure = False
            oof_vector = np.zeros(n_samples, dtype=float)
            start_time = time.perf_counter()
            best_val_score = -float("inf")
            best_fold_model: Any = None

            # Execute cross-validation folds using clone to prevent state leakage
            for train_idx, val_idx in splits_list:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                if has_non_numeric:
                    fold_model = build_sealed_pipeline(clone(base_model), X_train, target_column=target_column)
                else:
                    fold_model = clone(base_model)
                try:
                    fold_model.fit(X_train, y_train)
                except Exception as fit_err:
                    # Fallback to CPU if GPU driver failure
                    cpu_params = self.gpu_manager.get_fallback_cpu_params(name)
                    est_step = fold_model.named_steps["model"] if isinstance(fold_model, Pipeline) else fold_model
                    if hasattr(est_step, "set_params"):
                        try:
                            est_step.set_params(**cpu_params)
                            fold_model.fit(X_train, y_train)
                        except Exception as cpu_err:
                            msg = f"Model {name} failed on fold {len(fold_val_scores)+1}: {cpu_err}"
                            logger.warning(msg)
                            model_warnings.append(msg)
                            warnings.append(msg)
                            continue
                    else:
                        msg = f"Model {name} failed on fold {len(fold_val_scores)+1}: {fit_err}"
                        logger.warning(msg)
                        model_warnings.append(msg)
                        warnings.append(msg)
                        continue

                # Evaluate using unified Scoring Engine (eliminates model.score trap)
                try:
                    if needs_proba:
                        if hasattr(fold_model, "predict_proba"):
                            train_pred = fold_model.predict_proba(X_train)
                            val_pred = fold_model.predict_proba(X_val)
                            train_metric_in = train_pred[:, 1] if train_pred.ndim > 1 and train_pred.shape[1] == 2 else train_pred
                            val_metric_in = val_pred[:, 1] if val_pred.ndim > 1 and val_pred.shape[1] == 2 else val_pred
                        elif hasattr(fold_model, "decision_function"):
                            train_metric_in = fold_model.decision_function(X_train)
                            val_metric_in = fold_model.decision_function(X_val)
                        else:
                            train_metric_in = fold_model.predict(X_train)
                            val_metric_in = fold_model.predict(X_val)
                    else:
                        train_metric_in = fold_model.predict(X_train)
                        val_metric_in = fold_model.predict(X_val)

                    train_score = float(scorer_fn(y_train.to_numpy(), train_metric_in))
                    val_score = float(scorer_fn(y_val.to_numpy(), val_metric_in))
                    fold_train_scores.append(train_score)
                    fold_val_scores.append(val_score)

                    if val_score > best_val_score or best_fold_model is None:
                        best_val_score = val_score
                        best_fold_model = fold_model
                except Exception as sc_err:
                    # Defect C6 Fix: Never fall back to fold_model.score() which mixes metrics!
                    msg = f"Scorer '{metric_str}' failed on fold for {name}: {sc_err}."
                    logger.warning(msg)
                    model_warnings.append(msg)
                    warnings.append(msg)
                    has_scorer_failure = True

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

            # Fit final model: if has_non_numeric, use best_fold_model to prevent full-dataset preprocessor leakage
            if has_non_numeric:
                final_fitted = best_fold_model if best_fold_model is not None else fold_model
            else:
                final_fitted = clone(base_model)
                try:
                    final_fitted.fit(X, y)
                except Exception as final_fit_err:
                    logger.warning(f"Final refit on full dataset failed for {name}: {final_fit_err}. Using fold model.")
                    warnings.append(f"Final refit on full dataset failed for {name}: {final_fit_err}. Using fold model.")
                    final_fitted = fold_model

            fit_time = time.perf_counter() - start_time
            oof_predictions[name] = oof_vector

            # Compute inference latency (ms per sample)
            lat_start = time.perf_counter()
            test_slice = X.head(min(50, len(X)))
            _ = final_fitted.predict(test_slice)
            latency_ms = (time.perf_counter() - lat_start) * 1000.0 / len(test_slice)

            if has_scorer_failure or not fold_val_scores:
                mean_val = None
                std_val = 0.0
                mean_train = None
                overfit_gap = 0.0
            else:
                mean_val = float(np.mean(fold_val_scores))
                std_val = float(np.std(fold_val_scores))
                mean_train = float(np.mean(fold_train_scores)) if fold_train_scores else mean_val
                overfit_gap = float(abs(mean_train - mean_val))

            feat_importances = self._extract_feature_importances(final_fitted, feature_names)

            eval_dto = ModelEvaluationDTO(
                model_name=name,
                metric_name=metric_str,
                mean_cv_score=round(mean_val, 4) if mean_val is not None else None,
                std_cv_score=round(std_val, 4),
                fit_time_seconds=round(fit_time, 2),
                inference_latency_ms=round(latency_ms, 3),
                overfit_gap=round(overfit_gap, 4),
                feature_importances=feat_importances,
                warnings=model_warnings,
            )
            evaluations.append(eval_dto)
            fitted_models.append((name, final_fitted))

        # Sort leaderboard descending by validation score (handling None safely)
        evaluations.sort(
            key=lambda m: (m.mean_cv_score is not None, m.mean_cv_score if m.mean_cv_score is not None else -float("inf")),
            reverse=True,
        )

        # Select Stacking candidates using Pairwise OOF Prediction Diversity Guard (Kuncheva & Whitaker 2003)
        top_candidates = self._select_diverse_candidates(
            sorted_evaluations=evaluations,
            oof_dict=oof_predictions,
            max_candidates=3,
            correlation_threshold=0.95,
        )

        stacking_base = [(name, model) for name, model in fitted_models if name in top_candidates]

        if len(stacking_base) >= 2 and not has_non_numeric:
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
                    if needs_proba:
                        if hasattr(stacking_model, "predict_proba"):
                            stk_train_p = stacking_model.predict_proba(X)
                            stk_in = stk_train_p[:, 1] if stk_train_p.ndim > 1 and stk_train_p.shape[1] == 2 else stk_train_p
                        elif hasattr(stacking_model, "decision_function"):
                            stk_in = stacking_model.decision_function(X)
                        else:
                            stk_in = stacking_model.predict(X)
                    else:
                        stk_in = stacking_model.predict(X)

                    stack_train_score = float(scorer_fn(y.to_numpy(), stk_in))
                    stack_overfit_gap = float(abs(stack_train_score - stacking_score)) if stacking_score is not None else 0.0
                except Exception as ov_err:
                    stack_overfit_gap = 0.0
                    warnings.append(f"Stacking train score evaluation failed ({ov_err}); overfit gap recorded as 0.0.")

                if stacking_score is not None:
                    stacking_eval = ModelEvaluationDTO(
                        model_name="StackingEnsemble",
                        metric_name=metric_str,
                        mean_cv_score=round(stacking_score, 4),
                        std_cv_score=0.0,
                        fit_time_seconds=round(stack_fit_time, 2),
                        inference_latency_ms=round(stack_latency_ms, 3),
                        overfit_gap=round(stack_overfit_gap, 4),
                        feature_importances={},
                        warnings=list(self.stacking_engine.warnings_),
                    )
                    evaluations.append(stacking_eval)
                    evaluations.sort(
                        key=lambda m: (m.mean_cv_score is not None, m.mean_cv_score if m.mean_cv_score is not None else -float("inf")),
                        reverse=True,
                    )
                    fitted_models.append(("StackingEnsemble", stacking_model))
            except Exception as stack_err:
                msg = f"Stacking ensemble build failed: {stack_err}"
                logger.warning(msg)
                warnings.append(msg)
        elif len(stacking_base) >= 2 and has_non_numeric:
            warnings.append("Stacking ensemble skipped for non-numeric/mixed data to prevent full-dataset preprocessor leakage.")

        champion = evaluations[0] if evaluations else None
        self.champion_estimator = None
        if champion:
            for name, m in fitted_models:
                if name == champion.model_name:
                    self.champion_estimator = m
                    break
        if self.champion_estimator is None and fitted_models:
            self.champion_estimator = fitted_models[0][1]

        if isinstance(self.champion_estimator, Pipeline):
            self.champion_pipeline_ = self.champion_estimator
        elif self.champion_estimator is not None:
            self.champion_pipeline_ = Pipeline([("model", self.champion_estimator)])
        else:
            self.champion_pipeline_ = None

        # H4: Evaluate champion model on untouched outer holdout to eliminate Winner's Curse
        selection_cv_score = champion.mean_cv_score if (champion and champion.mean_cv_score is not None) else None
        champion_final_score: Optional[float] = None

        if holdout_df is not None and self.champion_pipeline_ is not None and len(holdout_df) > 0:
            X_hold = holdout_df.drop(columns=[target_column])
            if group_column and group_column in X_hold.columns:
                X_hold = X_hold.drop(columns=[group_column])
            y_hold = holdout_df[target_column]
            try:
                if needs_proba:
                    if hasattr(self.champion_pipeline_, "predict_proba"):
                        h_pred = self.champion_pipeline_.predict_proba(X_hold)
                        h_metric_in = h_pred[:, 1] if h_pred.ndim > 1 and h_pred.shape[1] == 2 else h_pred
                    elif hasattr(self.champion_pipeline_, "decision_function"):
                        h_metric_in = self.champion_pipeline_.decision_function(X_hold)
                    else:
                        h_metric_in = self.champion_pipeline_.predict(X_hold)
                else:
                    h_metric_in = self.champion_pipeline_.predict(X_hold)

                holdout_score = float(scorer_fn(y_hold.to_numpy(), h_metric_in))
                champion_final_score = round(holdout_score, 4)
                champion_score_source = "outer_holdout"
            except Exception as h_err:
                logger.warning(f"Champion evaluation on outer holdout failed: {h_err}")
                warnings.append(f"Champion evaluation on outer holdout failed: {h_err}")
                champion_final_score = selection_cv_score
                champion_score_source = "cross_validation"
        else:
            champion_final_score = selection_cv_score
            champion_score_source = "outer_holdout" if champion_score_source == "outer_holdout" else "unavailable"

        return TournamentLeaderboardDTO(
            task_type=task_type,
            primary_metric=metric_str,
            champion_model=champion.model_name if champion else "None",
            champion_score=champion_final_score,
            champion_score_source=champion_score_source,
            selection_cv_score=selection_cv_score,
            selection_index=sel_indices.tolist() if hasattr(sel_indices, "tolist") else list(sel_indices),
            holdout_index=hold_indices.tolist() if hasattr(hold_indices, "tolist") else list(hold_indices),
            evaluation_mode="held_out_test" if champion_score_source == "outer_holdout" else "in_sample",
            splitter=splitter_name,
            stacking_candidates=top_candidates,
            leaderboard=evaluations,
            warnings=warnings,
        )
