"""Bayesian Hyperparameter Optimization using Optuna TPE and Active MedianPruner."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from ml_mcp.schemas.tuning import OptunaStudyDTO

# Silence Optuna info logs to protect console tokens
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _get_metric_direction(metric: str) -> str:
    """Infer optimization direction for a given metric."""
    minimize_metrics = {
        "log_loss",
        "neg_log_loss",
        "mse",
        "mean_squared_error",
        "neg_mean_squared_error",
        "rmse",
        "root_mean_squared_error",
        "neg_root_mean_squared_error",
        "mae",
        "mean_absolute_error",
        "neg_mean_absolute_error",
    }
    return "minimize" if metric.lower() in minimize_metrics else "maximize"


def _get_sklearn_scoring(metric: str, task_type: str) -> str:
    """Map friendly metric name to scikit-learn scoring string."""
    metric_map = {
        "roc_auc": "roc_auc",
        "auc": "roc_auc",
        "pr_auc": "average_precision",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "f1": "f1_weighted" if task_type == "classification" else "r2",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "log_loss": "neg_log_loss",
        "r2": "r2",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    return metric_map.get(metric.lower(), metric)


class BayesianTuner:
    """Bayesian Hyperparameter Tuner powered by Optuna TPE & Active In-Loop MedianPruner.

    Theoretical Basis:
        - Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (KDD 2019).
          Optuna: A Next-generation Hyperparameter Optimization Framework.
    """

    def __init__(
        self,
        n_trials: int = 20,
        time_budget_secs: Optional[int] = None,
        cv_splits: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        from ml_mcp.config import get_settings
        self.n_trials = n_trials
        self.time_budget_secs = time_budget_secs
        self.cv_splits = cv_splits
        self.random_state = random_state if random_state is not None else get_settings().random_state

    def _sample_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Sample hyperparameters based on model family."""
        name = model_name.lower().replace("-", "_").replace(" ", "_")
        params: Dict[str, Any] = {}

        if "lightgbm" in name or "lgbm" in name:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 200),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbose": -1,
                "random_state": self.random_state,
            }
        elif "xgboost" in name or "xgb" in name:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "verbosity": 0,
                "random_state": self.random_state,
            }
        elif "catboost" in name:
            params = {
                "iterations": trial.suggest_int("iterations", 30, 200),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "verbose": 0,
                "random_seed": self.random_state,
            }
        elif "extra_trees" in name or "extratrees" in name:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "random_state": self.random_state,
            }
        else:
            # Default to RandomForest
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "random_state": self.random_state,
            }

        return params

    def _instantiate_model(self, model_name: str, task_type: str, params: Dict[str, Any]) -> BaseEstimator:
        """Instantiate an estimator with the sampled parameters."""
        name = model_name.lower().replace("-", "_").replace(" ", "_")

        if "lightgbm" in name or "lgbm" in name:
            import lightgbm as lgb
            if task_type == "classification":
                return lgb.LGBMClassifier(**params)
            return lgb.LGBMRegressor(**params)

        if "xgboost" in name or "xgb" in name:
            import xgboost as xgb
            if task_type == "classification":
                return xgb.XGBClassifier(**params)
            return xgb.XGBRegressor(**params)

        if "catboost" in name:
            import catboost as cb
            if task_type == "classification":
                return cb.CatBoostClassifier(**params)
            return cb.CatBoostRegressor(**params)

        if "extra_trees" in name or "extratrees" in name:
            if task_type == "classification":
                return ExtraTreesClassifier(**params)
            return ExtraTreesRegressor(**params)

        # Default RandomForest
        if task_type == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    def tune(
        self,
        model_name: str,
        X: Any,
        y: Any,
        task_type: str = "classification",
        metric: str = "roc_auc",
        direction: Optional[str] = None,
        group_column: Optional[str] = None,
        groups: Optional[Any] = None,
        study_name: Optional[str] = None,
    ) -> Tuple[OptunaStudyDTO, BaseEstimator]:
        """Execute Bayesian Optimization with In-Loop Median Pruning and Group Splitting."""
        groups_arr: Optional[np.ndarray] = None
        if isinstance(X, pd.DataFrame):
            if group_column and group_column in X.columns:
                groups_arr = np.asarray(X[group_column])
                X_clean = X.drop(columns=[group_column])
            else:
                groups_arr = np.asarray(groups) if groups is not None else None
                X_clean = X
            X_arr = np.asarray(X_clean)
        else:
            groups_arr = np.asarray(groups) if groups is not None else None
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        resolved_direction: Literal["minimize", "maximize"] = direction if direction in ["minimize", "maximize"] else _get_metric_direction(metric)  # type: ignore[assignment]
        optuna_direction = resolved_direction
        scoring_metric = _get_sklearn_scoring(metric, task_type)
        scorer = get_scorer(scoring_metric)

        # Build CV splitter with Group Support
        if groups_arr is not None:
            if task_type == "classification":
                cv = StratifiedGroupKFold(n_splits=self.cv_splits)
            else:
                cv = GroupKFold(n_splits=self.cv_splits)
        else:
            if task_type == "classification":
                cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=1)
        sampler = TPESampler(seed=self.random_state)

        study_id = study_name or f"optuna_{model_name}_{metric}"
        study = optuna.create_study(
            study_name=study_id,
            direction=optuna_direction,
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, model_name)
            estimator = self._instantiate_model(model_name, task_type, params)

            fold_scores: List[float] = []
            split_gen = cv.split(X_arr, y_arr, groups=groups_arr) if groups_arr is not None else cv.split(X_arr, y_arr)

            for step, (train_idx, val_idx) in enumerate(split_gen):
                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

                fold_est = clone(estimator)
                fold_est.fit(X_tr, y_tr)

                score = float(scorer(fold_est, X_va, y_va))
                fold_scores.append(score)

                # Report running intermediate score to Optuna MedianPruner
                intermediate_score = float(np.mean(fold_scores))
                if scoring_metric.startswith("neg_") and optuna_direction == "minimize":
                    report_score = -intermediate_score
                else:
                    report_score = intermediate_score

                trial.report(report_score, step=step)

                # Active In-Loop Early Stopping (Akiba et al. KDD 2019)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(fold_scores))
            if scoring_metric.startswith("neg_") and optuna_direction == "minimize":
                return -mean_score
            return mean_score

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.time_budget_secs,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = float(study.best_value)
        best_trial_number = int(study.best_trial.number)

        # Count pruned trials
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        study_dto = OptunaStudyDTO(
            study_name=study_id,
            best_trial_number=best_trial_number,
            best_params=best_params,
            best_value=best_value,
            total_trials=len(study.trials),
            pruned_trials=pruned_trials,
            direction=optuna_direction,
            metric=metric,
        )

        # Refit best estimator on full training set
        all_best_params = dict(best_params)
        best_estimator = self._instantiate_model(model_name, task_type, all_best_params)
        best_estimator.fit(X_arr, y_arr)

        return study_dto, best_estimator
