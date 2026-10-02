"""Optuna Bayesian Hyperparameter Optimization Engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ml_mcp.schemas.tuning import OptunaStudyDTO

logger = logging.getLogger(__name__)

# Enforce logging verbosity guard to protect console tokens
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_metric_direction(metric: str) -> str:
    """Determine optimization direction from metric name."""
    minimize_metrics = {"log_loss", "neg_log_loss", "rmse", "neg_root_mean_squared_error", "mae", "neg_mean_absolute_error", "mse"}
    if metric.lower() in minimize_metrics:
        return "minimize"
    return "maximize"


def _get_sklearn_scoring(metric: str, task_type: str) -> str:
    """Map user metric to standard scikit-learn scoring string."""
    metric_lower = metric.lower()
    if task_type == "classification":
        if "auc" in metric_lower or "roc" in metric_lower:
            return "roc_auc"
        if "pr" in metric_lower:
            return "average_precision"
        if "f1" in metric_lower:
            return "f1_macro"
        if "log_loss" in metric_lower:
            return "neg_log_loss"
        return "accuracy"
    else:
        if "rmse" in metric_lower:
            return "neg_root_mean_squared_error"
        if "mae" in metric_lower:
            return "neg_mean_absolute_error"
        return "r2"


class BayesianTuner:
    """Bayesian Hyperparameter Tuner powered by Optuna TPE & MedianPruner."""

    def __init__(
        self,
        n_trials: int = 20,
        time_budget_secs: Optional[int] = None,
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.time_budget_secs = time_budget_secs
        self.cv_splits = cv_splits
        self.random_state = random_state

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
        study_name: Optional[str] = None,
    ) -> Tuple[OptunaStudyDTO, BaseEstimator]:
        """Execute Bayesian Optimization with Cross-Validation and Early Pruning."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if direction is None:
            direction = _get_metric_direction(metric)

        optuna_direction = direction
        scoring_metric = _get_sklearn_scoring(metric, task_type)

        # Build CV splitter
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

            # Evaluate with Cross-Validation
            scores = cross_val_score(estimator, X_arr, y_arr, cv=cv, scoring=scoring_metric, n_jobs=1)
            mean_score = float(np.mean(scores))

            # Scikit-learn negates loss metrics (e.g. neg_log_loss, neg_root_mean_squared_error)
            if scoring_metric.startswith("neg_"):
                # If optimizing minimize direction, invert to positive for readability
                if optuna_direction == "minimize":
                    return -mean_score
                return mean_score

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
        all_best_params = self._sample_params(study.best_trial, model_name)
        # Update with chosen trial params
        all_best_params.update(best_params)
        best_estimator = self._instantiate_model(model_name, task_type, all_best_params)
        best_estimator.fit(X_arr, y_arr)

        return study_dto, best_estimator
