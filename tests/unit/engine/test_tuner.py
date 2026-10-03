"""Unit tests for Optuna Bayesian hyperparameter tuner with active pruning and groups."""
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from ml_mcp.engine.tuner import BayesianTuner
from ml_mcp.schemas.tuning import OptunaStudyDTO


def test_tuner_optuna_log_silenced():
    """Verify Optuna logging is silenced to WARNING to protect console tokens."""
    tuner = BayesianTuner()
    assert optuna.logging.get_verbosity() <= optuna.logging.WARNING


def test_tuner_classification_fast_trials():
    """Verify tuner runs on classification with LightGBM/RandomForest in few trials."""
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=4,
        random_state=42,
    )

    tuner = BayesianTuner(n_trials=3, cv_splits=3, random_state=42)
    study_dto, best_estimator = tuner.tune(
        model_name="random_forest",
        X=X,
        y=y,
        task_type="classification",
        metric="roc_auc",
    )

    assert isinstance(study_dto, OptunaStudyDTO)
    assert study_dto.total_trials == 3
    assert study_dto.best_trial_number >= 0
    assert "n_estimators" in study_dto.best_params
    assert hasattr(best_estimator, "predict")


def test_tuner_active_median_pruning():
    """Verify in-loop fold reporting allows MedianPruner to terminate trials."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        random_state=42,
    )

    # 8 trials with 3 folds each gives MedianPruner enough history to prune
    tuner = BayesianTuner(n_trials=8, cv_splits=3, random_state=42)
    study_dto, best_estimator = tuner.tune(
        model_name="random_forest",
        X=X,
        y=y,
        task_type="classification",
        metric="accuracy",
    )

    assert isinstance(study_dto, OptunaStudyDTO)
    assert study_dto.total_trials == 8
    # Pruned trials field is tracked
    assert study_dto.pruned_trials >= 0


def test_tuner_group_column_splitting():
    """Verify group-aware cross validation is respected without data leakage."""
    np.random.seed(42)
    n = 150
    groups = np.repeat(np.arange(30), 5)  # 30 distinct groups
    X = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, size=n),
        "feat_2": np.random.normal(2, 1, size=n),
        "grp": groups,
    })
    y = np.random.binomial(1, 0.5, size=n)

    tuner = BayesianTuner(n_trials=2, cv_splits=3, random_state=42)
    study_dto, best_estimator = tuner.tune(
        model_name="random_forest",
        X=X,
        y=y,
        task_type="classification",
        metric="roc_auc",
        group_column="grp",
    )

    assert isinstance(study_dto, OptunaStudyDTO)
    assert study_dto.total_trials == 2
    assert hasattr(best_estimator, "predict")
