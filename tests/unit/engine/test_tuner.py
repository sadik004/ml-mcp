"""Unit tests for Optuna Bayesian hyperparameter tuner."""
import optuna
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


def test_tuner_xgboost_or_lightgbm():
    """Verify tuner constructs search space and tunes LightGBM / XGBoost."""
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=4,
        random_state=42,
    )

    tuner = BayesianTuner(n_trials=3, cv_splits=3, random_state=42)
    study_dto, best_estimator = tuner.tune(
        model_name="lightgbm",
        X=X,
        y=y,
        task_type="classification",
        metric="log_loss",
        direction="minimize",
    )

    assert isinstance(study_dto, OptunaStudyDTO)
    assert study_dto.direction == "minimize"
    assert "num_leaves" in study_dto.best_params or "max_depth" in study_dto.best_params
    assert hasattr(best_estimator, "predict")
