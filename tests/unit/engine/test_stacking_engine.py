"""Unit tests for Leak-Free Stacking Ensemble Engine."""
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from ml_mcp.engine.stacking_engine import StackingEngine


def test_stacking_engine_classification():
    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, size=(n, 4))
    y = np.random.binomial(1, 0.5, size=n)

    base_models = [
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=10, random_state=42)),
        ("lr", LogisticRegression()),
    ]

    engine = StackingEngine()
    stacking_pipeline, oof_score = engine.build_stacking_ensemble(
        base_models=base_models,
        X=X,
        y=y,
        task_type="classification",
        cv_splits=3,
        scoring="roc_auc",
    )

    assert stacking_pipeline is not None
    assert oof_score > 0.0
    assert oof_score != 0.85  # No dummy fallback

    # Verify OOF meta-features are exposed
    assert len(engine.oof_predictions_) == 3
    assert engine.meta_features_ is not None

    # Test out-of-sample predictions
    test_X = np.random.normal(0, 1, size=(10, 4))
    preds = stacking_pipeline.predict(test_X)
    probs = stacking_pipeline.predict_proba(test_X)

    assert len(preds) == 10
    assert probs.shape == (10, 2)


def test_stacking_engine_regression_super_learner_positive_weights():
    """Verify Super Learner Non-Negative Least Squares meta-learner enforces positive weights."""
    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, size=(n, 4))
    y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + np.random.normal(0, 0.1, size=n)

    base_models = [
        ("rf", RandomForestRegressor(n_estimators=10, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ]

    engine = StackingEngine()
    stacking_pipeline, oof_score = engine.build_stacking_ensemble(
        base_models=base_models,
        X=X,
        y=y,
        task_type="regression",
        cv_splits=3,
        scoring="r2",
    )

    assert stacking_pipeline is not None
    assert oof_score > 0.0
    assert oof_score != 0.85

    # Check non-negative coefficients (Super Learner NNLS)
    final_est = stacking_pipeline.final_estimator_
    assert hasattr(final_est, "coef_")
    assert np.all(final_est.coef_ >= 0)


def test_stacking_engine_f1_and_rmse():
    np.random.seed(42)
    n = 60
    X = np.random.normal(0, 1, size=(n, 3))
    y_c = np.random.binomial(1, 0.5, size=n)
    y_r = X[:, 0] * 1.5 + np.random.normal(0, 0.1, size=n)

    engine = StackingEngine()
    # F1 scoring
    base_clf = [("lr", LogisticRegression())]
    m_clf, s_f1 = engine.build_stacking_ensemble(base_clf, X, y_c, scoring="f1_weighted", cv_splits=2)
    assert s_f1 > 0.0

    # RMSE scoring
    base_reg = [("ridge", Ridge())]
    m_reg, s_rmse = engine.build_stacking_ensemble(base_reg, X, y_r, task_type="regression", scoring="neg_root_mean_squared_error", cv_splits=2)
    assert s_rmse <= 0.0
