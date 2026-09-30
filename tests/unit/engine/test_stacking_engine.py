"""Unit tests for Leak-Free Stacking Ensemble Engine."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

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
    )

    assert stacking_pipeline is not None
    assert oof_score > 0.0

    # Test out-of-sample predictions
    test_X = np.random.normal(0, 1, size=(10, 4))
    preds = stacking_pipeline.predict(test_X)
    probs = stacking_pipeline.predict_proba(test_X)

    assert len(preds) == 10
    assert probs.shape == (10, 2)
