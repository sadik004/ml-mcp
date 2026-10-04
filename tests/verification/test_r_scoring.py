"""RED Verification Suite: Scoring Honesty & Metric Integrity (C1, C2, C6, H9).

Enforces Rule T3 (independent sklearn/numpy oracles) and Rule T4 (no answer-faking mocks).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss

from ml_mcp.engine.stacking_engine import StackingEngine
from ml_mcp.engine.tournament import TournamentArena


def test_stacking_average_precision_is_true_ap() -> None:
    """C1: Stacking with average_precision must evaluate true AP, not fall through to accuracy."""
    X, y = make_classification(n_samples=400, n_features=10, random_state=42)
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ("lr", LogisticRegression(random_state=42)),
    ]
    engine = StackingEngine()
    model, score = engine.build_stacking_ensemble(
        base_models=base_models,
        X=X,
        y=y,
        task_type="classification",
        scoring="average_precision",
    )

    # Defect C1 verification:
    # Stacking engine must expose raw oof_meta_proba or compute true AP
    # Currently it returns accuracy (0.9025) while true AP is 0.8392
    assert hasattr(engine, "oof_meta_proba_"), "StackingEngine must expose oof_meta_proba_ for independent verification"
    oracle_ap = float(average_precision_score(y, engine.oof_meta_proba_[:, 1]))
    assert abs(score - oracle_ap) < 1e-12, f"Reported score {score} does not match sklearn AP {oracle_ap}"
    oracle_acc = float(accuracy_score(y, (engine.oof_meta_proba_[:, 1] >= 0.5).astype(int)))
    assert abs(score - oracle_acc) > 1e-4, f"Metric Impersonation Detected! Reported AP {score} equals accuracy {oracle_acc}"


def test_stacking_neg_log_loss_is_negative_log_loss() -> None:
    """C1: Stacking with neg_log_loss must report strictly negative log loss, not accuracy."""
    X, y = make_classification(n_samples=300, n_features=8, random_state=42)
    base_models = [
        ("lr", LogisticRegression(random_state=42)),
    ]
    engine = StackingEngine()
    model, score = engine.build_stacking_ensemble(
        base_models=base_models,
        X=X,
        y=y,
        task_type="classification",
        scoring="neg_log_loss",
    )
    assert hasattr(engine, "oof_meta_proba_")
    oracle_nll = -float(log_loss(y, engine.oof_meta_proba_))
    assert abs(score - oracle_nll) < 1e-12
    assert score < 0.0, f"neg_log_loss must be negative, got {score}"


def test_stacking_unknown_scoring_raises() -> None:
    """H9: Unknown metric must raise ValueError rather than silently defaulting to accuracy."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    base_models = [("lr", LogisticRegression(random_state=42))]
    engine = StackingEngine()
    with pytest.raises(ValueError) as exc_info:
        engine.build_stacking_ensemble(
            base_models=base_models,
            X=X,
            y=y,
            task_type="classification",
            scoring="unknown_scoring_typo",
        )
    assert "scoring" in str(exc_info.value).lower() or "metric" in str(exc_info.value).lower()



def test_stacking_oof_failure_never_returns_insample() -> None:
    """C2: OOF cross_val_predict failure must not return in-sample model.score() without warning."""
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    base_models = [("lr", LogisticRegression(random_state=42))]
    engine = StackingEngine()

    with patch("ml_mcp.engine.stacking_engine.cross_val_predict", side_effect=RuntimeError("Injected OOF failure")):
        model, score = engine.build_stacking_ensemble(
            base_models=base_models,
            X=X,
            y=y,
            task_type="classification",
            scoring="roc_auc",
        )
        # Must never return in-sample accuracy as OOF score
        in_sample = float(model.score(X, y))
        assert score is None or abs(score - in_sample) > 1e-6, "Must not return in-sample score on OOF failure"
        assert hasattr(engine, "warnings_")
        assert any("Injected OOF failure" in w for w in engine.warnings_)


def test_tournament_fold_scorer_failure_not_mixed() -> None:
    """C6: Scorer failure on a fold must not fall back to estimator.score() and average mixed metrics."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    arena = TournamentArena()
    # Inject fold failure for scoring
    with patch("sklearn.metrics.roc_auc_score", side_effect=[0.85, RuntimeError("Scorer fail"), 0.82, 0.84, 0.86]):
        results = arena.run_tournament(
            df=df,
            target_column="target",
            task_type="classification",
            scoring="roc_auc",
        )
        # Candidate must not average roc_auc with accuracy
        for model_eval in results.leaderboard:
            assert model_eval.cv_score is None or len(model_eval.warnings) > 0


def test_tournament_regression_group_uses_groupkfold() -> None:
    """H9: Regression with group_column must use GroupKFold, never StratifiedGroupKFold on continuous y."""
    X, y = make_regression(n_samples=200, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
    df["target"] = y
    df["group_id"] = np.repeat(np.arange(20), 10)
    arena = TournamentArena()
    report = arena.run_tournament(
        df=df,
        target_column="target",
        task_type="regression",
        scoring="r2",
        group_column="group_id",
    )
    assert report is not None
    assert getattr(report, "splitter", "") == "GroupKFold"
