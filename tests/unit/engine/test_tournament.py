"""Unit tests for 8-Model Tournament Arena with metric alignment and diversity guard."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.tournament import TournamentArena
from ml_mcp.schemas.tournament import TournamentLeaderboardDTO


def test_tournament_arena_fast_execution():
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "num_1": np.random.normal(0, 1, size=n),
        "num_2": np.random.uniform(10, 50, size=n),
        "target": np.random.binomial(1, 0.4, size=n),
    })

    arena = TournamentArena(cv_splits=2, fast_mode=True)
    leaderboard_dto = arena.run_tournament(
        df=X,
        target_column="target",
        task_type="classification",
    )

    assert isinstance(leaderboard_dto, TournamentLeaderboardDTO)
    assert leaderboard_dto.champion_model is not None
    assert len(leaderboard_dto.leaderboard) >= 5
    assert len(leaderboard_dto.stacking_candidates) >= 2

    # Check that each evaluation model has calculated latency and overfit gap
    for model_eval in leaderboard_dto.leaderboard:
        assert model_eval.fit_time_seconds >= 0.0
        assert model_eval.inference_latency_ms >= 0.0
        assert model_eval.overfit_gap >= 0.0
        assert len(model_eval.feature_importances) <= 10


def test_tournament_metric_alignment_and_diversity_guard():
    """Verify tournament honors scoring metric (e.g. roc_auc) and diversity filter works."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feat_a": np.random.normal(0, 1, size=n),
        "feat_b": np.random.normal(5, 2, size=n),
        "feat_c": np.random.uniform(0, 1, size=n),
        "target": np.random.binomial(1, 0.5, size=n),
    })

    arena = TournamentArena(cv_splits=2, fast_mode=True)
    res = arena.run_tournament(
        df=df,
        target_column="target",
        task_type="classification",
        scoring="roc_auc",
    )

    # 1. Metric alignment check
    assert res.primary_metric == "roc_auc"
    for m in res.leaderboard:
        assert m.metric_name == "roc_auc"

    # 2. Check stacking candidate selection (max 3 diverse models)
    assert 2 <= len(res.stacking_candidates) <= 3

    # 3. Check StackingEnsemble is not using dummy stats
    stack_eval = next((m for m in res.leaderboard if m.model_name == "StackingEnsemble"), None)
    if stack_eval:
        assert stack_eval.mean_cv_score > 0.0
        assert stack_eval.metric_name == "roc_auc"
        assert stack_eval.inference_latency_ms > 0.0


def test_tournament_gradient_boosters_roster_participation():
    """Verify that LightGBM and CatBoost do not crash with duplicate verbose kwargs
    and successfully participate in the competitive tournament."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "target": np.random.binomial(1, 0.5, size=n),
    })

    arena = TournamentArena(cv_splits=2, fast_mode=True)
    res = arena.run_tournament(df=df, target_column="target", task_type="classification")
    models = [m.model_name for m in res.leaderboard]

    # Verify both LGBM and CatBoost successfully participated
    assert "LGBMClassifier" in models, f"LGBMClassifier missing from tournament roster: {models}"
    assert "CatBoostClassifier" in models, f"CatBoostClassifier missing from tournament roster: {models}"


def test_tournament_surfaces_model_exclusion_warning():
    """Verify that when a model fails to import or initialize, it records a structured warning."""
    from unittest.mock import patch
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "target": np.random.binomial(1, 0.5, size=n),
    })

    arena = TournamentArena(cv_splits=2, fast_mode=True)
    with patch.dict("sys.modules", {"lightgbm": None}):
        res = arena.run_tournament(df=df, target_column="target", task_type="classification")

    models = [m.model_name for m in res.leaderboard]
    assert "LGBMClassifier" not in models
    assert any("LGBMClassifier excluded from tournament:" in w for w in res.warnings)


def test_tournament_regression_fast():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "target": np.random.randn(n) * 2.0 + 1.0,
    })
    arena = TournamentArena(cv_splits=2, fast_mode=True)
    res = arena.run_tournament(df=df, target_column="target", task_type="regression")
    assert res.task_type == "regression"
    assert res.champion_model is not None
    assert len(res.leaderboard) >= 4


