"""Unit tests for 8-Model Tournament Arena."""
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

    # Check that each evaluation model has overfit_gap and latency
    for model_eval in leaderboard_dto.leaderboard:
        assert model_eval.fit_time_seconds >= 0.0
        assert model_eval.inference_latency_ms >= 0.0
        assert model_eval.overfit_gap >= 0.0
        assert len(model_eval.feature_importances) <= 10
