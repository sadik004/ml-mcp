"""Unit tests for Tournament and Leaderboard DTOs."""
import pytest
from pydantic import ValidationError

from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.schemas.tournament import (
    ModelEvaluationDTO,
    TournamentLeaderboardDTO,
    LineageDTO,
)


def test_model_evaluation_dto_feature_importance_guard():
    # 25 features provided
    raw_importances = {f"feat_{i}": float(i) for i in range(25)}

    dto = ModelEvaluationDTO(
        model_name="CatBoostClassifier",
        metric_name="pr_auc",
        mean_cv_score=0.8942,
        std_cv_score=0.012,
        fit_time_seconds=14.5,
        inference_latency_ms=0.8,
        overfit_gap=0.021,
        feature_importances=raw_importances,
    )

    # Must be trimmed to top 10 features sorted descending
    assert len(dto.feature_importances) == 10
    assert "feat_24" in dto.feature_importances
    assert "feat_23" in dto.feature_importances
    assert "feat_0" not in dto.feature_importances
    # Verify descending sort
    values = list(dto.feature_importances.values())
    assert values == sorted(values, reverse=True)


def test_tournament_leaderboard_dto():
    m1 = ModelEvaluationDTO(
        model_name="CatBoostClassifier",
        metric_name="pr_auc",
        mean_cv_score=0.8942,
        std_cv_score=0.012,
        fit_time_seconds=14.5,
        inference_latency_ms=0.8,
        overfit_gap=0.021,
    )
    m2 = ModelEvaluationDTO(
        model_name="LightGBMClassifier",
        metric_name="pr_auc",
        mean_cv_score=0.8871,
        std_cv_score=0.015,
        fit_time_seconds=4.2,
        inference_latency_ms=0.3,
        overfit_gap=0.035,
    )

    dto = TournamentLeaderboardDTO(
        task_type="classification",
        primary_metric="pr_auc",
        champion_model="CatBoostClassifier",
        champion_score=0.8942,
        stacking_candidates=["CatBoostClassifier", "LightGBMClassifier"],
        leaderboard=[m1, m2],
    )

    assert dto.champion_model == "CatBoostClassifier"
    assert len(dto.leaderboard) == 2

    # Compact representation test
    compact = dto.to_compact()
    assert compact["champion_model"] == "CatBoostClassifier"
    assert "leaderboard_summary" in compact
    assert len(str(compact)) < 500

    # JSON sanitizer check
    sanitized = sanitize_for_json(dto.model_dump())
    assert isinstance(sanitized, dict)


def test_lineage_dto():
    dto = LineageDTO(
        dataset_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        row_count=50000,
        column_count=20,
        random_seed=42,
        git_commit="73f5f2b",
        checkpoint_path="/content/drive/MyDrive/ml_mcp/checkpoints/job_123.joblib",
    )

    assert dto.random_seed == 42
    assert "checkpoints" in dto.checkpoint_path
