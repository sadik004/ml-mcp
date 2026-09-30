"""Tournament Leaderboard, Model Evaluation, and Lineage DTOs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator
from ml_mcp.schemas.base import BaseDTO


class ModelEvaluationDTO(BaseDTO):
    """Performance evaluation metrics for a single model across 5-fold cross validation."""

    model_name: str = Field(description="Algorithm name (e.g. CatBoostClassifier)")
    metric_name: str = Field(description="Primary evaluation metric name")
    mean_cv_score: float = Field(description="Mean score across all validation folds")
    std_cv_score: float = Field(ge=0.0, description="Standard deviation across validation folds")
    fit_time_seconds: float = Field(ge=0.0, description="Total wall-clock training time in seconds")
    inference_latency_ms: float = Field(ge=0.0, description="Single-record inference latency in ms")
    overfit_gap: float = Field(
        default=0.0, ge=0.0, description="Absolute performance divergence between train and val"
    )
    feature_importances: Dict[str, float] = Field(
        default_factory=dict,
        description="Top predictive features capped to top 10 for token protection",
    )

    @field_validator("feature_importances", mode="before")
    @classmethod
    def cap_feature_importances(cls, val: Any) -> Dict[str, float]:
        """Automatically sorts and caps feature importances to top 10 features descending."""
        if not isinstance(val, dict):
            return {}
        # Sort descending by importance score
        sorted_pairs = sorted(val.items(), key=lambda item: float(item[1]), reverse=True)
        # Cap to top 10
        top_k = sorted_pairs[:10]
        return {str(k): round(float(v), 5) for k, v in top_k}


class TournamentLeaderboardDTO(BaseDTO):
    """Leaderboard summary ranking all evaluated models in the 8-model tournament."""

    task_type: str = Field(description="classification or regression")
    primary_metric: str = Field(description="Optimized selection metric")
    champion_model: str = Field(description="Name of the winning model")
    champion_score: float = Field(description="Cross-validation score of the champion")
    stacking_candidates: List[str] = Field(
        default_factory=list, description="Top 3 distinct models selected for stacking blend"
    )
    leaderboard: List[ModelEvaluationDTO] = Field(
        default_factory=list, description="Ranked list of evaluated models"
    )

    def to_compact(self) -> Dict[str, Any]:
        """Compact leaderboard view containing only champion and top 3 summary."""
        return {
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            "champion_model": self.champion_model,
            "champion_score": round(self.champion_score, 4),
            "stacking_candidates": self.stacking_candidates,
            "leaderboard_summary": [
                {
                    "model": m.model_name,
                    "mean_score": round(m.mean_cv_score, 4),
                    "fit_time_s": round(m.fit_time_seconds, 2),
                }
                for m in self.leaderboard[:5]
            ],
        }


class LineageDTO(BaseDTO):
    """Lineage and provenance metadata tracking dataset integrity and experiment artifacts."""

    dataset_hash: str = Field(description="SHA-256 fingerprint of training data")
    row_count: int = Field(ge=0, description="Row count at execution")
    column_count: int = Field(ge=0, description="Column count at execution")
    random_seed: int = Field(default=42, description="Reproducibility random seed")
    git_commit: Optional[str] = Field(default=None, description="Active git commit hash")
    checkpoint_path: str = Field(description="Storage path to the checkpoint .joblib")
