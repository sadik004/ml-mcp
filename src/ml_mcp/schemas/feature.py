"""Pydantic schemas and DTOs for Phase 2 feature engineering and selection."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field
from ml_mcp.schemas.base import BaseDTO


class GroupBySpecDTO(BaseDTO):
    cat_col: str
    num_col: str
    aggregations: List[str] = ["mean", "std"]
    create_relative_diff: bool = True
    create_relative_ratio: bool = True
    create_zscore: bool = True


class FeaturePruningReportDTO(BaseDTO):
    original_feature_count: int
    pruned_feature_count: int
    selected_features: List[str]
    dropped_features: List[str]
    feature_importances: Dict[str, float]

class FeaturePipelineReportDTO(BaseDTO):
    """Unified Phase 2 Feature Pipeline Generation Report DTO."""

    original_shape: List[int] = Field(description="Original [rows, columns]")
    transformed_shape: List[int] = Field(description="Transformed [rows, columns]")
    synthesized_features: List[str] = Field(default_factory=list, description="Newly generated high-signal features")
    pruned_features: List[str] = Field(default_factory=list, description="Pruned zero/negative importance features")
    retained_features: List[str] = Field(default_factory=list, description="Final high-signal feature set")
    class_weights: Optional[Dict[str, float]] = Field(default=None, description="Computed class balancing sample weights")
    preprocessor_artifact_path: Optional[str] = Field(default=None, description="Path to fitted joblib pipeline artifact")
    transformed_dataset_path: Optional[str] = Field(default=None, description="Path to saved transformed CSV dataset artifact")
    receipt_card: str = Field(description="Human-readable Markdown feature receipt card")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "original_shape": self.original_shape,
            "transformed_shape": self.transformed_shape,
            "synthesized_count": len(self.synthesized_features),
            "synthesized_features": self.synthesized_features,
            "pruned_count": len(self.pruned_features),
            "retained_count": len(self.retained_features),
            "class_weights": self.class_weights,
            "transformed_dataset_path": self.transformed_dataset_path,
            "receipt_card": self.receipt_card,
        }

