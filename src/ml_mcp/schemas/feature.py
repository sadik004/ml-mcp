"""Pydantic schemas and DTOs for Phase 2 feature engineering and selection."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
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
