"""Data Hygiene, Pre-Flight Audit, and Leakage DTOs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field
from ml_mcp.schemas.base import BaseDTO


class AuditReportDTO(BaseDTO):
    """Pre-flight statistical and integrity scan report of a dataset."""

    row_count: int = Field(ge=0, description="Total number of rows")
    column_count: int = Field(ge=0, description="Total number of columns")
    missing_cells: int = Field(ge=0, description="Total missing cell count across all columns")
    missing_ratio: float = Field(ge=0.0, le=1.0, description="Proportion of missing values")
    infinite_count: int = Field(ge=0, description="Count of positive/negative infinite values")
    sentinel_count: int = Field(ge=0, description="Detected legacy sentinel values (-999, 999, etc.)")
    duplicate_rows: int = Field(ge=0, description="Count of completely duplicated rows")
    class_imbalance_ratio: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Majority class proportion"
    )
    recommended_metric: str = Field(
        default="accuracy", description="Auto-selected metric (accuracy, pr_auc, f1_weighted)"
    )
    group_column_candidate: Optional[str] = Field(
        default=None, description="Identified entity identifier for GroupKFold"
    )
    has_temporal_order: bool = Field(
        default=False, description="Flag indicating datetime sequence necessitating TimeSeriesSplit"
    )
    column_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed per-column dtypes, cardinality, and skew"
    )

    def to_compact(self) -> Dict[str, Any]:
        """Excludes detailed per-column breakdown to protect token budget."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_ratio": round(self.missing_ratio, 4),
            "infinite_count": self.infinite_count,
            "sentinel_count": self.sentinel_count,
            "duplicate_rows": self.duplicate_rows,
            "class_imbalance_ratio": round(self.class_imbalance_ratio, 4) if self.class_imbalance_ratio else None,
            "recommended_metric": self.recommended_metric,
            "group_column_candidate": self.group_column_candidate,
            "has_temporal_order": self.has_temporal_order,
        }


class TargetLeakageReportDTO(BaseDTO):
    """Diagnostic report detecting post-event features or direct target leakage."""

    target_column: str = Field(description="Name of the target variable")
    leaked_features: List[str] = Field(
        default_factory=list, description="Suspicious features exhibiting >=0.95 correlation"
    )
    correlation_matrix: Dict[str, float] = Field(
        default_factory=dict, description="Feature correlations with the target"
    )
    mutual_info_scores: Dict[str, float] = Field(
        default_factory=dict, description="Mutual information scores with the target"
    )
    has_critical_leakage: bool = Field(
        default=False, description="True if any feature exceeds leakage threshold"
    )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "has_critical_leakage": self.has_critical_leakage,
            "leaked_features": self.leaked_features,
            "highest_correlation": (
                round(max(self.correlation_matrix.values()), 4) if self.correlation_matrix else 0.0
            ),
        }


class TextFeatureReportDTO(BaseDTO):
    """Report identifying free-form text columns requiring dedicated TF-IDF embedding."""

    text_columns: List[str] = Field(
        default_factory=list, description="Columns qualified as free-form natural language"
    )
    avg_char_lengths: Dict[str, float] = Field(
        default_factory=dict, description="Average character length per column"
    )
    unique_token_counts: Dict[str, int] = Field(
        default_factory=dict, description="Distinct token count estimation"
    )
    recommended_strategy: str = Field(
        default="tfidf_sublinear", description="Recommended vectorization strategy"
    )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "text_columns": self.text_columns,
            "column_count": len(self.text_columns),
            "recommended_strategy": self.recommended_strategy,
        }
