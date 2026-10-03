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
    id_memorization_columns: List[str] = Field(
        default_factory=list, description="High-entropy / unique ID columns flagged for removal to prevent memorization"
    )
    target_skewness: Optional[float] = Field(
        default=None, description="Fisher-Pearson skewness coefficient for continuous targets"
    )
    recommended_split_strategy: str = Field(
        default="stratified_kfold", description="Recommended CV splitting strategy (stratified, group, time_series, kfold)"
    )
    missingness_mechanisms: Dict[str, str] = Field(
        default_factory=dict, description="Missingness mechanism classification (MCAR vs MNAR) per missing column"
    )
    benford_anomalies: List[str] = Field(
        default_factory=list, description="Columns exhibiting significant first-digit deviation from Benford's Law (p < 0.01)"
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
            "id_memorization_columns": self.id_memorization_columns,
            "target_skewness": round(self.target_skewness, 4) if self.target_skewness is not None else None,
            "recommended_split_strategy": self.recommended_split_strategy,
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
    cramers_v_scores: Dict[str, float] = Field(
        default_factory=dict, description="Bias-corrected Cramér's V scores for categorical predictors"
    )
    chatterjee_scores: Dict[str, float] = Field(
        default_factory=dict, description="Chatterjee rank correlation xi scores (JASA 2021)"
    )
    pps_scores: Dict[str, float] = Field(
        default_factory=dict, description="Predictive Power Score single-feature decision tree scores"
    )
    has_critical_leakage: bool = Field(
        default=False, description="True if any feature exceeds leakage threshold"
    )

    def to_compact(self) -> Dict[str, Any]:
        max_corr = max(self.correlation_matrix.values()) if self.correlation_matrix else 0.0
        max_v = max(self.cramers_v_scores.values()) if self.cramers_v_scores else 0.0
        return {
            "target_column": self.target_column,
            "has_critical_leakage": self.has_critical_leakage,
            "leaked_features": self.leaked_features,
            "highest_correlation": round(max_corr, 4),
            "highest_cramers_v": round(max_v, 4),
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


class LabelErrorSampleDTO(BaseDTO):
    """Information regarding an individual detected corrupted label."""

    sample_index: int = Field(description="Row index of the suspicious sample")
    given_label: Any = Field(description="Original annotated ground truth label")
    suggested_label: Any = Field(description="Mathematically estimated true label")
    confidence: float = Field(description="Model confidence for suggested label")


class LabelErrorReportDTO(BaseDTO):
    """MIT Confident Learning diagnostic report for label noise in dataset."""

    total_samples: int = Field(description="Total evaluated samples")
    total_errors: int = Field(description="Total detected label errors")
    error_rate: float = Field(description="Estimated label noise ratio")
    task_type: str = Field(default="classification", description="Task type evaluated (classification or regression)")
    class_thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Per-class self-confidence thresholds t_j (for classification)"
    )
    flagged_samples: List[LabelErrorSampleDTO] = Field(
        default_factory=list, description="Samples flagged as label errors"
    )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "task_type": self.task_type,
            "class_thresholds": {k: round(v, 4) for k, v in self.class_thresholds.items()},
            "flagged_count": len(self.flagged_samples),
        }


class ColumnConstraintViolationDTO(BaseDTO):
    """Constraint validation violation metrics for a single column."""

    column: str = Field(description="Column name")
    violation_count: int = Field(description="Number of rows violating constraint")
    violation_rate: float = Field(description="Fraction of rows violating constraint")
    rule_broken: str = Field(description="Description of violated constraint rule")


class ConstraintValidationReportDTO(BaseDTO):
    """Amazon Deequ-style physical and domain constraint verification report."""

    total_rows: int = Field(description="Total evaluated rows")
    passed: bool = Field(description="True if zero violations found across all columns")
    total_violations: int = Field(description="Total violation count across all columns")
    violations_by_column: List[ColumnConstraintViolationDTO] = Field(
        default_factory=list, description="Detailed per-column violation breakdown"
    )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "passed": self.passed,
            "total_violations": self.total_violations,
            "violation_columns_count": len(self.violations_by_column),
            "violations_summary": [
                {
                    "column": v.column,
                    "violations": v.violation_count,
                    "rate": round(v.violation_rate, 4),
                    "rule": v.rule_broken,
                }
                for v in self.violations_by_column
            ],
        }


class CollinearPairDTO(BaseDTO):
    """Collinear feature pair comparison and selection decision."""
    feature_a: str = Field(description="First collinear feature candidate")
    feature_b: str = Field(description="Second collinear feature candidate")
    correlation: float = Field(description="Pairwise absolute correlation coefficient")
    kept: str = Field(description="Feature retained based on predictive signal")
    dropped: str = Field(description="Inferior feature pruned from dataset")
    selection_metric: str = Field(description="Metric used for competition (pearson, anova_f, variance_non_null)")


class DatasetLineageDTO(BaseDTO):
    """Complete provenance record binding dataset state to code commit."""
    job_id: str = Field(description="Unique experiment job identifier")
    dataset_hash: str = Field(description="SHA-256 digest of dataset raw bytes")
    git_commit: Optional[str] = Field(default=None, description="Repository commit SHA-256")
    total_rows: int = Field(ge=0, description="Total rows in dataset snapshot")
    total_columns: int = Field(ge=0, description="Total columns in dataset snapshot")
    timestamp: str = Field(description="ISO 8601 creation timestamp")


class CollinearityReportDTO(BaseDTO):
    """Report detailing multicollinearity, VIF, SVD condition number, and pruned features."""
    threshold_corr: float = Field(description="Pairwise correlation cutoff threshold")
    vif_threshold: float = Field(description="VIF cutoff threshold")
    spectral_condition_number: Optional[float] = Field(
        default=None, description="SVD condition number kappa(X) = sigma_max / sigma_min"
    )
    vif_scores: Dict[str, float] = Field(default_factory=dict, description="VIF score per feature")
    high_vif_features: List[str] = Field(default_factory=list, description="Features exceeding VIF threshold")
    collinear_pairs: List[Dict[str, Any]] = Field(default_factory=list, description="Pairwise collinear candidates")
    variance_decomposition: Optional[Dict[str, Any]] = Field(
        default=None, description="Belsley variance decomposition proportions table and condition indices"
    )
    dropped_features: List[str] = Field(default_factory=list, description="Features pruned to eliminate collinearity")
    remaining_features_count: int = Field(ge=0, description="Count of retained features")
    selection_metric: str = Field(default="competitive", description="Signal metric used for feature survival")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "spectral_condition_number": (
                round(self.spectral_condition_number, 2) if self.spectral_condition_number is not None else None
            ),
            "high_vif_count": len(self.high_vif_features),
            "dropped_count": len(self.dropped_features),
            "dropped_features": self.dropped_features,
            "remaining_features_count": self.remaining_features_count,
        }
