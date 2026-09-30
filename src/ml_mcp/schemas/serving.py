"""Serving, Batch Prediction, Drift Monitoring, and Model Packaging DTOs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field
from ml_mcp.schemas.base import BaseDTO


class BatchPredictDTO(BaseDTO):
    """High-throughput batch inference and Kaggle submission integrity report."""

    input_csv_path: str = Field(description="Source path of unlabelled test data")
    output_csv_path: str = Field(description="Destination path of generated predictions")
    rows_processed: int = Field(ge=0, description="Total records scored")
    id_column_verified: bool = Field(description="True if ID sequence strictly matches input")
    has_nan_or_inf: bool = Field(description="True if any NaN or Inf escaped into predictions")
    kaggle_submission_ready: bool = Field(
        description="True if ID matched, zero NaNs, and exact row count verified"
    )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "output_csv_path": self.output_csv_path,
            "rows_processed": self.rows_processed,
            "id_column_verified": self.id_column_verified,
            "has_nan_or_inf": self.has_nan_or_inf,
            "kaggle_submission_ready": self.kaggle_submission_ready,
        }


class DataDriftReportDTO(BaseDTO):
    """Population Stability Index (PSI) and Kolmogorov-Smirnov drift detection report."""

    psi_score: float = Field(ge=0.0, description="Overall Population Stability Index")
    drift_status: str = Field(description="no_drift (PSI<0.1), moderate_drift, severe_drift (PSI>0.25)")
    ks_p_value: float = Field(ge=0.0, le=1.0, description="Minimum KS-test p-value across numerical features")
    retraining_recommended: bool = Field(description="True if severe drift detected")
    drifted_features: List[str] = Field(default_factory=list, description="List of columns with high PSI")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "psi_score": round(self.psi_score, 4),
            "drift_status": self.drift_status,
            "ks_p_value": round(self.ks_p_value, 4),
            "retraining_recommended": self.retraining_recommended,
            "drifted_features": self.drifted_features,
        }


class ModelExportDTO(BaseDTO):
    """Artifact bundle report containing serialized model, ONNX graph, and model card."""

    joblib_path: str = Field(description="Path to atomic .joblib pipeline file")
    onnx_path: str = Field(description="Path to optimized ONNX graph")
    model_card_path: str = Field(description="Path to synthesized MODEL_CARD.md")
    p95_latency_ms: float = Field(ge=0.0, description="95th percentile inference latency in ms")
    p99_latency_ms: float = Field(ge=0.0, description="99th percentile inference latency in ms")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "joblib_path": self.joblib_path,
            "onnx_path": self.onnx_path,
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
        }
