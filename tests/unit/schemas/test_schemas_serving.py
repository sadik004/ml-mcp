"""Unit tests for Serving, Inference, Drift and Model Card DTOs."""
import pytest
from pydantic import ValidationError

from ml_mcp.schemas.serving import (
    BatchPredictDTO,
    DataDriftReportDTO,
    ModelExportDTO,
)


def test_batch_predict_dto():
    dto = BatchPredictDTO(
        input_csv_path="/content/test.csv",
        output_csv_path="/content/submission.csv",
        rows_processed=20000,
        id_column_verified=True,
        has_nan_or_inf=False,
        kaggle_submission_ready=True,
    )

    assert dto.kaggle_submission_ready is True
    assert dto.rows_processed == 20000


def test_batch_predict_dto_invalid_kaggle():
    dto = BatchPredictDTO(
        input_csv_path="/content/test.csv",
        output_csv_path="/content/submission.csv",
        rows_processed=20000,
        id_column_verified=False,
        has_nan_or_inf=True,
        kaggle_submission_ready=False,
    )
    assert dto.kaggle_submission_ready is False


def test_data_drift_report_dto():
    dto = DataDriftReportDTO(
        psi_score=0.082,
        drift_status="no_drift",
        ks_p_value=0.24,
        retraining_recommended=False,
        drifted_features=[],
    )

    assert dto.drift_status == "no_drift"
    assert dto.retraining_recommended is False


def test_model_export_dto():
    dto = ModelExportDTO(
        joblib_path="/content/drive/MyDrive/ml_mcp/models/champion.joblib",
        onnx_path="/content/drive/MyDrive/ml_mcp/models/champion.onnx",
        model_card_path="/content/drive/MyDrive/ml_mcp/models/MODEL_CARD.md",
        p95_latency_ms=1.2,
        p99_latency_ms=2.4,
    )

    assert dto.p95_latency_ms == 1.2
    assert "MODEL_CARD.md" in dto.model_card_path
