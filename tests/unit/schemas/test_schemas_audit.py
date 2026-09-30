"""Unit tests for Audit and Data Hygiene DTOs."""
import pytest
from pydantic import ValidationError

from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.schemas.audit import AuditReportDTO, TargetLeakageReportDTO, TextFeatureReportDTO


def test_audit_report_dto_valid():
    dto = AuditReportDTO(
        row_count=1000,
        column_count=12,
        missing_cells=50,
        missing_ratio=0.0041,
        infinite_count=0,
        sentinel_count=5,
        duplicate_rows=0,
        class_imbalance_ratio=0.88,
        recommended_metric="pr_auc",
        group_column_candidate="customer_id",
        has_temporal_order=True,
        column_details={"age": {"dtype": "int64", "missing": 0}},
    )

    assert dto.row_count == 1000
    assert dto.recommended_metric == "pr_auc"
    assert dto.view == "compact"

    # Compact representation test
    compact = dto.to_compact()
    assert "column_details" not in compact
    assert compact["row_count"] == 1000
    assert compact["recommended_metric"] == "pr_auc"
    assert len(str(compact)) < 400

    # JSON sanitizer compatibility
    sanitized = sanitize_for_json(dto.model_dump())
    assert isinstance(sanitized, dict)


def test_audit_report_dto_extra_forbidden():
    with pytest.raises(ValidationError):
        AuditReportDTO(
            row_count=100,
            column_count=5,
            missing_cells=0,
            missing_ratio=0.0,
            infinite_count=0,
            sentinel_count=0,
            duplicate_rows=0,
            class_imbalance_ratio=0.5,
            recommended_metric="accuracy",
            unexpected_field="hack",
        )


def test_target_leakage_dto():
    dto = TargetLeakageReportDTO(
        target_column="churn",
        leaked_features=["post_churn_survey_rating", "cancellation_date"],
        correlation_matrix={"post_churn_survey_rating": 0.98},
        mutual_info_scores={"post_churn_survey_rating": 0.92},
        has_critical_leakage=True,
    )

    assert dto.has_critical_leakage is True
    compact = dto.to_compact()
    assert compact["has_critical_leakage"] is True
    assert "post_churn_survey_rating" in compact["leaked_features"]


def test_text_feature_dto():
    dto = TextFeatureReportDTO(
        text_columns=["customer_review", "support_chat"],
        avg_char_lengths={"customer_review": 142.5, "support_chat": 85.0},
        unique_token_counts={"customer_review": 1520, "support_chat": 890},
        recommended_strategy="tfidf_sublinear",
    )

    assert len(dto.text_columns) == 2
    assert dto.recommended_strategy == "tfidf_sublinear"
