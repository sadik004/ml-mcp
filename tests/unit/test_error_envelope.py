"""Unit tests for self-healing structured error envelopes and fuzzy suggestions."""
import pytest

from ml_mcp.engine.error_envelope import format_error_envelope, suggest_close_matches


def test_suggest_close_matches():
    candidates = ["target_churn", "monthly_charges", "total_payments", "customer_id"]
    
    # Close match
    suggestions = suggest_close_matches("target_chrun", candidates)
    assert "target_churn" in suggestions

    suggestions_charges = suggest_close_matches("mothly_charge", candidates)
    assert "monthly_charges" in suggestions_charges

    # Completely unrelated
    assert suggest_close_matches("xylophone_banana", candidates) == []


def test_format_error_envelope_basic():
    envelope = format_error_envelope(
        error_type="ValidationError",
        message="Column 'churn_target' not found in dataset.",
        details={"attempted_column": "churn_target"},
        candidates=["target_churn", "customer_age"],
    )

    assert envelope["status"] == "error"
    assert envelope["error_type"] == "ValidationError"
    assert "Column 'churn_target' not found" in envelope["message"]
    assert "target_churn" in envelope["suggestions"]
    assert envelope["retryable"] is True


def test_format_error_envelope_exception_handling():
    try:
        raise ValueError("Invalid hyperparameter: learning_rate must be positive")
    except ValueError as exc:
        envelope = format_error_envelope(
            error_type="ValueError",
            message=str(exc),
            exception=exc,
            retryable=False,
        )

    assert envelope["status"] == "error"
    assert envelope["error_type"] == "ValueError"
    assert "learning_rate must be positive" in envelope["message"]
    assert envelope["retryable"] is False
