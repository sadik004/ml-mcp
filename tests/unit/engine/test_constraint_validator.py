"""Unit tests for Amazon Deequ-style Constraint Validator."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.constraint_validator import ConstraintValidator
from ml_mcp.schemas.audit import ConstraintValidationReportDTO


def test_constraint_validator_explicit_rules():
    df = pd.DataFrame({
        "age": [25, 40, -5, 180, 35],       # -5 violates min/non_negative, 180 violates max
        "price": [10.0, 20.0, -1.0, 15.0, 50.0],  # -1 violates non_negative
        "status": ["active", "pending", "corrupted", "active", "active"], # "corrupted" violates allowed
        "rating": [4.5, 3.8, np.nan, 4.9, 5.0],    # np.nan violates is_complete
    })

    constraints = {
        "age": {"min": 0, "max": 120, "non_negative": True},
        "price": {"non_negative": True},
        "status": {"allowed_values": ["active", "pending", "closed"]},
        "rating": {"is_complete": True},
    }

    validator = ConstraintValidator()
    report = validator.validate_constraints(df, constraints=constraints)

    assert isinstance(report, ConstraintValidationReportDTO)
    assert report.passed is False
    assert report.total_violations > 0

    cols_with_violations = [v.column for v in report.violations_by_column]
    assert "age" in cols_with_violations
    assert "price" in cols_with_violations
    assert "status" in cols_with_violations
    assert "rating" in cols_with_violations


def test_constraint_validator_iqr_automated():
    np.random.seed(42)
    # Generate 100 normal observations
    vals = np.random.normal(50, 5, 100).tolist()
    # Injected extreme physical outliers: 500.0 and -200.0
    vals.append(500.0)
    vals.append(-200.0)

    df = pd.DataFrame({"sensor_reading": vals})

    validator = ConstraintValidator()
    report = validator.validate_constraints(df)

    assert report.passed is False
    assert report.total_violations >= 2
    assert report.violations_by_column[0].column == "sensor_reading"
    assert "iqr_range_violation" in report.violations_by_column[0].rule_broken
