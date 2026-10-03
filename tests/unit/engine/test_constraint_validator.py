"""Unit tests for Constraint Validator with Hubert Medcouple adjusted boxplot."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.constraint_validator import ConstraintValidator
from ml_mcp.schemas.audit import ConstraintValidationReportDTO


def test_constraint_validator_explicit_rules():
    df = pd.DataFrame({
        "age": [25, 40, -5, 180, 35],
        "price": [10.0, 20.0, -1.0, 15.0, 50.0],
        "status": ["active", "pending", "corrupted", "active", "active"],
        "rating": [4.5, 3.8, np.nan, 4.9, 5.0],
    })

    constraints = {
        "age": {"min": 0, "max": 120, "non_negative": True},
        "price": {"non_negative": True},
        "status": {"allowed_values": ["active", "pending", "closed"]},
        "rating": {"is_complete": True},
    }

    validator = ConstraintValidator(constraints=constraints)
    report = validator.validate_constraints(df)

    assert isinstance(report, ConstraintValidationReportDTO)
    assert report.passed is False
    assert report.total_violations >= 4


def test_constraint_validator_medcouple_skewed_data():
    """Verify Medcouple adjusted boxplot prevents false-positive outlier alarms on right-skewed log-normal data."""
    np.random.seed(42)
    n = 200
    # Log-normal distribution (strongly right-skewed, like salaries or web latency)
    skewed_salaries = np.random.lognormal(mean=3.0, sigma=0.8, size=n)
    df = pd.DataFrame({"salary": skewed_salaries})

    validator = ConstraintValidator()
    mc = validator.calculate_medcouple(df["salary"])
    # Medcouple for right-skewed distribution is strictly positive
    assert mc > 0.0

    report = validator.validate_constraints(df)
    # Adjusted boxplot should report minimal or 0 false violations
    assert report.total_violations < 10
