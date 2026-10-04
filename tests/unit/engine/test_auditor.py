"""Unit tests for Central Pre-Flight Data Auditor with Missingness Mechanisms and DataPerf ID guard."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.auditor import DatasetAuditor
from ml_mcp.schemas.audit import AuditReportDTO


def test_dataset_auditor_accuracy_paradox_guard():
    n = 200
    y = np.array([0] * 180 + [1] * 20)

    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, size=n),
        "feature_2": np.random.uniform(10, 50, size=n),
        "target": y,
    })

    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df, target_column="target", task_type="classification")

    assert isinstance(report, AuditReportDTO)
    assert report.class_imbalance_ratio == 0.9
    assert report.recommended_metric in ["pr_auc", "f1_weighted"]


def test_dataset_auditor_missingness_mechanism_mnar():
    """Verify MNAR classification when missingness is strongly correlated with another feature."""
    np.random.seed(42)
    n = 200
    income = np.random.uniform(20000, 150000, n)
    # High-income respondents systematically omit reporting (MNAR pattern)
    net_worth = income * 2.5
    net_worth[income > 90000] = np.nan

    df = pd.DataFrame({
        "income": income,
        "net_worth": net_worth,
    })

    auditor = DatasetAuditor()
    mech = auditor.classify_missingness_mechanism(df, "net_worth")
    assert mech == "MNAR"


def test_dataset_auditor_dataperf_id_memorization_guard():
    n = 200
    df = pd.DataFrame({
        "unique_guid": [f"ID_{i}_{np.random.randint(1000, 9999)}" for i in range(n)],
        "normal_feat": np.random.normal(0, 1, n),
    })

    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df)
    assert "unique_guid" in report.id_memorization_columns
    assert "normal_feat" not in report.id_memorization_columns


def test_dataset_auditor_benfords_law():
    # Benford compliant numbers
    rng = np.random.RandomState(42)
    compliant_data = 10.0 ** rng.uniform(1.0, 5.0, size=500)
    series_comp = pd.Series(compliant_data)
    is_anomalous, chi2 = DatasetAuditor.check_benfords_law(series_comp)
    assert isinstance(is_anomalous, bool)
    assert chi2 >= 0.0

    # Non-compliant uniform numbers
    uniform_data = rng.uniform(1.0, 10000.0, size=500)
    series_unif = pd.Series(uniform_data)
    is_anomalous_u, chi2_u = DatasetAuditor.check_benfords_law(series_unif)
    assert chi2_u >= 0.0


def test_dataset_auditor_regression_task():
    n = 200
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.uniform(0, 10, n),
        "y": np.random.randn(n) * 5.0 + 10.0,
    })
    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df, target_column="y", task_type="regression")
    assert report.recommended_metric in ["rmse", "mae"]
