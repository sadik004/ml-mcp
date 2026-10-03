"""Unit tests for Central Pre-Flight Data Auditor."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.auditor import DatasetAuditor
from ml_mcp.schemas.audit import AuditReportDTO


def test_dataset_auditor_accuracy_paradox_guard():
    # Severe class imbalance: 90% class 0, 10% class 1
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
    # Accuracy Paradox: accuracy is strictly banned! Auto-switches to pr_auc
    assert report.recommended_metric == "pr_auc"


def test_dataset_auditor_balanced_class():
    # Balanced class: 50% class 0, 50% class 1
    n = 100
    y = np.array([0] * 50 + [1] * 50)

    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, size=n),
        "target": y,
    })

    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df, target_column="target", task_type="classification")

    assert report.class_imbalance_ratio == 0.5
    assert report.recommended_metric == "accuracy"


def test_dataset_auditor_group_and_temporal_guards():
    # Customer entity with repeated visits and datetime timestamps
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    customer_ids = [f"cust_{i % 10}" for i in range(100)]  # 10 customers repeating

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "transaction_date": dates,
        "amount": np.random.uniform(10, 500, size=100),
        "is_fraud": np.random.binomial(1, 0.05, size=100),
    })

    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df, target_column="is_fraud", task_type="classification")

    # Group Guard detects customer_id
    assert report.group_column_candidate == "customer_id"
    assert report.recommended_split_strategy == "group_kfold"
    # Temporal Guard detects transaction_date
    assert report.has_temporal_order is True


def test_dataset_auditor_entropy_id_memorization_guard():
    # High-entropy synthetic primary key / GUID column that would cause memorization
    n = 150
    user_uuids = [f"uuid_record_{i}_{np.random.randint(1000, 9999)}" for i in range(n)]
    regular_feature = np.random.choice(["A", "B", "C"], size=n)
    target = np.random.binomial(1, 0.5, size=n)

    df = pd.DataFrame({
        "session_hash_key": user_uuids,
        "category": regular_feature,
        "churn": target,
    })

    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df, target_column="churn", task_type="classification")

    # ID Memorization Guard flags 100% unique high-entropy column
    assert "session_hash_key" in report.id_memorization_columns
    assert "category" not in report.id_memorization_columns


def test_dataset_auditor_target_skewness_regression():
    # Exponential right-skewed regression target
    np.random.seed(42)
    n = 200
    skewed_y = np.exp(np.random.normal(3.0, 1.2, size=n))

    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, size=n),
        "house_price": skewed_y,
    })

    auditor = DatasetAuditor()
    report = auditor.audit_dataset(df, target_column="house_price", task_type="regression")

    assert report.target_skewness is not None
    assert report.target_skewness > 1.0  # Captures right-skewness
    assert report.recommended_split_strategy == "kfold"
