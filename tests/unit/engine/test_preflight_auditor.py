"""Unit tests for Phase 1 Master Pre-Flight Quality Auditor."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.preflight_auditor import PreflightAuditor
from ml_mcp.schemas.audit import PreflightAuditReportDTO


def test_preflight_auditor_clean_dataset():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, n),
        "feat_2": np.random.uniform(10, 50, n),
        "target": np.random.binomial(1, 0.3, n),
    })

    auditor = PreflightAuditor()
    report = auditor.audit(df, target_column="target", task_type="classification", dataset_name="clean_test.csv")

    assert isinstance(report, PreflightAuditReportDTO)
    assert report.readiness_verdict in ("PASSED", "CONDITIONAL_PASS")
    assert report.health_score >= 70
    assert len(report.sha256_hash) == 64
    assert report.row_count == n
    assert report.column_count == 3
    assert "EXECUTIVE READINESS VERDICT" in report.executive_card


def test_preflight_auditor_detects_leakage_blocker():
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.5, n)
    # Perfect copy target leakage
    leaked_x = y.copy().astype(float)

    df = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, n),
        "direct_leaker": leaked_x,
        "target": y,
    })

    auditor = PreflightAuditor()
    report = auditor.audit(df, target_column="target", task_type="classification")

    assert report.readiness_verdict == "BLOCKED"
    assert report.health_score < 70
    assert any("LEAKAGE" in rf for rf in report.critical_red_flags)
