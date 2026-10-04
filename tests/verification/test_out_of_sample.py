"""Permuted-label (in-sample) fraud guard tests.

Mathematical Invariant:
When the target y is independently permuted (pure noise with zero real feature signal),
any honest out-of-sample / out-of-fold evaluation metric (AUC, accuracy, F-beta)
MUST collapse to chance level (e.g. AUC or Accuracy in [0.35, 0.65]).

If an implementation fits a model and evaluates in-sample, the overfitted model
will memorize the noise and falsely report high scores (AUC > 0.85, Accuracy > 0.90, score=1.0).
Every test here must FAIL (RED) against the current codebase before Phase P3 remediation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from ml_mcp.server import mcp


@pytest.fixture
def permuted_dataset_csv(tmp_path: Path) -> str:
    """Fixture producing a dataset with permuted target labels (pure noise)."""
    np.random.seed(42)
    n = 120
    X = np.random.randn(n, 6)
    # y is purely random binomial, independent of X
    y = np.random.binomial(1, 0.5, size=n)

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
    df["target"] = y

    csv_path = tmp_path / "permuted_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def _get_tool_fn(name: str):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert name in tools, f"Tool '{name}' not registered on FastMCP server"
    return tools[name].fn


# ------------------------------------------------------------------------------
# 1. ml_calibrate_probabilities
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permuted_labels_calibrate_probabilities(permuted_dataset_csv: str):
    """Calibrate probabilities must evaluate out-of-sample.

    On permuted noise labels, post-calibration Brier must reflect chance (>= 0.20)
    and must NEVER claim is_well_calibrated=True.
    """
    tool_fn = _get_tool_fn("ml_calibrate_probabilities")
    res = await tool_fn(
        csv_path=permuted_dataset_csv,
        target_column="target",
        model_name="lightgbm",
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    # Fraud check: Calibrator must evaluate on honest out-of-fold predictions
    # Today's code evaluates in-sample and lacks oof_brier / oof_ece fields
    assert "oof_brier" in res or res.get("evaluation_mode") == "out_of_fold", (
        f"In-Sample Evaluation Fraud: ml_calibrate_probabilities evaluated in-sample on training set! "
        f"Missing 'oof_brier' out-of-fold evaluation metric. Output: {res}"
    )
    # On permuted noise labels, out-of-fold Brier score must be near chance level (~0.25)
    oof_brier = res.get("oof_brier", res.get("post_brier_score", 0.0))
    assert oof_brier >= 0.22, (
        f"Overfit fraud: reported Brier={oof_brier} on permuted labels is suspiciously low (expected >= 0.22)"
    )


# ------------------------------------------------------------------------------
# 2. ml_tune_threshold_and_errors
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permuted_labels_tune_threshold_and_errors(permuted_dataset_csv: str):
    """Threshold tuning must evaluate on held-out predictions.

    On permuted noise labels, F-beta score cannot exceed chance level (<= 0.65).
    In-sample fitting achieves > 0.85.
    """
    tool_fn = _get_tool_fn("ml_tune_threshold_and_errors")
    res = await tool_fn(
        csv_path=permuted_dataset_csv,
        target_column="target",
        criterion="f_beta",
        beta=1.0,
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    f_beta = res.get("f_beta_score", 0.0)
    assert f_beta <= 0.65, (
        f"Overfit fraud in ml_tune_threshold_and_errors: reported F-beta={f_beta:.4f} "
        f"on permuted noise labels. Expected chance level <= 0.65 via out-of-fold evaluation."
    )


# ------------------------------------------------------------------------------
# 3. ml_stress_test_and_fairness
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permuted_labels_stress_test_and_fairness(permuted_dataset_csv: str):
    """Stress testing must report baseline performance on an untouched holdout set.

    On permuted noise labels, baseline accuracy must be chance level (0.35 <= acc <= 0.65).
    In-sample fitting achieves >= 0.90.
    """
    tool_fn = _get_tool_fn("ml_stress_test_and_fairness")
    res = await tool_fn(
        csv_path=permuted_dataset_csv,
        target_column="target",
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    baseline_score = res.get("baseline_score", 1.0)
    assert 0.35 <= baseline_score <= 0.65, (
        f"Overfit fraud in ml_stress_test_and_fairness: baseline score={baseline_score:.4f} "
        f"on permuted labels. Expected out-of-sample chance accuracy in [0.35, 0.65]."
    )


# ------------------------------------------------------------------------------
# 4. ml_export_and_document
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permuted_labels_export_and_document(permuted_dataset_csv: str, tmp_path: Path):
    """Export and document must compute real held-out metrics, never hardcoding score: 1.0.

    On permuted labels, any evaluated validation score must be chance level (<= 0.65).
    """
    tool_fn = _get_tool_fn("ml_export_and_document")
    out_dir = tmp_path / "bundle_export"
    res = await tool_fn(
        csv_path=permuted_dataset_csv,
        target_column="target",
        output_dir=str(out_dir),
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    score = res.get("score")
    # Must NOT be hardcoded 1.0, and must be <= 0.65 on permuted data
    assert score != 1.0, "Fraud in ml_export_and_document: returned hardcoded placeholder 'score': 1.0"
    assert score is not None and score <= 0.65, (
        f"Fraud in ml_export_and_document: reported score={score} on permuted labels, "
        f"expected held-out chance score <= 0.65."
    )


# ------------------------------------------------------------------------------
# 5. ml_conformal_risk_control
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permuted_labels_conformal_risk_control(permuted_dataset_csv: str):
    """CRC must enforce target risk strictly with ZERO slack.

    When empirical risk exceeds target risk, guarantee_satisfied MUST be False.
    """
    tool_fn = _get_tool_fn("ml_conformal_risk_control")
    # Request strict FNR target risk of 0.05 on permuted noise
    res = await tool_fn(
        csv_path=permuted_dataset_csv,
        target_column="target",
        loss_type="fnr",
        target_risk=0.05,
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    # CRC must explicitly document and enforce honest out_of_sample evaluation
    assert res.get("evaluation_mode") == "out_of_sample", (
        f"In-Sample Evaluation Fraud: ml_conformal_risk_control missing explicit 'evaluation_mode': 'out_of_sample'! Output: {res}"
    )
    emp_risk = res.get("empirical_risk")
    satisfied = res.get("guarantee_satisfied", False)
    if emp_risk is not None:
        assert not (emp_risk > 0.05 and satisfied), (
            f"Zero Slack Violation: empirical_risk={emp_risk} > target_risk=0.05 with guarantee_satisfied=True"
        )
    else:
        assert not satisfied, "Guarantee cannot be satisfied when empirical risk is None"



# ------------------------------------------------------------------------------
# 6. ml_certify_safety_and_decisions
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permuted_labels_certify_safety_and_decisions(permuted_dataset_csv: str):
    """Certification must REFUSE when honest out-of-fold predictions are missing.

    In-sample probabilities must never be passed off as certified.
    """
    tool_fn = _get_tool_fn("ml_certify_safety_and_decisions")
    # Calling without oof_path and without allow_in_sample_diagnostic
    res = await tool_fn(
        csv_path=permuted_dataset_csv,
        target_column="target",
        train_ephemeral=True,
        allow_in_sample_diagnostic=False,
    )

    # Must refuse certification or return refusal envelope
    status = res.get("certification_status", "")
    assert status == "refused" or "error" in res, (
        f"Fraud in ml_certify_safety_and_decisions: granted certification '{status}' "
        f"without out-of-fold predictions on ephemeral model."
    )
