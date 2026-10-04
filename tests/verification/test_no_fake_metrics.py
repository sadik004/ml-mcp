"""Hardcoded metric and synthetic guarantee fraud guard tests.

Mathematical Invariants:
1. No default fallback metric values: Calling eval dashboard without metrics must REFUSE,
   never returning hardcoded placeholder 'Score: 0.95' or 'Latency: 0.8'.
2. Zero Slack Policy: In Conformal Risk Control, if empirical_risk > target_risk,
   guarantee_satisfied MUST be False. Adding literal slack (+0.05) is statistical fraud.
3. Undefined Rates: When test set has 0 positive instances, False Negative Rate (FNR)
   is mathematically 0/0 (undefined). It must return None with a warning, NEVER 0.0.
4. Serving API Integrity: Generating serving schemas without feature names must REFUSE,
   never fabricating dummy ['f1', 'f2', 'f3'] inputs.

Every test here must FAIL (RED) against today's codebase.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from ml_mcp.server import mcp


def _get_tool_fn(name: str):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert name in tools, f"Tool '{name}' not registered on FastMCP server"
    return tools[name].fn


# ------------------------------------------------------------------------------
# 1. generate_eval_dashboard must not return hardcoded 0.95 / 0.8
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_eval_dashboard_no_hardcoded_defaults(tmp_path: Path):
    """Calling generate_eval_dashboard without metrics must not inject 0.95 or 0.8."""
    tool_fn = _get_tool_fn("ml_generate_eval_dashboard")
    res = await tool_fn(
        output_html_path=str(tmp_path / "dashboard.html"),
        project_name="untested_project",
    )

    html = res.get("dashboard_html", "")
    # Check that hardcoded placeholder 0.95 / 0.8 was NOT injected
    assert "0.95" not in html, "Fraud detected: generate_eval_dashboard returned hardcoded 'Score: 0.95'!"
    assert "0.8" not in html, "Fraud detected: generate_eval_dashboard returned hardcoded 'Latency: 0.8'!"
    # Must refuse or flag incomplete metrics
    assert res.get("status") in ("refused", "error", "warning"), (
        f"Expected refusal when metrics are omitted, got status='{res.get('status')}'"
    )


# ------------------------------------------------------------------------------
# 2. CRC: Zero slack on violated guarantee
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_crc_zero_slack_on_violated_guarantee(tmp_path: Path):
    """CRC must reject guarantee when empirical risk exceeds target risk (zero slack).

    If target_risk=0.10 and empirical_risk=0.12, guarantee_satisfied MUST be False.
    Today's code has `empirical_risk <= target_risk + 0.05` which falsely marks True!
    """
    tool_fn = _get_tool_fn("ml_conformal_risk_control")

    # Construct dataset where empirical risk falls strictly above target_risk but within old +0.05 slack zone
    np.random.seed(14)
    n = 150
    X = np.random.randn(n, 4)
    # Balanced classes
    y = np.random.binomial(1, 0.5, size=n)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    df["target"] = y

    csv_path = tmp_path / "crc_slack_test.csv"
    df.to_csv(csv_path, index=False)

    # Set tight target risk
    res = await tool_fn(
        csv_path=str(csv_path),
        target_column="target",
        loss_type="fnr",
        target_risk=0.08,
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    emp_risk = res.get("empirical_risk", 0.0)
    target_risk = res.get("target_risk", 0.08)

    # On seed 14, emp_risk ~ 0.0909 > target_risk (0.08) and <= 0.08 + 0.05 (0.13)
    assert emp_risk is not None and emp_risk > target_risk, (
        f"Test fixture condition violated: emp_risk ({emp_risk}) should exceed target_risk ({target_risk})"
    )
    assert res.get("guarantee_satisfied") is False, (
        f"Zero Slack Violation: empirical_risk ({emp_risk}) > target_risk ({target_risk}), "
        f"but guarantee_satisfied was marked True due to literal +0.05 slack in source!"
    )


# ------------------------------------------------------------------------------
# 3. CRC FNR with zero positives must return None, not 0.0
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_crc_fnr_zero_positives_returns_none(tmp_path: Path):
    """When test set has 0 positive samples, FNR is undefined (0/0).

    It must return empirical_risk = None (or raise/warn), NEVER 0.0.
    """
    tool_fn = _get_tool_fn("ml_conformal_risk_control")

    # Dataset with train & cal having positive classes, but test having 0 positives
    from ml_mcp.services.safety_service import SafetyService
    svc = SafetyService()

    np.random.seed(42)
    n = 100
    df = pd.DataFrame(np.random.randn(n, 3), columns=["f1", "f2", "f3"])
    # Put positives in train (idx 11) and cal (idx 26), ensuring test split has exactly 0 positives
    y = np.zeros(n, dtype=int)
    y[11] = 1
    y[26] = 1
    df["target"] = y

    csv_path = tmp_path / "zero_positives.csv"
    df.to_csv(csv_path, index=False)

    res = await tool_fn(
        csv_path=str(csv_path),
        target_column="target",
        loss_type="fnr",
        target_risk=0.10,
    )

    # Line 393 sets empirical_risk = None if not np.any(pos_mask)
    emp_risk = res.get("empirical_risk")
    assert emp_risk is None, (
        f"Metric Impersonation: FNR with zero positive samples returned empirical_risk=0.0! "
        f"0/0 is mathematically undefined; it must be None with an audit warning."
    )
    assert "warnings" in res and any("undefined" in str(w).lower() for w in res.get("warnings", []))


# ------------------------------------------------------------------------------
# 4. generate_serving_api without feature_names must not generate f1, f2, f3
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_serving_api_no_dummy_features(tmp_path: Path):
    """Calling generate_serving_api without feature_names must REFUSE or fail cleanly,

    never silently creating fake schemas with ['f1', 'f2', 'f3'].
    """
    tool_fn = _get_tool_fn("ml_generate_serving_api")
    out_dir = tmp_path / "serving_output"

    res = await tool_fn(
        output_dir=str(out_dir),
        model_name="test_model",
        feature_names=None,
    )

    # Inspect generated files if any
    gen_files = res.get("generated_files", [])
    for fpath in gen_files:
        content = Path(fpath).read_text(encoding="utf-8")
        assert "f1:" not in content and "f2:" not in content and "f3:" not in content, (
            f"Silent Dummy Feature Injection: {fpath} contains placeholder ['f1', 'f2', 'f3'] "
            f"instead of requiring actual dataset feature metadata!"
        )
    assert res.get("status") in ("refused", "error"), (
        f"Expected refusal for missing feature_names, got '{res.get('status')}'"
    )
