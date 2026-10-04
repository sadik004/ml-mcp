"""Visible-fallback fraud guard tests.

Mathematical Invariant:
When an optional heavy gradient boosting engine (LightGBM / XGBoost) fails to import
or initialize, any automatic model substitution (e.g. HistGradientBoostingClassifier)
MUST be visibly surfaced to the user in a structured `warnings: list[str]` audit trail
identifying the exact substitute estimator.

Silently swallowing import errors and running an unannounced surrogate estimator
violates Cardinal Sin #4 (Silent Exception Masking & Zombie Substitutions).

Every test here must FAIL (RED) against today's codebase.
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml_mcp.server import mcp


@pytest.fixture
def fallback_dataset_csv(tmp_path: Path) -> str:
    np.random.seed(42)
    n = 60
    X = np.random.randn(n, 3)
    y = np.random.binomial(1, 0.5, size=n)
    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3"])
    df["target"] = y
    csv_file = tmp_path / "fallback_test.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def _get_tool_fn(name: str):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert name in tools, f"Tool '{name}' not registered on FastMCP server"
    return tools[name].fn


def _make_block_import(blocked_module: str):
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(f"{blocked_module}."):
            raise ImportError(f"Simulated missing dependency: {blocked_module}")
        return real_import(name, *args, **kwargs)

    return mock_import


# ------------------------------------------------------------------------------
# Fallback Site 1: ml_calibrate_probabilities (LightGBM fallback)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_warning_calibrate_probabilities_lightgbm(fallback_dataset_csv: str):
    """ml_calibrate_probabilities must record a visible warning when LightGBM is substituted."""
    tool_fn = _get_tool_fn("ml_calibrate_probabilities")
    with patch("builtins.__import__", side_effect=_make_block_import("lightgbm")):
        res = await tool_fn(
            csv_path=fallback_dataset_csv,
            target_column="target",
            model_name="lightgbm",
        )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    warnings = res.get("warnings", [])
    assert any("HistGradientBoostingClassifier" in str(w) or "fallback" in str(w).lower() for w in warnings), (
        f"Silent Fallback Violation: ml_calibrate_probabilities substituted LightGBM without "
        f"surfacing substitute estimator in warnings list! Output: {res}"
    )


# ------------------------------------------------------------------------------
# Fallback Site 2: ml_calibrate_probabilities (XGBoost fallback)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_warning_calibrate_probabilities_xgboost(fallback_dataset_csv: str):
    """ml_calibrate_probabilities must record a visible warning when XGBoost is substituted."""
    tool_fn = _get_tool_fn("ml_calibrate_probabilities")
    with patch("builtins.__import__", side_effect=_make_block_import("xgboost")):
        res = await tool_fn(
            csv_path=fallback_dataset_csv,
            target_column="target",
            model_name="xgboost",
        )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    warnings = res.get("warnings", [])
    assert any("HistGradientBoostingClassifier" in str(w) or "fallback" in str(w).lower() for w in warnings), (
        f"Silent Fallback Violation: ml_calibrate_probabilities substituted XGBoost without "
        f"surfacing substitute estimator in warnings list! Output: {res}"
    )


# ------------------------------------------------------------------------------
# Fallback Site 3: ml_tune_threshold_and_errors (LightGBM fallback)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_warning_tune_threshold_lightgbm(fallback_dataset_csv: str):
    """ml_tune_threshold_and_errors must record a visible warning when LightGBM is substituted."""
    tool_fn = _get_tool_fn("ml_tune_threshold_and_errors")
    with patch("builtins.__import__", side_effect=_make_block_import("lightgbm")):
        res = await tool_fn(
            csv_path=fallback_dataset_csv,
            target_column="target",
            model_name="lightgbm",
        )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    warnings = res.get("warnings", [])
    assert any("HistGradientBoostingClassifier" in str(w) or "fallback" in str(w).lower() for w in warnings), (
        f"Silent Fallback Violation: ml_tune_threshold_and_errors substituted LightGBM without "
        f"surfacing substitute estimator in warnings list! Output: {res}"
    )


# ------------------------------------------------------------------------------
# Fallback Site 4: ml_tune_threshold_and_errors (XGBoost fallback)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_warning_tune_threshold_xgboost(fallback_dataset_csv: str):
    """ml_tune_threshold_and_errors must record a visible warning when XGBoost is substituted."""
    tool_fn = _get_tool_fn("ml_tune_threshold_and_errors")
    with patch("builtins.__import__", side_effect=_make_block_import("xgboost")):
        res = await tool_fn(
            csv_path=fallback_dataset_csv,
            target_column="target",
            model_name="xgboost",
        )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    warnings = res.get("warnings", [])
    assert any("HistGradientBoostingClassifier" in str(w) or "fallback" in str(w).lower() for w in warnings), (
        f"Silent Fallback Violation: ml_tune_threshold_and_errors substituted XGBoost without "
        f"surfacing substitute estimator in warnings list! Output: {res}"
    )


# ------------------------------------------------------------------------------
# Fallback Site 5: ml_explain_predictions (LightGBM fallback)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_warning_explain_predictions_lightgbm(fallback_dataset_csv: str):
    """ml_explain_predictions must record a visible warning when LightGBM is substituted."""
    tool_fn = _get_tool_fn("ml_explain_predictions")
    with patch("builtins.__import__", side_effect=_make_block_import("lightgbm")):
        res = await tool_fn(
            csv_path=fallback_dataset_csv,
            target_column="target",
            model_name="lightgbm",
        )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    warnings = res.get("warnings", [])
    assert any("HistGradientBoostingClassifier" in str(w) or "fallback" in str(w).lower() for w in warnings), (
        f"Silent Fallback Violation: ml_explain_predictions substituted LightGBM without "
        f"surfacing substitute estimator in warnings list! Output: {res}"
    )


# ------------------------------------------------------------------------------
# Fallback Site 6: ml_explain_predictions (XGBoost fallback)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_warning_explain_predictions_xgboost(fallback_dataset_csv: str):
    """ml_explain_predictions must record a visible warning when XGBoost is substituted."""
    tool_fn = _get_tool_fn("ml_explain_predictions")
    with patch("builtins.__import__", side_effect=_make_block_import("xgboost")):
        res = await tool_fn(
            csv_path=fallback_dataset_csv,
            target_column="target",
            model_name="xgboost",
        )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    warnings = res.get("warnings", [])
    assert any("HistGradientBoostingClassifier" in str(w) or "fallback" in str(w).lower() for w in warnings), (
        f"Silent Fallback Violation: ml_explain_predictions substituted XGBoost without "
        f"surfacing substitute estimator in warnings list! Output: {res}"
    )
