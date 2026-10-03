"""Unit tests for the 4 Phase Master Orchestrator MCP Tools."""
import pytest
import numpy as np
import pandas as pd

from ml_mcp.server import mcp


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.uniform(10, 100, n),
        "target": np.random.choice([0, 1], size=n),
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.mark.asyncio
async def test_ml_preflight_audit_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_preflight_audit" in tools
    
    tool_fn = tools["ml_preflight_audit"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", view="compact")
    
    assert "error" not in res
    assert "executive_card" in res or "lineage_sha256" in res


@pytest.mark.asyncio
async def test_ml_prepare_feature_pipeline_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_prepare_feature_pipeline" in tools
    
    tool_fn = tools["ml_prepare_feature_pipeline"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", view="compact")
    
    assert "error" not in res
    assert "transformed_shape" in res
    assert "receipt_card" in res


@pytest.mark.asyncio
async def test_ml_run_model_tournament_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_run_model_tournament" in tools
    
    tool_fn = tools["ml_run_model_tournament"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", tune_trials=2, n_splits=2, view="compact")
    
    assert "error" not in res
    assert "champion_model" in res or "tournament_card" in res


@pytest.mark.asyncio
async def test_ml_certify_safety_and_decisions_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_certify_safety_and_decisions" in tools
    
    tool_fn = tools["ml_certify_safety_and_decisions"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", view="compact")
    
    assert "error" not in res
    assert "optimal_threshold" in res
    assert "safety_card" in res
