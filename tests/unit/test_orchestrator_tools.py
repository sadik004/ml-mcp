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
    assert "executive_card" in res
    assert "sha256_hash" in res
    assert "health_score" in res
    assert "readiness_verdict" in res


@pytest.mark.asyncio
async def test_ml_prepare_feature_pipeline_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_prepare_feature_pipeline" in tools

    tool_fn = tools["ml_prepare_feature_pipeline"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", view="compact")

    assert "error" not in res
    assert "transformed_shape" in res
    assert "receipt_card" in res
    assert "transformed_dataset_path" in res


@pytest.mark.asyncio
async def test_ml_run_model_tournament_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_run_model_tournament" in tools

    tool_fn = tools["ml_run_model_tournament"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", tune_trials=2, n_splits=2, view="compact")

    assert "error" not in res
    assert "champion_architecture" in res
    assert "tournament_card" in res
    assert "validation_protocol" in res
    assert "nested" in res["validation_protocol"].lower()


@pytest.mark.asyncio
async def test_ml_certify_safety_and_decisions_tool(sample_csv):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert "ml_certify_safety_and_decisions" in tools

    tool_fn = tools["ml_certify_safety_and_decisions"].fn
    res = await tool_fn(csv_path=sample_csv, target_column="target", view="compact")

    assert "error" not in res
    assert "optimal_threshold" in res
    assert "safety_card" in res
    assert "calibrated_ece" in res
    assert "conformal_coverage_pct" in res
    assert "false_negative_reduction_pct" in res


@pytest.mark.asyncio
async def test_end_to_end_orchestrator_chaining(sample_csv):
    """Test full Phase 2 -> Phase 3 -> Phase 4 seamless artifact passing without manual path specification."""
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}

    # Phase 2: Feature Pipeline saves transformed_dataset.csv and feature_metadata.json
    p2_fn = tools["ml_prepare_feature_pipeline"].fn
    p2_res = await p2_fn(csv_path=sample_csv, target_column="target", view="compact")
    assert "transformed_dataset_path" in p2_res

    # Phase 3: Tournament runs with csv_path=None, auto-consuming Phase 2 output
    p3_fn = tools["ml_run_model_tournament"].fn
    p3_res = await p3_fn(tune_trials=2, n_splits=2, view="compact")
    assert "error" not in p3_res
    assert "champion_architecture" in p3_res

    # Phase 4: Safety & Decisions runs with csv_path=None, model_path=None, oof_path=None
    p4_fn = tools["ml_certify_safety_and_decisions"].fn
    p4_res = await p4_fn(view="compact")
    assert "error" not in p4_res
    assert "optimal_threshold" in p4_res
    assert "calibrated_ece" in p4_res
