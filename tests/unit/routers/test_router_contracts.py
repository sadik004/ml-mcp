"""Unit tests for FastMCP routers verifying error envelopes, JSON sanitization, and dependency injection."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock
from mcp.server.fastmcp import FastMCP

from ml_mcp.routers.audit_router import register_audit_tools
from ml_mcp.routers.feature_router import register_feature_tools
from ml_mcp.routers.model_router import register_model_tools
from ml_mcp.routers.safety_router import register_safety_tools
from ml_mcp.routers.serving_router import register_serving_tools
from ml_mcp.routers.colab_router import register_colab_tools
from ml_mcp.core.errors import InsufficientSamples, CertificationRefused


@pytest.fixture
def test_mcp():
    return FastMCP("test-server")


@pytest.mark.asyncio
async def test_audit_router_error_envelope(test_mcp):
    mock_service = MagicMock()
    mock_service.audit_dataset.side_effect = InsufficientSamples("Dataset has only 3 rows, minimum required is 10")
    register_audit_tools(test_mcp, service=mock_service)

    # FastMCP tools are stored in _tool_manager._tools or as registered functions
    tool_fn = test_mcp._tool_manager._tools["ml_audit_dataset"].fn
    res = await tool_fn(csv_path="test.csv", target_column="target")

    assert res["status"] == "error"
    assert res["error_code"] == "INSUFFICIENT_SAMPLES"
    assert res["failed_step"] == "ml_audit_dataset"
    assert "test.csv" in res["inputs_provided"]["csv_path"]
    assert len(res["remediation_suggestions"]) > 0


@pytest.mark.asyncio
async def test_feature_router_error_envelope(test_mcp):
    mock_service = MagicMock()
    mock_service.synthesize_features.side_effect = ValueError("Synthetic feature error")
    register_feature_tools(test_mcp, service=mock_service)

    tool_fn = test_mcp._tool_manager._tools["ml_synthesize_features"].fn
    res = await tool_fn(csv_path="data.csv", target_column="y")

    assert res["status"] == "error"
    assert res["failed_step"] == "ml_synthesize_features"
    assert "data.csv" in res["inputs_provided"]["csv_path"]


@pytest.mark.asyncio
async def test_model_router_error_envelope(test_mcp):
    mock_service = MagicMock()
    mock_service.benchmark_models.side_effect = RuntimeError("GPU out of memory")
    register_model_tools(test_mcp, service=mock_service)

    tool_fn = test_mcp._tool_manager._tools["ml_benchmark_models"].fn
    res = await tool_fn(csv_path="data.csv", target_column="y")

    assert res["status"] == "error"
    assert res["failed_step"] == "ml_benchmark_models"
    assert "GPU out of memory" in res["error_message"]


@pytest.mark.asyncio
async def test_safety_router_error_envelope(test_mcp):
    mock_service = MagicMock()
    mock_service.certify_safety_and_decisions.side_effect = CertificationRefused("Holdout predictions required")
    register_safety_tools(test_mcp, service=mock_service)

    tool_fn = test_mcp._tool_manager._tools["ml_certify_safety_and_decisions"].fn
    res = await tool_fn(csv_path="data.csv", target_column="y")

    assert res["status"] == "error"
    assert res["error_code"] == "CERTIFICATION_REFUSED"
    assert "Holdout predictions required" in res["error_message"]


@pytest.mark.asyncio
async def test_serving_router_error_envelope(test_mcp):
    mock_service = MagicMock()
    mock_service.batch_predict.side_effect = FileNotFoundError("Model checkpoint not found")
    register_serving_tools(test_mcp, service=mock_service)

    tool_fn = test_mcp._tool_manager._tools["ml_batch_predict"].fn
    res = await tool_fn(model_path="missing.joblib", input_csv_path="in.csv", output_csv_path="out.csv")

    assert res["status"] == "error"
    assert res["failed_step"] == "ml_batch_predict"
    assert "Model checkpoint not found" in res["error_message"]


@pytest.mark.asyncio
async def test_colab_router_error_envelope(test_mcp):
    mock_service = MagicMock()
    mock_service.colab_execute.side_effect = ConnectionError("Colab runtime disconnected")
    register_colab_tools(test_mcp, service=mock_service)

    tool_fn = test_mcp._tool_manager._tools["ml_colab_execute"].fn
    res = await tool_fn(code="print(1)")

    assert res["status"] == "error"
    assert res["failed_step"] == "ml_colab_execute"
    assert "Colab runtime disconnected" in res["error_message"]


@pytest.mark.asyncio
async def test_router_json_sanitization(test_mcp):
    mock_service = MagicMock()
    mock_service.audit_dataset.return_value = {
        "score": np.float64(0.925),
        "nan_metric": float("nan"),
        "inf_metric": float("inf"),
        "array_val": np.array([1, 2, 3]),
    }
    register_audit_tools(test_mcp, service=mock_service)

    tool_fn = test_mcp._tool_manager._tools["ml_audit_dataset"].fn
    res = await tool_fn(csv_path="clean.csv")

    assert isinstance(res["score"], float)
    assert res["nan_metric"] is None
    assert res["inf_metric"] is None
    assert isinstance(res["array_val"], list)
