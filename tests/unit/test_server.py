"""Unit tests for FastMCP server instantiation and ml_ping diagnostic tool."""
import pytest

from ml_mcp.server import mcp, ml_ping


def test_server_instance():
    assert mcp is not None
    assert mcp.name == "ml.mcp"


@pytest.mark.asyncio
async def test_ml_ping_diagnostics():
    report = await ml_ping()

    assert report["status"] == "ok"
    assert "python_version" in report
    assert "platform" in report
    assert "cuda_available" in report
    assert "is_colab" in report
    assert "storage_writable" in report
