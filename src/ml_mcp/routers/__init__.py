"""Routers package for FastMCP tool registration in ML-MCP Clean Architecture."""
from mcp.server.fastmcp import FastMCP

from ml_mcp.routers.audit_router import register_audit_tools
from ml_mcp.routers.colab_router import register_colab_tools
from ml_mcp.routers.feature_router import register_feature_tools
from ml_mcp.routers.model_router import register_model_tools
from ml_mcp.routers.safety_router import register_safety_tools
from ml_mcp.routers.serving_router import register_serving_tools


def register_all_routers(mcp: FastMCP) -> None:
    """Register all 6 domain routers (41 tools) onto the given FastMCP instance."""
    register_audit_tools(mcp)
    register_feature_tools(mcp)
    register_model_tools(mcp)
    register_safety_tools(mcp)
    register_serving_tools(mcp)
    register_colab_tools(mcp)


__all__ = [
    "register_all_routers",
    "register_audit_tools",
    "register_feature_tools",
    "register_model_tools",
    "register_safety_tools",
    "register_serving_tools",
    "register_colab_tools",
]
