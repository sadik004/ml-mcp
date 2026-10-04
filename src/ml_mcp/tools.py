"""FastMCP Tool Registrations delegating to Domain Routers (3-Tier Clean Architecture)."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP

from ml_mcp.routers import (
    register_all_routers,
    register_audit_tools,
    register_colab_tools,
    register_feature_tools,
    register_model_tools,
    register_safety_tools,
    register_serving_tools,
)
from ml_mcp.services.base import persist_processed_dataframe

logger = logging.getLogger(__name__)


def _persist_processed_dataframe(
    df: pd.DataFrame,
    source_path: str,
    suffix: str,
    output_path: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper for persisting processed DataFrame."""
    return persist_processed_dataframe(
        df=df,
        source_path=source_path,
        suffix=suffix,
        output_path=output_path,
    )


def register_all_tools(mcp: FastMCP) -> None:
    """Register all domain routers onto FastMCP instance."""
    register_all_routers(mcp)


__all__ = [
    "register_all_tools",
    "register_audit_tools",
    "register_feature_tools",
    "register_model_tools",
    "register_safety_tools",
    "register_serving_tools",
    "register_colab_tools",
    "_persist_processed_dataframe",
]
