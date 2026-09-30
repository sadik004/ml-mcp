"""FastMCP server initialization and diagnostic health-check tools."""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

from ml_mcp.config import get_settings
from ml_mcp.engine.json_sanitizer import sanitize_for_json

# Instantiate FastMCP server instance
mcp = FastMCP("ml.mcp")


@mcp.tool()
async def ml_ping() -> Dict[str, Any]:
    """Diagnostic health check tool for ml.mcp server and runtime environment.

    Returns:
        JSON report with Python version, platform, CUDA acceleration, and storage writability.
    """
    settings = get_settings()

    # Hardware CUDA detection
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_count = int(torch.cuda.device_count())
            cuda_device_name = str(torch.cuda.get_device_name(0))
    except (ImportError, Exception):
        pass

    storage_status = settings.validate_storage()

    report: Dict[str, Any] = {
        "status": "ok",
        "server_name": "ml.mcp",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "is_colab": settings.is_colab,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "storage_writable": storage_status.get("is_writable", False),
        "storage_path": storage_status.get("storage_path"),
        "drive_mounted": storage_status.get("is_drive_mounted", False),
    }

    return sanitize_for_json(report)


if __name__ == "__main__":
    settings = get_settings()
    # Stdio or SSE runner
    mcp.run(transport="stdio")
