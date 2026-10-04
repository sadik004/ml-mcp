"""Google Colab Cloud Compute & Notebook MCP Tool Router."""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from mcp.server.fastmcp import FastMCP

from ml_mcp.engine.error_envelope import format_error_envelope
from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.services.colab_service import ColabService

logger = logging.getLogger(__name__)


def register_colab_tools(mcp: FastMCP, service: Optional[ColabService] = None) -> None:
    """Register Google Colab Remote Compute and notebook generation tools onto FastMCP instance."""
    colab_svc = service or ColabService()

    @mcp.tool()
    async def ml_generate_colab_notebook(
        output_ipynb_path: str,
        project_name: str = "ML_Project",
        dataset_name: str = "dataset.csv",
        target_column: str = "target",
    ) -> Dict[str, Any]:
        """Generate turnkey Jupyter Notebook (.ipynb) for Google Colab GPU execution."""
        try:
            res = colab_svc.generate_colab_notebook(
                output_ipynb_path=output_ipynb_path,
                project_name=project_name,
                dataset_name=dataset_name,
                target_column=target_column,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_generate_colab_notebook", ["output_ipynb_path"])

    @mcp.tool()
    async def ml_cancel_job(job_id: str) -> Dict[str, Any]:
        """Cancel an active long-running training or tuning job gracefully."""
        try:
            res = colab_svc.cancel_job(job_id=job_id)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_cancel_job", ["job_id"])

    @mcp.tool()
    async def ml_colab_status(session: Optional[str] = None) -> Dict[str, Any]:
        """Check active Google Colab GPU hardware, VRAM, and connection health."""
        try:
            res = colab_svc.colab_status(session=session)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_colab_status", ["session"])

    @mcp.tool()
    async def ml_colab_execute(
        code: str,
        session: Optional[str] = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Execute arbitrary Python / ML code directly on the remote Google Colab GPU or CPU runtime."""
        try:
            res = colab_svc.colab_execute(code=code, session=session, timeout=timeout)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_colab_execute", ["code"])

    @mcp.tool()
    async def ml_colab_upload(
        local_path: str,
        remote_path: str = "/content/data.csv",
        session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local dataset or script to the remote Google Colab cloud filesystem."""
        try:
            res = colab_svc.colab_upload(local_path=local_path, remote_path=remote_path, session=session)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_colab_upload", ["local_path", "remote_path"])

    @mcp.tool()
    async def ml_colab_download(
        remote_path: str,
        local_path: str,
        session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download trained models, ONNX artifacts, or metrics from Google Colab to local storage."""
        try:
            res = colab_svc.colab_download(remote_path=remote_path, local_path=local_path, session=session)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_colab_download", ["remote_path", "local_path"])

    @mcp.tool()
    async def ml_colab_stop(session: Optional[str] = None) -> Dict[str, Any]:
        """Release the Google Colab GPU runtime to conserve compute units when work is done."""
        try:
            res = colab_svc.colab_stop(session=session)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_colab_stop", ["session"])

    @mcp.tool()
    async def ml_colab_provision(
        session: str = "gpu",
        accelerator: Literal["T4", "A100", "L4", "CPU"] = "T4",
        high_mem: bool = False,
    ) -> Dict[str, Any]:
        """Provision a fresh Google Colab compute runtime with requested GPU acceleration."""
        try:
            res = colab_svc.colab_provision(session=session, accelerator=accelerator, high_mem=high_mem)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_colab_provision", ["session", "accelerator"])
