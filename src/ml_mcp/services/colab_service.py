"""Colab Cloud Compute, Job Management & Notebook Generation Service."""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from ml_mcp.colab_bridge import ColabCloudRunner
from ml_mcp.engine.colab_generator import ColabNotebookGenerator
from ml_mcp.services.job_registry import get_job_registry

logger = logging.getLogger(__name__)


class ColabService:
    """Orchestrates Google Colab remote runtime interactions, jobs, and notebook synthesis."""

    def __init__(self, runner: Optional[ColabCloudRunner] = None) -> None:
        self._runner = runner

    @property
    def runner(self) -> ColabCloudRunner:
        """Lazy initialization of ColabCloudRunner on first use."""
        if self._runner is None:
            self._runner = ColabCloudRunner()
        return self._runner

    def generate_colab_notebook(
        self,
        output_ipynb_path: str,
        project_name: str = "ML_Project",
        dataset_name: str = "dataset.csv",
        target_column: str = "target",
    ) -> Dict[str, Any]:
        """Generate turnkey Jupyter Notebook (.ipynb) for Google Colab GPU execution."""
        generator = ColabNotebookGenerator()
        dto = generator.generate_notebook(
            output_ipynb_path,
            project_name=project_name,
            dataset_name=dataset_name,
            target_column=target_column,
        )
        return dto.to_compact()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel an active long-running training or tuning job gracefully."""
        return get_job_registry().cancel_job(job_id)

    def colab_status(self, session: Optional[str] = None) -> Dict[str, Any]:
        """Check active Google Colab GPU hardware, VRAM, and connection health."""
        return self.runner.get_status(session_name=session)

    def colab_execute(
        self,
        code: str,
        session: Optional[str] = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Execute arbitrary Python / ML code directly on the remote Google Colab runtime."""
        return self.runner.execute_code(code, session=session, timeout=timeout)

    def colab_upload(
        self,
        local_path: str,
        remote_path: str = "/content/data.csv",
        session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local dataset or script to the remote Google Colab cloud filesystem."""
        return self.runner.upload_file(local_path, remote_path, session=session)

    def colab_download(
        self,
        remote_path: str,
        local_path: str,
        session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download trained models, ONNX artifacts, or metrics from Google Colab."""
        return self.runner.download_file(remote_path, local_path, session=session)

    def colab_stop(self, session: Optional[str] = None) -> Dict[str, Any]:
        """Release the Google Colab GPU runtime to conserve compute units."""
        return self.runner.stop_session(session=session)

    def colab_provision(
        self,
        session: str = "gpu",
        accelerator: Literal["T4", "A100", "L4", "CPU"] = "T4",
        high_mem: bool = False,
    ) -> Dict[str, Any]:
        """Provision a fresh Google Colab compute runtime with requested GPU acceleration."""
        return self.runner.provision_session(
            session_name=session, accelerator=accelerator, high_mem=high_mem
        )
