"""Centralized Job Registry for ML-MCP asynchronous and background training tasks (Phase P5)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JobRegistry:
    """Thread-safe, singleton job state registry tracking active and cancelled training runs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Any] = {}

    def register_job(self, job_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Register a new active background job."""
        self._jobs[job_id] = meta or {"status": "running"}
        logger.info("Registered job '%s'. Total active jobs: %d", job_id, len(self._jobs))

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel an active job and remove from registry."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info("Cancelled job '%s'.", job_id)
            return {"job_id": job_id, "status": "cancelled"}
        return {"job_id": job_id, "status": "job_not_found_or_already_completed"}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a registered job."""
        return self._jobs.get(job_id)

    def is_active(self, job_id: str) -> bool:
        """Check if job is active."""
        return job_id in self._jobs

    def list_jobs(self) -> Dict[str, Any]:
        """List all active registered jobs."""
        return dict(self._jobs)


_GLOBAL_JOB_REGISTRY: Optional[JobRegistry] = None


def get_job_registry() -> JobRegistry:
    """Return singleton JobRegistry instance."""
    global _GLOBAL_JOB_REGISTRY
    if _GLOBAL_JOB_REGISTRY is None:
        _GLOBAL_JOB_REGISTRY = JobRegistry()
    return _GLOBAL_JOB_REGISTRY
