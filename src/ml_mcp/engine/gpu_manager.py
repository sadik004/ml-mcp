"""Hardware GPU acceleration manager with resilient CPU fallback."""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class GPUManager:
    """Detects CUDA capabilities and supplies model-tailored acceleration flags with CPU fallback."""

    def __init__(self, force_cpu: bool = False) -> None:
        self.force_cpu = force_cpu

    def is_cuda_available(self) -> bool:
        """Determines if NVIDIA CUDA acceleration is operational."""
        if self.force_cpu:
            return False
        try:
            import torch
            return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        except Exception:
            return False

    def get_hardware_status(self) -> Dict[str, Any]:
        """Provides hardware status for diagnostic and reporting tools."""
        cuda_avail = self.is_cuda_available()
        device_name = "CPU"
        device_count = 0

        if cuda_avail:
            try:
                import torch
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
            except Exception as e:
                logger.debug(f"Failed querying CUDA device details: {e}")

        return {
            "cuda_available": cuda_avail,
            "device_name": device_name,
            "device_count": device_count,
            "force_cpu": self.force_cpu,
        }

    def get_xgb_params(self) -> Dict[str, Any]:
        """Returns hardware-optimized hyperparameters for XGBoost."""
        if self.is_cuda_available():
            return {"device": "cuda", "tree_method": "hist"}
        return {"device": "cpu", "tree_method": "hist", "n_jobs": -1}

    def get_lgb_params(self) -> Dict[str, Any]:
        """Returns hardware-optimized hyperparameters for LightGBM."""
        if self.is_cuda_available():
            return {"device_type": "gpu"}
        return {"device_type": "cpu", "n_jobs": 1, "verbose": -1}

    def get_catboost_params(self) -> Dict[str, Any]:
        """Returns hardware-optimized hyperparameters for CatBoost."""
        if self.is_cuda_available():
            return {"task_type": "GPU", "verbose": 0}
        return {"task_type": "CPU", "thread_count": 1, "verbose": 0}

    def get_fallback_cpu_params(self, model_family: str) -> Dict[str, Any]:
        """Provides guaranteed CPU-safe fallback parameters upon driver failure or CUDA OOM."""
        if model_family.lower() == "xgboost":
            return {"device": "cpu", "tree_method": "hist", "n_jobs": -1}
        elif model_family.lower() == "lightgbm":
            return {"device_type": "cpu", "n_jobs": 1, "verbose": -1}
        elif model_family.lower() == "catboost":
            return {"task_type": "CPU", "thread_count": 1, "verbose": 0}
        return {"n_jobs": 1}

