"""Unit tests for GPU Manager and hardware acceleration fallback."""
import pytest

from ml_mcp.engine.gpu_manager import GPUManager


def test_gpu_manager_detection():
    manager = GPUManager()
    status = manager.get_hardware_status()

    assert "cuda_available" in status
    assert "device_name" in status
    assert "device_count" in status
    assert isinstance(status["cuda_available"], bool)


def test_gpu_manager_model_params_cpu():
    # Force CPU configuration
    manager = GPUManager(force_cpu=True)

    xgb_params = manager.get_xgb_params()
    assert xgb_params["device"] == "cpu"
    assert xgb_params["tree_method"] == "hist"

    lgb_params = manager.get_lgb_params()
    assert lgb_params["device_type"] == "cpu"

    cb_params = manager.get_catboost_params()
    assert cb_params["task_type"] == "CPU"


def test_gpu_manager_resilient_fallback():
    manager = GPUManager()
    # Emulate GPU failure
    fallback_params = manager.get_fallback_cpu_params("xgboost")
    assert fallback_params["device"] == "cpu"
    assert fallback_params["n_jobs"] == -1
