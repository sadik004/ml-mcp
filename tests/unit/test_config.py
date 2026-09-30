"""Unit tests for configuration manager and Colab environment detection."""
import os
import sys
from pathlib import Path
import pytest

from ml_mcp.config import Settings, get_settings


def test_default_settings():
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
    assert isinstance(settings.is_colab, bool)
    assert settings.max_json_depth == 64


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ML_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("ML_MCP_PORT", "9090")
    monkeypatch.setenv("ML_MCP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ML_MCP_STORAGE_BACKEND", "local")

    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 9090
    assert settings.log_level == "DEBUG"
    assert settings.storage_backend == "local"


def test_colab_detection(monkeypatch):
    # Simulate Colab environment
    monkeypatch.setitem(sys.modules, "google.colab", object())
    settings = Settings()
    assert settings.is_colab is True

    # Simulate Non-Colab environment
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    settings_non_colab = Settings()
    # unless running on colab VM, it should be False
    assert settings_non_colab.is_colab is False


def test_validate_storage_path(tmp_path):
    settings = Settings(drive_root=str(tmp_path / "ml_mcp"))
    status = settings.validate_storage()
    assert status["is_writable"] is True
    assert Path(status["storage_path"]).exists()
