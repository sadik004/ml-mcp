"""Configuration management and Colab cloud environment validation."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def is_running_in_colab() -> bool:
    """Detects whether current execution environment is Google Colab."""
    if "google.colab" in sys.modules:
        return True
    if os.environ.get("COLAB_GPU") is not None:
        return True
    if Path("/content").exists() and Path("/root/.colab").exists():
        return True
    return False


class Settings(BaseSettings):
    """Central settings for ml.mcp server and Colab storage backends."""

    model_config = SettingsConfigDict(
        env_prefix="ML_MCP_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="FastMCP / SSE server bind host")
    port: int = Field(default=8000, description="FastMCP / SSE server bind port")
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")

    storage_backend: Literal["local", "drive"] = Field(
        default="drive" if is_running_in_colab() else "local",
        description="Persistent artifact and checkpoint storage destination",
    )
    drive_root: str = Field(
        default="/content/drive/MyDrive/ml_mcp" if is_running_in_colab() else ".artifacts",
        description="Root path for checkpoint persistence and reports",
    )

    max_json_depth: int = Field(
        default=64,
        description="Safety threshold for JSON recursive sanitization",
    )

    @property
    def is_colab(self) -> bool:
        """Dynamic check of Colab environment."""
        return is_running_in_colab()

    def validate_storage(self) -> Dict[str, Any]:
        """Validates if target storage path exists and is writable.

        Returns:
            Dictionary with path status, writability, and Drive mount indicators.
        """
        path = Path(self.drive_root).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_probe"
            test_file.write_text("ok")
            test_file.unlink(missing_ok=True)
            is_writable = True
        except Exception:
            is_writable = False

        drive_mounted = False
        if self.is_colab:
            drive_mounted = Path("/content/drive/MyDrive").exists()

        return {
            "storage_path": str(path),
            "is_writable": is_writable,
            "is_colab": self.is_colab,
            "is_drive_mounted": drive_mounted,
        }


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Returns singleton settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
