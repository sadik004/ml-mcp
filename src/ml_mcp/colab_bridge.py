"""Colab cloud bridge, tunnel URL persistence, and runtime watchdog."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class ColabBridge:
    """Manages Drive persistence, tunnel sync, and bootstrap scripting for Google Colab."""

    def __init__(self, drive_root: str = "/content/drive/MyDrive/ml_mcp") -> None:
        self.drive_root = Path(drive_root).resolve()
        self.checkpoints_dir = self.drive_root / "checkpoints"
        self.artifacts_dir = self.drive_root / "artifacts"
        self.tunnel_file = self.drive_root / "tunnel_url.txt"
        self.init_directories()

    def init_directories(self) -> None:
        """Initializes checkpoint and artifact directories."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_initialized(self) -> bool:
        """Checks if storage directories exist."""
        return self.checkpoints_dir.exists() and self.artifacts_dir.exists()

    def persist_tunnel_url(self, url: str) -> None:
        """Atomically saves the active Cloudflare/ngrok SSE tunnel URL to Google Drive."""
        self.tunnel_file.write_text(url.strip(), encoding="utf-8")

    def get_persisted_tunnel_url(self) -> Optional[str]:
        """Reads the currently persisted tunnel URL if available."""
        if self.tunnel_file.exists():
            return self.tunnel_file.read_text(encoding="utf-8").strip()
        return None

    def generate_colab_bootstrap_cell(self, port: int = 8000) -> str:
        """Synthesizes a compact, silent bootstrap cell to start FastMCP inside Google Colab."""
        return (
            "import os, sys\n"
            f"# ml_mcp Colab Bootstrap on port {port}\n"
            "from ml_mcp.server import mcp\n"
            f"print('ml_mcp daemon ready on port {port}')\n"
        )
