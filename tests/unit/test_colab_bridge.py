"""Unit tests for Colab bridge, tunnel watchdog and Drive persistence."""
from pathlib import Path
import pytest

from ml_mcp.colab_bridge import ColabBridge


def test_colab_bridge_initialization(tmp_path):
    bridge = ColabBridge(drive_root=str(tmp_path / "ml_mcp"))
    assert bridge.is_initialized is True
    assert (tmp_path / "ml_mcp" / "checkpoints").exists()
    assert (tmp_path / "ml_mcp" / "artifacts").exists()


def test_tunnel_url_persistence(tmp_path):
    bridge = ColabBridge(drive_root=str(tmp_path / "ml_mcp"))
    sample_url = "https://silent-galaxy-trycloudflare.com/sse"

    bridge.persist_tunnel_url(sample_url)
    loaded_url = bridge.get_persisted_tunnel_url()

    assert loaded_url == sample_url
    assert (tmp_path / "ml_mcp" / "tunnel_url.txt").read_text().strip() == sample_url


def test_colab_bootstrap_script_synthesis(tmp_path):
    bridge = ColabBridge(drive_root=str(tmp_path / "ml_mcp"))
    script = bridge.generate_colab_bootstrap_cell(port=8000)

    assert "import" in script
    assert "8000" in script
    assert "ml_mcp" in script
