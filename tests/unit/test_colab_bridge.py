"""Unit tests for Colab bridge, cloud runner, hardware probes and Drive persistence."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ml_mcp.colab_bridge import ColabBridge, ColabCloudRunner


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


def test_colab_cloud_runner_cli_resolution():
    runner = ColabCloudRunner()
    assert runner.base_cmd is not None
    assert len(runner.base_cmd) >= 1
    assert runner.cli_exe == runner.base_cmd[0]


@patch("subprocess.run")
def test_colab_cloud_runner_get_status_gpu(mock_subproc):
    """Verify GPU probe correctly parses VRAM, RAM, and sets cuda_verified=True."""
    mock_subproc.return_value = MagicMock(
        returncode=0,
        stdout="PROBE:CUDA=True|DEV=Tesla T4|VRAM_TOT_GB=14.56|VRAM_USE_GB=1.20|RAM_TOT_GB=12.67|RAM_AVAIL_GB=11.40\n",
        stderr="",
    )

    runner = ColabCloudRunner()
    status = runner.get_status(session_name="gpu")

    assert status["status"] == "connected"
    assert status["hardware"] == "Tesla T4"
    assert status["cuda_available"] is True
    assert status["cuda_verified"] is True
    assert status["vram_total_mb"] == pytest.approx(14.56 * 1024.0, rel=1e-3)
    assert status["vram_allocated_mb"] == pytest.approx(1.20 * 1024.0, rel=1e-3)
    assert status["ram_total_gb"] == 12.67
    assert status["ram_available_gb"] == 11.40


@patch("subprocess.run")
def test_colab_cloud_runner_get_status_cpu_no_fake_guarantees(mock_subproc):
    """Verify CPU probe never fakes cuda_verified or returns fake T4 hardware."""
    mock_subproc.return_value = MagicMock(
        returncode=0,
        stdout="PROBE:CUDA=False|DEV=CPU|VRAM_TOT_GB=0.00|VRAM_USE_GB=0.00|RAM_TOT_GB=12.67|RAM_AVAIL_GB=10.03\n",
        stderr="",
    )

    runner = ColabCloudRunner()
    status = runner.get_status(session_name="cpu_session")

    assert status["status"] == "connected"
    assert status["hardware"] == "CPU"
    assert status["cuda_available"] is False
    assert status["cuda_verified"] is False
    assert status["vram_total_mb"] == 0.0
    assert status["vram_allocated_mb"] == 0.0


@patch("subprocess.run")
def test_colab_cloud_runner_execute_code(mock_subproc):
    mock_subproc.return_value = MagicMock(
        returncode=0,
        stdout="Computation Result: 42\n",
        stderr="",
    )

    runner = ColabCloudRunner()
    res = runner.execute_code("print('Computation Result:', 6*7)", session="gpu")

    assert res["success"] is True
    assert "42" in res["output"]
    assert res["exit_code"] == 0
    assert "execution_time_seconds" in res


def test_colab_cloud_runner_upload_missing_file():
    runner = ColabCloudRunner()
    res = runner.upload_file("non_existent_file_path_12345.csv", "/content/data.csv")
    assert res["success"] is False
    assert "Local source file does not exist" in res["error"]


@patch("subprocess.run")
def test_colab_cloud_runner_upload_success(mock_subproc, tmp_path):
    local_file = tmp_path / "sample.csv"
    local_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    mock_subproc.return_value = MagicMock(returncode=0, stdout="uploaded", stderr="")

    runner = ColabCloudRunner()
    res = runner.upload_file(str(local_file), "/content/sample.csv", session="gpu")

    assert res["success"] is True
    assert res["bytes_transferred"] > 0
    assert res["remote_path"] == "/content/sample.csv"


@patch("subprocess.run")
def test_colab_cloud_runner_provision_and_stop(mock_subproc):
    mock_subproc.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

    runner = ColabCloudRunner()
    prov_res = runner.provision_session(session_name="t4-cluster", accelerator="T4", high_mem=True)
    assert prov_res["success"] is True
    assert prov_res["accelerator"] == "T4"

    stop_res = runner.stop_session(session="t4-cluster")
    assert stop_res["success"] is True
