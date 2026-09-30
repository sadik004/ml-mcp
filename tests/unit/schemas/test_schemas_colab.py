"""Unit tests for Colab Notebook and Session DTOs."""
import pytest

from ml_mcp.schemas.colab import ColabNotebookDTO, ColabSessionDTO


def test_colab_notebook_dto():
    dto = ColabNotebookDTO(
        notebook_path="/content/ml_grandmaster_run.ipynb",
        cell_count=12,
        cuda_enabled=True,
        mixed_precision=True,
        drive_checkpoint_dir="/content/drive/MyDrive/ml_mcp/checkpoints",
        kaggle_slug="titanic",
    )

    assert dto.cell_count == 12
    assert dto.cuda_enabled is True
    assert dto.kaggle_slug == "titanic"
    compact = dto.to_compact()
    assert compact["notebook_path"] == "/content/ml_grandmaster_run.ipynb"
    assert compact["cuda_enabled"] is True


def test_colab_session_dto():
    dto = ColabSessionDTO(
        tunnel_url="https://bright-galaxy-trycloudflare.com/sse",
        is_alive=True,
        vram_allocated_mb=1450.0,
        vram_total_mb=15360.0,
        active_job_id="job_kaggle_grandmaster_01",
    )

    assert dto.is_alive is True
    assert dto.vram_total_mb == 15360.0
