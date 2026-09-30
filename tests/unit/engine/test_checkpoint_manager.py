"""Unit tests for Checkpoint Manager and Experiment Lineage tracking."""
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.checkpoint_manager import CheckpointManager
from ml_mcp.schemas.tournament import LineageDTO


def test_checkpoint_manager_save_and_load(tmp_path):
    manager = CheckpointManager(storage_root=str(tmp_path))

    model = LogisticRegression()
    job_id = "test_job_42"

    saved_path = manager.save_checkpoint(model=model, job_id=job_id)
    assert Path(saved_path).exists()
    assert saved_path.endswith(".joblib")

    # Load back and verify
    loaded_model = manager.load_checkpoint(job_id=job_id)
    assert isinstance(loaded_model, LogisticRegression)


def test_checkpoint_manager_lineage_metadata(tmp_path):
    manager = CheckpointManager(storage_root=str(tmp_path))

    lineage = manager.record_lineage(
        dataset_content=b"sample_dataset_bytes_12345",
        row_count=500,
        column_count=8,
        random_seed=42,
        job_id="lineage_job_01",
    )

    assert isinstance(lineage, LineageDTO)
    assert len(lineage.dataset_hash) == 64
    assert lineage.row_count == 500
    assert lineage.random_seed == 42
    assert (tmp_path / "checkpoints" / "lineage_job_01_lineage.json").exists()
