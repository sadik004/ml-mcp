"""Unit tests for Colab Notebook Generator."""
import json
import os
import pytest

from ml_mcp.engine.colab_generator import ColabNotebookGenerator
from ml_mcp.schemas.colab import ColabNotebookDTO


def test_colab_generator_creates_valid_ipynb(tmp_path):
    """Verify colab generator synthesizes a valid JSON .ipynb notebook."""
    generator = ColabNotebookGenerator()
    out_file = tmp_path / "train_champion.ipynb"

    dto = generator.generate_notebook(
        output_path=str(out_file),
        project_name="Fraud_Detection",
        dataset_name="transactions.csv",
        target_column="is_fraud",
        task_type="classification",
    )

    assert isinstance(dto, ColabNotebookDTO)
    assert os.path.exists(dto.notebook_path)
    assert dto.cell_count >= 6
    assert dto.cuda_enabled is True
    assert dto.drive_checkpoint_dir == "/content/drive/MyDrive/ml_mcp/checkpoints"

    # Parse and verify .ipynb JSON structure
    with open(dto.notebook_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    assert "cells" in nb_json
    assert "metadata" in nb_json
    assert nb_json["nbformat"] == 4
    assert len(nb_json["cells"]) == dto.cell_count
