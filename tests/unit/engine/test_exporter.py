"""Unit tests for Model Exporter, Dry-Run Verification, and Model Card Generator."""
import os
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.exporter import ModelExporter
from ml_mcp.schemas.serving import ModelExportDTO


def test_model_exporter_round_trip_and_card(tmp_path):
    """Verify exporter creates .joblib, runs dry-run test, and writes MODEL_CARD.md."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    exporter = ModelExporter()
    dto = exporter.export_model_bundle(
        model=clf,
        sample_input=X[:2],
        output_dir=str(tmp_path),
        model_name="Champion_RandomForest",
        metrics={"accuracy": 0.95, "f1_macro": 0.94},
        dataset_hash="a1b2c3d4e5f67890",
    )

    assert isinstance(dto, ModelExportDTO)
    assert os.path.exists(dto.joblib_path)
    assert os.path.exists(dto.model_card_path)
    assert dto.p95_latency_ms >= 0.0

    # Verify model card content
    with open(dto.model_card_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Champion_RandomForest" in content
    assert "accuracy" in content
    assert "a1b2c3d4e5f67890" in content
