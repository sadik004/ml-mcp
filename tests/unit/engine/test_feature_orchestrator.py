"""Unit tests for Phase 2 Feature Pipeline Master Orchestrator."""
import os
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.feature_orchestrator import FeaturePipelineOrchestrator
from ml_mcp.schemas.feature import FeaturePipelineReportDTO


def test_feature_pipeline_orchestrator_execution(tmp_path):
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "time_seconds": np.linspace(0, 86400, n),
        "V1": np.random.normal(0, 1, n),
        "V2": np.random.normal(0, 1, n),
        "V3": np.random.normal(0, 1, n),
        "V4": np.random.normal(0, 1, n),
        "amount": np.random.exponential(100, n),
        "target": np.random.binomial(1, 0.2, n),
    })

    orchestrator = FeaturePipelineOrchestrator(artifact_dir=str(tmp_path))
    report = orchestrator.prepare_pipeline(
        df=df,
        target_column="target",
        task_type="classification",
        enable_synthesis=True,
        enable_pruning=False,
    )

    assert isinstance(report, FeaturePipelineReportDTO)
    assert report.original_shape == [150, 6]
    assert report.transformed_shape[0] == 150
    # Synthesized time harmonics and latent L2 norm
    assert len(report.synthesized_features) >= 1
    assert "time_seconds_hour" in report.synthesized_features or "V_l2_norm" in report.synthesized_features
    assert report.class_weights is not None
    assert "0" in report.class_weights and "1" in report.class_weights
    assert os.path.exists(report.preprocessor_artifact_path)
    assert "FEATURE PIPELINE GENERATION COMPLETE" in report.receipt_card
