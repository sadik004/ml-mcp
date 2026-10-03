"""Unit tests for Phase 3 Model Tournament & Anti-Overfit Tuning Orchestrator."""
import os
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.tournament_orchestrator import TournamentOrchestrator
from ml_mcp.schemas.tournament import TournamentAndTuningReportDTO


def test_tournament_orchestrator_classification(tmp_path):
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, n),
        "feat_2": np.random.normal(0, 1, n),
        "feat_3": np.random.normal(0, 1, n),
    })
    y = pd.Series(np.random.binomial(1, 0.3, n))

    orchestrator = TournamentOrchestrator(artifact_dir=str(tmp_path))
    report = orchestrator.run_tournament_and_tuning(
        X=df,
        y=y,
        task_type="classification",
        primary_metric="roc_auc",
        n_splits=3,
        tune_trials=5,
    )

    assert isinstance(report, TournamentAndTuningReportDTO)
    assert report.champion_score > 0.0
    assert report.gap_status in ("HEALTHY", "MODERATE", "SEVERE_OVERFIT")
    assert os.path.exists(report.champion_artifact_path)
    assert os.path.exists(report.oof_predictions_path)
    assert "MODEL TOURNAMENT & TUNING COMPLETE" in report.tournament_card
