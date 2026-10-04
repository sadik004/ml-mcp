"""RED Verification Suite: Evaluation Mode Honesty & Selection Bias (H1, H2, H4).

Enforces Rule T3 (independent sklearn oracles) and Rule T4 (spies observe without answering).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml_mcp.engine.tournament import TournamentArena
from ml_mcp.repositories.storage_repository import LocalDiskStorageRepository
from ml_mcp.services.safety_service import SafetyService


def test_loaded_model_same_csv_not_held_out() -> None:
    """H1: When loaded model was trained on the provided CSV, evaluation_mode must NOT claim 'held_out_test'."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    df["target"] = y

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_file, index=False)
        repo = LocalDiskStorageRepository()
        service = SafetyService(repository=repo)

        # Train model on this exact CSV and persist
        m = LogisticRegression().fit(X, y)
        saved_path = repo.save_model(m, str(Path(tmp_dir) / "m.joblib"))

        # Tune threshold with saved model on same data
        res = service.tune_threshold_and_errors(
            csv_path=str(csv_file),
            target_column="target",
            model_path=saved_path,
        )
        mode = res.get("evaluation_mode")
        assert mode != "held_out_test", f"H1 Defect: in-sample model evaluation claimed held_out_test mode: {mode}"


def test_small_n_not_labelled_held_out() -> None:
    """H1: When n < 20, p_te = p_tr exactly; evaluation_mode must NOT be 'held_out_test'."""
    X, y = make_classification(n_samples=15, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    df["target"] = y


    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_file, index=False)
        repo = LocalDiskStorageRepository()
        service = SafetyService(repository=repo)

        res = service.tune_threshold_and_errors(
            csv_path=str(csv_file),
            target_column="target",
        )
        mode = res.get("evaluation_mode")
        assert mode != "held_out_test", "When n < 20, p_te is completely in-sample, cannot claim held_out_test!"


def test_threshold_tuned_on_oof() -> None:
    """H2: Decision threshold must be optimized on out-of-fold predictions, not in-sample p_tr."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(4)])
    df["target"] = y

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_file, index=False)
        repo = LocalDiskStorageRepository()
        service = SafetyService(repository=repo)

        # Check if oof_predict_proba was invoked during threshold tuning
        with patch("ml_mcp.services.evaluation.oof_predict_proba", wraps=None) as spy_oof:
            res = service.tune_threshold_and_errors(
                csv_path=str(csv_file),
                target_column="target",
            )
            # Currently safety_service line 227 fits on train and uses in-sample predict_proba(X_train)
            assert spy_oof.call_count >= 1, "Threshold optimization did not use out-of-fold cross-validation probabilities!"


def test_champion_score_from_outer_holdout() -> None:
    """H4: Tournament champion_score must be evaluated on an untouched outer holdout split to prevent Winner's Curse."""
    X, y = make_classification(n_samples=250, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(6)])
    df["target"] = y

    arena = TournamentArena()
    report = arena.run_tournament(
        df=df,
        target_column="target",
        task_type="classification",
    )
    # The arena must register champion_score_source as outer_holdout
    assert hasattr(report, "champion_score_source"), "TournamentLeaderboardDTO must report champion_score_source"
    assert report.champion_score_source == "outer_holdout"
