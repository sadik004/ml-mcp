"""RED Verification Suite: Data Leakage & Sealed Pipeline Invariants (C5).

Enforces Rule T3 (independent oracles) and Rule T4 (spies observe without answering).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ml_mcp.engine.tournament import TournamentArena
from ml_mcp.repositories.storage_repository import LocalDiskStorageRepository
from ml_mcp.services.safety_service import SafetyService


class _SpyPreprocessor(BaseEstimator, TransformerMixin):
    """Spy transformer to record sample sizes during fit calls without altering behavior."""

    fit_sizes: list[int] = []

    def fit(self, X: Any, y: Any = None) -> "_SpyPreprocessor":
        self.fit_sizes.append(len(X))
        return self

    def transform(self, X: Any) -> Any:
        return X

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        self.fit_sizes.append(len(X))
        return X


def test_tournament_preprocessor_never_sees_full_data_in_cv() -> None:
    """C5: Preprocessor must never run fit or fit_transform on full dataset prior to cross-validation splits."""
    n_samples = 300
    df = pd.DataFrame({
        "num": np.random.randn(n_samples),
        "cat": ["A", "B", "C"] * (n_samples // 3),
        "target": np.random.randint(0, 2, n_samples),
    })

    arena = TournamentArena()
    from sklearn.compose import ColumnTransformer
    orig_fit = ColumnTransformer.fit_transform
    recorded_sizes: list[int] = []

    def spy_fit_transform(self: Any, X: Any, y: Any = None, **kwargs: Any) -> Any:
        recorded_sizes.append(len(X))
        return orig_fit(self, X, y, **kwargs)

    with patch.object(ColumnTransformer, "fit_transform", side_effect=spy_fit_transform, autospec=True):
        arena.run_tournament(
            df=df,
            target_column="target",
            task_type="classification",
            scoring="roc_auc",
        )
        for sz in recorded_sizes:
            assert sz < n_samples, f"Data Leakage Detected! Preprocessor fit_transform called on full {n_samples} samples!"


def test_safety_service_calibrate_fits_inside_split() -> None:
    """C5: SafetyService.calibrate_probabilities must not fit preprocessor on full data."""
    n_samples = 200
    df = pd.DataFrame({
        "num": np.random.randn(n_samples),
        "cat": ["X", "Y"] * (n_samples // 2),
        "target": np.random.randint(0, 2, n_samples),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_file, index=False)
        repo = LocalDiskStorageRepository()
        service = SafetyService(repository=repo)

        from sklearn.compose import ColumnTransformer
        orig_fit = ColumnTransformer.fit_transform
        recorded_sizes: list[int] = []

        def spy_fit_transform(self: Any, X: Any, y: Any = None, **kwargs: Any) -> Any:
            recorded_sizes.append(len(X))
            return orig_fit(self, X, y, **kwargs)

        with patch.object(ColumnTransformer, "fit_transform", side_effect=spy_fit_transform, autospec=True):
            service.calibrate_probabilities(
                csv_path=str(csv_file),
                target_column="target",
                method="temperature",
            )
            for sz in recorded_sizes:
                assert sz < n_samples, f"Leakage: preprocessor fit on full {n_samples} dataset!"




def test_persisted_artifact_serves_raw_features() -> None:
    """C5: Persisted tournament model must be a complete Pipeline accepting raw features (including strings)."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0] * 50,
        "category": ["low", "medium", "high", "low"] * 50,
        "target": [0, 1, 0, 1] * 50,
    })
    arena = TournamentArena()
    report = arena.run_tournament(
        df=df,
        target_column="target",
        task_type="classification",
    )
    # The champion artifact must be a Pipeline that encapsulates preprocessing
    assert hasattr(arena, "champion_pipeline_"), "Champion must be saved as full Pipeline with preprocessing"
    raw_query = pd.DataFrame({"num": [2.5], "category": ["medium"]})
    # Must predict successfully on raw strings without manual external pre-transform
    probs = arena.champion_pipeline_.predict_proba(raw_query)
    assert probs.shape == (1, 2)


def test_user_model_refit_warns() -> None:
    """H1/Hygiene: If calibrate_probabilities clones and refits a user-provided model, it must emit a warning."""
    df = pd.DataFrame({
        "x": np.random.randn(50),
        "target": np.random.randint(0, 2, 50),
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "data.csv"
        df.to_csv(csv_file, index=False)
        repo = LocalDiskStorageRepository()
        service = SafetyService(repository=repo)

        # Train a mock model to disk
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression().fit(df[["x"]], df["target"])
        saved_model_path = repo.save_model(m, str(Path(tmp_dir) / "dummy_user_model.joblib"))

        report = service.calibrate_probabilities(
            csv_path=str(csv_file),
            target_column="target",
            model_path=saved_model_path,
        )
        warnings_list = report.get("warnings", []) if isinstance(report, dict) else getattr(report, "warnings", [])
        assert any("refit" in w.lower() or "clone" in w.lower() for w in warnings_list)


