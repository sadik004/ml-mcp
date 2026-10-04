"""Artifact round-trip fraud guard tests.

Mathematical Invariant:
Any tool that creates, trains, ensembles, or exports a machine learning model
MUST persist a fully fitted, operational scikit-learn estimator.
The serialized artifact must support:
`joblib.load()` -> `check_is_fitted()` -> `predict_proba()`
yielding valid, finite probability vectors that match in-memory outputs.

Returning unfitted stub dicts, string summaries without artifacts, or zombie estimators
violates Cardinal Sin #4 (Zombie Models & Fake Serialization).

Every test here must FAIL (RED) against today's codebase where artifacts are missing or stubbed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from ml_mcp.server import mcp


@pytest.fixture
def roundtrip_dataset_csv(tmp_path: Path) -> str:
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 4)
    # Binary target with clear signal
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    df["target"] = y
    csv_file = tmp_path / "roundtrip_test.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def _get_tool_fn(name: str):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert name in tools, f"Tool '{name}' not registered on FastMCP server"
    return tools[name].fn


# ------------------------------------------------------------------------------
# 1. ml_create_ensemble artifact persistence & round-trip
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_ensemble_artifact_roundtrip(roundtrip_dataset_csv: str, tmp_path: Path):
    """ml_create_ensemble must persist a real, fitted StackingClassifier estimator.

    Today's implementation returns only string names and does NOT persist a model artifact.
    """
    tool_fn = _get_tool_fn("ml_create_ensemble")
    res = await tool_fn(
        csv_path=roundtrip_dataset_csv,
        target_column="target",
        task_type="classification",
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"

    # Check for artifact path in response
    artifact_path = res.get("artifact_path") or res.get("model_path")
    assert artifact_path is not None, (
        f"Zombie Artifact Fraud: ml_create_ensemble returned only metadata without persisting "
        f"a fitted ensemble estimator! Output: {res}"
    )
    assert os.path.exists(artifact_path), f"Persisted artifact path '{artifact_path}' does not exist on disk!"

    # Load and verify it is a real fitted estimator
    loaded_model = joblib.load(artifact_path)
    check_is_fitted(loaded_model)

    # Verify predict_proba works
    sample_X = np.random.randn(5, 4)
    probs = loaded_model.predict_proba(sample_X)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


# ------------------------------------------------------------------------------
# 2. ml_export_and_document round-trip
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_export_and_document_artifact_roundtrip(roundtrip_dataset_csv: str, tmp_path: Path):
    """ml_export_and_document must produce loadable .joblib artifact that matches predict_proba."""
    tool_fn = _get_tool_fn("ml_export_and_document")
    out_dir = tmp_path / "export_roundtrip"

    res = await tool_fn(
        csv_path=roundtrip_dataset_csv,
        target_column="target",
        output_dir=str(out_dir),
        model_name="test_champion",
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"

    joblib_file = out_dir / "test_champion.joblib"
    assert joblib_file.exists(), f"Expected joblib file at {joblib_file}"

    loaded_model = joblib.load(str(joblib_file))
    check_is_fitted(loaded_model)

    sample_X = np.random.randn(3, 4)
    probs = loaded_model.predict_proba(sample_X)
    assert probs.shape == (3, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


# ------------------------------------------------------------------------------
# 3. ml_run_model_tournament champion artifact round-trip
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tournament_champion_artifact_roundtrip(roundtrip_dataset_csv: str):
    """ml_run_model_tournament must persist a fully fitted champion estimator."""
    tool_fn = _get_tool_fn("ml_run_model_tournament")
    res = await tool_fn(
        csv_path=roundtrip_dataset_csv,
        target_column="target",
        tune_trials=2,
        n_splits=2,
        view="compact",
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    champion_path = res.get("champion_model_path", ".artifacts/models/champion_model.joblib")
    assert os.path.exists(champion_path), f"Champion model artifact '{champion_path}' does not exist on disk!"

    loaded_model = joblib.load(champion_path)
    check_is_fitted(loaded_model)

    sample_X = np.random.randn(2, 4)
    probs = loaded_model.predict_proba(sample_X)
    assert probs.shape == (2, 2)


# ------------------------------------------------------------------------------
# 4. ml_calibrate_probabilities artifact persistence round-trip
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_calibrate_probabilities_artifact_roundtrip(roundtrip_dataset_csv: str, tmp_path: Path):
    """ml_calibrate_probabilities must persist the calibrated model artifact on disk."""
    tool_fn = _get_tool_fn("ml_calibrate_probabilities")
    cal_out = tmp_path / "calibrated_model.joblib"

    res = await tool_fn(
        csv_path=roundtrip_dataset_csv,
        target_column="target",
        output_model_path=str(cal_out),
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    # Persisted calibrated estimator must exist on disk and be loadable
    assert cal_out.exists(), (
        f"Zombie Artifact Fraud: ml_calibrate_probabilities returned metrics without persisting "
        f"the calibrated model estimator to output_model_path '{cal_out}'!"
    )
    loaded_calibrator = joblib.load(str(cal_out))
    check_is_fitted(loaded_calibrator)


# ------------------------------------------------------------------------------
# 5. ml_export_and_document end-to-end pipeline round-trip on mixed types
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_export_pipeline_mixed_types_roundtrip(tmp_path: Path):
    """ml_export_and_document must persist full end-to-end Pipeline with preprocessing,

    not a disconnected raw classifier that fails on raw tabular input.
    """
    tool_fn = _get_tool_fn("ml_export_and_document")
    out_dir = tmp_path / "pipeline_export"

    # Dataset with categorical string column
    df = pd.DataFrame({
        "num_col": [1.0, 2.0, 3.0, 4.0] * 10,
        "cat_col": ["low", "high", "medium", "low"] * 10,
        "target": [0, 1, 0, 1] * 10,
    })
    csv_file = tmp_path / "mixed_data.csv"
    df.to_csv(csv_file, index=False)

    res = await tool_fn(
        csv_path=str(csv_file),
        target_column="target",
        output_dir=str(out_dir),
        model_name="pipeline_champ",
    )

    assert "error" not in res, f"Tool crashed: {res.get('error')}"
    joblib_file = out_dir / "pipeline_champ.joblib"
    assert joblib_file.exists()

    loaded_pipe = joblib.load(str(joblib_file))
    # Must accept raw DataFrame with string column 'cat_col' without crashing
    raw_test_input = pd.DataFrame({"num_col": [1.5], "cat_col": ["high"]})
    probs = loaded_pipe.predict_proba(raw_test_input)
    assert probs.shape == (1, 2)

