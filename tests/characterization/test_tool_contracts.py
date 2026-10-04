"""Characterization contract tests locking public MCP tool interfaces.

These tests call each registered tool with deterministic fixtures and snapshot
their output keys and value types (not values) into `tests/characterization/snapshots/`.

Purpose:
Prevent regressions or unexpected schema breaks during Phase P5 clean architecture refactoring.
These tests MUST BE GREEN (PASSING) before and after the refactor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from ml_mcp.server import mcp

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"


@pytest.fixture(scope="session")
def standard_dataset_csv(tmp_path_factory) -> str:
    tmp_path = tmp_path_factory.mktemp("contracts")
    np.random.seed(42)
    n = 60
    X = np.random.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3", "feat_4"])
    df["target"] = y
    df["text_feat"] = ["sample text narrative log" if i % 2 == 0 else "error exception warning" for i in range(n)]
    csv_file = tmp_path / "contract_data.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def _get_tool_fn(name: str):
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    assert name in tools, f"Tool '{name}' not found in registered tools"
    return tools[name].fn


def _verify_and_snapshot(tool_name: str, res: Dict[str, Any]) -> None:
    assert isinstance(res, dict), f"Tool {tool_name} did not return a dict: {type(res)}"
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snap_file = SNAPSHOTS_DIR / f"{tool_name}.json"

    # Contract schema: mapping of key to type
    schema = {k: type(v).__name__ for k, v in sorted(res.items())}

    if not snap_file.exists():
        snap_file.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    else:
        saved_schema = json.loads(snap_file.read_text(encoding="utf-8"))
        # Verify that all saved keys exist in current response
        for k in saved_schema:
            assert k in res, f"Contract violation: tool '{tool_name}' missing expected key '{k}'"


# ------------------------------------------------------------------------------
# 1. Audit Tools
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contract_ml_ping():
    fn = _get_tool_fn("ml_ping")
    res = await fn()
    _verify_and_snapshot("ml_ping", res)


@pytest.mark.asyncio
async def test_contract_ml_preflight_audit(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_preflight_audit")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", view="compact")
    _verify_and_snapshot("ml_preflight_audit", res)


@pytest.mark.asyncio
async def test_contract_ml_audit_dataset(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_audit_dataset")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", view="compact")
    _verify_and_snapshot("ml_audit_dataset", res)


@pytest.mark.asyncio
async def test_contract_ml_detect_target_leakage(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_detect_target_leakage")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_detect_target_leakage", res)


@pytest.mark.asyncio
async def test_contract_ml_check_collinearity(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_check_collinearity")
    res = await fn(csv_path=standard_dataset_csv)
    _verify_and_snapshot("ml_check_collinearity", res)


@pytest.mark.asyncio
async def test_contract_ml_detect_label_errors(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_detect_label_errors")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_detect_label_errors", res)


@pytest.mark.asyncio
async def test_contract_ml_verify_constraints(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_verify_constraints")
    res = await fn(csv_path=standard_dataset_csv)
    _verify_and_snapshot("ml_verify_constraints", res)


# ------------------------------------------------------------------------------
# 2. Feature Tools
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contract_ml_prepare_feature_pipeline(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_prepare_feature_pipeline")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", view="compact")
    _verify_and_snapshot("ml_prepare_feature_pipeline", res)


@pytest.mark.asyncio
async def test_contract_ml_auto_clean_and_pipe(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_auto_clean_and_pipe")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_auto_clean_and_pipe", res)


@pytest.mark.asyncio
async def test_contract_ml_handle_text_features(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_handle_text_features")
    res = await fn(csv_path=standard_dataset_csv)
    _verify_and_snapshot("ml_handle_text_features", res)


@pytest.mark.asyncio
async def test_contract_ml_balance_classes(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_balance_classes")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_balance_classes", res)


@pytest.mark.asyncio
async def test_contract_ml_synthesize_features(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_synthesize_features")
    res = await fn(csv_path=standard_dataset_csv)
    _verify_and_snapshot("ml_synthesize_features", res)


@pytest.mark.asyncio
async def test_contract_ml_prune_features(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_prune_features")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", top_k=2)
    _verify_and_snapshot("ml_prune_features", res)


@pytest.mark.asyncio
async def test_contract_ml_transform_target(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_transform_target")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", method="log1p")
    _verify_and_snapshot("ml_transform_target", res)


# ------------------------------------------------------------------------------
# 3. Model Tools
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contract_ml_run_model_tournament(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_run_model_tournament")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", tune_trials=1, n_splits=2, view="compact")
    _verify_and_snapshot("ml_run_model_tournament", res)


@pytest.mark.asyncio
async def test_contract_ml_benchmark_models(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_benchmark_models")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", fast_mode=True, cv_splits=2)
    _verify_and_snapshot("ml_benchmark_models", res)


@pytest.mark.asyncio
async def test_contract_ml_create_ensemble(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_create_ensemble")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_create_ensemble", res)


@pytest.mark.asyncio
async def test_contract_ml_track_lineage(standard_dataset_csv: str, tmp_path: Path):
    fn = _get_tool_fn("ml_track_lineage")
    res = await fn(csv_path=standard_dataset_csv, checkpoint_dir=str(tmp_path / "checkpoints"))
    _verify_and_snapshot("ml_track_lineage", res)


@pytest.mark.asyncio
async def test_contract_ml_tune_hyperparameters(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_tune_hyperparameters")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", n_trials=1, model_name="lightgbm")
    _verify_and_snapshot("ml_tune_hyperparameters", res)


# ------------------------------------------------------------------------------
# 4. Safety Tools
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contract_ml_calibrate_probabilities(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_calibrate_probabilities")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_calibrate_probabilities", res)


@pytest.mark.asyncio
async def test_contract_ml_tune_threshold_and_errors(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_tune_threshold_and_errors")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_tune_threshold_and_errors", res)


@pytest.mark.asyncio
async def test_contract_ml_explain_predictions(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_explain_predictions")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", instance_index=0)
    _verify_and_snapshot("ml_explain_predictions", res)


@pytest.mark.asyncio
async def test_contract_ml_detect_ood(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_detect_ood")
    res = await fn(train_csv_path=standard_dataset_csv, test_csv_path=standard_dataset_csv)
    _verify_and_snapshot("ml_detect_ood", res)


@pytest.mark.asyncio
async def test_contract_ml_stress_test_and_fairness(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_stress_test_and_fairness")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_stress_test_and_fairness", res)


@pytest.mark.asyncio
async def test_contract_ml_conformal_risk_control(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_conformal_risk_control")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_conformal_risk_control", res)


@pytest.mark.asyncio
async def test_contract_ml_certify_safety_and_decisions(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_certify_safety_and_decisions")
    res = await fn(
        csv_path=standard_dataset_csv,
        target_column="target",
        train_ephemeral=True,
        allow_in_sample_diagnostic=True,
        view="compact",
    )
    _verify_and_snapshot("ml_certify_safety_and_decisions", res)


# ------------------------------------------------------------------------------
# 5. Serving Tools
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contract_ml_export_and_document(standard_dataset_csv: str, tmp_path: Path):
    fn = _get_tool_fn("ml_export_and_document")
    res = await fn(csv_path=standard_dataset_csv, target_column="target", output_dir=str(tmp_path / "exp"))
    _verify_and_snapshot("ml_export_and_document", res)


@pytest.mark.asyncio
async def test_contract_ml_optimize_inference(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_optimize_inference")
    res = await fn(csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_optimize_inference", res)


@pytest.mark.asyncio
async def test_contract_ml_generate_eval_dashboard(tmp_path: Path):
    fn = _get_tool_fn("ml_generate_eval_dashboard")
    res = await fn(
        output_html_path=str(tmp_path / "d.html"),
        project_name="contract_model",
        metrics={"ROC-AUC": 0.95, "F1-Macro": 0.92},
    )
    _verify_and_snapshot("ml_generate_eval_dashboard", res)


@pytest.mark.asyncio
async def test_contract_ml_generate_serving_api(tmp_path: Path):
    fn = _get_tool_fn("ml_generate_serving_api")
    res = await fn(output_dir=str(tmp_path / "srv"), model_name="srv_model", feature_names=["f1", "f2"])
    _verify_and_snapshot("ml_generate_serving_api", res)


@pytest.mark.asyncio
async def test_contract_ml_generate_docker_spec(tmp_path: Path):
    fn = _get_tool_fn("ml_generate_docker_spec")
    res = await fn(output_dir=str(tmp_path / "docker"))
    _verify_and_snapshot("ml_generate_docker_spec", res)


@pytest.mark.asyncio
async def test_contract_ml_monitor_drift(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_monitor_drift")
    res = await fn(reference_csv_path=standard_dataset_csv, current_csv_path=standard_dataset_csv)
    _verify_and_snapshot("ml_monitor_drift", res)


@pytest.mark.asyncio
async def test_contract_ml_pseudo_label_loop(standard_dataset_csv: str):
    fn = _get_tool_fn("ml_pseudo_label_loop")
    res = await fn(train_csv_path=standard_dataset_csv, unlabelled_csv_path=standard_dataset_csv, target_column="target")
    _verify_and_snapshot("ml_pseudo_label_loop", res)


# ------------------------------------------------------------------------------
# 6. Colab Tools
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contract_ml_colab_status():
    fn = _get_tool_fn("ml_colab_status")
    res = await fn()
    _verify_and_snapshot("ml_colab_status", res)


@pytest.mark.asyncio
async def test_contract_ml_generate_colab_notebook(tmp_path: Path):
    fn = _get_tool_fn("ml_generate_colab_notebook")
    res = await fn(output_ipynb_path=str(tmp_path / "nb.ipynb"))
    _verify_and_snapshot("ml_generate_colab_notebook", res)


@pytest.mark.asyncio
async def test_contract_ml_cancel_job():
    fn = _get_tool_fn("ml_cancel_job")
    res = await fn(job_id="dummy_job_123")
    _verify_and_snapshot("ml_cancel_job", res)


@pytest.mark.asyncio
async def test_contract_ml_colab_execute():
    fn = _get_tool_fn("ml_colab_execute")
    res = await fn(code="print('contract_test')", session="gpu", timeout=10.0)
    _verify_and_snapshot("ml_colab_execute", res)


@pytest.mark.asyncio
async def test_contract_ml_colab_upload(tmp_path: Path):
    fn = _get_tool_fn("ml_colab_upload")
    dummy = tmp_path / "upload_test.txt"
    dummy.write_text("hello colab", encoding="utf-8")
    res = await fn(local_path=str(dummy), remote_path="/content/upload_test.txt")
    _verify_and_snapshot("ml_colab_upload", res)


@pytest.mark.asyncio
async def test_contract_ml_colab_download(tmp_path: Path):
    fn = _get_tool_fn("ml_colab_download")
    target = tmp_path / "download_test.txt"
    res = await fn(remote_path="/content/upload_test.txt", local_path=str(target))
    _verify_and_snapshot("ml_colab_download", res)


@pytest.mark.asyncio
async def test_contract_ml_colab_provision():
    fn = _get_tool_fn("ml_colab_provision")
    res = await fn(accelerator="gpu")
    _verify_and_snapshot("ml_colab_provision", res)


@pytest.mark.asyncio
async def test_contract_ml_colab_stop():
    fn = _get_tool_fn("ml_colab_stop")
    res = await fn(session="nonexistent_session")
    _verify_and_snapshot("ml_colab_stop", res)

