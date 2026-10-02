"""Unit tests for FastMCP server instantiation and ml_ping diagnostic tool."""
import pytest

from ml_mcp.server import mcp, ml_ping


def test_server_instance():
    assert mcp is not None
    assert mcp.name == "ml.mcp"


@pytest.mark.asyncio
async def test_ml_ping_diagnostics():
    report = await ml_ping()

    assert report["status"] == "ok"
    assert "python_version" in report
    assert "platform" in report
    assert "cuda_available" in report
    assert "is_colab" in report
    assert "storage_writable" in report


def test_all_25_tools_registered():
    """Verify that all 25 catalog tools + ml_ping are registered on the FastMCP instance."""
    expected_tools = [
        "ml_ping",
        "ml_audit_dataset",
        "ml_detect_target_leakage",
        "ml_check_collinearity",
        "ml_detect_label_errors",
        "ml_verify_constraints",
        "ml_handle_text_features",
        "ml_auto_clean_and_pipe",
        "ml_balance_classes",
        "ml_synthesize_features",
        "ml_prune_features",
        "ml_transform_target",
        "ml_benchmark_models",
        "ml_create_ensemble",
        "ml_track_lineage",
        "ml_tune_hyperparameters",
        "ml_calibrate_probabilities",
        "ml_tune_threshold_and_errors",
        "ml_explain_predictions",
        "ml_detect_ood",
        "ml_stress_test_and_fairness",
        "ml_batch_predict",
        "ml_export_and_document",
        "ml_optimize_inference",
        "ml_generate_eval_dashboard",
        "ml_generate_serving_api",
        "ml_generate_docker_spec",
        "ml_monitor_drift",
        "ml_pseudo_label_loop",
        "ml_generate_colab_notebook",
        "ml_cancel_job",
    ]

    # In FastMCP, tools are tracked in mcp._tool_manager or via get_tools()
    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    for expected in expected_tools:
        assert expected in tool_names, f"Expected tool {expected} not registered on FastMCP"
    assert len(tool_names) >= 26
