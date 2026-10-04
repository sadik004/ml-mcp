"""Model Tournament & Lineage MCP Tool Router."""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from mcp.server.fastmcp import FastMCP

from ml_mcp.engine.error_envelope import format_error_envelope
from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.services.model_service import ModelService

logger = logging.getLogger(__name__)


def register_model_tools(mcp: FastMCP, service: Optional[ModelService] = None) -> None:
    """Register Phase 3 Model Tournament & Tuning tools onto FastMCP instance."""
    model_svc = service or ModelService()

    @mcp.tool()
    async def ml_run_model_tournament(
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
        task_type: Literal["classification", "regression"] = "classification",
        primary_metric: str = "pr_auc",
        n_splits: int = 5,
        tune_trials: int = 20,
        overfit_penalty_lambda: float = 1.0,
        class_weights: Optional[Dict[str, float]] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Phase 3 Master Model Tournament: Multi-model CV evaluation with Optuna TPE, Overfit Penalty, and Out-Of-Fold stacking."""
        try:
            res = model_svc.run_model_tournament(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                primary_metric=primary_metric,
                n_splits=n_splits,
                tune_trials=tune_trials,
                overfit_penalty_lambda=overfit_penalty_lambda,
                class_weights=class_weights,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_run_model_tournament", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_benchmark_models(
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        cv_splits: int = 5,
        scoring: Optional[str] = None,
        group_column: Optional[str] = None,
        fast_mode: bool = False,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Execute 8-model competitive arena with metric alignment, diversity-guarded stacking, and GPU acceleration."""
        try:
            res = model_svc.benchmark_models(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                cv_splits=cv_splits,
                scoring=scoring,
                group_column=group_column,
                fast_mode=fast_mode,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_benchmark_models", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_create_ensemble(
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build leak-free Stacking Ensemble from top models using Out-Of-Fold predictions."""
        try:
            res = model_svc.create_ensemble(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                artifact_path=artifact_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_create_ensemble", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_track_lineage(
        csv_path: str,
        checkpoint_dir: str = "artifacts/checkpoints",
    ) -> Dict[str, Any]:
        """Generate SHA-256 fingerprint and metadata lineage report for experiment artifacts."""
        try:
            res = model_svc.track_lineage(
                csv_path=csv_path,
                checkpoint_dir=checkpoint_dir,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_track_lineage", ["csv_path"])

    @mcp.tool()
    async def ml_tune_hyperparameters(
        csv_path: str,
        target_column: str,
        model_name: str = "lightgbm",
        n_trials: int = 10,
        task_type: str = "classification",
        metric: Optional[str] = None,
        group_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bayesian hyperparameter optimization via Optuna TPE and active MedianPruner."""
        try:
            res = model_svc.tune_hyperparameters(
                csv_path=csv_path,
                target_column=target_column,
                model_name=model_name,
                n_trials=n_trials,
                task_type=task_type,
                metric=metric,
                group_column=group_column,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_tune_hyperparameters", ["csv_path", "target_column", "model_name"])
