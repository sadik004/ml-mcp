"""Serving, Deployment & Monitoring MCP Tool Router."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ml_mcp.engine.error_envelope import format_error_envelope
from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.services.serving_service import ServingService

logger = logging.getLogger(__name__)


def register_serving_tools(mcp: FastMCP, service: Optional[ServingService] = None) -> None:
    """Register Phase 5 Serving, Deployment, and Monitoring tools onto FastMCP instance."""
    serving_svc = service or ServingService()

    @mcp.tool()
    async def ml_batch_predict(
        model_path: str,
        input_csv_path: str,
        output_csv_path: str,
        id_column: Optional[str] = None,
        task_type: str = "classification",
        optimal_threshold: Optional[float] = None,
        calibrator_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform chunked high-throughput batch inference with optional DCA optimal cutoff and calibration."""
        try:
            res = serving_svc.batch_predict(
                model_path=model_path,
                input_csv_path=input_csv_path,
                output_csv_path=output_csv_path,
                id_column=id_column,
                task_type=task_type,
                optimal_threshold=optimal_threshold,
                calibrator_path=calibrator_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_batch_predict", ["model_path", "input_csv_path", "output_csv_path"])

    @mcp.tool()
    async def ml_export_and_document(
        csv_path: str,
        target_column: str,
        output_dir: str = "artifacts/bundle",
        model_name: str = "champion_model",
    ) -> Dict[str, Any]:
        """Export atomic .joblib, .onnx, and synthesize official MODEL_CARD.md."""
        try:
            res = serving_svc.export_and_document(
                csv_path=csv_path,
                target_column=target_column,
                output_dir=output_dir,
                model_name=model_name,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_export_and_document", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_optimize_inference(
        csv_path: str,
        target_column: str,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert model to ONNX format with Level-3 graph fusion and benchmark single-sample P95/P99 latency."""
        try:
            res = serving_svc.optimize_inference(
                csv_path=csv_path,
                target_column=target_column,
                model_path=model_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_optimize_inference", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_generate_eval_dashboard(
        output_html_path: str,
        project_name: str = "ML Production Model",
        champion_model: str = "Champion",
        metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        holdout_csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize interactive single-file HTML evaluation dashboard with SVG charts."""
        try:
            res = serving_svc.generate_eval_dashboard(
                output_html_path=output_html_path,
                project_name=project_name,
                champion_model=champion_model,
                metrics=metrics,
                model_path=model_path,
                holdout_csv_path=holdout_csv_path,
                target_column=target_column,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_generate_eval_dashboard", ["output_html_path", "metrics", "model_path", "holdout_csv_path", "target_column"])

    @mcp.tool()
    async def ml_generate_serving_api(
        output_dir: str,
        model_name: str = "champion_model",
        feature_names: Optional[List[str]] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate 3-Tier clean architecture FastAPI serving router and Pydantic v2 schemas."""
        try:
            res = serving_svc.generate_serving_api(
                output_dir=output_dir,
                model_name=model_name,
                feature_names=feature_names,
                model_path=model_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_generate_serving_api", ["output_dir", "feature_names", "model_path"])

    @mcp.tool()
    async def ml_generate_docker_spec(
        output_dir: str,
        service_name: str = "ml-serving",
        port: int = 8000,
    ) -> Dict[str, Any]:
        """Generate hardened multi-stage Dockerfile and docker-compose.yml spec."""
        try:
            res = serving_svc.generate_docker_spec(
                output_dir=output_dir,
                service_name=service_name,
                port=port,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_generate_docker_spec", ["output_dir"])

    @mcp.tool()
    async def ml_monitor_drift(
        reference_csv_path: str,
        current_csv_path: str,
    ) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) and KS-test for data drift detection."""
        try:
            res = serving_svc.monitor_drift(
                reference_csv_path=reference_csv_path,
                current_csv_path=current_csv_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_monitor_drift", ["reference_csv_path", "current_csv_path"])

    @mcp.tool()
    async def ml_pseudo_label_loop(
        train_csv_path: str,
        unlabelled_csv_path: str,
        target_column: str,
        confidence_threshold: float = 0.95,
        alpha: float = 0.10,
    ) -> Dict[str, Any]:
        """Harvest high-confidence pseudo-labels using FlexMatch curriculum thresholds and conformal singletons."""
        try:
            res = await asyncio.to_thread(
                serving_svc.pseudo_label_loop,
                train_csv_path=train_csv_path,
                unlabelled_csv_path=unlabelled_csv_path,
                target_column=target_column,
                confidence_threshold=confidence_threshold,
                alpha=alpha,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_pseudo_label_loop", ["train_csv_path", "unlabelled_csv_path", "target_column"])
