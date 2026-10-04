"""Audit and Pre-flight MCP Tool Router."""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from mcp.server.fastmcp import FastMCP

from ml_mcp.engine.error_envelope import format_error_envelope
from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.services.audit_service import AuditService

logger = logging.getLogger(__name__)


def register_audit_tools(mcp: FastMCP, service: Optional[AuditService] = None) -> None:
    """Register Phase 1 Data Audit & Hygiene tools onto FastMCP instance."""
    audit_svc = service or AuditService()

    @mcp.tool()
    async def ml_preflight_audit(
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        dataset_name: str = "Dataset",
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Phase 1 Master Pre-Flight Audit: Unified SHA-256 Lineage, Leakage, Collinearity, Label Error and Constraint Verification."""
        try:
            res = audit_svc.preflight_audit(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                dataset_name=dataset_name,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_preflight_audit", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_audit_dataset(
        csv_path: str,
        target_column: Optional[str] = None,
        task_type: Literal["classification", "regression"] = "classification",
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Pre-flight statistical audit of dataset hygiene and leakage indicators."""
        try:
            res = audit_svc.audit_dataset(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_audit_dataset", {"csv_path": csv_path, "target_column": target_column})

    @mcp.tool()
    async def ml_detect_target_leakage(
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        threshold_correlation: float = 0.95,
        threshold_cramers_v: float = 0.90,
    ) -> Dict[str, Any]:
        """Scan for suspicious features with correlation >= 0.95, Cramer's V >= 0.90, or high mutual info."""
        try:
            res = audit_svc.detect_target_leakage(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                threshold_correlation=threshold_correlation,
                threshold_cramers_v=threshold_cramers_v,
                view="detailed",
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_detect_target_leakage", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_check_collinearity(
        csv_path: str,
        target_column: Optional[str] = None,
        vif_threshold: float = 10.0,
        correlation_cutoff: float = 0.90,
        condition_number_threshold: float = 30.0,
        output_path: Optional[str] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Detect multicollinear features using SVD condition number, pure NumPy VIF and Iterative Pruning."""
        try:
            res = audit_svc.check_collinearity(
                csv_path=csv_path,
                target_column=target_column,
                vif_threshold=vif_threshold,
                correlation_cutoff=correlation_cutoff,
                condition_number_threshold=condition_number_threshold,
                output_path=output_path,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_check_collinearity", ["csv_path"])

    @mcp.tool()
    async def ml_detect_label_errors(
        csv_path: str,
        target_column: str,
        cv_splits: int = 5,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Detect corrupt/noisy training labels via MIT Confident Learning (classification) or Residual Dispersion (regression)."""
        try:
            res = audit_svc.detect_label_errors(
                csv_path=csv_path,
                target_column=target_column,
                cv_splits=cv_splits,
                task_type=task_type,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_detect_label_errors", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_verify_constraints(
        csv_path: str,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Validate physical domain limits and automated Amazon Deequ IQR range constraints."""
        try:
            res = audit_svc.verify_constraints(
                csv_path=csv_path,
                constraints=constraints,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_verify_constraints", ["csv_path"])
