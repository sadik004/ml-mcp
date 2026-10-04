"""Safety, Calibration & Certification MCP Tool Router."""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from mcp.server.fastmcp import FastMCP

from ml_mcp.engine.error_envelope import format_error_envelope
from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.services.safety_service import SafetyService

logger = logging.getLogger(__name__)


def register_safety_tools(mcp: FastMCP, service: Optional[SafetyService] = None) -> None:
    """Register Phase 4 Model Safety, Calibration and Risk Control tools onto FastMCP instance."""
    safety_svc = service or SafetyService()

    @mcp.tool()
    async def ml_certify_safety_and_decisions(
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
        cost_fp: float = 5.0,
        cost_fn: float = 250.0,
        model_path: Optional[str] = None,
        oof_path: Optional[str] = None,
        train_ephemeral: bool = False,
        allow_in_sample_diagnostic: bool = False,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Phase 4 Master Model Certification: Probability calibration (Platt/Beta with ECE audit), Decision Curve Analysis (cost-loss threshold optimization for p*), Fast TreeSHAP attributions, Conformal Risk Control (95 percent coverage guarantee), and Helmholtz/IForest OOD anomaly cutoff."""
        try:
            res = safety_svc.certify_safety_and_decisions(
                csv_path=csv_path,
                target_column=target_column,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                model_path=model_path,
                oof_path=oof_path,
                train_ephemeral=train_ephemeral,
                allow_in_sample_diagnostic=allow_in_sample_diagnostic,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_certify_safety_and_decisions", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_calibrate_probabilities(
        csv_path: str,
        target_column: str,
        model_path: Optional[str] = None,
        model_name: str = "lightgbm",
        method: Optional[Literal["isotonic", "sigmoid", "temperature"]] = None,
        output_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calibrate classifier probabilities via Platt Scaling, Isotonic Regression, or Temperature Scaling."""
        try:
            res = safety_svc.calibrate_probabilities(
                csv_path=csv_path,
                target_column=target_column,
                model_path=model_path,
                model_name=model_name,
                method=method,
                output_model_path=output_model_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_calibrate_probabilities", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_tune_threshold_and_errors(
        csv_path: str,
        target_column: str,
        beta: float = 1.0,
        criterion: Literal["f_beta", "cost_loss"] = "f_beta",
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
        model_path: Optional[str] = None,
        model_name: str = "lightgbm",
    ) -> Dict[str, Any]:
        """Optimize classification decision threshold using cost-sensitive loss matrix and PR-curve cutoffs."""
        try:
            res = safety_svc.tune_threshold_and_errors(
                csv_path=csv_path,
                target_column=target_column,
                beta=beta,
                criterion=criterion,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                benefit_tp=benefit_tp,
                benefit_tn=benefit_tn,
                model_path=model_path,
                model_name=model_name,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_tune_threshold_and_errors", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_explain_predictions(
        csv_path: str,
        target_column: str,
        top_k: int = 10,
        instance_index: Optional[int] = None,
        model_path: Optional[str] = None,
        model_name: str = "lightgbm",
    ) -> Dict[str, Any]:
        """Compute sub-10s TreeSHAP feature attributions or local sample waterfall breakdown."""
        try:
            res = safety_svc.explain_predictions(
                csv_path=csv_path,
                target_column=target_column,
                top_k=top_k,
                instance_index=instance_index,
                model_path=model_path,
                model_name=model_name,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_explain_predictions", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_detect_ood(
        train_csv_path: str,
        test_csv_path: str,
        method: str = "isolation_forest",
    ) -> Dict[str, Any]:
        """Scan unlabelled test data for Out-of-Distribution anomalous samples."""
        try:
            res = safety_svc.detect_ood(
                train_csv_path=train_csv_path,
                test_csv_path=test_csv_path,
                method=method,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_detect_ood", ["train_csv_path", "test_csv_path"])

    @mcp.tool()
    async def ml_stress_test_and_fairness(
        csv_path: str,
        target_column: str,
        protected_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate noise perturbation degradation and demographic slice disparity."""
        try:
            res = safety_svc.stress_test_and_fairness(
                csv_path=csv_path,
                target_column=target_column,
                protected_column=protected_column,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_stress_test_and_fairness", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_conformal_risk_control(
        csv_path: str,
        target_column: str,
        loss_type: Literal["misclassification", "fnr", "asymmetric_cost"] = "misclassification",
        target_risk: float = 0.05,
        test_size: float = 0.3,
        cal_fraction: Optional[float] = None,
        mondrian: bool = False,
    ) -> Dict[str, Any]:
        """Apply Conformal Risk Control (CRC) providing mathematical guarantees E[loss] <= target_risk."""
        try:
            res = safety_svc.conformal_risk_control(
                csv_path=csv_path,
                target_column=target_column,
                loss_type=loss_type,
                target_risk=target_risk,
                test_size=test_size,
                cal_fraction=cal_fraction,
                mondrian=mondrian,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_conformal_risk_control", ["csv_path", "target_column"])
