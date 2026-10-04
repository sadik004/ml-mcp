"""Phase 1 Data Audit & Hygiene Service."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from ml_mcp.engine.auditor import DatasetAuditor
from ml_mcp.engine.collinearity import CollinearityFilter
from ml_mcp.engine.constraint_validator import ConstraintValidator
from ml_mcp.engine.label_error_detector import LabelErrorDetector
from ml_mcp.engine.leakage import TargetLeakageDetector
from ml_mcp.engine.preflight_auditor import PreflightAuditor
from ml_mcp.services.base import BaseService, persist_processed_dataframe


class AuditService(BaseService):
    """Orchestrates preflight audit, data hygiene, target leakage, collinearity, and constraints."""

    def preflight_audit(
        self,
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        dataset_name: str = "Dataset",
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        auditor = PreflightAuditor()
        report = auditor.audit(df, target_column=target_column, task_type=task_type, dataset_name=dataset_name)
        return report.to_compact() if view == "compact" else report.model_dump()

    def audit_dataset(
        self,
        csv_path: str,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        auditor = DatasetAuditor()
        tt: Literal["classification", "regression"] = "regression" if task_type == "regression" else "classification"
        report = auditor.audit_dataset(df, target_column=target_column, task_type=tt)
        return report.to_compact() if view == "compact" else report.model_dump()

    def detect_target_leakage(
        self,
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        threshold_correlation: float = 0.95,
        threshold_cramers_v: float = 0.90,
        drop_leakers: bool = False,
        output_path: Optional[str] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        detector = TargetLeakageDetector(
            threshold_correlation=threshold_correlation,
            threshold_cramers_v=threshold_cramers_v,
        )
        report = detector.detect_leakage(df, target_column=target_column, task_type=task_type)
        report.view = view
        res = report.to_compact() if view == "compact" else report.model_dump()
        if drop_leakers and len(report.leaked_features) > 0:
            df_clean = df.drop(columns=report.leaked_features)
            saved_path = persist_processed_dataframe(df_clean, csv_path, "clean", output_path)
            res["saved_clean_csv_path"] = saved_path
        return res

    def check_collinearity(
        self,
        csv_path: str,
        target_column: Optional[str] = None,
        vif_threshold: float = 10.0,
        correlation_cutoff: float = 0.90,
        condition_number_threshold: float = 30.0,
        prune: bool = False,
        output_path: Optional[str] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        fltr = CollinearityFilter(
            threshold_corr=correlation_cutoff,
            vif_threshold=vif_threshold,
            condition_number_threshold=condition_number_threshold,
        )
        report = fltr.filter_collinearity(df, target_column=target_column)

        # Dataset chaining: Persist processed/pruned dataframe to disk
        df_pruned = df.drop(columns=report.dropped_features) if report.dropped_features else df
        saved_path = persist_processed_dataframe(df_pruned, csv_path, "filtered", output_path)

        if view == "compact":
            res = {
                "processed_csv_path": saved_path,
                "vif_threshold": float(report.vif_threshold),
                "correlation_cutoff": float(report.threshold_corr),
                "high_vif_features": report.high_vif_features,
                "dropped_features": report.dropped_features,
                "collinear_pairs_count": len(report.collinear_pairs),
                "remaining_features_count": report.remaining_features_count,
            }
        else:
            res = {
                **report.model_dump(),
                "processed_csv_path": saved_path,
                "collinear_pairs_count": len(report.collinear_pairs),
                "correlation_cutoff": float(report.threshold_corr),
            }
        if prune and report.dropped_features:
            res["saved_pruned_csv_path"] = saved_path
        return res

    def detect_label_errors(
        self,
        csv_path: str,
        target_column: str,
        cv_splits: int = 5,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        threshold: float = 0.5,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        detector = LabelErrorDetector(cv_splits=cv_splits, random_state=self.settings.random_state)
        report = detector.detect_label_errors(df, target_column=target_column, task_type=task_type)
        return report.to_compact() if view == "compact" else report.model_dump()

    def verify_constraints(
        self,
        csv_path: str,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        validator = ConstraintValidator(constraints=constraints)
        report = validator.validate_constraints(df, constraints=constraints)
        return report.to_compact() if view == "compact" else report.model_dump()
