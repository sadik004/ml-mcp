"""Feature Engineering MCP Tool Router."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP

from ml_mcp.engine.error_envelope import format_error_envelope
from ml_mcp.engine.json_sanitizer import sanitize_for_json
from ml_mcp.services.feature_service import FeatureService

logger = logging.getLogger(__name__)


def register_feature_tools(mcp: FastMCP, service: Optional[FeatureService] = None) -> None:
    """Register Phase 2 Feature Engineering tools onto FastMCP instance."""
    feature_svc = service or FeatureService()

    @mcp.tool()
    async def ml_prepare_feature_pipeline(
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        enable_synthesis: bool = True,
        enable_pruning: bool = True,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        """Phase 2 Master Feature Engineering: Temporal harmonics, manifold outlier features, defensive scaling and imputation, cost-sensitive class balancing, and permutation pruning."""
        try:
            res = feature_svc.prepare_feature_pipeline(
                csv_path=csv_path,
                target_column=target_column,
                task_type=task_type,
                enable_synthesis=enable_synthesis,
                enable_pruning=enable_pruning,
                view=view,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_prepare_feature_pipeline", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_handle_text_features(csv_path: str) -> Dict[str, Any]:
        """Detect free-form natural language text features excluding UUIDs and hex hashes."""
        try:
            res = feature_svc.handle_text_features(csv_path=csv_path)
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_handle_text_features", ["csv_path"])

    @mcp.tool()
    async def ml_auto_clean_and_pipe(
        csv_path: str,
        target_column: Optional[str] = None,
        imputation_strategy: Literal["median", "mean", "iterative"] = "median",
    ) -> Dict[str, Any]:
        """Build zero-leakage ColumnTransformer with defensive omnipresent imputers (median/mean/MICE)."""
        try:
            res = feature_svc.auto_clean_and_pipe(
                csv_path=csv_path,
                target_column=target_column,
                imputation_strategy=imputation_strategy,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_auto_clean_and_pipe", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_balance_classes(
        csv_path: str,
        target_column: str,
    ) -> Dict[str, Any]:
        """Wrap pipeline with dynamic SMOTE / RandomOverSampler for minority classes."""
        try:
            res = feature_svc.balance_classes(
                csv_path=csv_path,
                target_column=target_column,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_balance_classes", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_synthesize_features(
        csv_path: str,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        period: float = 24.0,
        ratio_pairs: Optional[List[List[str]]] = None,
        group_specs: Optional[List[Dict[str, Any]]] = None,
        max_cardinality: int = 1000,
        smoothing: float = 10.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate cyclical sin/cos features, safe ratios, and ExploreKit group aggregations."""
        try:
            res = feature_svc.synthesize_features(
                csv_path=csv_path,
                time_column=time_column,
                period=period,
                ratio_pairs=ratio_pairs,
                group_specs=group_specs,
                max_cardinality=max_cardinality,
                smoothing=smoothing,
                output_path=output_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_synthesize_features", {"csv_path": csv_path, "target_column": target_column, "time_column": time_column})

    @mcp.tool()
    async def ml_prune_features(
        csv_path: str,
        target_column: str,
        top_k: Optional[int] = None,
        importance_threshold: float = 0.005,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        cv_splits: int = 3,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prune noisy and redundant features using gradient-boosted importance (OpenFE architecture)."""
        try:
            res = feature_svc.prune_features(
                csv_path=csv_path,
                target_column=target_column,
                top_k=top_k,
                importance_threshold=importance_threshold,
                task_type=task_type,
                cv_splits=cv_splits,
                output_path=output_path,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_prune_features", ["csv_path", "target_column"])

    @mcp.tool()
    async def ml_transform_target(
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
        values: Optional[List[float]] = None,
        skew_threshold: float = 1.5,
        method: Literal["auto", "log1p", "yeo-johnson"] = "auto",
    ) -> Dict[str, Any]:
        """Normalize skewed continuous target variables using log1p or Yeo-Johnson power transform."""
        try:
            res = feature_svc.transform_target(
                csv_path=csv_path,
                target_column=target_column,
                values=values,
                skew_threshold=skew_threshold,
                method=method,
            )
            return sanitize_for_json(res)
        except Exception as e:
            return format_error_envelope(e, "ml_transform_target", ["target_column", "values"])
