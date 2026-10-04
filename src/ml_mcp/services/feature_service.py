"""Phase 2 Feature Engineering, Synthesis & Balancing Service."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ml_mcp.engine.balancer import ClassBalancer
from ml_mcp.engine.feature_orchestrator import FeaturePipelineOrchestrator
from ml_mcp.engine.feature_pruner import GradientFeatureSelector
from ml_mcp.engine.feature_synthesizer import (
    CyclicalFeatureTransformer,
    GroupByAggregationTransformer,
    RatioFeatureTransformer,
)
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.engine.target_transformer import SkewedTargetTransformer
from ml_mcp.engine.text_handler import TextFeatureHandler
from ml_mcp.services.base import BaseService, persist_processed_dataframe


class FeatureService(BaseService):
    """Orchestrates feature preparation, synthesis, pruning, class balancing, and target transformations."""

    def prepare_feature_pipeline(
        self,
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        enable_synthesis: bool = True,
        enable_pruning: bool = True,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        orchestrator = FeaturePipelineOrchestrator()
        report = orchestrator.prepare_pipeline(
            df=df,
            target_column=target_column,
            task_type=task_type,
            enable_synthesis=enable_synthesis,
            enable_pruning=enable_pruning,
        )
        return report.to_compact() if view == "compact" else report.model_dump()

    def handle_text_features(self, csv_path: str) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        handler = TextFeatureHandler()
        report = handler.detect_text_features(df)
        return report.model_dump()

    def auto_clean_and_pipe(
        self,
        csv_path: str,
        target_column: Optional[str] = None,
        imputation_strategy: Literal["median", "mean", "iterative"] = "median",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        builder = DefensivePipelineBuilder(imputation_strategy=imputation_strategy)
        pipe = builder.build_pipeline(df, target_column=target_column)
        return {
            "status": "success",
            "imputation_strategy": imputation_strategy,
            "steps": [name for name, _ in pipe.steps],
            "pipeline_str": str(pipe),
        }

    def balance_classes(self, csv_path: str, target_column: str) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        y = df[target_column]
        balancer = ClassBalancer()
        sampler = balancer.get_resampler(y)
        return {
            "status": "success",
            "sampler_name": getattr(sampler, "__class__", type(sampler)).__name__,
        }

    def synthesize_features(
        self,
        csv_path: str,
        time_column: Optional[str] = None,
        period: float = 24.0,
        ratio_pairs: Optional[List[List[str]]] = None,
        group_specs: Optional[List[Dict[str, Any]]] = None,
        max_cardinality: int = 1000,
        smoothing: float = 10.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        features = [time_column] if time_column and time_column in df.columns else (num_cols[:1] if num_cols else [])
        cyclical_df = df.copy()
        if features:
            cyclical_tf = CyclicalFeatureTransformer(time_periods={col: period for col in features})
            cyclical_df = cyclical_tf.fit_transform(cyclical_df)

        resolved_pairs: List[Tuple[str, str, str]] = []
        if ratio_pairs:
            for pair in ratio_pairs:
                if len(pair) == 3:
                    resolved_pairs.append((pair[0], pair[1], pair[2]))
                elif len(pair) == 2:
                    resolved_pairs.append((pair[0], pair[1], f"{pair[0]}_per_{pair[1]}"))
        elif len(num_cols) >= 2:
            col1, col2 = num_cols[0], num_cols[1]
            resolved_pairs.append((col1, col2, f"{col1}_per_{col2}"))

        ratio_tf = RatioFeatureTransformer(ratio_pairs=resolved_pairs)
        synthesized_df = ratio_tf.fit_transform(cyclical_df)

        skipped_groups: List[Dict[str, Any]] = []
        if group_specs:
            group_tf = GroupByAggregationTransformer(group_specs=group_specs, max_cardinality=max_cardinality, smoothing=smoothing)
            synthesized_df = group_tf.fit_transform(synthesized_df)
            skipped_groups = getattr(group_tf, "skipped_specs_", [])

        new_columns = [c for c in synthesized_df.columns if c not in df.columns]

        saved_csv_path = persist_processed_dataframe(
            synthesized_df,
            source_path=csv_path,
            suffix="synthesized",
            output_path=output_path,
        )

        return {
            "status": "success",
            "processed_csv_path": saved_csv_path,
            "synthesized_columns_count": len(new_columns),
            "synthesized_columns": new_columns,
            "total_columns": len(synthesized_df.columns),
            "skipped_high_cardinality_groups": skipped_groups,
        }

    def prune_features(
        self,
        csv_path: str,
        target_column: str,
        top_k: Optional[int] = None,
        importance_threshold: float = 0.005,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        cv_splits: int = 3,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        selector = GradientFeatureSelector(top_k=top_k, importance_threshold=importance_threshold, task_type=task_type, cv=cv_splits)
        selector.fit(X, y)
        X_pruned = selector.transform(X)

        pruned_df = X_pruned.copy() if isinstance(X_pruned, pd.DataFrame) else pd.DataFrame(X_pruned, columns=selector.selected_features_)
        pruned_df[target_column] = y.values

        saved_csv_path = persist_processed_dataframe(
            pruned_df,
            source_path=csv_path,
            suffix="pruned",
            output_path=output_path,
        )

        report = selector.get_report()
        data_dump = report.model_dump()
        data_dump["processed_csv_path"] = saved_csv_path

        return {
            "status": "success",
            "processed_csv_path": saved_csv_path,
            "data": data_dump,
        }

    def transform_target(
        self,
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
        values: Optional[List[float]] = None,
        skew_threshold: float = 1.5,
        method: Literal["auto", "log1p", "yeo-johnson"] = "auto",
    ) -> Dict[str, Any]:
        if csv_path and target_column:
            df = self.repository.load_dataframe(csv_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in CSV.")
            y_data = df[target_column].dropna().values
        elif values is not None:
            y_data = np.array(values, dtype=np.float64)
        else:
            raise ValueError("Must provide either (csv_path and target_column) or values.")

        transformer = SkewedTargetTransformer(skew_threshold=skew_threshold)
        return transformer.transform_target(y_data, method=method)
