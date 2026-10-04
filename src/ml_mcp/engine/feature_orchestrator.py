"""Unified Phase 2 Feature Pipeline Master Orchestrator.

Coordinates all Phase 2 feature engineering engines:
1. High-Signal Feature Synthesis (Temporal harmonics sin/cos, Latent Manifold L2 Norm & Mahalanobis energy)
2. Zero-Leakage Defensive Pipeline Building (Adaptive RobustScaler, SimpleImputer, TargetEncoder)
3. Cost-Sensitive Class Balancing (Inverse frequency class weights)
4. Out-of-Fold Permutation / Gradient Feature Pruning (Eliminating curse of dimensionality)
5. Artifact Serialization (preprocessor.joblib and transformed matrix)
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from ml_mcp.config import get_settings
from ml_mcp.engine.balancer import ClassBalancer
from ml_mcp.engine.feature_pruner import GradientFeatureSelector
from ml_mcp.engine.feature_synthesizer import (
    AutomaticTemporalTransformer,
    LatentManifoldOutlierTransformer,
)
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.schemas.feature import FeaturePipelineReportDTO

logger = logging.getLogger(__name__)


class FeaturePipelineOrchestrator:
    """Master single-pass orchestrator executing the full Phase 2 Feature Engineering Suite."""

    def __init__(self, artifact_dir: str = ".artifacts/processed") -> None:
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def prepare_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        enable_synthesis: bool = True,
        enable_pruning: bool = True,
        train_indices: Optional[List[int]] = None,
    ) -> FeaturePipelineReportDTO:
        """Executes all Phase 2 transformations and generates ready-to-train artifacts."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        warnings: List[str] = []
        y = df[target_column]
        X_raw = df.drop(columns=[target_column]).copy()
        initial_features = list(X_raw.columns)
        initial_shape = [int(len(df)), int(len(initial_features))]

        # Zero-Leakage Split Partitioning
        if train_indices is not None:
            train_idx = np.asarray(train_indices, dtype=int)
        elif len(df) >= 20:
            from sklearn.model_selection import train_test_split
            train_idx, _ = train_test_split(
                np.arange(len(df)),
                test_size=0.20,
                random_state=get_settings().random_state,
                stratify=y if (task_type == "classification" and len(np.unique(y)) == 2) else None,
            )
        else:
            train_idx = np.arange(len(df))

        X_train = X_raw.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # 1. Feature Synthesis
        synthesized_cols: List[str] = []
        X_synth = X_raw.copy()
        if enable_synthesis:
            # Temporal Harmonics
            temp_trans = AutomaticTemporalTransformer()
            temp_trans.fit(X_train)
            X_synth = temp_trans.transform(X_synth)

            # Latent Manifold Outliers
            latent_trans = LatentManifoldOutlierTransformer()
            latent_trans.fit(temp_trans.transform(X_train))
            X_synth = latent_trans.transform(X_synth)

            synthesized_cols = [c for c in X_synth.columns if c not in initial_features]

        # 2. Zero-Leakage Defensive Preprocessor
        pipeline_builder = DefensivePipelineBuilder()
        preprocessor = pipeline_builder.build_pipeline(X_synth.iloc[train_idx], target_column=None)
        preprocessor.fit(X_synth.iloc[train_idx])
        X_transformed = preprocessor.transform(X_synth)

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Handle feature naming
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]
        else:
            feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]

        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)

        # 3. Class Balancing Weights
        class_weights: Optional[Dict[str, float]] = None
        if task_type == "classification" and len(np.unique(y_train)) >= 2:
            balancer = ClassBalancer()
            raw_weights = balancer.compute_class_weights(y_train)
            class_weights = {str(k): round(float(v), 4) for k, v in raw_weights.items()}

        # 4. Out-of-Fold Permutation Feature Pruning
        pruned_features: List[str] = []
        retained_features = list(X_transformed_df.columns)
        if enable_pruning and X_transformed_df.shape[1] > 8:
            try:
                selector = GradientFeatureSelector(task_type=task_type)
                selector.fit(X_transformed_df.iloc[train_idx], y_train)
                X_pruned = selector.transform(X_transformed_df)
                pruning_rep = selector.get_report()
                pruned_features = pruning_rep.dropped_features
                retained_features = pruning_rep.selected_features
                X_final = pd.DataFrame(X_pruned, columns=retained_features, index=df.index)
            except Exception as e:
                msg = f"Feature pruning fallback: {e}"
                logger.warning(msg)
                warnings.append(msg)
                X_final = X_transformed_df
        else:
            X_final = X_transformed_df

        final_shape = [int(len(X_final)), int(X_final.shape[1])]

        # 5. Persist Fitted Artifacts and Processed Dataset
        preprocessor_path = os.path.join(self.artifact_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)

        # Attach target column to produce fully ready transformed dataset for Phase 3
        transformed_df = X_final.copy()
        transformed_df[target_column] = y.values
        dataset_path = os.path.join(self.artifact_dir, "transformed_dataset.csv")
        transformed_df.to_csv(dataset_path, index=False)

        # Save metadata for downstream Phase 3 consumption
        import json
        meta_path = os.path.join(self.artifact_dir, "feature_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({
                    "target_column": target_column,
                    "task_type": task_type,
                    "class_weights": class_weights,
                    "retained_features": retained_features,
                    "train_samples_count": len(train_idx),
                    "zero_leakage_split": True,
                }, mf, indent=2)
        except Exception as e:
            logger.warning(f"Could not save feature metadata: {e}")

        # ----------------------------------------------------------------------
        # Executive Feature Receipt Card
        # ----------------------------------------------------------------------
        card_lines = [
            "=" * 88,
            "✨ ML-MCP PHASE 2: FEATURE PIPELINE GENERATION COMPLETE",
            f"Transformed Feature Matrix: ({final_shape[0]:,} samples × {final_shape[1]} features)",
            "=" * 88,
            "\n📊 FEATURE SPACE EVOLUTION:",
            f"  • Raw Input Features         : {initial_shape[1]} features",
            f"  • Synthesized New Features   : +{len(synthesized_cols)} ({', '.join(synthesized_cols) if synthesized_cols else 'None'})",
            f"  • Pruned Low-Signal Features : -{len(pruned_features)} ({', '.join(pruned_features[:5]) if pruned_features else 'None'})",
            f"  • Final High-Signal Space    : {final_shape[1]} production features",
            "\n⚙️ DEFENSIVE PIPELINE RECIPES:",
            "  • Imputer Safeguard          : SimpleImputer with Median/Mode (Guaranteed zero NaN inference)",
            "  • Adaptive Scalers           : RobustScaler (Skewed features) + StandardScaler (Normal features)",
            f"  • Saved Pipeline Artifact    : {preprocessor_path}",
        ]

        if class_weights:
            card_lines.append("\n⚖️ CLASS BALANCING WEIGHTS (Anti-SMOTE Cost-Sensitive):")
            for cls_val, w in class_weights.items():
                card_lines.append(f"  • Class {cls_val}: weight = {w:.4f}")

        card_lines.extend([
            "=" * 88,
            "👉 READY FOR PHASE 3: Competitive Model Arena & Anti-Overfit Bayesian Tuning!",
            "=" * 88,
        ])

        receipt_card = "\n".join(card_lines)

        return FeaturePipelineReportDTO(
            original_shape=initial_shape,
            transformed_shape=final_shape,
            synthesized_features=synthesized_cols,
            pruned_features=pruned_features,
            retained_features=retained_features,
            class_weights=class_weights,
            preprocessor_artifact_path=preprocessor_path,
            transformed_dataset_path=dataset_path,
            receipt_card=receipt_card,
            warnings=warnings,
        )
