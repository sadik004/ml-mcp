"""Phase 3 Model Tournament, Benchmarks, Tuning & Stacking Service."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Literal, Optional

import pandas as pd

from ml_mcp.engine.checkpoint_manager import CheckpointManager
from ml_mcp.engine.tournament import TournamentArena
from ml_mcp.engine.tournament_orchestrator import TournamentOrchestrator
from ml_mcp.engine.tuner import BayesianTuner
from ml_mcp.services.base import BaseService
from ml_mcp.services.job_registry import get_job_registry

logger = logging.getLogger(__name__)


class ModelService(BaseService):
    """Orchestrates model tournament, benchmarks, hyperparameter tuning, stacking, and line tracking."""

    def run_model_tournament(
        self,
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
        if not csv_path:
            for cand in [
                os.path.join(".artifacts", "processed", "transformed_dataset.csv"),
                os.path.join(os.getcwd(), ".artifacts", "processed", "transformed_dataset.csv"),
            ]:
                if self.repository.file_exists(cand):
                    csv_path = cand
                    break
            if not csv_path:
                raise ValueError("csv_path was not provided and no transformed dataset found at .artifacts/processed/transformed_dataset.csv")

        meta_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(csv_path)), "feature_metadata.json"),
            os.path.join(".artifacts", "processed", "feature_metadata.json"),
            os.path.join(os.getcwd(), ".artifacts", "processed", "feature_metadata.json"),
        ]
        for mf_path in meta_candidates:
            if self.repository.file_exists(mf_path):
                try:
                    with open(mf_path, "r", encoding="utf-8") as mf:
                        fmeta = json.load(mf)
                        if not target_column:
                            target_column = fmeta.get("target_column")
                        if not class_weights:
                            class_weights = fmeta.get("class_weights")
                        if "task_type" in fmeta and fmeta["task_type"] in ["classification", "regression"]:
                            task_type = fmeta["task_type"]
                    if target_column:
                        break
                except Exception as meta_err:
                    logger.warning("Failed parsing metadata candidate '%s': %s", mf_path, meta_err)

        if not target_column:
            raise ValueError("target_column must be specified if not found in feature_metadata.json")

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        orchestrator = TournamentOrchestrator()
        report = orchestrator.run_tournament_and_tuning(
            X=X,
            y=y,
            task_type=task_type,
            primary_metric=primary_metric,
            n_splits=n_splits,
            tune_trials=tune_trials,
            overfit_penalty_lambda=overfit_penalty_lambda,
            class_weights=class_weights,
        )
        return report.to_compact() if view == "compact" else report.model_dump()

    def benchmark_models(
        self,
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        cv_splits: int = 5,
        scoring: Optional[str] = None,
        group_column: Optional[str] = None,
        fast_mode: bool = False,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        arena = TournamentArena(cv_splits=cv_splits, fast_mode=fast_mode)
        leaderboard = arena.run_tournament(
            df,
            target_column=target_column,
            task_type=task_type,
            group_column=group_column,
            scoring=scoring,
        )
        return leaderboard.to_compact() if view == "compact" else leaderboard.model_dump()

    def create_ensemble(
        self,
        csv_path: str,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        arena = TournamentArena(cv_splits=3, fast_mode=True)
        leaderboard = arena.run_tournament(df, target_column=target_column, task_type=task_type)

        from ml_mcp.engine.stacking_engine import StackingEngine
        stack_engine = StackingEngine()

        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            base_models = [
                ("lr", LogisticRegression(max_iter=300, random_state=self.settings.random_state)),
                ("rf", RandomForestClassifier(n_estimators=10, random_state=self.settings.random_state)),
            ]
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            base_models = [
                ("ridge", Ridge(alpha=1.0, random_state=self.settings.random_state)),
                ("rf", RandomForestRegressor(n_estimators=10, random_state=self.settings.random_state)),
            ]

        has_non_numeric = any(X[col].dtype == "object" or isinstance(X[col].dtype, pd.StringDtype) or X[col].isnull().any() for col in X.columns)
        preprocessor = None
        if has_non_numeric:
            from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
            builder = DefensivePipelineBuilder()
            preprocessor = builder.build_pipeline(df, target_column=target_column)
            X = preprocessor.fit_transform(X, y)

        stack_model, oof_score = stack_engine.build_stacking_ensemble(
            base_models=base_models,
            X=X,
            y=y,
            task_type=task_type,
            cv_splits=3,
        )

        final_artifact = stack_model
        if preprocessor is not None:
            from sklearn.pipeline import Pipeline
            final_artifact = Pipeline([("preprocessor", preprocessor), ("ensemble", stack_model)])

        out_path = artifact_path or os.path.join(".artifacts", "models", "ensemble_model.joblib")
        self.repository.save_artifact(final_artifact, out_path)

        return {
            "status": "success",
            "artifact_path": os.path.abspath(out_path),
            "model_path": os.path.abspath(out_path),
            "champion": leaderboard.champion_model,
            "stacking_candidates": leaderboard.stacking_candidates,
            "oof_score": oof_score,
        }

    def track_lineage(
        self,
        csv_path: str,
        checkpoint_dir: str = "artifacts/checkpoints",
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        lineage = manager.create_lineage(df, checkpoint_path=os.path.join(checkpoint_dir, "model.joblib"))
        return lineage.model_dump()

    def tune_hyperparameters(
        self,
        csv_path: str,
        target_column: str,
        model_name: str = "lightgbm",
        n_trials: int = 10,
        task_type: str = "classification",
        metric: Optional[str] = None,
        group_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        default_metric = "roc_auc" if task_type == "classification" else "r2"
        eval_metric = metric or default_metric
        tuner = BayesianTuner(n_trials=n_trials)
        study_dto, _ = tuner.tune(
            model_name=model_name,
            X=X,
            y=y,
            task_type=task_type,
            metric=eval_metric,
            group_column=group_column,
        )
        return study_dto.to_compact()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return get_job_registry().cancel_job(job_id)
