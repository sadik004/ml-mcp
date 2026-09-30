"""Integration tests for the complete ML MCP pipeline."""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ml_mcp.engine.auditor import DatasetAuditor
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.engine.tournament import TournamentArena
from ml_mcp.engine.threshold import DecisionThresholdOptimizer
from ml_mcp.engine.explainer import TreeShapExplainer
from ml_mcp.engine.exporter import ModelExporter


def test_end_to_end_fast_pipeline(tmp_path):
    """Verify entire pipeline: Audit -> Pipe -> Tournament -> Threshold -> SHAP -> Export."""
    # 1. Synthetic dataset
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=42)
    feature_names = [f"f_{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # 2. Audit
    auditor = DatasetAuditor()
    audit_report = auditor.audit_dataset(df, target_column="target", task_type="classification")
    assert audit_report.row_count == 150

    # 3. Pipeline
    builder = DefensivePipelineBuilder()
    pipe = builder.build_pipeline(df, target_column="target")
    X_trans = pipe.fit_transform(df.drop(columns=["target"]), y)
    assert X_trans.shape[0] == 150

    # 4. Fast Tournament
    df_trans = pd.DataFrame(X_trans, columns=feature_names)
    df_trans["target"] = y
    arena = TournamentArena(cv_splits=2, fast_mode=True, force_cpu=True)
    leaderboard = arena.run_tournament(df_trans, target_column="target", task_type="classification")
    assert leaderboard.champion_model != ""
    assert len(leaderboard.leaderboard) >= 7

    # 5. Threshold Optimization
    best_model = arena.champion_estimator
    val_probas = best_model.predict_proba(X_trans)[:, 1]
    opt = DecisionThresholdOptimizer()
    thresh_report = opt.optimize(y, val_probas)
    assert thresh_report.optimal_threshold > 0.0

    # 6. SHAP Explainability
    explainer = TreeShapExplainer()
    shap_report = explainer.explain(best_model, X_trans, feature_names=feature_names)
    assert len(shap_report["top_features"]) <= 10

    # 7. Model Export
    exporter = ModelExporter()
    export_dto = exporter.export_model_bundle(
        model=best_model,
        sample_input=X_trans[:2],
        output_dir=str(tmp_path),
        model_name="Champion_Model",
    )
    assert os.path.exists(export_dto.joblib_path)
    assert os.path.exists(export_dto.model_card_path)
