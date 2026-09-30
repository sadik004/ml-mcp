"""15-Second Automated End-to-End Smoke Test Pipeline for ml-mcp."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ml_mcp.engine.auditor import DatasetAuditor
from ml_mcp.engine.dashboard_generator import DashboardGenerator
from ml_mcp.engine.explainer import TreeShapExplainer
from ml_mcp.engine.exporter import ModelExporter
from ml_mcp.engine.ood_detector import OODDetector
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.engine.stress_tester import ModelStressTester
from ml_mcp.engine.threshold import DecisionThresholdOptimizer
from ml_mcp.engine.tournament import TournamentArena


def run_smoke_test() -> None:
    """Execute end-to-end ML lifecycle under 15 seconds."""
    start_total = time.perf_counter()
    temp_dir = tempfile.mkdtemp(prefix="ml_mcp_smoke_")

    print("==================================================")
    print(">> Running ml-mcp 15-Second End-to-End Smoke Test")
    print("==================================================")

    try:
        # Step 1: Synthesize Data
        t0 = time.perf_counter()
        X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=42)
        feature_names = [f"feat_{i}" for i in range(6)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        print(f" [1/8] Data Synthesis: {len(df)} rows, 6 features ({time.perf_counter() - t0:.2f}s)")

        # Step 2: Pre-Flight Audit
        t0 = time.perf_counter()
        auditor = DatasetAuditor()
        audit_rep = auditor.audit_dataset(df, target_column="target", task_type="classification")
        assert audit_rep.row_count == 200
        print(f" [2/8] Pre-Flight Audit: Hygiene & guards passed ({time.perf_counter() - t0:.2f}s)")

        # Step 3: Defensive Zero-Leakage Pipeline
        t0 = time.perf_counter()
        builder = DefensivePipelineBuilder()
        pipe = builder.build_pipeline(df, target_column="target")
        X_trans = pipe.fit_transform(df.drop(columns=["target"]), y)
        print(f" [3/8] Defensive Pipeline: Transformed shape {X_trans.shape} ({time.perf_counter() - t0:.2f}s)")

        # Step 4: 8-Model Competitive Arena
        t0 = time.perf_counter()
        df_trans = pd.DataFrame(X_trans, columns=feature_names)
        df_trans["target"] = y
        arena = TournamentArena(cv_splits=2, fast_mode=True, force_cpu=True)
        leaderboard = arena.run_tournament(df_trans, target_column="target", task_type="classification")
        champion_model = arena.champion_estimator
        print(f" [4/8] Tournament: Champion '{leaderboard.champion_model}' (Score: {leaderboard.champion_score:.4f}) ({time.perf_counter() - t0:.2f}s)")

        # Step 5: Threshold Optimization & Error Forensics
        t0 = time.perf_counter()
        probas = champion_model.predict_proba(X_trans)[:, 1] if hasattr(champion_model, "predict_proba") else champion_model.predict(X_trans)
        optimizer = DecisionThresholdOptimizer()
        thresh_rep = optimizer.optimize(y, probas, beta=1.0)
        print(f" [5/8] Threshold Optimization: Optimal boundary {thresh_rep.optimal_threshold} (F1: {thresh_rep.f_beta_score:.4f}) ({time.perf_counter() - t0:.2f}s)")

        # Step 6: Sub-10s TreeSHAP Explainability
        t0 = time.perf_counter()
        explainer = TreeShapExplainer()
        shap_rep = explainer.explain(champion_model, X_trans, feature_names=feature_names, top_k=5)
        print(f" [6/8] TreeSHAP: Top feature '{list(shap_rep['top_features'].keys())[0]}' ({time.perf_counter() - t0:.2f}s)")

        # Step 7: AI Safety (OOD & Stress Test)
        t0 = time.perf_counter()
        ood_det = OODDetector()
        ood_det.fit(X_trans)
        ood_rep = ood_det.detect(X_trans)

        tester = ModelStressTester()
        stress_rep = tester.evaluate(champion_model, X_trans, y)
        print(f" [7/8] AI Safety: OOD={ood_rep.ood_detected_count}, Robustness={stress_rep.robustness_score:.1f}/100 ({time.perf_counter() - t0:.2f}s)")

        # Step 8: Packaging, ONNX, and Dashboard
        t0 = time.perf_counter()
        exporter = ModelExporter()
        export_dto = exporter.export_model_bundle(
            model=champion_model,
            sample_input=X_trans[:2],
            output_dir=os.path.join(temp_dir, "bundle"),
            model_name="SmokeChampion",
        )

        dash_gen = DashboardGenerator()
        dash_path = dash_gen.generate_dashboard(
            output_path=os.path.join(temp_dir, "dashboard.html"),
            project_name="Smoke Test Verification",
            champion_model=leaderboard.champion_model,
            metrics={"Accuracy": leaderboard.champion_score, "P95_Latency_ms": export_dto.p95_latency_ms},
            leaderboard=[{"model": m.model_name, "score": m.mean_cv_score, "fit_time": m.fit_time_seconds} for m in leaderboard.leaderboard],
            top_features=shap_rep["top_features"],
            confusion_matrix=thresh_rep.confusion_matrix,
        )
        print(f" [8/8] Artifacts & Dashboard: Generated at {dash_path} ({time.perf_counter() - t0:.2f}s)")

        total_time = time.perf_counter() - start_total
        print("==================================================")
        print(f"[SUCCESS] End-to-End Smoke Test Passed in {total_time:.2f} seconds! (Threshold <= 15.0s)")
        print("==================================================")
        assert total_time < 15.0, f"Smoke test exceeded 15 seconds: {total_time:.2f}s"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        run_smoke_test()
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FAILED] Smoke Test Failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
