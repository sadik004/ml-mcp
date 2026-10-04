"""Phase 5 Serving, Deployment, Drift Monitoring & Pseudo-labeling Service."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_mcp.engine.api_generator import APIGenerator
from ml_mcp.engine.batch_predictor import BatchPredictor
from ml_mcp.engine.dashboard_generator import DashboardGenerator
from ml_mcp.engine.docker_generator import DockerGenerator
from ml_mcp.engine.drift_monitor import DataDriftMonitor
from ml_mcp.engine.exporter import ModelExporter
from ml_mcp.engine.onnx_optimizer import ONNXOptimizer
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.engine.pseudo_labeler import PseudoLabeler
from ml_mcp.services.base import BaseService

logger = logging.getLogger(__name__)


class ServingService(BaseService):
    """Orchestrates model batch inference, packaging, ONNX export, serving API generation, and drift monitoring."""

    def batch_predict(
        self,
        model_path: str,
        input_csv_path: str,
        output_csv_path: str,
        id_column: Optional[str] = None,
        task_type: str = "classification",
        optimal_threshold: Optional[float] = None,
        calibrator_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform chunked high-throughput batch inference with optional DCA cutoff and calibration."""
        model = self.repository.load_model(model_path)
        calibrator = self.repository.load_model(calibrator_path) if calibrator_path else None
        predictor = BatchPredictor()
        dto = predictor.predict_csv(
            model,
            input_csv_path,
            output_csv_path,
            id_column=id_column,
            task_type=task_type,
            optimal_threshold=optimal_threshold,
            calibrator=calibrator,
        )
        return dto.to_compact()

    def export_and_document(
        self,
        csv_path: str,
        target_column: str,
        output_dir: str = "artifacts/bundle",
        model_name: str = "champion_model",
    ) -> Dict[str, Any]:
        """Export atomic .joblib, .onnx, and synthesize official MODEL_CARD.md."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, r2_score

        from ml_mcp.services.evaluation import fit_final, oof_predict

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        has_non_numeric = any(
            X[col].dtype == "object"
            or isinstance(X[col].dtype, pd.StringDtype)
            or X[col].isnull().any()
            for col in X.columns
        )
        base_clf = RandomForestClassifier(n_estimators=15, random_state=self.settings.random_state)
        is_classification = (
            y.dtype == "object"
            or str(y.dtype).startswith("cat")
            or len(np.unique(y)) <= 10
        )
        if not is_classification:
            from sklearn.ensemble import RandomForestRegressor
            base_clf = RandomForestRegressor(n_estimators=15, random_state=self.settings.random_state)

        if has_non_numeric:
            builder = DefensivePipelineBuilder()
            model_to_eval = builder.build_pipeline(df, target_column=target_column, estimator=base_clf)
        else:
            model_to_eval = base_clf

        oof_preds = oof_predict(model_to_eval, X, y, cv=self.settings.cv_splits)
        if is_classification:
            score = float(accuracy_score(y, oof_preds))
            metrics_dict = {"accuracy": score, "oof_score": score}
        else:
            score = float(r2_score(y, oof_preds))
            metrics_dict = {"r2": score, "oof_score": score}

        final_clf = fit_final(model_to_eval, X, y)
        exporter = ModelExporter()
        sample_input = X.iloc[:2] if isinstance(X, pd.DataFrame) else X[:2]
        dto = exporter.export_model_bundle(
            final_clf,
            sample_input,
            output_dir=output_dir,
            model_name=model_name,
            metrics=metrics_dict,
            score=score,
            trained_on="full_data",
        )
        return dto.to_compact()

    def optimize_inference(
        self,
        csv_path: str,
        target_column: str,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert model to ONNX format with graph optimization and benchmark latency."""
        from sklearn.ensemble import RandomForestClassifier

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        has_non_numeric = any(
            X[col].dtype == "object"
            or isinstance(X[col].dtype, pd.StringDtype)
            or X[col].isnull().any()
            for col in X.columns
        )
        if has_non_numeric:
            builder = DefensivePipelineBuilder()
            pipe = builder.build_pipeline(df, target_column=target_column)
            X = pipe.fit_transform(X, y)

        if model_path and os.path.exists(model_path):
            clf = self.repository.load_model(model_path)
        else:
            clf = RandomForestClassifier(n_estimators=25, random_state=self.settings.random_state)
            clf.fit(X, y)

        optimizer = ONNXOptimizer()
        benchmark_sample = X[:100] if len(X) >= 100 else X
        _, p95_ms, p99_ms = optimizer.convert_and_benchmark(clf, benchmark_sample)
        return {
            "format": "onnx",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "p95_latency_ms": p95_ms,
            "p99_latency_ms": p99_ms,
        }

    def generate_eval_dashboard(
        self,
        output_html_path: str,
        project_name: str = "ML Production Model",
        champion_model: str = "Champion",
        metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        holdout_csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize interactive single-file HTML evaluation dashboard."""
        import time

        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            f1_score,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )

        metric_dict: Dict[str, Any] = {}
        if metrics:
            metric_dict = dict(metrics)
        elif model_path and holdout_csv_path and target_column:
            if not self.repository.file_exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not self.repository.file_exists(holdout_csv_path):
                raise FileNotFoundError(f"Holdout CSV not found: {holdout_csv_path}")

            model = self.repository.load_model(model_path)
            df = self.repository.load_dataframe(holdout_csv_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in holdout dataset.")

            X_holdout = df.drop(columns=[target_column])
            y_holdout = df[target_column]

            t0 = time.perf_counter()
            preds = model.predict(X_holdout)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Classification vs Regression check
            is_classification = False
            if hasattr(model, "classes_") or hasattr(model, "predict_proba"):
                is_classification = True
            elif pd.api.types.is_integer_dtype(y_holdout.dtype) or y_holdout.nunique() <= 20:
                is_classification = True

            if is_classification:
                metric_dict["Accuracy"] = round(float(accuracy_score(y_holdout, preds)), 4)
                metric_dict["Macro-F1"] = round(float(f1_score(y_holdout, preds, average="macro", zero_division=0)), 4)
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X_holdout)
                        if probs.shape[1] == 2:
                            metric_dict["ROC-AUC"] = round(float(roc_auc_score(y_holdout, probs[:, 1])), 4)
                            metric_dict["Brier-Score"] = round(float(brier_score_loss(y_holdout, probs[:, 1])), 4)
                        else:
                            metric_dict["ROC-AUC"] = round(float(roc_auc_score(y_holdout, probs, multi_class="ovr")), 4)
                    except Exception as e:
                        logger.warning(f"Could not compute probability metrics: {e}")
            else:
                metric_dict["RMSE"] = round(float(np.sqrt(mean_squared_error(y_holdout, preds))), 4)
                metric_dict["R2"] = round(float(r2_score(y_holdout, preds)), 4)

            per_sample_latency = latency_ms / max(1, len(X_holdout))
            metric_dict["Latency_ms_per_sample"] = round(float(per_sample_latency), 3)
        else:
            raise ValueError(
                "Evaluation metrics missing: provide explicit 'metrics' dictionary or "
                "('model_path', 'holdout_csv_path', 'target_column') to compute empirical metrics on holdout data."
            )

        generator = DashboardGenerator()
        path = generator.generate_dashboard(
            output_path=output_html_path,
            project_name=project_name,
            champion_model=champion_model,
            metrics=metric_dict,
        )
        return {"dashboard_path": path, "status": "generated", "metrics": metric_dict}

    def generate_serving_api(
        self,
        output_dir: str,
        model_name: str = "champion_model",
        feature_names: Optional[List[str]] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate 3-Tier clean architecture FastAPI serving router and Pydantic v2 schemas."""
        feats = feature_names
        if not feats and model_path and os.path.exists(model_path):
            try:
                loaded = self.repository.load_model(model_path)
                if hasattr(loaded, "feature_names_in_"):
                    feats = list(loaded.feature_names_in_)
            except Exception as e:
                logger.warning(f"Could not extract feature_names_in_ from model: {e}")

        if not feats:
            raise ValueError(
                "feature_names are required to generate serving schemas. "
                "Pass feature_names explicitly or specify a model_path containing fitted feature_names_in_."
            )

        generator = APIGenerator()
        files = generator.generate_serving_scaffold(
            output_dir=output_dir, model_name=model_name, feature_names=feats
        )
        return {"generated_files": files, "status": "success"}

    def generate_docker_spec(
        self,
        output_dir: str,
        service_name: str = "ml-serving",
        port: int = 8000,
    ) -> Dict[str, Any]:
        """Generate hardened multi-stage Dockerfile and docker-compose.yml spec."""
        generator = DockerGenerator()
        files = generator.generate_docker_spec(
            output_dir=output_dir, service_name=service_name, port=port
        )
        return {"generated_files": files, "status": "success"}

    def monitor_drift(
        self,
        reference_csv_path: str,
        current_csv_path: str,
    ) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) and KS-test for data drift detection."""
        df_ref = self.repository.load_dataframe(reference_csv_path).select_dtypes(include=[np.number])
        df_curr = self.repository.load_dataframe(current_csv_path).select_dtypes(include=[np.number])
        monitor = DataDriftMonitor()
        report = monitor.detect_drift(df_ref, df_curr)
        return report.to_compact()

    def pseudo_label_loop(
        self,
        train_csv_path: str,
        unlabelled_csv_path: str,
        target_column: str,
        confidence_threshold: float = 0.95,
        alpha: float = 0.10,
    ) -> Dict[str, Any]:
        """Harvest high-confidence pseudo-labels using curriculum thresholds."""
        from sklearn.ensemble import RandomForestClassifier

        df_tr = self.repository.load_dataframe(train_csv_path)
        df_unlab = self.repository.load_dataframe(unlabelled_csv_path)
        X_tr = df_tr.drop(columns=[target_column])
        y_tr = df_tr[target_column]
        clf = RandomForestClassifier(n_estimators=15, random_state=self.settings.random_state)
        clf.fit(X_tr, y_tr)
        labeler = PseudoLabeler(
            confidence_threshold=confidence_threshold,
            alpha=alpha,
            random_state=self.settings.random_state,
        )
        stats, _ = labeler.refine_with_pseudo_labels(clf, X_tr, y_tr, df_unlab)
        return stats

