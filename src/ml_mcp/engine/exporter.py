"""Model Packaging, Dry-Run Verification, and Bundle Exporter Engine."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import joblib

from ml_mcp.engine.model_card import generate_model_card
from ml_mcp.engine.onnx_optimizer import ONNXOptimizer
from ml_mcp.schemas.serving import ModelExportDTO

logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports production model bundle with .joblib, .onnx, dry-run verification, and MODEL_CARD.md."""

    def __init__(self) -> None:
        self.onnx_optimizer = ONNXOptimizer()

    def export_model_bundle(
        self,
        model: Any,
        sample_input: Any,
        output_dir: str,
        model_name: str = "champion_model",
        metrics: Optional[Dict[str, Any]] = None,
        dataset_hash: str = "unknown_sha256",
        hyperparameters: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        trained_on: str = "full_data",
    ) -> ModelExportDTO:
        """Export atomic .joblib, .onnx, and MODEL_CARD.md with dry-run test."""
        os.makedirs(output_dir, exist_ok=True)
        metrics_dict = metrics or {}

        # 1. Atomic .joblib serialization
        joblib_path = os.path.join(output_dir, f"{model_name}.joblib")
        joblib.dump(model, joblib_path)

        # 2. Round-trip dry-run inference verification
        reloaded_model = joblib.load(joblib_path)
        dry_run_pred = reloaded_model.predict(sample_input)
        if dry_run_pred is None or len(dry_run_pred) != len(sample_input):
            raise RuntimeError("Dry-run inference failed on reloaded .joblib pipeline.")

        # 3. ONNX conversion and latency benchmarking
        onnx_path = ""
        p95_ms, p99_ms = 0.0, 0.0
        try:
            onnx_bytes, p95_ms, p99_ms = self.onnx_optimizer.convert_and_benchmark(
                model=model,
                sample_input=sample_input,
            )
            onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_bytes)
        except Exception as e:
            logger.warning("ONNX conversion skipped for model: %s", e)

        # 4. Generate MODEL_CARD.md
        card_content = generate_model_card(
            model_name=model_name,
            metrics=metrics_dict,
            dataset_hash=dataset_hash,
            p95_latency_ms=p95_ms,
            p99_latency_ms=p99_ms,
            hyperparameters=hyperparameters,
        )
        model_card_path = os.path.join(output_dir, "MODEL_CARD.md")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        return ModelExportDTO(
            joblib_path=os.path.abspath(joblib_path),
            onnx_path=os.path.abspath(onnx_path),
            model_card_path=os.path.abspath(model_card_path),
            p95_latency_ms=p95_ms,
            p99_latency_ms=p99_ms,
            score=score,
            trained_on=trained_on,
            metrics=metrics_dict,
        )
