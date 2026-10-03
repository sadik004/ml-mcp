"""Unit tests for FastAPI 3-Tier API Generator with lifespan handlers and Non-Root Dockerfile."""
import os
import pytest

from ml_mcp.engine.api_generator import APIGenerator
from ml_mcp.engine.docker_generator import DockerGenerator


def test_api_generator_synthesizes_lifespan_fastapi_code(tmp_path):
    """Verify API generator produces modern lifespan context manager and health endpoints."""
    generator = APIGenerator()
    out_dir = tmp_path / "serving_api"

    files = generator.generate_serving_scaffold(
        output_dir=str(out_dir),
        model_name="Champion_CatBoost",
        feature_names=["age", "income", "credit_score"],
        feature_types={"age": "int", "income": "float", "credit_score": "float"},
        task_type="classification",
    )

    assert os.path.exists(files["schemas.py"])
    assert os.path.exists(files["main.py"])

    with open(files["main.py"], "r", encoding="utf-8") as f:
        code = f.read()

    # Must contain modern asynccontextmanager lifespan handler
    assert "asynccontextmanager" in code
    assert "lifespan" in code
    # Must contain CNCF health probes
    assert "/healthz" in code
    assert "/readyz" in code


def test_docker_generator_synthesizes_non_root_dockerfile(tmp_path):
    """Verify Docker generator creates multi-stage Dockerfile with non-root appuser."""
    generator = DockerGenerator()
    out_dir = tmp_path / "docker"

    files = generator.generate_docker_spec(
        output_dir=str(out_dir),
        service_name="churn-predictor",
        port=8080,
    )

    with open(files["Dockerfile"], "r", encoding="utf-8") as f:
        d_code = f.read()

    # Enforce non-root security benchmark
    assert "USER appuser" in d_code
    assert "10001" in d_code
    assert "HEALTHCHECK" in d_code
