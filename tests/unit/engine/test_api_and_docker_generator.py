"""Unit tests for FastAPI 3-Tier API Generator and Dockerfile Synthesizer."""
import os
import pytest

from ml_mcp.engine.api_generator import APIGenerator
from ml_mcp.engine.docker_generator import DockerGenerator


def test_api_generator_synthesizes_fastapi_code(tmp_path):
    """Verify API generator produces valid Python FastAPI router and Pydantic schemas."""
    generator = APIGenerator()
    out_dir = tmp_path / "serving_api"

    files = generator.generate_serving_scaffold(
        output_dir=str(out_dir),
        model_name="Champion_CatBoost",
        feature_names=["age", "income", "credit_score"],
        feature_types={"age": "int", "income": "float", "credit_score": "float"},
        task_type="classification",
    )

    assert "main.py" in files
    assert "schemas.py" in files
    assert os.path.exists(files["main.py"])
    assert os.path.exists(files["schemas.py"])

    with open(files["main.py"], "r", encoding="utf-8") as f:
        main_code = f.read()
    assert "FastAPI" in main_code
    assert "/predict" in main_code
    assert "/health" in main_code


def test_docker_generator_synthesizes_dockerfile_and_compose(tmp_path):
    """Verify Docker generator creates multi-stage Dockerfile and docker-compose.yml."""
    generator = DockerGenerator()
    out_dir = tmp_path / "docker_spec"

    files = generator.generate_docker_spec(
        output_dir=str(out_dir),
        service_name="ml-champion-serving",
        port=8000,
    )

    assert "Dockerfile" in files
    assert "docker-compose.yml" in files
    assert os.path.exists(files["Dockerfile"])
    assert os.path.exists(files["docker-compose.yml"])

    with open(files["Dockerfile"], "r", encoding="utf-8") as f:
        dockerfile = f.read()
    assert "FROM python:" in dockerfile
    assert "EXPOSE 8000" in dockerfile
