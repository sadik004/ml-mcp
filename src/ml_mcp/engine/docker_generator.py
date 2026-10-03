"""Production Multi-Stage Dockerfile and Docker Compose Generator with Non-Root CIS Security."""
from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


class DockerGenerator:
    """Generates hardened, multi-stage Dockerfiles and docker-compose.yml enforcing CIS Non-Root benchmarks."""

    def __init__(self) -> None:
        pass

    def generate_docker_spec(
        self,
        output_dir: str,
        service_name: str = "ml-serving-api",
        port: int = 8000,
    ) -> Dict[str, str]:
        """Synthesize multi-stage Dockerfile and docker-compose.yml with non-root security."""
        os.makedirs(output_dir, exist_ok=True)

        dockerfile_content = f"""# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal Distroless/Slim Production Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# CIS Benchmark: Create and enforce unprivileged non-root user (appuser:10001)
RUN addgroup --system --gid 10001 appuser && \
    adduser --system --uid 10001 --ingroup appuser --home /app --no-create-home appuser

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . /app

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/healthz')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}", "--workers", "2"]
"""
        dockerfile_path = os.path.join(output_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        compose_content = f"""version: '3.8'

services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{port}:{port}"
    environment:
      - MODEL_PATH=/app/model.joblib
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2048M
"""
        compose_path = os.path.join(output_dir, "docker-compose.yml")
        with open(compose_path, "w", encoding="utf-8") as f:
            f.write(compose_content)

        return {"Dockerfile": dockerfile_path, "docker-compose.yml": compose_path}
