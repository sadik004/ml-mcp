"""Production Multi-Stage Dockerfile and Docker Compose Generator."""
from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


class DockerGenerator:
    """Generates hardened, multi-stage Dockerfiles and docker-compose.yml for ML model serving."""

    def __init__(self) -> None:
        pass

    def generate_docker_spec(
        self,
        output_dir: str,
        service_name: str = "ml-serving-api",
        port: int = 8000,
    ) -> Dict[str, str]:
        """Synthesize multi-stage Dockerfile and docker-compose.yml."""
        os.makedirs(output_dir, exist_ok=True)

        dockerfile_content = f"""# Multi-Stage Hardened Production Dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Minimal Runtime Stage
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root user for enterprise container security
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
"""
        dockerfile_path = os.path.join(output_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        compose_content = f"""version: "3.8"

services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: {service_name}
    restart: unless-stopped
    ports:
      - "{port}:{port}"
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 5s
      retries: 3
"""
        compose_path = os.path.join(output_dir, "docker-compose.yml")
        with open(compose_path, "w", encoding="utf-8") as f:
            f.write(compose_content)

        return {"Dockerfile": os.path.abspath(dockerfile_path), "docker-compose.yml": os.path.abspath(compose_path)}
