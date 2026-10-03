"""FastAPI 3-Tier Clean Architecture Serving Code Generator with modern lifespan handlers."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class APIGenerator:
    """Synthesizes production-ready 3-tier FastAPI router code, lifespan lifecycle, and Pydantic v2 DTOs."""

    def __init__(self) -> None:
        pass

    def generate_serving_scaffold(
        self,
        output_dir: str,
        model_name: str,
        feature_names: List[str],
        feature_types: Optional[Dict[str, str]] = None,
        task_type: str = "classification",
    ) -> Dict[str, str]:
        """Synthesize schemas.py and main.py matching enterprise FastAPI lifespan standards."""
        os.makedirs(output_dir, exist_ok=True)
        f_types = feature_types or {f: "float" for f in feature_names}

        # 1. Generate schemas.py
        fields_code = ""
        for feat in feature_names:
            py_type = f_types.get(feat, "float")
            fields_code += f"    {feat}: {py_type}\n"

        schemas_py = f"""\"\"\"Pydantic v2 Request/Response Data Transfer Objects.\"\"\"
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    \"\"\"Input features payload for real-time inference.\"\"\"
{fields_code}

class PredictionResponse(BaseModel):
    \"\"\"Inference response envelope.\"\"\"
    prediction: Any
    confidence_score: Optional[float] = None
    model_version: str = "{model_name}"
    latency_ms: float

class HealthResponse(BaseModel):
    \"\"\"CNCF Health probe status.\"\"\"
    status: str
    model_loaded: bool
"""
        schemas_path = os.path.join(output_dir, "schemas.py")
        with open(schemas_path, "w", encoding="utf-8") as f:
            f.write(schemas_py)

        # 2. Generate main.py with modern asynccontextmanager lifespan handler
        main_py = f"""\"\"\"FastAPI Model Serving Microservice with Lifespan Lifecycle Management.\"\"\"
import os
import time
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from schemas import PredictionRequest, PredictionResponse, HealthResponse

# Global model container
ml_models = {{}}

@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"Enterprise lifespan context manager loading model artifacts at startup.\"\"\"
    model_path = os.getenv("MODEL_PATH", "model.joblib")
    if os.path.exists(model_path):
        ml_models["model"] = joblib.load(model_path)
    else:
        ml_models["model"] = None
    yield
    # Clean up resources on shutdown
    ml_models.clear()

app = FastAPI(
    title="{model_name} Serving API",
    version="1.0.0",
    description="Production-grade 3-Tier Clean ML Inference API",
    lifespan=lifespan,
)

@app.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    \"\"\"Kubernetes / CNCF Liveness Probe.\"\"\"
    return HealthResponse(status="alive", model_loaded=ml_models.get("model") is not None)

@app.get("/readyz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def readiness_check():
    \"\"\"Kubernetes / CNCF Readiness Probe.\"\"\"
    is_ready = ml_models.get("model") is not None
    if not is_ready:
        raise HTTPException(status_code=503, detail="Model artifact not yet loaded")
    return HealthResponse(status="ready", model_loaded=True)

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    \"\"\"Execute real-time model scoring.\"\"\"
    model = ml_models.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    df_in = pd.DataFrame([payload.model_dump()])
    
    pred = model.predict(df_in)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(df_in)[0].max())

    latency = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        prediction=pred,
        confidence_score=conf,
        model_version="{model_name}",
        latency_ms=round(latency, 2),
    )
"""
        main_path = os.path.join(output_dir, "main.py")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(main_py)

        return {"schemas.py": schemas_path, "main.py": main_path}
