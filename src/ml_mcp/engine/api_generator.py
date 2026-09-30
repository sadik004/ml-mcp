"""FastAPI 3-Tier Clean Architecture Serving Code Generator."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class APIGenerator:
    """Synthesizes production-ready 3-tier FastAPI router code and Pydantic v2 DTOs."""

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
        """Synthesize schemas.py and main.py matching enterprise FastAPI standards."""
        os.makedirs(output_dir, exist_ok=True)
        types_map = feature_types or {f: "float" for f in feature_names}

        # 1. Generate schemas.py
        fields_code = ""
        for feat in feature_names:
            py_type = types_map.get(feat, "float")
            if py_type not in ["int", "float", "str", "bool"]:
                py_type = "float"
            fields_code += f"    {feat}: {py_type} = Field(..., description='Feature value for {feat}')\n"

        schemas_py = f'''"""Pydantic v2 Data Transfer Objects for {model_name} Serving."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequestDTO(BaseModel):
    """Inference request schema with strict field validation."""
    model_config = ConfigDict(extra="forbid")

{fields_code}


class BatchPredictionRequestDTO(BaseModel):
    """Batch inference request payload."""
    records: List[PredictionRequestDTO] = Field(..., description="List of feature records")


class PredictionResponseDTO(BaseModel):
    """Single-record prediction response schema."""
    prediction: Any = Field(..., description="Model prediction output")
    confidence: Optional[float] = Field(None, description="Prediction probability confidence")
    latency_ms: float = Field(..., description="Inference execution time in milliseconds")


class HealthResponseDTO(BaseModel):
    """Service liveness and readiness response."""
    status: str = "healthy"
    model_name: str = "{model_name}"
    version: str = "1.0.0"
'''
        schemas_path = os.path.join(output_dir, "schemas.py")
        with open(schemas_path, "w", encoding="utf-8") as f:
            f.write(schemas_py)

        # 2. Generate main.py
        main_py = f'''"""Production FastAPI Serving Application for {model_name}."""
from __future__ import annotations

import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from schemas import (
    PredictionRequestDTO,
    BatchPredictionRequestDTO,
    PredictionResponseDTO,
    HealthResponseDTO,
)

app = FastAPI(
    title="{model_name} Inference API",
    description="3-Tier Clean Architecture Production Endpoint generated via ml-mcp",
    version="1.0.0",
)

MODEL_PATH = "{model_name}.joblib"
model = None

@app.on_event("startup")
def load_pipeline():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load model pipeline from {{MODEL_PATH}}: {{e}}")

@app.get("/health", response_model=HealthResponseDTO, tags=["Health"])
def health_check():
    return HealthResponseDTO()

@app.post("/predict", response_model=PredictionResponseDTO, tags=["Inference"])
def predict(request: PredictionRequestDTO):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    start_time = time.perf_counter()
    input_data = pd.DataFrame([request.model_dump()])
    
    try:
        pred = model.predict(input_data)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(input_data)[0]
            confidence = float(np.max(probas))
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return PredictionResponseDTO(
            prediction=pred if not isinstance(pred, np.generic) else pred.item(),
            confidence=confidence,
            latency_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {{str(e)}}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
'''
        main_path = os.path.join(output_dir, "main.py")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(main_py)

        return {"schemas.py": os.path.abspath(schemas_path), "main.py": os.path.abspath(main_path)}
