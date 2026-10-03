"""FastAPI 3-Tier Clean Architecture Serving Code Generator with modern lifespan handlers and ONNX hybrid support."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class APIGenerator:
    """Synthesizes production-ready 3-tier FastAPI router code, lifespan lifecycle, requirements.txt, and ONNX runtime support."""

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
        """Synthesize schemas.py, main.py, and requirements.txt matching enterprise FastAPI lifespan standards."""
        os.makedirs(output_dir, exist_ok=True)
        f_types = feature_types or {f: "float" for f in feature_names}

        # 1. Generate requirements.txt (PEP 508 Pinned Dependencies)
        requirements_txt = (
            "fastapi>=0.100.0\n"
            "uvicorn[standard]>=0.23.0\n"
            "pydantic>=2.0.0\n"
            "joblib>=1.3.0\n"
            "numpy>=1.24.0\n"
            "pandas>=2.0.0\n"
            "onnxruntime>=1.16.0\n"
            "scikit-learn>=1.3.0\n"
        )
        requirements_path = os.path.join(output_dir, "requirements.txt")
        with open(requirements_path, "w", encoding="utf-8") as f:
            f.write(requirements_txt)

        # 2. Generate schemas.py
        fields_code = ""
        for feat in feature_names:
            py_type = f_types.get(feat, "float")
            fields_code += f"    {feat}: {py_type}\n"

        schemas_py = (
            '"""Pydantic v2 Request/Response Data Transfer Objects."""\n'
            "from __future__ import annotations\n"
            "from typing import Any, Dict, List, Optional\n"
            "from pydantic import BaseModel, Field\n\n"
            "class PredictionRequest(BaseModel):\n"
            '    """Input features payload for real-time inference."""\n'
            f"{fields_code}\n"
            "class PredictionResponse(BaseModel):\n"
            '    """Inference response envelope."""\n'
            "    prediction: Any\n"
            "    confidence_score: Optional[float] = None\n"
            f'    model_version: str = "{model_name}"\n'
            "    runtime_engine: str\n"
            "    latency_ms: float\n\n"
            "class HealthResponse(BaseModel):\n"
            '    """CNCF Health probe status."""\n'
            "    status: str\n"
            "    model_loaded: bool\n"
            "    runtime_engine: str\n"
        )
        schemas_path = os.path.join(output_dir, "schemas.py")
        with open(schemas_path, "w", encoding="utf-8") as f:
            f.write(schemas_py)

        # 3. Generate main.py with ONNX Hybrid Loader & Zero-Copy NumPy inference
        feats_list_repr = repr(feature_names)
        main_py = (
            '"""FastAPI Model Serving Microservice with Lifespan Lifecycle Management and ONNX Acceleration."""\n'
            "import os\n"
            "import time\n"
            "from contextlib import asynccontextmanager\n"
            "import joblib\n"
            "import numpy as np\n"
            "from fastapi import FastAPI, HTTPException, status\n"
            "from schemas import PredictionRequest, PredictionResponse, HealthResponse\n\n"
            'runtime_state = {"model": None, "engine": "none", "input_name": None}\n'
            f"FEATURE_NAMES = {feats_list_repr}\n\n"
            "@asynccontextmanager\n"
            "async def lifespan(app: FastAPI):\n"
            f'    onnx_path = os.getenv("ONNX_MODEL_PATH", "{model_name}.onnx")\n'
            f'    joblib_path = os.getenv("MODEL_PATH", "{model_name}.joblib")\n\n'
            "    if os.path.exists(onnx_path):\n"
            "        try:\n"
            "            import onnxruntime as ort\n"
            "            sess_options = ort.SessionOptions()\n"
            "            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
            '            sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])\n'
            '            runtime_state["model"] = sess\n'
            '            runtime_state["engine"] = "onnx_runtime"\n'
            '            runtime_state["input_name"] = sess.get_inputs()[0].name\n'
            "        except Exception:\n"
            '            runtime_state["model"] = None\n\n'
            '    if runtime_state["model"] is None and os.path.exists(joblib_path):\n'
            '        runtime_state["model"] = joblib.load(joblib_path)\n'
            '        runtime_state["engine"] = "joblib_pipeline"\n\n'
            "    yield\n"
            "    runtime_state.clear()\n\n"
            "app = FastAPI(\n"
            f'    title="{model_name} Serving API",\n'
            '    version="1.0.0",\n'
            '    description="Production-grade 3-Tier Clean ML Inference API with ONNX Level-3 Acceleration",\n'
            "    lifespan=lifespan,\n"
            ")\n\n"
            '@app.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)\n'
            "async def health_check():\n"
            "    return HealthResponse(\n"
            '        status="alive",\n'
            '        model_loaded=runtime_state.get("model") is not None,\n'
            '        runtime_engine=runtime_state.get("engine", "none"),\n'
            "    )\n\n"
            '@app.get("/readyz", response_model=HealthResponse, status_code=status.HTTP_200_OK)\n'
            "async def readiness_check():\n"
            '    is_ready = runtime_state.get("model") is not None\n'
            "    if not is_ready:\n"
            '        raise HTTPException(status_code=503, detail="Model artifact not yet loaded")\n'
            "    return HealthResponse(\n"
            '        status="ready",\n'
            "        model_loaded=True,\n"
            '        runtime_engine=runtime_state.get("engine", "none"),\n'
            "    )\n\n"
            '@app.post("/predict", response_model=PredictionResponse)\n'
            "async def predict(payload: PredictionRequest):\n"
            '    model = runtime_state.get("model")\n'
            '    engine = runtime_state.get("engine", "none")\n'
            "    if model is None:\n"
            '        raise HTTPException(status_code=503, detail="Model not loaded")\n\n'
            "    start_time = time.perf_counter()\n"
            "    feat_vector = np.array([[getattr(payload, f) for f in FEATURE_NAMES]], dtype=np.float32)\n"
            "    pred = None\n"
            "    conf = None\n\n"
            '    if engine == "onnx_runtime":\n'
            '        input_name = runtime_state["input_name"]\n'
            "        raw_out = model.run(None, {input_name: feat_vector})\n"
            '        pred = int(raw_out[0][0]) if hasattr(raw_out[0][0], "item") else raw_out[0][0]\n'
            "        if len(raw_out) > 1 and isinstance(raw_out[1], list) and len(raw_out[1]) > 0:\n"
            "            prob_dict = raw_out[1][0]\n"
            "            conf = float(max(prob_dict.values())) if isinstance(prob_dict, dict) else None\n"
            "        elif len(raw_out) > 1 and isinstance(raw_out[1], np.ndarray):\n"
            "            conf = float(np.max(raw_out[1][0]))\n"
            "    else:\n"
            "        pred = model.predict(feat_vector)[0]\n"
            '        if hasattr(model, "predict_proba"):\n'
            "            conf = float(model.predict_proba(feat_vector)[0].max())\n\n"
            "    latency = (time.perf_counter() - start_time) * 1000.0\n"
            "    return PredictionResponse(\n"
            "        prediction=pred,\n"
            "        confidence_score=conf,\n"
            f'        model_version="{model_name}",\n'
            "        runtime_engine=engine,\n"
            "        latency_ms=round(latency, 2),\n"
            "    )\n"
        )
        main_path = os.path.join(output_dir, "main.py")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(main_py)

        return {"schemas.py": schemas_path, "main.py": main_path, "requirements.txt": requirements_path}
