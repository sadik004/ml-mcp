"""ONNX Inference Speed Optimization and Level-3 Graph Hardware Optimization Engine."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _apply_skl2onnx_compatibility_patch() -> None:
    """Fixes skl2onnx issue where mixed bool/uint8 lists in TreeEnsembleClassifier crash on ONNX 1.16+."""
    try:
        import skl2onnx.common._container as container_mod
        if getattr(container_mod.ModelComponentContainer, "_patched_onnx_bool_fix", False):
            return
        orig_add_node = container_mod.ModelComponentContainer.add_node

        def patched_add_node(self, op_type, inputs, outputs, op_domain="", op_version=None, name=None, **attrs):
            for k, v in list(attrs.items()):
                if isinstance(v, list) and any(isinstance(x, (bool, getattr(container_mod.np, "bool_", bool))) for x in v):
                    if any(isinstance(x, (int, getattr(container_mod.np, "integer", int))) for x in v):
                        attrs[k] = [int(x) if isinstance(x, (bool, getattr(container_mod.np, "bool_", bool))) else x for x in v]
            return orig_add_node(self, op_type, inputs, outputs, op_domain=op_domain, op_version=op_version, name=name, **attrs)

        container_mod.ModelComponentContainer.add_node = patched_add_node
        container_mod.ModelComponentContainer._patched_onnx_bool_fix = True
    except Exception as exc:
        logger.debug("skl2onnx patch skipped: %s", exc)


class ONNXOptimizer:
    """Converts models to ONNX graph with Level-3 graph fusion and benchmarks P95/P99 latency.

    Theoretical foundations:
        - ONNX Runtime Level-3 Hardware Graph Optimization: IEEE Micro 2022
        - Node Fusion, Constant Folding, and Dead Code Elimination
    """

    def __init__(self, benchmark_samples: int = 100) -> None:
        self.benchmark_samples = benchmark_samples
        _apply_skl2onnx_compatibility_patch()

    def convert_and_benchmark(
        self,
        model: Any,
        sample_input: Any,
        n_features: Optional[int] = None,
    ) -> Tuple[bytes, float, float]:
        """Convert model to ONNX bytes with ORT_ENABLE_ALL and benchmark single-sample P95/P99 latency."""
        sample_arr = (
            sample_input.to_numpy(dtype=np.float32)
            if isinstance(sample_input, pd.DataFrame)
            else np.asarray(sample_input, dtype=np.float32)
        )
        if sample_arr.ndim == 1:
            sample_arr = sample_arr.reshape(1, -1)

        num_features = n_features or sample_arr.shape[1]
        onnx_bytes = b""
        session = None

        import onnxruntime as ort
        model_type_str = str(type(model)).lower()

        try:
            if "lightgbm" in model_type_str:
                try:
                    import onnxmltools
                    from onnxmltools.convert.common.data_types import FloatTensorType
                    initial_type = [("float_input", FloatTensorType([None, num_features]))]
                    onx = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
                    onnx_bytes = onx.SerializeToString()
                except Exception:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                    initial_type = [("float_input", FloatTensorType([None, num_features]))]
                    onx = convert_sklearn(model, initial_types=initial_type)
                    onnx_bytes = onx.SerializeToString()
            elif "xgboost" in model_type_str:
                try:
                    import onnxmltools
                    from onnxmltools.convert.common.data_types import FloatTensorType
                    initial_type = [("float_input", FloatTensorType([None, num_features]))]
                    onx = onnxmltools.convert_xgboost(model, initial_types=initial_type)
                    onnx_bytes = onx.SerializeToString()
                except Exception:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                    initial_type = [("float_input", FloatTensorType([None, num_features]))]
                    onx = convert_sklearn(model, initial_types=initial_type)
                    onnx_bytes = onx.SerializeToString()
            else:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [("float_input", FloatTensorType([None, num_features]))]
                onx = convert_sklearn(model, initial_types=initial_type)
                onnx_bytes = onx.SerializeToString()

            # Configure ONNX Runtime with Level-3 Hardware Graph Optimizations (ORT_ENABLE_ALL)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(onnx_bytes, sess_options, providers=["CPUExecutionProvider"])
        except Exception as e:
            # Fail-fast: do not mask failure with silent dummy byte placeholder
            raise RuntimeError(f"ONNX conversion or session initialization failed for model {type(model)}: {e}") from e

        # Benchmark single-sample inference latency (P95 and P99)
        latencies_ms = []
        test_single = sample_arr[:1]

        input_name = session.get_inputs()[0].name
        for _ in range(self.benchmark_samples):
            start = time.perf_counter()
            _ = session.run(None, {input_name: test_single})
            elapsed = (time.perf_counter() - start) * 1000.0  # ms
            latencies_ms.append(elapsed)

        p95_latency = float(np.percentile(latencies_ms, 95))
        p99_latency = float(np.percentile(latencies_ms, 99))

        return onnx_bytes, round(p95_latency, 3), round(p99_latency, 3)
