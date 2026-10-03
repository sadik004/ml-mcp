"""Unit tests for ONNX Inference Optimizer with Level-3 Graph Fusion (ORT_ENABLE_ALL)."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.onnx_optimizer import ONNXOptimizer


def test_onnx_optimizer_conversion_and_latency():
    """Verify model converts to ONNX and benchmarks Level-3 P95/P99 latency."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = LogisticRegression()
    clf.fit(X, y)

    optimizer = ONNXOptimizer()
    onnx_bytes, p95_ms, p99_ms = optimizer.convert_and_benchmark(
        model=clf,
        sample_input=X[:5],
        n_features=4,
    )

    assert len(onnx_bytes) > 0
    assert p95_ms > 0.0
    assert p99_ms >= p95_ms
