"""Unit tests for ONNX Inference Optimizer and Latency Benchmarker."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.onnx_optimizer import ONNXOptimizer


def test_onnx_optimizer_conversion_and_latency():
    """Verify model converts to ONNX and benchmarks P95/P99 latency."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = LogisticRegression()
    clf.fit(X, y)

    optimizer = ONNXOptimizer()
    onnx_bytes, p95_ms, p99_ms = optimizer.convert_and_benchmark(
        model=clf,
        sample_input=X[:5],
        n_features=4,
    )

    assert onnx_bytes is not None
    assert len(onnx_bytes) > 0
    assert p95_ms >= 0.0
    assert p99_ms >= 0.0
    assert p99_ms >= p95_ms or abs(p99_ms - p95_ms) < 1.0


def test_onnx_optimizer_resilient_fallback():
    """Verify optimizer provides resilient fallback on unsupported objects."""
    optimizer = ONNXOptimizer()
    # Passing an arbitrary object without sklearn/onnx converter
    onnx_bytes, p95_ms, p99_ms = optimizer.convert_and_benchmark(
        model="unsupported_model_string",
        sample_input=np.zeros((2, 2)),
        n_features=2,
    )

    # Resilient fallback returns empty or mock bytes without crash
    assert onnx_bytes is not None
    assert p95_ms >= 0.0
    assert p99_ms >= 0.0
