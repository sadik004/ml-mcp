"""Unit tests for JSON sanitizer engine."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.json_sanitizer import sanitize_for_json, MaxRecursionDepthExceeded


def test_sanitize_primitives():
    assert sanitize_for_json(42) == 42
    assert sanitize_for_json(3.14) == 3.14
    assert sanitize_for_json("hello") == "hello"
    assert sanitize_for_json(True) is True
    assert sanitize_for_json(None) is None


def test_sanitize_numpy_scalar_types():
    assert sanitize_for_json(np.int64(100)) == 100
    assert isinstance(sanitize_for_json(np.int64(100)), int)

    assert sanitize_for_json(np.float64(2.718)) == pytest.approx(2.718)
    assert isinstance(sanitize_for_json(np.float64(2.718)), float)

    assert sanitize_for_json(np.bool_(True)) is True
    assert isinstance(sanitize_for_json(np.bool_(True)), bool)
    assert sanitize_for_json(np.bool_(False)) is False


def test_sanitize_nan_and_infinities():
    # np.nan -> None
    assert sanitize_for_json(np.nan) is None
    assert sanitize_for_json(float("nan")) is None

    # np.inf & -np.inf -> None
    assert sanitize_for_json(np.inf) is None
    assert sanitize_for_json(-np.inf) is None
    assert sanitize_for_json(float("inf")) is None
    assert sanitize_for_json(float("-inf")) is None


def test_sanitize_pandas_nas_and_nat():
    assert sanitize_for_json(pd.NA) is None
    assert sanitize_for_json(pd.NaT) is None


def test_sanitize_datetimes():
    ts = pd.Timestamp("2026-09-30T08:00:00")
    assert sanitize_for_json(ts) == "2026-09-30T08:00:00"

    np_dt = np.datetime64("2026-09-30T08:00:00")
    assert "2026-09-30" in sanitize_for_json(np_dt)

    np_nat = np.datetime64("NaT")
    assert sanitize_for_json(np_nat) is None


def test_sanitize_arrays_and_series():
    arr = np.array([1, 2, 3], dtype=np.int64)
    res = sanitize_for_json(arr)
    assert res == [1, 2, 3]
    assert all(isinstance(x, int) for x in res)

    series = pd.Series([10.5, np.nan, np.inf], index=["a", "b", "c"])
    s_res = sanitize_for_json(series)
    assert s_res == {"a": 10.5, "b": None, "c": None}


def test_sanitize_nested_complex_structures():
    nested = {
        "metrics": {
            "accuracy": np.float64(0.95),
            "fold_scores": np.array([0.94, 0.96, np.nan]),
        },
        "metadata": {
            "is_valid": np.bool_(True),
            "timestamp": pd.Timestamp("2026-01-01"),
            "tags": ("kaggle", "grandmaster", np.int32(1)),
            "flags": {np.bool_(False), np.bool_(True)},
        },
    }
    cleaned = sanitize_for_json(nested)
    assert cleaned["metrics"]["accuracy"] == 0.95
    assert cleaned["metrics"]["fold_scores"] == [0.94, 0.96, None]
    assert cleaned["metadata"]["is_valid"] is True
    assert cleaned["metadata"]["timestamp"] == "2026-01-01T00:00:00"
    assert cleaned["metadata"]["tags"] == ["kaggle", "grandmaster", 1]
    assert set(cleaned["metadata"]["flags"]) == {False, True}


def test_sanitize_max_depth_and_cyclic_protection():
    # Cyclic reference
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict

    with pytest.raises(MaxRecursionDepthExceeded):
        sanitize_for_json(cyclic_dict, max_depth=10)

    # Deeply nested list exceeding depth limit
    deep = []
    curr = deep
    for _ in range(15):
        nxt = []
        curr.append(nxt)
        curr = nxt

    with pytest.raises(MaxRecursionDepthExceeded):
        sanitize_for_json(deep, max_depth=10)
