"""Recursive JSON sanitizer engine for NumPy, Pandas, and Python primitives.

Guarantees zero serialization failures (e.g. 'Object of type int64 is not JSON serializable')
across all FastMCP tools and APIs.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


class MaxRecursionDepthExceeded(ValueError):
    """Raised when object traversal exceeds safe recursion limit or cyclic reference detected."""
    pass


def sanitize_for_json(obj: Any, max_depth: int = 64, _current_depth: int = 0) -> Any:
    """Recursively converts NumPy, Pandas, and complex objects into JSON-compliant native Python types.

    Transforms:
        - np.generic scalars (np.int64, np.float64, np.bool_) -> int, float, bool
        - np.ndarray -> list
        - pd.Series -> dict
        - np.nan, float('nan'), pd.NA, pd.NaT -> None
        - np.inf, -np.inf, float('inf'), float('-inf') -> None
        - pd.Timestamp, np.datetime64 -> ISO 8601 string
        - dict, list, tuple, set -> standard sanitized dict/list

    Raises:
        MaxRecursionDepthExceeded: If traversal depth exceeds `max_depth` (cyclic guard).
    """
    if _current_depth > max_depth:
        raise MaxRecursionDepthExceeded(
            f"Maximum recursion depth of {max_depth} exceeded during JSON sanitization."
        )

    # 1. None check
    if obj is None:
        return None

    # 2. Pandas NA / NaT types
    if obj is pd.NA or obj is pd.NaT:
        return None

    # 3. Floating NaNs and Infinities
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)

    # 4. NumPy integer types
    if isinstance(obj, (np.integer,)):
        return int(obj)

    # 5. NumPy boolean
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # 6. Datetime types (pd.Timestamp, np.datetime64)
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        if np.isnat(obj):
            return None
        return str(obj)

    # 7. Basic Python Primitives
    if isinstance(obj, (int, str, bool)):
        return obj

    # 8. Pandas Series
    if isinstance(obj, pd.Series):
        return {
            str(k): sanitize_for_json(v, max_depth, _current_depth + 1)
            for k, v in obj.to_dict().items()
        }

    # 9. NumPy ndarray
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(item, max_depth, _current_depth + 1) for item in obj.tolist()]

    # 10. Dictionaries
    if isinstance(obj, dict):
        return {
            str(k): sanitize_for_json(v, max_depth, _current_depth + 1)
            for k, v in obj.items()
        }

    # 11. Sequences (lists, tuples, sets)
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item, max_depth, _current_depth + 1) for item in obj]

    # Fallback to string representation if unknown non-primitive
    return str(obj)
