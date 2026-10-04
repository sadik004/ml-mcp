"""Rigorous statistical routines for confidence intervals and coverage uncertainty."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from scipy.stats import norm


def wilson_interval(
    k: int,
    n: int,
    conf: float = 0.95,
    min_n: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Calculates Wilson score confidence interval for binomial proportion (Wilson, 1927).

    Guarantees:
        - Bounded strictly in [0.0, 1.0]
        - Degenerate / zero samples return (None, None)
        - If min_n specified and n < min_n, returns (None, None)
    """
    if n <= 0:
        return None, None

    if min_n is not None and n < min_n:
        return None, None

    k_clean = max(0, min(k, n))
    p = float(k_clean / n)

    alpha = 1.0 - conf
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * (n ** 2))))

    low = max(0.0, center - spread)
    high = min(1.0, center + spread)

    return round(float(low), 4), round(float(high), 4)
