"""Estimator Resolution Engine with Transparent Fallback Auditing (Phase P4).

Enforces Zero Silent Masking by registering structured warnings whenever surrogate
estimators (e.g. HistGradientBoostingClassifier) are substituted for requested libraries.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Union

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier

from ml_mcp.core.errors import DependencyMissing
from ml_mcp.core.warnings import WarningsCollector

logger = logging.getLogger(__name__)


def resolve_classifier(
    model_name: str = "lightgbm",
    warnings: Optional[Union[List[str], WarningsCollector]] = None,
    strict: bool = False,
    random_state: Optional[int] = None,
) -> BaseEstimator:
    from ml_mcp.config import get_settings
    seed = random_state if random_state is not None else get_settings().random_state
    """Resolve classifier by name with transparent fallback logging into warnings.

    If strict=True: raises DependencyMissing when requested library is absent.
    If strict=False: falls back to HistGradientBoostingClassifier and registers
    an explicit warning in `warnings` identifying the substitute estimator.
    """
    name = model_name.lower().strip()

    def _add_warning(msg: str) -> None:
        if warnings is not None:
            if isinstance(warnings, list):
                if msg not in warnings:
                    warnings.append(msg)
            else:
                warnings.add("DEP_FALLBACK", msg)

    if "lightgbm" in name or "lgbm" in name:
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=50, random_state=seed, verbose=-1)
        except Exception as err:
            logger.info("LightGBM unavailable (%s), triggering fallback protocol.", err)
            if strict:
                raise DependencyMissing(f"LightGBM is required but unavailable: {err}") from err
            _add_warning(
                f"lightgbm unavailable ({err}); used HistGradientBoostingClassifier as surrogate fallback"
            )
            return HistGradientBoostingClassifier(random_state=seed)

    if "xgboost" in name or "xgb" in name:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=50, random_state=seed, eval_metric="logloss")
        except Exception as err:
            logger.info("XGBoost unavailable (%s), triggering fallback protocol.", err)
            if strict:
                raise DependencyMissing(f"XGBoost is required but unavailable: {err}") from err
            _add_warning(
                f"xgboost unavailable ({err}); used HistGradientBoostingClassifier as surrogate fallback"
            )
            return HistGradientBoostingClassifier(random_state=seed)

    if "hist" in name or "gradient_boost" in name:
        return HistGradientBoostingClassifier(random_state=seed)

    if strict:
        raise DependencyMissing(f"Requested model '{model_name}' is unavailable in strict mode.")
    _add_warning(
        f"Requested model '{model_name}' unavailable; used HistGradientBoostingClassifier as surrogate fallback"
    )
    return HistGradientBoostingClassifier(random_state=random_state)
