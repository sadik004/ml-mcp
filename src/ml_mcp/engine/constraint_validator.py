"""Amazon Deequ & Hubert Medcouple Adjusted Statistical Domain Constraint Validator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_mcp.config import get_settings
from ml_mcp.schemas.audit import (
    ColumnConstraintViolationDTO,
    ConstraintValidationReportDTO,
)


class ConstraintValidator:
    """Enforces declarative physical limits and Hubert & Vandervieren Medcouple adjusted statistical boundaries.

    Theoretical foundations:
        - Skewed Outlier Bounds: Hubert & Vandervieren (Computational Statistics) "An Adjusted Boxplot for Skewed Distributions"
        - Declarative Constraints: Amazon Deequ (VLDB / ACM SIGMOD 2022) & Grafberger et al. (VLDB 2023)
    """

    def __init__(self, constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.constraints = constraints or {}

    @staticmethod
    def calculate_medcouple(series: pd.Series) -> float:
        """Calculates Medcouple (MC) robust skewness estimator in [-1, 1] (Hubert & Vandervieren).

        Subsamples up to 500 rows for fast O(N_sub log N_sub) performance.
        """
        clean = series.dropna().to_numpy(dtype=np.float64)
        n = len(clean)
        if n < 10:
            return 0.0

        if n > 500:
            rng = np.random.RandomState(get_settings().random_state)
            clean = rng.choice(clean, 500, replace=False)
            n = len(clean)

        clean = np.sort(clean)
        med = float(np.median(clean))

        x_left = clean[clean <= med]
        x_right = clean[clean >= med]

        if len(x_left) == 0 or len(x_right) == 0:
            return 0.0

        # Vectorized kernel h(xi, xj) = ((xj - med) - (med - xi)) / (xj - xi)
        # Avoid outer product explosion: sample pairs if product > 50,000
        total_pairs = len(x_left) * len(x_right)
        if total_pairs > 50000:
            rng = np.random.RandomState(get_settings().random_state)
            i_idx = rng.choice(len(x_left), 200)
            j_idx = rng.choice(len(x_right), 200)
            xi = x_left[i_idx, np.newaxis]
            xj = x_right[np.newaxis, j_idx]
        else:
            xi = x_left[:, np.newaxis]
            xj = x_right[np.newaxis, :]

        denom = xj - xi
        valid = denom > 1e-12
        if not np.any(valid):
            return 0.0

        num = (xj - med) - (med - xi)
        h = num[valid] / denom[valid]
        mc = float(np.median(h))
        return float(np.clip(mc, -1.0, 1.0))

    def validate_constraints(
        self,
        df: pd.DataFrame,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ConstraintValidationReportDTO:
        """Validates all dataframe columns against user-specified rules or Medcouple adjusted statistical boundaries."""
        active_constraints = constraints if constraints is not None else self.constraints
        n_rows = len(df)
        if n_rows == 0:
            return ConstraintValidationReportDTO(
                total_rows=0,
                passed=True,
                total_violations=0,
                violations_by_column=[],
            )

        violations_by_column: List[ColumnConstraintViolationDTO] = []
        total_violations = 0

        # 1. User-Specified Explicit Constraints
        if active_constraints:
            for col, rules in active_constraints.items():
                if col not in df.columns:
                    continue
                col_data = df[col]

                # Non-negative check
                if rules.get("non_negative", False):
                    num_series = pd.to_numeric(col_data, errors="coerce")
                    neg_mask = num_series < 0.0
                    cnt = int(neg_mask.sum())
                    if cnt > 0:
                        total_violations += cnt
                        violations_by_column.append(
                            ColumnConstraintViolationDTO(
                                column=col,
                                violation_count=cnt,
                                violation_rate=round(float(cnt / n_rows), 4),
                                rule_broken="non_negative: values must be >= 0.0",
                            )
                        )

                # Min boundary
                if "min" in rules:
                    min_val = float(rules["min"])
                    num_series = pd.to_numeric(col_data, errors="coerce")
                    min_mask = num_series < min_val
                    cnt = int(min_mask.sum())
                    if cnt > 0:
                        total_violations += cnt
                        violations_by_column.append(
                            ColumnConstraintViolationDTO(
                                column=col,
                                violation_count=cnt,
                                violation_rate=round(float(cnt / n_rows), 4),
                                rule_broken=f"min_bound: values must be >= {min_val}",
                            )
                        )

                # Max boundary
                if "max" in rules:
                    max_val = float(rules["max"])
                    num_series = pd.to_numeric(col_data, errors="coerce")
                    max_mask = num_series > max_val
                    cnt = int(max_mask.sum())
                    if cnt > 0:
                        total_violations += cnt
                        violations_by_column.append(
                            ColumnConstraintViolationDTO(
                                column=col,
                                violation_count=cnt,
                                violation_rate=round(float(cnt / n_rows), 4),
                                rule_broken=f"max_bound: values must be <= {max_val}",
                            )
                        )

                # Discrete Allowed values
                if "allowed_values" in rules:
                    allowed = set(rules["allowed_values"])
                    unallowed_mask = ~col_data.isin(allowed) & col_data.notna()
                    cnt = int(unallowed_mask.sum())
                    if cnt > 0:
                        total_violations += cnt
                        violations_by_column.append(
                            ColumnConstraintViolationDTO(
                                column=col,
                                violation_count=cnt,
                                violation_rate=round(float(cnt / n_rows), 4),
                                rule_broken="allowed_values: discrete set constraint violated",
                            )
                        )

                # Completeness check
                if rules.get("is_complete", False):
                    nan_mask = col_data.isna()
                    cnt = int(nan_mask.sum())
                    if cnt > 0:
                        total_violations += cnt
                        violations_by_column.append(
                            ColumnConstraintViolationDTO(
                                column=col,
                                violation_count=cnt,
                                violation_rate=round(float(cnt / n_rows), 4),
                                rule_broken="is_complete: NaN/null entries forbidden",
                            )
                        )

        # 2. Hubert Medcouple Adjusted Boxplot for Numeric Columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if active_constraints and col in active_constraints and "min" in active_constraints[col] and "max" in active_constraints[col]:
                continue

            series = df[col].dropna()
            if len(series) < 15:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr > 1e-9:
                mc = self.calculate_medcouple(series)
                # Asymmetric adjusted bounds
                if mc >= 0:
                    lower_limit = q1 - 1.5 * np.exp(-4.0 * mc) * iqr
                    upper_limit = q3 + 1.5 * np.exp(3.0 * mc) * iqr
                else:
                    lower_limit = q1 - 1.5 * np.exp(-3.0 * mc) * iqr
                    upper_limit = q3 + 1.5 * np.exp(4.0 * mc) * iqr

                outlier_mask = (series < lower_limit) | (series > upper_limit)
                cnt = int(outlier_mask.sum())
                if cnt > 0 and not (active_constraints and col in active_constraints):
                    total_violations += cnt
                    violations_by_column.append(
                        ColumnConstraintViolationDTO(
                            column=col,
                            violation_count=cnt,
                            violation_rate=round(float(cnt / n_rows), 4),
                            rule_broken=f"medcouple_adjusted_iqr: outside [{round(lower_limit, 2)}, {round(upper_limit, 2)}] (MC={round(mc, 2)})",
                        )
                    )

        passed = (total_violations == 0)

        return ConstraintValidationReportDTO(
            total_rows=n_rows,
            passed=passed,
            total_violations=total_violations,
            violations_by_column=violations_by_column,
        )
