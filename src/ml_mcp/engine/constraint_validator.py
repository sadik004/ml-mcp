"""Amazon Deequ-style Physical and Statistical Domain Constraint Validator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ml_mcp.schemas.audit import (
    ColumnConstraintViolationDTO,
    ConstraintValidationReportDTO,
)


class ConstraintValidator:
    """Enforces declarative physical limits and automated IQR statistical range constraints."""

    def __init__(self, constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.constraints = constraints or {}

    def validate_constraints(
        self,
        df: pd.DataFrame,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ConstraintValidationReportDTO:
        """Validates all dataframe columns against user-specified rules or automated IQR boundaries.

        Args:
            df: Input dataset pandas DataFrame.
            constraints: Optional override dictionary of per-column physical limits.

        Returns:
            ConstraintValidationReportDTO with violation metrics and broken rules.
        """
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

        # If user explicitly supplied constraints, validate against them
        if active_constraints:
            for col, rules in active_constraints.items():
                if col not in df.columns:
                    continue

                col_data = df[col]

                # 1. Non-negative check
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

                # 2. Min boundary
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

                # 3. Max boundary
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

                # 4. Allowed values (categorical domain check)
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
                                rule_broken=f"allowed_values: discrete set constraint violated",
                            )
                        )

                # 5. Completeness check (is_complete)
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

        # Automatic Deequ IQR statistical bounds for numeric columns without explicit min/max
        for col in df.select_dtypes(include=[np.number]).columns:
            # Skip if explicitly covered by user min and max
            if active_constraints and col in active_constraints and "min" in active_constraints[col] and "max" in active_constraints[col]:
                continue

            series = df[col].dropna()
            if len(series) < 10:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr > 1e-9:
                lower_limit = q1 - 3.0 * iqr
                upper_limit = q3 + 3.0 * iqr

                outlier_mask = (series < lower_limit) | (series > upper_limit)
                cnt = int(outlier_mask.sum())
                if cnt > 0 and not (active_constraints and col in active_constraints):
                    total_violations += cnt
                    violations_by_column.append(
                        ColumnConstraintViolationDTO(
                            column=col,
                            violation_count=cnt,
                            violation_rate=round(float(cnt / n_rows), 4),
                            rule_broken=f"iqr_range_violation: outside [{round(lower_limit, 2)}, {round(upper_limit, 2)}]",
                        )
                    )

        passed = (total_violations == 0)

        return ConstraintValidationReportDTO(
            total_rows=n_rows,
            passed=passed,
            total_violations=total_violations,
            violations_by_column=violations_by_column,
        )
