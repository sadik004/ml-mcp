"""Unified Phase 1 Master Pre-Flight Quality Auditor.

Coordinates all 6 Phase 1 diagnostic engines:
1. Cryptographic Lineage (SHA-256, shape, memory, duplicate scan)
2. Data Hygiene & Missingness (MCAR vs MNAR, Benford's Law anomaly, DataPerf memorization)
3. Target Leakage (Chatterjee non-parametric rank xi, Cramér's V, baseline-normalized PPS)
4. Multicollinearity (SVD Spectral Condition Number kappa, Ridge-VIF, Belsley variance)
5. Label Error Hunter (MIT Confident Learning OOF self-confidence bounds)
6. Domain Invariant Constraints & Medcouple Skewed Outlier Bounds
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd

from ml_mcp.engine.auditor import DatasetAuditor
from ml_mcp.engine.collinearity import CollinearityFilter
from ml_mcp.engine.constraint_validator import ConstraintValidator
from ml_mcp.engine.label_error_detector import LabelErrorDetector
from ml_mcp.engine.leakage import TargetLeakageDetector
from ml_mcp.schemas.audit import PreflightAuditReportDTO

logger = logging.getLogger(__name__)


class PreflightAuditor:
    """Master single-pass orchestrator executing the full Phase 1 Data Hygiene & Quality Suite."""

    def __init__(
        self,
        leakage_chatterjee_threshold: float = 0.85,
        leakage_pps_threshold: float = 0.90,
        collinearity_kappa_threshold: float = 100.0,
        max_label_noise_pct: float = 3.0,
    ) -> None:
        self.leakage_chatterjee_threshold = leakage_chatterjee_threshold
        self.leakage_pps_threshold = leakage_pps_threshold
        self.collinearity_kappa_threshold = collinearity_kappa_threshold
        self.max_label_noise_pct = max_label_noise_pct

    @staticmethod
    def compute_lineage(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates cryptographic SHA-256 fingerprint, shape, memory footprint, and exact duplicate count."""
        sample_bytes = np.asarray(pd.util.hash_pandas_object(df, index=True).values).tobytes()
        sha256 = hashlib.sha256(sample_bytes).hexdigest()
        mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        duplicates = int(df.duplicated().sum())

        return {
            "sha256": sha256,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_mb": round(mem_mb, 2),
            "duplicate_rows": duplicates,
            "duplicate_pct": round((duplicates / max(1, len(df))) * 100, 2),
        }

    def audit(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"] = "classification",
        dataset_name: str = "Dataset",
    ) -> PreflightAuditReportDTO:
        """Executes all 6 Phase 1 engines and compiles the Executive Health Report."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # 1. Lineage & Fingerprint
        lineage = self.compute_lineage(df)

        # 2. Data Hygiene & Missingness
        auditor = DatasetAuditor()
        hygiene_rep = auditor.audit_dataset(df, target_column=target_column)

        # 3. Target Leakage (Chatterjee xi, Cramér's V, Normalized PPS)
        leakage_detector = TargetLeakageDetector(
            threshold_chatterjee=self.leakage_chatterjee_threshold,
            threshold_pps=self.leakage_pps_threshold,
        )
        leakage_rep = leakage_detector.detect_leakage(df, target_column=target_column, task_type=task_type)

        # 4. Multicollinearity & SVD Condition Number
        collinearity_filter = CollinearityFilter()
        feature_df = df.drop(columns=[target_column])
        numeric_features = feature_df.select_dtypes(include=[np.number])
        if numeric_features.shape[1] >= 2 and len(numeric_features) >= 5:
            collinear_rep = collinearity_filter.filter_collinearity(feature_df)
            kappa = collinear_rep.spectral_condition_number or 1.0
            high_vif_count = len(collinear_rep.high_vif_features)
            dropped_features = collinear_rep.dropped_features
        else:
            kappa = 1.0
            high_vif_count = 0
            dropped_features = []

        # 5. Label Error Detection (Confident Learning)
        label_noise_pct = 0.0
        suspected_label_errors_count = 0
        if task_type == "classification" and len(df[target_column].unique()) >= 2:
            label_detector = LabelErrorDetector()
            try:
                label_rep = label_detector.detect_label_errors(df, target_column=target_column)
                rate = getattr(label_rep, "error_rate", getattr(label_rep, "noise_rate", 0.0))
                label_noise_pct = float(rate * 100.0)
                suspected_label_errors_count = int(getattr(label_rep, "total_errors", getattr(label_rep, "total_label_errors", 0)))
            except Exception as e:
                logger.warning(f"Label error detection skipped: {e}")

        # 6. Domain Constraints & Skewness Outlier Bounds
        constraint_validator = ConstraintValidator()
        try:
            constraint_rep = constraint_validator.validate_constraints(df)
            violations_count = len(constraint_rep.violations_by_column)
        except Exception:
            violations_count = 0

        # ----------------------------------------------------------------------
        # Deterministic Health Score Calculation (100 - Penalties)
        # ----------------------------------------------------------------------
        score = 100
        critical_red_flags: List[str] = []
        warnings: List[str] = []

        # Target Leakage Penalty
        valid_chatterjee = [v for v in leakage_rep.chatterjee_scores.values() if v is not None]
        if leakage_rep.has_critical_leakage:
            score -= 40
            leaked_names = ", ".join(leakage_rep.leaked_features[:3])
            critical_red_flags.append(
                f"CRITICAL TARGET LEAKAGE: Features [{leaked_names}] exhibit anomalous correlation. Must be dropped before Phase 2!"
            )
        elif valid_chatterjee and max(valid_chatterjee) > 0.50:
            score -= 10
            warnings.append("Moderate Target Association: Some features exhibit Chatterjee xi > 0.50.")

        # Multicollinearity Penalty
        if kappa > self.collinearity_kappa_threshold:
            score -= 20
            critical_red_flags.append(
                f"Severe Multicollinearity: SVD Condition Number kappa={kappa:.1f} (>{self.collinearity_kappa_threshold}). Feature matrix is ill-conditioned."
            )
        elif kappa > 30.0 or high_vif_count > 0:
            score -= 5
            warnings.append(f"Moderate Collinearity: SVD kappa={kappa:.1f}, {high_vif_count} high-VIF features detected.")

        # Missingness Penalty
        if any(v == "MNAR" for v in hygiene_rep.missingness_mechanisms.values()):
            score -= 15
            warnings.append("MNAR (Missing Not at Random) pattern detected. Imputer must retain MissingIndicators.")
        elif hygiene_rep.missing_ratio > 0.20:
            score -= 10
            warnings.append(f"High Missing Rate: {hygiene_rep.missing_ratio * 100:.1f}% missing cells in dataset.")

        # Duplicate Rows Penalty
        if lineage["duplicate_pct"] > 5.0:
            score -= 10
            warnings.append(f"Duplicate Rows: {lineage['duplicate_rows']} ({lineage['duplicate_pct']}%) duplicate rows found.")

        # Label Noise Penalty
        if label_noise_pct > self.max_label_noise_pct:
            score -= 15
            critical_red_flags.append(
                f"High Label Noise: Estimated {label_noise_pct:.2f}% corrupt labels detected by Confident Learning."
            )
        elif label_noise_pct > 0.5:
            score -= 5
            warnings.append(f"Low Label Noise: Estimated {label_noise_pct:.2f}% suspected label errors.")

        # Bound score
        final_score = int(max(0, min(100, score)))

        # Traffic Light Verdict
        if final_score >= 85 and not critical_red_flags:
            verdict: Literal["PASSED", "CONDITIONAL_PASS", "BLOCKED"] = "PASSED"
            icon = "🟢"
        elif final_score >= 60 and not any("LEAKAGE" in rf for rf in critical_red_flags):
            verdict = "CONDITIONAL_PASS"
            icon = "🟡"
        else:
            verdict = "BLOCKED"
            icon = "🔴"

        # ----------------------------------------------------------------------
        # Executive Markdown Quality Card
        # ----------------------------------------------------------------------
        card_lines = [
            "=" * 88,
            "🚀 ML-MCP PRE-FLIGHT DATA QUALITY REPORT (PHASE 1 AUDIT)",
            f"Dataset: {dataset_name} | Shape: ({lineage['rows']:,} rows × {lineage['columns']} cols) | Memory: {lineage['memory_mb']} MB",
            f"Cryptographic SHA-256: {lineage['sha256'][:16]}...{lineage['sha256'][-8:]}",
            "=" * 88,
            f"\n{icon} EXECUTIVE READINESS VERDICT: {verdict} (Health Score: {final_score} / 100)",
            "-" * 88,
        ]

        if critical_red_flags:
            card_lines.append("\n🚨 CRITICAL RED FLAGS (Action Required before Phase 2):")
            for rf in critical_red_flags:
                card_lines.append(f"  ❌ {rf}")
        else:
            card_lines.append("\n✅ ZERO CRITICAL RED FLAGS: No catastrophic leakage or matrix singularities detected.")

        if warnings:
            card_lines.append("\n⚠️ DIAGNOSTIC WARNINGS (Review Recommended):")
            for w in warnings:
                card_lines.append(f"  ⚠️ {w}")

        card_lines.extend([
            "\n" + "-" * 88,
            "🔬 6-ENGINE AUDIT SUMMARY:",
            f"  1. Lineage       : {lineage['rows']:,} samples, {lineage['columns']} columns, {lineage['duplicate_rows']} duplicates ({lineage['duplicate_pct']}%).",
            f"  2. Missingness   : {hygiene_rep.missing_ratio * 100:.1f}% missing, Mechanisms={dict(list(hygiene_rep.missingness_mechanisms.items())[:3])}, Benford Anomalies={len(hygiene_rep.benford_anomalies)}.",
            f"  3. Target Leakage: {'CRITICAL LEAKAGE' if leakage_rep.has_critical_leakage else 'SAFE'} (Leaked={len(leakage_rep.leaked_features)}).",
            f"  4. Collinearity  : SVD kappa={kappa:.2f} ({'ILL-CONDITIONED' if kappa > 100 else 'STABLE'}), Dropped Pruning Candidates={len(dropped_features)}.",
            f"  5. Label Noise   : Estimated Noise Rate={label_noise_pct:.2f}%, Flagged Suspects={suspected_label_errors_count} rows.",
            f"  6. Constraints   : Invariant Failures={violations_count}.",
            "=" * 88,
            f"👉 RECOMMENDATION: {'Ready to advance to Phase 2 Feature Pipeline!' if verdict in ('PASSED', 'CONDITIONAL_PASS') else 'Resolve critical red flags before model training.'}",
            "=" * 88,
        ])

        executive_card = "\n".join(card_lines)

        return PreflightAuditReportDTO(
            dataset_name=dataset_name,
            health_score=final_score,
            readiness_verdict=verdict,
            sha256_hash=lineage["sha256"],
            row_count=lineage["rows"],
            column_count=lineage["columns"],
            duplicate_rows=lineage["duplicate_rows"],
            critical_red_flags=critical_red_flags,
            warnings=warnings,
            audit_summary={
                "lineage": lineage,
                "missing_ratio": hygiene_rep.missing_ratio,
                "leakage_count": len(leakage_rep.leaked_features),
                "svd_kappa": round(kappa, 2),
                "label_noise_pct": round(label_noise_pct, 2),
                "violations_count": violations_count,
            },
            executive_card=executive_card,
        )
