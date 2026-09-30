"""AI Safety, Out-of-Distribution (OOD), Stress Testing, and Slice Fairness DTOs."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import Field
from ml_mcp.schemas.base import BaseDTO


class OODReportDTO(BaseDTO):
    """Out-of-Distribution detection report using Isolation Forest or Mahalanobis Distance."""

    detector_name: str = Field(description="OOD model name (e.g. IsolationForest)")
    total_samples: int = Field(ge=1, description="Total samples scanned")
    ood_detected_count: int = Field(ge=0, description="Count of anomalous out-of-distribution records")
    ood_ratio: float = Field(ge=0.0, le=1.0, description="Proportion of samples flagged as OOD")
    anomaly_threshold: float = Field(description="Decision threshold separating in-distribution from OOD")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "total_samples": self.total_samples,
            "ood_detected_count": self.ood_detected_count,
            "ood_ratio": round(self.ood_ratio, 4),
        }


class StressTestReportDTO(BaseDTO):
    """Stress test audit injecting noise perturbations to evaluate model fragility."""

    perturbation_type: str = Field(description="gaussian_noise, feature_swap, or extreme_outlier")
    noise_level: float = Field(description="Relative amplitude of injected noise")
    baseline_score: float = Field(description="Performance score on unperturbed test set")
    stressed_score: float = Field(description="Performance score after noise injection")
    degradation_percentage: float = Field(description="Percentage degradation under stress")
    robustness_score: float = Field(ge=0.0, le=100.0, description="Robustness index (0 to 100)")
    is_stress_passed: bool = Field(description="True if degradation <= 10%")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "perturbation_type": self.perturbation_type,
            "robustness_score": round(self.robustness_score, 2),
            "degradation_percentage": round(self.degradation_percentage, 2),
            "is_stress_passed": self.is_stress_passed,
        }


class SliceFairnessDTO(BaseDTO):
    """Sub-group fairness and disparate impact audit report."""

    protected_attribute: str = Field(description="Audited categorical slice attribute (e.g. region, gender)")
    subgroup_scores: Dict[str, float] = Field(description="Validation scores across distinct slices")
    max_disparity: float = Field(ge=0.0, description="Maximum absolute score disparity between groups")
    disparate_impact_ratio: float = Field(
        ge=0.0, description="Ratio of min group score to max group score"
    )
    parity_violated: bool = Field(description="True if disparate impact ratio < 0.80 (80% rule)")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "protected_attribute": self.protected_attribute,
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
            "max_disparity": round(self.max_disparity, 4),
            "parity_violated": self.parity_violated,
            "subgroup_summary": {k: round(v, 4) for k, v in self.subgroup_scores.items()},
        }
