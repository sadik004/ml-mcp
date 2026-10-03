"""AI Safety, Out-of-Distribution (OOD), Stress Testing, and Slice Fairness DTOs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
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


class CRCPredictionDTO(BaseDTO):
    """Individual prediction instance with conformal prediction set and triage status."""

    sample_index: int = Field(description="Index of sample")
    prediction_set: list[int] = Field(description="List of predicted class labels included in conformal set")
    set_size: int = Field(description="Size of the prediction set")
    is_ambiguous: bool = Field(description="True if |set| > 1, requiring human-in-the-loop review")
    is_empty: bool = Field(description="True if |set| == 0 (anomaly / out-of-distribution)")
    needs_human_review: bool = Field(description="True if ambiguous or empty")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Class probabilities")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "prediction_set": self.prediction_set,
            "needs_human_review": self.needs_human_review,
        }


class CRCReportDTO(BaseDTO):
    """Conformal Risk Control (CRC) report with mathematical risk guarantees."""

    loss_function: str = Field(description="Type of controlled loss: misclassification, fnr, or asymmetric_cost")
    target_risk: float = Field(ge=0.0, le=1.0, description="User-specified maximum risk bound alpha: E[loss] <= alpha")
    empirical_risk: float = Field(description="Empirical risk evaluated on calibration/validation split")
    calibrated_lambda: float = Field(description="Optimal threshold parameter lambda achieving risk control")
    guarantee_satisfied: bool = Field(description="True if empirical risk <= target_risk")
    total_cal_samples: int = Field(ge=1, description="Number of calibration samples used")
    average_set_size: float = Field(description="Average cardinality of prediction sets |C(X)|")
    ambiguity_rate: float = Field(ge=0.0, le=1.0, description="Proportion of samples with |C(X)| > 1")
    empty_set_rate: float = Field(ge=0.0, le=1.0, description="Proportion of samples with empty set |C(X)| == 0")
    human_triage_count: int = Field(ge=0, description="Total samples routed to human review")
    mondrian_conditional: bool = Field(default=False, description="True if group/class-conditional calibration applied")
    per_class_thresholds: Optional[Dict[str, float]] = Field(default=None, description="Per-class lambda thresholds if Mondrian")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "loss_function": self.loss_function,
            "target_risk": round(self.target_risk, 4),
            "empirical_risk": round(self.empirical_risk, 4),
            "calibrated_lambda": round(self.calibrated_lambda, 4),
            "guarantee_satisfied": self.guarantee_satisfied,
            "average_set_size": round(self.average_set_size, 2),
            "ambiguity_rate": round(self.ambiguity_rate, 4),
            "human_triage_count": self.human_triage_count,
        }

class SafetyCertificateReportDTO(BaseDTO):
    """Unified Phase 4 Master Safety, Calibration & Decision Certificate Report DTO."""

    optimal_threshold: float = Field(description="DCA-optimal cutoff threshold p*")
    naive_50_dollar_loss: float = Field(description="Expected loss using naive 0.50 threshold")
    optimal_dollar_loss: float = Field(description="Expected loss using optimal p* threshold")
    net_dollar_savings: float = Field(description="Dollar savings achieved by optimal thresholding")
    false_positive_reduction_pct: float = Field(description="Percentage reduction in false positives")
    raw_ece: float = Field(description="Raw Expected Calibration Error before Platt/Beta scaling")
    calibrated_ece: float = Field(description="Calibrated Expected Calibration Error")
    top_shap_drivers: List[str] = Field(default_factory=list, description="Top decision-driving features by TreeSHAP")
    conformal_coverage_pct: float = Field(description="Realized empirical coverage percentage (target 95%)")
    conformal_singleton_pct: float = Field(description="Percentage of samples with unique non-ambiguous prediction set")
    ood_cutoff_boundary: float = Field(description="Helmholtz Free Energy 99th percentile cutoff")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or diagnostic notices")
    safety_card: str = Field(description="Formatted human-readable Markdown safety certificate card")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "net_dollar_savings": round(self.net_dollar_savings, 2),
            "false_positive_reduction_pct": round(self.false_positive_reduction_pct, 1),
            "calibrated_ece": round(self.calibrated_ece * 100, 2),
            "conformal_coverage_pct": round(self.conformal_coverage_pct, 1),
            "top_shap_drivers": self.top_shap_drivers[:5],
            "warnings": self.warnings,
            "safety_card": self.safety_card,
        }

