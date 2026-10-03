"""Hyperparameter Tuning, Probability Calibration, and Decision Threshold DTOs."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import Field
from ml_mcp.schemas.base import BaseDTO


class OptunaStudyDTO(BaseDTO):
    """Result report from Optuna Bayesian hyperparameter optimization."""

    study_name: str = Field(description="Name of the Optuna study")
    best_trial_number: int = Field(ge=0, description="Index of the best performing trial")
    best_params: Dict[str, Any] = Field(description="Optimal hyperparameters discovered")
    best_value: float = Field(description="Score achieved by the best trial")
    total_trials: int = Field(ge=1, description="Total trials attempted")
    pruned_trials: int = Field(default=0, ge=0, description="Trials terminated early by MedianPruner")
    direction: str = Field(default="maximize", description="maximize or minimize")
    metric: str = Field(description="Optimization metric name")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "best_trial_number": self.best_trial_number,
            "best_value": round(self.best_value, 4),
            "total_trials": self.total_trials,
            "pruned_trials": self.pruned_trials,
            "best_params": self.best_params,
        }


class CalibrationReportDTO(BaseDTO):
    """Report measuring probability calibration improvement (Platt / Isotonic / Temperature)."""

    method: str = Field(description="sigmoid (Platt Scaling), isotonic, or temperature")
    pre_brier_score: float = Field(default=0.0, ge=0.0, description="Brier score before calibration")
    post_brier_score: float = Field(default=0.0, ge=0.0, description="Brier score after calibration")
    brier_score_lift: float = Field(default=0.0, description="Improvement delta (positive is better)")
    is_well_calibrated: bool = Field(default=False, description="True if post-calibration Brier score <= 0.15")
    status: Optional[str] = Field(default=None, description="Execution status or skipped notes")
    pre_ece: Optional[float] = Field(default=None, description="Expected Calibration Error before calibration")
    post_ece: Optional[float] = Field(default=None, description="Expected Calibration Error after calibration")
    ece_lift: Optional[float] = Field(default=None, description="ECE improvement delta")
    adaptive_ece: Optional[float] = Field(default=None, description="Debiased Adaptive-Quantile ECE (Roelofs et al. 2022)")
    temperature: Optional[float] = Field(default=None, description="Optimized scaling temperature T (Guo et al. 2017)")
    conformal_coverage: Optional[float] = Field(default=None, description="Empirical marginal coverage of conformal set")
    conformal_alpha: Optional[float] = Field(default=None, description="Target significance level alpha")

    def to_compact(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "method": self.method,
            "pre_brier_score": round(self.pre_brier_score, 4),
            "post_brier_score": round(self.post_brier_score, 4),
            "brier_score_lift": round(self.brier_score_lift, 4),
            "is_well_calibrated": self.is_well_calibrated,
        }
        if self.status:
            res["status"] = self.status
        if self.pre_ece is not None and self.post_ece is not None:
            res["pre_ece"] = round(self.pre_ece, 4)
            res["post_ece"] = round(self.post_ece, 4)
            if self.ece_lift is not None:
                res["ece_lift"] = round(self.ece_lift, 4)
        if self.adaptive_ece is not None:
            res["adaptive_ece"] = round(self.adaptive_ece, 4)
        if self.temperature is not None:
            res["temperature"] = round(self.temperature, 4)
        if self.conformal_coverage is not None:
            res["conformal_coverage"] = round(self.conformal_coverage, 4)
        return res


class ThresholdReportDTO(BaseDTO):
    """Decision threshold tuning, cost-loss matrix, and error forensics report."""

    default_threshold: float = Field(default=0.50, description="Default 0.50 classification boundary")
    optimal_threshold: float = Field(description="Tuned decision boundary maximizing objective")
    f_beta_score: float = Field(description="Optimal F-beta score achieved")
    precision: float = Field(ge=0.0, le=1.0, description="Precision at optimal threshold")
    recall: float = Field(ge=0.0, le=1.0, description="Recall at optimal threshold")
    confusion_matrix: Dict[str, int] = Field(description="Counts for tn, fp, fn, tp")
    false_positive_count: int = Field(ge=0, description="Count of hard false positives")
    false_negative_count: int = Field(ge=0, description="Count of hard false negatives")
    total_cost_optimal: Optional[float] = Field(default=None, description="Total financial loss at optimal threshold")
    total_cost_default: Optional[float] = Field(default=None, description="Total financial loss at default 0.50 threshold")
    cost_savings: Optional[float] = Field(default=None, description="Financial loss saved by threshold optimization")
    analytical_cost_threshold: Optional[float] = Field(default=None, description="Sheng & Ling (2014) closed-form theoretical threshold")

    def to_compact(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "f_beta_score": round(self.f_beta_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "confusion_matrix": self.confusion_matrix,
        }
        if self.total_cost_optimal is not None and self.total_cost_default is not None:
            res["total_cost_optimal"] = round(self.total_cost_optimal, 2)
            res["total_cost_default"] = round(self.total_cost_default, 2)
            if self.cost_savings is not None:
                res["cost_savings"] = round(self.cost_savings, 2)
        if self.analytical_cost_threshold is not None:
            res["analytical_cost_threshold"] = round(self.analytical_cost_threshold, 4)
        return res
