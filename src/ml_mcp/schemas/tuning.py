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
    """Report measuring probability calibration improvement (Platt Scaling / Isotonic)."""

    method: str = Field(description="sigmoid (Platt Scaling) or isotonic")
    pre_brier_score: float = Field(ge=0.0, description="Brier score before calibration")
    post_brier_score: float = Field(ge=0.0, description="Brier score after calibration")
    brier_score_lift: float = Field(description="Improvement delta (positive is better)")
    is_well_calibrated: bool = Field(description="True if post-calibration Brier score <= 0.10")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "pre_brier_score": round(self.pre_brier_score, 4),
            "post_brier_score": round(self.post_brier_score, 4),
            "brier_score_lift": round(self.brier_score_lift, 4),
            "is_well_calibrated": self.is_well_calibrated,
        }


class ThresholdReportDTO(BaseDTO):
    """Decision threshold tuning and error forensics report."""

    default_threshold: float = Field(default=0.50, description="Default 0.50 classification boundary")
    optimal_threshold: float = Field(description="Tuned decision boundary maximizing F-beta")
    f_beta_score: float = Field(description="Optimal F-beta score achieved")
    precision: float = Field(ge=0.0, le=1.0, description="Precision at optimal threshold")
    recall: float = Field(ge=0.0, le=1.0, description="Recall at optimal threshold")
    confusion_matrix: Dict[str, int] = Field(description="Counts for tn, fp, fn, tp")
    false_positive_count: int = Field(ge=0, description="Count of hard false positives")
    false_negative_count: int = Field(ge=0, description="Count of hard false negatives")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "f_beta_score": round(self.f_beta_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "confusion_matrix": self.confusion_matrix,
        }
