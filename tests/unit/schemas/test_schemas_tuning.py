"""Unit tests for Hyperparameter Tuning and Calibration DTOs."""
import pytest

from ml_mcp.schemas.tuning import (
    OptunaStudyDTO,
    CalibrationReportDTO,
    ThresholdReportDTO,
)


def test_optuna_study_dto():
    dto = OptunaStudyDTO(
        study_name="optuna_catboost_tuning",
        best_trial_number=17,
        best_params={"depth": 6, "l2_leaf_reg": 3.5, "learning_rate": 0.04},
        best_value=0.9105,
        total_trials=50,
        pruned_trials=18,
        direction="maximize",
        metric="pr_auc",
    )

    assert dto.best_value == 0.9105
    assert dto.pruned_trials == 18
    compact = dto.to_compact()
    assert compact["best_trial_number"] == 17
    assert "best_params" in compact


def test_calibration_report_dto():
    dto = CalibrationReportDTO(
        method="sigmoid",
        pre_brier_score=0.152,
        post_brier_score=0.098,
        brier_score_lift=0.054,
        is_well_calibrated=True,
    )

    assert dto.is_well_calibrated is True
    assert dto.brier_score_lift == 0.054


def test_threshold_report_dto():
    dto = ThresholdReportDTO(
        default_threshold=0.50,
        optimal_threshold=0.38,
        f_beta_score=0.884,
        precision=0.86,
        recall=0.91,
        confusion_matrix={"tn": 850, "fp": 40, "fn": 15, "tp": 95},
        false_positive_count=40,
        false_negative_count=15,
    )

    assert dto.optimal_threshold == 0.38
    compact = dto.to_compact()
    assert compact["optimal_threshold"] == 0.38
    assert "confusion_matrix" in compact
