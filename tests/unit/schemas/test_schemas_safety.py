"""Unit tests for AI Safety, OOD, and Slice Fairness DTOs."""
import pytest

from ml_mcp.schemas.safety import (
    OODReportDTO,
    StressTestReportDTO,
    SliceFairnessDTO,
)


def test_ood_report_dto():
    dto = OODReportDTO(
        detector_name="IsolationForest",
        total_samples=1000,
        ood_detected_count=23,
        ood_ratio=0.023,
        anomaly_threshold=-0.42,
    )

    assert dto.ood_ratio == 0.023
    assert dto.ood_detected_count == 23


def test_stress_test_report_dto():
    dto = StressTestReportDTO(
        perturbation_type="gaussian_noise",
        noise_level=0.05,
        baseline_score=0.90,
        stressed_score=0.87,
        degradation_percentage=3.33,
        robustness_score=96.67,
        is_stress_passed=True,
    )

    assert dto.is_stress_passed is True
    assert dto.robustness_score == 96.67


def test_slice_fairness_dto():
    dto = SliceFairnessDTO(
        protected_attribute="region",
        subgroup_scores={"north": 0.88, "south": 0.86, "east": 0.87},
        max_disparity=0.02,
        disparate_impact_ratio=0.977,
        parity_violated=False,
    )

    assert dto.parity_violated is False
    assert dto.max_disparity == 0.02
