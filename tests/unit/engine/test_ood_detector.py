"""Unit tests for Out-of-Distribution (OOD) Detector Engine."""
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from ml_mcp.engine.ood_detector import OODDetector
from ml_mcp.schemas.safety import OODReportDTO


def test_ood_detector_isolation_forest_flags_outliers():
    """Verify Isolation Forest flags extreme outliers as OOD."""
    np.random.seed(42)
    # Generate in-distribution Gaussian data
    X_train, _ = make_blobs(n_samples=300, n_features=5, centers=1, cluster_std=1.0, random_state=42)
    
    # Generate test set: 90 normal points + 10 extreme outliers
    X_normal, _ = make_blobs(n_samples=90, n_features=5, centers=1, cluster_std=1.0, random_state=100)
    X_outliers = np.random.uniform(low=20.0, high=30.0, size=(10, 5))
    X_test = np.vstack([X_normal, X_outliers])

    detector = OODDetector(method="isolation_forest", contamination=0.10, random_state=42)
    detector.fit(X_train)
    report = detector.detect(X_test)

    assert isinstance(report, OODReportDTO)
    assert report.total_samples == 100
    assert report.ood_detected_count >= 8
    assert report.ood_ratio >= 0.08
    assert report.detector_name == "IsolationForest"


def test_ood_detector_mahalanobis_handles_singular_matrix():
    """Verify Mahalanobis distance handles singular/collinear features with pseudo-inverse."""
    np.random.seed(42)
    # Collinear data where column 2 = 2 * column 1
    col1 = np.random.randn(200, 1)
    col2 = 2.0 * col1
    col3 = np.random.randn(200, 1)
    X_train = np.hstack([col1, col2, col3])

    detector = OODDetector(method="mahalanobis", contamination=0.05)
    detector.fit(X_train)

    # In-distribution test point and extreme outlier
    test_normal = np.array([[0.1, 0.2, 0.1]])
    test_extreme = np.array([[50.0, 100.0, 50.0]])
    X_test = np.vstack([test_normal, test_extreme])

    report = detector.detect(X_test)
    assert isinstance(report, OODReportDTO)
    assert report.total_samples == 2
    assert report.ood_detected_count == 1
    assert report.detector_name == "MahalanobisDistance"
