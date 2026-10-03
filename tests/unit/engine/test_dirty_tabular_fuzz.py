"""Adversarial Tabular Fuzz Testing Suite (DataPerf / ACM CACM Standards)."""
import numpy as np
import pandas as pd
import pytest
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.engine.calibrator import ProbabilityCalibrator
from sklearn.ensemble import RandomForestClassifier


def test_adversarial_dirty_strings_and_currency_symbols():
    """Verify pipeline handles currency symbols, commas, and dirty text defensively."""
    df = pd.DataFrame({
        "revenue": ["$1,250.00", "$4,900.50", "€350.20", "N/A", "$0.00", "--", "$10,000.00"],
        "category": ["A", "B", "A", "C", None, "B", "A"],
        "target": [0, 1, 0, 1, 0, 1, 0]
    })
    builder = DefensivePipelineBuilder()
    pipe = builder.build_pipeline(df, target_column="target")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Must fit and transform without raising unhandled ValueError
    X_trans = pipe.fit_transform(X, y)
    assert X_trans is not None
    assert X_trans.shape[0] == 7
    assert not np.isnan(X_trans).any()


def test_adversarial_extreme_zero_inflation():
    """Verify extreme 99% zero-inflated continuous features do not cause SVD / scaling collapse."""
    n_samples = 500
    zeros = np.zeros(n_samples)
    zeros[:5] = [10.5, 20.2, 5.1, 100.0, 50.0]  # 99% zero-inflated
    df = pd.DataFrame({
        "sparse_feat": zeros,
        "dense_feat": np.random.randn(n_samples),
        "target": np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])
    })
    builder = DefensivePipelineBuilder()
    pipe = builder.build_pipeline(df, target_column="target")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_trans = pipe.fit_transform(X, y)
    assert X_trans.shape[0] == n_samples
    assert not np.isnan(X_trans).any()


def test_multiclass_dirichlet_calibration():
    """Verify 3-class classification uses Dirichlet calibration over probability simplex."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=250, n_features=5, n_informative=3, n_classes=3, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    calibrator = ProbabilityCalibrator()
    report, calibrated_model = calibrator.calibrate(
        model=clf,
        X=X,
        y=y,
        task_type="classification",
        method="beta",  # Auto-routes to Dirichlet on K > 2
    )
    
    assert report.method == "dirichlet"
    cal_probs = calibrated_model.predict_proba(X[:10])
    assert cal_probs.shape[1] == 3
    # Check probability simplex sum(p) == 1.0
    row_sums = np.sum(cal_probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
