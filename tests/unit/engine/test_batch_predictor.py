"""Unit tests for Bulk Batch Predictor with DCA Cutoff and Kaggle Submission Integrity Guard."""
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.batch_predictor import BatchPredictor
from ml_mcp.schemas.serving import BatchPredictDTO


def test_batch_predictor_with_dca_optimal_threshold(tmp_path):
    """Verify batch predictor applies optimal Decision Curve Analysis threshold cutoff p*."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    test_ids = [f"id_{i:04d}" for i in range(100)]
    df_test = pd.DataFrame(X[:100], columns=[f"feat_{i}" for i in range(5)])
    df_test["row_id"] = test_ids

    input_path = tmp_path / "test.csv"
    output_path = tmp_path / "predictions.csv"
    df_test.to_csv(input_path, index=False)

    predictor = BatchPredictor(chunksize=30)
    # Apply asymmetric financial DCA cutoff p* = 0.28 instead of 0.50
    dto = predictor.predict_csv(
        model=clf,
        input_csv_path=str(input_path),
        output_csv_path=str(output_path),
        id_column="row_id",
        optimal_threshold=0.28,
    )

    assert isinstance(dto, BatchPredictDTO)
    assert dto.kaggle_submission_ready is True
    assert dto.rows_processed == 100
    assert dto.id_column_verified is True
    assert dto.has_nan_or_inf is False

    pred_df = pd.read_csv(output_path)
    assert "Threshold_Applied" in pred_df.columns
    assert pred_df["Threshold_Applied"].iloc[0] == 0.28


def test_batch_predictor_multiclass_and_regression(tmp_path):
    from sklearn.linear_model import Ridge, LogisticRegression
    # Regression
    X = np.random.randn(50, 3)
    y = X[:, 0] * 2.0
    reg = Ridge().fit(X, y)
    df_reg = pd.DataFrame(X, columns=["a", "b", "c"])
    input_path = tmp_path / "reg_in.csv"
    output_path = tmp_path / "reg_out.csv"
    df_reg.to_csv(input_path, index=False)
    predictor = BatchPredictor(chunksize=20)
    dto_reg = predictor.predict_csv(reg, str(input_path), str(output_path), task_type="regression")
    assert dto_reg.rows_processed == 50
    assert dto_reg.has_nan_or_inf is False

    # Multiclass
    y_mc = np.random.choice([0, 1, 2], size=50)
    clf_mc = LogisticRegression().fit(X, y_mc)
    input_mc = tmp_path / "mc_in.csv"
    output_mc = tmp_path / "mc_out.csv"
    df_reg.to_csv(input_mc, index=False)
    dto_mc = predictor.predict_csv(clf_mc, str(input_mc), str(output_mc), task_type="classification")
    assert dto_mc.rows_processed == 50
