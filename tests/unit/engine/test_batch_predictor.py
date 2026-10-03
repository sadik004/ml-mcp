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
