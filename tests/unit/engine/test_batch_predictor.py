"""Unit tests for Bulk Batch Predictor and Kaggle Submission Integrity Guard."""
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml_mcp.engine.batch_predictor import BatchPredictor
from ml_mcp.schemas.serving import BatchPredictDTO


def test_batch_predictor_classification_kaggle_submission_ready(tmp_path):
    """Verify batch predictor produces Kaggle submission ready predictions."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # Generate unlabelled test CSV with explicit ID column
    test_ids = [f"id_{i:04d}" for i in range(100)]
    X_test_df = pd.DataFrame(X[:100], columns=[f"feat_{i}" for i in range(5)])
    X_test_df.insert(0, "record_id", test_ids)

    input_csv = tmp_path / "test_unlabelled.csv"
    output_csv = tmp_path / "submission.csv"
    X_test_df.to_csv(input_csv, index=False)

    predictor = BatchPredictor(chunksize=50)
    dto = predictor.predict_csv(
        model=clf,
        input_csv_path=str(input_csv),
        output_csv_path=str(output_csv),
        id_column="record_id",
        feature_columns=[f"feat_{i}" for i in range(5)],
        task_type="classification",
    )

    assert isinstance(dto, BatchPredictDTO)
    assert dto.rows_processed == 100
    assert dto.id_column_verified is True
    assert dto.has_nan_or_inf is False
    assert dto.kaggle_submission_ready is True
    assert os.path.exists(output_csv)

    # Read output and verify columns
    out_df = pd.read_csv(output_csv)
    assert len(out_df) == 100
    assert list(out_df["record_id"]) == test_ids
    assert "Predicted_Label" in out_df.columns
    assert "Confidence_Score" in out_df.columns
    assert out_df["Confidence_Score"].between(0.0, 1.0).all()
