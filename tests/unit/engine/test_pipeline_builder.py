"""Unit tests for Defensive Zero-Leakage Pipeline Builder."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder


def test_pipeline_builder_unseen_nan_inference_trap_guard():
    # Training set has ZERO missing values
    train_df = pd.DataFrame({
        "num_1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat_1": ["A", "B", "A", "B", "C"],
        "target": [0, 1, 0, 1, 0],
    })

    builder = DefensivePipelineBuilder()
    pipeline = builder.build_pipeline(
        train_df,
        target_column="target",
        estimator=LogisticRegression(),
        high_cardinality_threshold=5,
    )

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    pipeline.fit(X_train, y_train)

    # Test set contains UNEXPECTED NaNs and unseen categories!
    test_df = pd.DataFrame({
        "num_1": [np.nan, 25.0, np.nan],
        "cat_1": [np.nan, "Z", "A"],  # "Z" is unseen category, np.nan is missing
    })

    # The pipeline MUST predict successfully without throwing ValueError: Input contains NaN!
    predictions = pipeline.predict(test_df)
    assert len(predictions) == 3
    assert all(p in [0, 1] for p in predictions)


def test_pipeline_builder_high_cardinality_target_encoding():
    # 60 distinct categories (exceeding threshold of 50)
    n = 300
    categories = [f"zip_{i % 60}" for i in range(n)]

    df = pd.DataFrame({
        "zip_code": categories,
        "income": np.random.uniform(30000, 120000, size=n),
        "default": np.random.binomial(1, 0.2, size=n),
    })

    builder = DefensivePipelineBuilder(high_cardinality_threshold=50)
    pipeline = builder.build_pipeline(
        df,
        target_column="default",
        estimator=LogisticRegression(),
    )

    X = df.drop(columns=["default"])
    y = df["default"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X.head(10))
    assert len(preds) == 10
