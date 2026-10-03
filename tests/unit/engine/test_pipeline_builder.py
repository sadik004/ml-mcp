"""Unit tests for Defensive Zero-Leakage Pipeline Builder with MNAR Indicators."""
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
        "cat_1": [np.nan, "Z", "A"],
    })

    # The pipeline MUST predict successfully without throwing ValueError: Input contains NaN!
    predictions = pipeline.predict(test_df)
    assert len(predictions) == 3
    assert all(p in [0, 1] for p in predictions)


def test_pipeline_builder_high_cardinality_target_encoding():
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


def test_pipeline_builder_mice_multivariate_imputation():
    df = pd.DataFrame({
        "age": [20.0, 25.0, 45.0, 52.0, np.nan, 30.0],
        "salary": [25000.0, 32000.0, 75000.0, 88000.0, 82000.0, 40000.0],
        "category": ["eng", "eng", "mgmt", "mgmt", "mgmt", "eng"],
        "target": [0, 0, 1, 1, 1, 0],
    })

    builder = DefensivePipelineBuilder(imputation_strategy="iterative")
    pipeline = builder.build_pipeline(df, target_column="target")

    X = df.drop(columns=["target"])
    y = df["target"]
    pipeline.fit(X, y)

    transformed_X = pipeline.named_steps["preprocessor"].transform(X)
    assert not np.isnan(transformed_X).any()


def test_pipeline_builder_missingness_indicator_preservation():
    # Verify that add_indicator=True preserves MNAR indicator flags (Jaeger et al. NeurIPS 2023)
    train_df = pd.DataFrame({
        "income": [50000.0, np.nan, 80000.0, 120000.0, np.nan, 95000.0],
        "debt": [1000.0, 2000.0, 1500.0, 5000.0, 4500.0, 3000.0],
        "target": [0, 1, 0, 0, 1, 0],
    })

    builder = DefensivePipelineBuilder(imputation_strategy="median")
    pipeline = builder.build_pipeline(train_df, target_column="target")

    X = train_df.drop(columns=["target"])
    y = train_df["target"]
    pipeline.fit(X, y)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)

    # 2 numeric columns + 1 missingness indicator for 'income' = 3 total features!
    assert transformed.shape[1] == 3
    assert not np.isnan(transformed).any()
