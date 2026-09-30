"""Unit tests for Sub-10s TreeSHAP explainability engine."""
import time
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_mcp.engine.explainer import TreeShapExplainer


def test_treeshap_explainer_fast_execution_and_token_guard():
    """Verify TreeSHAP executes in under 10 seconds and caps top 10 features."""
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=8,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(15)]
    X_df = pd.DataFrame(X, columns=feature_names)

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_df, y)

    explainer = TreeShapExplainer()
    start_time = time.perf_counter()
    report = explainer.explain(
        model=clf,
        X=X_df,
        feature_names=feature_names,
        top_k=10,
    )
    elapsed = time.perf_counter() - start_time

    assert elapsed < 10.0, f"SHAP execution took {elapsed:.2f}s, expected < 10.0s"
    assert "top_features" in report
    assert len(report["top_features"]) <= 10
    assert "feature_directions" in report
    assert len(report["feature_directions"]) <= 10
    assert "execution_time_seconds" in report
    assert report["execution_time_seconds"] < 10.0


def test_treeshap_explainer_fallback_for_linear_model():
    """Verify non-tree models gracefully fall back without throwing exceptions."""
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    feature_names = [f"f_{i}" for i in range(6)]
    clf = LogisticRegression()
    clf.fit(X, y)

    explainer = TreeShapExplainer()
    report = explainer.explain(
        model=clf,
        X=X,
        feature_names=feature_names,
        top_k=5,
    )

    assert "top_features" in report
    assert len(report["top_features"]) <= 5
    assert report["explainer_type"] in ["LinearExplainer", "PermutationExplainer", "CoefficientsFallback"]
