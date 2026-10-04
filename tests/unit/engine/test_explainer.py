"""Unit tests for Sub-10s TreeSHAP and Local Waterfall Attribution."""
import time
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ml_mcp.engine.explainer import TreeShapExplainer


def test_treeshap_explainer_fast_execution_and_token_guard():
    """Verify TreeSHAP executes in under 10 seconds and caps top 10 features."""
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=6,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(12)]
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
    assert report["execution_time_seconds"] < 10.0


def test_treeshap_local_waterfall_attribution():
    """Verify local instance waterfall attribution breakdown."""
    X, y = make_classification(n_samples=150, n_features=6, random_state=42)
    feature_names = [f"f_{i}" for i in range(6)]
    X_df = pd.DataFrame(X, columns=feature_names)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_df, y)

    explainer = TreeShapExplainer()
    instance_report = explainer.explain_instance(
        model=clf,
        x_row=X_df.iloc[[0]],
        background_X=X_df,
        top_k=5,
    )

    assert "prediction" in instance_report
    assert "base_value" in instance_report
    assert "waterfall_steps" in instance_report
    assert len(instance_report["waterfall_steps"]) <= 5
    assert "top_positive_drivers" in instance_report
    assert "top_negative_drivers" in instance_report


def test_treeshap_zero_fake_permutation_fallback():
    """Verify non-tree models (e.g. KNN) use genuine permutation importance with zero fake 1.0."""
    X, y = make_classification(n_samples=80, n_features=5, random_state=42)
    feature_names = [f"f_{i}" for i in range(5)]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    explainer = TreeShapExplainer()
    report = explainer.explain(
        model=knn,
        X=X,
        feature_names=feature_names,
        top_k=4,
    )

    assert report["explainer_type"] == "PermutationExplainer"
    assert "top_features" in report
    # Critical: verify NO fake 1.0 dummy weights
    for feat_name, imp in report["top_features"].items():
        assert imp != 1.0 or len(report["top_features"]) == 1


def test_treeshap_local_waterfall_non_tree():
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    feature_names = [f"f_{i}" for i in range(4)]
    X_df = pd.DataFrame(X, columns=feature_names)
    clf = LogisticRegression().fit(X_df, y)

    explainer = TreeShapExplainer()
    instance_report = explainer.explain_instance(
        model=clf,
        x_row=X_df.iloc[[0]],
        background_X=X_df,
        top_k=3,
    )
    assert "waterfall_steps" in instance_report
    assert len(instance_report["waterfall_steps"]) <= 3
