"""Unit tests for Self-Contained Interactive HTML Dashboard Generator."""
import os
import pytest

from ml_mcp.engine.dashboard_generator import DashboardGenerator


def test_dashboard_generator_creates_self_contained_html(tmp_path):
    """Verify dashboard generator creates valid HTML file with KPI cards and charts."""
    generator = DashboardGenerator()
    out_file = tmp_path / "dashboard.html"

    metrics = {
        "Accuracy": 0.952,
        "F1-Macro": 0.941,
        "ROC-AUC": 0.985,
        "Single Latency (ms)": 0.82,
    }
    leaderboard = [
        {"model": "CatBoostClassifier", "score": 0.952, "fit_time": 4.2},
        {"model": "LGBMClassifier", "score": 0.948, "fit_time": 1.8},
        {"model": "XGBClassifier", "score": 0.945, "fit_time": 2.1},
    ]
    top_features = {
        "transaction_amount": 0.35,
        "user_age": 0.22,
        "hour_of_day": 0.18,
    }

    path = generator.generate_dashboard(
        output_path=str(out_file),
        project_name="Fraud Defense Arena",
        champion_model="CatBoostClassifier",
        metrics=metrics,
        leaderboard=leaderboard,
        top_features=top_features,
        confusion_matrix={"tn": 850, "fp": 30, "fn": 15, "tp": 105},
    )

    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "<!DOCTYPE html>" in html
    assert "Fraud Defense Arena" in html
    assert "CatBoostClassifier" in html
    assert "transaction_amount" in html
    assert "<svg" in html
