"""Unit tests for Self-Contained Interactive HTML Dashboard Generator with pure SVG DCA & Calibration charts."""
import os
import pytest

from ml_mcp.engine.dashboard_generator import DashboardGenerator


def test_dashboard_generator_creates_dca_and_calibration_svgs(tmp_path):
    """Verify dashboard generator creates valid HTML file with pure SVG DCA and Calibration diagrams."""
    generator = DashboardGenerator()
    out_file = tmp_path / "dashboard.html"

    metrics = {
        "ROC-AUC": 0.978,
        "Macro-F1": 0.954,
        "Optimal DCA Net Benefit": 0.32,
    }

    path = generator.generate_dashboard(
        output_path=str(out_file),
        project_name="Fraud Detection Engine",
        champion_model="Ensemble_Stacker",
        metrics=metrics,
    )

    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    # Must contain pure SVG charts for DCA and Calibration
    assert "<svg" in html
    assert "Decision Curve Analysis" in html
    assert "Beta Calibration Reliability" in html
    assert "Model Net Benefit" in html
