"""Single-File Self-Contained Interactive HTML Evaluation Dashboard Generator."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generates zero-dependency interactive single-file HTML evaluation dashboards with pure SVG charts."""

    def __init__(self) -> None:
        pass

    def _render_kpi_cards(self, metrics: Dict[str, Any]) -> str:
        """Render CSS flexbox KPI summary cards."""
        cards_html = ""
        for name, val in metrics.items():
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            cards_html += f"""
            <div class="kpi-card">
                <div class="kpi-title">{name}</div>
                <div class="kpi-value">{val_str}</div>
            </div>
            """
        return cards_html

    def _render_leaderboard_table(self, leaderboard: List[Dict[str, Any]]) -> str:
        """Render styled HTML table for tournament results."""
        rows_html = ""
        for i, item in enumerate(leaderboard, 1):
            model = item.get("model", f"Model_{i}")
            score = item.get("score", 0.0)
            fit_time = item.get("fit_time", 0.0)
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            time_str = f"{fit_time:.2f}s" if isinstance(fit_time, float) else str(fit_time)
            badge_class = "gold-badge" if i == 1 else "rank-badge"

            rows_html += f"""
            <tr>
                <td><span class="{badge_class}">#{i}</span></td>
                <td><strong>{model}</strong></td>
                <td>{score_str}</td>
                <td>{time_str}</td>
            </tr>
            """
        return rows_html

    def _render_svg_confusion_matrix(self, cm: Dict[str, int]) -> str:
        """Render pure inline SVG 2x2 confusion matrix heatmap."""
        tn = cm.get("tn", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        tp = cm.get("tp", 0)

        svg = f"""
        <svg width="240" height="240" viewBox="0 0 240 240" class="svg-chart">
            <!-- TN -->
            <rect x="20" y="20" width="90" height="90" fill="#10b981" fill-opacity="0.8" rx="6"/>
            <text x="65" y="60" text-anchor="middle" fill="#fff" font-size="14" font-weight="bold">TN: {tn}</text>
            <text x="65" y="80" text-anchor="middle" fill="#e2e8f0" font-size="10">Actual 0 / Pred 0</text>

            <!-- FP -->
            <rect x="130" y="20" width="90" height="90" fill="#ef4444" fill-opacity="0.7" rx="6"/>
            <text x="175" y="60" text-anchor="middle" fill="#fff" font-size="14" font-weight="bold">FP: {fp}</text>
            <text x="175" y="80" text-anchor="middle" fill="#e2e8f0" font-size="10">Actual 0 / Pred 1</text>

            <!-- FN -->
            <rect x="20" y="130" width="90" height="90" fill="#ef4444" fill-opacity="0.7" rx="6"/>
            <text x="65" y="170" text-anchor="middle" fill="#fff" font-size="14" font-weight="bold">FN: {fn}</text>
            <text x="65" y="190" text-anchor="middle" fill="#e2e8f0" font-size="10">Actual 1 / Pred 0</text>

            <!-- TP -->
            <rect x="130" y="130" width="90" height="90" fill="#10b981" fill-opacity="0.8" rx="6"/>
            <text x="175" y="170" text-anchor="middle" fill="#fff" font-size="14" font-weight="bold">TP: {tp}</text>
            <text x="175" y="190" text-anchor="middle" fill="#e2e8f0" font-size="10">Actual 1 / Pred 1</text>
        </svg>
        """
        return svg

    def _render_svg_shap_bars(self, top_features: Dict[str, float]) -> str:
        """Render pure inline SVG horizontal bar chart for top features."""
        if not top_features:
            return "<p>No feature attributions available.</p>"

        max_val = max(top_features.values()) if top_features else 1.0
        max_val = max(max_val, 1e-6)

        y_offset = 25
        bar_height = 20
        chart_height = max(180, len(top_features) * 35 + 20)

        svg = f"""<svg width="450" height="{chart_height}" class="svg-chart">"""
        for feat, val in list(top_features.items())[:10]:
            width = int((val / max_val) * 220)
            svg += f"""
            <text x="10" y="{y_offset + 14}" fill="#cbd5e1" font-size="11" font-family="monospace">{feat[:20]}</text>
            <rect x="160" y="{y_offset}" width="{max(5, width)}" height="{bar_height}" fill="#3b82f6" rx="4"/>
            <text x="{170 + max(5, width)}" y="{y_offset + 14}" fill="#94a3b8" font-size="10">{val:.4f}</text>
            """
            y_offset += 30

        svg += "</svg>"
        return svg

    def generate_dashboard(
        self,
        output_path: str,
        project_name: str,
        champion_model: str,
        metrics: Dict[str, Any],
        leaderboard: Optional[List[Dict[str, Any]]] = None,
        top_features: Optional[Dict[str, float]] = None,
        confusion_matrix: Optional[Dict[str, int]] = None,
    ) -> str:
        """Generate complete standalone HTML dashboard."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        kpi_html = self._render_kpi_cards(metrics)
        leaderboard_html = self._render_leaderboard_table(leaderboard or [])
        cm_svg = self._render_svg_confusion_matrix(confusion_matrix or {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        shap_svg = self._render_svg_shap_bars(top_features or {})

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - ML Evaluation Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --success: #10b981;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; color: var(--text-muted); font-size: 14px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-title {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 22px; font-weight: bold; color: var(--text-main); }}
        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px;
        }}
        .panel h2 {{ margin-top: 0; font-size: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); font-weight: 600; }}
        .gold-badge {{ background: #eab308; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
        .rank-badge {{ background: var(--border); color: #fff; padding: 2px 8px; border-radius: 4px; }}
        .svg-chart {{ display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 {project_name}</h1>
        <p>Champion Model: <strong>{champion_model}</strong> | Automated Production Artifact</p>
    </div>

    <div class="kpi-grid">
        {kpi_html}
    </div>

    <div class="grid-2col">
        <div class="panel">
            <h2>🏆 Tournament Arena Leaderboard</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model Architecture</th>
                        <th>CV Score</th>
                        <th>Training Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {leaderboard_html}
                </tbody>
            </table>
        </div>

        <div class="panel">
            <h2>🎯 Decision Confusion Matrix</h2>
            <div style="display:flex; justify-content:center; align-items:center; height:240px;">
                {cm_svg}
            </div>
        </div>
    </div>

    <div class="panel">
        <h2>🔍 Top 10 TreeSHAP Feature Attributions</h2>
        {shap_svg}
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return os.path.abspath(output_path)
