# 5-Minute Quickstart Tutorial: `ml-mcp`

> **Objective:** From raw tabular CSV to Audited Dataset, Stacking Ensemble Champion, TreeSHAP Insights, Docker Container, and Standalone HTML Dashboard in under 5 minutes.

---

## 1. Prerequisites

Ensure you have Python 3.10+ installed and install `ml-mcp`:

```bash
git clone https://github.com/sadik004/ml-mcp.git
cd ml-mcp
pip install -e .
```

Verify your installation by running the 15-second smoke test:

```bash
python scripts/smoke_test.py
```

Expected output:
```
==================================================
>> Running ml-mcp 15-Second End-to-End Smoke Test
==================================================
 [1/8] Data Synthesis: 200 rows, 6 features (0.02s)
 [2/8] Pre-Flight Audit: Hygiene & guards passed (0.00s)
 [3/8] Defensive Pipeline: Transformed shape (200, 6) (0.01s)
 [4/8] Tournament: Champion 'StackingEnsemble' (Score: 0.9196) (5.53s)
 [5/8] Threshold Optimization: Optimal boundary 0.3565 (F1: 0.7712) (0.01s)
 [6/8] TreeSHAP: Top feature 'feat_3' (0.00s)
 [7/8] AI Safety: OOD=10, Robustness=99.3/100 (0.20s)
 [8/8] Artifacts & Dashboard: Generated at C:\...\dashboard.html (0.33s)
==================================================
[SUCCESS] End-to-End Smoke Test Passed in 6.09 seconds! (Threshold <= 15.0s)
==================================================
```

---

## 2. Walkthrough via Python API

You can use `ml-mcp` either directly as a Python library or via FastMCP JSON-RPC. Here is the full programmatic workflow:

### Step 1: Pre-Flight Audit & Hygiene Check

```python
import pandas as pd
from ml_mcp.engine.auditor import PreFlightAuditor

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

# Audit dataset for sentinels (-999, ?), missing values, and ID columns
auditor = PreFlightAuditor(target_column="churn")
audit_report = auditor.audit(df)

print(f"Data Health: {audit_report.health_status}")
print(f"Sentinel Warnings: {audit_report.sentinels_detected}")
print(f"Class Balance Ratio: {audit_report.class_balance_ratio}")
```

### Step 2: Build Zero-Leakage Defensive Pipeline

```python
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder

builder = DefensivePipelineBuilder()
pipeline = builder.build(
    numeric_features=["age", "tenure", "monthly_charges"],
    categorical_features=["contract_type", "payment_method"],
    scaler="robust",
    handle_unknown_categories=True
)

# Fit strictly on training split
X_transformed = pipeline.fit_transform(df.drop(columns=["churn"]))
```

### Step 3: Run 8-Model Competitive Arena

```python
from ml_mcp.engine.tournament import TournamentArena

arena = TournamentArena(cv_folds=5, metric="f1", task_type="classification")
leaderboard = arena.run(df, target_column="churn")

print(f"Champion Model: {leaderboard.champion_name}")
for rank, model in enumerate(leaderboard.rankings, 1):
    print(f"#{rank} {model.name}: {model.mean_score:.4f} (+/- {model.std_score:.4f})")
```

### Step 4: Asymmetric Cost Threshold Optimization

In churn or fraud prediction, missing a positive case (False Negative) is often 5x to 10x more costly than a false alarm (False Positive):

```python
from ml_mcp.engine.threshold import CostMatrixThresholdOptimizer

optimizer = CostMatrixThresholdOptimizer()
opt_res = optimizer.optimize(
    y_true=y_val,
    y_prob=champion_probs,
    cost_fp=50.0,   # Cost of reaching out to a non-churning customer
    cost_fn=500.0,  # Cost of losing a churning customer
)

print(f"Optimal Decision Threshold: {opt_res.optimal_threshold:.4f} (Default was 0.5000)")
print(f"Total Business Cost Saved: ${opt_res.cost_savings:,.2f}")
```

### Step 5: Sub-10s TreeSHAP Explainability

```python
from ml_mcp.engine.explainer import Sub10sTreeSHAPExplainer

explainer = Sub10sTreeSHAPExplainer()
shap_res = explainer.explain(
    model=arena.champion_estimator,
    X_background=X_transformed[:100],
    X_query=X_transformed[:500]
)

print("Top 3 Driving Features:")
for feat, score in shap_res.top_features[:3]:
    print(f" - {feat}: {score:.4f}")
```

### Step 6: Generate Standalone HTML Dashboard

```python
from ml_mcp.engine.dashboard_generator import StandaloneHTMLDashboardGenerator

dash_gen = StandaloneHTMLDashboardGenerator()
html_content = dash_gen.generate(
    model_name=leaderboard.champion_name,
    metrics={"Accuracy": 0.92, "F1-Score": 0.88, "ROC-AUC": 0.94},
    top_features=shap_res.top_features[:5],
    output_path="churn_dashboard.html"
)
print("Dashboard generated at churn_dashboard.html (open in any browser)!")
```

### Step 7: Export Production Serving Bundle & Dockerfile

```python
from ml_mcp.engine.api_generator import FastAPIRouterGenerator
from ml_mcp.engine.docker_generator import DockerSpecGenerator

# 1. Generate 3-Tier FastAPI Router
api_gen = FastAPIRouterGenerator()
api_gen.generate(
    model_path="models/champion.joblib",
    features=["age", "tenure", "monthly_charges", "contract_type", "payment_method"],
    output_dir="serving"
)

# 2. Generate Multi-Stage Dockerfile & Docker Compose
docker_gen = DockerSpecGenerator()
docker_gen.generate(output_dir="serving")

print("Serving bundle ready in ./serving. Build with: docker compose up --build")
```

---

## 3. Starting the FastMCP Server

To use `ml-mcp` interactively through LLM agents:

```bash
python -m ml_mcp.server
```

The server listens on `stdio` and accepts all 25 standard MCP JSON-RPC tool calls.
