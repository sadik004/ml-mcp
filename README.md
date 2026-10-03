<div align="center">

# 🧠 `ml-mcp`: Machine Learning Model Context Protocol Server
### *Enterprise & Kaggle Grandmaster Edition*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP%20v2.0-8A2BE2.svg?style=for-the-badge&logo=anthropic&logoColor=white)](https://github.com/modelcontextprotocol)
[![Tests Passing](https://img.shields.io/badge/tests-100%20passed-success.svg?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![Smoke Test](https://img.shields.io/badge/smoke%20test-6.09s%20passed-green.svg?style=for-the-badge&logo=speedtest&logoColor=white)](#-the-15-second-smoke-test)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Sub--millisecond-orange.svg?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Docker Ready](https://img.shields.io/badge/docker-multi--stage%20non--root-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>A production-grade, 3-tier clean architecture Model Context Protocol (MCP) server engineered for autonomous AI agents and data science teams.</b><br>
  Dual-topology execution across <b>Local IDEs (Antigravity IDE, Claude Code, Cursor)</b> and <b>Google Colab Cloud GPUs (NVIDIA T4 / L4 / A100)</b>.
</p>

[Quickstart Guide](docs/QUICKSTART_GUIDE.md) • [System Architecture](docs/ARCHITECTURE.md) • [25-Tool Reference](docs/TOOL_REFERENCE.md) • [Colab Integration](docs/COLAB_INTEGRATION.md)

---

</div>

## 📌 Table of Contents

- [The Core Mission](#-the-core-mission)
- [Why `ml-mcp`? (The Problem vs Solution)](#-why-ml-mcp-the-problem-vs-solution)
- [System Architecture & 3-Tier Clean Pattern](#-system-architecture--3-tier-clean-pattern)
- [Dual-Topology Execution (Local + Colab GPU)](#-dual-topology-execution-local--colab-gpu)
- [The 25 FastMCP Production Tools](#-the-25-fastmcp-production-tools)
- [Core Defensive Invariants](#-core-defensive-invariants)
- [Quickstart: Local Installation](#-quickstart-local-installation)
- [Client Integration (Claude Desktop & Antigravity IDE)](#-client-integration-claude-desktop--antigravity-ide)
- [The 15-Second Smoke Test](#-the-15-second-smoke-test)
- [Automated Testing & Quality Gates](#-automated-testing--quality-gates)
- [Production Deployment with Docker & FastAPI](#-production-deployment-with-docker--fastapi)
- [Repository Structure](#-repository-structure)
- [License & Authors](#-license--authors)

---

## 🎯 The Core Mission

`ml-mcp` equips autonomous coding agents (Claude, Gemini, GPT-4) with **Kaggle Grandmaster-grade machine learning superpowers**. Rather than generating fragile, unvalidated code blocks that fail on missing values or cause catastrophic target leakage, agents call `ml-mcp`'s 25 defensive tools via standard MCP JSON-RPC.

From raw tabular data auditing to cross-validated model tournaments, Optuna Bayesian optimization, sub-10s TreeSHAP attributions, out-of-distribution detection, ONNX export, and standalone HTML dashboards—`ml-mcp` runs the entire pipeline deterministically, defensively, and reproducibly.

---

## ⚖️ Why `ml-mcp`? (The Problem vs Solution)

| Classic Agent ML Failures | `ml-mcp` Defensive Guarantees |
|---|---|
| **Catastrophic Target Leakage:** Scaling and imputer statistics computed across the whole dataset before splitting. | **Zero-Leakage Invariant:** All transformations fit strictly inside cross-validation training folds using `DefensivePipelineBuilder`. |
| **Accuracy Paradox:** Naive 99% accuracy on imbalanced data by simply predicting the majority class. | **Cost-Sensitive Learning (Anti-SMOTE):** Automatic fallback to Macro-F1, ROC-AUC, or PR-AUC metrics, coupled with exact inverse-frequency sample weights ($w_i = \frac{N}{K \cdot N_{y_i}}$) and Random Resampling with Shrinkage. |
| **Silent NaN Inferences:** Production models crash when unobserved missing values appear in future inference requests. | **Omnipresent Median/Mode Imputers:** Every pipeline retains a fallback imputer regardless of whether training data had NaNs. |
| **Context Window Exhaustion:** Raw arrays, massive correlation grids, and thousands of predictions blow up LLM token limits. | **Token Shield Architecture:** `view="compact"` returns $\le 15$ essential metrics; `view="detailed"` provides complete distributions. |
| **Scientific Type JSON Crashes:** NumPy floats, Pandas Series, and NaNs trigger serialization errors. | **Recursive JSON Sanitizer:** Seamlessly converts NumPy scalars, NaNs, and Datetimes up to depth 30 before serialization. |
| **Arbitrary 0.500 Decision Thresholds:** Blindly applying 0.500 thresholds despite asymmetric financial costs (e.g., fraud vs churn). | **Asymmetric Cost Matrix Optimizer:** Computes exact optimal cutoff boundaries minimizing real-world dollar loss. |

---

## 🏗️ System Architecture & 3-Tier Clean Pattern

`ml-mcp` enforces a strict Separation of Concerns across three architectural tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MCP PROTOCOL CLIENT                             │
│         (Claude Desktop / Antigravity IDE / Cursor / OpenCode)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ FastMCP JSON-RPC 2.0 (Stdio / SSE)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 1: TOOLS & ROUTERS LAYER                         │
│                      (src/ml_mcp/tools.py)                             │
│  - 25 FastMCP `@mcp.tool()` entry points                              │
│  - Token Shield: view="compact" | "detailed" budgeting                 │
│  - Parameter sanitization & boundary validation                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Typed Parameters
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 2: DETERMINISTIC ENGINE LAYER                   │
│                     (src/ml_mcp/engine/*.py)                           │
│  - Pure Domain Logic & Mathematical Formulations                       │
│  - Zero MCP or HTTP protocol dependencies                              │
│  - Memory-safe, GPU-aware, chunked operations                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pure DTOs
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  TIER 3: DATA CONTRACTS & SCHEMAS                      │
│                    (src/ml_mcp/schemas/*.py)                           │
│  - Pydantic v2 DTOs with extra="forbid"                                │
│  - Recursive JSON Sanitizer (NumPy / Pandas -> Native JSON)            │
└────────────────────────────────────────────────────────────────────────┘
```

Detailed architectural diagrams and mechanics are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## ⚡ Dual-Topology Execution (Local + Colab GPU)

`ml-mcp` natively supports **hybrid execution**:

```
                       ┌───────────────────────────────┐
                       │   Local Machine (IDE Agent)   │
                       │   - Fast Data Audit (< 1s)    │
                       │   - Defensive Pipeline Build  │
                       │   - Decision Threshold Tuning │
                       │   - Standalone Dashboard Gen  │
                       └──────────────┬────────────────┘
                                      │
               Remote SSE Tunnel      │ Optional Delegation
               (ngrok / colab-mcp)    │ for Heavy Compute
                                      ▼
                       ┌───────────────────────────────┐
                       │    Google Colab Cloud GPU     │
                       │    (NVIDIA T4 / L4 / A100)    │
                       │   - 8-Model Competitive Arena │
                       │   - Out-of-Fold Stacking      │
                       │   - Optuna 100-Trial Search   │
                       │   - Sub-10s Fast TreeSHAP     │
                       │   - Google Drive Checkpoints  │
                       └───────────────────────────────┘
```

Step-by-step setup instructions for Google Colab are available in [docs/COLAB_INTEGRATION.md](docs/COLAB_INTEGRATION.md).

---

## 🛠️ The 37 FastMCP Production Tools

All tools are decorated with @mcp.tool(), protected by the Token Shield, and return sanitized JSON-RPC responses:

| # | Tool Identifier | Category | Primary Functionality |
|---|---|---|---|
| **1** | ml_ping | **Core** | Server health, Python environment, platform & CUDA GPU detection. |
| **2** | ml_audit_dataset | **Phase 1: Hygiene** | Pre-flight audit for sentinels (-999, ?), nulls, constant features, duplicates, and class imbalance. |
| **3** | ml_detect_target_leakage | **Phase 1: Hygiene** | Identifies target leakage, future timestamp features, and perfect predictors. |
| **4** | ml_check_collinearity | **Phase 1: Hygiene** | Variance Inflation Factor (VIF) and pairwise correlation matrix analysis with competitive drop rule. |
| **5** | ml_detect_label_errors | **Phase 1: Hygiene** | Detects corrupt/noisy labels via MIT Confident Learning (Northcutt et al., 2021). |
| **6** | ml_verify_constraints | **Phase 1: Hygiene** | Validates physical limits, non-negativity, and automated Amazon Deequ 3x-IQR statistical outlier rules. |
| **7** | ml_track_lineage | **Phase 1: Lineage** | Cryptographic artifact & dataset lineage tracking with SHA-256 digests and Git commit hashes. |
| **8** | ml_handle_text_features | **Phase 2: Features** | Detects free-form natural language text features while filtering UUIDs and random hashes. |
| **9** | ml_auto_clean_and_pipe | **Phase 2: Features** | Zero-leakage Scikit-Learn ColumnTransformer preprocessing pipeline with median/mode imputers and scalers. |
| **10** | ml_balance_classes | **Phase 2: Features** | Cost-sensitive sample weighting (anti-SMOTE, Wallace et al. 2021), RandomUnderSampler, and RandomOverSampler with shrinkage. |
| **11** | ml_synthesize_features | **Phase 2: Features** | Generates cyclical sin/cos features, safe ratios, and ExploreKit group aggregations with empirical Bayes smoothing. |
| **12** | ml_prune_features | **Phase 2: Features** | Prunes noisy features using gradient-boosted out-of-fold (OOF) cross-validated permutation importance. |
| **13** | ml_transform_target | **Phase 2: Features** | Linearizes highly skewed continuous targets (Log1p, Box-Cox, and Yeo-Johnson transformations). |
| **14** | ml_benchmark_models | **Phase 3: Arena** | Competitive tournament across LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, and Ridge/Logistic. |
| **15** | ml_create_ensemble | **Phase 3: Arena** | Multi-model Stacking and Voting ensembles with out-of-fold (OOF) meta-learners. |
| **16** | ml_tune_hyperparameters | **Phase 3: Tuning** | Optuna Bayesian optimization with pruning over hyperparameter spaces. |
| **17** | ml_pseudo_label_loop | **Phase 3: Semi-Supervised** | Extracts high-confidence pseudo-labels from unlabeled pools to expand training sets. |
| **18** | ml_calibrate_probabilities | **Phase 4: Safety** | 3-Parameter Beta Calibration (Kull et al.) and Equal-Frequency Adaptive ECE (Roelofs et al. 2022) with Platt/Isotonic fallbacks. |
| **19** | ml_tune_threshold_and_errors | **Phase 4: Safety** | Decision Curve Analysis (DCA Net Benefit, Vickers & Elkin) vs Treat All/None, with Sheng & Ling cost matrix thresholding. |
| **20** | ml_explain_predictions | **Phase 4: Safety** | Sub-10s TreeSHAP local attributions and global feature importance rankings. |
| **21** | ml_detect_ood | **Phase 4: Safety** | Helmholtz Free Energy OOD scoring ($E(x) = -T \log \sum \exp(f_i/T)$, Liu et al. NeurIPS 2020) and Mahalanobis distance. |
| **22** | ml_stress_test_and_fairness | **Phase 4: Safety** | Covariance-preserving manifold stress ($\\delta \\sim \\mathcal{N}(0, \\epsilon^2 \\Sigma)$) and intersectional subgroup fairness (Kearns et al. ICML 2018). |
| **23** | ml_conformal_risk_control | **Phase 4: Safety** | Mondrian (Class-Conditional) Conformal Prediction ($P(Y \\in C(X) \\mid Y=k) \\ge 1-\\alpha$) and RAPS regularized prediction sets. |
| **24** | ml_batch_predict | **Phase 5: Serving** | High-throughput chunked batch inference on massive CSV datasets. |
| **25** | ml_export_and_document | **Phase 5: Governance** | Auto-synthesizes Mitchell et al. production MODEL_CARD.md. |
| **26** | ml_optimize_inference | **Phase 5: Serving** | Converts Scikit-Learn / LightGBM models into ONNX runtime graph format with fp16/int8 quantization. |
| **27** | ml_generate_eval_dashboard | **Phase 5: Visuals** | Generates standalone HTML evaluation dashboard with interactive confusion matrices and charts. |
| **28** | ml_generate_serving_api | **Phase 5: Serving** | Scaffolds a production-ready 3-tier FastAPI serving microservice with typed Pydantic schemas. |
| **29** | ml_generate_docker_spec | **Phase 5: Deployment** | Generates minimal, non-root, multi-stage production Dockerfile and docker-compose.yml. |
| **30** | ml_monitor_drift | **Phase 5: Monitoring** | Population Stability Index (PSI) and Wasserstein Distance data drift monitoring. |
| **31** | ml_generate_colab_notebook | **Phase 6: Cloud GPU** | Generates executable .ipynb Colab notebook for remote cloud execution. |
| **32** | ml_colab_status | **Phase 6: Cloud GPU** | Checks remote Google Colab GPU health, Tesla T4 VRAM availability, and bridge connectivity. |
| **33** | ml_colab_execute | **Phase 6: Cloud GPU** | Executes arbitrary Python and Machine Learning code directly on remote Google Colab GPU. |
| **34** | ml_colab_upload | **Phase 6: Cloud GPU** | Uploads local CSV datasets, models, or scripts directly into Google Colab filesystem. |
| **35** | ml_colab_download | **Phase 6: Cloud GPU** | Downloads trained models, checkpoints, or metric reports from Colab to local workspace. |
| **36** | ml_colab_stop | **Phase 6: Cloud GPU** | Terminates remote Google Colab GPU session to release cloud compute resources. |
| **37** | ml_cancel_job | **Phase 6: Cloud GPU** | Cancels long-running background asynchronous ML jobs. |

Complete schemas, inputs, and example JSON payloads are in [docs/TOOL_REFERENCE.md](docs/TOOL_REFERENCE.md) and deep-dives in [docs/tools/](docs/tools/README.md).

## 🛡️ Core Defensive Invariants

1. **Zero Data Leakage:** Imputers and scalers are fitted strictly inside `StratifiedKFold` training splits.
2. **Omnipresent Median Imputers:** Models never throw unhandled NaN exceptions during production inference.
3. **Recursive JSON Sanitization:** High-dimensional arrays, NumPy scalars, and pandas `NaT`/`NA` values are converted into valid JSON primitives.
4. **Token Shield:** Outputs default to `view="compact"` ($\le 15$ key metrics) to protect LLM context windows.
5. **Accuracy Paradox Guard:** High class imbalance automatically shifts objective metrics to Macro-F1 or PR-AUC.

---

## 🚀 Quickstart: Local Installation

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/sadik004/ml-mcp.git
cd ml-mcp

# Install in editable mode with all dependencies
pip install -e .
```

### 2. Verify Server Health

```bash
python -m ml_mcp.server
```

---

## 🔌 Client Integration (Claude Desktop & Antigravity IDE)

Add `ml-mcp` to your MCP configuration file:

### On Windows (`%APPDATA%\Claude\claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "ml-mcp": {
      "command": "python",
      "args": ["-m", "ml_mcp.server"],
      "cwd": "C:\\path\\to\\ml-mcp"
    }
  }
}
```

### On macOS / Linux (`~/.config/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "ml-mcp": {
      "command": "python3",
      "args": ["-m", "ml_mcp.server"],
      "cwd": "/path/to/ml-mcp"
    }
  }
}
```

---

## ⏱️ The 15-Second Smoke Test

`ml-mcp` provides a 1-click end-to-end smoke test script that verifies the full machine learning lifecycle in under 15 seconds:

```bash
python scripts/smoke_test.py
```

### Smoke Test Checkpoints:
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
 [8/8] Artifacts & Dashboard: Generated at dashboard.html (0.33s)
==================================================
[SUCCESS] End-to-End Smoke Test Passed in 6.09 seconds! (Threshold <= 15.0s)
==================================================
```

---

## 🧪 Automated Testing & Quality Gates

The test suite contains **100 comprehensive unit and integration tests** validating every engine and tool:

```bash
# Run complete test suite with timing and summary
pytest tests/ -v
```

```
============================== 100 passed in 24.72s ===============================
```

### Test Coverage Highlights:
- `tests/unit/test_json_sanitizer.py`: NumPy floats, ints, NaNs, infinities, and Pandas timestamps.
- `tests/unit/engine/test_auditor.py`: Sentinel detection, constant columns, null percentages.
- `tests/unit/engine/test_pipeline_builder.py`: Zero-leakage imputer and robust scaling.
- `tests/unit/engine/test_tournament.py`: 8-model competitive rankings and stacking.
- `tests/unit/engine/test_tuner.py`: Optuna Bayesian sweeps and pruning.
- `tests/unit/engine/test_threshold.py`: Cost-matrix boundary optimization.
- `tests/unit/engine/test_explainer.py`: Sub-10s TreeSHAP attributions.
- `tests/unit/engine/test_ood_detector.py`: Mahalanobis distance & Isolation Forest OOD detection.
- `tests/unit/engine/test_onnx_optimizer.py`: ONNX FP32/FP16 conversion and round-trip fidelity.
- `tests/unit/engine/test_dashboard_generator.py`: Standalone HTML/SVG generation.
- `tests/integration/test_full_pipeline.py`: Complete multi-engine workflow test.
- `tests/integration/test_smoke_script.py`: Automated assertion of the 15-second smoke test script.

---

## 🐳 Production Deployment with Docker & FastAPI

`ml-mcp` automatically generates production deployment artifacts:

```bash
# Generate serving bundle via MCP tool or python
python -c "
from ml_mcp.engine.api_generator import FastAPIRouterGenerator
from ml_mcp.engine.docker_generator import DockerSpecGenerator

FastAPIRouterGenerator().generate('models/champion.joblib', ['feat_1', 'feat_2'], 'serving')
DockerSpecGenerator().generate('serving')
"

# Build and run the non-root container
cd serving
docker compose up --build
```

Endpoints provided:
- `GET /health`: Model status, memory footprint, and latency metrics.
- `POST /predict`: Input validation via Pydantic DTOs and sub-millisecond inference.

---

## 📁 Repository Structure

```
ml-mcp/
├── docs/
│   ├── ARCHITECTURE.md          # 3-tier clean architecture & topology codex
│   ├── TOOL_REFERENCE.md        # Comprehensive 25-tool reference guide
│   ├── COLAB_INTEGRATION.md     # Google Colab Cloud GPU acceleration guide
│   └── QUICKSTART_GUIDE.md      # 5-minute practical tutorial
├── scripts/
│   └── smoke_test.py            # 15-second 1-click end-to-end smoke test
├── src/
│   └── ml_mcp/
│       ├── __init__.py
│       ├── config.py            # Environment & GPU configurations
│       ├── server.py            # FastMCP Server initialization
│       ├── tools.py             # 25 FastMCP `@mcp.tool()` registrations
│       ├── engine/              # Deterministic Machine Learning Engines
│       │   ├── api_generator.py
│       │   ├── auditor.py
│       │   ├── balancer.py
│       │   ├── batch_predictor.py
│       │   ├── calibrator.py
│       │   ├── checkpoint_manager.py
│       │   ├── colab_bridge.py
│       │   ├── colab_generator.py
│       │   ├── collinearity.py
│       │   ├── dashboard_generator.py
│       │   ├── docker_generator.py
│       │   ├── drift_monitor.py
│       │   ├── explainer.py
│       │   ├── exporter.py
│       │   ├── fairness_auditor.py
│       │   ├── feature_synthesizer.py
│       │   ├── gpu_manager.py
│       │   ├── json_sanitizer.py
│       │   ├── leakage.py
│       │   ├── model_card.py
│       │   ├── onnx_optimizer.py
│       │   ├── pipeline_builder.py
│       │   ├── pseudo_labeler.py
│       │   ├── sentinel_hunter.py
│       │   ├── stacking_engine.py
│       │   ├── stress_tester.py
│       │   ├── target_transformer.py
│       │   ├── text_handler.py
│       │   ├── threshold.py
│       │   ├── tournament.py
│       │   └── tuner.py
│       └── schemas/             # Pydantic v2 DTOs (Strict extra="forbid")
│           ├── audit.py
│           ├── base.py
│           ├── colab.py
│           ├── safety.py
│           ├── serving.py
│           ├── tournament.py
│           └── tuning.py
├── tests/
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── test_smoke_script.py
│   └── unit/
│       ├── test_config.py
│       ├── test_json_sanitizer.py
│       ├── test_server.py
│       └── engine/              # 16 unit test suites for all engines
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📄 License & Authors

- **License:** Distributed under the [MIT License](https://opensource.org/licenses/MIT).
- **Author & Maintainer:** [sadik004](https://github.com/sadik004)
- **Engineered for:** Pair programming with Antigravity IDE, Claude Code, and Google Colab.
