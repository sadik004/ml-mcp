# Enterprise Architecture & System Topology: `ml-mcp`

> **Document Version:** 1.0.0  
> **System Classification:** Model Context Protocol (MCP) Server for Machine Learning  
> **Governance:** Strict 3-Tier Clean Architecture, Zero-Leakage Pipeline, Token-Shield Defenses

---

## 1. System Overview & Mission

`ml-mcp` is an enterprise-grade, standalone Machine Learning Model Context Protocol server. It provides LLM agents (running in Antigravity IDE, Claude Code, Cursor, or OpenCode) and human engineers with a unified, defensive interface to perform end-to-end data science and competitive Kaggle-grade machine learning workflows.

The system is designed for **Dual-Topology Execution**:
1. **Local Mode:** Stdio-based communication inside IDEs for rapid data auditing, pipeline synthesis, and lightweight tree/linear model training.
2. **Colab Cloud GPU Mode:** Coupled natively with `googlecolab/colab-mcp` or secure SSE tunnels, utilizing NVIDIA GPUs (T4, L4, A100) for large-scale Optuna tuning, LightGBM/XGBoost/CatBoost tournaments, and Stacking ensembles.

---

## 2. 3-Tier Clean Architecture

`ml-mcp` strictly enforces Clean Architecture and Separation of Concerns across all modules:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MCP PROTOCOL CLIENT                             │
│         (Claude Desktop / Antigravity IDE / Cursor / OpenCode)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ FastMCP JSON-RPC (Stdio / SSE)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 1: TOOLS & ROUTERS LAYER                         │
│                      (src/ml_mcp/tools.py)                             │
│  - 25 FastMCP `@mcp.tool()` entry points                              │
│  - Token Shield: view="compact" | "detailed" token budgeting           │
│  - Defensive Immutability & Parameter Validation                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Typed Requests / Parameters
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 2: DETERMINISTIC ENGINE LAYER                   │
│                     (src/ml_mcp/engine/*.py)                           │
│  - Pure Domain Logic & Mathematical Formulations                       │
│  - Zero HTTP or MCP Protocol Dependencies                              │
│  - Memory-Safe, Exception-Bounded, GPU-Aware Orchestration             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pure DTO Instantiations
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  TIER 3: DATA CONTRACTS & SCHEMAS                      │
│                    (src/ml_mcp/schemas/*.py)                           │
│  - Pydantic v2 DTOs with strict extra="forbid"                         │
│  - Strict Type Hinting & Value Range Validation                        │
│  - Recursive JSON Sanitizer (NumPy / Pandas -> Native JSON)            │
└────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

1. **Routers / Tools Layer (`src/ml_mcp/tools.py`, `src/ml_mcp/server.py`):**
   - Receives JSON-RPC tool calls from LLM clients.
   - Enforces token budgeting via the **Token Shield** (`view="compact"` returns $\le 15$ key metrics; `view="detailed"` returns full distributions).
   - Sanitizes all returned dictionaries and DTOs using `sanitize_for_json` before serializing back to the client.

2. **Engine Layer (`src/ml_mcp/engine/`):**
   - Contains zero MCP-specific code. All engine classes and functions are purely deterministic, testable Python functions operating on Pandas DataFrames, NumPy arrays, Scikit-Learn estimators, and PyTorch/ONNX runtimes.
   - Modules include:
     - `auditor.py`, `sentinel_hunter.py`, `leakage.py`, `collinearity.py`, `text_handler.py` (Data Hygiene).
     - `pipeline_builder.py`, `feature_synthesizer.py`, `target_transformer.py`, `balancer.py` (Defensive Preprocessing).
     - `tournament.py`, `stacking_engine.py`, `gpu_manager.py`, `checkpoint_manager.py` (Competitive Arena).
     - `tuner.py`, `calibrator.py` (Bayesian Search & Calibration).
     - `threshold.py`, `explainer.py` (Decision Theory & Sub-10s TreeSHAP).
     - `ood_detector.py`, `stress_tester.py`, `fairness_auditor.py` (AI Safety).
     - `onnx_optimizer.py`, `batch_predictor.py`, `pseudo_labeler.py`, `model_card.py`, `exporter.py` (Serving & Inference).
     - `colab_generator.py`, `dashboard_generator.py`, `drift_monitor.py`, `api_generator.py`, `docker_generator.py` (Delivery & Monitoring).

3. **Data Contracts Layer (`src/ml_mcp/schemas/`):**
   - All tool responses are guaranteed to validate against immutable Pydantic v2 models.
   - Sub-modules: `base.py`, `audit.py`, `tournament.py`, `tuning.py`, `safety.py`, `serving.py`, `colab.py`.

---

## 3. Core Architectural Invariants

### Invariant 1: Zero Data Leakage
Preprocessors, transformers, and imputers are **never** fit on validation or test folds.
- Cross-validation splits in `tournament.py` use `StratifiedKFold` (for classification) or `KFold` (for regression).
- In every fold, the pipeline is fit strictly on `train_idx` and evaluated on `val_idx`.
- Target encoders and scaling statistics are never computed across the entire dataset before splitting.

### Invariant 2: Omnipresent Defensive Imputation
All generated Scikit-Learn pipelines inject an initial median/mode imputer step (`SimpleImputer(strategy="median")` for numeric features, and `SimpleImputer(strategy="most_frequent")` for categorical features). Even if a training dataset contains zero null values, the imputer remains in the pipeline to prevent catastrophic `NaN` runtime exceptions when deployed in production inference.

### Invariant 3: Recursive JSON Sanitization
FastMCP tool outputs must serialize to standard JSON. Scientific Python objects (`numpy.float32`, `numpy.int64`, `numpy.ndarray`, `pandas.Series`, `pandas.NA`, `pd.Timestamp`, `np.nan`, `np.inf`) break standard JSON encoders.
- The `sanitize_for_json` utility traverses nested dictionaries, tuples, lists, and dataclasses up to depth 30.
- Converts `NaN`, `+Inf`, and `-Inf` to `None` (JSON `null`).
- Converts all NumPy scalars to native Python `float`, `int`, and `bool`.
- Converts date/time objects to ISO 8601 strings.

### Invariant 4: Token Shield Architecture
LLMs possess finite context windows. Dumping full distribution tables or raw predictions into tool responses exhausts context and inflates inference latency.
- Every read-heavy or report-heavy tool accepts a `view: Literal["compact", "detailed"] = "compact"` argument.
- `compact` view provides high-signal summaries: top 5 features, overall accuracy/F1/ROC-AUC, top 3 tournament champions, and binary health statuses.
- `detailed` view supplies full per-fold CV matrices, raw correlation grids, complete classification reports, and quantile bins.

### Invariant 5: Accuracy Paradox Protection
In heavily imbalanced datasets (e.g., 99% Class 0, 1% Class 1), a naive classifier predicting all zeros achieves 99% accuracy.
- `ml-mcp` automatically detects class imbalance during pre-flight audits.
- The tournament engine defaults to Macro-F1, ROC-AUC, or PR-AUC as the selection metric for imbalanced classification tasks, preventing naive majority-class collapse.

---

## 4. Execution Topology: Local vs Colab GPU

```
┌────────────────────────────────────────────────────────┐
│                   LOCAL ENVIRONMENT                    │
│             (Windows / Linux / macOS IDE)              │
│                                                        │
│  - fastmcp stdio server                                │
│  - Rapid data audit & VIF check (< 1s)                 │
│  - Fast local decision tree / linear models            │
│  - Standalone HTML Dashboard visualization             │
│  - ONNX runtime batch prediction                       │
└───────────────────────────┬────────────────────────────┘
                            │
               Optional Remote Tunnel (SSE / SSH)
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                GOOGLE COLAB CLOUD GPU                  │
│             (NVIDIA T4 / L4 / A100 VRAM)               │
│                                                        │
│  - Automatic device="cuda" detection                   │
│  - 8-Model Tournament (LightGBM, XGBoost, CatBoost)    │
│  - Optuna 100-Trial Bayesian Hyperparameter Search     │
│  - Out-of-Fold Stacking Regressor / Classifier         │
│  - Sub-10s TreeSHAP explainability                     │
│  - Automated Google Drive Checkpoint Persistence       │
│    (/content/drive/MyDrive/ml_mcp/checkpoints/)        │
└────────────────────────────────────────────────────────┘
```

---

## 5. Security & Isolation

- **Non-Root Execution:** The generated Docker runtime enforces `USER appuser` with a non-privileged UID (10001).
- **Directory Traversal Protection:** All file paths passed to export and checkpoint tools are validated against allowed workspace boundaries using `os.path.abspath` resolution.
- **Resource Constraints:** Batch inference incorporates chunked processing (`chunk_size=10000`) to guarantee that memory consumption remains strictly bounded regardless of dataset dimensions.
