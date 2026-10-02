# FastMCP Tool Reference Guide: `ml-mcp`

> **Total Production Tools:** 25  
> **Server Implementation:** `src/ml_mcp/server.py` & `src/ml_mcp/tools.py`  
> **Communication Protocol:** Model Context Protocol (MCP) JSON-RPC 2.0 (Stdio & SSE)

> **Phased Architectural Curriculum:** Explore the full 5-phase lifecycle in [docs/phases/](phases/README.md).
>
> **Deep-Dive Architectural Guides:** Detailed human stories, mathematical models, and internal mechanics are documented in [docs/tools/](tools/README.md):
> - [ml_benchmark_models](tools/ml_benchmark_models.md)
> - [ml_detect_target_leakage](tools/ml_detect_target_leakage.md)
> - [ml_track_lineage](tools/ml_track_lineage.md)
> - [ml_pseudo_label_loop](tools/ml_pseudo_label_loop.md)

---

## Tool Category Index

1. [Core & Environment](#1-core--environment) (`ml_ping`)
2. [Data Hygiene & Pre-flight Auditing](#2-data-hygiene--pre-flight-auditing) (`ml_audit_dataset`, `ml_detect_leakage`, `ml_check_collinearity`)
3. [Defensive Preprocessing & Engineering](#3-defensive-preprocessing--engineering) (`ml_build_pipeline`, `ml_transform_target`, `ml_balance_data`, `ml_synthesize_features`)
4. [Competitive Arena & Stacking](#4-competitive-arena--stacking) (`ml_run_tournament`, `ml_train_stacking`)
5. [Optimization & Calibration](#5-optimization--calibration) (`ml_tune_hyperparameters`, `ml_calibrate_probabilities`)
6. [Decision Theory & Explainability](#6-decision-theory--explainability) (`ml_optimize_threshold`, `ml_explain_shap`)
7. [AI Safety & Robustness](#7-ai-safety--robustness) (`ml_detect_ood`, `ml_stress_test`, `ml_audit_fairness`)
8. [Inference Speed & Serving](#8-inference-speed--serving) (`ml_export_onnx`, `ml_batch_predict`, `ml_pseudo_label`, `ml_generate_model_card`)
9. [Delivery & Continuous Monitoring](#9-delivery--continuous-monitoring) (`ml_export_colab_notebook`, `ml_generate_dashboard`, `ml_monitor_drift`, `ml_generate_serving_bundle`)

---

## 1. Core & Environment

### `ml_ping`
Verifies server health, active Python environment, runtime platform, and GPU acceleration status.
- **Parameters:** None
- **Returns:**
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "cuda_available": false,
    "device": "cpu",
    "gpu_count": 0,
    "platform": "Windows-11-...",
    "python_version": "3.14.0",
    "registered_tools_count": 25
  }
  ```

---

## 2. Data Hygiene & Pre-flight Auditing

### `ml_audit_dataset`
Performs a comprehensive pre-flight sanity audit on a tabular CSV dataset.
- **Parameters:**
  - `csv_path` (`str`, required): Absolute or relative path to CSV file.
  - `target_column` (`str`, optional): Name of the prediction target column.
  - `view` (`"compact" | "detailed"`, default: `"compact"`): Token shield view mode.
- **Outputs Checked:**
  - Sentinel values (`-999`, `-1`, `9999`, `"?"`, `"N/A"`, `"missing"`).
  - Missing value percentages and near-empty columns ($\ge 90\%$ null).
  - Constant features (zero variance) and high-cardinality ID columns ($\ge 95\%$ unique).
  - Row duplicates and class imbalance ratios.

### `ml_detect_leakage`
Audits features for target leakage and unrealistically high correlation with the target.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Target label or dependent variable.
  - `correlation_threshold` (`float`, default: `0.95`): Pearson/Spearman cutoff for suspicious features.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Key Warnings:** Flags perfect predictors (AUC = 1.0 or $R^2 = 1.0$) and future-dated timestamp columns.

### `ml_check_collinearity`
Calculates Variance Inflation Factor (VIF) and Pearson/Spearman correlation matrices.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `vif_threshold` (`float`, default: `10.0`): Cutoff for multi-collinear features.
  - `correlation_cutoff` (`float`, default: `0.90`): Maximum allowed pairwise feature correlation.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Recommendation:** Generates a list of redundant features to drop without losing model capacity.

---

## 3. Defensive Preprocessing & Engineering

### `ml_build_pipeline`
Constructs a zero-leakage, reproducible Scikit-Learn `ColumnTransformer` and `Pipeline`.
- **Parameters:**
  - `numeric_columns` (`list[str]`, required): List of continuous numeric features.
  - `categorical_columns` (`list[str]`, required): List of discrete categorical features.
  - `scaler` (`"robust" | "standard" | "minmax"`, default: `"robust"`): Numeric scaling method.
  - `cat_imputer` (`"most_frequent" | "constant"`, default: `"most_frequent"`): Categorical imputer.
  - `num_imputer` (`"median" | "mean"`, default: `"median"`): Numeric imputer.
- **Safeguard:** Injects omnipresent imputers to handle future unobserved production NaNs.

### `ml_transform_target`
Linearizes highly skewed continuous targets using Log1p or Box-Cox transformations.
- **Parameters:**
  - `values` (`list[float]`, required): Raw target vector.
  - `method` (`"auto" | "log1p" | "box-cox"`, default: `"auto"`): Transformation technique.
- **Outputs:** Transformed values, optimal lambda parameter, and inverse-transform function handles.

### `ml_balance_data`
Mitigates extreme classification class imbalance using synthetic resampling techniques.
- **Parameters:**
  - `csv_path` (`str`, required): Training dataset.
  - `target_column` (`str`, required): Binary or multi-class target column.
  - `method` (`"smote" | "borderline" | "random"`, default: `"smote"`): Balancing strategy.
  - `sampling_strategy` (`float | str`, default: `"auto"`): Desired minority-to-majority ratio.

### `ml_synthesize_features`
Generates interaction terms, polynomial combinations, and domain feature ratios.
- **Parameters:**
  - `csv_path` (`str`, required): Input CSV.
  - `feature_pairs` (`list[tuple[str, str]]`, optional): Specific feature interaction pairs.
  - `include_polynomials` (`bool`, default: `false`): Include degree-2 polynomial expansion.

---

## 4. Competitive Arena & Stacking

### `ml_run_tournament`
Launches an 8-model competitive tournament across LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Gradient Boosting, Ridge/Logistic Regression, and Multi-Layer Perceptron (MLP).
- **Parameters:**
  - `csv_path` (`str`, required): Tabular dataset.
  - `target_column` (`str`, required): Prediction target.
  - `task_type` (`"classification" | "regression"`, default: `"classification"`)
  - `metric` (`str`, default: `"f1"`): Evaluation metric (`"f1"`, `"roc_auc"`, `"accuracy"`, `"rmse"`, `"r2"`).
  - `cv_folds` (`int`, default: `5`): Stratified K-Fold cross-validation count.
  - `device` (`"auto" | "cuda" | "cpu"`, default: `"auto"`): Hardware accelerator.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Returns:** Leaderboard ranking, champion model name, per-model scores, and standard deviations.

### `ml_train_stacking`
Builds an Out-of-Fold (OOF) Stacking Ensemble using the top $K$ tournament models as base estimators and Logistic Regression / Ridge as the meta-learner.
- **Parameters:**
  - `csv_path` (`str`, required): Tabular dataset.
  - `target_column` (`str`, required): Target column.
  - `top_k` (`int`, default: `3`): Number of base estimators to ensemble.
  - `cv_folds` (`int`, default: `5`): Cross-validation folds for OOF predictions.
- **Guarantee:** Strict leak-free cross-validation guarantees meta-features do not overfit.

---

## 5. Optimization & Calibration

### `ml_tune_hyperparameters`
Executes Optuna Bayesian Optimization over tree depth, learning rate, regularization, and subsampling.
- **Parameters:**
  - `model_name` (`str`, required): Name of estimator (`"lightgbm"`, `"xgboost"`, `"random_forest"`).
  - `csv_path` (`str`, required): Training dataset.
  - `target_column` (`str`, required): Target column.
  - `n_trials` (`int`, default: `30`): Optuna trials count.
  - `timeout_seconds` (`int`, default: `300`): Hard execution time boundary.
- **Pruning:** Integrates Median Pruner to abort unpromising trials within early epochs.

### `ml_calibrate_probabilities`
Calibrates model output probabilities to match true empirical likelihoods.
- **Parameters:**
  - `model_path` (`str`, required): Path to pickled/joblib model.
  - `csv_path` (`str`, required): Validation dataset.
  - `target_column` (`str`, required): Target labels.
  - `method` (`"platt" | "isotonic"`, default: `"platt"`): Calibration algorithm.
- **Outputs:** Brier score reduction, Expected Calibration Error (ECE), and calibration curve bins.

---

## 6. Decision Theory & Explainability

### `ml_optimize_threshold`
Finds the mathematically optimal classification decision boundary given an asymmetric business cost matrix.
- **Parameters:**
  - `probabilities` (`list[float]`, required): Predicted positive class probabilities.
  - `y_true` (`list[int]`, required): Ground truth binary labels.
  - `cost_fp` (`float`, default: `1.0`): Financial cost of False Positive.
  - `cost_fn` (`float`, default: `5.0`): Financial cost of False Negative.
  - `metric` (`"cost" | "f1" | "youden"`, default: `"f1"`): Optimization objective.
- **Returns:** Optimal decision threshold (e.g., `0.342` instead of default `0.500`), net business savings, confusion matrix at optimal threshold.

### `ml_explain_shap`
Generates sub-10 second TreeSHAP local and global attributions for tree models.
- **Parameters:**
  - `model_path` (`str`, required): Fitted estimator path.
  - `csv_path` (`str`, required): Sample dataset for explanation.
  - `sample_size` (`int`, default: `100`): Background sample size for fast calculation.
  - `top_k` (`int`, default: `10`): Number of top influential features to report.
- **Fallback:** Uses kernel explainer or permutation importance if model is non-tree.

---

## 7. AI Safety & Robustness

### `ml_detect_ood`
Identifies Out-of-Distribution (OOD) test inputs using Mahalanobis Distance and Isolation Forest.
- **Parameters:**
  - `train_csv_path` (`str`, required): In-distribution training reference data.
  - `test_csv_path` (`str`, required): Query test observations to audit.
  - `contamination` (`float`, default: `0.05`): Expected anomalous fraction.
- **Output:** Anomaly scores, binary OOD flags, and feature drift attribution.

### `ml_stress_test`
Subject models to simulated real-world data corruption: Gaussian noise and Missing Completely At Random (MCAR).
- **Parameters:**
  - `model_path` (`str`, required): Saved model artifact.
  - `csv_path` (`str`, required): Clean test dataset.
  - `target_column` (`str`, required): Target labels.
  - `noise_levels` (`list[float]`, default: `[0.01, 0.05, 0.10]`): Noise variances.
  - `missing_rates` (`list[float]`, default: `[0.05, 0.10, 0.20]`): Fraction of masked values.
- **Score:** Comprehensive Robustness Score ($0 - 100$).

### `ml_audit_fairness`
Audits sub-group demographic parity, disparate impact ratios, and equal opportunity margins.
- **Parameters:**
  - `csv_path` (`str`, required): Evaluation dataset with predictions.
  - `sensitive_column` (`str`, required): Protected demographic attribute (gender, age, race).
  - `target_column` (`str`, required): Ground truth label.
  - `prediction_column` (`str`, required): Binary predicted class.
- **Governance:** Checks EEOC 80% (4/5ths) disparate impact rule compliance.

---

## 8. Inference Speed & Serving

### `ml_export_onnx`
Converts Scikit-Learn pipelines and GBDT models into high-performance Open Neural Network Exchange (ONNX) format.
- **Parameters:**
  - `model_path` (`str`, required): Input `.joblib` model.
  - `output_path` (`str`, required): Target `.onnx` output path.
  - `target_dtype` (`"float32" | "float16"`, default: `"float32"`): Floating-point precision.
- **Verification:** Automatically performs round-trip inference comparison against original estimator.

### `ml_batch_predict`
Executes memory-safe, chunked batch predictions for massive datasets with Kaggle submission validation.
- **Parameters:**
  - `model_path` (`str`, required): Model file (`.joblib` or `.onnx`).
  - `test_csv_path` (`str`, required): Unlabeled test CSV.
  - `output_csv_path` (`str`, required): Destination CSV for predictions.
  - `id_column` (`str`, optional): ID column to preserve for Kaggle submissions.
  - `chunk_size` (`int`, default: `10000`): Chunk size for memory-bounded processing.

### `ml_pseudo_label`
Extracts high-confidence predictions on unlabeled data to enrich the training set.
- **Parameters:**
  - `model_path` (`str`, required): Trained model.
  - `unlabeled_csv_path` (`str`, required): Unlabeled observations.
  - `confidence_threshold` (`float`, default: `0.90`): Minimum probability cutoff.
  - `max_pseudo_labels` (`int`, default: `1000`): Maximum pseudo-labels to generate.

### `ml_generate_model_card`
Synthesizes a production-ready `MODEL_CARD.md` adhering to Mitchell et al. standards.
- **Parameters:**
  - `model_name` (`str`, required): Name of model.
  - `metrics` (`dict`, required): Performance dictionary (Accuracy, F1, ROC-AUC, Latency).
  - `intended_use` (`str`, optional): Production application context.
  - `output_path` (`str`, default: `"MODEL_CARD.md"`): Destination file path.

---

## 9. Delivery & Continuous Monitoring

### `ml_export_colab_notebook`
Generates a complete, executable 8-cell Jupyter `.ipynb` notebook ready to upload directly to Google Colab.
- **Parameters:**
  - `target_column` (`str`, required): Target feature.
  - `output_path` (`str`, default: `"ml_pipeline.ipynb"`): Output notebook path.
  - `task_type` (`"classification" | "regression"`, default: `"classification"`)
  - `dataset_url` (`str`, optional): Remote CSV URL (e.g. Kaggle / raw GitHub link).

### `ml_generate_dashboard`
Compiles an interactive, standalone HTML dashboard containing embedded SVG confusion matrices and SHAP bar charts.
- **Parameters:**
  - `model_name` (`str`, required): Champion model name.
  - `metrics` (`dict`, required): Model evaluation metrics.
  - `top_features` (`list[tuple[str, float]]`, required): Feature importances.
  - `output_path` (`str`, default: `"dashboard.html"`): Output file location.
- **Feature:** Zero external CDN dependencies; renders natively in any browser offline.

### `ml_monitor_drift`
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS-test) statistics between reference training distributions and current production inputs.
- **Parameters:**
  - `reference_csv_path` (`str`, required): Historical baseline data.
  - `current_csv_path` (`str`, required): Live incoming production data.
  - `psi_threshold` (`float`, default: `0.2`): PSI cutoff indicating substantial drift.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Traffic Light System:** Green ($PSI < 0.1$), Yellow ($0.1 \le PSI < 0.2$), Red ($PSI \ge 0.2$, triggering retrain alarm).

### `ml_generate_serving_bundle`
Synthesizes a clean 3-tier FastAPI router (`app/routers/predict.py`) and a production multi-stage Docker build specification (`Dockerfile` & `docker-compose.yml`).
- **Parameters:**
  - `model_path` (`str`, required): Trained model artifact.
  - `features` (`list[str]`, required): Expected input features.
  - `output_dir` (`str`, default: `"serving_bundle"`): Destination directory.
- **Includes:** Pydantic input models, `/health`, `/predict`, non-root container user, and automated container healthcheck.
