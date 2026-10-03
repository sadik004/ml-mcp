# FastMCP Tool Reference Guide: `ml-mcp`

> **Total Production Tools:** 37  
> **Server Implementation:** `src/ml_mcp/server.py` & `src/ml_mcp/tools.py`  
> **Communication Protocol:** Model Context Protocol (MCP) JSON-RPC 2.0 (Stdio & SSE)

> **Phased Architectural Curriculum:** Explore the full 5-phase lifecycle in [docs/phases/](phases/README.md).
>
> **Deep-Dive Architectural Guides:** Detailed human stories, mathematical models, and internal mechanics are documented in [docs/tools/](tools/README.md).

---

## Tool Category Index

1. [Core & Environment](#1-core--environment) (`ml_ping`)
2. [Phase 1: Data Audit & Hygiene](#2-phase-1-data-audit--hygiene) (`ml_audit_dataset`, `ml_detect_target_leakage`, `ml_check_collinearity`, `ml_detect_label_errors`, `ml_verify_constraints`, `ml_track_lineage`)
3. [Phase 2: Defensive Feature Engineering](#3-phase-2-defensive-feature-engineering) (`ml_handle_text_features`, `ml_auto_clean_and_pipe`, `ml_balance_classes`, `ml_synthesize_features`, `ml_prune_features`, `ml_transform_target`)
4. [Phase 3: Tournament, Tuning & Stacking](#4-phase-3-tournament-tuning--stacking) (`ml_benchmark_models`, `ml_create_ensemble`, `ml_tune_hyperparameters`, `ml_pseudo_label_loop`)
5. [Phase 4: Calibration, Safety & Explainability](#5-phase-4-calibration-safety--explainability) (`ml_calibrate_probabilities`, `ml_tune_threshold_and_errors`, `ml_explain_predictions`, `ml_detect_ood`, `ml_stress_test_and_fairness`, `ml_conformal_risk_control`)
6. [Phase 5: Inference, Packaging & Operations](#6-phase-5-inference-packaging--operations) (`ml_batch_predict`, `ml_export_and_document`, `ml_optimize_inference`, `ml_generate_eval_dashboard`, `ml_generate_serving_api`, `ml_generate_docker_spec`, `ml_monitor_drift`)
7. [Phase 6: Remote Cloud & Google Colab GPU](#7-phase-6-remote-cloud--google-colab-gpu) (`ml_generate_colab_notebook`, `ml_colab_status`, `ml_colab_execute`, `ml_colab_upload`, `ml_colab_download`, `ml_colab_stop`, `ml_cancel_job`)

---

## 1. Core & Environment

### `ml_ping`
Verifies server health, active Python environment, runtime platform, and GPU acceleration status.
- **Parameters:** None
- **Returns:** JSON object containing status, python version, platform, and registered tools count.

---

## 2. Phase 1: Data Audit & Hygiene

### `ml_audit_dataset`
Performs a comprehensive pre-flight sanity audit on a tabular CSV dataset.
- **Parameters:**
  - `csv_path` (`str`, required): Absolute or relative path to CSV file.
  - `target_column` (`str`, optional): Name of the prediction target column.
  - `view` (`"compact" | "detailed"`, default: `"compact"`): Token shield view mode.
- **Outputs Checked:** Sentinel values, missing percentages, zero-variance constants, high-cardinality IDs, row duplicates, and class imbalance.

### `ml_detect_target_leakage`
Audits features for target leakage and unrealistically high correlation with the target.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Target label or dependent variable.
  - `correlation_threshold` (`float`, default: `0.95`): Pearson/Spearman cutoff for suspicious features.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Key Warnings:** Flags perfect predictors (AUC = 1.0 or R^2 = 1.0) and future-dated timestamp columns.

### `ml_check_collinearity`
Calculates Variance Inflation Factor (VIF) and Pearson/Spearman correlation matrices.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, optional): Target column used for ANOVA F-score competitive twin retention.
  - `vif_threshold` (`float`, default: `10.0`): Cutoff for multi-collinear features.
  - `correlation_cutoff` (`float`, default: `0.90`): Maximum allowed pairwise feature correlation.
  - `output_path` (`str`, optional): Path to save pruned dataset.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Recommendation:** Generates a list of redundant features to drop without losing model capacity.

### `ml_detect_label_errors`
Detects corrupt, mislabeled, or noisy ground truth labels using MIT Confident Learning (Northcutt et al., 2021).
- **Parameters:**
  - `csv_path` (`str`, required): Path to training dataset CSV.
  - `target_column` (`str`, required): Ground truth label column.
  - `cv_splits` (`int`, default: `5`): Out-of-fold cross-validation folds.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Outputs:** Confident joint matrix, estimated noise rate, self-confidence thresholds, and ranked suspicious sample indices.

### `ml_verify_constraints`
Validates physical limits, non-negativity, and automated 3x-IQR statistical outlier limits inspired by Amazon Deequ (VLDB 2018).
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `constraints` (`dict`, optional): User-specified constraints (e.g. min, max, allowed values, regex).
  - `view` (`"compact" | "detailed"`, default: `"compact"`)
- **Outputs:** Constraint violation counts, failure percentages, and schema integrity scorecard.

### `ml_track_lineage`
Cryptographic artifact and dataset lineage tracker computing SHA-256 digests and Git commit hashes.
- **Parameters:**
  - `job_id` (`str`, required): Tracking job identifier.
  - `data_path` (`str`, required): Path to dataset artifact.
  - `model_path` (`str`, optional): Path to model serialized checkpoint.
  - `parameters` (`dict`, optional): Execution hyperparameters.
  - `metrics` (`dict`, optional): Evaluation metric key-value pairs.
- **Outputs:** Immutable lineage JSON record with git_commit_sha, data_sha256, and timestamps.

---

## 3. Phase 2: Defensive Feature Engineering

### `ml_handle_text_features`
Detects and embeds free-form natural language text columns while rejecting random UUIDs and hashes.
- **Parameters:**
  - `csv_path` (`str`, required): Path to CSV dataset.
- **Outputs:** Text column detections, token statistics, and representation recommendations.

### `ml_auto_clean_and_pipe`
Constructs a zero-leakage, reproducible Scikit-Learn `ColumnTransformer` and preprocessing pipeline.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Prediction target.
  - `numeric_imputer` (`"median" | "mean" | "knn"`, default: `"median"`)
  - `categorical_imputer` (`"most_frequent" | "constant"`, default: `"most_frequent"`)
  - `scaler` (`"robust" | "standard" | "minmax"`, default: `"robust"`)
  - `output_dir` (`str`, optional): Directory to persist pickled pipeline.
- **Outputs:** Persisted pipeline.joblib, transformed sample preview, and column routing manifest.

### `ml_balance_classes`
Rebalances imbalanced classification datasets using SMOTE, ADASYN, Random Under Sampler, or Balanced Class Weights.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Classification target.
  - `strategy` (`"auto" | "smote" | "undersample" | "weights"`, default: `"auto"`)
  - `output_path` (`str`, optional): Path to save balanced dataset.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_synthesize_features`
Generates cyclical sin/cos features, safe ratios, and ExploreKit group aggregations with empirical Bayes $m$-estimate smoothing and cardinality guardrails.
- **Parameters:**
  - `csv_path` (`str`, required): Absolute or relative path to CSV dataset.
  - `time_column` (`str`, optional): Name of cyclical temporal column (e.g. `"hour"`, `"month"`).
  - `period` (`float`, default: `24.0`): Periodicity of the cycle (24 for hours, 7 for days, 12 for months).
  - `ratio_pairs` (`list[list[str]]`, optional): Feature pairs for safe ratio computation (`[["num_col", "den_col"]]`).
  - `group_specs` (`list[dict]`, optional): ExploreKit group-by aggregation specifications (`cat_col`, `num_col`, `aggregations`, etc.).
  - `max_cardinality` (`int`, default: `1000`): Maximum unique levels allowed for categorical grouping (with automatic 20% ratio guard when $N \ge 50$) to prevent singleton overfitting.
  - `smoothing` (`float`, default: `10.0`): CatBoost empirical Bayes $m$-estimate smoothing parameter for shrinking small-sample category means toward the global prior.
  - `output_path` (`str`, optional): Destination path to save enriched dataset (defaults to `processed/{name}_synthesized.csv`).
- **Outputs:** Transformed CSV path, generated features list, and transformation metadata.

### `ml_prune_features`
Prunes noisy and uninformative features using gradient-boosted out-of-fold (OOF) cross-validated permutation importance (OpenFE architecture, Breiman 2001, Hooker 2019).
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Prediction target column name.
  - `top_k` (`int`, optional): Retain at most top K features by importance rank.
  - `importance_threshold` (`float`, default: `0.005`): Minimum normalized permutation importance (0.5%) required to retain a feature.
  - `task_type` (`"auto" | "classification" | "regression"`, default: `"auto"`): Learning task type.
  - `cv_splits` (`int`, default: `3`): Number of cross-validation folds for out-of-fold permutation importance evaluation (with fallback for small datasets $N < 15$).
  - `output_path` (`str`, optional): Destination path to save pruned dataset (defaults to `processed/{name}_pruned.csv`).
- **Outputs:** Pruned CSV path, original feature count, pruned count, retained feature list, and normalized permutation importance scores.

### `ml_transform_target`
Applies defensive target transformations (Box-Cox, Yeo-Johnson, Log1p) for skewed regression targets.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Regression target column.
  - `method` (`"auto" | "log1p" | "box-cox" | "yeo-johnson"`, default: `"auto"`)
  - `output_path` (`str`, optional): Path to save transformed dataset.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

---

## 4. Phase 3: Tournament, Tuning & Stacking

### `ml_benchmark_models`
Runs an automated tournament across LightGBM, XGBoost, CatBoost, Random Forest, and Ridge/LogisticRegression with metric alignment, state isolation, and diversity-guarded stacking.
- **Parameters:**
  - `csv_path` (`str`, required): Path to preprocessed CSV.
  - `target_column` (`str`, required): Prediction target column name.
  - `task_type` (`"classification" | "regression"`, default: `"classification"`)
  - `cv_splits` (`int`, default: `5`): Cross-validation fold count.
  - `scoring` (`str`, optional): Target evaluation metric (e.g. "roc_auc", "f1", "f1_macro", "accuracy", "r2", "neg_root_mean_squared_error").
  - `group_column` (`str`, optional): Column name for group-aware StratifiedGroupKFold / GroupKFold cross-validation.
  - `fast_mode` (`bool`, default: `False`): Truncates tree counts (30 trees) for low-latency testing.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_create_ensemble`
Constructs multi-model Stacking and Voting ensembles with out-of-fold meta-learners.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset.
  - `target_column` (`str`, required): Prediction target.
  - `models` (`list[str]`, optional): Models to include in ensemble.
  - `meta_learner` (`str`, default: `"logistic"` / `"ridge"`): Meta-model.
  - `output_path` (`str`, optional): Path to persist ensemble checkpoint.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_tune_hyperparameters`
Conducts Bayesian optimization with Optuna TPE and active in-loop MedianPruner.
- **Parameters:**
  - `csv_path` (`str`, required): Path to dataset CSV.
  - `target_column` (`str`, required): Prediction target.
  - `model_name` (`str`, default: `"lightgbm"`): Model architecture to tune ("lightgbm", "xgboost", "catboost", "random_forest", "extra_trees").
  - `n_trials` (`int`, default: `10`): Number of Bayesian optimization trials.
  - `task_type` (`str`, default: `"classification"`): "classification" or "regression".
  - `metric` (`str`, optional): Target metric (defaults to "roc_auc" for classification, "r2" for regression).
  - `group_column` (`str`, optional): Column name for group-aware cross-validation.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_pseudo_label_loop`
Executes confidence-gated semi-supervised pseudo-labeling for unlabeled datasets.
- **Parameters:**
  - `labeled_csv_path` (`str`, required): Path to ground-truth training set.
  - `unlabeled_csv_path` (`str`, required): Path to unlabeled test set.
  - `target_column` (`str`, required): Prediction target column.
  - `confidence_threshold` (`float`, default: `0.90`): Minimum probability for pseudo-labeling.
  - `output_path` (`str`, optional): Path to save expanded dataset.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

---

## 5. Phase 4: Calibration, Safety & Explainability

### `ml_calibrate_probabilities`
Calibrates model output probabilities via Platt Scaling, Isotonic Regression, or Temperature Scaling with simplex normalization and adaptive ECE.
- **Parameters:**
  - `csv_path` (`str`, required): Evaluation dataset CSV.
  - `target_column` (`str`, required): Classification ground truth column.
  - `model_path` (`str`, optional): Path to persisted model checkpoint (.joblib).
  - `model_name` (`str`, default: `"lightgbm"`): Fallback model architecture if model_path not supplied.
  - `method` (`"isotonic" | "sigmoid" | "temperature"`, optional): Calibration technique.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_tune_threshold_and_errors`
Optimizes classification decision thresholds using PR-curve exact cutoffs and Sheng & Ling (2014) cost-sensitive loss matrix.
- **Parameters:**
  - `csv_path` (`str`, required): Validation dataset CSV.
  - `target_column` (`str`, required): Ground truth binary label.
  - `beta` (`float`, default: `1.0`): F-beta metric weight favoring recall.
  - `criterion` (`"f_beta" | "cost_loss"`, default: `"f_beta"`): Optimization goal.
  - `cost_fp` (`float`, default: `1.0`): Economic cost per False Positive.
  - `cost_fn` (`float`, default: `5.0`): Economic cost per False Negative.
  - `benefit_tp` (`float`, default: `0.0`): Economic benefit per True Positive.
  - `benefit_tn` (`float`, default: `0.0`): Economic benefit per True Negative.
  - `model_path` (`str`, optional): Path to persisted model checkpoint (.joblib).
  - `model_name` (`str`, default: `"lightgbm"`): Model architecture if model_path not supplied.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_explain_predictions`
Computes sub-10s TreeSHAP global feature attributions or Lundberg et al. (Nature MI 2020) local sample waterfall breakdowns.
- **Parameters:**
  - `csv_path` (`str`, required): Dataset CSV.
  - `target_column` (`str`, required): Target label.
  - `top_k` (`int`, default: `10`): Max features in attribution report.
  - `instance_index` (`int`, optional): Row index for single-sample local waterfall breakdown.
  - `model_path` (`str`, optional): Path to persisted model checkpoint (.joblib).
  - `model_name` (`str`, default: `"lightgbm"`): Model architecture if model_path not supplied.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_detect_ood`
Calculates Mahalanobis distance and Isolation Forest anomaly scores to detect Out-of-Distribution inputs.
- **Parameters:**
  - `reference_csv_path` (`str`, required): In-distribution training reference CSV.
  - `query_csv_path` (`str`, required): Query inference CSV to audit.
  - `threshold_percentile` (`float`, default: `95.0`): Anomaly detection percentile.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_stress_test_and_fairness`
Performs adversarial input perturbation (Gaussian noise, missing injection) and sub-group fairness slice audits.
- **Parameters:**
  - `csv_path` (`str`, required): Evaluation dataset CSV.
  - `target_column` (`str`, required): Ground truth label.
  - `model_path` (`str`, required): Model checkpoint.
  - `protected_attributes` (`list[str]`, optional): Demographics / fairness columns.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

### `ml_conformal_risk_control`
Provides rigorous finite-sample mathematical error-rate guarantees via Split Conformal Prediction and Learn-then-Test (Angelopoulos & Bates, 2021).
- **Parameters:**
  - `csv_path` (`str`, required): Calibration set CSV.
  - `target_column` (`str`, required): Ground truth target.
  - `model_path` (`str`, required): Model checkpoint.
  - `alpha` (`float`, default: `0.10`): Maximum tolerated risk bound (1 - alpha = 90% coverage).
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

---

## 6. Phase 5: Inference, Packaging & Operations

### `ml_batch_predict`
Executes high-throughput chunked batch inference on massive CSV datasets.
- **Parameters:**
  - `csv_path` (`str`, required): Test input CSV.
  - `model_path` (`str`, required): Model checkpoint.
  - `output_path` (`str`, optional): Path to save predictions CSV.
  - `chunk_size` (`int`, default: `10000`): Chunk size for OOM avoidance.

### `ml_export_and_document`
Generates comprehensive Hugging Face / Mitchell et al. Model Card markdown documentation.
- **Parameters:**
  - `model_path` (`str`, required): Model checkpoint.
  - `model_name` (`str`, required): Model name.
  - `metrics` (`dict`, required): Performance score dictionary.
  - `output_path` (`str`, optional): Path to write `MODEL_CARD.md`.

### `ml_optimize_inference`
Converts Scikit-Learn / LightGBM models into ONNX runtime graph format with fp16/int8 quantization.
- **Parameters:**
  - `model_path` (`str`, required): Scikit-Learn `.joblib` model.
  - `output_path` (`str`, optional): Target `.onnx` path.
  - `quantize` (`bool`, default: `False`): Apply dynamic int8 quantization.

### `ml_generate_eval_dashboard`
Generates a standalone, beautiful HTML/JavaScript evaluation dashboard with interactive charts.
- **Parameters:**
  - `metrics` (`dict`, required): Model evaluation metrics.
  - `confusion_matrix` (`list[list[int]]`, optional): Confusion matrix.
  - `feature_importances` (`dict[str, float]`, optional): Feature importances.
  - `output_path` (`str`, optional): Path to write `eval_dashboard.html`.

### `ml_generate_serving_api`
Scaffolds a production-ready FastAPI serving microservice with Pydantic request/response schemas.
- **Parameters:**
  - `model_path` (`str`, required): Path to model checkpoint.
  - `feature_names` (`list[str]`, required): Expected input features.
  - `output_dir` (`str`, optional): Target directory for API code.

### `ml_generate_docker_spec`
Generates minimal, non-root, multi-stage production Dockerfile and `docker-compose.yml` for serving.
- **Parameters:**
  - `app_dir` (`str`, required): Root directory of serving API.
  - `port` (`int`, default: `8000`): HTTP serving port.

### `ml_monitor_drift`
Audits data drift between baseline reference and production inference using Population Stability Index (PSI) and Wasserstein Distance.
- **Parameters:**
  - `baseline_csv_path` (`str`, required): Training baseline CSV.
  - `current_csv_path` (`str`, required): Current production inference CSV.
  - `psi_threshold` (`float`, default: `0.20`): PSI drift alert threshold.
  - `view` (`"compact" | "detailed"`, default: `"compact"`)

---

## 7. Phase 6: Remote Cloud & Google Colab GPU

### `ml_generate_colab_notebook`
Generates a complete, production-ready `.ipynb` Jupyter notebook for remote execution on Google Colab.
- **Parameters:**
  - `pipeline_steps` (`list[str]`, required): Selected pipeline stages.
  - `dataset_url` (`str`, optional): Direct download link or Google Drive link for dataset.
  - `output_path` (`str`, optional): Path to write notebook file.

### `ml_colab_status`
Checks remote Google Colab GPU health, Tesla T4 VRAM availability, and bridge connectivity.
- **Parameters:**
  - `session` (`str`, default: `"gpu"`): Colab session identifier.

### `ml_colab_execute`
Executes arbitrary Python and Machine Learning code directly on remote Google Colab GPU.
- **Parameters:**
  - `code` (`str`, required): Python script or bash command to run on Colab.
  - `session` (`str`, default: `"gpu"`): Colab session ID.
  - `timeout` (`int`, default: `120`): Execution timeout in seconds.

### `ml_colab_upload`
Uploads local CSV datasets, models, or scripts directly into Google Colab filesystem.
- **Parameters:**
  - `local_path` (`str`, required): Local file path.
  - `remote_path` (`str`, required): Target destination on Colab.
  - `session` (`str`, default: `"gpu"`)

### `ml_colab_download`
Downloads trained models, checkpoints, or metric reports from Colab to local workspace.
- **Parameters:**
  - `remote_path` (`str`, required): File path on Google Colab.
  - `local_path` (`str`, required): Local target file path.
  - `session` (`str`, default: `"gpu"`)

### `ml_colab_stop`
Terminates remote Google Colab GPU session to release cloud compute resources.
- **Parameters:**
  - `session` (`str`, default: `"gpu"`)

### `ml_cancel_job`
Cancels long-running background asynchronous ML jobs.
- **Parameters:**
  - `job_id` (`str`, required): Unique job ID to abort.
