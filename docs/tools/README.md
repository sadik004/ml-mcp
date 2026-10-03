# ML-MCP Deep-Dive Tool Guides

Comprehensive architectural guides, human histories, mathematical foundations, parameters, and internal engine mechanics for ml-mcp tools.

---

## Available Deep-Dive Guides

| Tool | Category | Theoretical SOTA Reference | Guide Link |
| :--- | :--- | :--- | :--- |
| **ml_audit_dataset** | Phase 1: Data Audit & Hygiene | Dirac-Delta Sentinels & Hubert Medcouple | [ml_audit_dataset.md](ml_audit_dataset.md) |
| **ml_detect_target_leakage** | Phase 1: Data Audit & Hygiene | Chatterjee (JASA 2021) & Cramér's V (2023) | [ml_detect_target_leakage.md](ml_detect_target_leakage.md) |
| **ml_check_collinearity** | Phase 1: Data Audit & Hygiene | SVD Condition Number & Belsley Decomposition | [ml_check_collinearity.md](ml_check_collinearity.md) |
| **ml_detect_label_errors** | Phase 1: Data Audit & Hygiene | Northcutt et al. (Confident Learning) | [ml_detect_label_errors.md](ml_detect_label_errors.md) |
| **ml_verify_constraints** | Phase 1: Data Audit & Hygiene | Schema Domain Boundary Checking | [ml_verify_constraints.md](ml_verify_constraints.md) |
| **ml_track_lineage** | Data Provenance & Tracking | SHA-256 Provenance & Experiment Tracking | [ml_track_lineage.md](ml_track_lineage.md) |
| **ml_auto_clean_and_pipe** | Phase 2: Feature Engineering | Pipeline Imputation & Target Encoding | [ml_auto_clean_and_pipe.md](ml_auto_clean_and_pipe.md) |
| **ml_handle_text_features** | Phase 2: Feature Engineering | TF-IDF & TruncatedSVD NLP Reduction | [ml_handle_text_features.md](ml_handle_text_features.md) |
| **ml_balance_classes** | Phase 2: Resampling & Balance | BorderlineSMOTE & Adaptive Resampling | [ml_balance_classes.md](ml_balance_classes.md) |
| **ml_synthesize_features** | Phase 2: Feature Synthesis | High-Order Polynomial Crosses | [ml_synthesize_features.md](ml_synthesize_features.md) |
| **ml_prune_features** | Phase 2: Feature Selection | Mutual Information & Lasso Stability | [ml_prune_features.md](ml_prune_features.md) |
| **ml_transform_target** | Phase 2: Target Transformation | Yeo-Johnson & Box-Cox Normality | [ml_transform_target.md](ml_transform_target.md) |
| **ml_benchmark_models** | Phase 3: Model Tournament | 5-Fold Stratified Multi-Model Sweep | [ml_benchmark_models.md](ml_benchmark_models.md) |
| **ml_tune_hyperparameters** | Phase 3: Bayesian Optimization | Optuna Tree-Structured Parzen Estimator | [ml_tune_hyperparameters.md](ml_tune_hyperparameters.md) |
| **ml_create_ensemble** | Phase 3: Meta-Learning | Ridge Meta-Learner Stacking Classifier | [ml_create_ensemble.md](ml_create_ensemble.md) |
| **ml_conformal_risk_control**| Phase 3: Safe Prediction Sets | Mondrian Conformal & RAPS (NeurIPS 2021) | [ml_conformal_risk_control.md](ml_conformal_risk_control.md) |
| **ml_calibrate_probabilities**| Phase 3: Probability Calibration | 3-Parameter Beta Calibration & Adaptive ECE | [ml_calibrate_probabilities.md](ml_calibrate_probabilities.md) |
| **ml_tune_threshold_and_errors**| Phase 3: Decision Thresholding| Vickers DCA Net Benefit & Expected Cost | [ml_tune_threshold_and_errors.md](ml_tune_threshold_and_errors.md) |
| **ml_explain_predictions** | Phase 3: Explainable AI (XAI) | Lundberg TreeSHAP & Attribution | [ml_explain_predictions.md](ml_explain_predictions.md) |
| **ml_detect_ood** | Phase 3: Out-of-Distribution | Helmholtz Free Energy (NeurIPS 2020) & Mahalanobis | [ml_detect_ood.md](ml_detect_ood.md) |
| **ml_stress_test_and_fairness**| Phase 3: Stress & Fairness | Covariance Manifold Noise & Subgroup Disparity | [ml_stress_test_and_fairness.md](ml_stress_test_and_fairness.md) |
| **ml_pseudo_label_loop** | Phase 4: Production Serving & Loop | FlexMatch (NeurIPS 2021) & Conformal Singletons | [ml_pseudo_label_loop.md](ml_pseudo_label_loop.md) |
| **ml_optimize_inference** | Phase 4: Production Serving & Loop | ONNX Runtime Level-3 Hardware Fusion (IEEE 2022) | [ml_optimize_inference.md](ml_optimize_inference.md) |
| **ml_batch_predict** | Phase 4: Production Serving & Loop | VLDB 2022 Streaming & Calibrated DCA Cutoffs | [ml_batch_predict.md](ml_batch_predict.md) |
| **ml_generate_serving_api** | Phase 4: Production Serving & Loop | Modern ASGI Lifespan & Cloud-Native CNCF Probes | [ml_generate_serving_api.md](ml_generate_serving_api.md) |
| **ml_generate_docker_spec** | Phase 4: Production Serving & Loop | CIS Docker Benchmark v1.6 & Non-Root appuser | [ml_generate_docker_spec.md](ml_generate_docker_spec.md) |
| **ml_generate_eval_dashboard**| Phase 4: Production Serving & Loop | Pure SVG Decision Curve & Reliability Vectors | [ml_generate_eval_dashboard.md](ml_generate_eval_dashboard.md) |
| **ml_monitor_drift** | Phase 4: Production Serving & Loop | Population Stability Index & Wasserstein-1 Drift | [ml_monitor_drift.md](ml_monitor_drift.md) |

---

*These documents are automatically updated and audited to reflect the real Python engine implementations.*
