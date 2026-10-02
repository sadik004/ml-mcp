# ML-MCP Phased Architecture & Topic Curriculum

An enterprise-grade, phased breakdown of the machine learning lifecycle implemented in `ml-mcp`. Each phase represents an architectural milestone containing specialized topic guides.

```mermaid
flowchart LR
    P1["Phase 1: Data Audit & Hygiene"] --> P2["Phase 2: Feature Engineering"]
    P2 --> P3["Phase 3: Model Training & Tuning"]
    P3 --> P4["Phase 4: Safety & Risk Validation"]
    P4 --> P5["Phase 5: Serving & MLOps"]
```

---

## Phased Structure & Topic Directory

### 🔍 [Phase 1: Data Audit & Hygiene](phase-01-data-audit-and-hygiene/)
> **Mission:** Pre-flight statistical auditing, cryptographic data integrity, and leakage prevention before any modeling begins.
- **Topic 1:** [ml_audit_dataset](phase-01-data-audit-and-hygiene/ml_audit_dataset.md) — Digital stethoscope, SentinelHunter & Accuracy Paradox Guard.
- **Topic 2:** [ml_detect_target_leakage](phase-01-data-audit-and-hygiene/ml_detect_target_leakage.md) — Pearson correlation & mutual information leakage detection.
- **Topic 3:** [ml_track_lineage](phase-01-data-audit-and-hygiene/ml_track_lineage.md) — SHA-256 dataset fingerprinting and Git-linked provenance metadata.

---

### ⚙️ [Phase 2: Feature Engineering & Preprocessing](phase-02-feature-engineering/)
> **Mission:** Transforming raw tabular data, NLP feature extraction, class balancing, and automated defensive pipelines.
- *Upcoming topics: `ml_auto_clean_and_pipe`, `ml_handle_text_features`, `ml_balance_classes`, `ml_synthesize_features`.*

---

### 🏆 [Phase 3: Model Training, Benchmarking & Refinement](phase-03-model-training-and-refinement/)
> **Mission:** Competitive arena cross-validation, hyperparameter tuning, cloud GPU execution, and semi-supervised refinement.
- **Topic 1:** [ml_benchmark_models](phase-03-model-training-and-refinement/ml_benchmark_models.md) — 8-model competitive arena, overfitting guards, latency profiling & stacking.
- **Topic 2:** [ml_pseudo_label_loop](phase-03-model-training-and-refinement/ml_pseudo_label_loop.md) — Harvesting >=98% confident unlabelled predictions to boost dataset size and accuracy.

---

### 🛡️ [Phase 4: Validation, Risk & AI Safety](phase-04-validation-and-safety/)
> **Mission:** Conformal risk control, probability calibration, decision thresholds, stress testing, and out-of-distribution detection.
- *Upcoming topics: `ml_conformal_risk_control`, `ml_calibrate_probabilities`, `ml_explain_predictions`, `ml_detect_ood`.*

---

### 🚀 [Phase 5: Production Serving & MLOps](phase-05-serving-and-mlops/)
> **Mission:** ONNX quantization, 3-tier clean architecture FastAPI endpoints, Dockerization, and real-time drift monitoring.
- *Upcoming topics: `ml_optimize_inference`, `ml_generate_serving_api`, `ml_generate_docker_spec`, `ml_monitor_drift`.*
