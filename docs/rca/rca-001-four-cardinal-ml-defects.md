# RCA-001: Root Cause Analysis on Four Critical ML Governance Defects

| Metadata | Details |
| :--- | :--- |
| **Incident ID** | RCA-001 |
| **Severity** | CRITICAL (Fiduciary, Safety & Capital Risk) |
| **Date** | 2026-10-03 |
| **Status** | RESOLVED & CODIFIED |
| **Resolution Commit** | `1a06ad4` |
| **Author** | Antigravity AI (Reviewed by Lead Architect) |

---

## 1. Executive Summary
During the unification of the Phase 1–4 Master Orchestrators for `ml-mcp`, four severe algorithmic and architectural defects were identified. If deployed to production in a banking, fintech, or clinical setting, these defects would have resulted in multi-million dollar capital loss, miscalibrated thresholding, and catastrophic false-negative security escapes.

This document details the root causes, catastrophic real-world failure modes, mathematical corrections applied, and the permanent ethical and governance directives added to the global agent codex.

---

## 2. Detailed Defect Analysis & Root Causes

### Defect 1: Hardcoded Conformal Risk Control Guarantees
- **What happened:** In `SafetyOrchestrator.certify_model()`, the variables `coverage_pct` and `singleton_pct` were hardcoded to `95.0` and `94.0` in the certificate output, completely discarding the empirical calculations.
- **Root Cause:** Rushing to produce green unit tests and attractive markdown receipts without wiring the empirical holdout verification pass.
- **Catastrophic Failure Mode:** In automated fraud detection, leadership believes 95% of fraudulent events are mathematically bounded within the prediction set. In reality, actual empirical coverage may have fallen to 60%, silently letting 40% of fraudulent transactions pass without triage.
- **Correction Applied:** Implemented split-conformal validation. Probabilities are partitioned into calibration (50%) and test (50%) splits. $\hat{\lambda}$ is calibrated on the calibration split, and actual realized coverage $\frac{1}{N} \sum \mathbb{I}(y_i \in C(x_i))$ and singleton ratios are computed strictly on the untouched test partition.

---

### Defect 2: Metric Impersonation (Brier Score Disguised as ECE)
- **What happened:** In `SafetyOrchestrator`, `cal_rep.raw_brier_score` was assigned directly to `raw_ece`.
- **Root Cause:** Conceptual carelessness during rapid integration; conflating distinct probability scoring metrics under the assumption that both measure calibration errors.
- **Catastrophic Failure Mode:** Brier score is a strictly proper scoring rule combining calibration and refinement. Expected Calibration Error (ECE) specifically measures calibration divergence across probability bins. Reporting Brier as ECE misleads decision-threshold engines into optimizing cutoffs on false confidence bounds, producing severe false-positive or false-negative cascades in loan underwriting.
- **Correction Applied:** Explicitly integrated `calculate_ece(y_arr, probs, n_bins=10)` (Naeini et al. AAAI 2015; Guo et al. ICML 2017) to compute authentic 10-bin calibration error for both uncalibrated and Beta/Platt calibrated probabilities.

---

### Defect 3: In-Sample Hyperparameter Optimization & Selection Bias
- **What happened:** Hyperparameter tuning was performed over cross-validation folds, and the validation score from those same folds was reported as the champion's generalization performance.
- **Root Cause:** Standard competitive ML shortcuts (single-loop CV) that overlook hyperparameter optimization selection bias.
- **Catastrophic Failure Mode:** Cawley & Talbot (JMLR 2010) and Varma & Simon (2006) demonstrate that tuning hyperparameters on the evaluation loop leaks information and produces optimistic performance estimates. A model claiming 98% PR-AUC in testing drops to 80% upon live traffic deployment.
- **Correction Applied:** Enforced **Nested Cross-Validation** ($K$-outer folds for generalization evaluation $\times$ $k$-inner folds for Optuna Bayesian search). Outer folds remain completely unseen during tuning.

---

### Defect 4: Unfitted Ensemble Dictionaries & Silent Exception Swallowing
- **What happened:** The Stacking champion returned an ad-hoc dictionary of estimators where base models were not fit on the full dataset, and multiple components wrapped runtime errors in `except Exception: pass` returning arbitrary dummy numbers (`-0.5`, `0.0`).
- **Root Cause:** "Fault tolerance" misapplied to safety-critical algorithms. Swallowing exceptions was used to prevent test crashes instead of properly handling domain edge cases.
- **Catastrophic Failure Mode:** Calling `.predict()` on the serialized model object in production throws an `AttributeError` or scores inputs with untrained weights. Fallback defaults hide failures from site reliability engineering (SRE) monitoring.
- **Correction Applied:** Used official `sklearn.ensemble.StackingClassifier` / `StackingRegressor` fitted end-to-end on all training data before serialization. Replaced silent exception blocks with an explicit user-facing `warnings: List[str]` audit trail in all DTO reports.

---

## 3. Permanent Codification in Agent Harness & Rules
To ensure these defects are never repeated, the following binding laws were added to the global agent codex:
- `C:\Users\User\.gemini\config\AGENTS.md` (Section 6: ML Mathematical Rigor)
- `C:\Users\User\.gemini\config\rules\clean-architecture.md` (Section 6)
- `C:\Users\User\.gemini\config\rules\ml-production-safety.md` (Dedicated Governance Directive)
- `C:\Users\User\.gemini\config\skills\fastapi-production\SKILL.md` (The 4 Cardinal Sins)

---

## 4. Verification & Quality Gates
- **Unit Test Suite:** 136 tests passed (100% passing rate).
- **Execution Time:** 45.56s.
- **Git Commit:** `1a06ad4`.
