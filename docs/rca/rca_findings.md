# Root Cause Analysis (RCA) Report: Audit Findings C1–C8

## Executive Summary
This document provides complete, mathematically rigorous Root Cause Analyses (RCA) and remediation documentation for findings C1 through C8 discovered during the audit of `ml-mcp`. In accordance with the Anti-Fraud and Mathematical Rigor Directives ("জিরো ফ্রড ইঞ্জিনিয়ারিং"), all hardcoded heuristics, synthetic statistical guarantees, silent exception masking, and data leakage channels have been completely eliminated.

---

## RCA C1: Fake Conformal Prediction Coverage Guarantees
- **Classification**: Critical Fraud Risk (The Sin of Fake Statistical Guarantees)
- **Root Cause**:
  `ml_conformal_risk_control` and `ml_certify_safety_and_decisions` previously returned hardcoded coverage figures (e.g. `empirical_coverage = 0.950` or `realized_coverage = 0.95`, `singleton_percentage = 82.0%`) rather than calculating realized empirical coverage on an untouched holdout validation split.
- **Catastrophic Impact**:
  Presenting false statistical guarantees creates catastrophic real-world failures (e.g., misdiagnosed patients or undetected fraudulent transactions) when live models fail to achieve nominal coverage.
- **Mathematical Remediation**:
  1. Conformal prediction sets are constructed using true disjoint calibration sets ($D_{\text{cal}}$) and evaluated on untouched validation holdouts ($D_{\text{val}}$).
  2. Non-conformity scores $s_i = 1 - \hat{p}(y_i \mid x_i)$ are calibrated using finite-sample quantile adjustment:
     $$\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
  3. Realized coverage is computed empirically strictly as:
     $$\text{empirical\_coverage} = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \mathbb{I}[y_i \in C(x_i)]$$
  4. Tested via `test_guard_a_conformal_no_fake_guarantees` and `test_conformal_coverage_disjoint_split`.

---

## RCA C2: Metric Impersonation & False ECE Calculation
- **Classification**: High Malpractice Risk (The Sin of Metric Impersonation)
- **Root Cause**:
  In several decision certification tools, Brier score ($\frac{1}{N} \sum (\hat{p}_i - y_i)^2$) was computed and passed directly into fields labeled as Expected Calibration Error (`ece`), falsely implying probability calibration.
- **Catastrophic Impact**:
  Downstream risk engines rely on ECE to select decision thresholds $p^*$. Conflating Brier score with ECE leads to incorrect cutoff deployment, inducing false-positive dollar cascades.
- **Mathematical Remediation**:
  1. Strict mathematical separation of Brier score, Negative Log-Likelihood (NLL), and ECE.
  2. ECE is calculated strictly through equal-frequency or adaptive reliability binning ($M=10$ or adaptive):
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  3. Tested via `test_adaptive_ece_elimination_of_sample_bias` and regression guards.

---

## RCA C3: Out-of-Distribution (OOD) Covariance Scale Invariance & Instability
- **Classification**: High Numerical Instability
- **Root Cause**:
  The Mahalanobis distance calculation $(x - \mu)^T \Sigma^{-1} (x - \mu)$ operated directly on unstandardized raw feature matrices without condition checking. If features had differing scales or were collinear, $\Sigma$ became singular, triggering runtime crashes or unbounded distances.
- **Catastrophic Impact**:
  OOD detection returned false positives or crashed entirely in production on dirty tabular inputs.
- **Mathematical Remediation**:
  1. Added mandatory internal `StandardScaler` normalization before computing the empirical covariance matrix $\Sigma$.
  2. Applied Ledoit-Wolf or Tikhonov diagonal shrinkage $\Sigma_{\text{reg}} = \Sigma + 10^{-5} I$.
  3. Used SVD pseudo-inverse fallback (`np.linalg.pinv`) if Cholesky / matrix inversion condition number exceeds $10^6$.
  4. Tested via `test_guard_b_ood_tabular_scale_invariance_and_stability` and `test_ood_mahalanobis_scale_invariance`.

---

## RCA C4: Data Leakage in Tuning & Feature Preparation
- **Classification**: High Overfitting Risk (The Overfitting Mirage)
- **Root Cause**:
  Feature transformers (imputers, target encoders, synthesizers, and selectors) were fitted on entire datasets prior to cross-validation splits, leaking out-of-fold target and distribution statistics into training folds.
- **Catastrophic Impact**:
  Validation metrics inflated unrealistically; deployed models collapsed when evaluated on real live data.
- **Mathematical Remediation**:
  1. Enforced strict encapsulation of preprocessors, imbalanced samplers, and selectors inside Scikit-Learn `Pipeline` / `ColumnTransformer` constructs.
  2. In Bayesian tuning (`tuner.py`), all preprocessing steps are fit exclusively within the inner training folds of `KFold`/`StratifiedKFold`.
  3. Out-of-fold cross-validation is mandatory for feature pruning (`feature_pruner.py`).
  4. Tested via `test_guard_d_preprocessor_zero_leakage` and `test_feature_pipeline_zero_leakage`.

---

## RCA C5: Zombie Models & Silent Exception Masking
- **Classification**: Enterprise Defect (Silent Exception Swallowing)
- **Root Cause**:
  `tournament.py` and `stacking_engine.py` caught blanket `Exception`s and silently masked failures with placeholder scores (`0.0`, `-0.5`) or dummy ensemble models with hardcoded weights (`0.85`).
- **Catastrophic Impact**:
  Broken pipelines silently returned un-trainable models; bugs went undetected until production scoring crashed.
- **Mathematical Remediation**:
  1. Blanket `except: pass` eliminated across the entire codebase (verified by Gate G1 AST checker).
  2. Excluded models register transparently in structured `warnings: List[str]` audit trails within the returned DTOs.
  3. Ensembles are validated as fully fitted estimators before serialization.
  4. Tested via `test_guard_f_tournament_reports_warnings_on_failures` and `scripts/check_no_silent_except.py`.

---

## RCA C6: Target Leakage Evaluation Crashes Masked as Clean Data
- **Classification**: Data Integrity Malpractice
- **Root Cause**:
  When non-linear correlation estimators (Chatterjee $\xi$, Cramér's V, or PPS) threw exceptions on edge-case data, the leakage detector caught the error and reported `has_critical_leakage = False`.
- **Catastrophic Impact**:
  Severely leaked features bypassed the preflight audit, contaminating the entire ML development lifecycle.
- **Mathematical Remediation**:
  1. Calculation failures are explicitly recorded as warnings and flagged as unverified.
  2. If a leakage check fails, the audit status reflects inconclusive verification rather than a clean pass.
  3. Tested via `test_guard_e_leakage_crashed_evaluation_distinguishable`.

---

## RCA C7: Ephemeral Training & Certification Without OOF Probabilities
- **Classification**: Architectural Transparency Defect
- **Root Cause**:
  `ml_certify_safety_and_decisions` previously attempted to silently train a temporary model or evaluate calibration metrics on training data if out-of-fold probabilities (`oof_probs`) were not provided.
- **Catastrophic Impact**:
  Gives clients the illusion that a certified safety audit was performed when in reality in-sample overfitted probabilities were evaluated.
- **Mathematical Remediation**:
  1. By default, `ml_certify_safety_and_decisions` **REFUSES** certification if `oof_probs` is missing.
  2. `train_ephemeral=False` is strictly enforced as the default. Without `model_path` or explicit `train_ephemeral=True`, certification fails clearly.
  3. When `train_ephemeral=True` is explicitly opted in, the certificate records:
     `training_mode = "ephemeral"` and `certification_status = "limited"`.
  4. In-sample diagnostic metrics require explicit opt-in via `allow_in_sample_diagnostic=True` and are explicitly tagged with `in_sample=True`.
  5. Tested via `test_safety_orchestrator_refuses_without_oof_probs` and `test_in_sample_leakage_rejected_as_certified`.

---

## RCA C8: Error Envelope Standard & Diagnostic Consistency
- **Classification**: API Robustness & Type Safety
- **Root Cause**:
  Errors previously resulted in unstructured strings or unhandled Python tracebacks flooding the LLM context window.
- **Catastrophic Impact**:
  Agent loops exhausted token budgets on repeated invalid tool arguments without actionable error recovery guidance.
- **Mathematical Remediation**:
  1. Created unified `format_error_envelope` with difflib fuzzy matching for misspelled columns and parameter names.
  2. Fully typed with Pydantic DTOs and strict Mypy compliance (0 issues across 54 source files).
  3. Tested via `test_error_envelope.py`.
