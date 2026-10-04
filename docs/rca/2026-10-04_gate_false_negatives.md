# RCA: Quality Gate False Negatives & The Illusion of 10/10 Certification

**Date:** 2026-10-04  
**Author:** Apprentice Backend / ML Systems Engineer  
**Status:** RESOLVED & PREVENTED  
**Scope:** Remediation v3 / Defect Isolation (C1–C6, H1–H9, T1–T11, G0–G10)

---

## 1. Symptom: The 10/10 Green Mirage
Prior to Remediation v3, `scripts/quality_gate.py` reported 10/10 gates PASSING (100% green exit code 0). However, deep adversarial auditing revealed that 6 Critical (C1–C6) and 9 High (H1–H9) architectural and mathematical defects were actively present in production code:
- **C1/C2 (Scoring Impersonation):** Brier score returned as Expected Calibration Error; stacking meta-learners fitted on training probabilities instead of out-of-fold meta-features.
- **C3/C4 (Calibrator Breakage):** Mismatched array lengths on small holdouts; string/object labels crash `LabelEncoder` at inference.
- **C5 (Data Leakage):** Full-data preprocessors fitted before cross-validation splits, inflating tournament champion scores.
- **C6 (Group Leakage):** Entity IDs leaking across cross-validation folds due to lack of `GroupKFold`.

Despite these severe defects, the quality gate reported `10/10 PASS [OK]`.

---

## 2. Root Cause Analysis (RCA)

### RCA-1: Substring Matching & Surface-Level Regex Checks
The original quality gate used simple substring and regex searches to verify invariants:
- A search for `"except Exception: pass"` only matched exact literal formatting. Any multiline try/except with a log statement or fallback passed undetected.
- Brier-to-ECE impersonation went undetected because the function returned a dictionary containing the key `"ece"`, regardless of what mathematical formula generated the value.

### RCA-2: Happy-Path Testing Without Adversarial Oracles
Tests in the test suite tested standard happy-path inputs (e.g., standard Iris or synthetic blobs with numeric columns only).
- When models were tested with shuffled/permuted labels, tests did not assert that AUC collapsed to ~0.50.
- When optional dependencies were mocked, tests checked only that no exception was thrown, rather than verifying that an explicit fallback was recorded in `warnings: List[str]`.

### RCA-3: Absence of Cryptographic Test Locks
Without test integrity gates, test assertions could be modified, weakened, or bypassed by automated agents during refactoring ("answer-faking" or assertion relaxation).

---

## 3. The Multi-Layered Remediation & Fix

### Layer 1: Cryptographic Anti-Gaming Gate (G0 & Rules T1–T11)
- Implemented `scripts/check_test_integrity.py` which uses Python's Abstract Syntax Tree (`ast`) to parse all test files and reject:
  - Any test file modification that changes the SHA-256 hash locked in `tests/verification/LOCK.json`.
  - Any mock overriding the function under test (Rule T4).
  - Any assertions with tautological truth (`assert True`, `assert 1 == 1`).
  - Missing Cloud Offloading telemetry verification (`--require-colab`).

### Layer 2: True Mathematical Oracles (Phase P2 & Rules T2, T3)
- Implemented independent mathematical reference implementations (e.g., `_reference_ece` with equal-width binning $\sum |B_m|/N |\text{acc} - \text{conf}|$) to verify production metrics against independent formulas.
- Dynamic Scikit-Learn metric dispatch in `src/ml_mcp/engine/scoring.py` guaranteeing exact metric definitions for 26 distinct classification and regression metrics.

### Layer 3: Label Permutation & Selection Bias Stress Tests (Gate G3)
- Implemented `tests/verification/test_out_of_sample.py`: under randomly permuted labels, training accuracy may be high, but out-of-fold and holdout metrics MUST collapse to chance level ($\text{AUC} \le 0.60, R^2 \le 0.05$).

### Layer 4: Strict Pipeline Sealing (Gate G5 & Phase P4)
- Wrapped all estimators in `Pipeline` with fold-level preprocessors (`build_sealed_pipeline`), ensuring that preprocessors (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) are fitted strictly within training folds with zero leakage onto holdout sets.

---

## 4. Prevention & Continuous Enforcement
1. **Gate G0 in CI/CD:** Every PR and test run runs `py scripts/check_test_integrity.py --require-colab` prior to any code evaluation.
2. **Mandatory Out-of-Sample Verification:** All reported scores expose `evaluation_mode` (`"out_of_fold"`, `"holdout"`, or `"in_sample"`) and `warnings`.
3. **Automated Quality Gate Scorecard:** All 11 gates must achieve exit code 0 (`py scripts/quality_gate.py --all`) on every release.
