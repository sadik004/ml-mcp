# Baseline Quality Gate Report (Phase P0: Truth Baseline)

**Date:** 2026-10-04  
**Hardware & Environment:** Live Google Colab Cloud Session (NVIDIA Tesla T4 GPU, 14.56 GB VRAM, 12.67 GB System RAM, Linux 6.6.122+, Python 3.13.15, Pytest 8.4.2)  
**Execution Runner:** `scripts/colab_gate.py` executing `scripts/quality_gate.py`  
**Execution Timestamp:** 2026-10-03 18:16:00 UTC  
**Machine Exit Code:** `7` (3 / 10 Gates Passed — 30.0%)

---

## 1. Executive Summary & Honest Starting Score

In strict adherence to the **"জিরো ফ্রড ইঞ্জিনিয়ারিং" (Zero Fraud Engineering)** and **Radical Anti-Sycophancy ("জিরো ত্যালবাজি" পলিসি)** governance directives, this document records the uncompromised, machine-checked baseline score for `ml-mcp` across all 10 enterprise quality gates.

No human overrides, manual assertions, or synthetic scores are recognized. The codebase enters Phase P1 with an honest starting score of **3 / 10 Gates Passed**.

| Gate | Gate Name | Status | Machine Evidence |
| :--- | :--- | :---: | :--- |
| **G1** | Zero Silent Handlers Gate | **✅ PASS** | PASSED: 0 silent exception handlers found in `src/ml_mcp`. |
| **G2** | No Hardcoded Metrics / Zero Guarantee Slack | **❌ FAIL** | Found 4 violations: `exporter.py:35` (`score: 1.0`), `serving_service.py:132` (`Score: 0.95`, `Latency: 0.8`), `safety_service.py:408` (literal slack `+ 0.05`). |
| **G3** | Honest Out-Of-Sample Metrics | **❌ FAIL** | `test_out_of_sample.py` not implemented yet (scheduled P1). |
| **G4** | Visible Fallbacks in Warnings | **❌ FAIL** | `test_fallback_warnings.py` not implemented yet (scheduled P1). |
| **G5** | Fitted Estimators Round-Trip | **❌ FAIL** | `test_artifact_roundtrip.py` not implemented yet (scheduled P1). |
| **G6** | Statistical Guarantees Honesty & Validity | **❌ FAIL** | `test_coverage_validity.py` not implemented yet (scheduled P6). |
| **G7** | Test Suite Green & High Coverage | **✅ PASS** | 187 passed, 36 warnings in 117.52s on Tesla T4 GPU. |
| **G8** | Ruff Linter & Strict Mypy Type Safety | **✅ PASS** | Ruff: exit 0; Mypy: exit 0 (71 source files checked). |
| **G9** | 3-Tier Clean Architecture Invariants | **❌ FAIL** | 8 direct file I/O calls in `src/ml_mcp/services/` (`pd.read_csv`, `joblib.load/dump`); 75 hardcoded `random_state=42` literals. |
| **G10** | Test-Backed Documentation Integrity | **❌ FAIL** | ROADMAP guarantees not yet tagged with `[test: path::name]` (scheduled P7). |

---

## 2. Gate Breakdown & Audit Diagnostics

### Gate G1: Zero Silent Handlers (Status: PASS)
- **Scanner**: AST parser scanning every `try...except` block in `src/ml_mcp/`.
- **Invariant**: Any empty `except: pass`, `except Exception: pass`, or `except: continue` without structured logging/warning is an immediate gate failure.
- **Finding**: 0 violations found.

### Gate G2: No Hardcoded Metrics / Zero Guarantee Slack (Status: FAIL)
- **Scanner**: AST dictionary/return scanner + comparison checker.
- **Detected Violations**:
  1. `src/ml_mcp/services/exporter.py:35`: Hardcoded metric `'score': 1.0` in mock/stub export dictionary.
  2. `src/ml_mcp/services/serving_service.py:132`: Hardcoded default metric `'Score': 0.95`.
  3. `src/ml_mcp/services/serving_service.py:132`: Hardcoded default metric `'Latency': 0.8`.
  4. `src/ml_mcp/services/safety_service.py:408`: Guarantee comparison has literal slack `realized_coverage + 0.05 >= target_coverage`.
- **Impact**: Violates Cardinal Sin #1 and #2. Slacks and default metric placeholders create illusion of certified accuracy without empirical measurement.

### Gate G3: Honest Out-Of-Sample Metrics (Status: FAIL)
- **Invariant**: Metric-reporting tools (`ml_benchmark_models`, `ml_calibrate_probabilities`, `ml_tune_threshold_and_errors`) must evaluate strictly on held-out / out-of-fold data. If labels are randomly permuted, reported AUC must collapse to ~0.50 (within statistical tolerance), not reflect in-sample memorization.
- **Remediation**: Scheduled for implementation in Phase P1 (`tests/verification/test_out_of_sample.py`) and resolution in Phase P3.

### Gate G4: Visible Fallbacks in Warnings (Status: FAIL)
- **Invariant**: If an optional library (`lightgbm`, `xgboost`, `catboost`) fails or a fallback model is used, the response dictionary must contain a structured `warnings: list[str]` explicitly naming the fallback and cause.
- **Remediation**: Scheduled for implementation in Phase P1 (`tests/verification/test_fallback_warnings.py`) and resolution in Phase P4.

### Gate G5: Fitted Estimators Round-Trip (Status: FAIL)
- **Invariant**: All ensemble models and pipeline estimators returned or saved must be fully fitted Scikit-Learn estimators capable of `joblib.dump` -> `joblib.load` -> `predict_proba` round-trip without `NotFittedError`.
- **Remediation**: Scheduled for implementation in Phase P1 (`tests/verification/test_artifact_roundtrip.py`) and resolution in Phase P5.

### Gate G6: Statistical Guarantees Honesty & Validity (Status: FAIL)
- **Invariant**: Conformal prediction sets must report Wilson score binomial confidence intervals, enforce $n \ge 100$ minimum calibration size, and maintain statistical validity across seeds.
- **Remediation**: Scheduled for Phase P6 (`tests/verification/test_coverage_validity.py`).

### Gate G7: Test Suite Green & High Coverage (Status: PASS)
- **Pytest Output**: 187 passed, 36 warnings in 117.52s on Google Colab NVIDIA Tesla T4 GPU.
- **Note**: Re-measurement of strict line coverage will be enforced after P1–P6 code refactorings.

### Gate G8: Ruff Linter & Strict Mypy Type Safety (Status: PASS)
- **Ruff**: Exit code 0 (all lint checks clean).
- **Mypy**: Exit code 0 (`Success: no issues found in 71 source files`).

### Gate G9: 3-Tier Clean Architecture Invariants (Status: FAIL)
- **Scanner**: AST inspection for raw disk I/O in `services/` and hardcoded seeds across `src/ml_mcp/`.
- **Detected Violations**:
  - `serving_service.py`: contains raw `pd.read_csv` and `joblib` calls.
  - `feature_service.py`: contains raw `pd.read_csv`.
  - `model_service.py`: contains raw `pd.read_csv` and `joblib` calls.
  - `safety_service.py`: contains raw `pd.read_csv` and `joblib` calls.
  - `audit_service.py`: contains raw `pd.read_csv`.
  - Found 75 occurrences of hardcoded `'random_state=42'` literals (must route through central `Settings.seed`).
- **Remediation**: Scheduled for Phase P5.

### Gate G10: Test-Backed Documentation Integrity (Status: FAIL)
- **Scanner**: ROADMAP parser asserting that every performance, safety, and metric claim is anchored to an automated test via `[test: path::test_name]`.
- **Remediation**: Scheduled for Phase P7.

---

## 3. Phase P1 RED Fraud Verification Results

**Colab Execution Date:** 2026-10-04 (18:35 UTC)  
**Total Tests Executed:** 249 tests (230 PASSED, 19 FAILED)  
**Machine Exit Code:** `8` (2 / 10 Gates Passed: G1, G8)

### Summary of RED Fraud Guards (19/19 Failing as Expected)
Per the mandatory **RED-Before-GREEN** rule, all fraud detection guards must fail against the unfixed codebase:

| Fraud Test Suite | Target Gate | Tests | Failing (RED) | Passing | Root Cause Verified |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `tests/verification/test_out_of_sample.py` | **G3** | 6 | **6** | 0 | In-sample memorization on permuted noise (F-beta 0.86, Accuracy 1.0, Score 1.0, missing OOF evaluation). |
| `tests/verification/test_no_fake_metrics.py` | **G2** | 4 | **4** | 0 | Hardcoded `Score: 0.95`, `+0.05` CRC slack, `0.0` undefined FNR, and `f1, f2, f3` dummy schemas. |
| `tests/verification/test_fallback_warnings.py` | **G4** | 6 | **6** | 0 | 6 fallback sites substitute `HistGradientBoostingClassifier` without surfacing warnings to user. |
| `tests/verification/test_artifact_roundtrip.py` | **G5** | 5 | **3** | 2 | `ml_create_ensemble` produces no persisted artifact; calibrator has no persistence; raw estimators exported without pipelines. |

### Characterization Contract Suite (41/41 GREEN)
- **Suite**: `tests/characterization/test_tool_contracts.py`
- **Result**: **41 passed in 19.81s** on Colab Tesla T4 GPU.
- **Artifacts**: Snapshots saved in `tests/characterization/snapshots/` locking public input/output schemas for all 41 tools across routers before P5 refactoring.

---

## 4. Immediate Next Step

Proceed directly to **Phase P2: Fake Metrics Elimination (`P2_fake_metrics.md`)**:
- Eliminate `{"Score": 0.95, "Latency": 0.8}` from `serving_service.py:132`.
- Eliminate `+ 0.05` literal slack from `safety_service.py:408`.
- Enforce undefined FNR returns `None` with warning in `safety_service.py:393`.
- Eliminate placeholder `["f1", "f2", "f3"]` in `serving_service.py:149`.
- Turn **Gate G2 GREEN**!

