# RCA-v1: Root Cause Analysis on Manual Quality Gate Deficiencies and v2 Remediation

| Document Metadata | Value |
| :--- | :--- |
| **Document ID** | RCA-V1-MANUAL-GATES |
| **Severity** | CRITICAL (Governance & Statistical Integrity Failure) |
| **Scope** | Failure modes of v1 manual verification gates vs automated v2 machine gates |
| **Author** | Antigravity AI Engineering Team |
| **Status** | RESOLVED & CODIFIED |

---

## 1. Executive Summary

In Phase v1 of `ml-mcp`, quality gates relied on manual checklist reviews and standard unit test coverage passes. While test suites reported 100% green status, post-audit forensic review revealed four cardinal machine learning malpractice defects and multiple documentation regressions:
1. **Hardcoded Conformal Metric (0.95)**: Synthetic statistical guarantees presented as empirical reality.
2. **Conformal Risk Control Slack (`+ 0.05`)**: Invisible tolerance additions masking guarantee violations.
3. **In-Sample Metric Impersonation**: Generalization metrics computed on training samples or unseparated folds.
4. **Silent Exception Swallowing & Zombie Defaults**: `except Exception: pass` returning fake zeroes or arbitrary constants.
5. **Stale Documentation Claims**: Unbacked literature citations framed as verified operational guarantees.

This document investigates why v1 manual verification failed to detect these defects and formalizes how the v2 Automated Quality Gate architecture (`scripts/quality_gate.py`, Gates G1–G10) permanently prevents recurrence.

---

## 2. Root Cause Analysis by Defect

### 2.1 Defect 1: The 0.95 Hardcoded Conformal Metric Miss
- **v1 Failure Mode**: The v1 test suite asserted `assert "coverage_pct" in report` and `assert report["coverage_pct"] >= 0.90`. Because the underlying engine hardcoded `coverage_pct = 95.0`, the assertion passed trivially with 0% awareness of real data distributions.
- **Why Manual Gate Missed It**: Human manual review focused on interface compliance (the existence of output keys and green tests) rather than behavioral perturbation or adversarial data tests.
- **v2 Machine Prevention**:
  - **Gate G2 (AST Metric Scanner)**: Parses the AST of all files in `src/ml_mcp/`, rejecting any dictionary with keys like `coverage`, `ece`, `auc`, or `accuracy` mapped to literal float constants.
  - **Gate G6 (Multi-Seed Empirical Validity)**: Executes `test_multi_seed_conformal_coverage_validity` across 200 random seeds on untouched holdouts, asserting that empirical coverage tracks $\ge 1 - \alpha$ within finite-sample Wilson confidence bounds $[L_{\text{wilson}}, U_{\text{wilson}}]$.

---

### 2.2 Defect 2: Conformal Risk Control Slack Miss (`+ 0.05` Tolerance)
- **v1 Failure Mode**: In `conformal_risk_control.py`, a comparison `if realized_loss <= target_loss + 0.05:` was introduced to reduce test flakiness on small datasets. This +5% slack converted theoretical risk control into an uncalibrated heuristic.
- **Why Manual Gate Missed It**: The manual reviewer assumed the addition was an acceptable numerical engineering tolerance rather than a mathematical violation of the risk control bound.
- **v2 Machine Prevention**:
  - **Gate G2 (AST Guarantee Slack Scanner)**: Scans every `ast.Compare` node in `src/ml_mcp/`. Any comparison against a guarantee bound containing an additive constant (`+ 0.05`, `+ 0.01`) triggers an immediate gate failure.
  - **Verification Test**: `tests/verification/test_no_fake_metrics.py::test_crc_zero_slack_on_violated_guarantee` tests adversarial datasets where realized loss is $0.12$ against target $0.10$, proving that `guarantee_satisfied` is strictly `False`.

---

### 2.3 Defect 3: In-Sample Metric Impersonation
- **v1 Failure Mode**: In safety and calibration tools, when holdout predictions were not supplied, engines computed ECE, Brier score, and accuracy directly on the training set without informing the caller or flagging data leakage.
- **Why Manual Gate Missed It**: In-sample metrics produce higher, cleaner numbers that visually resemble well-performing models during manual demonstration runs.
- **v2 Machine Prevention**:
  - **Gate G3 (Permuted Label Out-Of-Sample Gate)**: Executes `tests/verification/test_out_of_sample.py` with shuffled targets ($y_{\text{perm}}$). Because random permutations destroy all predictive signal, genuine out-of-sample evaluation yields near-zero accuracy/AUC and high loss. Any tool returning high accuracy on permuted data is mathematically exposed as leaking training data.
  - **Explicit Invariant**: Unfitted models or missing holdouts must use cross-validated out-of-fold scoring (`oof_predict`, `oof_predict_proba`), never training-set resubstitution.

---

### 2.4 Defect 4: Silent Fallbacks & Exception Swallowing
- **v1 Failure Mode**: Optional libraries (LightGBM, XGBoost, CatBoost) were handled via bare `try...except Exception: return -0.5` or `except: pass`, substituting `HistGradientBoostingClassifier` without audit evidence.
- **Why Manual Gate Missed It**: Manual testing ran in environments where some optional packages were installed, masking that missing dependencies caused silent degradation.
- **v2 Machine Prevention**:
  - **Gate G1 (AST 0-Silent-Handler Gate)**: Scans every `ast.ExceptHandler` in `src/ml_mcp/`. Every catch block must either log an error, raise an exception, or record a structured message in a warnings list.
  - **Gate G4 (Fallback Warnings Audit)**: `tests/verification/test_fallback_warnings.py` mocks missing packages and asserts that every surrogate fallback registers an explicit, user-facing notice in `warnings: List[str]`.

---

### 2.5 Defect 5: Stale Documentation Claims
- **v1 Failure Mode**: `ROADMAP.md` and `README.md` listed extensive research papers and theoretical capabilities without verifying whether each claim had corresponding test assertions in the test suite.
- **Why Manual Gate Missed It**: Documentation and code were treated as separate workstreams; markdown files were authored as marketing materials rather than verified contracts.
- **v2 Machine Prevention**:
  - **Gate G10 (Test-Backed Documentation Integrity)**: Parses every guarantee statement in `ROADMAP.md`, verifies the presence of `[test: path::test_name]` tags, and confirms that every tagged test file and test function exists in the workspace.
  - **Literature Reclassification**: Pure academic citations not directly verified by a test assertion are strictly categorized under `Scientific & Algorithmic Foundations (Literature References)`.

---

## 3. The 10 Automated Machine Gates (v2 Matrix)

| Gate | Target Defect / Invariant | Enforcement Mechanism | Status |
| :--- | :--- | :--- | :---: |
| **G1** | Zero silent exception handlers | AST analysis of all try/except blocks (`scripts/check_no_silent_except.py`) | ✅ PASS |
| **G2** | No hardcoded metrics or guarantee slacks | AST scanner for literal metric dicts and `+ 0.05` comparisons | ✅ PASS |
| **G3** | Honest out-of-sample evaluation | Permuted-label tests across all 6 core services (`test_out_of_sample.py`) | ✅ PASS |
| **G4** | Transparent fallback warnings | Estimator resolver audit and warnings assertions (`test_fallback_warnings.py`) | ✅ PASS |
| **G5** | Real fitted estimator round-trips | Serialization and deserialization round-trip tests (`test_artifact_roundtrip.py`) | ✅ PASS |
| **G6** | Statistical guarantees honesty | Wilson CI intervals, min-n sample bounds, 200-seed simulation (`test_coverage_validity.py`) | ✅ PASS |
| **G7** | Test suite 100% green & coverage $\ge 85\%$ | Full automated pytest execution on Colab GPU | ✅ PASS |
| **G8** | Ruff linter & strict Mypy typing | Ruff check and Mypy strict type verification across all source files | ✅ PASS |
| **G9** | 3-tier clean architecture & seed hygiene | Protocol repository verification and zero literal `random_state=42` literals | ✅ PASS |
| **G10** | Test-backed documentation integrity | Machine parser verifying `[test: ...]` tags in `ROADMAP.md` | ✅ PASS |

---

## 4. Conclusion & Governance Directive

Manual review is inherently susceptible to confirmation bias, fatigue, and interface superficiality. True fiduciary machine learning engineering demands automated, adversarial, AST-level quality gates that execute automatically on remote compute hardware before any release.
