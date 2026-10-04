# Baseline Test & Coverage Report (Phase 0)

**Date:** 2026-10-03  
**Environment:** Google Colab Cloud GPU/CPU Runtime (Linux 6.6.122+, Python 3.13.15, Pytest 8.4.2)  
**Execution Runtime:** 131.43s via `ml_colab_execute`

---

## 1. Summary Metrics

| Metric | Result |
| :--- | :--- |
| **Total Test Count** | 152 |
| **Passing Tests** | 143 |
| **Failing Tests** | 9 |
| **Skipped / Warning** | 23 warnings |
| **Total Code Coverage** | **71%** (Target: >= 85%) |
| **Engine Code Coverage** | **78.4%** |
| **Tools Monolith Coverage** | **18%** |

---

## 2. Baseline Failures Breakdown

1. `tests/integration/test_smoke_script.py::test_smoke_test_script_execution`  
   - Root Cause: Paths to test datasets relative to root.
2. `tests/unit/test_colab_bridge.py`:
   - `test_colab_cloud_runner_get_status_gpu`
   - `test_colab_cloud_runner_get_status_cpu_no_fake_guarantees`  
   - Root Cause: Colab CLI mock expectations.
3. `tests/unit/test_orchestrator_tools.py`:
   - `test_ml_preflight_audit_tool`
   - `test_ml_prepare_feature_pipeline_tool`
   - `test_ml_run_model_tournament_tool`
   - `test_ml_certify_safety_and_decisions_tool`
   - `test_end_to_end_orchestrator_chaining`  
   - Root Cause: FastMCP tool async execution / loop binding in pytest.
4. `tests/unit/test_server.py::test_ml_ping_diagnostics`  
   - Root Cause: Missing `pytest.mark.asyncio` on async tool test.

---

## 3. Verification Test Status (`tests/verification/test_reproduce_audit_findings.py`)

- `test_a_conformal_hardcoded_fallback`: FAILED (because hardcoded 95% was removed in source, returning `None`).
- `test_b_energy_ood_tabular_instability`: PASSED (Issue B still exists in source).
- `test_c_calibrator_in_sample_leakage`: PASSED (Issue C still exists in source).
- `test_d_feature_orchestrator_preprocessor_leakage`: PASSED (Issue D still exists in source).
- `test_e_leakage_silent_zero_indistinguishable`: PASSED (Issue E still exists in source).
- `test_f_tournament_swallowed_exceptions`: FAILED (LGBM duplicate verbose was fixed in source).

---

## 4. Next Phase: Phase 1 Remediation

Rewriting `test_reproduce_audit_findings.py` into regression guards asserting the correct behavior (RED phase), then fixing C1-C4, H5-H7, M3-M4 in `calibrator.py` and `safety_orchestrator.py`.
