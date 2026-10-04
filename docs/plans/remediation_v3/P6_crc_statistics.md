# P6: Conformal / CRC Statistical Validity (H5–H8 + quantile)

**Turns GREEN:** `tests/verification/test_r_crc.py` (9 tests) and, on Colab, `tests/verification/heavy/test_r_coverage_sim.py` (2 tests).
**Files:** `engine/conformal_risk_control.py`, `engine/calibrator.py` (quantile only), `services/safety_service.py` L460-495, `schemas/safety.py`.

## Changes

1. **H5:** `asymmetric_cost` takes an explicit `cost_matrix` (no magic 0.2/1.0). At construction, verify that the loss is monotone non-increasing in λ for the nested-set family; otherwise raise `ValueError("loss not monotone in lambda; CRC guarantee invalid")`.
2. **H6:** the risk UCB uses Hoeffding, `mean + sqrt(log(1/δ)/(2n))`, for losses bounded in [0,1]. Clopper-Pearson is used only when the losses are verified to be in {0,1}. Report `ucb_method`. `coverage_ci` is only emitted for the miscoverage loss.
3. **H7:** when no λ in the grid meets the target, set `status="infeasible"` and add the warning `"INFEASIBLE: n_c=<n> < 1/alpha-1=<k> for class <c>"`. λ=1.0 is still returned so the output is usable, but it is flagged.
4. **H8:** `guarantee_satisfied = risk_ucb <= alpha`. For Mondrian, `calibrated_lambda=None` and per-class λ goes into `per_class_lambda: Dict[label, float]`.
5. **Quantile:** `np.quantile(scores, level, method="higher")` with `level = ceil((n+1)(1-α))/n`. If `level > 1`: `q_hat = inf`, `coverage_guaranteed=False`, and a warning.
6. **RAPS naming:** rename it to `"raps_like"` (non-randomized) in code, docs and DTOs.
7. **Performance:** vectorize the λ grid search. Sort scores once and compute risk via `np.searchsorted` with cumulative sums, O((G + n) log n) instead of the Python O(G·n) loop. The P1 tests pin the result to be identical to the old loop's output on valid inputs.

## Colab run (T11)

1. `ml_colab_status` to confirm the session.
2. `ml_colab_upload` the repo snapshot, then `ml_colab_execute`: `pytest tests/verification/heavy -q --junitxml=heavy.xml`.
3. `ml_colab_download` `heavy.xml` and write `docs/rca/colab_runs/<HEAD>.json` with `{commit, exit_code, passed, failed, duration}`.

## Exit criteria

```powershell
python -m pytest tests/verification/test_r_crc.py -q          # exit 0, 9 passed
python scripts/check_test_integrity.py --require-colab        # exit 0 (colab_runs/<HEAD>.json exit_code 0)
python -m pytest tests/unit tests/integration -q              # exit 0
```
