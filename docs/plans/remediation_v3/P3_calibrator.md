# P3: Calibrator Correctness (C3, C4, H3, hygiene)

**Turns GREEN:** `tests/verification/test_r_calibrator.py` (7 tests; the ECE guard must stay green).
**Files:** `engine/calibrator.py`, `schemas/tuning.py` (CalibrationReportDTO).

## Changes

1. **Label encoding at entry** (`ProbabilityCalibrator.calibrate`): `le = LabelEncoder().fit(y)` and `y_enc = le.transform(y)`. All internal maths (`bincount`, ECE, Brier, column indexing at L445/448) use `y_enc`. `classes_ = le.classes_` is stored on the calibrated model, and `predict` returns `classes_[argmax]`.
2. **C3:** in the holdout branch, `post_probas = eval_post` so that every array is the same length.
3. **H3:** `evaluation_mode` is set per path: `"out_of_fold"` only when `cross_val_predict` succeeded; otherwise `"in_sample"` plus a warning. In that case conformal coverage is reported with `coverage_valid=False`.
4. **Fitted-state hygiene:** move `lr_` and `temperature_` out of `__init__` so they are assigned only in `fit`, which makes `check_is_fitted` work. Fix the `VectorScalingCalibrator` copy-paste message at L276.
5. **Docs:** remove the "Debiased (Roelofs 2022)" claim from `calculate_adaptive_ece` and describe it as equal-mass binning.
6. **`is_well_calibrated`:** replace the magic 0.15/0.10 with `ece_tolerance: float` taken from `config.py`. Drop the Brier condition because a Brier threshold depends on the base rate.

(The quantile `method="higher"` change belongs to P6, so that statistics changes stay in one phase.)

## Exit criteria

```powershell
python -m pytest tests/verification/test_r_calibrator.py -q   # exit 0, 7 passed
python -m pytest tests/unit tests/integration -q              # exit 0
python scripts/check_test_integrity.py                        # exit 0
python -m ruff check src ; python -m mypy --strict src/ml_mcp # exit 0
```
