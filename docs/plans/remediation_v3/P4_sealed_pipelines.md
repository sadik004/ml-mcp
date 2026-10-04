# P4: Sealed Pipelines, No Preprocessing Leakage (C5)

**Turns GREEN:** `tests/verification/test_r_leakage.py` (5 tests).
**Files:** `engine/tournament.py`, `services/safety_service.py`, `services/evaluation.py`.

## Changes

1. Replace the 5 copy-pasted `has_non_numeric → pipe.fit_transform(X, y)` blocks with one helper, `evaluation.prepare_estimator(X, base_estimator) -> Pipeline`. It returns `build_sealed_pipeline(preprocessor, estimator)` and does **not** fit.
2. Every CV and score path uses `oof_predict_proba(pipeline, X_raw, y, cv)`, so the preprocessor is fit inside each training fold.
3. Train/test paths in `safety_service` (L147-151, 199-203, 292-296, 346-350) fit the pipeline on the train split only.
4. Persist the **whole** fitted `Pipeline` (preprocessor + model) and assert `isinstance(artifact, Pipeline)` before saving.
5. `calibrate_probabilities(model_path=...)`: if a clone is refit, append the warning `"user model was cloned and refit on the provided data; returned artifact is not the original model"`.
6. `stress_test_and_fairness` L398-400: dropping categoricals now produces a warning that lists the dropped columns. Rename the misleading `cal_fraction` usage, or document it as the test fraction in the DTO field description.

## Exit criteria

```powershell
python -m pytest tests/verification/test_r_leakage.py -q      # exit 0, 5 passed
python -m pytest tests/unit tests/integration -q              # exit 0
python scripts/check_test_integrity.py                        # exit 0
rg -n "fit_transform\(" src/ml_mcp/engine/tournament.py src/ml_mcp/services/safety_service.py   # 0 hits
```
