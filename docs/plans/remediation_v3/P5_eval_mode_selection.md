# P5: Honest Evaluation Mode and Selection Bias (H1, H2, H4)

**Turns GREEN:** `tests/verification/test_r_eval_mode.py` (5 tests).
**Files:** `services/safety_service.py`, `engine/tournament.py`, `engine/tournament_orchestrator.py`, `schemas/tournament.py`, `schemas/tuning.py`.

## Changes

1. **Evaluation-mode enum** (`Literal["out_of_fold","held_out_test","in_sample","unknown_provenance"]`) shared by every DTO. Free-text strings are no longer accepted.
2. **H1:** a loaded `model_path` has unknown training rows, so it gets `"unknown_provenance"` plus a warning. If the artifact metadata stores a training-data fingerprint (SHA-256 of the CSV) and it matches the current CSV, the mode is `"in_sample"`. When n < 20 there is no test split, so the mode is `"in_sample"`, never `"held_out_test"`.
3. **H2:** threshold search runs on `oof_predict_proba` of the train split. The test split is used only for the final reported metrics. Return `oof_proba` so the test can recompute the optimum.
4. **H4 (winner's curse):**
   - `TournamentArena.run_tournament`: before CV, split off an outer holdout with `holdout_fraction` from config, stratified or group-aware. Selection runs on the rest; `champion_score` = the champion's score on the holdout. The max CV mean is still reported, but as `selection_cv_score`.
   - Expose `selection_index`, `holdout_index`, `champion_score_source="outer_holdout"`.
   - `tournament_orchestrator.py` L254-267: the arena runs on the inner-training part of each outer fold only, so outer folds are never reused for model selection.
   - If the data is too small for a holdout (fewer than `min_holdout_n` rows from config), set `champion_score=None`, `champion_score_source="unavailable"` and add a warning. No dummy 0.0.

## Exit criteria

```powershell
python -m pytest tests/verification/test_r_eval_mode.py -q    # exit 0, 5 passed
python -m pytest tests/unit tests/integration -q              # exit 0
python scripts/check_test_integrity.py                        # exit 0
```
