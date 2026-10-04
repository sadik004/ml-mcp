# P2: Scoring Honesty (C1, C2, C6, H9)

**Turns GREEN:** `tests/verification/test_r_scoring.py` (6 tests). Locked tests must not be edited (T1).
**Files:** `engine/stacking_engine.py`, `engine/tournament.py`, `schemas/tournament.py`, and a new `engine/scoring.py`.

## Changes

1. **New `engine/scoring.py`**: one source of truth.
   - `resolve_scorer(name: str, task: Literal["binary","multiclass","regression"]) -> Callable[[y, oof], float]`
   - It maps names to `sklearn.metrics` functions using a `dict` (O(1)). An unknown name raises `ValueError(f"unsupported scoring '{name}'; allowed: {sorted(keys)}")`.
   - Probability metrics (`roc_auc`, `average_precision`, `neg_log_loss`, `brier`) take OOF probabilities; label metrics take argmax mapped through `classes_`.
2. **stacking_engine.py**
   - Delete the `if/elif` chain at L114-125 and call `resolve_scorer`.
   - Expose `oof_meta_proba: np.ndarray` on the result.
   - Delete the `except Exception:` → `stacking_model.score(X, y)` fallback (L136-139). On failure: `score=None` and `warnings.append(f"cross_val_predict failed: {type(e).__name__}: {e}")`. An in-sample number must never be produced.
3. **tournament.py**
   - Per-fold scorer failure: mark the model `cv_score=None`, add a warning, and exclude it from ranking. The `estimator.score()` fallback at L298-305 is removed.
   - Splitter choice through a dict: `(task, has_groups) -> splitter class`; regression + groups → `GroupKFold`. Expose `res.splitter: str`.
4. **schemas/tournament.py**: `score: Optional[float]`, `splitter: str`, `warnings: List[str]`.

## Not allowed
- Catching the injected `RuntimeError` and returning any number.
- Adding `average_precision` to a "skip" list so the test stops exercising it.

## Exit criteria

```powershell
python -m pytest tests/verification/test_r_scoring.py -q                 # exit 0, 6 passed
python -m pytest tests/unit tests/integration -q                         # exit 0 (no regressions)
python scripts/check_test_integrity.py                                   # exit 0 (lock intact)
python -m ruff check src ; python -m mypy --strict src/ml_mcp            # exit 0
```
