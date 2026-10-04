# P7: Gate Hardening (close the false negatives)

**Turns GREEN:** `tests/verification/test_r_gate_meta.py` (6 tests), including `test_gates_zero_on_src`.
**Files:** `scripts/check_no_silent_except.py` (G1), `scripts/quality_gate.py` (G2, seed check), `src/ml_mcp/config.py`, and the 12+ files that still hold hardcoded seeds.

## Changes

1. **G1 (AST):** flag any `except` / `except Exception` / `except BaseException` whose body contains neither a `raise` nor a call to `<x>.warnings.append` / `warnings.append` / `logger.warning|error|exception`. Returning a value from such a handler counts as a violation.
2. **G2 (AST):** flag any float or int literal assigned to a target, keyword, or conditional-expression branch whose name matches `(score|auc|ece|brier|coverage|lambda|gap|accuracy|f1|precision|recall)` (case-insensitive). Allow-list: `config.py` and `# metric-literal-ok: <reason>` comments, and every use of that comment is listed in the gate report.
3. **Seed check (AST):** flag any integer literal used as the default or the keyword value of a parameter named `seed`, `random_state` or `random_seed`. Allowed: `settings.random_state` and parameters typed `Optional[int] = None` that resolve from config.
4. **Fix src:** add `random_state: int` to `config.py` and thread it through every one of the 16 sites. Remove `overfit_gap = 0.0`, `champion_score ... else 0.0` and `calibrated_lambda ... else 0.0` (most were already removed in P5/P6).
5. **G3 update:** the gate requires that `tests/verification/test_r_*` contain at least one fault-injection test per module with a fallback branch (stacking, tournament, calibrator, safety_service).

## Exit criteria

```powershell
python -m pytest tests/verification/test_r_gate_meta.py -q    # exit 0, 6 passed
python scripts/quality_gate.py --only G0,G1,G2,G9             # exit 0
python scripts/check_test_integrity.py                        # exit 0
```
