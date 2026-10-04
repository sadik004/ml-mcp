# P0: Test-Integrity Infrastructure (G0)

**Goal:** make the contract in README.md enforceable by a machine *before* any test or fix gets written.
**Touches:** `scripts/check_test_integrity.py`, `scripts/quality_gate.py`, `tests/verification/conftest.py`, `pyproject.toml`. Zero changes to `src/`.

## Tasks

1. `scripts/check_test_integrity.py`: one AST-based checker with sub-checks T1, T2, T3, T4, T5, T6, T7, T8, T9, T11. It exits 1 with `file:line rule message` for each violation.
2. `tests/verification/conftest.py`: a `pytest_collection_modifyitems` hook that raises `pytest.UsageError` if any item under `tests/verification/` carries a `skip`/`skipif`/`xfail` marker.
3. `pyproject.toml` `[tool.pytest.ini_options]`: `xfail_strict = true`, `addopts = "--strict-markers -p no:cacheprovider"`.
4. `scripts/quality_gate.py`: register **G0 Test Integrity** as the first gate. If G0 fails, the run aborts.
5. Create empty `RED_EVIDENCE.md`, `LOCK_CHANGES.md`, `PHASE_LOG.md`.
6. `scripts/lock_tests.py --write`: writes `LOCK.json` = `{ "files": {path: sha256}, "ids": [sorted node ids] }`. It refuses to run if `RED_EVIDENCE.md` is missing any id.

## Tests for the checker itself (`tests/unit/scripts/test_check_test_integrity.py`)

The checker gets its own tests, using a temporary directory of synthetic test files:

| Test | Input file content | Expected |
|---|---|---|
| `test_flags_skip_marker` | `@pytest.mark.skip` on a `test_r_x` | violation T2 |
| `test_flags_early_return` | `if cond: return` before an assertion | violation T2 |
| `test_flags_try_around_assert` | `try: assert ... except AssertionError: pass` | violation T2 |
| `test_flags_return_value_mock` | `patch(..., return_value=0.9)` | violation T4 |
| `test_allows_wraps_spy` | `patch.object(X, "fit", wraps=X.fit, autospec=True)` | no violation |
| `test_flags_loose_approx` | `pytest.approx(a, rel=0.05)` with no `# bound:` comment | violation T6 |
| `test_allows_declared_bound` | same, with `# bound: 1-alpha-3*se` | no violation |
| `test_flags_trivial_assert` | the only assertion is `assert result is not None` | violation T8 |
| `test_flags_src_test_detection` | `src/x.py` contains `os.environ.get("PYTEST_CURRENT_TEST")` | violation T7 |
| `test_flags_hash_drift` | file edited after `LOCK.json` was written, no LOCK_CHANGES entry | violation T1 |
| `test_flags_removed_id` | an id is in LOCK but no longer collected | violation T9 |
| `test_flags_missing_red_evidence` | an id is not in `RED_EVIDENCE.md` | violation T5 |

## Exit criteria

```powershell
python -m pytest tests/unit/scripts/test_check_test_integrity.py -q   # exit 0, 12 passed
python scripts/check_test_integrity.py --mode bootstrap               # exit 0 (no test_r_* yet)
python -m ruff check scripts/ ; python -m mypy --strict scripts/check_test_integrity.py  # exit 0
```

## Forbidden in this phase
- Any edit under `src/ml_mcp/`.
- Writing any `test_r_*` file (that is P1).
