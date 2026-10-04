# Remediation v3: Master Plan and Test-Integrity Contract

Source of findings: `code_review.md` (C1–C6, H1–H9, P2 items, gate blind spots).
Status: **AWAITING APPROVAL**. No production code changes until P0 + P1 are approved.

## Phase map

| Phase | File | Findings | Depends on | Exit gate |
|---|---|---|---|---|
| P0 | [P0_test_integrity.md](P0_test_integrity.md) | Anti-gaming infrastructure | none | G0 green |
| P1 | [P1_red_tests.md](P1_red_tests.md) | Convert every finding into a failing test | P0 | All `test_r_*` RED, evidence logged, LOCK written |
| P2 | [P2_scoring_honesty.md](P2_scoring_honesty.md) | C1, C2, C6, H9 | P1 | `test_r_scoring.py` green |
| P3 | [P3_calibrator.md](P3_calibrator.md) | C3, C4, H3, calibrator P2 items | P1 | `test_r_calibrator.py` green |
| P4 | [P4_sealed_pipelines.md](P4_sealed_pipelines.md) | C5, refit-warning, DRY | P2 | `test_r_leakage.py` green |
| P5 | [P5_eval_mode_selection.md](P5_eval_mode_selection.md) | H1, H2, H4 | P4 | `test_r_eval_mode.py` green |
| P6 | [P6_crc_statistics.md](P6_crc_statistics.md) | H5–H8, quantile, RAPS naming, vectorize | P3 | `test_r_crc.py` + Colab heavy green |
| P7 | [P7_gate_hardening.md](P7_gate_hardening.md) | G1, G2, seed check, G3 blind spots | P2–P6 | `test_r_gate_meta.py` green, 0 violations on `src/` |
| P8 | [P8_final_gate_docs.md](P8_final_gate_docs.md) | Colab full run, scorecard, RCA, ROADMAP | P7 | `quality_gate.py` 11/11, exit 0 |

One phase at a time. A phase is closed only when its exit commands return exit code 0 **and** the output is pasted into `PHASE_LOG.md`.

---

## The Test-Integrity Contract (binding, enforced by G0)

These rules exist so that a test cannot be bent to make code look correct. Each rule names the mechanism that enforces it. A rule that is only written down and not enforced does not count.

| # | Rule | Enforcement |
|---|---|---|
| T1 | **Test lock.** After P1 is approved, the SHA-256 of every `tests/verification/test_r_*.py` is frozen in `tests/verification/LOCK.json`. | G0 recomputes the hashes. Any mismatch fails unless `LOCK_CHANGES.md` has an entry with the old/new hash, the reason, and `Approved-by: user`. |
| T2 | **No escape hatches.** No `skip`, `skipif`, `xfail`, `importorskip`, early `return`, or `try/except` around assertions in `tests/verification/`. | `tests/verification/conftest.py` collection hook plus the G0 AST scan. `xfail_strict = true` in `pyproject.toml`. |
| T3 | **Independent oracle.** The expected value is computed inside the test, from raw arrays, using sklearn/numpy/scipy. It is never read from a field the tool reports about itself. | Code review checklist plus G0 AST: every `test_r_*` must import at least one of `sklearn.metrics`, `numpy`, or `scipy.stats`. |
| T4 | **Mocks may only observe or break; they may not answer.** Allowed: `wraps=` spies (call-through) and `side_effect=<Exception>` fault injection at a named external boundary. Banned: `return_value=` on the unit under test or its metric functions. | G0 AST scan bans `return_value=` and `.return_value =` in `test_r_*`. |
| T5 | **RED evidence.** Every new test must fail on the pre-fix commit. A test that passes before the fix is invalid. | `RED_EVIDENCE.md` lists the commit hash, test id and failure line for each test. G0 checks that every `test_r_*` id appears there. |
| T6 | **Tolerances are declared, not tuned.** Deterministic metrics use `abs(a - b) < 1e-12`. Statistical claims use a bound derived from a formula (e.g. `1-α - 3·SE`) with ≥ 5 fixed seeds listed in the test. | G0 AST bans `pytest.approx(..., rel=` or `abs=` greater than `1e-9` in `test_r_*`, unless the line carries `# bound: <formula>`. |
| T7 | **Production code must not detect tests.** | G0 greps `src/` for `pytest`, `PYTEST_CURRENT_TEST`, `unittest`, `sys.modules["pytest"]`. Zero hits is required. |
| T8 | **Meaningful assertions.** Each test needs ≥ 1 assertion that compares against a value. A test whose only checks are `assert True`, `assert x`, or `assert x is not None` is banned. | G0 AST check. |
| T9 | **No test deletion or renaming.** The set of test ids in `test_r_*` may only grow. | G0 compares collected ids against `LOCK.json["ids"]`. |
| T10 | **2-strike rule.** If the same test fails after two fix attempts, stop. Write an RCA in `docs/rca/`, re-plan, and ask for approval. | Manual; logged in `PHASE_LOG.md`. |
| T11 | **Heavy compute runs on Colab.** Simulation tests live in `tests/verification/heavy/` and run via `ml_colab_execute`. | G0 requires `docs/rca/colab_runs/<commit>.json` with `exit_code: 0` for the current HEAD before P6 and P8 can close. |

> [!IMPORTANT]
> Changing a locked test is allowed **only** when the test itself is mathematically wrong (wrong oracle, wrong formula). "The code can't pass it" is never a valid reason. Every change needs your explicit approval in `LOCK_CHANGES.md`.

## Files created by this plan

```
docs/plans/remediation_v3/
  README.md  P0..P8 *.md  PHASE_LOG.md  RED_EVIDENCE.md  LOCK_CHANGES.md
tests/verification/
  conftest.py  LOCK.json
  test_r_scoring.py  test_r_calibrator.py  test_r_leakage.py
  test_r_eval_mode.py  test_r_crc.py  test_r_gate_meta.py
  heavy/test_r_coverage_sim.py
  fixtures/bad_code/*.py  fixtures/clean_code/*.py
scripts/check_test_integrity.py   (G0)
```
