# P8: Final Gate, Documentation, RCA

**Goal:** prove the whole system on a clean Colab runtime and make the docs tell the truth.

## Tasks

1. `ml_colab_status`, then upload the HEAD snapshot and run `python scripts/quality_gate.py --all --json gate.json` plus `pytest tests -q`, including `heavy/`.
2. Download `gate.json` and the junit XML into `docs/rca/colab_runs/<HEAD>.json`.
3. Update `docs/rca/final_scorecard.md`. Every row cites the gate id, the evidence line and the commit. G0 must be the first row.
4. Write `docs/rca/2026-10-04_gate_false_negatives.md` covering: symptom (10/10 while C1–C6 existed), root cause (substring and literal-pattern gates, happy-path tests only), fix (P0 contract + P7 AST gates), and prevention (the locked RED suite).
5. `ROADMAP.md`: every guarantee line carries `[test: tests/verification/test_r_x.py::test_id]` (G10 verifies the tags).
6. Update `docs/TOOL_REFERENCE.md` with the new DTO fields: `evaluation_mode` enum, `warnings`, `champion_score_source`, `status`, `ucb_method`, `per_class_lambda`, `splitter`.
7. Update the `fastapi-production` skill file with the patterns taught here: scorer registry, sealed pipeline, test lock.

## Exit criteria

```powershell
python scripts/quality_gate.py --all          # exit 0, 11/11 (G0..G10)
python -m pytest tests -q                     # exit 0
python scripts/check_test_integrity.py --require-colab   # exit 0
git diff --stat <P1-lock-commit> -- tests/verification/test_r_*.py   # empty, OR every hunk listed in LOCK_CHANGES.md
```
