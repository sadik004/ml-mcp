# P1: RED Tests (every finding becomes a failing, locked test)

**Goal:** encode C1–C6, H1–H9 and the gate blind spots as tests that fail on the current HEAD. The point is to fix the definition of "correct" *before* anyone touches the code, so the fix cannot redefine it.
**Touches:** `tests/verification/test_r_*.py`, `tests/verification/heavy/`, `tests/verification/fixtures/`. Zero changes to `src/`.

## Shared fixtures (`tests/verification/conftest.py`)

- `SEEDS = (0, 1, 2, 3, 4)`. Fixed and listed explicitly (T6).
- `make_binary(n, labels=(0, 1), pos_rate=0.3, seed)` uses `sklearn.datasets.make_classification` and then remaps labels.
- `make_noise(n, seed)`: features independent of y (permuted labels). Used for winner's-curse and leakage tests.
- `csv_path(tmp_path, df)` writes the CSV that the service layer reads.

## Oracle rule (T3)

Every expected value is recomputed in the test from **raw** arrays the tool returns (`oof_proba`, `y_true`, `prediction_sets`, ...). If the tool does not return the raw arrays yet, the test asserts that they exist, which is itself RED until P2–P6 add them.

---

## `test_r_scoring.py` (C1, C2, C6, H9) → fixed in P2

| Test id | Setup | Assertion (oracle) | Why RED now |
|---|---|---|---|
| `test_stacking_average_precision_is_true_ap` | binary, n=400, `scoring="average_precision"` | `abs(res.score - average_precision_score(y, res.oof_meta_proba[:,1])) < 1e-12` and `abs(res.score - accuracy_score(y, argmax)) > 1e-6` | C1: returns accuracy |
| `test_stacking_neg_log_loss_is_negative_log_loss` | same, `scoring="neg_log_loss"` | `abs(res.score + log_loss(y, res.oof_meta_proba)) < 1e-12`, `res.score < 0` | C1 |
| `test_stacking_unknown_scoring_raises` | `scoring="acuracy"` (typo) | `pytest.raises(ValueError, match="scoring")` | H9: silently becomes accuracy |
| `test_stacking_oof_failure_never_returns_insample` | fault injection: `patch("...stacking_engine.cross_val_predict", side_effect=RuntimeError("boom"))` | `res.score is None` and some `w` in `res.warnings` has `"cross_val_predict" in w` | C2: returns `score(X, y)` |
| `test_tournament_fold_scorer_failure_not_mixed` | a scorer wrapper raising on fold 2 only (`side_effect=[v, v, ValueError, v, v]` via `wraps`) | that model's `cv_score is None` or it is excluded, plus a warning; never `mean([auc, auc, acc, ...])` | C6 |
| `test_tournament_regression_group_uses_groupkfold` | regression, continuous y, `group_column="g"` | no exception; `res.splitter == "GroupKFold"` | H9: StratifiedGroupKFold crash |

## `test_r_calibrator.py` (C3, C4, H3, calibrator hygiene) → fixed in P3

| Test id | Setup | Assertion | Why RED now |
|---|---|---|---|
| `test_holdout_branch_lengths_consistent` | n=9 (forces holdout branch, R2) | no exception; `len(report.y_eval) == report.post_probas.shape[0]` | C3 crash |
| `test_labels_one_two_n400` | labels {1,2}, n=400 (R3) | no exception; `set(model.predict(X)) <= {1, 2}` | C4 crash |
| `test_string_labels` | labels {"no","yes"} | no exception; `set(model.predict(X)) <= {"no","yes"}`; `model.classes_.tolist() == ["no","yes"]` | C4 TypeError |
| `test_ece_matches_reference_binning` | fixed probas/labels | `abs(calculate_ece(y,p,10) - ref_ece(y,p,10)) < 1e-12`, where `ref_ece` is written in the test with equal-width bins, Σ\|B\|/N·\|acc−conf\| | regression guard (must still be GREEN; recorded as guard in RED_EVIDENCE) |
| `test_fallback_mode_is_not_out_of_fold` | fault injection on `cross_val_predict` inside the calibrator | `report.evaluation_mode != "out_of_fold"` and a warning mentions `"in_sample"` | H3 hardcoded |
| `test_calibrators_unfitted_before_fit` | `TemperatureScaler()`, `BetaCalibrator()` | `pytest.raises(NotFittedError)` from `check_is_fitted` | attrs set in `__init__` |
| `test_adaptive_ece_docstring_no_debiased_claim` | `inspect.getdoc(calculate_adaptive_ece)` | `"Debiased" not in doc` | doc impersonation |

## `test_r_leakage.py` (C5) → fixed in P4

Spy technique (T4-compliant): `patch.object(<PreprocessorClass>, "fit", autospec=True, side_effect=record_and_call)`. Here `record_and_call` stores `len(X)` and calls the real `fit`, so it observes without answering.

| Test id | Setup | Assertion | Why RED now |
|---|---|---|---|
| `test_tournament_preprocessor_never_sees_full_data_in_cv` | mixed numeric + categorical, n=300, cv=5 | every recorded fit size during CV is `<= ceil(300*4/5)`; a single fit on all 300 is allowed only **after** the CV scores are produced (final refit) | C5 `fit_transform` on full X |
| `test_safety_service_paths_fit_inside_split` | parametrized over `calibrate_probabilities`, `tune_threshold_and_errors`, `stress_test_and_fairness`, `explain_predictions` | the preprocessor fit size equals the train-split size, never n | C5 ×4 call sites |
| `test_persisted_artifact_serves_raw_features` | run tournament, load saved artifact | `artifact.predict_proba(raw_df_with_strings)` works, and the artifact `isinstance(Pipeline)` | pipeline not persisted |
| `test_target_leak_signal_not_inflated` | `make_noise`, plus a categorical column with unique values per row | over 5 seeds, mean OOF AUC is ≤ `0.5 + 3*se` (`# bound: 0.5+3*se`) | leakage inflates the score |
| `test_user_model_refit_warns` | pass `model_path` | some warning contains `"refit"`/`"clone"` | silent clone + refit |

## `test_r_eval_mode.py` (H1, H2, H4) → fixed in P5

| Test id | Setup | Assertion | Why RED now |
|---|---|---|---|
| `test_loaded_model_same_csv_not_held_out` | train and persist on csv A, then call `tune_threshold_and_errors(model_path, csv A)` | `res.evaluation_mode in {"in_sample", "unknown_provenance"}` plus a warning | H1 |
| `test_small_n_not_labelled_held_out` | n=15 | `res.evaluation_mode != "held_out_test"` | H1 `p_te = p_tr` |
| `test_threshold_tuned_on_oof` | spy `wraps` on `evaluation.oof_predict_proba` | spy called ≥ 1 during threshold search; threshold equals the brute-force optimum recomputed in the test from the returned `oof_proba` | H2 in-sample |
| `test_champion_score_from_outer_holdout` | tournament, 9 models | `res.champion_score_source == "outer_holdout"`; `set(res.holdout_index).isdisjoint(res.selection_index)` | H4 |
| `test_winners_curse_on_noise` | `make_noise`, 5 seeds | mean `champion_holdout_auc` ≤ `0.5 + 3*se` (`# bound: 0.5+3*se`) | H4 max-of-9 is biased upwards |

## `test_r_crc.py` (H5–H8, quantile) → fixed in P6

| Test id | Setup | Assertion | Why RED now |
|---|---|---|---|
| `test_non_monotone_cost_rejected` | `asymmetric_cost` with the current cost matrix | `pytest.raises(ValueError, match="monotone")` | H5 |
| `test_monotone_loss_property` | valid cost matrix, λ grid | the loss vector is non-increasing in λ for every sample (`np.all(np.diff(L, axis=1) <= 0)`) | H5 |
| `test_ucb_hoeffding_for_bounded_loss` | fractional losses ∈ [0,1], n=500, δ=0.1 | `abs(res.risk_ucb - (mean + sqrt(log(1/δ)/(2n)))) < 1e-12` (or Hoeffding-Bentkus if chosen; the formula is fixed in the test) | H6 Clopper-Pearson |
| `test_infeasible_mondrian_flagged` | Mondrian, one class with n_c=10, α=0.05 | `"INFEASIBLE" in " ".join(res.warnings)` and `res.status == "infeasible"` | H7 silent λ=1 |
| `test_mondrian_lambda_is_none` | Mondrian, feasible | `res.calibrated_lambda is None` | H8 dummy 0.0 |
| `test_guarantee_uses_ucb_not_point` | construct emp risk ≤ α but UCB > α | `res.guarantee_satisfied is False` | H8 overclaim |
| `test_split_conformal_quantile_higher` | known scores, n=99, α=0.1 | `q_hat == np.quantile(s, ceil((n+1)*(1-α))/n, method="higher")` exactly | linear interpolation |
| `test_quantile_level_above_one_is_inf` | n=5, α=0.05 | `q_hat == inf` and a warning | clipped level |
| `test_raps_named_honestly` | method metadata / docstring | contains `"RAPS-like"` or `u`-randomization is present | naming |

## `heavy/test_r_coverage_sim.py` (Colab only, T11) → closes in P6

- `test_marginal_coverage_simulation`: 500 repetitions × 5 seeds, n_cal=500, α=0.1. Assert `mean_cov >= 0.9 - 3*sqrt(0.9*0.1/(500*reps))` (`# bound:` comment).
- `test_crc_risk_simulation`: same design for CRC risk ≤ α + 3·SE.

## `test_r_gate_meta.py` (gate blind spots) → fixed in P7

Fixture files with known bad and clean code. Each checker must catch every bad fixture and none of the clean ones.

| Test id | Bad fixture | Expected |
|---|---|---|
| `test_g1_flags_broad_except_without_warning` | `except Exception: return model.score(X, y)` | 1 violation |
| `test_g1_allows_except_with_warning_append` | `except Exception as e: warnings.append(str(e)); raise` | 0 |
| `test_g2_flags_metric_float_assignment` | `overfit_gap = 0.0`, `calibrated_lambda=... else 0.0` | 2 violations |
| `test_seed_check_flags_all_forms` | `random_state: int = 42`, `random_seed=42`, `TPESampler(seed=42)`, `_splitter(seed=42)` | 4 violations |
| `test_seed_check_allows_config_seed` | `random_state=settings.random_state` | 0 |
| `test_gates_zero_on_src` | run G1, G2 and the seed check on `src/ml_mcp` | 0 violations (RED now: 16 seeds plus C2) |

## Procedure (strict order)

1. Write all files listed above.
2. Run `python -m pytest tests/verification -q -rA > docs/plans/remediation_v3/red_run.txt`. **Expected: every `test_r_*` FAILS** except tests explicitly marked as guards in RED_EVIDENCE.
3. Fill `RED_EVIDENCE.md`: `| test id | commit | failure line | guard? |`.
4. If any non-guard test passes on HEAD, the test is wrong. Rewrite it (it is not locked yet).
5. User reviews and approves → `python scripts/lock_tests.py --write` → commit `test(verification): lock RED suite v3`.

## Exit criteria

```powershell
python scripts/check_test_integrity.py            # exit 0 (T1–T9 satisfied, LOCK present)
python -m pytest tests/verification -q            # EXPECTED exit 1, failures == number of non-guard tests
```
