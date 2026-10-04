# RED Test Evidence Register

| Test ID | File | Commit | Failure Line | Guard? | Failure Snippet / Reason |
|---|---|---|---|---|---|
| test_stacking_average_precision_is_true_ap | test_r_scoring.py | HEAD | L44 | False | Metric Impersonation: AP returned accuracy score 0.9025 instead of true AP 0.8392 |
| test_stacking_neg_log_loss_is_negative_log_loss | test_r_scoring.py | HEAD | L66 | False | Metric Impersonation: neg_log_loss returned accuracy score instead of negative value |
| test_stacking_unknown_scoring_raises | test_r_scoring.py | HEAD | L77 | False | Silent fallback to accuracy instead of raising ValueError on invalid metric |
| test_stacking_oof_failure_never_returns_insample | test_r_scoring.py | HEAD | L100 | False | Returns in-sample stacking_model.score(X,y) on cross_val_predict failure without warning |
| test_tournament_fold_scorer_failure_not_mixed | test_r_scoring.py | HEAD | L119 | False | Fold failure falls back to estimator.score() averaging mixed metrics |
| test_tournament_regression_group_uses_groupkfold | test_r_scoring.py | HEAD | L129 | False | StratifiedGroupKFold crashes on continuous regression target |
| test_ece_matches_reference_binning | test_r_calibrator.py | HEAD | - | True | Regression guard for true reliability diagram equal-width binning formula |
| test_holdout_branch_lengths_consistent | test_r_calibrator.py | HEAD | L59 | False | Inconsistent sample length: y_eval/pre_probas sliced but post_probas remains full length |
| test_labels_one_two_n400 | test_r_calibrator.py | HEAD | L77 | False | np.bincount(y) with min=0 routes labels {1, 2} into corrupted holdout branch |
| test_string_labels | test_r_calibrator.py | HEAD | L100 | False | String labels raise TypeError/ValueError in bincount and metric calculation |
| test_fallback_mode_is_not_out_of_fold | test_r_calibrator.py | HEAD | L126 | False | Hardcoded evaluation_mode='out_of_fold' returned during in-sample CV fallback |
| test_calibrators_unfitted_before_fit | test_r_calibrator.py | HEAD | L133 | False | Fitted attributes lr_ and temperature_ set in __init__ breaking check_is_fitted |
| test_adaptive_ece_docstring_no_debiased_claim | test_r_calibrator.py | HEAD | L145 | False | Docstring falsely claims debiased Roelofs 2022 estimator |
| test_tournament_preprocessor_never_sees_full_data_in_cv | test_r_leakage.py | HEAD | L59 | False | Data Leakage: fit_transform called on full 300 rows prior to cross-validation splits |
| test_safety_service_calibrate_fits_inside_split | test_r_leakage.py | HEAD | L85 | False | Data Leakage: fit_transform called on full data during probability calibration |
| test_persisted_artifact_serves_raw_features | test_r_leakage.py | HEAD | L107 | False | Tournament does not encapsulate preprocessor into final serving Pipeline |
| test_user_model_refit_warns | test_r_leakage.py | HEAD | L138 | False | Model cloned and refitted without registering user-facing warning |
| test_loaded_model_same_csv_not_held_out | test_r_eval_mode.py | HEAD | L44 | False | Pre-trained model evaluated on training CSV falsely claims held_out_test mode |
| test_small_n_not_labelled_held_out | test_r_eval_mode.py | HEAD | L66 | False | When n < 20, p_te is completely in-sample but claims held_out_test |
| test_threshold_tuned_on_oof | test_r_eval_mode.py | HEAD | L88 | False | Decision threshold optimized on in-sample p_tr rather than OOF predictions |
| test_champion_score_from_outer_holdout | test_r_eval_mode.py | HEAD | L104 | False | Champion score evaluated on same folds used for selection (Winner's Curse) |
| test_non_monotone_asymmetric_cost_rejected | test_r_crc.py | HEAD | L21 | False | Asymmetric cost is non-monotone in set size, violating CRC theoretical guarantee |
| test_monotone_loss_property | test_r_crc.py | HEAD | - | True | Mathematical invariant check for monotone losses |
| test_infeasible_mondrian_flagged_with_warning | test_r_crc.py | HEAD | L63 | False | Silently returns lambda=1.0 when sample size is insufficient for target risk |
| test_mondrian_lambda_is_none_or_explicit | test_r_crc.py | HEAD | - | True | Mondrian class-conditional returns dictionary of lambdas |
| test_split_conformal_quantile_higher | test_r_crc.py | HEAD | L93 | False | Conformal quantile uses default linear interpolation instead of method='higher' |
| test_quantile_level_above_one_is_inf | test_r_crc.py | HEAD | L104 | False | Small-sample quantile level > 1 clips instead of setting q_hat=inf with warning |
| test_raps_named_honestly | test_r_crc.py | HEAD | L116 | False | Non-randomized prediction sets mislabeled as true randomized RAPS |
| test_g1_flags_broad_except_without_warning | test_r_gate_meta.py | HEAD | - | True | Guard test: AST checker detects silent broad except |
| test_g1_allows_except_with_warning_append | test_r_gate_meta.py | HEAD | - | True | Guard test: AST checker permits documented exceptions |
| test_seed_check_flags_all_hardcoded_forms | test_r_gate_meta.py | HEAD | - | True | Guard test: AST checker catches hardcoded seed literals |
| test_current_src_contains_zero_violations_gate | test_r_gate_meta.py | HEAD | L106 | False | Current src contains 16 hardcoded seeds and C2 fallback |
| test_marginal_coverage_simulation | test_r_coverage_sim.py | HEAD | - | True | Colab Monte Carlo simulation guard for marginal conformal coverage |
