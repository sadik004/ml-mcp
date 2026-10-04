"""Unified Phase 3 Master Model Tournament & Anti-Overfit Tuning Orchestrator.

Honest-validation protocol (Cawley & Talbot, JMLR 2010; Varma & Simon, 2006):
1. Arena: stratified K-fold baseline across candidate models (model selection only).
2. Nested CV: for every OUTER fold, Optuna tunes the champion on an INNER CV of the
   outer-train part only, then the tuned model is scored on the untouched outer-val part.
   The reported champion score is therefore free of hyperparameter-selection bias.
   Tuning fitness = val - lambda * max(0, train - val) - gamma * std_val.
3. Stacking: a real, fully fitted scikit-learn Stacking estimator is evaluated on the same
   outer folds (paired comparison). It is adopted only if it beats the tuned single model
   by at least KISS_MIN_LIFT in mean outer-fold score.
4. Final artifact: tuned (or stacked) model fitted on all data; honest OOF predictions
   (outer folds) are saved for downstream calibration / conformal steps in Phase 4.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from ml_mcp.config import get_settings
from ml_mcp.schemas.tournament import TournamentAndTuningReportDTO

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

KISS_MIN_LIFT = 0.003
VARIANCE_PENALTY_GAMMA = 0.5
INNER_SPLITS = 3


class TournamentOrchestrator:
    """Master single-pass orchestrator executing the full Phase 3 Tournament & Tuning suite."""

    def __init__(self, artifact_dir: str = ".artifacts/models") -> None:
        self.artifact_dir = artifact_dir

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _gap_status(gap: float) -> str:
        gap = max(0.0, gap)
        if gap <= 0.03:
            return "HEALTHY"
        elif gap <= 0.08:
            return "MODERATE"
        return "SEVERE_OVERFIT"

    @staticmethod
    def _score(y_true: np.ndarray, pred: np.ndarray, task_type: str, metric: str) -> float:
        if task_type == "classification":
            if metric in ("pr_auc", "average_precision"):
                return float(average_precision_score(y_true, pred))
            return float(roc_auc_score(y_true, pred))
        return float(r2_score(y_true, pred))

    @staticmethod
    def _predict(model: Any, X: np.ndarray, task_type: str) -> np.ndarray:
        if task_type == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    @staticmethod
    def _splitter(task_type: str, y: np.ndarray, n_splits: int, seed: Optional[int] = None):
        seed_val = seed if seed is not None else get_settings().random_state
        if task_type == "classification":
            min_class = int(np.bincount(y.astype(int)).min())
            if min_class < n_splits:
                raise ValueError(
                    f"Smallest class has {min_class} samples, fewer than n_splits={n_splits}. "
                    "Reduce n_splits or collect more minority samples."
                )
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_val)
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed_val)

    @staticmethod
    def _get_candidate_models(task_type: str, class_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        seed = get_settings().random_state
        if task_type == "classification":
            scale_pos = 1.0
            cw: Optional[Dict[int, float]] = None
            if class_weights and "1" in class_weights and "0" in class_weights:
                scale_pos = float(class_weights["1"] / max(1e-6, class_weights["0"]))
                cw = {0: float(class_weights["0"]), 1: float(class_weights["1"])}

            models: Dict[str, Any] = {
                "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=6, random_state=seed, n_jobs=-1, class_weight=cw),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=80, max_depth=6, random_state=seed, n_jobs=-1, class_weight=cw),
                "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=80, random_state=seed, class_weight=cw),
                "LogisticRegression": LogisticRegression(max_iter=300, random_state=seed, class_weight=cw),
            }
            try:
                import lightgbm as lgb
                models["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.08, scale_pos_weight=scale_pos,
                    random_state=seed, verbose=-1, n_jobs=-1,
                )
            except ImportError as e:
                logger.info(f"LightGBM not installed, skipping in classifier candidates: {e}")
            return models

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=6, random_state=seed, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=80, max_depth=6, random_state=seed, n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=80, random_state=seed),
            "Ridge": Ridge(alpha=1.0),
        }
        try:
            import lightgbm as lgb
            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.08, random_state=seed, verbose=-1, n_jobs=-1,
            )
        except ImportError as e:
            logger.info(f"LightGBM not installed, skipping in regressor candidates: {e}")
        return models

    @staticmethod
    def _suggest_params(trial: optuna.Trial, name: str) -> Dict[str, Any]:
        """Per-architecture search space (each model only receives parameters it supports)."""
        if name == "LightGBM":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            }
        if name in ("RandomForest", "ExtraTrees"):
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        if name == "HistGradientBoosting":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            }
        if name == "LogisticRegression":
            return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
        if name == "Ridge":
            return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
        return {}

    def _tune(
        self,
        name: str,
        base: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        metric: str,
        n_trials: int,
        lam: float,
        timeout_s: float,
    ) -> Tuple[Dict[str, Any], int, bool]:
        """Bayesian tuning on ONE training set with an inner CV. Returns (params, trials_done, timed_out)."""
        inner = self._splitter(task_type, y, INNER_SPLITS)
        splits = list(inner.split(X, y))

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, name)
            tr_s, va_s = [], []
            for tr, va in splits:
                m = clone(base).set_params(**params)
                m.fit(X[tr], y[tr])
                tr_s.append(self._score(y[tr], self._predict(m, X[tr], task_type), task_type, metric))
                va_s.append(self._score(y[va], self._predict(m, X[va], task_type), task_type, metric))
            va_m, va_sd = float(np.mean(va_s)), float(np.std(va_s))
            gap = max(0.0, float(np.mean(tr_s)) - va_m)
            return va_m - lam * gap - VARIANCE_PENALTY_GAMMA * va_sd

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=get_settings().random_state),
            pruner=optuna.pruners.NopPruner(),
        )
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
        done = len(study.trials)
        timed_out = done < n_trials and (time.time() - t0) >= timeout_s
        if done == 0:
            return {}, 0, timed_out
        return dict(study.best_params), done, timed_out

    @staticmethod
    def _build_stack(task_type: str, tuned: Any, other: Any) -> Any:
        estimators = [("m1", clone(tuned)), ("m2", clone(other))]
        if task_type == "classification":
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=300, random_state=get_settings().random_state),
                stack_method="predict_proba",
                cv=INNER_SPLITS,
                n_jobs=1,
            )
        return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=INNER_SPLITS, n_jobs=1)

    # --------------------------------------------------------------------- main
    def run_tournament_and_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"] = "classification",
        primary_metric: str = "pr_auc",
        class_weights: Optional[Dict[str, float]] = None,
        n_splits: int = 5,
        tune_trials: int = 20,
        overfit_penalty_lambda: float = 1.0,
        tune_timeout_s: float = 120.0,
    ) -> TournamentAndTuningReportDTO:
        """Arena -> nested-CV tuning -> paired stacking test -> fitted champion artifact."""
        os.makedirs(self.artifact_dir, exist_ok=True)
        X_arr = X.to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float64)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        warnings: List[str] = []

        if task_type == "classification" and len(np.unique(y_arr)) != 2:
            raise ValueError("Phase 3 currently supports binary classification (exactly 2 classes in target).")
        if task_type == "classification":
            y_arr = y_arr.astype(int)

        outer = self._splitter(task_type, y_arr, n_splits)
        outer_splits = list(outer.split(X_arr, y_arr))

        # ---------------- Stage 3.1: Arena (model selection; untuned) ----------------
        models = self._get_candidate_models(task_type, class_weights)
        arena_results: List[Dict[str, Any]] = []
        for name, base_model in models.items():
            t0 = time.time()
            scores = []
            for tr, va in outer_splits:
                m = clone(base_model).fit(X_arr[tr], y_arr[tr])
                scores.append(self._score(y_arr[va], self._predict(m, X_arr[va], task_type), task_type, primary_metric))
            arena_results.append({
                "model": name,
                "mean_score": round(float(np.mean(scores)), 4),
                "std_score": round(float(np.std(scores)), 4),
                "fit_time_s": round(time.time() - t0, 2),
            })
        arena_results.sort(key=lambda r: r["mean_score"], reverse=True)
        top1_name = arena_results[0]["model"]
        top1_base_score = arena_results[0]["mean_score"]
        top2_name = arena_results[1]["model"] if len(arena_results) > 1 else top1_name
        base1, base2 = models[top1_name], models[top2_name]

        # ---------------- Stage 3.2: Nested CV tuning + paired stacking test ----------------
        oof_tuned = np.zeros(len(y_arr))
        oof_stack = np.zeros(len(y_arr))
        tr_scores, va_scores, stack_scores = [], [], []
        trials_done_total = 0
        timeouts = 0

        for fold_i, (tr, va) in enumerate(outer_splits):
            params, done, timed_out = self._tune(
                top1_name, base1, X_arr[tr], y_arr[tr], task_type, primary_metric,
                tune_trials, overfit_penalty_lambda, tune_timeout_s,
            )
            trials_done_total += done
            timeouts += int(timed_out)
            tuned = clone(base1).set_params(**params).fit(X_arr[tr], y_arr[tr])
            tr_scores.append(self._score(y_arr[tr], self._predict(tuned, X_arr[tr], task_type), task_type, primary_metric))
            p_va = self._predict(tuned, X_arr[va], task_type)
            oof_tuned[va] = p_va
            va_scores.append(self._score(y_arr[va], p_va, task_type, primary_metric))

            stack = self._build_stack(task_type, tuned, base2).fit(X_arr[tr], y_arr[tr])
            s_va = self._predict(stack, X_arr[va], task_type)
            oof_stack[va] = s_va
            stack_scores.append(self._score(y_arr[va], s_va, task_type, primary_metric))

        if timeouts:
            warnings.append(f"Optuna time budget ({tune_timeout_s}s) hit in {timeouts} study(ies); fewer than {tune_trials} trials ran.")

        nested_va = float(np.mean(va_scores))
        nested_tr = float(np.mean(tr_scores))
        cv_fold_std = float(np.std(va_scores))
        gen_gap = nested_tr - nested_va
        gap_status = self._gap_status(gen_gap)
        stack_mean = float(np.mean(stack_scores))
        is_stacking_adopted = (stack_mean - nested_va) >= KISS_MIN_LIFT

        # ---------------- Stage 3.3: Final fit on all data ----------------
        final_params, final_done, final_timeout = self._tune(
            top1_name, base1, X_arr, y_arr, task_type, primary_metric,
            tune_trials, overfit_penalty_lambda, tune_timeout_s,
        )
        trials_done_total += final_done
        if final_timeout:
            warnings.append("Final-fit Optuna study hit its time budget; hyperparameters may be under-explored.")
        final_tuned = clone(base1).set_params(**final_params)

        if is_stacking_adopted:
            champion_arch = f"Stacking_Ensemble({top1_name}_Tuned + {top2_name})"
            champion_score = stack_mean
            final_oof = oof_stack
            final_model = self._build_stack(task_type, final_tuned, base2).fit(X_arr, y_arr)
        else:
            champion_arch = f"Single_Champion_{top1_name}_Tuned"
            champion_score = nested_va
            final_oof = oof_tuned
            final_model = final_tuned.fit(X_arr, y_arr)

        tuning_lift = champion_score - top1_base_score

        champion_path = os.path.join(self.artifact_dir, "champion_model.joblib")
        oof_path = os.path.join(self.artifact_dir, "oof_predictions.npy")
        joblib.dump(final_model, champion_path)
        np.save(oof_path, final_oof)

        meta_path = os.path.join(self.artifact_dir, "tournament_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as tmf:
                json.dump({
                    "target_column": getattr(y, "name", "target") or "target",
                    "task_type": task_type,
                    "champion_architecture": champion_arch,
                    "primary_metric": primary_metric,
                }, tmf, indent=2)
        except Exception as e:
            logger.warning(f"Could not save tournament metadata: {e}")

        protocol = f"nested CV: {n_splits} outer folds x {INNER_SPLITS} inner folds, {tune_trials} trials/study"

        card_lines = [
            "=" * 88,
            "🏆 ML-MCP PHASE 3: MODEL TOURNAMENT & TUNING COMPLETE",
            f"Selected Champion Architecture : {champion_arch}",
            f"Primary Metric ({primary_metric})      : {champion_score:.4f} (untuned arena: {top1_base_score:.4f}, Lift: {tuning_lift:+.4f})",
            f"Validation Protocol            : {protocol}",
            f"Generalization Gap (train-val) : {gen_gap:+.4f} [{gap_status}]",
            "=" * 88,
            "\n📋 ARENA LEADERBOARD (untuned, model selection only):",
        ]
        for idx, row in enumerate(arena_results):
            medal = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else "  "))
            card_lines.append(f"  {medal} #{idx+1} {row['model']:<22} | Score: {row['mean_score']:.4f} (±{row['std_score']:.4f}) | Time: {row['fit_time_s']}s")
        card_lines.extend([
            f"\n🎯 NESTED-CV TUNING ({top1_name}):",
            f"  • Outer Train vs Outer Val   : Train={nested_tr:.4f}, Val={nested_va:.4f} (Gap={gen_gap:+.4f})",
            f"  • Fold Std (outer val)       : {cv_fold_std:.4f}",
            f"  • Tuning trials completed    : {trials_done_total}",
            f"  • Final Hyperparameters      : {final_params}",
            "\n🤝 STACKING (paired, same outer folds):",
            f"  • Stack Mean Score           : {stack_mean:.4f} vs Tuned Single {nested_va:.4f}",
            f"  • KISS Gate Verdict          : {'ADOPTED' if is_stacking_adopted else 'REJECTED'} (required lift >= +{KISS_MIN_LIFT})",
        ])
        if warnings:
            card_lines.append("\n⚠️ WARNINGS:")
            card_lines.extend(f"  • {w}" for w in warnings)
        card_lines.extend([
            "\n💾 SAVED ARTIFACTS:",
            f"  • Champion Model (fitted)    : {champion_path}",
            f"  • Honest OOF Predictions     : {oof_path}",
            "=" * 88,
            "👉 READY FOR PHASE 4: Calibration, DCA Thresholding, Conformal & Explainability!",
            "=" * 88,
        ])

        return TournamentAndTuningReportDTO(
            champion_architecture=champion_arch,
            primary_metric=primary_metric,
            champion_score=round(champion_score, 4),
            baseline_champion_score=round(top1_base_score, 4),
            tuning_lift=round(tuning_lift, 4),
            generalization_gap=round(gen_gap, 4),
            gap_status=gap_status,
            cv_fold_std=round(cv_fold_std, 4),
            best_hyperparameters=final_params,
            is_stacking_adopted=is_stacking_adopted,
            champion_artifact_path=champion_path,
            oof_predictions_path=oof_path,
            tournament_card="\n".join(card_lines),
            validation_protocol=protocol,
            tuning_trials_completed=trials_done_total,
            warnings=warnings,
        )
