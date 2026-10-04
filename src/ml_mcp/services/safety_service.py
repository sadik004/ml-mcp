"""Phase 4 Calibration, Decision Theory, Safety Certification & Drift Service."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ml_mcp.engine.calibrator import ProbabilityCalibrator
from ml_mcp.engine.conformal_risk_control import ConformalRiskControlEngine
from ml_mcp.engine.explainer import TreeShapExplainer
from ml_mcp.engine.fairness_auditor import SliceFairnessAuditor
from ml_mcp.engine.ood_detector import OODDetector
from ml_mcp.engine.pipeline_builder import DefensivePipelineBuilder
from ml_mcp.engine.safety_orchestrator import SafetyOrchestrator
from ml_mcp.engine.stress_tester import ModelStressTester
from ml_mcp.engine.threshold import DecisionThresholdOptimizer
from ml_mcp.schemas.safety import CRCReportDTO
from ml_mcp.services.base import BaseService
from ml_mcp.services.evaluation import build_sealed_pipeline

logger = logging.getLogger(__name__)


class SafetyService(BaseService):
    """Orchestrates model safety certification, probability calibration, DCA thresholding, SHAP, and OOD detection."""

    def certify_safety_and_decisions(
        self,
        csv_path: Optional[str] = None,
        target_column: Optional[str] = None,
        cost_fp: float = 5.0,
        cost_fn: float = 250.0,
        model_path: Optional[str] = None,
        oof_path: Optional[str] = None,
        train_ephemeral: bool = False,
        allow_in_sample_diagnostic: bool = False,
        view: Literal["compact", "detailed"] = "compact",
    ) -> Dict[str, Any]:
        if not csv_path:
            for cand in [
                os.path.join(".artifacts", "processed", "transformed_dataset.csv"),
                os.path.join(os.getcwd(), ".artifacts", "processed", "transformed_dataset.csv"),
            ]:
                if os.path.exists(cand):
                    csv_path = cand
                    break
            if not csv_path:
                raise ValueError("csv_path was not provided and no transformed dataset found at .artifacts/processed/transformed_dataset.csv")

        if not target_column:
            for mf_path in [
                os.path.join(os.path.dirname(os.path.abspath(csv_path)), "feature_metadata.json"),
                os.path.join(".artifacts", "models", "tournament_metadata.json"),
                os.path.join(".artifacts", "processed", "feature_metadata.json"),
                os.path.join(os.getcwd(), ".artifacts", "processed", "feature_metadata.json"),
            ]:
                if self.repository.file_exists(mf_path):
                    try:
                        target_column = self.repository.read_json(mf_path).get("target_column")
                        if target_column:
                            break
                    except Exception as meta_err:
                        logger.warning(f"Failed parsing metadata candidate '{mf_path}': {meta_err}")
        if not target_column:
            raise ValueError("target_column must be specified if not found in feature_metadata.json")

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if not model_path:
            default_model = os.path.join(".artifacts", "models", "champion_model.joblib")
            if self.repository.file_exists(default_model):
                model_path = default_model

        training_mode = "persisted"
        if model_path and self.repository.file_exists(model_path):
            model = self.repository.load_model(model_path)
        elif train_ephemeral:
            from sklearn.ensemble import HistGradientBoostingClassifier
            training_mode = "ephemeral"
            model = HistGradientBoostingClassifier(random_state=self.settings.random_state)
            model.fit(X, y)
        else:
            raise ValueError(
                "Certification Refused: 'model_path' not provided and no persisted model found at .artifacts/models/champion_model.joblib. "
                "Independent certification requires a persisted model artifact. "
                "Pass train_ephemeral=True explicitly if you intend to run an ephemeral development/testing evaluation."
            )

        if not oof_path and not train_ephemeral:
            default_oof = os.path.join(".artifacts", "models", "oof_predictions.npy")
            if self.repository.file_exists(default_oof):
                oof_path = default_oof

        oof_probs = None
        if oof_path and self.repository.file_exists(oof_path):
            try:
                oof_probs = self.repository.load_npy(oof_path)
                if len(oof_probs) != len(y):
                    if not allow_in_sample_diagnostic:
                        return {
                            "certification_status": "refused",
                            "reason": f"Mismatched OOF predictions length: got {len(oof_probs)}, expected {len(y)}",
                        }
                    oof_probs = None
            except Exception as e:
                raise ValueError(f"Corrupt or unreadable out-of-fold predictions file '{oof_path}': {e}")
        elif not allow_in_sample_diagnostic:
            return {
                "certification_status": "refused",
                "reason": (
                    "Certification Refused: Independent certification requires out-of-fold predictions (oof_path). "
                    "In-sample probabilities cannot be certified under ML mathematical rigor."
                ),
            }

        orchestrator = SafetyOrchestrator()
        report = orchestrator.certify_model(
            model=model,
            X=X,
            y=y,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            oof_probs=oof_probs,
            allow_in_sample_diagnostic=allow_in_sample_diagnostic,
            training_mode=training_mode,
        )
        return report.to_compact() if view == "compact" else report.model_dump()

    def calibrate_probabilities(
        self,
        csv_path: str,
        target_column: str,
        model_path: Optional[str] = None,
        model_name: str = "lightgbm",
        method: Optional[Literal["isotonic", "sigmoid", "temperature"]] = None,
        output_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        warnings_list: List[str] = []

        from sklearn.model_selection import train_test_split
        strat = y if len(np.unique(y)) <= 10 else None
        effective_test_size = self.settings.test_size
        if len(X) >= 40 and int(len(X) * effective_test_size) < self.settings.min_calibration_n:
            effective_test_size = max(self.settings.test_size, min(0.5, float(self.settings.min_calibration_n) / len(X)))
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=effective_test_size, random_state=self.settings.random_state, stratify=strat
            )
        except Exception:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=effective_test_size, random_state=self.settings.random_state
            )

        clf = None
        if model_path and self.repository.file_exists(model_path):
            clf = self.repository.load_model(model_path)
            warnings_list.append("user model was cloned and refit on the provided data; returned artifact is not the original model")
        else:
            from ml_mcp.services.estimator_resolver import resolve_classifier
            clf = resolve_classifier(model_name, warnings=warnings_list, strict=False, random_state=self.settings.random_state)

        # Sealed pipeline fitted only on train split (C5: zero full-data preprocessor leakage)
        pipeline = build_sealed_pipeline(clf, X_tr, target_column=target_column)
        pipeline.fit(X_tr, y_tr)

        calibrator = ProbabilityCalibrator()
        report, calibrated_model = calibrator.calibrate(model=pipeline, X=X_val, y=y_val, method=method)

        if output_model_path:
            self.repository.save_artifact(calibrated_model, output_model_path)

        compact = report.to_compact()
        if output_model_path:
            compact["model_path"] = os.path.abspath(output_model_path)
            compact["artifact_path"] = os.path.abspath(output_model_path)

        if "warnings" not in compact:
            compact["warnings"] = []
        for w in warnings_list:
            if w not in compact["warnings"]:
                compact["warnings"].append(w)
        return compact

    def tune_threshold_and_errors(
        self,
        csv_path: str,
        target_column: str,
        beta: float = 1.0,
        criterion: Literal["f_beta", "cost_loss"] = "f_beta",
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
        benefit_tp: float = 0.0,
        benefit_tn: float = 0.0,
        model_path: Optional[str] = None,
        model_name: str = "lightgbm",
    ) -> Dict[str, Any]:
        from sklearn.model_selection import train_test_split

        import ml_mcp.services.evaluation as eval_svc

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        warnings_list: List[str] = []

        if len(X) < 20:
            evaluation_mode = "in_sample"
            warnings_list.append("Sample size n < 20 is too small for held-out evaluation; evaluated in_sample.")
            from ml_mcp.services.estimator_resolver import resolve_classifier
            clf = resolve_classifier(model_name, warnings=warnings_list, strict=False, random_state=self.settings.random_state)
            pipe = build_sealed_pipeline(clf, X, target_column=target_column)
            pipe.fit(X, y)
            p_tr = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X)
            p_te = p_tr
            y_tr, y_te = y, y
        elif model_path and self.repository.file_exists(model_path):
            evaluation_mode = "unknown_provenance"
            warnings_list.append("User model loaded from disk; training data provenance unknown, evaluation_mode marked unknown_provenance.")
            clf = self.repository.load_model(model_path)
            strat = y if len(np.unique(y)) <= 10 else None
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=self.settings.test_size, random_state=self.settings.random_state, stratify=strat
                )
            except Exception:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=self.settings.test_size, random_state=self.settings.random_state
                )
            p_tr = clf.predict_proba(X_tr)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_tr)
            p_te = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_te)
        else:
            evaluation_mode = "held_out_test"
            strat = y if len(np.unique(y)) <= 10 else None
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=self.settings.test_size, random_state=self.settings.random_state, stratify=strat
                )
            except Exception:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=self.settings.test_size, random_state=self.settings.random_state
                )

            from ml_mcp.services.estimator_resolver import resolve_classifier
            clf = resolve_classifier(model_name, warnings=warnings_list, strict=False, random_state=self.settings.random_state)

            # H2: Threshold search MUST run on out-of-fold predictions of the training split
            oof_res = eval_svc.oof_predict_proba(clf, X_tr, y_tr)
            if isinstance(oof_res, np.ndarray):
                p_tr = oof_res[:, 1] if oof_res.ndim > 1 and oof_res.shape[1] == 2 else oof_res
            else:
                # If mocked or non-array returned
                pipe_m = build_sealed_pipeline(clf, X_tr, target_column=target_column)
                pipe_m.fit(X_tr, y_tr)
                p_tr = pipe_m.predict_proba(X_tr)[:, 1] if hasattr(pipe_m, "predict_proba") else pipe_m.predict(X_tr)

            # Fit pipe on train split to predict on held-out test split
            pipe = build_sealed_pipeline(clf, X_tr, target_column=target_column)
            pipe.fit(X_tr, y_tr)
            p_te = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_te)

        optimizer = DecisionThresholdOptimizer()
        report_tr = optimizer.optimize(
            y_true=np.asarray(y_tr),

            y_probas=p_tr,
            beta=beta,
            criterion=criterion,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            benefit_tp=benefit_tp,
            benefit_tn=benefit_tn,
        )
        opt_threshold = report_tr.optimal_threshold

        # Evaluate performance metrics honestly on untouched held-out test predictions
        y_te_arr = np.asarray(y_te, dtype=int)
        y_pred_te = (p_te >= opt_threshold).astype(int)
        tp = int(np.sum((y_te_arr == 1) & (y_pred_te == 1)))
        fp = int(np.sum((y_te_arr == 0) & (y_pred_te == 1)))
        fn = int(np.sum((y_te_arr == 1) & (y_pred_te == 0)))
        tn = int(np.sum((y_te_arr == 0) & (y_pred_te == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        beta_sq = beta ** 2
        denom = (beta_sq * prec) + rec
        honest_f_beta = float(((1 + beta_sq) * prec * rec) / denom) if denom > 0 else 0.0

        opt_cost = optimizer.calculate_financial_loss(tp, fp, fn, tn, cost_fp, cost_fn, benefit_tp, benefit_tn)
        default_pred_te = (p_te >= 0.50).astype(int)
        tp_def = int(np.sum((y_te_arr == 1) & (default_pred_te == 1)))
        fp_def = int(np.sum((y_te_arr == 0) & (default_pred_te == 1)))
        fn_def = int(np.sum((y_te_arr == 1) & (default_pred_te == 0)))
        tn_def = int(np.sum((y_te_arr == 0) & (default_pred_te == 0)))
        def_cost = optimizer.calculate_financial_loss(tp_def, fp_def, fn_def, tn_def, cost_fp, cost_fn, benefit_tp, benefit_tn)

        compact_res = report_tr.to_compact()
        compact_res["f_beta_score"] = round(honest_f_beta, 4)
        compact_res["precision"] = round(prec, 4)
        compact_res["recall"] = round(rec, 4)
        compact_res["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        compact_res["total_cost_optimal"] = round(opt_cost, 2)
        compact_res["total_cost_default"] = round(def_cost, 2)
        compact_res["cost_savings"] = round(def_cost - opt_cost, 2)
        compact_res["evaluation_mode"] = evaluation_mode
        compact_res["warnings"] = warnings_list
        compact_res["oof_proba"] = p_tr.tolist() if hasattr(p_tr, "tolist") else list(p_tr)
        return compact_res

    def explain_predictions(
        self,
        csv_path: str,
        target_column: str,
        top_k: int = 10,
        instance_index: Optional[int] = None,
        model_path: Optional[str] = None,
        model_name: str = "lightgbm",
    ) -> Dict[str, Any]:

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        has_non_numeric = any(X[col].dtype == "object" or isinstance(X[col].dtype, pd.StringDtype) or X[col].isnull().any() for col in X.columns)
        if has_non_numeric:
            builder = DefensivePipelineBuilder()
            pipe = builder.build_pipeline(df, target_column=target_column)
            pipe.fit(X, y)
            X_trans = pipe.transform(X)
            X = pd.DataFrame(X_trans, columns=[f"f_{i}" for i in range(X_trans.shape[1])], index=X.index)

        warnings_list: List[str] = []
        clf = None
        if model_path and self.repository.file_exists(model_path):
            clf = self.repository.load_model(model_path)
        else:
            from ml_mcp.services.estimator_resolver import resolve_classifier
            clf = resolve_classifier(model_name, warnings=warnings_list, strict=False, random_state=self.settings.random_state)
            clf.fit(X, y)

        explainer = TreeShapExplainer()
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f_{i}" for i in range(X.shape[1])]

        if instance_index is not None:
            row_slice = X.iloc[[instance_index]] if hasattr(X, "iloc") else X[[instance_index]]
            res = explainer.explain_instance(clf, row_slice, background_X=X, feature_names=feature_names, top_k=top_k)
        else:
            res = explainer.explain(clf, X, feature_names=feature_names, top_k=top_k)
        if isinstance(res, dict):
            res["explains"] = "fitted_model"
            res["warnings"] = warnings_list
        return res

    def detect_ood(
        self,
        train_csv_path: str,
        test_csv_path: str,
        method: str = "isolation_forest",
    ) -> Dict[str, Any]:
        df_train = self.repository.load_dataframe(train_csv_path).select_dtypes(include=[np.number])
        df_test = self.repository.load_dataframe(test_csv_path).select_dtypes(include=[np.number])
        detector = OODDetector(method=method)
        detector.fit(df_train)
        report = detector.detect(df_test)
        return report.to_compact()

    def stress_test_and_fairness(
        self,
        csv_path: str,
        target_column: str,
        protected_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = self.repository.load_dataframe(csv_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split into 70% train and 30% held-out test to evaluate honest baseline and stress degradation
        strat = y if len(np.unique(y)) <= 10 else None
        try:
            X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
                X, y, df, test_size=self.settings.test_size, random_state=self.settings.random_state, stratify=strat
            )
        except Exception:
            X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
                X, y, df, test_size=self.settings.test_size, random_state=self.settings.random_state
            )

        base_clf = RandomForestClassifier(n_estimators=15, random_state=self.settings.random_state)
        pipeline = build_sealed_pipeline(base_clf, X_train, target_column=target_column)
        pipeline.fit(X_train, y_train)

        tester = ModelStressTester()
        stress_report = tester.evaluate(pipeline, X_test, y_test)

        fairness_dict = {}
        if protected_column and protected_column in df.columns:
            auditor = SliceFairnessAuditor()
            f_report = auditor.audit(pipeline, X_test, y_test, protected_series=df_test[protected_column], protected_attribute=protected_column)
            fairness_dict = f_report.to_compact()

        return {
            "baseline_score": stress_report.baseline_score,
            "stress_test": stress_report.to_compact(),
            "slice_fairness": fairness_dict,
        }

    def conformal_risk_control(
        self,
        csv_path: str,
        target_column: str,
        loss_type: Literal["misclassification", "fnr", "asymmetric_cost"] = "misclassification",
        target_risk: float = 0.05,
        test_size: float = 0.3,
        cal_fraction: Optional[float] = None,
        mondrian: bool = False,
    ) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = self.repository.load_dataframe(csv_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV.")

        raw_features = df.drop(columns=[target_column])
        numeric_cols = raw_features.select_dtypes(include=[np.number]).columns.tolist()
        dropped_cols = [c for c in raw_features.columns if c not in numeric_cols]
        warnings_list: List[str] = []
        if dropped_cols:
            warnings_list.append(f"Non-numeric columns dropped for conformal risk control: {dropped_cols}")

        X = raw_features[numeric_cols].fillna(0)
        if X.shape[1] == 0:
            X = pd.get_dummies(raw_features, drop_first=True)

        y_raw = df[target_column]
        _, y = np.unique(y_raw, return_inverse=True)

        effective_cal_frac = cal_fraction if cal_fraction is not None else self.settings.cal_fraction
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size or self.settings.test_size, random_state=self.settings.random_state
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=effective_cal_frac, random_state=self.settings.random_state
        )

        clf = RandomForestClassifier(n_estimators=30, random_state=self.settings.random_state)
        clf.fit(X_train, y_train)

        probs_cal = clf.predict_proba(X_cal)
        probs_test = clf.predict_proba(X_test)

        crc = ConformalRiskControlEngine()
        calibrated_lambda, _ = crc.calibrate(
            probs_cal=probs_cal,
            y_cal=y_cal,
            loss_type=loss_type,
            target_risk=target_risk,
            mondrian=mondrian,
        )

        psets_test, triage_records = crc.predict_and_triage(probs_test, calibrated_lambda)
        test_losses = crc.evaluate_loss(psets_test, y_test, loss_type=loss_type)
        if loss_type == "fnr":
            pos_mask = (y_test == 1)
            if np.any(pos_mask):
                eval_losses = test_losses[pos_mask]
                empirical_risk = float(np.mean(eval_losses))
                n_eval = int(np.sum(pos_mask))
                k_loss = float(np.sum(eval_losses))
            else:
                empirical_risk = None
                n_eval = 0
                k_loss = 0.0
                warnings_list.append(
                    "FNR evaluation undefined: test set contains 0 positive instances (0/0 is mathematically undefined)."
                )
        else:
            empirical_risk = float(np.mean(test_losses)) if len(test_losses) > 0 else None
            n_eval = len(test_losses)
            k_loss = float(np.sum(test_losses)) if len(test_losses) > 0 else 0.0

        min_n = self.settings.min_calibration_n
        coverage_ci_low: Optional[float] = None
        coverage_ci_high: Optional[float] = None

        if len(y_cal) < min_n:
            empirical_risk = None
            holdout_risk_ucb = None
            holdout_check_passed = None
            warnings_list.append(
                f"INSUFFICIENT_N: calibration samples ({len(y_cal)}) below minimum calibration threshold ({min_n})."
            )
        elif empirical_risk is not None and n_eval > 0:
            from scipy.stats import beta
            k_clip = min(float(n_eval), max(0.0, float(k_loss)))
            if k_clip >= n_eval:
                holdout_risk_ucb = 1.0
            elif k_clip <= 0.0:
                holdout_risk_ucb = float(1.0 - (1.0 - 0.95) ** (1.0 / n_eval))
            else:
                holdout_risk_ucb = float(beta.ppf(0.95, k_clip + 1, n_eval - k_clip))
            holdout_check_passed = bool(holdout_risk_ucb <= target_risk)

            from ml_mcp.engine.stats import wilson_interval
            k_covered = int(max(0, n_eval - int(k_loss)))
            coverage_ci_low, coverage_ci_high = wilson_interval(k_covered, n_eval, conf=0.95, min_n=min_n // 2)
        else:
            holdout_risk_ucb = None
            holdout_check_passed = None

        guarantee_satisfied = bool(empirical_risk is not None and empirical_risk <= target_risk)

        set_sizes = [len(s) for s in psets_test]
        avg_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0
        ambiguity_count = sum(1 for s in set_sizes if s > 1)
        empty_count = sum(1 for s in set_sizes if s == 0)
        triage_count = sum(1 for r in triage_records if r["needs_human_review"])

        report = CRCReportDTO(
            loss_function=loss_type,
            target_risk=float(target_risk),
            empirical_risk=empirical_risk,
            calibrated_lambda=float(calibrated_lambda) if isinstance(calibrated_lambda, (int, float)) else 0.0,
            guarantee_satisfied=guarantee_satisfied,
            total_cal_samples=len(y_cal),
            average_set_size=float(avg_set_size),
            ambiguity_rate=float(ambiguity_count / len(y_test)) if len(y_test) > 0 else 0.0,
            empty_set_rate=float(empty_count / len(y_test)) if len(y_test) > 0 else 0.0,
            human_triage_count=int(triage_count),
            mondrian_conditional=mondrian,
            per_class_thresholds={str(k): round(v, 4) for k, v in calibrated_lambda.items()} if isinstance(calibrated_lambda, dict) else None,
            coverage_ci_low=coverage_ci_low,
            coverage_ci_high=coverage_ci_high,
            holdout_risk_ucb=holdout_risk_ucb,
            holdout_check_passed=holdout_check_passed,
            warnings=warnings_list,
        )

        compact = report.to_compact()
        compact["triage_samples_preview"] = triage_records[:5]
        return compact

