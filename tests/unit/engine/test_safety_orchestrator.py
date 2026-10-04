import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from ml_mcp.engine.safety_orchestrator import SafetyOrchestrator


def test_safety_orchestrator_refuses_without_oof_probs():
    """Decision 1 Gate: Certification MUST be refused if oof_probs is missing without allow_in_sample_diagnostic."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
    y = pd.Series((X['f1'] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    with pytest.raises(ValueError, match="Certification Refused: Independent certification requires out-of-fold predictions"):
        orchestrator.certify_model(model=clf, X=X, y=y)


def test_safety_orchestrator_allows_in_sample_with_explicit_flag():
    """Decision 1 Gate: If allow_in_sample_diagnostic=True, metrics must be labeled in_sample=True and status=refused."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
    y = pd.Series((X['f1'] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    cert = orchestrator.certify_model(model=clf, X=X, y=y, allow_in_sample_diagnostic=True)

    assert cert.in_sample is True
    assert cert.certification_status == "refused"
    assert "DIAGNOSTIC IN-SAMPLE (UNCERTIFIED)" in cert.safety_card
    assert any("DIAGNOSTIC ONLY" in w for w in cert.warnings)


def test_safety_orchestrator_execution_with_oof_probs():
    """Honest independent certification with out-of-fold probabilities."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.uniform(0, 10, n),
    })
    y = pd.Series((X['f1'] + 0.5 * X['f2'] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    oof_probs = cross_val_predict(clf, X, y, cv=3, method="predict_proba")
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    cert = orchestrator.certify_model(
        model=clf,
        X=X,
        y=y,
        oof_probs=oof_probs,
        cost_fp=5.0,
        cost_fn=250.0,
        training_mode="persisted",
    )

    assert cert.in_sample is False
    assert cert.certification_status == "certified"
    assert cert.optimal_threshold > 0.0
    assert cert.optimal_threshold <= 1.0
    assert cert.optimal_dollar_loss <= cert.naive_50_dollar_loss or cert.net_dollar_savings >= 0.0
    assert len(cert.top_shap_drivers) > 0
    assert 'SAFETY, CALIBRATION & DECISION CERTIFICATE' in cert.safety_card
    assert 'OUT-OF-FOLD INDEPENDENT CERTIFICATION' in cert.safety_card


def test_safety_orchestrator_conformal_failure_regression():
    from unittest.mock import patch
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
    y = pd.Series((X['f1'] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    oof_probs = np.random.uniform(0.1, 0.9, size=(n, 2))
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    with patch.object(orchestrator.conformal_engine, "get_prediction_sets", side_effect=RuntimeError("Mock conformal error")):
        cert = orchestrator.certify_model(model=clf, X=X, y=y, oof_probs=oof_probs)

    # Must NOT return hardcoded 95.0 / 90.0
    assert cert.conformal_coverage_pct is None
    assert cert.conformal_singleton_pct is None
    assert "FAILED / UNCALIBRATED" in cert.safety_card
    assert any("Conformal evaluation failed: Mock conformal error" in w for w in cert.warnings)


def test_safety_orchestrator_ood_failure_regression():
    from unittest.mock import patch
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
    y = pd.Series((X['f1'] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    oof_probs = np.random.uniform(0.1, 0.9, size=(n, 2))
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    with patch.object(orchestrator.ood_engine, "fit", side_effect=ValueError("Mock OOD error")):
        cert = orchestrator.certify_model(model=clf, X=X, y=y, oof_probs=oof_probs)

    # Must NOT return dummy -0.5
    assert cert.ood_cutoff_boundary is None
    assert "FAILED / UNAVAILABLE" in cert.safety_card
    assert any("OOD detector failed: Mock OOD error" in w for w in cert.warnings)
