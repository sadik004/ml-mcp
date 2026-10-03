import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml_mcp.engine.safety_orchestrator import SafetyOrchestrator

def test_safety_orchestrator_execution():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.uniform(0, 10, n),
    })
    y = pd.Series((X['f1'] + 0.5 * X['f2'] > 0).astype(int))

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    orchestrator = SafetyOrchestrator()
    cert = orchestrator.certify_model(
        model=clf,
        X=X,
        y=y,
        cost_fp=5.0,
        cost_fn=250.0
    )

    assert cert.optimal_threshold > 0.0
    assert cert.optimal_threshold <= 1.0
    assert cert.optimal_dollar_loss <= cert.naive_50_dollar_loss or cert.net_dollar_savings >= 0.0
    assert len(cert.top_shap_drivers) > 0
    assert 'SAFETY, CALIBRATION & DECISION CERTIFICATE' in cert.safety_card
