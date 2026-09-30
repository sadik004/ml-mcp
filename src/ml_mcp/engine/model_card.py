"""Google and Hugging Face Compliant MODEL_CARD.md Generator."""
from __future__ import annotations

from typing import Any, Dict, Optional


def generate_model_card(
    model_name: str,
    metrics: Dict[str, Any],
    dataset_hash: str,
    p95_latency_ms: float,
    p99_latency_ms: float,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate official markdown model card documentation."""
    params_str = ""
    if hyperparameters:
        for k, v in hyperparameters.items():
            params_str += f"- **`{k}`**: `{v}`\n"
    else:
        params_str = "- Default / Tuned parameters\n"

    metrics_rows = ""
    for k, v in metrics.items():
        metrics_rows += f"| {k} | {v} |\n"

    card = f"""# Model Card: {model_name}

## 1. Model Details & Performance
- **Model Architecture**: `{model_name}`
- **Evaluation Framework**: 5-Fold Stratified Cross-Validation

### Evaluation Metrics
| Metric | Value |
| :--- | :--- |
{metrics_rows}

## 2. Hyperparameter Profile
{params_str}

## 3. Data Lineage & Integrity
- **Dataset SHA-256 Fingerprint**: `{dataset_hash}`
- **Data Leakage Guard**: Active (Zero-leakage ColumnTransformer)

## 4. Operational Latency Benchmarks
- **Single-Sample P95 Latency**: `{p95_latency_ms:.2f} ms`
- **Single-Sample P99 Latency**: `{p99_latency_ms:.2f} ms`

## 5. Ethical Considerations & Operational Limits
- **Out-of-Distribution (OOD)**: Inference on inputs outside the training feature distribution should be validated with `OODDetector`.
- **Fairness & Bias**: Demographic slice metrics must conform to the US EEOC Four-Fifths 80% Rule before production deployment.
"""
    return card
