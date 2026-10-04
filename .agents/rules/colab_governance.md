# Colab Compute Governance & Automatic Offloading Directive

1. **Compute Routing Policy**:
   - Heavy Machine Learning tasks, parameter tuning, benchmarks, high-cardinality feature engineering, neural network/transformer operations, and long test suites must execute on Google Colab GPU/TPU via `ml_colab_execute`.
   - Never wait for the user to prompt or remind to use Colab. Automatically choose Colab compute when facing non-trivial training or inference loops.

2. **Session Pre-Check**:
   - Query `ml_colab_status` to ensure runtime connectivity.
   - If session is down or unprovisioned, initialize/sync or notify cleanly with exact state.
