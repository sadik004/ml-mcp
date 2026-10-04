# Workspace Operational Directives & Compute Governance: ml-mcp

## ⚡ 1. Colab Compute-First Protocol (Strictly Enforced)
- **Zero Local Compute Waste**: For any heavy ML workloads—including model training, hyperparameter sweeps (Optuna), large dataset benchmarks, feature synthesis, text embeddings, or extensive verification runs—**DO NOT strain the local host CPU/RAM**.
- **Proactive Cloud Offloading**: Proactively utilize Google Colab compute power via `ml_colab_execute`, `ml_colab_upload`, `ml_colab_download`, and `ml_colab_status` (or `ColabCloudRunner`).
- **Never Wait for User Reminders**: The assistant must never require repeated reminders from the user to leverage Colab compute power. Always prioritize and route heavy computational jobs to the active Colab runtime (GPU/TPU) automatically.

---

## 🔬 2. Machine Learning Mathematical Rigor ("জিরো ফ্রড ইঞ্জিনিয়ারিং")
Every ML workflow must adhere strictly to mathematical invariants:
1. **Zero Fake Statistical Guarantees**: Empirical coverage $P(y \in C(x))$ and singleton sets must be computed from untouched holdouts, never hardcoded.
2. **True Metric Binning**: Expected Calibration Error (ECE) must use actual reliability diagram binning ($\sum |B_m|/N |\text{acc} - \text{conf}|$). Never conflate Brier score with ECE.
3. **Strict Out-of-Fold Cross-Validation**: Nested CV is mandatory. Preprocessors, synthesizers, and feature selectors must be fit strictly within training folds with zero leakage.
4. **Zero Silent Fallbacks**: Ensemble artifacts must be fully fitted Scikit-Learn estimators. No silent exception swallowing (`except Exception: pass`) with dummy fallbacks.

---

## 🛠️ 3. Execution & Verification Invariants
- Verify Colab session status proactively using `ml_colab_status` before launching compute jobs.
- If a remote job finishes, sync back necessary checkpoints or artifacts cleanly to the workspace.
- Enforce clean 3-tier architecture inside `src/ml_mcp/` and maintain 100% test passing gates.
