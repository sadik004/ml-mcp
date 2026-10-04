# Workspace Operational Directives & Compute Governance: ml-mcp

## ⚡ 1. Colab Compute-First Protocol (Strictly Enforced)
- **Zero Local Compute Waste**: For any heavy ML workloads—including model training, hyperparameter sweeps (Optuna), large dataset benchmarks, feature synthesis, text embeddings, or extensive verification runs—**DO NOT strain the local host CPU/RAM**.
- **Proactive Cloud Offloading**: Proactively utilize Google Colab compute power via `ml_colab_execute`, `ml_colab_upload`, `ml_colab_download`, and `ml_colab_status` (or `ColabCloudRunner`).
- **Never Wait for User Reminders**: The assistant must never require repeated reminders from the user to leverage Colab compute power. Always prioritize and route heavy computational jobs to the active Colab runtime (GPU/TPU) automatically.

---

## 🔒 2. Adversarial Anti-Gaming & Test Integrity Harness (Permanent Law)
The assistant must operate strictly under the 6-Ring Adversarial Harness at all times:
1. **Ring 1: Cryptographic Test Lock (`LOCK.json`)**:
   - Verification test files in `tests/verification/` are sealed with SHA-256 hashes.
   - The assistant is strictly prohibited from modifying, weakening, deleting, or bypassing any locked test.
2. **Ring 2: AST Anti-Gaming Audit (`check_test_integrity.py`)**:
   - Never mock the function or class under test to return fixed/canned answers (Rule T4).
   - Never write tautological assertions (`assert True`, `assert 1 == 1`).
   - Gate G0 must execute with exit code 0 before declaring any task complete.
3. **Ring 3: Independent Mathematical Oracles**:
   - Never verify code using its own helper functions.
   - All tests for statistical metrics (ECE, Brier, NLL, CRC loss) must assert against independent, pure-math formulas written directly in the test suite.
4. **Ring 4: Adversarial Permutation Stress Gate**:
   - Out-of-fold and holdout evaluation metrics under shuffled/permuted labels must collapse to chance level ($\text{AUC} \le 0.60, R^2 \le 0.05$).
   - Any model maintaining high scores under permuted labels is immediately flagged as in-sample leakage fraud.
5. **Ring 5: Defensive Sealed Pipelines**:
   - Zero preprocessor leakage: `StandardScaler`, `SimpleImputer`, `OneHotEncoder` must NEVER be fit on the full dataset before cross-validation.
   - All models must be encapsulated in sealed Scikit-Learn `Pipeline` objects fitting exclusively inside training splits.
6. **Ring 6: Multi-Gate Machine Quality Scorecard (G0–G10)**:
   - All 11 quality gates must pass cleanly with exit code 0. Zero tolerance for unverified claims.

---

## 🔬 3. Machine Learning Mathematical Rigor ("জিরো ফ্রড ইঞ্জিনিয়ারিং")
Every ML workflow must adhere strictly to mathematical invariants:
1. **Zero Fake Statistical Guarantees**: Empirical coverage $P(y \in C(x))$ and singleton sets must be computed from untouched holdouts, never hardcoded.
2. **True Metric Binning**: Expected Calibration Error (ECE) must use actual reliability diagram binning ($\sum |B_m|/N |\text{acc} - \text{conf}|$). Never conflate Brier score with ECE.
3. **Strict Out-of-Fold Cross-Validation**: Nested CV is mandatory. Preprocessors, synthesizers, and feature selectors must be fit strictly within training folds with zero leakage.
4. **Zero Silent Fallbacks**: Ensemble artifacts must be fully fitted Scikit-Learn estimators. No silent exception swallowing (`except Exception: pass`) with dummy fallbacks.

---

## 🛠️ 4. Execution & Verification Invariants
- Verify Colab session status proactively using `ml_colab_status` before launching compute jobs.
- If a remote job finishes, sync back necessary checkpoints or artifacts cleanly to the workspace.
- Enforce clean 3-tier architecture inside `src/ml_mcp/` and maintain 100% test passing gates.
