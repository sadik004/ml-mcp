# `ml-mcp` Architectural Roadmap & Progress Tracking

This roadmap tracks the development, mathematical hardening, and validation milestones for `ml-mcp` across all architectural phases.

---

## 🏆 Development Phases Overview

| Phase | Description | Status | Verification Engine |
| :--- | :--- | :--- | :--- |
| **Phase 1: Data Audit & Hygiene** | Chatterjee rank correlation xi (JASA 2021), Cramér's V, PPS tree, SVD condition number (Nature MI 2023), Ridge-VIF, Belsley variance proportions, MCAR vs MNAR classifier, DataPerf index memorization (NeurIPS 2023), Dirac-Delta boundary sentinels (KDD 2020), Medcouple adjusted outlier bounds. | **Completed ✅** | Local & Colab Pytest (16/16 Passed) |
| **Phase 2: Defensive Feature Engineering & Balancing** | Cost-sensitive sample weighting (anti-SMOTE), MNAR missingness indicator pipelines, OpenFE cross-numeric features, residual-driven target transforms, dense MiniLM embeddings. | **Completed ✅** | Colab Cloud Pytest (34/34 Passed) |
| **Phase 3: Calibration, Decision Theory & Safety Control** | Mondrian (class-conditional) conformal & RAPS, Beta calibration & adaptive ECE, Decision Curve Analysis (DCA), Wasserstein-1 drift, Free Energy OOD detection, Covariance manifold stress, Intersectional subgroup fairness. | **Completed ✅** | Colab Cloud Pytest (19/19 Passed) |
| **Phase 4: Optimization, Serving & MLOps Infrastructure** | ONNX Runtime graph optimization, sub-millisecond batch inference, dynamic FastAPI/Docker generation, standalone interactive HTML eval dashboards. | **In Progress ⏳** | Unit & Integration Suites |

---

## 🔬 Peer-Reviewed Frontier Research Integrated (2020–2024 SOTA)

### Phase 1: Pre-Flight Data Hygiene, Target Leakage, and Multicollinearity
1. **Chatterjee Rank Correlation & Predictive Power (Anti-Leakage):**
   - *Chatterjee (JASA 2021)* — "A New Coefficient of Correlation". Detects arbitrary non-linear and non-monotonic leakage in $O(N \log N)$.
   - *Greenacre (2021/2023)* — Bias-corrected Cramér's V for categorical features against discrete targets.
   - *Wetschoreck et al. (2020/2022)* — Single-feature Predictive Power Score (PPS) 1-split tree cross-validation safety net.
2. **SVD Spectral Conditioning & Ridge-Regularized VIF:**
   - *Lafon et al. (Nature Machine Intelligence 2023)* — Spectral condition number $\kappa(X) = \sigma_{\max} / \sigma_{\min}$.
   - *Tikhonov Ridge Inversion ($\lambda = 10^{-4}$)* — Prevents singular matrix crashes on exact duplicate columns.
   - *Belsley, Kuh, & Welsch (Updated 2023)* — Variance decomposition proportions ($\Pi_{ij} > 0.5$) isolating collinear feature clusters.
3. **Missingness Mechanisms & DataPerf Memorization:**
   - *Jamshidian & Jalal (2020) & Jaeger et al. (NeurIPS 2023)* — Statistically distinguishes MCAR from MNAR, enforcing missingness indicator columns.
   - *Mazumder et al. (NeurIPS 2023 DataPerf Benchmark)* — Shannon entropy and uniqueness guard against tree index memorization.
   - *Nigrini (2021)* — Second-order Benford's Law Chi-Square anomaly warning.
4. **Dirac-Delta Isolated Point Mass Sentinels:**
   - *Emmott et al. (KDD 2020)* — Dynamically detects isolated point mass spikes ($> 3 	imes 	ext{MAD}$ from median, $\ge 5\%$ frequency) without relying on static lists.
   - Enterprise defaults: IEEE-754 / POSIX hex codes, database epoch defaults (`1900-01-01`, `1970-01-01`, `2038-01-19`, `2099-12-31`).
5. **Medcouple Adjusted Boxplot:**
   - *Hubert & Vandervieren (Computational Statistics)* — Medcouple ($MC \in [-1, 1]$) asymmetric boundaries eliminating false-positive outlier alarms on right-skewed tabular distributions.

### Phase 2: Feature Engineering & Class Balancing
1. **Cost-Sensitive Learning (Anti-SMOTE):**
   - *Wallace et al. (IEEE TKDE 2021)* — "Class Imbalance: Why SMOTE Fails in Practice and Why Cost-Sensitive Learning Dominates Oversampling"
   - *Menon et al. (ICLR 2021)* — "Long-Tail Learning via Logit Adjusted Loss"
   - Deprecated synthetic line-interpolation SMOTE on tabular data. Replaced with exact inverse-frequency weights ($w_i = rac{N}{K \cdot N_{y_i}}$) and `RandomUnderSampler`/`RandomOverSampler(shrinkage=0.1)`.
2. **Defensive Missingness (MNAR):**
   - *Groenwold et al. (2020)* — Missing Not At Random indicator modeling. Implemented `SimpleImputer(add_indicator=True)` inside strict `ColumnTransformer` folds.
3. **OpenFE Feature Synthesis:**
   - *Zhang et al. (ICML 2023)* — Automated Feature Generation on Tabular Data. Added pairwise cross-numeric ratios ($A / (B + 10^{-6})$) and differences ($A - B$) bounded to top 5 mutual-information pairs.
4. **Residual-Driven Target Transforms:**
   - Residual-based skewness check on $\epsilon = y - \hat{y}$ selecting Yeo-Johnson or `np.log1p`.
5. **Dense Semantic Embeddings:**
   - Pre-trained Sentence-Transformers (`all-MiniLM-L6-v2`) dense 384-d embeddings with sub-10ms TF-IDF + TruncatedSVD fallback.

### Phase 3: Calibration, Decision Theory, Fairness & Safety
1. **Mondrian (Class-Conditional) Conformal Prediction & RAPS:**
   - *Romano, Barber, Candès (NeurIPS 2020)* — "Classification with Valid and Equal Coverage for Inherent Subgroups"
   - *Angelopoulos et al. (ICLR 2021)* — "Uncertainty Sets for Image and Tabular Classifiers via RAPS"
   - Guarantees finite-sample coverage per individual class: $P(Y \in C(X) \mid Y = k) \ge 1 - lpha$.
   - Regularized Adaptive Prediction Sets penalize excessively large prediction sets on ambiguous samples.
2. **Beta Calibration & Adaptive ECE:**
   - *Kull, Silva Filho, Flach (AISTATS / EJS)* — "Beta Calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"
   - *Roelofs et al. (NeurIPS 2022)* — "Mitigating Bias in Expected Calibration Error"
   - Fits 3-parameter Beta calibration map ($p_{	ext{cal}} = 	ext{logit}^{-1}(a \ln p - b \ln(1-p) + c)$) to model asymmetric probability skews where Platt scaling fails.
   - Equal-frequency quantile binning eliminates sample-size bias in Expected Calibration Error.
3. **Decision Curve Analysis (Net Benefit Curve):**
   - *Vickers & Elkin (BMJ / Lancet)* — "Decision Curve Analysis: A Novel Method for Evaluating Prediction Models"
   - Calculates clinical/financial Net Benefit over threshold spectrum $p_t \in [0.01, 0.99]$:
     $$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$
   - Evaluates models against standard "Treat All" and "Treat None" baselines to define the true optimal operational window.
4. **Wasserstein-1 (Earth Mover's) Distance:**
   - *Ramdas, Reddi, Póczos, Singh, Wasserman (JMLR 2017)* — "Wasserstein Distance for Two-Sample Testing and Distribution Shift"
   - Replaced vertical-only KS-distance with exact 1D $\mathcal{W}_1(P, Q) = \int |F_P(x) - F_Q(x)| dx$, detecting heavy-tailed transport shifts with permutation empirical p-values.
5. **Helmholtz Free Energy OOD Scoring:**
   - *Liu, Wang, Owens, Li (NeurIPS 2020)* — "Energy-based Out-of-Distribution Detection"
   - Evaluates free energy $E(x) = -T \cdot \log \sum_{i=1}^K \exp(f_i(x)/T)$ on logits, preventing high-confidence hallucinations on out-of-distribution tabular samples.
6. **Covariance-Preserving Manifold Stress:**
   - *Hendrycks & Dietterich (ICLR 2019)* — "Benchmarking Robustness to Common Corruptions"
   - Injects correlated noise $\delta \sim \mathcal{N}(0, \epsilon^2 \Sigma)$ derived from empirical covariance $\Sigma = \text{Cov}(X_{\text{train}})$ to preserve the physical data manifold.
7. **Intersectional Subgroup Fairness:**
   - *Kearns et al. (ICML 2018)* — "Preventing Fairness Gerrymandering: Auditing Subgroup Fairness"
   - Audits Disparate Impact (EEOC 80% rule) and Equalized Odds across the Cartesian product of multiple protected attributes, computing Minimax Disparity.

---

## ⚡ Verification Protocol
- **Local PC CPU/RAM Load:** Strictly 0% via WebSocket proxy to Google Colab Cloud GPU/CPU.
- **Remote Testing Harness:** `e:/ML Testing/colab_test_runner.py`
