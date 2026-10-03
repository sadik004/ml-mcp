# `ml_stress_test_and_fairness` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Hendrycks & Dietterich (ICLR 2019) — *"Benchmarking Robustness to Common Corruptions"*
> - Kearns et al. (ICML 2018) — *"Preventing Fairness Gerrymandering: Auditing Subgroup Fairness"*
> - Hardt et al. (NeurIPS 2016) — *"Equality of Opportunity in Supervised Learning"*

---

## 1. Covariance-Preserving Manifold Stress Testing
Injecting independent uniform noise into tabular features destroys natural physical correlations (e.g., generating records with `age=18, tenure=25` or `car_speed=120, engine_rpm=500`).

`ml-mcp` estimates the empirical covariance matrix $\Sigma = \text{Cov}(X_{\text{train}})$ and injects correlated multivariate Gaussian noise:
$$\delta \sim \mathcal{N}\left(0, \epsilon^2 \cdot \Sigma\right)$$
This ensures perturbations lie directly on the real-world data manifold, accurately stress-testing decision boundary stability.

---

## 2. Intersectional Subgroup Fairness
Auditing single protected attributes independently leaves models vulnerable to **Fairness Gerrymandering**, where a model appears fair when examining Gender alone or Age alone, but severely discriminates against intersectional minority groups (e.g., `Gender=Female & AgeBracket=Elderly`).

`ml-mcp` audits the full Cartesian product across protected attributes:
- **Disparate Impact (EEOC 80% Rule):**
  $$\text{DIR} = \frac{\min_{g} P(\hat{Y}=1 \mid G=g)}{\max_{g} P(\hat{Y}=1 \mid G=g)}$$
- **Equalized Odds Disparity:** Auditing True Positive Rate (TPR) and False Positive Rate (FPR) parity across all subgroups.
- **Minimax Disparity:** Identifying the worst-off intersectional subgroup to prevent systemic demographic harm.
