# `ml_detect_target_leakage` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Chatterjee (Journal of the American Statistical Association - JASA 2021) — *"A New Coefficient of Correlation"*
> - Greenacre (Springer 2021/2023) — *"Bias-Corrected Cramér's V"*
> - Wetschoreck et al. (2020/2022) — *"Predictive Power Score (PPS)"*

---

## 1. Chatterjee's Non-Parametric Rank Correlation $\xi_n(X, Y) \in [0, 1]$
Classical Pearson and Spearman correlations assume linear or monotonic relationships. If a feature has a non-monotonic functional dependency on the target (e.g. $Y = X^2$ or $Y = \sin(X)$), Pearson correlation is approximately $0.0$, allowing severe data leakage to bypass undetected.

`ml-mcp` implements **Chatterjee's Correlation $\xi_n$**:
$$\xi_n(X, Y) = 1 - \frac{3 \sum_{i=1}^{n-1} |r_{i+1} - r_i|}{n^2 - 1}$$
where $r_i$ is the rank of $Y$ after ordering by $X$.
- $\xi_n \to 1$ if and only if $Y = f(X)$ for some measurable non-constant function.
- $\xi_n \to 0$ if and only if $X$ and $Y$ are independent.
- Runs in $O(N \log N)$ time with zero memory explosion.
- Flags $\xi_n \ge 0.85$ as critical target leakage.

---

## 2. Bias-Corrected Cramér's V for Categorical Predictors
Categorical features and discrete target variables are evaluated via Greenacre's unbiased Cramér's $\tilde{V}$, applying Yates shrinkage to eliminate sample-size inflation on high-cardinality splits. Flags $\tilde{V} \ge 0.90$ as post-event leakage.

---

## 3. Predictive Power Score (PPS) Tree Safety Net
For borderline or high-signal features, a fast 1-split Decision Tree is cross-validated against the target. Out-of-fold scores $\ge 0.98$ trigger an automated non-linear leakage alarm.
