# `ml_check_collinearity` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Lafon et al. (Nature Machine Intelligence 2023) — *"Spectral Condition Number and Singular Value Thresholding"*
> - Belsley, Kuh, & Welsch (Updated 2023) — *"Variance Decomposition Proportions"*
> - Tikhonov SVD Regularization ($\lambda = 10^{-4}$)

---

## 1. SVD Spectral Condition Number $\kappa(X)$
Computes $\kappa(X) = \sigma_{\max} / \sigma_{\min}$ via Singular Value Decomposition of standardized features.
- $\kappa(X) > 30.0$: Moderate-to-severe multicollinearity across multiple features.
- $\kappa(X) > 100.0$: Catastrophic matrix ill-conditioning.

---

## 2. Ridge-Regularized SVD-VIF
Classical VIF inverts the correlation matrix $(X^T X)^{-1}$. When exact duplicate columns or near-singular features exist, classical VIF crashes.
`ml-mcp` implements regularized inverse:
$$\text{VIF}_j = \left[ (R + \lambda I)^{-1} \right]_{jj} \quad (\lambda = 10^{-4})$$
Iterative competitive drop rule prunes the feature with the lowest target correlation until all remaining features satisfy $\text{VIF} \le 10.0$ and $\kappa(X) \le 30.0$.

---

## 3. Belsley Variance Decomposition Proportions
Decomposes parameter variances across singular values:
$$\Pi_{jk} = \frac{\phi_{jk}}{\phi_j} \quad \text{where } \phi_{jk} = \frac{v_{jk}^2}{\sigma_k^2}$$
Identifies collinear groups where condition index $\mu_k > 30.0$ accounts for variance proportion $\Pi_{jk} > 0.50$ across two or more features.
