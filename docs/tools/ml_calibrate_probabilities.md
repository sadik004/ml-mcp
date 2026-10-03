# `ml_calibrate_probabilities` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Kull, Silva Filho, Flach (AISTATS / Electronic Journal of Statistics) — *"Beta Calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"*
> - Roelofs et al. (NeurIPS 2022) — *"Mitigating Bias in Expected Calibration Error"*
> - Guo et al. (ICML 2017) — *"On Calibration of Modern Neural Networks"*

---

## 1. Beta Calibration for Skewed Tabular Data
Platt scaling assumes a symmetric logistic sigmoid transformation $p_{\text{cal}} = \sigma(A \cdot f(x) + B)$. On real-world imbalanced tabular datasets, prediction score distributions are heavily right-skewed or left-skewed, causing Platt scaling to fail.

`ml-mcp` implements **3-Parameter Beta Calibration**:
$$p_{\text{cal}} = \frac{1}{1 + \frac{1}{e^c} \cdot \frac{(1 - p)^b}{p^a}}$$
Linearized in log-odds space:
$$\text{logit}(p_{\text{cal}}) = a \ln(p) - b \ln(1 - p) + c$$
Fitting parameters $a, b \ge 0$ and $c \in \mathbb{R}$ captures severe probability asymmetry and boundaries without over-fitting.

---

## 2. Equal-Frequency Adaptive ECE
Standard Expected Calibration Error (ECE) uses 10 equal-width bins $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$. On imbalanced datasets, 95% of samples fall into the first bin, leaving remaining bins sparse or empty and introducing severe estimator variance.

`ml-mcp` implements **Adaptive-Quantile ECE**:
$$\text{ECE}_{\text{adaptive}} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
where each bin $B_b$ contains exactly $\lfloor N / B \rfloor$ samples based on empirical quantiles, eliminating sample-size bias.

---

## 3. Multiclass Dirichlet Calibration for Probability Simplex
For multiclass problems ($K > 2$), standard Beta calibration or Platt scaling fails because independent binary transformations do not preserve the probability simplex condition:
$$\sum_{k=1}^K P(Y = k \mid X) = 1$$
`ml-mcp` implements **Dirichlet Calibration with L2 Off-Diagonal Regularization** (Kull et al., NeurIPS 2019):
$$\ln P(Y = k \mid X) = \sum_{j=1}^K w_{kj} \ln(p_j) + b_k$$
Formulated as multinomial logistic regression over the log-probabilities $\ln(\mathbf{p})$, Dirichlet calibration guarantees well-calibrated posterior probabilities over multi-class classifications without degenerating to uncalibrated softmax or heuristics.
