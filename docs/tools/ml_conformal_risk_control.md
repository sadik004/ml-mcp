# `ml_conformal_risk_control` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Romano, Barber, Candès (NeurIPS 2020) — *"Classification with Valid and Equal Coverage for Inherent Subgroups"* (Mondrian Conformal Prediction)
> - Angelopoulos et al. (ICLR 2021) — *"Uncertainty Sets for Image and Tabular Classifiers via RAPS"*
> - Angelopoulos & Bates (2022) — *"A Gentle Introduction to Conformal Prediction and Distribution-Free UQ"*

---

## 1. The Core Problem: Why Softmax Probabilities Fail
Standard deep learning and gradient boosted decision tree classifiers produce output probabilities via the Softmax or Sigmoid operator. Under distribution shift, class imbalance, or adversarial noise, these raw scores are frequently overconfident. A model may output $P(Y = \text{Fraud}) = 0.991$ while being completely incorrect.

Conformal Risk Control guarantees finite-sample, distribution-free statistical coverage:
$$\mathbb{P}(Y \in C(X)) \ge 1 - \alpha$$
where $\alpha \in (0, 1)$ is the user-specified error tolerance (e.g., $\alpha = 0.05$ guarantees 95% coverage).

---

## 2. Mondrian (Class-Conditional) Conformal Prediction
Marginal conformal prediction guarantees average coverage across the entire population, but often suffers from **coverage under-representation** on rare minority classes (e.g., achieving 99% coverage on benign samples, but only 60% coverage on critical fraud/cancer cases).

`ml-mcp` implements **Mondrian (Class-Conditional) Conformal Prediction**:
$$\hat{q}_k = \text{Quantile}\left( \{s_i : y_i = k\}, \frac{\lceil (n_k + 1)(1 - \alpha) \rceil}{n_k} \right)$$

This provides mathematical finite-sample guarantees for every individual class $k$:
$$\mathbb{P}(Y \in C(X) \mid Y = k) \ge 1 - \alpha$$

---

## 3. Regularized Adaptive Prediction Sets (RAPS)
When models face high-entropy, ambiguous inputs, standard adaptive prediction sets (APS) can balloon to include almost all classes, rendering prediction sets practically useless.

RAPS penalizes excessively large sets using cumulative softmax regularization:
$$s(x, y) = \sum_{j=1}^{\text{rank}(y)} \pi_j(x) + \lambda \cdot \max(0, \text{rank}(y) - k_{\text{reg}}) + u \cdot \pi_{\text{rank}(y)}(x)$$
- $\lambda$: Regularization penalty weight.
- $k_{\text{reg}}$: Free set size threshold before regularization applies.
- Minimizes set size variance while preserving exact marginal and conditional coverage guarantees.

---

## 4. Human-in-the-Loop Triage Escalation
Prediction sets with cardinality $|C(X)| > 1$ or $|C(X)| = 0$ represent high-uncertainty instances. `ml-mcp` flags these samples for automated human review (triage escalation) while automatically passing $|C(X)| = 1$ high-confidence samples directly through the automated pipeline.
