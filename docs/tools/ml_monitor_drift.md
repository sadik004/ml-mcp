# `ml_monitor_drift` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Yurdakul (2020) — *"Statistical Properties of Population Stability Index (PSI)"*
> - Villani (Springer 2009) — *"Optimal Transport: Old and New (Wasserstein-1 Earth Mover's Distance)"*
> - Massey (JASA 1951) — *"The Kolmogorov-Smirnov Test for Goodness of Fit"*

---

## 1. Population Stability Index (PSI)
Quantifies distribution shift between baseline reference dataset $P$ and production serving dataset $Q$ across $B$ equal-frequency quantiles:
$$\text{PSI} = \sum_{b=1}^B \left( Q_b - P_b \right) \times \ln\left( \frac{Q_b}{P_b} \right)$$
- $\text{PSI} < 0.10$: No significant drift (Stable).
- $0.10 \le \text{PSI} < 0.25$: Moderate drift (Requires monitoring).
- $\text{PSI} \ge 0.25$: Severe drift (Model retraining triggered).

---

## 2. Wasserstein-1 (Earth Mover's Distance)
Measures minimum probability mass transportation cost between empirical distributions:
$$\mathcal{W}_1(u, v) = \int_{-\infty}^{+\infty} |U(x) - V(x)| dx$$
Unlike KL-divergence, Wasserstein-1 remains finite, stable, and continuous even when distributions have disjoint supports or extreme boundary anomalies.
