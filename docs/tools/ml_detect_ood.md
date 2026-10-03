# `ml_detect_ood` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Liu, Wang, Owens, Li (NeurIPS 2020) — *"Energy-based Out-of-Distribution Detection"*
> - Hendrycks et al. (ICLR 2017) — *"A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"*

---

## 1. Free Energy vs Softmax Confidence
Maximum Softmax Probability (MSP) often produces overconfident predictions on anomalous or out-of-distribution tabular samples due to the normalizing denominator of softmax.

`ml-mcp` implements **Helmholtz Free Energy Scoring**:
$$E(x; f) = -T \cdot \log \sum_{i=1}^K \exp\left(\frac{f_i(x)}{T}\right)$$
- Samples with $E(x) > \tau$ are identified as Out-of-Distribution.
- Energy scoring maps smoothly to the log-marginal likelihood $\log p(x)$, providing superior separation on extreme unseen feature regimes.
