# `ml_tune_threshold_and_errors` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Vickers & Elkin (BMJ / Lancet) — *"Decision Curve Analysis: A Novel Method for Evaluating Prediction Models"*
> - Hernández-Orallo, Flach, & Ferri (JMLR 2013) — *"Brier Curves and Cost Curves: An Extensive Investigation of Evaluating and Comparing Prediction Models"*
> - Sheng & Ling (IEEE TKDE 2014) — *"Thresholding for Making Better Decisions in Cost-Sensitive Learning"*

---

## 1. Decision Curve Analysis (DCA Net Benefit)
In real-world business and clinical deployments, maximizing statistical metrics like F1-score or ROC-AUC does not guarantee net positive real-world utility. 

Decision Curve Analysis calculates the **Net Benefit** across the decision threshold spectrum $p_t \in [0.01, 0.99]$:
$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$

### Baselines for Comparison:
- **Treat All Policy:**
  $$\text{Net Benefit}_{\text{all}}(p_t) = \frac{\text{Positives}}{N} - \frac{\text{Negatives}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$
- **Treat None Policy:**
  $$\text{Net Benefit}_{\text{none}} = 0.0$$

A model should only be deployed at threshold $p_t$ if $\text{Net Benefit}_{\text{model}}(p_t) > \max(\text{Net Benefit}_{\text{all}}(p_t), 0)$. `ml-mcp` calculates the exact operational interval where the model provides superior economic utility over naive policies.
