# `ml_generate_eval_dashboard` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Vickers et al. (Annals of Internal Medicine 2021) — *"Decision Curve Analysis (Net Benefit)"*
> - ACM FAccT 2022 — *"Transparent Model Cards and Production Verification Dashboards"*

---

## 1. Pure SVG Decision Curve Analysis (Net Benefit)
Decision Curve Analysis evaluates clinical and commercial utility across varying decision thresholds $p_t$:
$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$
The dashboard embeds a standalone, pure SVG vector curve rendering:
- **Model Curve (Cyan/Blue):** Model net benefit across $[0.01, 0.99]$.
- **Treat All Strategy (Red Dash):** Net benefit if intervention is given to all individuals.
- **Treat None Strategy (Gray Baseline):** Zero net benefit baseline.

---

## 2. Pure SVG Beta Calibration Reliability Diagram
Visualizes model calibration without heavy JavaScript or external charting libraries:
- Compares **Mean Predicted Probability** against **Empirical Observed Proportion** across adaptive quantile bins.
- Ideal calibration diagonal ($y = x$) highlighted as reference guide.
- Fully portable: Single self-contained HTML artifact viewable offline or inside firewalled environments.
