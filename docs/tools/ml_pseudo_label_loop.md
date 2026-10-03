# `ml_pseudo_label_loop` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Zhang et al. (NeurIPS 2021) — *"FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo-Labeling"*
> - Wang et al. (ICLR 2023) — *"FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning"*
> - Angelopoulos et al. (2023) — *"Conformal Prediction: A Gentle Introduction & Singleton Set Uncertainty Filtering"*

---

## 1. The Confirmation Bias & Majority Collapse Problem
In standard pseudo-labeling, a static confidence threshold (e.g. 0.95 or 0.98) is applied uniformly across all classes. On imbalanced tabular datasets:
1. **Majority Class Collapse:** Easy majority-class samples routinely score $\ge 0.98$, while difficult or minority classes almost never pass the fixed bar.
2. **Confirmation Bias:** Early erroneous predictions on minority boundaries receive pseudo-labels, continually reinforcing model delusions in iterative training loops.

---

## 2. FlexMatch Dynamic Curriculum Thresholding
`ml-mcp` implements **Class-Adaptive Dynamic Thresholding** based on curriculum learning status $\sigma_c(t)$:
$$	au_c(t) = 	au_{\text{base}} \cdot \max\left(0.70, \frac{\sigma_c(t) + 1}{\max_{c'} \sigma_{c'}(t) + 1}\right)$$
where $\sigma_c(t)$ represents the number of unlabelled observations where class $c$ had the argmax probability.
- **Difficult / Minority Classes:** If $\sigma_c(t) < \max \sigma$, the threshold $	au_c(t)$ lowers dynamically (bounded at $0.70 \cdot \tau_{\text{base}}$), allowing the model to harvest informative minority instances.
- **Easy / Majority Classes:** When $\sigma_c(t)$ approaches the maximum, $	au_c(t) \to \tau_{\text{base}}$, preventing flood of uncalibrated majority labels.

---

## 3. Conformal Prediction Singleton Safety Gate
To prevent false-positive pseudo-labels on ambiguous boundary samples, predictions are passed through an inductive **Conformal Prediction Safety Filter**:
1. Conformal non-conformity scores are computed on calibration subsets:
   $$s_i = 1 - \hat{P}(Y = y_i \mid X_i)$$
2. Given significance level $\alpha$ (e.g. 0.10 for 90% finite-sample coverage guarantee), quantile threshold $\hat{q}$ is determined:
   $$\hat{q} = \text{Quantile}\left(\left\{s_i\right\}; \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)$$
3. Conformal uncertainty set $C(x)$ is formed:
   $$C(x) = \left\{ c \in \mathcal{Y} : 1 - \hat{P}(Y = c \mid x) \le \hat{q} \right\}$$
4. **Singleton Requirement:** An unlabelled sample $x$ is harvested **if and only if** predicted confidence $\ge \tau_c(t)$ **AND** $|C(x)| = 1$. If $|C(x)| > 1$ (model is split between classes) or $|C(x)| = 0$ (outlier), the sample is strictly rejected.

---

## 4. Telemetry & Audit Return Contract
The tool returns:
- `harvested_count`: Total reliable pseudo-labels added to training pool.
- `original_train_size`: Starting baseline sample count.
- `refined_train_size`: New augmented dataset count.
- `class_thresholds`: Effective dynamic thresholds $\tau_c(t)$ for each class.
- `per_class_harvested`: Exact breakdown of harvested pseudo-labels per category.
- `selection_ratios`: Proportion of unlabelled candidates retained per class.
