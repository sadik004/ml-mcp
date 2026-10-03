# `ml_batch_predict` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Raasveldt & Mühleisen (VLDB 2022) — *"DuckDB: An Embeddable Analytical Database"*
> - Crankshaw et al. (MLSys InferLine) — *"Latency-Aware High-Throughput Model Serving Pipelines"*
> - Vickers et al. (Annals of Internal Medicine 2021) — *"Decision Curve Analysis Cutoff Decision Rules"*

---

## 1. Calibrated & DCA-Aligned Decision Thresholding
Legacy batch scoring applies a hardcoded binary decision threshold:
$$\hat{y} = \mathbb{I}\left(P(Y=1 \mid X) \ge 0.50\right)$$
In production healthcare, fraud detection, and financial risk models, $0.50$ is catastrophic:
- If false-negative cost is $10\times$ false-positive cost, the optimal Decision Curve Analysis (DCA) cutoff is $p^* \approx 0.09$.
- `ml_batch_predict` natively integrates calibrated models ($P_{\text{cal}}(Y=1 \mid X)$) and optimal cutoff $p^*$:
$$\hat{y}_{\text{optimal}} = \mathbb{I}\left(P_{\text{cal}}(Y=1 \mid X) \ge p^*\right)$$

---

## 2. Zero-Copy Chunked Streaming Pipeline
To prevent Out-Of-Memory (OOM) failures on multi-gigabyte batch inference inputs:
1. **Memory-Bounded Chunking:** Files are streamed in chunks of `chunksize=5000` rows using low-overhead pandas iterators.
2. **ID-Column Preservation:** Primary keys (`id_column`) are preserved without type coercion.
3. **NaN & Inf Invariant Guards:** Real-time imputation and sentinel value checks verify no corrupt tensors enter the scoring engine.
4. **Row-Count Conservation Guarantee:** Strict assert verifies $N_{\text{in}} == N_{\text{out}}$ before flushing.
