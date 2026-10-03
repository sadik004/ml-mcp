# `ml_optimize_inference` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - ONNX Runtime Architecture & Graph Optimizations (IEEE Micro 2022)
> - Chen et al. (USENIX ATC 2021) — *"Fast Tensor Compilation & Kernel Fusion for Decision Trees"*

---

## 1. Level-3 Hardware Graph Optimization (`ORT_ENABLE_ALL`)
Standard Python inference suffers from GIL overhead, intermediate numpy array allocations, and non-fused tree traversals. 
`ml-mcp` utilizes **ONNX Runtime (ORT)** configured with comprehensive Level-3 graph optimization:
```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1
```
Optimization stages executed:
1. **Constant Folding:** Compile-time evaluation of constant sub-expressions.
2. **Dead Node Elimination:** Removal of unreachable sub-graphs and unused intermediate feature transformations.
3. **Node Fusion & Kernel Inlining:** Merging normalization + tree-ensemble score accumulation into vectorized single-pass C++ kernels.

---

## 2. Dedicated Multi-Engine GBDT Converters
Gradient Boosted Decision Trees (LightGBM, XGBoost, CatBoost) and Scikit-Learn pipelines feature heterogeneous internal data structures. `ml-mcp` provides resilient, multi-engine conversion cascades:
- **LightGBM:** Converted directly via native `onnxmltools.convert_lightgbm` with strictly typed `FloatTensorType` / `DoubleTensorType`.
- **XGBoost:** Converted via `onnxmltools.convert_xgboost` preserving tree depth and split pointers.
- **Scikit-Learn:** Converted via `skl2onnx.to_onnx` with automatic feature type discovery and pipeline composition.

---

## 3. Benchmarking & P95/P99 Latency Profiling
Real-world SLA violations occur at the tail percentiles. `ml_optimize_inference` benchmarks 100 single-sample warm inference passes:
- **P95 Latency (ms):** 95th percentile response time.
- **P99 Latency (ms):** 99th percentile worst-case response time.
- **Throughput:** Single-core inferences per second.
