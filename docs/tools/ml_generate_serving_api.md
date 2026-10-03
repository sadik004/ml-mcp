# `ml_generate_serving_api` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - FastAPI 0.100+ Standards & ASGI Lifespan Protocol (PEP 550 / Starlette)
> - Fowler & Clean Architecture Patterns: Separation of Serving and Domain

---

## 1. Modern `lifespan` Context Manager (Zero Startup Deprecations)
FastAPI deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` in favor of asynchronous lifespan context managers:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and artifacts once at application initialization
    model_holder["model"] = load_artifacts()
    yield
    # Graceful shutdown, flush buffers and release GPU/threads
    model_holder.clear()

app = FastAPI(lifespan=lifespan)
```
Benefits:
- Atomic artifact loading with fail-fast validation.
- Zero race conditions during worker spin-up.
- Unified exception handling during startup and shutdown.

---

## 2. CNCF Cloud-Native Health Probes
The generated service provides dedicated probes for Kubernetes and container orchestrators:
- **`/healthz` (Liveness):** Confirms the HTTP event loop is responsive.
- **`/readyz` (Readiness):** Confirms model weights and pipelines are fully loaded in memory and ready to serve traffic.
- **`/predict`:** Pydantic v2 typed single-observation inference with strict type validation.
- **`/predict/batch`:** Vectorized high-throughput bulk inference.
