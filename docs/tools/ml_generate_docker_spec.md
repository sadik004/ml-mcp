# `ml_generate_docker_spec` — Deep-Dive Architectural Guide

> **Theoretical Basis:** 
> - Center for Internet Security (CIS) Docker Benchmark v1.6
> - Linux Container Hardening: Rootless Execution & Multi-Stage Builds

---

## 1. Unprivileged Non-Root Security (`appuser:10001`)
Running containers as `root` (UID 0) presents a severe privilege escalation vulnerability. The generated Dockerfile enforces:
```dockerfile
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/sh -m appuser
USER appuser
```
- Completely eliminates root filesystem tampering.
- Compliant with strict enterprise Kubernetes PodSecurityStandards (`restricted`).

---

## 2. Production Multi-Stage Build & Healthchecks
1. **Multi-Stage Separation:** Dependencies are compiled in a temporary builder stage; final image retains only runtime artifacts and wheels, minimizing attack surface.
2. **Container HEALTHCHECK:** Integrated native health probe:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1
```
