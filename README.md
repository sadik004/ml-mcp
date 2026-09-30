# Machine Learning MCP Server (`ml-mcp`)
> **Enterprise & Kaggle Grandmaster Edition**

A production-grade, 3-tier clean architecture Model Context Protocol (MCP) server designed for dual-mode execution on **Local IDEs (Antigravity, Claude Code)** and **Google Colab Cloud GPU (NVIDIA T4/L4/A100)**.

---

## ⚡ Key Highlights
- **3-Tier Clean Architecture:** Strict separation between Routers/Tools, Deterministic Engines, and Pydantic v2 DTO Schemas.
- **Recursive JSON Sanitizer:** Seamlessly converts `np.generic`, `np.ndarray`, `pd.Series`, `pd.NA`, `pd.Timestamp`, `np.inf`, and `np.nan` into native JSON types.
- **Colab Cloud Symbiosis:** Works natively with `googlecolab/colab-mcp` or via FastMCP SSE tunnels with automated Drive persistence (`/content/drive/MyDrive/ml_mcp/`).
- **Defensive Safeguards:** Zero data leakage pipelines, omnipresent imputers, accuracy paradox metric guards, out-of-distribution detection, and sub-10s TreeSHAP explainability.

---

## 🚀 Installation & Local Run

```bash
# 1. Clone repository
git clone https://github.com/sadik004/ml-mcp.git
cd ml-mcp

# 2. Install dependencies
pip install -e .

# 3. Run unit tests
pytest tests/unit/ -v

# 4. Start FastMCP Stdio server
python -m ml_mcp.server
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src/ml_mcp
```
