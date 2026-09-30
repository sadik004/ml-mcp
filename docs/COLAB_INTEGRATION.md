# Google Colab Integration & Cloud GPU Acceleration Guide

> **Ecosystem:** `ml-mcp` + Google Colab (`googlecolab/colab-mcp`)  
> **Target Hardware:** NVIDIA T4 / L4 / A100 GPUs (16GB - 40GB VRAM)  
> **Persistence Target:** `/content/drive/MyDrive/ml_mcp/`

---

## 1. Why Colab Symbiosis?

Local developer machines frequently suffer from limited RAM, no dedicated CUDA GPU, or CPU thermal throttling when executing heavy machine learning workloads such as:
- 8-model competitive tournaments across multi-thousand row datasets.
- 100-trial Optuna Bayesian hyperparameter sweeps.
- Sub-10s TreeSHAP calculation over large validation partitions.
- Stacking ensemble out-of-fold matrix generation.

`ml-mcp` bridges local IDE agents (Antigravity IDE, Claude Code, Cursor) with Google Colab's free and pro cloud GPU infrastructure.

---

## 2. Architecture & Topology

```
┌─────────────────────────────────┐
│     LOCAL CLIENT (IDE / AGENT)  │
│  - Antigravity IDE              │
│  - Claude Desktop / Claude Code │
└────────────────┬────────────────┘
                 │
                 │ SSE / FastMCP Tunnel
                 │ (ngrok / cloudflared / colab-mcp)
                 ▼
┌─────────────────────────────────┐
│     GOOGLE COLAB GPU INSTANCE   │
│  - NVIDIA T4 / L4 GPU (16GB)    │
│  - CUDA-accelerated GBDTs       │
│  - FastMCP Server running on    │
│    localhost:8000               │
└────────────────┬────────────────┘
                 │
                 │ Auto Checkpoint Sync
                 ▼
┌─────────────────────────────────┐
│       GOOGLE DRIVE STORAGE      │
│  - /content/drive/MyDrive/      │
│    ml_mcp/checkpoints/          │
│    ml_mcp/models/               │
└─────────────────────────────────┘
```

---

## 3. Running `ml-mcp` on Google Colab

### Step 1: Open a Colab Notebook with GPU
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime -> Change runtime type**.
3. Select **T4 GPU** (or L4 / A100 if Colab Pro) and click **Save**.

### Step 2: Install `ml-mcp` & Dependencies
Paste the following cell into your Colab notebook:

```python
# 1. Mount Google Drive for automatic model checkpointing
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone and install ml-mcp
!git clone https://github.com/sadik004/ml-mcp.git /content/ml-mcp
%cd /content/ml-mcp
!pip install -q -e .
!pip install -q pyngrok lightgbm xgboost catboost onnxruntime shap
```

### Step 3: Launch FastMCP Server with SSE Tunnel

```python
import os
from pyngrok import ngrok
from ml_mcp.server import mcp

# Set up ngrok auth token (get free token from https://dashboard.ngrok.com)
NGROK_TOKEN = "your_ngrok_auth_token_here"
ngrok.set_auth_token(NGROK_TOKEN)

# Expose port 8000
public_url = ngrok.connect(8000).public_url
print(f"🚀 ml-mcp Remote SSE Endpoint: {public_url}/sse")

# Start MCP Server with SSE transport
mcp.run(transport="sse", port=8000, host="0.0.0.0")
```

---

## 4. Connecting Your Local IDE Agent to Colab

### For Claude Desktop / Antigravity IDE (`claude_desktop_config.json`)

Update your local MCP configuration file:

```json
{
  "mcpServers": {
    "ml-mcp-colab": {
      "url": "https://your-unique-ngrok-subdomain.ngrok-free.app/sse"
    }
  }
}
```

Now, when you prompt your agent:
> *"Audit the dataset at `https://raw.githubusercontent.com/.../train.csv` and run an 8-model tournament with GPU acceleration."*

The local agent delegates the heavy computation directly to the NVIDIA GPU running on Colab.

---

## 5. Google Drive Checkpoint Management

`ml-mcp` includes an automated Drive Checkpoint Manager (`src/ml_mcp/engine/checkpoint_manager.py`).

When running on Colab (`COLAB_GPU=1` or `os.path.exists("/content/drive")`):
- All trained champions and stacking models are saved to:  
  `/content/drive/MyDrive/ml_mcp/checkpoints/<timestamp>_<model_name>.joblib`
- Even if Colab disconnects or times out, your trained models and artifacts are securely retained in Google Drive.
- Checkpoints can be reloaded immediately in subsequent sessions using:
  ```python
  from ml_mcp.engine.checkpoint_manager import CheckpointManager
  ckpt = CheckpointManager()
  model = ckpt.load_latest_checkpoint()
  ```
