"""Google Colab Notebook Synthesis and Cloud Session DTOs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field
from ml_mcp.schemas.base import BaseDTO


class ColabNotebookDTO(BaseDTO):
    """Metadata report for generated 1-click Colab executable notebook (.ipynb)."""

    notebook_path: str = Field(description="Destination file path of generated .ipynb")
    cell_count: int = Field(ge=1, description="Total number of structured execution cells")
    cuda_enabled: bool = Field(default=True, description="True if GPU acceleration flags injected")
    mixed_precision: bool = Field(default=True, description="True if FP16 mixed precision enabled")
    drive_checkpoint_dir: str = Field(
        default="/content/drive/MyDrive/ml_mcp/checkpoints",
        description="Google Drive directory where checkpoints are persisted",
    )
    kaggle_slug: Optional[str] = Field(
        default=None, description="Optional Kaggle competition or dataset identifier"
    )

    def to_compact(self) -> Dict[str, Any]:
        return {
            "notebook_path": self.notebook_path,
            "cell_count": self.cell_count,
            "cuda_enabled": self.cuda_enabled,
            "drive_checkpoint_dir": self.drive_checkpoint_dir,
            "kaggle_slug": self.kaggle_slug,
        }


class ColabSessionDTO(BaseDTO):
    """Vitality and hardware monitoring report for active Colab cloud session."""

    session_name: str = Field(default="gpu", description="Identifier name of the Colab session")
    status: str = Field(default="connected", description="Session lifecycle status: connected, idle, busy, disconnected, error")
    hardware: str = Field(default="CPU", description="Detected device accelerator name (e.g. Tesla T4, CPU)")
    cuda_available: bool = Field(default=False, description="True if CUDA is verified active on the remote runtime")
    vram_allocated_mb: float = Field(ge=0.0, default=0.0, description="Active CUDA VRAM consumption in MB")
    vram_total_mb: float = Field(ge=0.0, default=0.0, description="Total GPU VRAM available in MB (e.g. 15360 for T4)")
    ram_total_gb: float = Field(ge=0.0, default=0.0, description="Total system RAM in GB")
    ram_available_gb: float = Field(ge=0.0, default=0.0, description="Available system RAM in GB")
    endpoint: Optional[str] = Field(default=None, description="Internal Colab backend endpoint ID")
    tunnel_url: Optional[str] = Field(default=None, description="Active Cloudflare/ngrok SSE tunnel URL")
    is_alive: bool = Field(default=True, description="True if Colab runtime is active and responsive")
    active_job_id: Optional[str] = Field(default=None, description="Active long-running job ID if executing")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal operational warnings or hardware notices")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "session_name": self.session_name,
            "status": self.status,
            "hardware": self.hardware,
            "cuda_available": self.cuda_available,
            "is_alive": self.is_alive,
            "vram_allocated_mb": round(self.vram_allocated_mb, 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_available_gb": round(self.ram_available_gb, 2),
            "endpoint": self.endpoint,
            "tunnel_url": self.tunnel_url,
            "active_job_id": self.active_job_id,
            "warnings": self.warnings,
        }
