"""Google Colab Notebook Synthesis and Cloud Session DTOs."""
from __future__ import annotations

from typing import Any, Dict, Optional
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

    tunnel_url: Optional[str] = Field(default=None, description="Active Cloudflare/ngrok SSE tunnel URL")
    is_alive: bool = Field(default=True, description="True if Colab runtime is active and responsive")
    vram_allocated_mb: float = Field(ge=0.0, description="Active CUDA VRAM consumption in MB")
    vram_total_mb: float = Field(ge=0.0, description="Total GPU VRAM available in MB (e.g. 15360 for T4)")
    active_job_id: Optional[str] = Field(default=None, description="Active long-running job ID if executing")

    def to_compact(self) -> Dict[str, Any]:
        return {
            "is_alive": self.is_alive,
            "tunnel_url": self.tunnel_url,
            "vram_allocated_mb": round(self.vram_allocated_mb, 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
            "active_job_id": self.active_job_id,
        }
