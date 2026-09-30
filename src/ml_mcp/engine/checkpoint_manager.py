"""Checkpoint manager and experiment lineage persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional
import joblib

from ml_mcp.config import get_settings
from ml_mcp.schemas.tournament import LineageDTO


class CheckpointManager:
    """Handles atomic .joblib serialization, Drive persistence, and lineage tracking."""

    def __init__(self, storage_root: Optional[str] = None) -> None:
        if storage_root is not None:
            self.root = Path(storage_root).resolve()
        else:
            settings = get_settings()
            self.root = Path(settings.drive_root).resolve()

        self.checkpoints_dir = self.root / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model: Any, job_id: str) -> str:
        """Atomically serializes model pipeline to .joblib on disk."""
        target_path = self.checkpoints_dir / f"{job_id}.joblib"
        joblib.dump(model, target_path, compress=3)
        return str(target_path)

    def load_checkpoint(self, job_id: str) -> Any:
        """Loads serialized model pipeline from .joblib."""
        target_path = self.checkpoints_dir / f"{job_id}.joblib"
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint for job '{job_id}' not found at {target_path}")
        return joblib.load(target_path)

    def record_lineage(
        self,
        dataset_content: bytes,
        row_count: int,
        column_count: int,
        random_seed: int = 42,
        git_commit: Optional[str] = None,
        job_id: str = "default_job",
    ) -> LineageDTO:
        """Computes SHA-256 dataset digest and records experiment lineage."""
        dataset_hash = hashlib.sha256(dataset_content).hexdigest()
        checkpoint_path = str(self.checkpoints_dir / f"{job_id}.joblib")

        dto = LineageDTO(
            dataset_hash=dataset_hash,
            row_count=row_count,
            column_count=column_count,
            random_seed=random_seed,
            git_commit=git_commit,
            checkpoint_path=checkpoint_path,
        )

        metadata_path = self.checkpoints_dir / f"{job_id}_lineage.json"
        metadata_path.write_text(json.dumps(dto.model_dump(), indent=2), encoding="utf-8")
        return dto
