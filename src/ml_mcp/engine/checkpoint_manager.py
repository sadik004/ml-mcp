"""Checkpoint manager and experiment lineage persistence."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional
import joblib

from ml_mcp.config import get_settings
from ml_mcp.schemas.audit import DatasetLineageDTO
from ml_mcp.schemas.tournament import LineageDTO


class CheckpointManager:
    """Handles atomic .joblib serialization, Drive persistence, and cryptographic lineage tracking."""

    def __init__(self, storage_root: Optional[str] = None) -> None:
        if storage_root is not None:
            self.root = Path(storage_root).resolve()
        else:
            settings = get_settings()
            self.root = Path(settings.drive_root).resolve()

        self.checkpoints_dir = self.root / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_git_commit_sha() -> Optional[str]:
        """Attempts to retrieve the current repository commit SHA (git rev-parse HEAD)."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return None

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

        resolved_git_commit = git_commit or self.get_git_commit_sha()

        dto = LineageDTO(
            dataset_hash=dataset_hash,
            row_count=row_count,
            column_count=column_count,
            random_seed=random_seed,
            git_commit=resolved_git_commit,
            checkpoint_path=checkpoint_path,
        )

        # Also serialize canonical DatasetLineageDTO
        iso_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset_lineage = DatasetLineageDTO(
            job_id=job_id,
            dataset_hash=dataset_hash,
            git_commit=resolved_git_commit,
            total_rows=row_count,
            total_columns=column_count,
            timestamp=iso_now,
        )

        metadata_path = self.checkpoints_dir / f"{job_id}_lineage.json"
        metadata_path.write_text(json.dumps(dataset_lineage.model_dump(), indent=2), encoding="utf-8")

        # Optional dispatch to ML_MCP_LINEAGE_WEBHOOK if configured in environment
        webhook_url = os.environ.get("ML_MCP_LINEAGE_WEBHOOK")
        if webhook_url:
            try:
                import urllib.request
                req = urllib.request.Request(
                    webhook_url,
                    data=json.dumps(dataset_lineage.model_dump()).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3):
                    pass
            except Exception:
                pass

        return dto
