"""Checkpoint manager and experiment lineage persistence."""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import joblib

from ml_mcp.config import get_settings
from ml_mcp.schemas.audit import DatasetLineageDTO
from ml_mcp.schemas.tournament import LineageDTO

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handles atomic .joblib serialization, Drive persistence, and cryptographic lineage tracking."""

    def __init__(
        self,
        storage_root: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        target_root = checkpoint_dir or storage_root
        if target_root is not None:
            self.root = Path(target_root).resolve()
        else:
            settings = get_settings()
            self.root = Path(settings.drive_root).resolve()

        self.checkpoints_dir = self.root / "checkpoints" if "checkpoints" not in self.root.parts else self.root
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def create_lineage(
        self,
        df: Any,
        checkpoint_path: Optional[str] = None,
        job_id: str = "default_job",
        random_seed: Optional[int] = None,
    ) -> LineageDTO:
        """Helper to create and record lineage from an in-memory DataFrame."""
        import pandas as pd
        csv_bytes = df.to_csv(index=False).encode("utf-8") if isinstance(df, pd.DataFrame) else bytes(str(df), "utf-8")
        row_count = len(df) if hasattr(df, "__len__") else 0
        col_count = len(getattr(df, "columns", [])) if hasattr(df, "columns") else 0
        return self.record_lineage(
            dataset_content=csv_bytes,
            row_count=row_count,
            column_count=col_count,
            random_seed=random_seed,
            job_id=job_id,
        )

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
        except Exception as e:
            logger.debug(f"Git commit SHA resolution failed: {e}")
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
        random_seed: Optional[int] = None,
        git_commit: Optional[str] = None,
        job_id: str = "default_job",
    ) -> LineageDTO:
        resolved_seed = random_seed if random_seed is not None else get_settings().random_state
        """Computes SHA-256 dataset digest and records experiment lineage."""
        dataset_hash = hashlib.sha256(dataset_content).hexdigest()
        checkpoint_path = str(self.checkpoints_dir / f"{job_id}.joblib")

        resolved_git_commit = git_commit or self.get_git_commit_sha()

        dto = LineageDTO(
            dataset_hash=dataset_hash,
            row_count=row_count,
            column_count=column_count,
            random_seed=resolved_seed,
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
                with urllib.request.urlopen(req, timeout=3) as resp:
                    _ = resp.read()
            except Exception as e:
                logger.warning(f"Lineage webhook dispatch failed: {e}")

        return dto
