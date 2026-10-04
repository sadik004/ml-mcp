"""Base service utilities, abstract dependencies, and file persistence for ML-MCP services."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_mcp.config import Settings, get_settings
from ml_mcp.repositories.storage_repository import (
    LocalDiskStorageRepository,
    StorageRepositoryProtocol,
)


def persist_processed_dataframe(
    df: pd.DataFrame,
    source_path: str,
    suffix: str,
    output_path: Optional[str] = None,
) -> str:
    """Atomically saves processed DataFrame to disk and returns absolute path."""
    if output_path:
        out_file = Path(output_path).resolve()
    else:
        src = Path(source_path).resolve()
        stem = src.stem
        # Avoid chaining double suffixes like _synthesized_synthesized
        clean_stem = re.sub(r"_(synthesized|pruned|filtered)$", "", stem)
        settings = get_settings()
        dest_dir = Path(settings.drive_root).resolve() / "processed"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_file = dest_dir / f"{clean_stem}_{suffix}.csv"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return str(out_file)


class BaseService:
    """Base class for all domain application services with injected repository protocols."""

    def __init__(
        self,
        repository: Optional[StorageRepositoryProtocol] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.repository: StorageRepositoryProtocol = repository or LocalDiskStorageRepository()
        self.settings: Settings = settings or get_settings()
