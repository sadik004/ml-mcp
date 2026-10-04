"""Data persistence, dataset access, and artifact storage abstractions for ML-MCP (Phase P5)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd

from ml_mcp.config import get_settings


@runtime_checkable
class DatasetRepositoryProtocol(Protocol):
    """Protocol for reading and writing raw and processed tabular datasets."""

    def read_csv(self, path: str) -> pd.DataFrame: ...
    def load_dataframe(self, path: str) -> pd.DataFrame: ...
    def write_processed(
        self,
        df: pd.DataFrame,
        source_path: str,
        suffix: str,
        output_path: Optional[str] = None,
    ) -> str: ...


@runtime_checkable
class ArtifactRepositoryProtocol(Protocol):
    """Protocol for managing serializable model artifacts, checkpoints, and metrics."""

    def load_model(self, path: str) -> Any: ...
    def save_model(self, model: Any, path: str) -> str: ...
    def load_artifact(self, path: str) -> Any: ...
    def save_artifact(self, model: Any, path: str) -> str: ...
    def load_npy(self, path: str) -> np.ndarray: ...
    def save_npy(self, arr: np.ndarray, path: str) -> str: ...
    def read_json(self, path: str) -> Dict[str, Any]: ...
    def write_json(self, data: Dict[str, Any], path: str) -> str: ...
    def file_exists(self, path: str) -> bool: ...


@runtime_checkable
class StorageRepositoryProtocol(DatasetRepositoryProtocol, ArtifactRepositoryProtocol, Protocol):
    """Unified repository protocol combining dataset and artifact persistence."""
    ...


class LocalDiskStorageRepository:
    """Concrete storage repository operating on local filesystem paths."""

    def read_csv(self, path: str) -> pd.DataFrame:
        return self.load_dataframe(path)

    def load_dataframe(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at path: {path}")
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def save_dataframe(self, df: pd.DataFrame, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df.to_csv(path, index=False)
        return os.path.abspath(path)

    def write_processed(
        self,
        df: pd.DataFrame,
        source_path: str,
        suffix: str,
        output_path: Optional[str] = None,
    ) -> str:
        if output_path:
            out_file = Path(output_path).resolve()
        else:
            src = Path(source_path).resolve()
            stem = src.stem
            clean_stem = re.sub(r"_(synthesized|pruned|filtered)$", "", stem)
            settings = get_settings()
            dest_dir = Path(settings.drive_root).resolve() / "processed"
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_file = dest_dir / f"{clean_stem}_{suffix}.csv"

        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        return str(out_file)

    def load_model(self, path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at path: {path}")
        return joblib.load(path)

    def save_model(self, model: Any, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(model, path)
        return os.path.abspath(path)

    def load_artifact(self, path: str) -> Any:
        return self.load_model(path)

    def save_artifact(self, model: Any, path: str) -> str:
        return self.save_model(model, path)

    def load_npy(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Array file not found at path: {path}")
        return np.load(path)

    def save_npy(self, arr: np.ndarray, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, arr)
        return os.path.abspath(path)

    def read_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found at path: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, data: Dict[str, Any], path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return os.path.abspath(path)

    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)


# Concrete aliases for explicit separation
LocalDiskDatasetRepository = LocalDiskStorageRepository
LocalDiskArtifactRepository = LocalDiskStorageRepository
