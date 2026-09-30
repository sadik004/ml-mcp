"""Bulk Batch Predictor Engine with Kaggle Submission Integrity Guard."""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ml_mcp.schemas.serving import BatchPredictDTO

logger = logging.getLogger(__name__)


class BatchPredictor:
    """High-throughput chunked batch scoring engine enforcing Kaggle submission invariants."""

    def __init__(self, chunksize: int = 5000) -> None:
        self.chunksize = chunksize

    def predict_csv(
        self,
        model: Any,
        input_csv_path: str,
        output_csv_path: str,
        id_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        task_type: str = "classification",
    ) -> BatchPredictDTO:
        """Score unlabelled CSV data in memory-efficient chunks and verify submission integrity."""
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV not found at: {input_csv_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

        input_ids: List[Any] = []
        output_chunks: List[pd.DataFrame] = []
        total_rows = 0

        # Read in chunks
        for chunk in pd.read_csv(input_csv_path, chunksize=self.chunksize):
            total_rows += len(chunk)

            # Preserve ID vector
            if id_column and id_column in chunk.columns:
                chunk_ids = chunk[id_column].tolist()
                input_ids.extend(chunk_ids)
            else:
                chunk_ids = list(range(total_rows - len(chunk), total_rows))

            # Select features
            if feature_columns:
                X_chunk = chunk[feature_columns]
            elif id_column and id_column in chunk.columns:
                X_chunk = chunk.drop(columns=[id_column])
            else:
                X_chunk = chunk

            # Inference
            res_df = pd.DataFrame()
            if id_column:
                res_df[id_column] = chunk_ids

            if task_type == "classification":
                preds = model.predict(X_chunk)
                res_df["Predicted_Label"] = preds

                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_chunk)
                    max_conf = np.max(probas, axis=1)
                    # Float precision clamp
                    res_df["Confidence_Score"] = np.clip(max_conf, 0.0, 1.0)
                else:
                    res_df["Confidence_Score"] = 1.0
            else:
                preds = model.predict(X_chunk)
                res_df["Prediction"] = preds

            output_chunks.append(res_df)

        # Concatenate and save
        final_df = pd.concat(output_chunks, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)

        # Kaggle Submission Integrity Verification
        has_nan_or_inf = bool(
            final_df.isna().any().any()
            or np.isinf(final_df.select_dtypes(include=[np.number]).to_numpy()).any()
        )

        id_column_verified = True
        if id_column and id_column in final_df.columns:
            id_column_verified = (final_df[id_column].tolist() == input_ids)

        kaggle_submission_ready = bool(
            id_column_verified
            and not has_nan_or_inf
            and len(final_df) == total_rows
        )

        return BatchPredictDTO(
            input_csv_path=input_csv_path,
            output_csv_path=output_csv_path,
            rows_processed=total_rows,
            id_column_verified=id_column_verified,
            has_nan_or_inf=has_nan_or_inf,
            kaggle_submission_ready=kaggle_submission_ready,
        )
