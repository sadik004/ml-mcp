"""Bulk Batch Predictor Engine with Calibrated Decision Theory (DCA) and Kaggle Submission Integrity Guard."""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ml_mcp.schemas.serving import BatchPredictDTO

logger = logging.getLogger(__name__)


class BatchPredictor:
    """High-throughput chunked batch scoring engine enforcing InferLine invariants and DCA optimal cutoffs.

    Theoretical foundations:
        - Out-of-Core Analytical Streaming: Raasveldt et al. (VLDB 2022)
        - InferLine Pipeline Invariants: MLSys 2021
        - Decision Curve Analysis Cutoff p*: Vickers & Elkin (BMJ/Lancet)
    """

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
        optimal_threshold: Optional[float] = None,
        calibrator: Optional[Any] = None,
    ) -> BatchPredictDTO:
        """Score unlabelled CSV data in memory-bounded chunks with optional calibrated probabilities and optimal DCA cutoff."""
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV not found at: {input_csv_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

        input_ids: List[Any] = []
        output_chunks: List[pd.DataFrame] = []
        total_rows = 0

        # Effective decision cutoff: use optimal DCA threshold if specified, else 0.50
        threshold_cutoff = optimal_threshold if optimal_threshold is not None else 0.50

        # Read in memory-bounded chunks
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
                # Get raw probabilities
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_chunk)
                    if probas.shape[1] == 2:
                        p1 = probas[:, 1]
                        # Apply calibrator if supplied
                        if calibrator is not None and hasattr(calibrator, "predict_proba"):
                            cal_probas = calibrator.predict_proba(p1.reshape(-1, 1))
                            p1 = cal_probas[:, 1] if cal_probas.ndim == 2 else cal_probas
                        # Decision rule with optimal DCA cutoff
                        preds = (p1 >= threshold_cutoff).astype(int)
                        res_df["Predicted_Label"] = preds
                        res_df["Confidence_Score"] = np.clip(p1, 0.0, 1.0)
                        res_df["Threshold_Applied"] = round(threshold_cutoff, 4)
                    else:
                        preds = np.argmax(probas, axis=1)
                        res_df["Predicted_Label"] = preds
                        res_df["Confidence_Score"] = np.clip(np.max(probas, axis=1), 0.0, 1.0)
                else:
                    preds = model.predict(X_chunk)
                    res_df["Predicted_Label"] = preds
                    res_df["Confidence_Score"] = 1.0
            else:
                preds = model.predict(X_chunk)
                res_df["Prediction"] = preds

            output_chunks.append(res_df)

        # Concatenate and save
        final_df = pd.concat(output_chunks, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)

        # Kaggle & InferLine Submission Integrity Verification
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
