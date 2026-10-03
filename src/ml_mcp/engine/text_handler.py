"""Free-form natural language text feature detector and dense embedding extractor (Reimers & Gurevych)."""
from __future__ import annotations

import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ml_mcp.schemas.audit import TextFeatureReportDTO


class TextFeatureHandler:
    """Detects free-form natural language text features and extracts compact dense embeddings.

    Theoretical Basis:
        - Reimers, N., & Gurevych, I. (EMNLP 2019). "Sentence-BERT: Sentence Embeddings using
          Siamese BERT-Networks." MiniLM-L6-v2 produces dense 384-dimensional semantic embeddings.
        - High-cardinality and free-form text expanded into sparse high-dimensional bag-of-words
          degrades GBDT cache locality and tree depth; dense representations compress semantic signal
          into compact float32 vectors.
    """

    # Regex for UUID4 or general 32-36 hex characters
    UUID_REGEX = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )
    HEX_HASH_REGEX = re.compile(r"^[0-9a-fA-F]{24,64}$")

    def __init__(self, min_avg_char_length: float = 30.0, min_avg_words: float = 3.0) -> None:
        self.min_avg_char_length = min_avg_char_length
        self.min_avg_words = min_avg_words

    def is_hex_or_uuid(self, sample_series: pd.Series) -> bool:
        """Returns True if the series contains UUIDs or hex hashes instead of prose."""
        non_null = sample_series.dropna().astype(str).str.strip()
        if non_null.empty:
            return False

        sample_subset = non_null.head(20)
        uuid_matches = sum(bool(self.UUID_REGEX.match(val)) for val in sample_subset)
        hex_matches = sum(bool(self.HEX_HASH_REGEX.match(val)) for val in sample_subset)

        # If >50% match UUID or hex pattern, it is an identifier, not natural text
        if (uuid_matches / len(sample_subset) > 0.5) or (hex_matches / len(sample_subset) > 0.5):
            return True
        return False

    def analyze_text_features(self, df: pd.DataFrame) -> TextFeatureReportDTO:
        """Scans string/object columns to identify natural language prose.

        Returns:
            TextFeatureReportDTO containing recognized text columns and strategy.
        """
        text_cols: List[str] = []
        avg_char_lens: Dict[str, float] = {}
        unique_tokens: Dict[str, int] = {}

        for col in df.columns:
            series = df[col]
            if series.dtype == "object" or isinstance(series.dtype, pd.StringDtype):
                valid_str = series.dropna().astype(str)
                if valid_str.empty or len(valid_str) < 2:
                    continue

                # Filter out UUIDs and hex hashes
                if self.is_hex_or_uuid(valid_str):
                    continue

                # Check average character length
                char_lens = valid_str.str.len()
                avg_len = float(char_lens.mean())

                # Check average word count (separated by whitespace)
                word_counts = valid_str.str.split().str.len()
                avg_words = float(word_counts.mean())

                if avg_len >= self.min_avg_char_length and avg_words >= self.min_avg_words:
                    text_cols.append(col)
                    avg_char_lens[col] = round(avg_len, 1)

                    # Estimate distinct vocabulary size
                    sample_text = " ".join(valid_str.head(100).tolist()).lower()
                    unique_tokens[col] = len(set(sample_text.split()))

        return TextFeatureReportDTO(
            text_columns=text_cols,
            avg_char_lengths=avg_char_lens,
            unique_token_counts=unique_tokens,
            recommended_strategy="dense_embedding",
        )

    # Alias for FastMCP tool compatibility
    detect_text_features = analyze_text_features

    def extract_dense_embeddings(self, text_series: pd.Series, n_components: int = 16) -> np.ndarray:
        """Extracts dense semantic representations using MiniLM-L6-v2 or fast TF-IDF + TruncatedSVD fallback.

        Returns:
            2D numpy array of shape (n_samples, n_dimensions).
        """
        cleaned_texts = text_series.fillna("").astype(str).tolist()
        if not cleaned_texts:
            return np.empty((0, n_components), dtype=np.float32)

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(cleaned_texts, show_progress_bar=False)
            return np.asarray(embeddings, dtype=np.float32)
        except Exception:
            # Sub-10ms fallback for MCP / non-GPU environments without torch
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            tfidf = TfidfVectorizer(max_features=64, sublinear_tf=True)
            X_tfidf = tfidf.fit_transform(cleaned_texts)

            # SVD components cannot exceed feature or sample count
            comp = min(n_components, X_tfidf.shape[1], max(1, X_tfidf.shape[0] - 1))
            if comp < 1:
                return np.zeros((len(cleaned_texts), 1), dtype=np.float32)

            svd = TruncatedSVD(n_components=comp, random_state=42)
            dense = svd.fit_transform(X_tfidf)
            return dense.astype(np.float32)
