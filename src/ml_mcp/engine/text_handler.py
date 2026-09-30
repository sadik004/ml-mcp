"""Free-form natural language text feature detector with UUID and hash filtering."""
from __future__ import annotations

import re
from typing import Dict, List
import pandas as pd

from ml_mcp.schemas.audit import TextFeatureReportDTO


class TextFeatureHandler:
    """Detects free-form natural language text features while filtering out high-entropy IDs."""

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
            recommended_strategy="tfidf_sublinear",
        )
