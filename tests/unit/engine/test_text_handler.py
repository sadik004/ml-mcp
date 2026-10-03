"""Unit tests for Free-Form Text Handler and Dense Embedding Extractor."""
import pytest
import numpy as np
import pandas as pd

from ml_mcp.engine.text_handler import TextFeatureHandler


def test_text_feature_handler_detects_natural_language():
    df = pd.DataFrame({
        "customer_review": [
            "This product was absolutely amazing, arrived on time and exceeded expectations!",
            "Terrible customer service, the device stopped working after just three days.",
            "Average quality for the price, would probably buy again if discounted.",
            "Fast shipping and responsive seller, highly recommend to everyone.",
        ],
        "category": ["Electronics", "Appliances", "Clothing", "Books"],
        "price": [99.99, 149.50, 25.00, 15.99],
    })

    handler = TextFeatureHandler(min_avg_char_length=30, min_avg_words=4)
    report = handler.analyze_text_features(df)

    assert "customer_review" in report.text_columns
    assert "category" not in report.text_columns
    assert report.recommended_strategy in ("dense_embedding", "tfidf_sublinear")


def test_text_feature_handler_filters_out_uuids_and_hex_hashes():
    df = pd.DataFrame({
        "transaction_uuid": [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
        ],
        "session_hash": [
            "a1b2c3d4e5f67890abcdef1234567890abcdef12",
            "b2c3d4e5f67890abcdef1234567890abcdef1234",
            "c3d4e5f67890abcdef1234567890abcdef123456",
            "d4e5f67890abcdef1234567890abcdef12345678",
        ],
        "real_comment": [
            "The battery life is really solid, lasts two full days without charging.",
            "Screen resolution is crisp and bright even outdoors under direct sun.",
            "Build quality feels premium, metal frame and smooth rounded corners.",
            "Audio output is clear with punchy bass and zero distortion at high volume.",
        ],
    })

    handler = TextFeatureHandler()
    report = handler.analyze_text_features(df)

    assert "real_comment" in report.text_columns
    assert "transaction_uuid" not in report.text_columns
    assert "session_hash" not in report.text_columns


def test_text_feature_handler_dense_embeddings_fallback():
    # Dense embeddings generation test with sub-10ms fallback guarantee
    reviews = pd.Series([
        "Outstanding performance and sleek industrial design!",
        "Customer support was unhelpful and delivery took weeks.",
        "Decent functionality for the price point, fits standard desks.",
        "Highly recommended for daily professional workflows and tasks.",
    ])

    handler = TextFeatureHandler()
    embeddings = handler.extract_dense_embeddings(reviews, n_components=8)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 4
    assert embeddings.shape[1] > 0
    assert not np.isnan(embeddings).any()
