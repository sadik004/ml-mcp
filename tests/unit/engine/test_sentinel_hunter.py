"""Unit tests for Sentinel Hunter with false-positive guard."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.sentinel_hunter import SentinelHunter


def test_sentinel_hunter_standard_sentinels():
    df = pd.DataFrame({
        "age": [25, 30, -999, 45, 999],
        "income": [50000.0, -999.0, 75000.0, 9999.0, 60000.0],
        "gender": ["M", "F", "?", "NULL", "N/A"],
    })

    hunter = SentinelHunter()
    cleaned_df, report = hunter.mask_sentinels(df)

    # All standard numeric and categorical sentinels masked
    assert cleaned_df["age"].isna().sum() == 2
    assert cleaned_df["income"].isna().sum() == 2  # -999.0 and 9999.0
    assert cleaned_df["gender"].isna().sum() == 3  # "?", "NULL", "N/A"
    assert report["total_sentinels_masked"] == 7


def test_sentinel_hunter_false_positive_guard_temperature():
    # Genuine negative-scale data: temperature in Celsius
    # -1 is a valid temperature, not a sentinel!
    df = pd.DataFrame({
        "temperature": [-10.5, -5.0, -1.0, 0.0, 4.5, 12.0, -1.0, 15.0],
    })

    hunter = SentinelHunter()
    cleaned_df, report = hunter.mask_sentinels(df)

    # -1.0 must NOT be converted to NaN because column contains genuine negative values
    assert cleaned_df["temperature"].isna().sum() == 0
    assert (-1.0 in cleaned_df["temperature"].values)


def test_sentinel_hunter_positive_scale_minus_one_masked():
    # Strictly positive scale data: age (0-100) or count
    # -1 is used as missing code
    df = pd.DataFrame({
        "age": [22, 35, 40, -1, 55, 60, -1, 30, 28, 45],
    })

    hunter = SentinelHunter()
    cleaned_df, report = hunter.mask_sentinels(df)

    # -1 must be masked as NaN because all valid ages are >= 0
    assert cleaned_df["age"].isna().sum() == 2
    assert -1 not in cleaned_df["age"].dropna().values
