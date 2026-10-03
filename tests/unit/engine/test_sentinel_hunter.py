"""Unit tests for Sentinel Hunter with Dirac-Delta isolated mass and enterprise defaults."""
import numpy as np
import pandas as pd
import pytest

from ml_mcp.engine.sentinel_hunter import SentinelHunter


def test_sentinel_hunter_standard_and_enterprise_sentinels():
    df = pd.DataFrame({
        "age": [25, 30, -999, 45, 999],
        "income": [50000.0, -999.0, 75000.0, 9999.0, 60000.0],
        "gender": ["M", "F", "?", "NULL", "N/A"],
        "date_joined": ["2023-01-01", "1900-01-01", "2023-05-15", "1970-01-01", "2024-02-01"],
    })

    hunter = SentinelHunter()
    cleaned_df, report = hunter.mask_sentinels(df)

    assert cleaned_df["age"].isna().sum() == 2
    assert cleaned_df["income"].isna().sum() == 2
    assert cleaned_df["gender"].isna().sum() == 3
    assert cleaned_df["date_joined"].isna().sum() == 2  # 1900-01-01 and 1970-01-01 masked


def test_sentinel_hunter_dirac_delta_isolated_mass():
    """Verify Dirac-Delta mass spike far from median is dynamically masked without static list."""
    np.random.seed(42)
    n = 200
    # Normal distribution centered at 50, MAD ~ 5
    vals = list(np.random.normal(50, 5, 185))
    # Add a 7.5% frequency spike at 888.0 (extreme isolated Dirac mass)
    vals.extend([888.0] * 15)

    df = pd.DataFrame({"sensor_reading": vals})
    hunter = SentinelHunter(spike_threshold=0.05)
    cleaned_df, report = hunter.mask_sentinels(df)

    # 888.0 was not in any static set, but Dirac boundary detection masked it!
    assert cleaned_df["sensor_reading"].isna().sum() == 15
