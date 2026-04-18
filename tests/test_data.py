"""Unit tests for data acquisition, return calculations, and validation."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.data.pipeline import prepare_market_dataset


def test_return_calculations_exact():
    """Verify that percentage and log returns match exact analytical formulas."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    # Prices: 100, 110, 99, 99
    # Returns: NaN/0, +10%, -10%, 0%
    raw_df = pd.DataFrame({
        "Open": [100.0, 105.0, 108.0, 99.0],
        "High": [102.0, 112.0, 110.0, 100.0],
        "Low": [98.0, 104.0, 97.0, 98.0],
        "Close": [100.0, 110.0, 99.0, 99.0],
        "Adj Close": [100.0, 110.0, 99.0, 99.0],
        "Volume": [1000, 1200, 1100, 900]
    }, index=dates)

    dataset = prepare_market_dataset(raw_df, symbol="TEST")
    expected_pct = np.array([0.0, 0.10, -0.10, 0.0])
    expected_log = np.array([0.0, np.log(1.10), np.log(0.90), 0.0])

    np.testing.assert_allclose(dataset.returns.values, expected_pct, atol=1e-7)
    np.testing.assert_allclose(dataset.log_returns.values, expected_log, atol=1e-7)


def test_parkinson_volatility():
    """Verify daily Parkinson volatility formula against hand calculation."""
    # High = 110, Low = 100 -> HL ratio = 1.10
    # var = (1 / (4 * ln 2)) * (ln 1.10)^2
    hl = 1.10
    expected_vol = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(hl) ** 2))

    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    raw_df = pd.DataFrame({
        "Open": [100.0, 100.0],
        "High": [100.0, 110.0],
        "Low": [100.0, 100.0],
        "Close": [100.0, 105.0],
        "Adj Close": [100.0, 105.0],
        "Volume": [1000, 1000]
    }, index=dates)

    dataset = prepare_market_dataset(raw_df, symbol="TEST")
    assert np.isclose(dataset.parkinson_vol.iloc[1], expected_vol, atol=1e-6)
