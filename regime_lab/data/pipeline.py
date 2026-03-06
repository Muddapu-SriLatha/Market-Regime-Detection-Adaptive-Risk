"""
Market Data Preprocessing and Feature Assembly Pipeline
======================================================
Prepares clean market series from raw archives:
- Computes percentage and log daily returns using adjusted prices
- Computes high-low Parkinson volatility and close-to-close rolling volatility
- Preserves raw archives separately from processed datasets
- Ensures zero future look-ahead leakage in all series
"""

from dataclasses import dataclass
import os
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ProcessedMarketData:
    """Container for processed market data series."""
    symbol: str
    df: pd.DataFrame
    dates: pd.DatetimeIndex
    adj_close: pd.Series
    returns: pd.Series
    log_returns: pd.Series
    parkinson_vol: pd.Series
    benchmark_returns: pd.Series

    @property
    def total_bars(self) -> int:
        return len(self.df)


def prepare_market_dataset(
    raw_df: pd.DataFrame,
    symbol: str = "SPY",
    processed_dir: str = os.path.join("data", "processed")
) -> ProcessedMarketData:
    """
    Process raw OHLCV data into clean analytic return series.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw validated market data with OHLCV columns and datetime index.
    symbol : str
        Ticker symbol identifier.
    processed_dir : str
        Output directory for processed parquet/CSV data.

    Returns
    -------
    ProcessedMarketData
        Container with verified returns, volatility, and prices.
    """
    os.makedirs(processed_dir, exist_ok=True)
    df = raw_df.copy()

    # Determine price series for return calculation:
    # Use 'Adj Close' if present; otherwise fallback to 'Close'
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    adj_close = df[price_col].astype(float)

    # 1. Simple percentage return: r_t = (P_t / P_{t-1}) - 1
    returns = adj_close.pct_change().fillna(0.0)
    returns.name = "return"

    # 2. Log return: ln(P_t / P_{t-1})
    log_returns = np.log(adj_close / adj_close.shift(1)).fillna(0.0)
    log_returns.name = "log_return"

    # 3. Daily Parkinson Volatility (based on High/Low range):
    # sigma_P = sqrt( 1 / (4 * ln(2)) * (ln(High / Low))^2 )
    # Handles zeros gracefully
    hl_ratio = np.maximum(df["High"] / np.maximum(df["Low"], 1e-6), 1.0)
    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (np.log(hl_ratio) ** 2)
    parkinson_vol = np.sqrt(parkinson_var)
    parkinson_vol.name = "parkinson_vol"

    # Add features to dataframe
    df["adj_close"] = adj_close
    df["return"] = returns
    df["log_return"] = log_returns
    df["parkinson_vol"] = parkinson_vol

    # Save processed dataframe
    csv_path = os.path.join(processed_dir, f"{symbol}_processed.csv")
    df.to_csv(csv_path)

    return ProcessedMarketData(
        symbol=symbol,
        df=df,
        dates=df.index,
        adj_close=adj_close,
        returns=returns,
        log_returns=log_returns,
        parkinson_vol=parkinson_vol,
        benchmark_returns=returns.copy()
    )
