"""
Underlying Trading Signal Module
================================
Generates the base trading signal shared identically by both:
- Strategy A: Fixed-Risk Baseline
- Strategy B: Adaptive-Risk Strategy

Ensures perfectly fair comparison by isolating the sole variable under research:
the regime-dependent exposure multiplier.
"""

from typing import Optional
import numpy as np
import pandas as pd


def compute_trend_signal(
    prices: pd.Series,
    window: int = 200
) -> pd.Series:
    """
    Compute long-term trend filter signal based on 200-day Simple Moving Average.
    
    Formula:
        Signal_t = 1.0 if Close_t > SMA_200_t else 0.0 (Cash)
    
    Parameters
    ----------
    prices : pd.Series
        Historical close or adjusted close prices.
    window : int, default 200
        Moving average window length in trading bars.

    Returns
    -------
    pd.Series
        Binary base trend signal series {0.0, 1.0} with DatetimeIndex.
    """
    if len(prices) < window:
        raise ValueError(f"Need at least {window} price bars to compute {window}-day trend signal.")

    sma = prices.rolling(window=window, min_periods=window).mean()
    signal = pd.Series(0.0, index=prices.index, dtype=float)

    # Bullish when close strictly above SMA, else 0 (cash)
    valid_mask = ~sma.isna()
    signal.loc[valid_mask] = np.where(prices[valid_mask] > sma[valid_mask], 1.0, 0.0)

    # Warmup period is NaN / 0.0
    signal.loc[~valid_mask] = 0.0
    signal.name = f"trend_signal_{window}"
    return signal
