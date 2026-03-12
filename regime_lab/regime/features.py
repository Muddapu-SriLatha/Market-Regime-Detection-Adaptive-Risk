"""
Regime Feature Construction Module
==================================
Constructs strictly backward-looking statistical features for regime classification:
1. Short-term Realized Volatility (21 trading days, annualized)
2. Long-term Realized Volatility (63 trading days, annualized)
3. Volatility Term-Structure Ratio (Short-vol / Long-vol)
4. Risk-Adjusted Momentum Score (21-day return normalized by 21-day vol)
5. Parkinson High-Low Volatility (21-day rolling)

All calculations are strictly causal: feature at time t uses only observations <= t.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RegimeFeatures:
    """Encapsulates computed feature matrix alongside metadata."""
    feature_matrix: pd.DataFrame
    feature_names: list
    dates: pd.DatetimeIndex
    short_vol_window: int
    long_vol_window: int
    momentum_window: int


def compute_regime_features(
    prices: pd.Series,
    highs: Optional[pd.Series] = None,
    lows: Optional[pd.Series] = None,
    short_vol_window: int = 21,
    long_vol_window: int = 63,
    momentum_window: int = 21,
    annualization_factor: float = np.sqrt(252.0)
) -> RegimeFeatures:
    """
    Compute backward-looking features for regime clustering.

    Parameters
    ----------
    prices : pd.Series
        Historical close or adjusted close prices with DatetimeIndex.
    highs : pd.Series, optional
        Historical high prices.
    lows : pd.Series, optional
        Historical low prices.
    short_vol_window : int, default 21
        Lookback for short-term realized volatility (~1 trading month).
    long_vol_window : int, default 63
        Lookback for long-term realized volatility (~3 trading months).
    momentum_window : int, default 21
        Lookback for trailing momentum.
    annualization_factor : float, default sqrt(252)
        Multiplier to annualize daily volatility.

    Returns
    -------
    RegimeFeatures
        Container with causal feature dataframe and metadata.
    """
    if len(prices) < long_vol_window:
        raise ValueError(f"Need at least {long_vol_window} price bars to calculate regime features.")

    # 1. Log returns: ln(P_t / P_{t-1})
    log_returns = np.log(prices / prices.shift(1))

    # 2. Short-term realized volatility (annualized)
    # sigma_short = sqrt(252) * std(log_returns, ddof=1) over 21 days
    short_vol = log_returns.rolling(window=short_vol_window, min_periods=short_vol_window).std(ddof=1) * annualization_factor
    short_vol.name = "short_vol"

    # 3. Long-term realized volatility (annualized)
    # sigma_long = sqrt(252) * std(log_returns, ddof=1) over 63 days
    long_vol = log_returns.rolling(window=long_vol_window, min_periods=long_vol_window).std(ddof=1) * annualization_factor
    long_vol.name = "long_vol"

    # 4. Volatility Term-Structure Ratio (Short Vol / Long Vol)
    # Identifies volatility spikes and expansions vs quiet contractions
    vol_ratio = short_vol / np.maximum(long_vol, 1e-6)
    vol_ratio.name = "vol_ratio"

    # 5. Risk-adjusted Momentum Score
    # Trailing 21-day log return normalized by the 21-day volatility
    cum_ret_21 = log_returns.rolling(window=momentum_window, min_periods=momentum_window).sum()
    daily_vol_21 = log_returns.rolling(window=momentum_window, min_periods=momentum_window).std(ddof=1)
    norm_momentum = cum_ret_21 / (np.maximum(daily_vol_21, 1e-6) * np.sqrt(momentum_window))
    norm_momentum.name = "norm_momentum"

    # 6. Assemble feature dictionary
    feature_dict = {
        "short_vol": short_vol,
        "long_vol": long_vol,
        "vol_ratio": vol_ratio,
        "norm_momentum": norm_momentum
    }

    if highs is not None and lows is not None:
        hl_ratio = np.maximum(highs / np.maximum(lows, 1e-6), 1.0)
        parkinson_daily_var = (1.0 / (4.0 * np.log(2.0))) * (np.log(hl_ratio) ** 2)
        parkinson_21 = np.sqrt(parkinson_daily_var.rolling(window=short_vol_window, min_periods=short_vol_window).mean()) * annualization_factor
        parkinson_21.name = "parkinson_vol_21"
        feature_dict["parkinson_vol_21"] = parkinson_21

    features_df = pd.DataFrame(feature_dict, index=prices.index)
    feature_names = list(feature_dict.keys())

    return RegimeFeatures(
        feature_matrix=features_df,
        feature_names=feature_names,
        dates=prices.index,
        short_vol_window=short_vol_window,
        long_vol_window=long_vol_window,
        momentum_window=momentum_window
    )
