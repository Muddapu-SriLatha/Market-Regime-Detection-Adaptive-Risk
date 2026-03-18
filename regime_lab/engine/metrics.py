"""
Performance and Risk Analytics Module
=====================================
Calculates standard institutional risk-adjusted return and drawdown statistics.
Handles all numerical edge cases safely (e.g. zero volatility, negative equity).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    calmar_ratio: float
    annualized_turnover: float
    total_cost_drag: float
    win_rate: float
    profit_factor: float
    downside_volatility: float
    total_bars: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def sanitize_returns(returns: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Validate returns and remove NaNs/Infs."""
    arr = np.asarray(returns, dtype=float)
    clean = arr[np.isfinite(arr)]
    return clean


def calculate_cumulative_return(returns: np.ndarray) -> float:
    """Compute total compounded return over the entire period."""
    if len(returns) == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    return float(wealth[-1] - 1.0)


def calculate_cagr(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Formula:
        CAGR = (1 + TotalReturn) ^ (periods_per_year / N) - 1
    """
    n = len(returns)
    if n == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    final_wealth = wealth[-1]
    if final_wealth <= 0:
        return -1.0
    return float((final_wealth ** (periods_per_year / n)) - 1.0)


def calculate_annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate sample annualized volatility."""
    if len(returns) < 2:
        return 0.0
    daily_std = float(np.std(returns, ddof=1))
    return float(daily_std * np.sqrt(periods_per_year))


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio using daily excess returns.
    
    Formula:
        Sharpe = sqrt(periods_per_year) * (mean(r) - rf_daily) / std(r)
    """
    if len(returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / periods_per_year
    excess = returns - rf_daily
    daily_std = np.std(returns, ddof=1)
    if daily_std < 1e-8:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / daily_std)


def calculate_downside_volatility(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized semi-deviation / downside risk below target_return."""
    if len(returns) < 2:
        return 0.0
    downside_diff = np.minimum(returns - target_return, 0.0)
    downside_var = np.mean(downside_diff ** 2)
    return float(np.sqrt(downside_var * periods_per_year))


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / periods_per_year
    excess = returns - rf_daily
    downside_vol = calculate_downside_volatility(returns, target_return=rf_daily, periods_per_year=periods_per_year)
    if downside_vol < 1e-8:
        return 0.0
    return float((np.mean(excess) * periods_per_year) / downside_vol)


def calculate_drawdown_series(returns: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Calculate underwater drawdown series, maximum drawdown, and max duration.

    Returns
    -------
    Tuple[np.ndarray, float, int]
        Drawdown series [0.0, -1.0], Max Drawdown, Max Drawdown Duration (bars).
    """
    if len(returns) == 0:
        return np.array([]), 0.0, 0

    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peak) / peak

    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Drawdown duration tracking
    current_duration = 0
    max_duration = 0
    for dd in drawdowns:
        if dd < 0:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0

    return drawdowns, abs(max_dd), max_duration


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio = CAGR / |MaxDrawdown|."""
    if max_drawdown < 1e-6:
        return 0.0
    return float(cagr / max_drawdown)


def calculate_all_metrics(
    returns: Union[np.ndarray, pd.Series],
    turnover: Optional[Union[np.ndarray, pd.Series]] = None,
    cost_drag: Optional[Union[np.ndarray, pd.Series]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Compute full institutional performance profile.
    """
    arr = sanitize_returns(returns)
    n = len(arr)

    tot_ret = calculate_cumulative_return(arr)
    cagr = calculate_cagr(arr, periods_per_year=periods_per_year)
    ann_vol = calculate_annualized_volatility(arr, periods_per_year=periods_per_year)
    sharpe = calculate_sharpe_ratio(arr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(arr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    _, max_dd, max_dd_bars = calculate_drawdown_series(arr)
    calmar = calculate_calmar_ratio(cagr, max_dd)
    downside_vol = calculate_downside_volatility(arr, periods_per_year=periods_per_year)

    # Win rate & Profit factor
    pos = arr[arr > 0]
    neg = arr[arr < 0]
    win_rate = float(len(pos) / n) if n > 0 else 0.0
    sum_gains = float(np.sum(pos)) if len(pos) > 0 else 0.0
    sum_losses = abs(float(np.sum(neg))) if len(neg) > 0 else 0.0
    profit_factor = float(sum_gains / sum_losses) if sum_losses > 1e-8 else (10.0 if sum_gains > 0 else 0.0)

    # Turnover & Cost drag
    if turnover is not None:
        t_arr = np.asarray(turnover, dtype=float)
        tot_turnover = float(np.sum(t_arr))
        ann_turnover = float(tot_turnover * (periods_per_year / n)) if n > 0 else 0.0
    else:
        ann_turnover = 0.0

    if cost_drag is not None:
        tot_drag = float(np.sum(cost_drag))
    else:
        tot_drag = 0.0

    return PerformanceMetrics(
        total_return=tot_ret,
        cagr=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_bars=max_dd_bars,
        calmar_ratio=calmar,
        annualized_turnover=ann_turnover,
        total_cost_drag=tot_drag,
        win_rate=win_rate,
        profit_factor=profit_factor,
        downside_volatility=downside_vol,
        total_bars=n
    )
