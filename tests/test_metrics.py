"""Unit tests for performance metrics with hand-calculated deterministic benchmarks."""
import numpy as np
import pytest

from regime_lab.engine.metrics import (
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_drawdown_series,
    calculate_calmar_ratio,
    calculate_all_metrics
)


def test_deterministic_cagr():
    """Verify CAGR against exact compound growth formula."""
    # Doubling wealth in 252 days = 100% CAGR
    # r = 2^(1/252) - 1 per day for 252 days
    daily_r = (2.0 ** (1.0 / 252.0)) - 1.0
    returns = np.full(252, daily_r)
    cagr = calculate_cagr(returns, periods_per_year=252)
    assert np.isclose(cagr, 1.0, atol=1e-5)


def test_drawdown_series_exact():
    """Verify drawdown calculation on deterministic hand-calculated sequence."""
    # Wealth: 100 -> 120 -> 90 -> 108
    # Returns: [0.20, -0.25, 0.20]
    # Peak:    [120,  120,   120]
    # Drawdowns: [0.0, -25%, -10%]
    # Max Drawdown: 25% = 0.25
    returns = np.array([0.20, -0.25, 0.20])
    dd_series, max_dd, max_dur = calculate_drawdown_series(returns)

    assert np.isclose(max_dd, 0.25, atol=1e-6)
    assert np.isclose(dd_series[1], -0.25, atol=1e-6)
    assert np.isclose(dd_series[2], -0.10, atol=1e-6)


def test_sharpe_ratio_deterministic():
    """Verify annualized Sharpe ratio formula."""
    # Daily mean return = 0.001, Daily std = 0.01, rf = 0
    # Annualized Sharpe = sqrt(252) * 0.001 / 0.01 = 0.1 * sqrt(252) ~ 1.58745
    np.random.seed(42)
    daily_ret = np.array([0.011, 0.001, -0.009, 0.003, 0.009, -0.005, 0.008, 0.002])
    mean_r = np.mean(daily_ret)
    std_r = np.std(daily_ret, ddof=1)
    expected_sharpe = np.sqrt(252.0) * mean_r / std_r

    calc_sharpe = calculate_sharpe_ratio(daily_ret, risk_free_rate=0.0, periods_per_year=252)
    assert np.isclose(calc_sharpe, expected_sharpe, atol=1e-6)
