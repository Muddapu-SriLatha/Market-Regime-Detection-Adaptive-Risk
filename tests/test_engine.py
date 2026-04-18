"""Unit tests for backtest execution lag, turnover, and transaction costs."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.engine.cost_model import TransactionCostModel
from regime_lab.engine.backtest import BacktestEngine


def test_strict_one_bar_execution_lag():
    """Verify that target exposure at close t is executed at t+1 and earns return t+1."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    # Prices: 100 -> 110 (+10%) -> 121 (+10%) -> 121 (0%)
    prices = pd.Series([100.0, 110.0, 121.0, 121.0], index=dates)

    # Signal generated at t=0: 1.0 (should NOT earn return 100->110)
    # Signal generated at t=1: 1.0 (should earn return 110->121)
    # Signal generated at t=2: 0.0 (should exit at t=3)
    target_exp = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)

    engine = BacktestEngine(cost_model=TransactionCostModel(0, 0, 0))  # zero cost
    res = engine.run(prices, target_exp)

    # At t=0: pos = 0, return = 0
    # At t=1: pos = 1.0, return = +10%
    # At t=2: pos = 1.0, return = +10%
    # At t=3: pos = 0.0, return = 0%
    expected_positions = np.array([0.0, 1.0, 1.0, 0.0])
    expected_gross_returns = np.array([0.0, 0.10, 0.10, 0.0])

    np.testing.assert_allclose(res.positions.values, expected_positions)
    np.testing.assert_allclose(res.gross_returns.values, expected_gross_returns, atol=1e-7)


def test_turnover_and_cost_deduction():
    """Verify turnover computation and exact linear friction deduction."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=dates)  # flat prices

    # Target exposure: 0 -> 1.0 -> 0.5 -> 0.0
    target_exp = pd.Series([1.0, 0.5, 0.0, 0.0], index=dates)

    # Total linear friction: 10 bps = 0.0010
    cost_model = TransactionCostModel(commission_bps=2.0, half_spread_bps=4.0, slippage_bps=4.0)
    assert np.isclose(cost_model.total_linear_rate, 0.0010)

    engine = BacktestEngine(cost_model=cost_model)
    res = engine.run(prices, target_exp)

    # Positions: [0.0, 1.0, 0.5, 0.0]
    # Turnover: [0.0, 1.0, 0.5, 0.5] -> total turnover = 2.0
    expected_turnover = np.array([0.0, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(res.turnover.values, expected_turnover)

    # Expected cost per bar = turnover * 0.0010
    expected_costs = expected_turnover * 0.0010
    # Net returns = 0 - costs = -expected_costs
    np.testing.assert_allclose(res.net_returns.values, -expected_costs, atol=1e-8)
