"""
Vectorized Backtesting Engine
=============================
Simulates strategy performance with strict protection against look-ahead bias:
- Target exposure chosen at bar close t is executed at bar open/close t+1
- Position held during bar t is strictly: position[t] = target_exposure[t - 1]
- Deducts exact linear transaction frictions per unit of turnover
- Tracks gross and net equity, drawdowns, and turnover
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from regime_lab.engine.cost_model import TransactionCostModel
from regime_lab.engine.metrics import calculate_all_metrics, PerformanceMetrics, calculate_drawdown_series


@dataclass
class BacktestResult:
    """Encapsulates backtest simulation output and analytics."""
    strategy_name: str
    dates: pd.DatetimeIndex
    prices: pd.Series
    target_exposure: pd.Series
    positions: pd.Series
    turnover: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    gross_equity: pd.Series
    net_equity: pd.Series
    gross_metrics: PerformanceMetrics
    net_metrics: PerformanceMetrics
    drawdowns: pd.Series
    cost_model: TransactionCostModel

    def summary_table(self) -> pd.DataFrame:
        """Return formatted gross vs net performance comparison."""
        rows = [
            ("Cumulative Net Return", f"{self.net_metrics.total_return:.2%}"),
            ("CAGR (Annualized)", f"{self.net_metrics.cagr:.2%}"),
            ("Annualized Volatility", f"{self.net_metrics.annualized_volatility:.2%}"),
            ("Sharpe Ratio (Rf=0%)", f"{self.net_metrics.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{self.net_metrics.sortino_ratio:.3f}"),
            ("Max Drawdown", f"{self.net_metrics.max_drawdown:.2%}"),
            ("Calmar Ratio", f"{self.net_metrics.calmar_ratio:.3f}"),
            ("Annualized Turnover", f"{self.net_metrics.annualized_turnover:.2f}x"),
            ("Total Friction Drag", f"{self.net_metrics.total_cost_drag * 1e4:.1f} bps"),
            ("Win Rate", f"{self.net_metrics.win_rate:.2%}"),
            ("Profit Factor", f"{self.net_metrics.profit_factor:.3f}")
        ]
        return pd.DataFrame(rows, columns=["Metric", self.strategy_name])


class BacktestEngine:
    """
    Simulates portfolio equity series under strict chronological execution constraints.
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        prices: pd.Series,
        target_exposure: pd.Series,
        strategy_name: str = "Strategy"
    ) -> BacktestResult:
        """
        Execute backtest.

        Parameters
        ----------
        prices : pd.Series
            Asset close/adjusted close series with DatetimeIndex.
        target_exposure : pd.Series
            Target exposure generated at bar close t.
        strategy_name : str
            Display label.
        """
        if len(prices) != len(target_exposure):
            raise ValueError(f"Prices ({len(prices)}) != target_exposure ({len(target_exposure)})")
        if len(prices) < 3:
            raise ValueError("Need at least 3 price bars to backtest.")

        dates = prices.index
        p_arr = prices.values.astype(float)
        exp_arr = target_exposure.values.astype(float)
        n = len(p_arr)

        # 1. Underlying asset return: r_t = (P_t / P_{t-1}) - 1
        asset_returns = np.zeros(n)
        asset_returns[1:] = (p_arr[1:] / p_arr[:-1]) - 1.0

        # 2. Strict 1-bar execution lag:
        # Position held during bar t is based on target exposure at t-1
        positions = np.zeros(n)
        positions[1:] = exp_arr[:-1]

        # 3. Strategy gross return
        gross_returns = positions * asset_returns

        # 4. Turnover: |pos[t] - pos[t-1]|
        prev_positions = np.zeros(n)
        prev_positions[1:] = positions[:-1]
        turnover = np.abs(positions - prev_positions)

        # 5. Transaction costs
        cost_drag = self.cost_model.calculate_cost(turnover)
        net_returns = gross_returns - cost_drag

        # 6. Cumulative equity index (starting at 1.0)
        gross_equity = np.cumprod(1.0 + gross_returns)
        net_equity = np.cumprod(1.0 + net_returns)

        # 7. Drawdown series
        dd_series, _, _ = calculate_drawdown_series(net_returns)

        # 8. Metrics
        gross_metrics = calculate_all_metrics(
            gross_returns,
            turnover=turnover,
            cost_drag=np.zeros(n),
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        net_metrics = calculate_all_metrics(
            net_returns,
            turnover=turnover,
            cost_drag=cost_drag,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )

        return BacktestResult(
            strategy_name=strategy_name,
            dates=dates,
            prices=pd.Series(p_arr, index=dates),
            target_exposure=pd.Series(exp_arr, index=dates),
            positions=pd.Series(positions, index=dates),
            turnover=pd.Series(turnover, index=dates),
            gross_returns=pd.Series(gross_returns, index=dates),
            net_returns=pd.Series(net_returns, index=dates),
            gross_equity=pd.Series(gross_equity, index=dates),
            net_equity=pd.Series(net_equity, index=dates),
            gross_metrics=gross_metrics,
            net_metrics=net_metrics,
            drawdowns=pd.Series(dd_series, index=dates),
            cost_model=self.cost_model
        )
