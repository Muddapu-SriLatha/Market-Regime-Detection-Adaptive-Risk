"""
Transaction Cost & Friction Modeling Module
============================================
Accurately models trading frictions for institutional and retail liquid ETF execution:
- Proportional Brokerage Commission (basis points)
- Bid-Ask Spread Cost (half-spread incurred per unit of turnover)
- Execution Slippage (basis points)
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd


@dataclass
class TransactionCostModel:
    """
    Linear transaction friction model.

    Parameters
    ----------
    commission_bps : float, default 1.0
        Commission in basis points (1 bp = 0.0001 = 0.01%).
    half_spread_bps : float, default 2.0
        Half the effective bid-ask spread in basis points.
    slippage_bps : float, default 2.0
        Expected execution slippage in basis points.
    """
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 2.0

    @property
    def total_linear_bps(self) -> float:
        """Total linear basis points deducted per unit of portfolio turnover."""
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    @property
    def total_linear_rate(self) -> float:
        """Total linear friction as a decimal rate (e.g. 5 bps -> 0.0005)."""
        return self.total_linear_bps * 1e-4

    def calculate_cost(
        self,
        turnover: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        """
        Calculate cost drag incurred for a given turnover series.
        
        Formula:
            cost_drag = turnover * total_linear_rate
        """
        return turnover * self.total_linear_rate
