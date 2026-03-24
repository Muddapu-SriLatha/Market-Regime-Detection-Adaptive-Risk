"""
Strategy A: Fixed-Risk Baseline
================================
Maintains a constant target exposure (1.0x) whenever the underlying trend signal
is bullish, and 0.0x (cash) when bearish.
"""

from typing import Optional
import numpy as np
import pandas as pd


class FixedRiskStrategy:
    """
    Fixed-Risk baseline strategy maintaining static 1.0x exposure to the underlying signal.
    """

    def __init__(self, target_exposure: float = 1.0, name: str = "Strategy A (Fixed Risk)"):
        self.target_exposure = target_exposure
        self.name = name

    def generate_target_exposure(
        self,
        base_signal: pd.Series,
        regimes: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generate target portfolio exposure at bar close t.

        Parameters
        ----------
        base_signal : pd.Series
            Underlying trading signal (e.g. 1.0 for bullish, 0.0 for bearish).
        regimes : pd.Series, optional
            Ignored by FixedRiskStrategy to preserve constant risk allocation.

        Returns
        -------
        pd.Series
            Target exposure series [0.0, 1.0] at bar close t.
        """
        target = base_signal * self.target_exposure
        target.name = "exposure_fixed"
        return target
