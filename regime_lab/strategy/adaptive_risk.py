"""
Strategy B: Adaptive-Risk Strategy
==================================
Modulates portfolio exposure according to the statistically identified market regime:
- Regime 0 (Low Risk / Calm Bull): 1.0x full exposure
- Regime 1 (Medium Risk / Transitional): 0.6x moderate defensive de-risking
- Regime 2 (High Risk / Turbulent Crisis): 0.2x high defensive de-risking
- Bearish base signal: 0.0x cash

Uses the EXACT same base signal as Strategy A to ensure perfectly fair comparison.
Uses only current-bar regime information (which was computed from observations <= t).
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class AdaptiveRiskStrategy:
    """
    Adaptive-Risk strategy that scales position exposure dynamically by regime state.
    """

    def __init__(
        self,
        exposure_map: Optional[Dict[int, float]] = None,
        default_exposure: float = 1.0,
        name: str = "Strategy B (Adaptive Risk)"
    ):
        self.exposure_map = exposure_map or {
            0: 1.0,  # Low Risk
            1: 0.6,  # Medium Risk
            2: 0.2   # High Risk
        }
        self.default_exposure = default_exposure
        self.name = name

    def generate_target_exposure(
        self,
        base_signal: pd.Series,
        regimes: pd.Series
    ) -> pd.Series:
        """
        Generate target portfolio exposure at bar close t based on current regime.

        Parameters
        ----------
        base_signal : pd.Series
            Underlying trading signal (1.0 = bullish, 0.0 = bearish).
        regimes : pd.Series
            Detected regime labels (0, 1, 2) available at close t.

        Returns
        -------
        pd.Series
            Target exposure series [0.0, 1.0] at bar close t.
        """
        if len(base_signal) != len(regimes):
            raise ValueError(
                f"Length mismatch: base_signal ({len(base_signal)}) != regimes ({len(regimes)})"
            )

        # Map regime labels to exposure multipliers
        # Fallback to default_exposure if regime is NaN (e.g. during warmup)
        multipliers = pd.Series(self.default_exposure, index=regimes.index, dtype=float)
        for r, mult in self.exposure_map.items():
            multipliers.loc[regimes == r] = mult

        target = base_signal * multipliers
        target.name = "exposure_adaptive"
        return target
