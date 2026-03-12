"""
Historical Volatility Quantile Threshold Regime Detector
========================================================
Benchmark statistical regime detector using empirical realized volatility thresholds.
Thresholds are computed strictly on the in-sample training period:
- Regime 0 (Low Risk): Volatility <= 33.3rd percentile of training volatility
- Regime 1 (Medium Risk): 33.3rd percentile < Volatility <= 66.7th percentile
- Regime 2 (High Risk): Volatility > 66.7th percentile
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class VolThresholdSummary:
    """Summary of volatility quantile thresholds."""
    low_threshold: float
    high_threshold: float
    training_frequencies: Dict[int, float]


class VolatilityThresholdDetector:
    """
    Classifies regimes based on fixed empirical quantile thresholds fitted on training data.
    """

    def __init__(self, low_quantile: float = 0.3333, high_quantile: float = 0.6667):
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.low_threshold: Optional[float] = None
        self.high_threshold: Optional[float] = None
        self.is_fitted: bool = False
        self.summary_: Optional[VolThresholdSummary] = None

    def fit(self, train_features: pd.DataFrame) -> "VolatilityThresholdDetector":
        """
        Compute volatility quantile cutoffs strictly on training data.
        """
        if "short_vol" not in train_features.columns:
            raise ValueError("Features must include 'short_vol' column.")

        clean_vol = train_features["short_vol"].dropna()
        if len(clean_vol) < 50:
            raise ValueError(f"Insufficient training observations ({len(clean_vol)}) for quantile calculation.")

        self.low_threshold = float(np.percentile(clean_vol, self.low_quantile * 100.0))
        self.high_threshold = float(np.percentile(clean_vol, self.high_quantile * 100.0))

        # Training frequencies
        r0 = np.mean(clean_vol <= self.low_threshold)
        r2 = np.mean(clean_vol > self.high_threshold)
        r1 = 1.0 - r0 - r2

        self.summary_ = VolThresholdSummary(
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            training_frequencies={0: float(r0), 1: float(r1), 2: float(r2)}
        )
        self.is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Assign regime labels (0, 1, 2) based on frozen training thresholds.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        vol = features["short_vol"]
        labels = pd.Series(index=features.index, dtype=float)
        labels[:] = np.nan

        valid_mask = ~vol.isna()
        v = vol[valid_mask]

        preds = np.zeros(len(v), dtype=float)
        preds[v > self.low_threshold] = 1.0
        preds[v > self.high_threshold] = 2.0

        labels.loc[valid_mask] = preds
        return labels
