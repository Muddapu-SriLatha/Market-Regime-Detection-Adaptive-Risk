"""Unit tests for strategy construction, signal fairness, and exposure constraints."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.strategy.signals import compute_trend_signal
from regime_lab.strategy.fixed_risk import FixedRiskStrategy
from regime_lab.strategy.adaptive_risk import AdaptiveRiskStrategy


def test_signal_fairness_and_identical_base():
    """Verify that both strategies receive identical base signals and bounds."""
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    # Linear price trend: 100 to 350
    prices = pd.Series(np.linspace(100, 350, 250), index=dates)

    base_sig = compute_trend_signal(prices, window=50)

    # Regimes: cycle 0, 1, 2
    regimes = pd.Series(np.tile([0, 1, 2], len(dates))[:len(dates)], index=dates)

    strat_a = FixedRiskStrategy(target_exposure=1.0)
    strat_b = AdaptiveRiskStrategy(exposure_map={0: 1.0, 1: 0.6, 2: 0.2})

    exp_a = strat_a.generate_target_exposure(base_sig)
    exp_b = strat_b.generate_target_exposure(base_sig, regimes)

    # Exposure constraints: 0 <= exposure <= 1.0
    assert (exp_a >= 0.0).all() and (exp_a <= 1.0).all()
    assert (exp_b >= 0.0).all() and (exp_b <= 1.0).all()

    # When base signal is 0.0 (bearish / cash), both MUST have 0.0 exposure
    bear_mask = (base_sig == 0.0)
    assert (exp_a[bear_mask] == 0.0).all()
    assert (exp_b[bear_mask] == 0.0).all()

    # When base signal is 1.0 (bullish), Strategy A has 1.0, Strategy B has regime multiplier
    bull_mask = (base_sig == 1.0)
    assert (exp_a[bull_mask] == 1.0).all()

    r0_bull = bull_mask & (regimes == 0)
    r1_bull = bull_mask & (regimes == 1)
    r2_bull = bull_mask & (regimes == 2)

    assert (exp_b[r0_bull] == 1.0).all()
    assert np.allclose(exp_b[r1_bull], 0.6)
    assert np.allclose(exp_b[r2_bull], 0.2)
