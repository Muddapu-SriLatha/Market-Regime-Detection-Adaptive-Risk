"""Unit tests for regime feature construction and statistical detectors."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.regime.features import compute_regime_features
from regime_lab.regime.gmm_detector import GMMRegimeDetector
from regime_lab.regime.vol_threshold_detector import VolatilityThresholdDetector


def test_feature_causality_and_lookback():
    """Verify that features at time t depend strictly on observations <= t."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))), index=dates)

    feat = compute_regime_features(prices, short_vol_window=21, long_vol_window=63, momentum_window=21)
    df = feat.feature_matrix

    # Check lookback warmup periods
    # Because prices -> log_returns consumes 1 bar (index 0 is NaN),
    # short_vol requires 21 returns, so indices 0..20 (21 rows) are NaN, and index 21 is the first valid value.
    assert df["short_vol"].iloc[:21].isna().all()
    assert not np.isnan(df["short_vol"].iloc[21])

    # Long vol requires 63 returns, so indices 0..62 (63 rows) are NaN, and index 63 is the first valid value.
    assert df["long_vol"].iloc[:63].isna().all()
    assert not np.isnan(df["long_vol"].iloc[63])


def test_gmm_deterministic_label_ordering():
    """Verify that GMM orders regimes strictly by ascending volatility (0 = lowest, 2 = highest)."""
    # Create synthetic dataset with 3 distinct volatility regimes
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2010-01-01", periods=n, freq="B")

    # Regime 0: vol ~ 0.10
    # Regime 1: vol ~ 0.20
    # Regime 2: vol ~ 0.40
    vol_vals = np.concatenate([
        np.full(100, 0.10) + np.random.normal(0, 0.01, 100),
        np.full(100, 0.20) + np.random.normal(0, 0.01, 100),
        np.full(100, 0.40) + np.random.normal(0, 0.02, 100)
    ])
    feat_df = pd.DataFrame({
        "short_vol": vol_vals,
        "vol_ratio": np.ones(n),
        "norm_momentum": np.zeros(n)
    }, index=dates)

    detector = GMMRegimeDetector(n_regimes=3, random_state=42)
    detector.fit(feat_df)

    # Check summary ordering
    means = detector.summary_.cluster_means
    mean_vols = [detector.summary_.mean_volatility_by_regime[r] for r in range(3)]

    assert mean_vols[0] < mean_vols[1] < mean_vols[2], "Regimes must be sorted by ascending volatility"


def test_volatility_threshold_detector():
    """Verify that quantile thresholding assigns regimes accurately."""
    dates = pd.date_range("2010-01-01", periods=300, freq="B")
    vols = np.linspace(0.05, 0.35, 300)
    feat_df = pd.DataFrame({"short_vol": vols}, index=dates)

    detector = VolatilityThresholdDetector(low_quantile=0.3333, high_quantile=0.6667)
    detector.fit(feat_df)

    preds = detector.predict(feat_df)
    # Check that predictions contain all three classes and are monotonic
    assert set(preds.unique()) == {0.0, 1.0, 2.0}
    assert preds.iloc[0] == 0.0
    assert preds.iloc[-1] == 2.0
