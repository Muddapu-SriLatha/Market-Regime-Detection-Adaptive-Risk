"""
Central Research Configuration
==============================
Contains all explicitly agreed-upon research parameters:
- Instrument & Benchmark
- Date ranges & chronological split boundaries
- Market frequency
- Feature extraction & lookback parameters
- Regime detection parameters
- Strategy exposure multipliers
- Transaction friction parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass(frozen=True)
class ResearchConfig:
    # Asset & Data Specifications
    symbol: str = "SPY"
    benchmark_symbol: str = "SPY"
    start_date: str = "2000-01-01"
    end_date: str = "2026-02-01"
    frequency: str = "1d"
    data_source: str = "Yahoo Finance (via yfinance)"
    
    # Chronological Partitioning (Zero-leakage boundaries)
    train_start: str = "2000-01-01"
    train_end: str = "2015-12-31"      # ~16 years: dot-com, 2008 GFC, 2011 EU crisis
    val_start: str = "2016-01-01"
    val_end: str = "2019-12-31"        # 4 years: 2017 low-vol grind, 2018 vol spike
    test_start: str = "2020-01-01"
    test_end: Optional[str] = None     # Through latest available (COVID, 2022 rate shock, 2023-2026)
    
    # Feature Engineering Lookbacks (Trading days)
    short_vol_window: int = 21         # ~1 month realized volatility
    long_vol_window: int = 63          # ~3 months realized volatility
    momentum_window: int = 21          # ~1 month trailing return
    trend_filter_window: int = 200     # 200-day SMA long-term trend filter
    
    # Regime Detector Settings
    n_regimes: int = 3                 # 0: Low Risk, 1: Medium Risk, 2: High Risk
    random_state: int = 42             # For reproducible GMM/KMeans fitting
    covariance_type: str = "full"      # GMM covariance type
    
    # Adaptive Risk Sizing Multipliers
    fixed_exposure: float = 1.0        # Strategy A baseline exposure when bullish
    adaptive_exposures: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # Low Risk regime: full 1.0x exposure
        1: 0.6,   # Medium Risk regime: moderate defensive de-risking (0.6x)
        2: 0.2    # High Risk regime: high defensive de-risking (0.2x)
    })
    bearish_exposure: float = 0.0      # When underlying trend signal is bearish (Cash)
    
    # Transaction Friction Assumptions (Liquid ETF standard: 5 bps round-trip)
    commission_bps: float = 1.0        # 0.01%
    half_spread_bps: float = 2.0       # 0.02%
    slippage_bps: float = 2.0          # 0.02%
    
    # Annualization Convention
    periods_per_year: int = 252
    risk_free_rate: float = 0.0        # Documented baseline risk-free rate
    
    # Directory Paths
    raw_data_dir: str = os.path.join("data", "raw")
    processed_data_dir: str = os.path.join("data", "processed")
    reports_dir: str = "reports"
    plots_dir: str = os.path.join("reports", "plots")


DEFAULT_CONFIG = ResearchConfig()
