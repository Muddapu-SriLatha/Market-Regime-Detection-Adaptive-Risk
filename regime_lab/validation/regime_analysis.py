"""
Regime Performance Breakdown & Transition Analytics
===================================================
Analyzes the empirical behavior of the market and strategies across each detected regime:
- Regime frequency, average duration, and transition dynamics
- Performance contrast: Strategy A vs Strategy B vs Market per regime
- Exposure tracking: verifies exposure was scaled as intended
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from regime_lab.engine.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_drawdown_series
)


@dataclass
class RegimeStats:
    """Summary statistics for a single regime state."""
    regime_id: int
    name: str
    total_bars: int
    frequency_pct: float
    avg_duration_bars: float
    asset_cagr: float
    asset_volatility: float
    fixed_strategy_cagr: float
    fixed_strategy_volatility: float
    fixed_strategy_sharpe: float
    adaptive_strategy_cagr: float
    adaptive_strategy_volatility: float
    adaptive_strategy_sharpe: float
    avg_adaptive_exposure: float


@dataclass
class RegimePerformanceSummary:
    """Consolidated regime diagnostics and transition dynamics."""
    regime_stats: Dict[int, RegimeStats]
    transition_matrix: pd.DataFrame
    comparison_table: pd.DataFrame


def analyze_performance_by_regime(
    regimes: pd.Series,
    asset_returns: pd.Series,
    fixed_returns: pd.Series,
    adaptive_returns: pd.Series,
    adaptive_exposure: pd.Series,
    n_regimes: int = 3
) -> RegimePerformanceSummary:
    """
    Compute rigorous per-regime performance metrics and transition matrix.
    """
    valid_mask = ~regimes.isna() & ~asset_returns.isna()
    r_series = regimes[valid_mask].astype(int)
    asset_ret = asset_returns[valid_mask]
    fix_ret = fixed_returns[valid_mask]
    adp_ret = adaptive_returns[valid_mask]
    adp_exp = adaptive_exposure[valid_mask]

    total_bars = len(r_series)
    regime_names = {
        0: "Low Risk (Calm Bull)",
        1: "Medium Risk (Transitional)",
        2: "High Risk (Turbulent Crisis)"
    }

    stats_dict: Dict[int, RegimeStats] = {}
    rows = []

    for r in range(n_regimes):
        mask = (r_series == r)
        r_bars = int(np.sum(mask))
        freq = float(r_bars / total_bars) if total_bars > 0 else 0.0

        if r_bars > 5:
            # Durations
            r_bool = (r_series == r).values
            transitions = (r_bool[1:] != r_bool[:-1])
            blocks = np.split(r_bool, np.where(transitions)[0] + 1)
            durations = [len(b) for b in blocks if b[0]]
            avg_dur = float(np.mean(durations)) if len(durations) > 0 else 0.0

            # Asset stats in regime r
            a_cagr = calculate_cagr(asset_ret[mask].values)
            a_vol = calculate_annualized_volatility(asset_ret[mask].values)

            # Fixed strategy stats in regime r
            f_cagr = calculate_cagr(fix_ret[mask].values)
            f_vol = calculate_annualized_volatility(fix_ret[mask].values)
            f_sharpe = calculate_sharpe_ratio(fix_ret[mask].values)

            # Adaptive strategy stats in regime r
            b_cagr = calculate_cagr(adp_ret[mask].values)
            b_vol = calculate_annualized_volatility(adp_ret[mask].values)
            b_sharpe = calculate_sharpe_ratio(adp_ret[mask].values)

            avg_exp = float(np.mean(adp_exp[mask]))
        else:
            avg_dur = 0.0
            a_cagr, a_vol = 0.0, 0.0
            f_cagr, f_vol, f_sharpe = 0.0, 0.0, 0.0
            b_cagr, b_vol, b_sharpe = 0.0, 0.0, 0.0
            avg_exp = 0.0

        stats = RegimeStats(
            regime_id=r,
            name=regime_names.get(r, f"Regime {r}"),
            total_bars=r_bars,
            frequency_pct=freq,
            avg_duration_bars=avg_dur,
            asset_cagr=a_cagr,
            asset_volatility=a_vol,
            fixed_strategy_cagr=f_cagr,
            fixed_strategy_volatility=f_vol,
            fixed_strategy_sharpe=f_sharpe,
            adaptive_strategy_cagr=b_cagr,
            adaptive_strategy_volatility=b_vol,
            adaptive_strategy_sharpe=b_sharpe,
            avg_adaptive_exposure=avg_exp
        )
        stats_dict[r] = stats

        rows.append({
            "Regime": stats.name,
            "Bars": stats.total_bars,
            "Frequency": f"{stats.frequency_pct:.1%}",
            "Avg Duration": f"{stats.avg_duration_bars:.1f} days",
            "Avg Exposure": f"{stats.avg_adaptive_exposure:.2f}x",
            "Asset Vol": f"{stats.asset_volatility:.1%}",
            "Fixed CAGR": f"{stats.fixed_strategy_cagr:.1%}",
            "Fixed Sharpe": f"{stats.fixed_strategy_sharpe:.2f}",
            "Adaptive CAGR": f"{stats.adaptive_strategy_cagr:.1%}",
            "Adaptive Sharpe": f"{stats.adaptive_strategy_sharpe:.2f}"
        })

    # Transition Matrix P(R_t = j | R_{t-1} = i)
    r_arr = r_series.values
    t_mat = np.zeros((n_regimes, n_regimes))
    for t in range(1, len(r_arr)):
        i, j = r_arr[t-1], r_arr[t]
        t_mat[i, j] += 1

    row_sums = t_mat.sum(axis=1, keepdims=True)
    t_prob = np.divide(t_mat, row_sums, out=np.zeros_like(t_mat), where=row_sums > 0)
    col_labels = [f"To {regime_names[r]}" for r in range(n_regimes)]
    row_labels = [f"From {regime_names[r]}" for r in range(n_regimes)]
    transition_df = pd.DataFrame(t_prob, index=row_labels, columns=col_labels)

    comp_df = pd.DataFrame(rows)
    return RegimePerformanceSummary(
        regime_stats=stats_dict,
        transition_matrix=transition_df,
        comparison_table=comp_df
    )
