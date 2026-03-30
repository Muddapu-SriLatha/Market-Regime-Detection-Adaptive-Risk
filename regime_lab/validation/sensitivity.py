"""
Robustness and Sensitivity Analysis Module
==========================================
Systematically stress-tests research findings:
1. Friction cost sensitivity (0 bps, 5 bps, 10 bps, 15 bps)
2. Exposure configuration sensitivity (alternative risk scaling vectors)
3. Historical sub-period stress tests (2008 GFC, 2020 COVID, 2022 Rate Shock)
4. Empirical block bootstrap for difference in Sharpe ratio
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from regime_lab.engine.cost_model import TransactionCostModel
from regime_lab.engine.backtest import BacktestEngine, BacktestResult
from regime_lab.engine.metrics import calculate_sharpe_ratio, calculate_cagr, calculate_annualized_volatility


@dataclass
class SensitivityResults:
    """Encapsulates sensitivity analysis tables and statistical tests."""
    cost_sensitivity: pd.DataFrame
    exposure_sensitivity: pd.DataFrame
    subperiod_performance: pd.DataFrame
    bootstrap_sharpe_diff_mean: float
    bootstrap_sharpe_ci_lower: float
    bootstrap_sharpe_ci_upper: float
    p_value_sharpe_diff: float


def run_sensitivity_experiments(
    prices: pd.Series,
    base_signal: pd.Series,
    regimes: pd.Series,
    test_start: str = "2020-01-01",
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> SensitivityResults:
    """
    Execute sensitivity experiments across frictions, exposure rules, and crisis subperiods.
    """
    # 1. Cost Sensitivity (Evaluated on full Out-of-Sample Test period)
    test_mask = prices.index >= pd.Timestamp(test_start)
    p_test = prices.loc[test_mask]
    sig_test = base_signal.loc[test_mask]
    reg_test = regimes.loc[test_mask]

    # Baseline exposures
    exp_fixed = sig_test * 1.0
    mult_map = {0: 1.0, 1: 0.6, 2: 0.2}
    exp_adapt = sig_test * reg_test.map(mult_map).fillna(1.0)

    cost_levels_bps = [0.0, 5.0, 10.0, 15.0]
    cost_rows = []

    for bps in cost_levels_bps:
        cost_mod = TransactionCostModel(commission_bps=bps * 0.2, half_spread_bps=bps * 0.4, slippage_bps=bps * 0.4)
        engine = BacktestEngine(cost_model=cost_mod)

        res_fixed = engine.run(p_test, exp_fixed, strategy_name="Fixed Risk")
        res_adapt = engine.run(p_test, exp_adapt, strategy_name="Adaptive Risk")

        cost_rows.append({
            "Cost Level (bps)": f"{bps:.1f} bps",
            "Fixed CAGR": f"{res_fixed.net_metrics.cagr:.2%}",
            "Fixed Sharpe": f"{res_fixed.net_metrics.sharpe_ratio:.3f}",
            "Fixed MaxDD": f"{res_fixed.net_metrics.max_drawdown:.2%}",
            "Adaptive CAGR": f"{res_adapt.net_metrics.cagr:.2%}",
            "Adaptive Sharpe": f"{res_adapt.net_metrics.sharpe_ratio:.3f}",
            "Adaptive MaxDD": f"{res_adapt.net_metrics.max_drawdown:.2%}",
            "Sharpe Diff (B - A)": f"{res_adapt.net_metrics.sharpe_ratio - res_fixed.net_metrics.sharpe_ratio:+.3f}"
        })
    cost_df = pd.DataFrame(cost_rows)

    # 2. Exposure Multiplier Sensitivity
    exposure_configs = {
        "Balanced (1.0 / 0.6 / 0.2)": {0: 1.0, 1: 0.6, 2: 0.2},
        "Defensive (1.0 / 0.5 / 0.0)": {0: 1.0, 1: 0.5, 2: 0.0},
        "Mild (1.0 / 0.8 / 0.4)": {0: 1.0, 1: 0.8, 2: 0.4},
        "Aggressive De-risking (1.0 / 0.4 / 0.0)": {0: 1.0, 1: 0.4, 2: 0.0}
    }

    base_engine = BacktestEngine(cost_model=TransactionCostModel(commission_bps=1.0, half_spread_bps=2.0, slippage_bps=2.0))
    res_fixed_base = base_engine.run(p_test, exp_fixed, strategy_name="Fixed Risk")

    exp_rows = []
    for cfg_name, mapping in exposure_configs.items():
        exp_cand = sig_test * reg_test.map(mapping).fillna(1.0)
        res_cand = base_engine.run(p_test, exp_cand, strategy_name=cfg_name)

        exp_rows.append({
            "Exposure Config [Low / Med / High]": cfg_name,
            "CAGR": f"{res_cand.net_metrics.cagr:.2%}",
            "Annual Vol": f"{res_cand.net_metrics.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{res_cand.net_metrics.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{res_cand.net_metrics.sortino_ratio:.3f}",
            "Max Drawdown": f"{res_cand.net_metrics.max_drawdown:.2%}",
            "Turnover": f"{res_cand.net_metrics.annualized_turnover:.2f}x",
            "Sharpe vs Fixed": f"{res_cand.net_metrics.sharpe_ratio - res_fixed_base.net_metrics.sharpe_ratio:+.3f}"
        })
    exposure_df = pd.DataFrame(exp_rows)

    # 3. Sub-Period Historical Stress Tests
    subperiods = [
        ("2008 GFC Crash", "2007-10-01", "2009-03-31"),
        ("2011 Debt Crisis", "2011-04-01", "2011-12-31"),
        ("2017 Low-Vol Grind", "2017-01-01", "2017-12-31"),
        ("2020 COVID Shock (OOS)", "2020-01-01", "2020-12-31"),
        ("2022 Rate Shock Bear (OOS)", "2022-01-01", "2022-12-31"),
        ("2023-2026 Expansion (OOS)", "2023-01-01", "2026-02-01")
    ]

    sub_rows = []
    for sp_name, s_start, s_end in subperiods:
        mask = (prices.index >= pd.Timestamp(s_start)) & (prices.index <= pd.Timestamp(s_end))
        if np.sum(mask) < 20:
            continue
        p_sub = prices.loc[mask]
        sig_sub = base_signal.loc[mask]
        reg_sub = regimes.loc[mask]

        exp_f_sub = sig_sub * 1.0
        exp_a_sub = sig_sub * reg_sub.map(mult_map).fillna(1.0)

        res_f_sub = base_engine.run(p_sub, exp_f_sub, strategy_name="Fixed")
        res_a_sub = base_engine.run(p_sub, exp_a_sub, strategy_name="Adaptive")

        # Asset Buy-and-Hold
        asset_ret_sub = (p_sub.values[1:] / p_sub.values[:-1]) - 1.0
        asset_cagr = calculate_cagr(asset_ret_sub)

        sub_rows.append({
            "Market Episode": sp_name,
            "Bars": len(p_sub),
            "Asset CAGR": f"{asset_cagr:.1%}",
            "Fixed Net CAGR": f"{res_f_sub.net_metrics.cagr:.1%}",
            "Fixed MaxDD": f"{res_f_sub.net_metrics.max_drawdown:.1%}",
            "Adaptive Net CAGR": f"{res_a_sub.net_metrics.cagr:.1%}",
            "Adaptive MaxDD": f"{res_a_sub.net_metrics.max_drawdown:.1%}",
            "MaxDD Reduction": f"{res_f_sub.net_metrics.max_drawdown - res_a_sub.net_metrics.max_drawdown:+.1%}"
        })
    subperiod_df = pd.DataFrame(sub_rows)

    # 4. Empirical Stationary Bootstrap for Sharpe Ratio Difference
    rng = np.random.default_rng(random_state)
    ret_fixed = res_fixed_base.net_returns.values
    ret_adapt = res_cand.net_returns.values  # or baseline adaptive
    # Use baseline adaptive
    res_adapt_base = base_engine.run(p_test, exp_adapt, strategy_name="Adaptive Base")
    ret_adapt = res_adapt_base.net_returns.values

    n_samples = len(ret_fixed)
    sharpe_diffs = np.zeros(n_bootstrap)
    block_size = 10  # 10-day blocks to preserve serial correlation
    n_blocks = int(np.ceil(n_samples / block_size))

    for b in range(n_bootstrap):
        # Block bootstrap sampling
        start_indices = rng.integers(0, n_samples - block_size, size=n_blocks)
        indices = np.concatenate([np.arange(idx, idx + block_size) for idx in start_indices])[:n_samples]

        b_fix = ret_fixed[indices]
        b_adp = ret_adapt[indices]

        sh_fix = calculate_sharpe_ratio(b_fix)
        sh_adp = calculate_sharpe_ratio(b_adp)
        sharpe_diffs[b] = sh_adp - sh_fix

    mean_diff = float(np.mean(sharpe_diffs))
    ci_lower = float(np.percentile(sharpe_diffs, 2.5))
    ci_upper = float(np.percentile(sharpe_diffs, 97.5))
    # Empirical two-tailed p-value against null hypothesis of 0 difference
    p_val = float(np.mean(sharpe_diffs <= 0.0) if mean_diff > 0 else np.mean(sharpe_diffs >= 0.0)) * 2.0
    p_val = min(p_val, 1.0)

    return SensitivityResults(
        cost_sensitivity=cost_df,
        exposure_sensitivity=exposure_df,
        subperiod_performance=subperiod_df,
        bootstrap_sharpe_diff_mean=mean_diff,
        bootstrap_sharpe_ci_lower=ci_lower,
        bootstrap_sharpe_ci_upper=ci_upper,
        p_value_sharpe_diff=p_val
    )
