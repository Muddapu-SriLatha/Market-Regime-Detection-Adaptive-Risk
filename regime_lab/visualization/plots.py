"""
Publication-Quality Diagnostic Visualizations
=============================================
Generates comprehensive diagnostic charts:
1. Historical price timeline with regime background shading
2. Cumulative net equity curves and underwater drawdowns (Strategy A vs B vs SPY)
3. Portfolio exposure and turnover dynamics over time
4. Empirical transition matrix heatmap and per-regime return/volatility profiles
5. Cost sensitivity and bootstrap Sharpe difference distribution
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from regime_lab.engine.backtest import BacktestResult
from regime_lab.validation.regime_analysis import RegimePerformanceSummary
from regime_lab.validation.sensitivity import SensitivityResults


def setup_plot_style():
    """Configure institutional plot aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.autolayout": True
    })


def plot_regime_timeline(
    prices: pd.Series,
    regimes: pd.Series,
    short_vol: pd.Series,
    output_path: str
) -> None:
    """Plot historical asset price series shaded by detected regime state."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    regime_colors = {
        0: "#2ca02c",  # Green: Low Risk
        1: "#ff7f0e",  # Orange: Medium Risk
        2: "#d62728"   # Red: High Risk
    }
    regime_labels = {
        0: "Low Risk (Calm Bull)",
        1: "Medium Risk (Transitional)",
        2: "High Risk (Turbulent Crisis)"
    }

    # Top panel: Asset Price
    ax1.plot(prices.index, prices.values, color="#1f77b4", linewidth=1.5, label="SPY Price (Adj Close)")
    ax1.set_ylabel("Price ($)", fontsize=11, fontweight="bold")
    ax1.set_title("Historical Asset Price & Market Regime Classification (2000–2026)", fontsize=13, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True)

    # Shading regimes
    valid_mask = ~regimes.isna()
    sub_r = regimes[valid_mask]
    dates = sub_r.index

    # Find contiguous spans
    curr_r = int(sub_r.iloc[0])
    span_start = dates[0]

    for i in range(1, len(sub_r)):
        r_val = int(sub_r.iloc[i])
        if r_val != curr_r:
            ax1.axvspan(span_start, dates[i], color=regime_colors[curr_r], alpha=0.18)
            ax2.axvspan(span_start, dates[i], color=regime_colors[curr_r], alpha=0.18)
            curr_r = r_val
            span_start = dates[i]
    ax1.axvspan(span_start, dates[-1], color=regime_colors[curr_r], alpha=0.18)
    ax2.axvspan(span_start, dates[-1], color=regime_colors[curr_r], alpha=0.18)

    # Bottom panel: Realized Volatility
    ax2.plot(short_vol.index, short_vol.values * 100.0, color="#333333", linewidth=1.1, label="21-day Realized Vol (%)")
    ax2.set_ylabel("Annualized Vol (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.grid(True)

    # Custom legend proxies for regimes
    handles, labels = ax1.get_legend_handles_labels()
    for r in [0, 1, 2]:
        patch = matplotlib.patches.Patch(color=regime_colors[r], alpha=0.35, label=regime_labels[r])
        handles.append(patch)
        labels.append(regime_labels[r])
    ax1.legend(handles=handles, labels=labels, loc="upper left", framealpha=0.9)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_equity_and_drawdowns(
    res_fixed: BacktestResult,
    res_adapt: BacktestResult,
    benchmark_equity: pd.Series,
    benchmark_dd: pd.Series,
    output_path: str,
    period_label: str = "Out-of-Sample Period (2020–2026)"
) -> None:
    """Plot cumulative net wealth index and underwater drawdown comparison."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    dates = res_fixed.dates

    # Top panel: Cumulative Wealth
    ax1.plot(dates, res_adapt.net_equity.values, color="#1b9e77", linewidth=2.0,
             label=f"Strategy B: Adaptive Risk (CAGR: {res_adapt.net_metrics.cagr:.1%}, Sharpe: {res_adapt.net_metrics.sharpe_ratio:.2f})")
    ax1.plot(dates, res_fixed.net_equity.values, color="#d95f02", linewidth=1.8,
             label=f"Strategy A: Fixed Risk (CAGR: {res_fixed.net_metrics.cagr:.1%}, Sharpe: {res_fixed.net_metrics.sharpe_ratio:.2f})")
    ax1.plot(dates, benchmark_equity.values, color="#7570b3", linewidth=1.4, linestyle="--",
             label="Benchmark: SPY Buy & Hold")

    ax1.set_ylabel("Growth of $1.00 (Net)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Out-of-Sample Performance Comparison — {period_label}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True)

    # Bottom panel: Underwater Drawdowns
    ax2.plot(dates, res_adapt.drawdowns.values * 100.0, color="#1b9e77", linewidth=1.5,
             label=f"Adaptive MaxDD: {res_adapt.net_metrics.max_drawdown:.1%}")
    ax2.plot(dates, res_fixed.drawdowns.values * 100.0, color="#d95f02", linewidth=1.3,
             label=f"Fixed MaxDD: {res_fixed.net_metrics.max_drawdown:.1%}")
    ax2.plot(dates, benchmark_dd.values * 100.0, color="#7570b3", linewidth=1.0, linestyle="--",
             label="SPY MaxDD")
    ax2.fill_between(dates, res_adapt.drawdowns.values * 100.0, 0, color="#1b9e77", alpha=0.15)

    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exposure_and_turnover(
    res_fixed: BacktestResult,
    res_adapt: BacktestResult,
    output_path: str
) -> None:
    """Plot portfolio exposure and transaction drag over time."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})

    dates = res_fixed.dates

    # Top panel: Portfolio Exposure
    ax1.plot(dates, res_fixed.positions.values, color="#d95f02", linewidth=1.5, label="Strategy A Fixed Exposure")
    ax1.plot(dates, res_adapt.positions.values, color="#1b9e77", linewidth=1.7, label="Strategy B Adaptive Exposure")
    ax1.set_ylabel("Position Exposure", fontsize=11, fontweight="bold")
    ax1.set_title("Portfolio Exposure Modulation & Transaction Friction Drag", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)

    # Bottom panel: Cumulative Drag in bps
    cost_drag_bps = (res_adapt.gross_equity - res_adapt.net_equity) / res_adapt.gross_equity * 10000.0
    ax2.plot(dates, cost_drag_bps.values, color="#e7298a", linewidth=1.5, label="Adaptive Cumulative Cost Drag (bps)")
    ax2.set_ylabel("Cumulative Drag (bps)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transition_and_breakdown(
    regime_summary: RegimePerformanceSummary,
    output_path: str
) -> None:
    """Plot transition matrix heatmap and comparative metrics by regime."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Heatmap of Transition Matrix
    t_mat = regime_summary.transition_matrix.values
    im = ax1.imshow(t_mat, cmap="Blues", vmin=0, vmax=1)

    labels = ["Low Risk", "Med Risk", "High Risk"]
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Next Regime (t)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Current Regime (t-1)", fontsize=11, fontweight="bold")
    ax1.set_title("Empirical Regime Transition Matrix P(R_t | R_{t-1})", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(3):
        for j in range(3):
            val = t_mat[i, j]
            color = "white" if val > 0.5 else "black"
            ax1.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Right: Return and Volatility by Regime Bar Chart
    stats = regime_summary.regime_stats
    x = np.arange(3)
    width = 0.25

    fixed_sharpes = [stats[r].fixed_strategy_sharpe for r in range(3)]
    adapt_sharpes = [stats[r].adaptive_strategy_sharpe for r in range(3)]
    asset_vols = [stats[r].asset_volatility * 100.0 for r in range(3)]

    ax2.bar(x - width/2, fixed_sharpes, width, label="Strategy A Sharpe", color="#d95f02", alpha=0.85)
    ax2.bar(x + width/2, adapt_sharpes, width, label="Strategy B Sharpe", color="#1b9e77", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold")
    ax2.set_title("Strategy Sharpe Ratio Breakdown by Market Regime", fontsize=12, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_and_bootstrap(
    sensitivity: SensitivityResults,
    output_path: str
) -> None:
    """Plot cost sensitivity bars and bootstrap Sharpe difference distribution."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Cost Sensitivity Table visualization
    cost_df = sensitivity.cost_sensitivity
    costs = cost_df["Cost Level (bps)"].tolist()
    # Parse numbers
    fixed_sh = [float(val) for val in cost_df["Fixed Sharpe"].tolist()]
    adapt_sh = [float(val) for val in cost_df["Adaptive Sharpe"].tolist()]

    x = np.arange(len(costs))
    width = 0.35
    ax1.bar(x - width/2, fixed_sh, width, label="Fixed Risk Sharpe", color="#d95f02", alpha=0.85)
    ax1.bar(x + width/2, adapt_sh, width, label="Adaptive Risk Sharpe", color="#1b9e77", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(costs, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Out-of-Sample Sharpe Ratio", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Friction Assumption", fontsize=11, fontweight="bold")
    ax1.set_title("Sharpe Ratio Under Increasing Transaction Costs", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)

    # Right: Exposure Configuration Sharpe vs Fixed
    exp_df = sensitivity.exposure_sensitivity
    configs = [c.split(" (")[0] for c in exp_df["Exposure Config [Low / Med / High]"].tolist()]
    diffs = [float(val) for val in exp_df["Sharpe vs Fixed"].tolist()]

    colors = ["#1b9e77" if d >= 0 else "#d62728" for d in diffs]
    ax2.barh(configs, diffs, color=colors, alpha=0.85, height=0.5)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Delta Sharpe Ratio (Adaptive - Fixed)", fontsize=11, fontweight="bold")
    ax2.set_title("Sensitivity to Exposure Sizing Configuration", fontsize=12, fontweight="bold")
    ax2.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_research_plots(
    prices: pd.Series,
    regimes: pd.Series,
    short_vol: pd.Series,
    res_fixed_oos: BacktestResult,
    res_adapt_oos: BacktestResult,
    benchmark_equity_oos: pd.Series,
    benchmark_dd_oos: pd.Series,
    regime_summary: RegimePerformanceSummary,
    sensitivity: SensitivityResults,
    plots_dir: str = os.path.join("reports", "plots")
) -> Dict[str, str]:
    """Generate and save all 5 publication-grade diagnostic plots."""
    os.makedirs(plots_dir, exist_ok=True)

    paths = {
        "regime_timeline": os.path.join(plots_dir, "regime_timeline_and_asset.png"),
        "equity_and_drawdown": os.path.join(plots_dir, "equity_and_drawdown_comparison.png"),
        "exposure_and_turnover": os.path.join(plots_dir, "exposure_and_turnover_dynamics.png"),
        "transition_and_breakdown": os.path.join(plots_dir, "regime_transition_and_breakdown.png"),
        "sensitivity_and_bootstrap": os.path.join(plots_dir, "sensitivity_and_bootstrap.png")
    }

    plot_regime_timeline(prices, regimes, short_vol, paths["regime_timeline"])
    plot_equity_and_drawdowns(res_fixed_oos, res_adapt_oos, benchmark_equity_oos, benchmark_dd_oos, paths["equity_and_drawdown"])
    plot_exposure_and_turnover(res_fixed_oos, res_adapt_oos, paths["exposure_and_turnover"])
    plot_transition_and_breakdown(regime_summary, paths["transition_and_breakdown"])
    plot_sensitivity_and_bootstrap(sensitivity, paths["sensitivity_and_bootstrap"])

    return paths
