"""
Reproducible Quantitative Research Experiment Entry Point
=========================================================
Executes the full research pipeline:
1. Historical data acquisition, provenance logging, and validation
2. Causal feature engineering
3. Chronological partitioning (Train / Validation / Test)
4. Model fitting strictly on In-Sample training data
5. Forward regime prediction & target exposure generation
6. Fair backtesting with 1-bar execution lag and friction deductions
7. In-depth regime breakdown and transition matrix estimation
8. Multi-dimensional robustness and bootstrap sensitivity analysis
9. Automated report generation and publication-quality diagnostic plotting
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regime_lab.config import DEFAULT_CONFIG, ResearchConfig
from regime_lab.data.loader import MarketDataLoader
from regime_lab.data.pipeline import prepare_market_dataset
from regime_lab.regime.features import compute_regime_features
from regime_lab.regime.gmm_detector import GMMRegimeDetector
from regime_lab.regime.vol_threshold_detector import VolatilityThresholdDetector
from regime_lab.strategy.signals import compute_trend_signal
from regime_lab.strategy.fixed_risk import FixedRiskStrategy
from regime_lab.strategy.adaptive_risk import AdaptiveRiskStrategy
from regime_lab.engine.cost_model import TransactionCostModel
from regime_lab.engine.backtest import BacktestEngine, BacktestResult
from regime_lab.engine.metrics import calculate_drawdown_series, calculate_all_metrics
from regime_lab.validation.partitioner import ChronologicalPartitioner
from regime_lab.validation.regime_analysis import analyze_performance_by_regime
from regime_lab.validation.sensitivity import run_sensitivity_experiments
from regime_lab.visualization.plots import generate_all_research_plots


def run_experiment(config: ResearchConfig = DEFAULT_CONFIG) -> Dict:
    """Run full quantitative experiment pipeline."""
    print("=" * 80)
    print("MARKET REGIME DETECTION & ADAPTIVE RISK RESEARCH PIPELINE")
    print("=" * 80)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Instrument: {config.symbol} | Benchmark: {config.benchmark_symbol}")
    print(f"History: {config.start_date} to {config.end_date}")
    print(f"Train: {config.train_start} to {config.train_end} | Val: {config.val_start} to {config.val_end} | Test: {config.test_start} to Present")
    print(f"Friction Assumption: {config.commission_bps + config.half_spread_bps + config.slippage_bps:.1f} bps round-trip")
    print("-" * 80)

    # 1. Market Data Pipeline
    print("\n[1/7] Fetching and validating historical market data...")
    loader = MarketDataLoader(raw_data_dir=config.raw_data_dir)
    raw_df, val_report, data_meta = loader.fetch_data(
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date
    )
    print(f"  Downloaded {len(raw_df)} trading bars from {val_report.start_date} to {val_report.end_date}.")
    print(f"  Data quality checks: {val_report.duplicate_count} duplicates, {val_report.missing_value_count} NaNs, {val_report.invalid_price_count} price errors.")
    if val_report.warnings:
        for w in val_report.warnings:
            print(f"  Warning: {w}")

    processed_data = prepare_market_dataset(
        raw_df=raw_df,
        symbol=config.symbol,
        processed_dir=config.processed_data_dir
    )

    # 2. Causal Feature Engineering
    print("\n[2/7] Computing causal backward-looking regime features...")
    regime_features = compute_regime_features(
        prices=processed_data.adj_close,
        highs=raw_df["High"],
        lows=raw_df["Low"],
        short_vol_window=config.short_vol_window,
        long_vol_window=config.long_vol_window,
        momentum_window=config.momentum_window
    )
    feat_df = regime_features.feature_matrix
    print(f"  Features generated: {regime_features.feature_names}")
    print(f"  Warmup periods: {config.long_vol_window} bars (up to {feat_df.dropna().index[0].date()}).")

    # 3. Base Trading Signal (Fair Comparison Foundation)
    print("\n[3/7] Generating 200-day trend filter underlying signal...")
    base_signal = compute_trend_signal(
        prices=processed_data.adj_close,
        window=config.trend_filter_window
    )
    bull_pct = float(base_signal.mean())
    print(f"  Base signal generated. Market is in Bullish Trend (> 200 SMA) {bull_pct:.1%} of bars.")

    # 4. Chronological Partitioning
    print("\n[4/7] Chronological time series partitioning (zero leakage)...")
    partitioner = ChronologicalPartitioner(
        train_end=config.train_end,
        val_end=config.val_end,
        test_start=config.test_start
    )
    splits = partitioner.partition(processed_data.df)
    print(splits.summary().to_string(index=False))

    # 5. Fit Regime Detectors Strictly on In-Sample Training Set
    print("\n[5/7] Fitting statistical regime models strictly on In-Sample training set...")
    train_feat = feat_df.loc[splits.train.data.index]
    
    # GMM Detector
    gmm_detector = GMMRegimeDetector(
        n_regimes=config.n_regimes,
        covariance_type=config.covariance_type,
        random_state=config.random_state
    )
    gmm_detector.fit(train_feat)
    print("  GMM Model Fitted Successfully:")
    print(gmm_detector.summary_.cluster_means.to_string())

    # Volatility Threshold Benchmark Detector
    vol_thresh_detector = VolatilityThresholdDetector()
    vol_thresh_detector.fit(train_feat)
    print(f"  Benchmark Vol Quantile Thresholds: Low <= {vol_thresh_detector.low_threshold:.1%}, High > {vol_thresh_detector.high_threshold:.1%}")

    # Forward Predictions across all dates
    regimes_gmm = gmm_detector.predict(feat_df)
    regimes_vol_thresh = vol_thresh_detector.predict(feat_df)

    # 6. Build Strategies & Execute Backtests
    print("\n[6/7] Simulating Strategy A (Fixed Risk) vs Strategy B (Adaptive Risk)...")
    strat_a = FixedRiskStrategy(target_exposure=config.fixed_exposure)
    strat_b = AdaptiveRiskStrategy(exposure_map=config.adaptive_exposures)

    # Target exposure signals at bar close t
    exp_fixed = strat_a.generate_target_exposure(base_signal)
    exp_adapt = strat_b.generate_target_exposure(base_signal, regimes_gmm)

    cost_model = TransactionCostModel(
        commission_bps=config.commission_bps,
        half_spread_bps=config.half_spread_bps,
        slippage_bps=config.slippage_bps
    )
    engine = BacktestEngine(
        cost_model=cost_model,
        risk_free_rate=config.risk_free_rate,
        periods_per_year=config.periods_per_year
    )

    # Slices for In-Sample (Train), Validation, and Out-of-Sample (Test)
    test_idx = splits.test.data.index
    p_test = processed_data.adj_close.loc[test_idx]
    exp_f_test = exp_fixed.loc[test_idx]
    exp_a_test = exp_adapt.loc[test_idx]

    res_fixed_oos = engine.run(p_test, exp_f_test, strategy_name="Strategy A (Fixed Risk)")
    res_adapt_oos = engine.run(p_test, exp_a_test, strategy_name="Strategy B (Adaptive Risk)")

    # Full history runs
    p_full = processed_data.adj_close
    res_fixed_full = engine.run(p_full, exp_fixed, strategy_name="Strategy A Full")
    res_adapt_full = engine.run(p_full, exp_adapt, strategy_name="Strategy B Full")

    # Benchmark SPY Buy & Hold (OOS)
    bench_ret_oos = (p_test.values[1:] / p_test.values[:-1]) - 1.0
    bench_full_ret = np.concatenate([[0.0], bench_ret_oos])
    bench_eq_oos = np.cumprod(1.0 + bench_full_ret)
    bench_dd_oos, bench_max_dd, _ = calculate_drawdown_series(bench_full_ret)
    bench_metrics_oos = calculate_all_metrics(bench_ret_oos)

    # Print Out-of-Sample Performance Comparison
    print("\n" + "=" * 50)
    print("OUT-OF-SAMPLE TEST PERFORMANCE (2020–2026)")
    print("=" * 50)
    oos_comparison = pd.DataFrame({
        "Metric": [
            "Cumulative Net Return",
            "CAGR (Annualized Return)",
            "Annualized Volatility",
            "Sharpe Ratio (Rf=0%)",
            "Sortino Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Annualized Turnover",
            "Total Cost Drag (bps)",
            "Win Rate",
            "Profit Factor"
        ],
        "Strategy A: Fixed Risk": [
            f"{res_fixed_oos.net_metrics.total_return:.2%}",
            f"{res_fixed_oos.net_metrics.cagr:.2%}",
            f"{res_fixed_oos.net_metrics.annualized_volatility:.2%}",
            f"{res_fixed_oos.net_metrics.sharpe_ratio:.3f}",
            f"{res_fixed_oos.net_metrics.sortino_ratio:.3f}",
            f"{res_fixed_oos.net_metrics.max_drawdown:.2%}",
            f"{res_fixed_oos.net_metrics.calmar_ratio:.3f}",
            f"{res_fixed_oos.net_metrics.annualized_turnover:.2f}x",
            f"{res_fixed_oos.net_metrics.total_cost_drag * 1e4:.1f} bps",
            f"{res_fixed_oos.net_metrics.win_rate:.2%}",
            f"{res_fixed_oos.net_metrics.profit_factor:.3f}"
        ],
        "Strategy B: Adaptive Risk": [
            f"{res_adapt_oos.net_metrics.total_return:.2%}",
            f"{res_adapt_oos.net_metrics.cagr:.2%}",
            f"{res_adapt_oos.net_metrics.annualized_volatility:.2%}",
            f"{res_adapt_oos.net_metrics.sharpe_ratio:.3f}",
            f"{res_adapt_oos.net_metrics.sortino_ratio:.3f}",
            f"{res_adapt_oos.net_metrics.max_drawdown:.2%}",
            f"{res_adapt_oos.net_metrics.calmar_ratio:.3f}",
            f"{res_adapt_oos.net_metrics.annualized_turnover:.2f}x",
            f"{res_adapt_oos.net_metrics.total_cost_drag * 1e4:.1f} bps",
            f"{res_adapt_oos.net_metrics.win_rate:.2%}",
            f"{res_adapt_oos.net_metrics.profit_factor:.3f}"
        ],
        "Benchmark: SPY Buy & Hold": [
            f"{bench_metrics_oos.total_return:.2%}",
            f"{bench_metrics_oos.cagr:.2%}",
            f"{bench_metrics_oos.annualized_volatility:.2%}",
            f"{bench_metrics_oos.sharpe_ratio:.3f}",
            f"{bench_metrics_oos.sortino_ratio:.3f}",
            f"{bench_metrics_oos.max_drawdown:.2%}",
            f"{bench_metrics_oos.calmar_ratio:.3f}",
            "0.00x",
            "0.0 bps",
            f"{bench_metrics_oos.win_rate:.2%}",
            f"{bench_metrics_oos.profit_factor:.3f}"
        ]
    })
    print(oos_comparison.to_string(index=False))

    # 7. Regime Performance Breakdown & Robustness
    print("\n[7/7] Computing regime breakdown analytics and robustness testing...")
    regime_summary = analyze_performance_by_regime(
        regimes=regimes_gmm,
        asset_returns=processed_data.returns,
        fixed_returns=res_fixed_full.net_returns,
        adaptive_returns=res_adapt_full.net_returns,
        adaptive_exposure=res_adapt_full.positions,
        n_regimes=config.n_regimes
    )
    print("\nPerformance by Market Regime:")
    print(regime_summary.comparison_table.to_string(index=False))

    print("\nRegime Transition Matrix:")
    print(regime_summary.transition_matrix.to_string())

    # Sensitivity experiments
    sensitivity_results = run_sensitivity_experiments(
        prices=processed_data.adj_close,
        base_signal=base_signal,
        regimes=regimes_gmm,
        test_start=config.test_start,
        n_bootstrap=1000,
        random_state=config.random_state
    )
    print("\nTransaction Cost Sensitivity (Out-of-Sample):")
    print(sensitivity_results.cost_sensitivity.to_string(index=False))

    print("\nExposure Multiplier Sensitivity (Out-of-Sample):")
    print(sensitivity_results.exposure_sensitivity.to_string(index=False))

    print("\nHistorical Sub-Period Stress Testing:")
    print(sensitivity_results.subperiod_performance.to_string(index=False))

    print(f"\nBootstrap Sharpe Difference (Adaptive - Fixed):")
    print(f"  Mean Delta Sharpe: {sensitivity_results.bootstrap_sharpe_diff_mean:+.3f}")
    print(f"  95% Confidence Interval: [{sensitivity_results.bootstrap_sharpe_ci_lower:+.3f}, {sensitivity_results.bootstrap_sharpe_ci_upper:+.3f}]")
    print(f"  Empirical p-value: {sensitivity_results.p_value_sharpe_diff:.4f}")

    # Generate diagnostic plots
    plot_paths = generate_all_research_plots(
        prices=processed_data.adj_close,
        regimes=regimes_gmm,
        short_vol=feat_df["short_vol"],
        res_fixed_oos=res_fixed_oos,
        res_adapt_oos=res_adapt_oos,
        benchmark_equity_oos=pd.Series(bench_eq_oos, index=test_idx),
        benchmark_dd_oos=pd.Series(bench_dd_oos, index=test_idx),
        regime_summary=regime_summary,
        sensitivity=sensitivity_results,
        plots_dir=config.plots_dir
    )
    print("\nDiagnostic plots saved to:")
    for k, p in plot_paths.items():
        print(f"  - {k}: {p}")

    # Save results to JSON
    os.makedirs(config.reports_dir, exist_ok=True)
    results_json_path = os.path.join(config.reports_dir, "experiment_results.json")
    results_dict = {
        "metadata": {
            "symbol": config.symbol,
            "benchmark": config.benchmark_symbol,
            "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frictions_bps": config.commission_bps + config.half_spread_bps + config.slippage_bps,
            "train_period": [config.train_start, config.train_end],
            "validation_period": [config.val_start, config.val_end],
            "test_period": [config.test_start, str(test_idx[-1].date())]
        },
        "out_of_sample_metrics": {
            "strategy_a_fixed": res_fixed_oos.net_metrics.to_dict(),
            "strategy_b_adaptive": res_adapt_oos.net_metrics.to_dict(),
            "benchmark_spy": bench_metrics_oos.to_dict(),
            "delta_sharpe": res_adapt_oos.net_metrics.sharpe_ratio - res_fixed_oos.net_metrics.sharpe_ratio,
            "delta_max_drawdown": res_fixed_oos.net_metrics.max_drawdown - res_adapt_oos.net_metrics.max_drawdown
        },
        "bootstrap_results": {
            "mean_delta_sharpe": sensitivity_results.bootstrap_sharpe_diff_mean,
            "ci_95_lower": sensitivity_results.bootstrap_sharpe_ci_lower,
            "ci_95_upper": sensitivity_results.bootstrap_sharpe_ci_upper,
            "p_value": sensitivity_results.p_value_sharpe_diff
        },
        "plot_paths": plot_paths
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nMachine-readable results archived at: {results_json_path}")

    # Save summary tables to CSV
    oos_comparison.to_csv(os.path.join(config.reports_dir, "oos_comparison.csv"), index=False)
    regime_summary.comparison_table.to_csv(os.path.join(config.reports_dir, "regime_performance.csv"), index=False)
    regime_summary.transition_matrix.to_csv(os.path.join(config.reports_dir, "regime_transitions.csv"))
    sensitivity_results.cost_sensitivity.to_csv(os.path.join(config.reports_dir, "cost_sensitivity.csv"), index=False)
    sensitivity_results.subperiod_performance.to_csv(os.path.join(config.reports_dir, "subperiod_stress_tests.csv"), index=False)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return results_dict


if __name__ == "__main__":
    run_experiment()
