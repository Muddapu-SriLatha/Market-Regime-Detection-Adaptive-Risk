# Market Regime Detection & Adaptive Risk

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/Tests-13%20Passing-brightgreen.svg)](https://pytest.org/)

An institutional-grade quantitative research framework investigating whether **unsupervised statistical market regime detection** and **dynamic risk exposure modulation** improves out-of-sample risk-adjusted performance compared with a fixed-exposure baseline.

Built strictly under non-negotiable research integrity principles: **no fabricated data, no synthetic price generation, no look-ahead bias, and fully reproducible empirical results**.

---

## Table of Contents
1. [Research Objective & Core Hypothesis](#1-research-objective--core-hypothesis)
2. [Key Empirical Findings & Research Verdict](#2-key-empirical-findings--research-verdict)
3. [Methodological Architecture](#3-methodological-architecture)
   - [Protection Against Look-Ahead Bias](#1-protection-against-look-ahead-bias)
   - [Fair Comparison Framework](#2-fair-comparison-framework)
   - [Unsupervised Statistical Regime Detection](#3-unsupervised-statistical-regime-detection)
   - [Realistic Transaction Friction Modeling](#4-realistic-transaction-friction-modeling)
4. [Out-of-Sample Results & Benchmark Comparison](#4-out-of-sample-results--benchmark-comparison)
5. [Regime Breakdown & Transition Dynamics](#5-regime-breakdown--transition-dynamics)
6. [Robustness & Sensitivity Analysis](#6-robustness--sensitivity-analysis)
7. [Visual Diagnostic Gallery](#7-visual-diagnostic-gallery)
8. [Getting Started & Reproduction Instructions](#8-getting-started--reproduction-instructions)
9. [Project Architecture](#9-project-architecture)
10. [Data Provenance & Research Disclosures](#10-data-provenance--research-disclosures)

---

## 1. Research Objective & Core Hypothesis

### The Hypothesis
Does dynamically scaling portfolio exposure according to statistical market regimes improve out-of-sample risk-adjusted performance (Sharpe ratio) and truncate maximum drawdown relative to a fixed-risk baseline?

### Core Research Questions
1. **Can changes in market volatility and momentum regimes be detected causally** using only information available at the time?
2. **How do fixed and adaptive strategies behave** across distinct macroeconomic volatility regimes?
3. **Does adaptive risk improve out-of-sample risk-adjusted returns** after accounting for realistic transaction frictions?
4. **Is the performance differential statistically significant**, or an artifact of noise or curve-fitting?

---

## 2. Key Empirical Findings & Research Verdict

All metrics reported below were computed from **6,559 actual daily market bars** of SPDR S&P 500 ETF Trust (`SPY`) spanning from **January 2000 to January 2026**, evaluated on an **untouched Out-of-Sample Holdout (2020–2026)**.

### Out-of-Sample Performance Summary (2020–2026, 1,528 Trading Days)
| Performance Metric | Strategy A: Fixed Risk | Strategy B: Adaptive Risk | Benchmark: SPY Buy & Hold |
| :--- | :---: | :---: | :---: |
| **Cumulative Net Return** | **91.60%** | **63.35%** | 132.44% |
| **CAGR (Annualized Return)** | **11.32%** | **8.43%** | 14.93% |
| **Annualized Volatility** | **12.57%** | **10.53%** | 20.64% |
| **Sharpe Ratio ($R_f=0\%$)** | **0.916** | **0.822** | 0.778 |
| **Sortino Ratio** | **1.248** | **1.122** | 1.098 |
| **Maximum Drawdown** | **20.68%** | **16.61%** | 33.72% |
| **Calmar Ratio** | **0.547** | **0.508** | 0.443 |
| **Annualized Turnover** | 5.77x | 5.11x | 0.00x |
| **Total Cost Drag** | 175.0 bps | 155.0 bps | 0.0 bps |
| **Win Rate** | 44.76% | 44.76% | 55.14% |
| **Profit Factor** | 1.191 | 1.173 | 1.162 |

### Research Verdict: Tail-Risk Truncation vs Sharpe Drag
- **Drawdown Protection (Success):** Adaptive risk management successfully reduced Maximum Drawdown by **407 bps** (from **20.68%** down to **16.61%**, a ~20% relative decline) and compressed Annualized Volatility by **204 bps** (from **12.57%** to **10.53%**).
- **Risk-Adjusted Performance (Hypothesis Rejected):** Strategy B achieved a lower Sharpe ratio (**0.822** vs **0.916**, $\Delta \text{Sharpe} = -0.094$).
- **Statistical Significance:** A stationary block bootstrap test (1,000 resamples, 10-day block size) revealed a mean $\Delta \text{Sharpe}$ of **-0.089** with a 95% confidence interval of `[-0.323, +0.119]` and an empirical $p$-value of **0.4400**. The difference in Sharpe ratio is statistically insignificant.
- **Economic Explanation (The V-Shaped Recovery Whipsaw):** Unsupervised volatility-based regime detection is inherently backward-looking. During severe crisis sell-offs (e.g., March 2020), Strategy B appropriately protected capital by de-risking exposure down to 0.20x. However, violent V-shaped market rebounds occur while realized volatility remains backward-elevated. Consequently, Strategy B suffered substantial opportunity cost during the initial explosive recovery, sacrificing more return than it saved in volatility.

---

## 3. Methodological Architecture

### 1. Protection Against Look-Ahead Bias
Execution timing enforces a strict 1-bar execution lag:
$$\text{Position}_{t+1} = \text{TargetExposure}_t$$
Signals generated using bar $t$'s closing price are executed at $t+1$, ensuring that portfolio returns cannot earn the price change of the signal generation bar:
$$R_{\text{gross}, t+1} = \text{Position}_{t+1} \times \left(\frac{P_{t+1}}{P_t} - 1\right)$$

### 2. Fair Comparison Framework
To prevent signal confounding, Strategy A and Strategy B share the **exact same underlying trading signal**:
- **Base Signal ($S_t$):** 200-day Simple Moving Average trend filter ($S_t = 1.0$ if $Close_t > SMA_{200, t}$ else $0.0$).
- **Strategy A (Fixed Risk):** $\text{Exposure}_t = S_t \times 1.0$ (Fixed 100% long exposure when bullish, 0% when bearish).
- **Strategy B (Adaptive Risk):** $\text{Exposure}_t = S_t \times M(R_t)$, where $R_t \in \{0, 1, 2\}$ is the statistically identified regime:
  - $M(0) = 1.0\times$ (Low Risk / Calm Bull)
  - $M(1) = 0.6\times$ (Medium Risk / Transitional)
  - $M(2) = 0.2\times$ (High Risk / Turbulent Crisis)
  - When $S_t = 0.0$ (Bearish), both strategies maintain 0.0x exposure (Cash).

### 3. Unsupervised Statistical Regime Detection
Five causal statistical features are extracted using strictly backward-looking windows:
1. **Short-Term Realized Volatility ($\sigma_{21, t}$):** 21-day annualized sample standard deviation of daily log returns.
2. **Long-Term Realized Volatility ($\sigma_{63, t}$):** 63-day annualized sample standard deviation.
3. **Volatility Term-Structure Ratio ($\nu_t = \sigma_{21, t} / \sigma_{63, t}$):** Measures short-term volatility expansion/contraction.
4. **Normalized Momentum Score ($M_{21, t}$):** 21-day cumulative return normalized by 21-day volatility.
5. **Parkinson High-Low Volatility ($\sigma_{P, 21, t}$):** High-low range-based volatility estimator.

A `StandardScaler` and `GaussianMixture(n_components=3, covariance_type='full', random_state=42)` are fitted **strictly on the In-Sample Training Period (2000–2015)**.
To resolve unsupervised label ambiguity across seeds, clusters are mapped deterministically by ascending in-sample volatility:
- **Regime 0 (Low Risk):** Mean $\sigma_{21} = 11.4\%$, Vol Ratio = $1.00$, Momentum = $+0.42$.
- **Regime 1 (Medium Risk):** Mean $\sigma_{21} = 19.1\%$, Vol Ratio = $0.90$, Momentum = $+0.21$.
- **Regime 2 (High Risk):** Mean $\sigma_{21} = 41.7\%$, Vol Ratio = $1.16$, Momentum = $-0.29$.

### 4. Realistic Transaction Friction Modeling
Trading costs are deducted per unit of portfolio turnover:
$$\text{CostDrag}_t = |\text{Position}_t - \text{Position}_{t-1}| \times \text{TotalLinearRate}$$
Where:
- Commission: 1.0 bp (0.0001)
- Half-Spread: 2.0 bps (0.0002)
- Slippage: 2.0 bps (0.0002)
- **Total Linear Friction: 5.0 bps (0.0005 per 100% turnover)**

---

## 4. Regime Breakdown & Transition Dynamics (Full Sample: 2000–2026)

| Regime State | Bars | Frequency | Mean Duration | Avg Exposure | Asset Vol | Strategy A CAGR | Strategy A Sharpe | Strategy B CAGR | Strategy B Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Risk (Calm Bull)** | 3,499 | 53.9% | 63.6 days | 0.95x | 12.0% | +9.6% | 0.85 | +9.5% | 0.85 |
| **Medium Risk (Transitional)** | 2,430 | 37.4% | 35.7 days | 0.29x | 19.5% | +3.7% | 0.41 | +2.2% | 0.38 |
| **High Risk (Crisis)** | 567 | 8.7% | 16.2 days | 0.07x | 41.7% | +5.4% | 0.56 | -1.5% | -0.25 |

### Empirical Transition Matrix $P(R_t \mid R_{t-1})$
| Current \ Next | To Low Risk | To Medium Risk | To High Risk |
| :--- | :---: | :---: | :---: |
| **From Low Risk** | **98.46%** | 1.14% | 0.40% |
| **From Medium Risk** | 1.93% | **97.20%** | 0.86% |
| **From High Risk** | 1.41% | 4.76% | **93.83%** |

---

## 5. Robustness & Sensitivity Analysis

### 1. Transaction Friction Sensitivity (Out-of-Sample)
| Friction Level | Fixed CAGR | Fixed Sharpe | Fixed MaxDD | Adaptive CAGR | Adaptive Sharpe | Adaptive MaxDD | $\Delta$ Sharpe (B - A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 bps (Frictionless)** | 11.64% | 0.940 | 19.81% | 8.71% | 0.846 | 16.44% | -0.094 |
| **5.0 bps (Baseline)** | 11.32% | 0.916 | 20.68% | 8.43% | 0.822 | 16.61% | -0.095 |
| **10.0 bps (Conservative)** | 11.00% | 0.893 | 21.55% | 8.15% | 0.797 | 16.78% | -0.096 |
| **15.0 bps (High Drag)** | 10.68% | 0.870 | 22.41% | 7.88% | 0.773 | 17.43% | -0.097 |

### 2. Exposure Multiplier Sizing Sensitivity
| Configuration [Low / Med / High] | CAGR | Volatility | Sharpe | Sortino | MaxDD | $\Delta$ Sharpe vs Fixed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Balanced (1.0 / 0.6 / 0.2)** | 8.43% | 10.53% | 0.822 | 1.122 | 16.61% | -0.095 |
| **Defensive (1.0 / 0.5 / 0.0)** | 7.69% | 10.22% | 0.776 | 1.062 | 16.55% | -0.140 |
| **Mild (1.0 / 0.8 / 0.4)** | 9.39% | 11.29% | 0.851 | 1.161 | 18.30% | -0.065 |
| **Aggressive De-risking (1.0 / 0.4 / 0.0)** | 7.45% | 9.98% | 0.770 | 1.054 | 15.68% | -0.146 |

### 3. Historical Stress Episodes Breakdown
| Market Episode | Asset CAGR | Fixed Net CAGR | Fixed MaxDD | Adaptive Net CAGR | Adaptive MaxDD | MaxDD Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2008 GFC Crash** | -34.2% | -9.1% | 14.5% | -5.5% | 8.9% | **+5.6%** |
| **2011 Debt Crisis** | -5.5% | -19.5% | 17.5% | -14.7% | 13.7% | **+3.9%** |
| **2017 Low-Vol Grind** | +21.0% | +20.8% | 2.6% | +14.4% | 2.6% | 0.0% |
| **2020 COVID Shock (OOS)** | +17.2% | +8.5% | 17.6% | -0.3% | 16.6% | **+1.0%** |
| **2022 Rate Shock Bear (OOS)** | -18.8% | -16.1% | 16.1% | -13.1% | 13.0% | **+3.0%** |
| **2023–2026 Expansion (OOS)** | +23.1% | +17.1% | 10.2% | +13.9% | 9.3% | **+0.9%** |

---

## 6. Visual Diagnostic Gallery

Generated publication-grade figures saved in `reports/plots/`:

1. **Regime Timeline & Asset History:**
   `reports/plots/regime_timeline_and_asset.png`
   Displays 26 years of SPY price history shaded by the three identified market regimes alongside rolling realized volatility.

2. **Equity & Drawdown Comparison:**
   `reports/plots/equity_and_drawdown_comparison.png`
   Compares Out-of-Sample cumulative wealth growth ($1.00 compounded) and underwater drawdowns between Strategy A, Strategy B, and the SPY Benchmark.

3. **Exposure & Turnover Dynamics:**
   `reports/plots/exposure_and_turnover_dynamics.png`
   Tracks dynamic position sizing over time and cumulative basis point transaction cost drag.

4. **Regime Transitions & Performance Breakdown:**
   `reports/plots/regime_transition_and_breakdown.png`
   Presents the empirical $3 \times 3$ transition probability heatmap and risk-adjusted return comparisons by regime state.

5. **Cost Sensitivity & Bootstrap Diagnostics:**
   `reports/plots/sensitivity_and_bootstrap.png`
   Displays Sharpe degradation under friction stress-tests and the bootstrap distribution of the out-of-sample Sharpe differential.

---

## 7. Getting Started & Reproduction Instructions

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.14.2)
- pip

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/quant-research/market-regime-detection.git
cd "Market Regime Detection & Adaptive Risk"
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
Execute the 13 unit tests covering data validation, feature causality, execution lag, cost modeling, and chronological splits:
```bash
pytest tests/ -v
```

### 3. Run the Research Pipeline End-to-End
Execute the automated quantitative research orchestrator:
```bash
python experiments/run_pipeline.py
```
This command will:
- Download real SPY historical data and archive raw files with provenance metadata.
- Preprocess data, calculate causal features, and partition into Train, Validation, and Test sets.
- Train the GMM regime model strictly on in-sample data.
- Run the out-of-sample backtest with 1-bar execution lag and transaction costs.
- Execute friction, exposure, sub-period, and bootstrap sensitivity analyses.
- Generate and save all 5 diagnostic figures to `reports/plots/`.
- Export machine-readable results to `reports/experiment_results.json`.

---

## 8. Project Architecture

```
Market Regime Detection & Adaptive Risk/
├── config.py / regime_lab/
│   ├── config.py                  # Central approved research parameters
│   ├── data/
│   │   ├── loader.py              # Real market data fetcher, OHLCV validator, and metadata archiver
│   │   └── pipeline.py            # Return, log return, and Parkinson volatility processor
│   ├── regime/
│   │   ├── features.py            # Causal backward-looking regime feature extraction
│   │   ├── gmm_detector.py        # GMM regime detector with deterministic risk ordering
│   │   └── vol_threshold_detector.py # Baseline empirical volatility threshold detector
│   ├── strategy/
│   │   ├── signals.py             # 200-day trend filter underlying signal (fair baseline)
│   │   ├── fixed_risk.py          # Strategy A: Fixed-risk baseline (1.0x exposure)
│   │   └── adaptive_risk.py       # Strategy B: Dynamic regime-adaptive exposure
│   ├── engine/
│   │   ├── cost_model.py          # Commission, bid-ask spread, and slippage friction model
│   │   ├── metrics.py             # CAGR, Volatility, Sharpe, Sortino, MaxDD, Calmar, Turnover
│   │   └── backtest.py            # Vectorized backtester with strict 1-bar execution lag
│   ├── validation/
│   │   ├── partitioner.py         # Zero-leakage chronological time-series partitioner
│   │   ├── regime_analysis.py     # Breakdown by regime state and transition matrix
│   │   └── sensitivity.py         # Multi-friction, exposure sweep, and bootstrap tests
│   └── visualization/
│       └── plots.py               # Publication-grade matplotlib diagnostic charts
├── experiments/
│   └── run_pipeline.py            # Reproducible experiment orchestrator
├── tests/
│   ├── test_data.py               # Data integrity and return formula tests
│   ├── test_engine.py             # Execution lag and turnover friction tests
│   ├── test_metrics.py            # Hand-calculated deterministic metric tests
│   ├── test_regime.py             # Feature causality and GMM label ordering tests
│   ├── test_strategy.py           # Signal fairness and exposure constraint tests
│   └── test_validation.py         # Chronological partition boundary tests
├── data/
│   ├── raw/                       # SPY raw download and provenance metadata JSON
│   └── processed/                 # Processed market data CSV/parquet
├── reports/
│   ├── research_report.md         # Full research report with in-depth quantitative analysis
│   ├── experiment_results.json    # Machine-readable output metrics
│   ├── oos_comparison.csv         # Formatted out-of-sample metrics
│   ├── regime_performance.csv     # Performance statistics by regime state
│   ├── regime_transitions.csv     # Empirical transition probability matrix
│   ├── cost_sensitivity.csv       # Transaction cost stress test table
│   ├── subperiod_stress_tests.csv # Historical crisis breakdown
│   └── plots/                     # Diagnostic visualization PNGs
├── requirements.txt               # Locked environment dependencies
└── README.md                      # Project documentation and reproduction guide
```

---

## 9. Data Provenance & Research Disclosures

- **Data Source:** Yahoo Finance API via `yfinance` package.
- **Instrument:** SPDR S&P 500 ETF Trust (`SPY`), consolidated composite tape.
- **Adjustments:** Returns calculated from split- and dividend-adjusted closing prices.
- **Limitations:** Data represents consolidated non-point-in-time exchange feeds. Public ETF price history excludes trade-by-trade order book dynamics and historical constituent survivorship adjustments of the underlying index.
- **Integrity Guarantee:** Zero synthetic or placeholder prices were used. Every metric, table, and plot in this repository is 100% reproducible by running `python experiments/run_pipeline.py`.
