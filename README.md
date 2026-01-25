# Bayesian-Regime-Detection-Strategy
# Bayesian Markov Switching Model for S&P 500 Volatility Regimes

A quantitative research framework that identifies latent market regimes ("Calm" vs. "Crisis") using a Bayesian Hidden Markov Model (HMM). The project implements a dynamic asset allocation strategy that successfully navigates historical "Lost Decades" and modern market crashes.

![Regime Map](analysis/regime_signals.png)
*Figure 1: The model (Red) identifying the Great Depression, 2008 GFC, and COVID-19 without look-ahead bias.*

## Strategy Performance
The strategy employs a lagged decision rule to switch between the S&P 500 and a Risk-Free asset. By filtering out high-volatility regimes, it significantly reduces drawdown while capturing long-term upside.

![Cumulative Wealth](analysis/cumulative_wealth.png)
*Figure 2: Growth of $1 invested (2000–2023). The strategy (Green) avoids the 2000 and 2008 crashes.*

| Metric (Full Century) | Benchmark (S&P 500) | HMM Strategy |
| :--- | :--- | :--- |
| **Sharpe Ratio** | 0.30 | **1.03** |
| **Max Drawdown** | -86% | **-33%** |
| **CAGR (Modern Era)** | 4.1% | **10.2%** |

## Robustness Across Cycles
Unlike simple trend-following, the HMM strategy generates alpha in diverse economic conditions, including the deflationary 1930s and the inflationary 1970s.

![Decade Returns](analysis/decade_returns.png)
*Figure 3: The strategy avoids "Lost Decades" (negative bars in Grey) by detecting structural regime shifts.*

## Technical Implementation
This project bridges the gap between statistical rigor and production engineering:
* **Bayesian Inference:** Implemented in **Stan** (Hamiltonian Monte Carlo) to capture parameter uncertainty and "sticky" regime transitions.
* **High-Performance Computing:** Core Forward-Backward algorithms re-engineered in **C++ (Rcpp)**, achieving a **30.5x speedup** over vectorized R code for real-time inference.
* **Stress Testing:** Validated against out-of-sample shocks including Black Monday (1987) and the Dot-Com Bubble (2000).

## Project Structure
* `analysis/`: Generated plots (Wealth curves, Regime maps).
* `models/`: Stan model files (`.stan`) and C++ source code (`.cpp`).
* `bayesian_hmm_sp500.Rmd`: Main research notebook and reporting engine.
* `fit_ms.rds`: Cached MCMC samples.

## Reproducibility
To run this analysis locally:
1.  Clone the repo.
2.  Open `bayesian-markov-switching-sp500.Rproj` in RStudio.
3.  Run `renv::restore()` to install dependencies (rstan, depmixS4, Rcpp).
4.  Knit the RMarkdown file.

---
*Author: Jack Reed*
