Each strategy is broken down into its own section, but the structure is consistent so everyone in the team knows exactly where to find specific details.

Strategy Overview

Name/identifier for the strategy. Brief description: what market inefficiency or opportunity is it targeting? Algorithmic Procedure

Mathematical or heuristic logic behind the strategy. Data inputs and features used (real-time price, fundamentals, alternative data). Frequency of signals (high-frequency, daily, weekly, etc.). Entry & Exit Criteria

Signal generation: thresholds, technical indicators, or model outputs that trigger entries. Position sizing logic: how the size of each position is determined. Exit conditions: stop-loss levels, profit-taking thresholds, time-based exit logic. Execution Algorithms

Explanation of how orders are routed (e.g., dark pools, limit orders, VWAP/TWAP). Slippage and transaction cost optimization methods. Risk Management per Strategy

In-strategy risk controls: stop losses, drawdown alerts, intraday kill switches. Capital allocation: maximum capital allocated to this strategy relative to the total portfolio. Backtesting & Valuation Linkages

Specific data sets used for backtesting this strategy. Historical performance metrics: Sharpe ratio, max drawdown, annualized return. How the strategy is integrated into the overall portfolio (cross-reference to overarching risk and allocation framework in the “Overarching Strategy Document”). Ongoing Monitoring & Review

KPIs for live performance: daily P/L, daily volatility, compliance with risk limits. Trigger points for re-tuning or shutting down the strategy if it underperforms.
