# TMRW – A Multi-Horizon AI Trading Framework
TMRW (Tomorrow Capital Research) is an open, research-grade sandbox for designing, testing and deploying systematic trading strategies across ultra-low-latency to multi-year horizons.
It marries principles from quantitative finance, statistical physics and modern machine-learning to pursue a single goal: maximum conservation & compounding of portfolio value in any market regime.
\



## 1. Core architecture

┌──────────┐    raw feeds  ┌────────┐  engineered  ┌────────┐ \
│   Data   │ ───────────▶ │Clean-up│ ───────────▶ │Features│ \
└──────────┘               └────────┘              └────────┘ \
                                       │ \
                          ┌────────────┼──────────────────┐ \
                          │            │                  │ \
                    ┌──────────┐ ┌──────────┐      ┌──────────┐ \
                    │ Risk AI  │ │ Stat AI  │ ...  │Forecast AI│ \
                    └──────────┘ └──────────┘      └──────────┘ \
                          │            │                  │ \
                          └────────────┴──────────────────┘ \
                                        ▼ \
                                   ┌────────┐ \
                                   │  OAI   │  →  Execution \
                                   └────────┘ \
                                   
- Risk AI – option-mitigation, concentration & extreme-value models
- Statistics AI – battery of regressions, factor & time-series tests
- Strategy AI – rule-based logic plus three RL agents (Speedy, Risky, Trendy) that learn bid/ask placement à la Avellaneda-Stoikov.
- Forecast AI – hidden-Markov state detection, MCMC, delta-vector and classical NN forecasters.
- The ensemble is organised in hierarchical “pods”—HFT, Short, Medium, Long—so layers can hedge or reinforce one another depending on regime.

## 2. Repository layout
Path	What you’ll find
notebooks/	• AI-trading.ipynb – end-to-end demo of the full engine
• mean-reversion-strategy.ipynb – statistical-arbitrage walkthrough
• Strategy.ipynb / research.ipynb – experimental RL & Monte-Carlo prototypes
util/	Architecture & corporate-roadmap diagrams, plus helper scripts
data/	Lightweight mock datasets for quick testing
docs/	Knowledge-base PDFs covering theory, risk, infra & business blueprint

## 3. Quick-start
´´´
git clone https://github.com/your-handle/TMRW.git
cd TMRW
conda env create -f environment.yml
conda activate tmrw
jupyter lab
´´´
Tip: each notebook is self-contained; start with AI-trading.ipynb, then drill into specialised notebooks as you explore.

## 4. Key features
End-to-end pipeline – from data ingestion & cleaning to live-trade execution hooks
Multi-agent RL market-making with adaptive risk-aversion & queue-position modelling
Statistical & machine-learning library of mean-reversion, VaR-, Bollinger-, and HMM-based strategies with performance benchmarks
Cyclical R&D framework – Expand → Establish → Excel → Repeat ensures continual strategy refresh and portfolio resilience
Research notes linking variational principles, mean-field games and quantum annealing to future strategy modules

## 5. Road-map
 Real-time gateway to Interactive Brokers / CCXT
 GPU-accelerated back-tester with walk-forward optimisations
 Live dashboards (Plotly / Dash) for risk & PnL telemetry
 Extended asset coverage: rates, commodities & exotic options
 Corporate-structure playbook for spinning the codebase into a multi-pod fund

## 6. Contributing
We welcome PRs, issue reports and theoretical discussion. If you have ideas, data, or capital to collaborate, reach out via the issues tab or mark@brezina.dk.
*Let’s turn insight into algorithms—and algorithms into out-performance.*
