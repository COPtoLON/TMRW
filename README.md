# TMRW – A Multi-Horizon AI Trading Framework
TMRW (Tomorrow Capital Research) is an ambitious, research-grade framework for building, back-testing and deploying algorithmic trading strategies that span millisecond-level market-making all the way out to multi-year, macro-thematic investing.  It combines modern machine-learning, mean-field-game theory and rigorous risk management into a single “pod” architecture designed to keep compounding capital in every market regime.  Although still early-stage (0 stars, many sections flagged “on hold” or “unfinished”), the repo already sketches a clear technical blueprint, a business plan and a multi-year product roadmap. 

## 1  Purpose & Vision

* **Core goal** – “maximum conservation & compounding of portfolio value” across any horizon or asset class.
* **Research sandbox** – meant both for internal fund use and for open-source collaboration; PRs and theoretical discussion are explicitly welcomed.
* **Business overlay** – the repo doubles as a hedge-fund playbook, outlining share classes, 30 %+ annual return targets and a cyclical “Expand → Establish → Excel → Repeat” growth mantra. 

## 2  System Architecture

| Layer      | Time-frame           | Primary engines             | Typical inventory | Main focus          |                                                                  
| ---------- | -------------------- | --------------------------- | ----------------- | ------------------- | 
| **HFT**    | sub-second → few min | Risk-AI + RL market-maker   | ≈ 0               | spread capture      |                                                                  
| **Short**  | minutes → days       | Stat-AI + volatility models | small             | mean-reversion      |                                                                 
| **Medium** | days → weeks         | Trend + macro factors       | medium            | swing & factor bets |                                                                 
| **Long**   | weeks → years        | Forecast-AI + fundamentals  | large             | thematic / macro    | 

Each layer is a “pod” that can hedge or reinforce the others depending on detected regime.  Inside every pod the pipeline flows

`Raw Data → Clean-up → Feature Store → (Risk AI | Stat AI | Strategy AI | Forecast AI) → Order-Execution (OAI)` 

### Key engines

* **Risk AI** – options-style mitigation, concentration caps, extreme-value monitors. 
* **Strategy AI** – three RL agents (Speedy, Risky, Trendy) that learn optimal bid/ask placement à la Avellaneda-Stoikov. 
* **Forecast AI** – Hidden-Markov-Model state detection, MCMC samplers and classical neural nets. 

## 3  Repository Layout

* **`notebooks/`** – walk-throughs such as `AI-trading.ipynb` (end-to-end engine) and a mean-reversion example (path referenced in README, file not yet committed). 
* **`Algorithm/`** – per-strategy templates covering signal logic, execution, risk limits and KPIs (currently “on hold”). 
* **`DATA/`** – lightweight mock feeds plus an exchange-connectivity checklist (Coinbase, Kraken, etc.). 
* **`util/`** – architecture diagrams, physics/micro-market simulations, helper scripts. 
* **White-papers & blueprints** – detailed markdown files on market modelling, mean-field theory and layered strategy design.

## 4  Notable Research Threads

* **Mean-Field-Game interpretation of markets** – limitations of classical MFGs (fixed agent count, homogeneous state vectors) and ideas for branching extensions that better match trading reality. 
* **Hierarchical reinforcement learning** – each pod’s RL agent reacts at its own cadence, allowing the ensemble to self-hedge across timescales. 
* **Expand–Establish–Excel loop** – a management framework for continually refreshing alpha while controlling downside. 

## 5  Roadmap & Gaps

| Planned feature                                            | Status    | Comment                                                                                         |
| ---------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------- |
| Real-time CCXT / Interactive Brokers gateway               | **TODO**  | live trading hooks still missing.                       |
| GPU-accelerated back-tester with walk-forward optimisation | **TODO**  | only design notes exist.                                   |
| Plotly/Dash dashboards for PnL & risk telemetry            | **TODO**  | targeted for v0.3.                                         |
| Extended asset coverage (rates, commodities, exotics)      | **TODO**  | design outlined in strategy docs.                              |
| Corporate-structure & hiring roadmap                       | **Draft** | high-level charts present; details flagged “still unfinished”.  |

## 6  Getting Started

````
bash
git clone https://github.com/COPtoLON/TMRW.git
cd TMRW
conda env create -f environment.yml   # file path listed, not yet pushed
conda activate tmrw
jupyter lab                            # walk through AI-trading.ipynb
``` 

## 7  Who Might Care  
* **Quant researchers** wanting a skeletal yet conceptually rich playground that mixes RL, HMMs and classic factor models.  
* **Fin-tech founders** looking for an open blueprint that spans code, infra and even fund governance.  
* **Academics** studying market micro-structure or MFGs who need an applied test-bed.  

## 8  Current Maturity Snapshot  
* **Commits:** 160 (active but pre-release). 
* **Stars/Forks:** 0/0 – early visibility stage. 
* **Language mix:** 100 % Python so far.
* **Documentation depth:** strong conceptual docs, but code implementations and environment files are still sparse or missing (e.g., `environment.yml`, notebooks). 

---

### Bottom line  
TMRW is less a finished library and more a comprehensive design dossier for a multi-horizon, AI-driven quant fund.  The architectural clarity, extensive theory notes and RL-first mindset make it a promising base for experimentation—provided contributors are ready to flesh out the missing code and connect the dots from white-paper to production.
::contentReference[oaicite:27]{index=27}
````


