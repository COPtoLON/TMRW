# Market expectations
This document is meant to serve as a manifest to describe the company's expectations about the market it would be operating in. It could also be considered an operational expectation plan.
It will cover all our overarching views on markets, competitor landscapes, and key theoretical underpinnings. It might also serve as a “living” market outlook and theoretical playbook.

## Macro & Micro Market considerations
The markets, this company will be operating in, will have multiple layers. There will be:
- Macroeconomic factors: interest rates, inflation, global economic indicators.
- Market microstructure considerations: liquidity, order book dynamics, transaction costs.
- Theoretical considerations (e.g., Efficient Market Hypothesis vs. Adaptive Market Hypothesis).
- external events - most any market experiences external events, EX: news, economic crisis, new policies.
- market participant effects - most markets are effected by other market participants, these can affect the market strategy or drain off of profits.

TO add to this, there ought to be considerations associated with the market participation, on the overall market behaviour, popularily called blue-ocean, red-ocean.
This will help associate whether the company expects to have to change their market participation state (Neutral, defense, attack, counter) often, or sparsely

## Operations considerations
There is a large range of possible actions in the market, these will be considered as operations.
- One can buy and hold, this is called going long
- One can borrow and sell an asset, this is called shorting
- One can buy options, futures or swaps with regards to a positions, in this way strategic implementations of positions can be taken into account
- One can buy an asset in one exchange and sell it higher in another exchange, in this way benefitting from differences in exchanges.
- One can borrow money, to perform the previous actions, this is called leveraging
- One can propose bid or ask prices in the market, in this way building an expectation of what the other market participants think of certain prices
- ...

## Participants
In respect to participants, a general consideration will be that some participants can be easily observed, some may be less observable.
In the market we may be able to track bid-ask spreads and market depth, but be unable to register what participant has placed what trades.
We may be needed to subscribe to larger systems or use external data providers, to keep track of movements in hindsight.

## Edges
THere are a few areas of the markets, which have yet to be fully optimized or in which various companies are already fighting to become the best.
These are the edges, these are the areas, where we can excel in the future, to become the best. Notable mentions are:
- Speed of execution, both in:
  - actual execution of the algorithm
  - Latency to the exchanges
  - Computational approximations
- Predictive quality, both accuracy and precision
- Coverage, the goal of covering as much of the market as possible, despite market impact
- stealth, keeping the execution of a trade hidden
- Most adaptable algorithms
- Best strategic decision making  




## Macro & Micro Market Considerations
Our target markets have multiple interconnected layers:
1. Macroeconomic Factors: Examples include interest rates, inflation trends, and global economic indicators.
2. Market Microstructure: Covers liquidity, order book dynamics, and transaction costs.
3. Theoretical Perspectives: Ranges from the Efficient Market Hypothesis (EMH) to the Adaptive Market Hypothesis (AMH), each influencing strategic approaches.
4. External Events: News releases, economic crises, or policy changes can significantly shift market conditions.
5. Participant Effects: Competitors or other market players can alter price flows, potentially impacting our strategies or eroding profits.

Additionally, we should consider “blue-ocean vs. red-ocean” scenarios—i.e., markets with uncontested vs. highly contested space—and whether we plan to change our participation mode (neutral, defensive, offensive, or reactive) frequently or infrequently.

## Operational Considerations
A broad range of tactical approaches is available in these markets, including:
- Long Positions – Buying and holding assets.
- Short Positions – Borrowing and selling assets in anticipation of price declines.
- Derivatives Usage – Incorporating options, futures, or swaps to refine positioning and risk profiles.
- Arbitrage – Exploiting price discrepancies across different exchanges.
- Leverage – Borrowing capital to scale or amplify positions.
- Market Making – Setting bid/ask quotes to gauge sentiment and capture spread opportunities.

## Market Participants
Participant visibility will vary:
- Observable – We can gauge some order flow directly, seeing bid-ask spreads and market depth.
- Less Observable – We often cannot identify which specific entity placed a trade without specialized data. Subscribing to more sophisticated market feeds or partnering with data providers may be necessary for deeper insights.

## Potential “Edges”
Certain areas remain under-optimized or heavily contested, presenting opportunities for differentiation:

- Execution Speed – Algorithmic efficiency and low-latency connectivity to exchanges.
- Computational & Predictive Accuracy – High-precision modeling and forecasting.
- Market Coverage – Maintaining broad coverage across multiple instruments or venues while minimizing impact.
- Stealth & Adaptability – Executing trades with minimal market signaling and pivoting strategies quickly when conditions change.
- Strategic Decision-Making – Informed, agile approaches that leverage real-time data for competitive advantage.
These edges represent our potential pathways to outperformance. By focusing on innovation in speed, predictive power, and flexible execution, we aim to thrive in a competitive market landscape.








## Model

### market dynamics
We can imagine a system consisting of multiple agents—referred to as partakers—each endowed with a certain measure. The overarching goal is for each agent to maximize its own measure, which might represent profit, wealth, or another relevant metric. This is a broad framework in which a trading algorithm would naturally be a sub-problem, aiming to maximize gains under certain constraints (e.g., risk limits, available market liquidity).

Hedge funds and proprietary trading firms such as Jane Street, Two Sigma, and Citadel engage in strategies that may be modeled in part by mean-field game (MFG) theory. Each firm (agent) interacts with the market (population), exerting small influences individually but collectively driving overall behavior.

Jane Street could be viewed as seeking an optimal strategy that accounts for large-population effects—this might be reflected in a mean-field approach.\
Two Sigma might specialize in systematically exploiting “edges” that arise from aggregate behavior.\
Citadel might “own the playing field” through broad market participation, typically gleaning minimal gains per transaction on large volume.\
Other funds may focus on volatility or on absorbing mispriced risk from weaker strategies.\
Despite nuanced differences, all of these perspectives can be placed under an umbrella of large-population, strategic decision-making, a natural setting for mean-field games.

### Mean-Field Game Theory
Mean-field game (MFG) theory studies how strategic decision-making unfolds among many “small” agents in a large population. Each agent’s individual influence on the system is negligible, yet collectively these agents drive its overall dynamics. The term “mean field” is borrowed from physics, where the behavior of large systems can often be approximated by examining one representative particle in an “average” field created by the others.

### Traditional Game Theory vs. MFG:
In standard (often two-player) game theory, we might rely on backward induction. This becomes difficult when dealing with a large number of players over continuous time. MFGs tackle this by letting the number of players go to infinity, introducing a representative agent whose behavior and optimal strategies become indicative of the whole population.

### Key PDEs:
Typically, MFGs in continuous time are associated with a coupled Hamilton–Jacobi–Bellman (HJB) equation (describing each agent’s optimal control) and a Fokker–Planck (or Kolmogorov) equation (tracking the aggregate distribution). Under suitable assumptions, an $N$-player Nash equilibrium converges to the MFG solution as $N \to \infty$

### Branching Versions & Extensions:
When agents can enter or leave the game (e.g., a new agent “spawned” under certain conditions), classical MFGs extend to branching mean-field games. This can be relevant for modeling demographic changes in an economy or repeated entry/exit in trading contexts.


### Mean-Field Game Equations (Classical Form)
A canonical MFG problem in continuous time and a state space $\mathbb{R}^d$ includes:

### Backward HJB Equation:

$$-\partial_t u(t,x) - \nu \Delta u(t,x) + H(x,m(t), \nabla u(t,x)) = 0,$$

subject to a terminal/goal condition $u(T,x)=G(x,m(T))$

### Forward Fokker–Planck (or Kolmogorov) Equation:

$$\partial_t m(t,x) - \nu \Delta m(t,x) - div(\nabla_p H(x,m(t), \nabla u(t,x)), m(t,x)) = 0,$$

with some initial condition $m(0)=m_0.$

Here:
- $u(t,x)$ is the value function for a typical agent starting at 
- $m(t)$ is the distribution of agent states at time
- $\nu$ is a diffusion coefficient (or viscosity parameter).
- $H$ is the Hamiltonian encoding the running cost and dynamics, while $G$ is a terminal cost functional.


[footnote 1 - Mean fields - daniel Lacker]()

