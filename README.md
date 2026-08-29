# Sequential Decision Analytics Policy Classes

A reproducible Operations Research implementation of Warren Powell's four policy meta-classes on one common sequential decision problem.

The repository uses a finite-horizon stochastic inventory system so that the distinction between the policy classes is visible in code rather than only in terminology. Every policy receives the same state, chooses the same decision variable, is exposed to the same exogenous-information process, uses the same transition function, and is evaluated with the same objective.

The four implemented policy classes are:

1. **Policy Function Approximation (PFA)** — an analytical state-to-decision rule.
2. **Cost Function Approximation (CFA)** — a parameterized deterministic optimization model solved at each decision epoch.
3. **Value Function Approximation (VFA)** — an optimization model that uses an approximation of downstream value.
4. **Direct Lookahead Approximation (DLA)** — a rolling-horizon optimization model that explicitly plans several periods into an approximate future.

An exact finite-horizon dynamic program is included only as a verification benchmark. It is not presented as a fifth policy class.

## Universal modeling framework

The inventory problem is written using the five elements of the Sequential Decision Analytics modeling framework.

### State variable

```text
S_t = (t, I_t)
```

where `I_t` is net inventory before ordering. Negative inventory represents backlog.

### Decision variable

```text
x_t = order quantity
0 <= x_t <= max_order
x_t integer
```

### Exogenous information

```text
W_{t+1} = customer demand during period t
```

Demand is discrete and stochastic. The demo uses a nonstationary probability schedule with low, medium, and peak-demand periods.

### Transition function

```text
I_{t+1} = I_t + x_t - W_{t+1}
```

### Objective

The objective is to minimize expected discounted operating cost:

```text
E[sum_t discount^t * (
    fixed_order_cost * 1{x_t > 0}
    + unit_order_cost * x_t
    + holding_cost * max(I_{t+1}, 0)
    + backlog_cost * max(-I_{t+1}, 0)
)]
```

This separation is deliberate: the **model** defines the sequential decision problem, while a **policy** defines how a decision is selected from the state.

## 1. Policy Function Approximation

`OrderUpToPFA` is a direct analytical mapping:

```text
x_t = clip(target_inventory - I_t, 0, max_order)
```

No optimization problem is solved inside the policy. The target parameter is tuned by simulation on a fixed set of training demand traces.

This is the simplest policy in the repository and corresponds to the PFA idea of mapping state information directly to an action.

## 2. Cost Function Approximation

`CFAOneStepPolicy` solves a deterministic surrogate optimization problem at every decision epoch.

For each feasible order quantity, it evaluates a parameterized approximation of the immediate operating cost using

```text
forecast_t = E[demand_t] + forecast_bias
```

and selects the order with minimum surrogate cost.

The `forecast_bias` and `backlog_multiplier` parameters change the deterministic cost model. In the demo, `forecast_bias` is tuned using simulation.

The important distinction from a PFA is structural: the CFA does not directly evaluate a closed-form decision rule. It solves an embedded optimization problem whose cost model has been parameterized to work better under uncertainty.

## 3. Value Function Approximation

`fit_coarse_vfa` builds a time-indexed, piecewise-linear approximation of the downstream value function on a coarse inventory grid.

At state `S_t`, the VFA policy selects

```text
argmin_x {
    order_cost(x)
    + E[
        inventory_cost(I_t + x - W_{t+1})
        + discount * V_hat_{t+1}(I_t + x - W_{t+1})
    ]
}
```

The value function is approximate because it is stored only at coarse inventory knots and interpolated between them.

A strong verification test uses grid spacing `1`. With every reachable integer inventory state represented explicitly, the VFA recursion reproduces the exact dynamic-programming action at the initial state.

## 4. Direct Lookahead Approximation

`DeterministicLookaheadPolicy` constructs a rolling deterministic planning problem each time a decision is needed.

The model looks several periods ahead using future mean-demand forecasts, solves the finite deterministic planning problem, implements only the first order, observes actual stochastic demand, and then replans.

Conceptually:

```text
observe current state
        |
        v
build approximate future over H periods
        |
        v
optimize the H-period plan
        |
        v
implement only the first decision
        |
        v
observe new information and repeat
```

This is a receding-horizon direct lookahead. It is intentionally deterministic; stochastic lookahead is a separate and substantially more expensive subclass of DLA.

## Exact dynamic-programming benchmark

`solve_exact_dp` solves the small finite-horizon Markov decision problem by backward induction using the full discrete demand distribution.

For each state it computes

```text
V_t(I) = min_x {
    order_cost(x)
    + E[
        inventory_cost(I + x - W)
        + discount * V_{t+1}(I + x - W)
    ]
}
```

This provides two forms of verification:

- a known optimal expected cost for the synthetic model,
- an exact policy against which approximate policy implementations can be checked.

The exact dynamic program is feasible here only because the state and action spaces are intentionally small.

## Policy tuning and evaluation

The policy-search classes use a clean train/validation split.

### Training

A fixed set of demand traces is generated once and reused for every parameter candidate. This is a Common Random Numbers design: parameter candidates are compared on identical exogenous-information realizations.

The demo tunes:

- the PFA order-up-to target,
- the CFA forecast bias.

### Out-of-sample evaluation

After tuning, all policies are evaluated on a new seed range that was not used during policy selection.

Reported metrics include:

- mean discounted cost,
- standard error,
- an approximate 95% confidence interval.

The exact DP's analytical expected cost is also reported separately from its Monte Carlo estimate.

## Reproducible demo result

With the repository defaults, 120 training replications, 500 independent validation replications, and seed `2026`, the current implementation produces approximately:

```text
Tuned PFA target:          6
Tuned CFA forecast bias:   2.0
Exact DP expected cost:    97.4493

Out-of-sample mean cost
PFA:                       106.4520
CFA:                       102.2960
VFA:                        98.9198
DLA:                       108.5618
Exact DP benchmark:         97.7774
```

These numbers are demonstration results for this synthetic instance. They are not claims that one policy class is generally superior to another. Changing the demand process, cost structure, information state, action constraints, approximation architecture, or tuning budget can change the ranking.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_comparison.py
├── src/sda_policies/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── dp.py
│   ├── experiment.py
│   ├── model.py
│   ├── policies.py
│   └── tuning.py
├── tests/
│   ├── test_benchmark_cli.py
│   ├── test_dp_and_tuning.py
│   ├── test_model.py
│   └── test_policies.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported.

## Run

```bash
python -m sda_policies \
  --training-replications 120 \
  --validation-replications 500 \
  --seed 2026
```

or

```bash
python examples/run_comparison.py
```

## Tests

```bash
python -m pytest
```

The test suite verifies:

- probability-model validation,
- state-transition and cost accounting,
- reproducible exogenous-information traces,
- direct PFA behavior,
- parameter-sensitive CFA decisions,
- VFA feasibility,
- exact/VFA agreement when the approximation grid is made exact,
- DLA action feasibility,
- exact dynamic programming on a one-period problem with a known solution,
- exact-policy simulation on random traces,
- deterministic CRN-based tuning,
- benchmark reproducibility,
- confidence-interval consistency,
- CLI JSON output.

GitHub Actions installs the package, compiles the source tree, and runs the test suite on Python 3.10 and Python 3.12.

## Methodological notes

- The four classes describe methods for making decisions, not four mutually exclusive academic fields.
- A PFA can be simple or very complex. A neural-network policy is still a PFA if it maps the state directly to an action.
- A CFA is not "just a cost function." The defining idea is a parameterized optimization model whose modified objective or constraints are tuned to perform well over time.
- A VFA is not synonymous with all reinforcement learning. It is the class of policies that uses an approximation of downstream value. Q-learning and many approximate dynamic-programming methods fall naturally in this class.
- A DLA explicitly plans forward. Deterministic model-predictive-control-style lookaheads and stochastic scenario-tree lookaheads are both members of the broader DLA class.
- Three of the four classes implemented here solve an embedded optimization problem. The PFA is the exception.
- The exact DP benchmark is intentionally small. It should not be interpreted as a scalable replacement for approximate policies.
- The confidence intervals summarize Monte Carlo sampling error for fixed selected policies. They do not fully account for the selection bias introduced during parameter tuning.

## References

- Warren B. Powell, **Policies — Sequential Decision Analytics**: https://warrenpowell.org/policies/
- Warren B. Powell, **Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions**: https://warrenpowell.org/rlso/
- Warren B. Powell, **Reinforcement Learning versus Sequential Decision Analytics**: https://warrenpowell.org/rlvssda/
- Warren B. Powell, **Bridging Decision Problems — Sequential Decision Analytics**: https://warrenpowell.org/bridgingdecisionproblems/

## License

MIT
