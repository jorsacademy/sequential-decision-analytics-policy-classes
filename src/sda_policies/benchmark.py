from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, stdev

from .dp import ExactDynamicProgrammingPolicy, solve_exact_dp
from .model import InventoryConfig, Policy, generate_demand_trace, seasonal_demo_config, simulate_policy
from .policies import CFAOneStepPolicy, DeterministicLookaheadPolicy, OrderUpToPFA, fit_coarse_vfa
from .tuning import grid_search_parameter


@dataclass(frozen=True)
class PolicySummary:
    name: str
    mean_cost: float
    standard_error: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class BenchmarkResult:
    tuned_pfa_target: int
    tuned_cfa_bias: float
    exact_expected_cost: float
    summaries: tuple[PolicySummary, ...]


def _summarize(name: str, costs: list[float]) -> PolicySummary:
    mean = fmean(costs)
    if len(costs) <= 1:
        se = 0.0
    else:
        se = stdev(costs) / math.sqrt(len(costs))
    radius = 1.96 * se
    return PolicySummary(
        name=name,
        mean_cost=mean,
        standard_error=se,
        ci95_low=mean - radius,
        ci95_high=mean + radius,
    )


def _evaluate(config: InventoryConfig, policy: Policy, traces: list[tuple[int, ...]]) -> PolicySummary:
    costs = [simulate_policy(config, policy, trace).total_discounted_cost for trace in traces]
    return _summarize(policy.name, costs)


def benchmark_policies(
    config: InventoryConfig | None = None,
    training_replications: int = 120,
    validation_replications: int = 500,
    seed: int = 2026,
) -> BenchmarkResult:
    """Tune policy-search parameters on CRN traces and compare policies out of sample."""
    cfg = config or seasonal_demo_config()
    if training_replications <= 0 or validation_replications <= 0:
        raise ValueError("replication counts must be positive")

    training_traces = [generate_demand_trace(cfg, seed + i) for i in range(training_replications)]
    validation_seed = seed + 100_000
    validation_traces = [
        generate_demand_trace(cfg, validation_seed + i) for i in range(validation_replications)
    ]

    pfa_tuning = grid_search_parameter(
        cfg,
        parameters=range(4, 19),
        policy_factory=lambda target: OrderUpToPFA(int(target)),
        traces=training_traces,
    )
    cfa_tuning = grid_search_parameter(
        cfg,
        parameters=(-2, -1, 0, 1, 2, 3, 4),
        policy_factory=lambda bias: CFAOneStepPolicy(forecast_bias=float(bias), backlog_multiplier=1.0),
        traces=training_traces,
    )

    exact = solve_exact_dp(cfg)
    policies: list[Policy] = [
        OrderUpToPFA(int(pfa_tuning.parameter)),
        CFAOneStepPolicy(forecast_bias=cfa_tuning.parameter, backlog_multiplier=1.0),
        fit_coarse_vfa(cfg, spacing=10),
        DeterministicLookaheadPolicy(lookahead_horizon=4, forecast_bias=1.0),
        ExactDynamicProgrammingPolicy(exact.policy_table),
    ]

    summaries = tuple(_evaluate(cfg, policy, validation_traces) for policy in policies)
    return BenchmarkResult(
        tuned_pfa_target=int(pfa_tuning.parameter),
        tuned_cfa_bias=cfa_tuning.parameter,
        exact_expected_cost=exact.expected_cost,
        summaries=summaries,
    )
