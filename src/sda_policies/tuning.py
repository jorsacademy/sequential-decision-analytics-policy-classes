from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

from .model import InventoryConfig, Policy, simulate_policy


@dataclass(frozen=True)
class TuningResult:
    parameter: float
    mean_cost: float
    all_results: tuple[tuple[float, float], ...]


def evaluate_on_traces(
    config: InventoryConfig,
    policy: Policy,
    traces: Sequence[Sequence[int]],
) -> float:
    if not traces:
        raise ValueError("at least one trace is required")
    return fmean(simulate_policy(config, policy, trace).total_discounted_cost for trace in traces)


def grid_search_parameter(
    config: InventoryConfig,
    parameters: Iterable[float],
    policy_factory,
    traces: Sequence[Sequence[int]],
) -> TuningResult:
    results: list[tuple[float, float]] = []
    for parameter in parameters:
        policy = policy_factory(parameter)
        mean_cost = evaluate_on_traces(config, policy, traces)
        results.append((float(parameter), mean_cost))
    if not results:
        raise ValueError("parameter grid cannot be empty")
    best_parameter, best_cost = min(results, key=lambda item: (item[1], item[0]))
    return TuningResult(
        parameter=best_parameter,
        mean_cost=best_cost,
        all_results=tuple(results),
    )
