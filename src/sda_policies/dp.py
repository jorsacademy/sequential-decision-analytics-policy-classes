from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from .model import InventoryConfig, InventoryState, inventory_cost, order_cost


@dataclass(frozen=True)
class ExactDPResult:
    expected_cost: float
    policy_table: dict[tuple[int, int], int]
    value_table: dict[tuple[int, int], float]


@dataclass(frozen=True)
class ExactDynamicProgrammingPolicy:
    """Exact finite-horizon MDP policy used only as a benchmark."""

    actions: dict[tuple[int, int], int]
    name: str = "exact-DP-benchmark"

    def decide(self, state: InventoryState, config: InventoryConfig) -> int:
        try:
            return self.actions[(state.time, state.inventory)]
        except KeyError as exc:
            raise ValueError("state is outside the exact DP policy table") from exc


def solve_exact_dp(config: InventoryConfig) -> ExactDPResult:
    """Solve the finite-horizon inventory MDP exactly on its reachable integer states."""
    actions: dict[tuple[int, int], int] = {}
    values: dict[tuple[int, int], float] = {}

    @lru_cache(maxsize=None)
    def value(time: int, inventory: int) -> float:
        if time == config.horizon:
            values[(time, inventory)] = 0.0
            return 0.0

        best_value = math.inf
        best_order = 0
        for order in range(config.max_order + 1):
            total = order_cost(order, config)
            expected = 0.0
            for demand, probability in zip(config.demand_values, config.probabilities_at(time)):
                ending = inventory + order - demand
                expected += probability * (
                    inventory_cost(ending, config)
                    + config.discount * value(time + 1, ending)
                )
            candidate = total + expected
            if candidate < best_value - 1e-12 or (
                math.isclose(candidate, best_value, abs_tol=1e-12) and order < best_order
            ):
                best_value = candidate
                best_order = order

        actions[(time, inventory)] = best_order
        values[(time, inventory)] = best_value
        return best_value

    expected = value(0, config.initial_inventory)
    return ExactDPResult(expected_cost=expected, policy_table=actions, value_table=values)
