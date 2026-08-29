from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from .model import (
    InventoryConfig,
    InventoryState,
    inventory_cost,
    order_cost,
)


@dataclass(frozen=True)
class OrderUpToPFA:
    """Policy Function Approximation: analytical order-up-to mapping S_t -> x_t."""

    target_inventory: int
    name: str = "PFA-order-up-to"

    def decide(self, state: InventoryState, config: InventoryConfig) -> int:
        return max(0, min(config.max_order, self.target_inventory - state.inventory))


@dataclass(frozen=True)
class CFAOneStepPolicy:
    """Cost Function Approximation using a parameterized deterministic one-step model."""

    forecast_bias: float = 0.0
    backlog_multiplier: float = 1.0
    name: str = "CFA-one-step"

    def __post_init__(self) -> None:
        if not math.isfinite(self.forecast_bias):
            raise ValueError("forecast_bias must be finite")
        if self.backlog_multiplier <= 0 or not math.isfinite(self.backlog_multiplier):
            raise ValueError("backlog_multiplier must be finite and positive")

    def surrogate_cost(self, state: InventoryState, order: int, config: InventoryConfig) -> float:
        forecast = max(0.0, config.mean_demand_at(state.time) + self.forecast_bias)
        projected = state.inventory + order - forecast
        holding = config.holding_cost * max(projected, 0.0)
        backlog = self.backlog_multiplier * config.backlog_cost * max(-projected, 0.0)
        return order_cost(order, config) + holding + backlog

    def decide(self, state: InventoryState, config: InventoryConfig) -> int:
        return min(
            range(config.max_order + 1),
            key=lambda q: (self.surrogate_cost(state, q, config), q),
        )


@dataclass(frozen=True)
class CoarseValueFunction:
    """Time-indexed piecewise-linear approximation of downstream value."""

    knots: tuple[int, ...]
    values_by_time: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.knots) < 2:
            raise ValueError("at least two value-function knots are required")
        if tuple(sorted(set(self.knots))) != self.knots:
            raise ValueError("value-function knots must be strictly increasing")
        if any(len(row) != len(self.knots) for row in self.values_by_time):
            raise ValueError("every value-function row must match the knot count")

    def value(self, time: int, inventory: int | float) -> float:
        if time < 0 or time >= len(self.values_by_time):
            raise ValueError("time is outside the fitted value-function horizon")
        row = self.values_by_time[time]
        x = float(inventory)
        if x <= self.knots[0]:
            return row[0]
        if x >= self.knots[-1]:
            return row[-1]

        lo = 0
        hi = len(self.knots) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.knots[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0 = self.knots[lo]
        x1 = self.knots[hi]
        weight = (x - x0) / (x1 - x0)
        return (1.0 - weight) * row[lo] + weight * row[hi]


@dataclass(frozen=True)
class VFAInventoryPolicy:
    """Policy based on an approximate downstream value function."""

    value_function: CoarseValueFunction
    name: str = "VFA-coarse-value"

    def decide(self, state: InventoryState, config: InventoryConfig) -> int:
        if state.time >= config.horizon:
            raise ValueError("no decision exists after the horizon")

        def expected_cost(order: int) -> float:
            immediate_order = order_cost(order, config)
            total = 0.0
            for demand, probability in zip(config.demand_values, config.probabilities_at(state.time)):
                ending = state.inventory + order - demand
                total += probability * (
                    inventory_cost(ending, config)
                    + config.discount * self.value_function.value(state.time + 1, ending)
                )
            return immediate_order + total

        return min(range(config.max_order + 1), key=lambda q: (expected_cost(q), q))


def _build_knots(config: InventoryConfig, spacing: int) -> tuple[int, ...]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    lower, upper = config.reachable_inventory_bounds()
    knots = list(range(lower, upper + 1, spacing))
    if knots[-1] != upper:
        knots.append(upper)
    return tuple(knots)


def fit_coarse_vfa(config: InventoryConfig, spacing: int = 6) -> VFAInventoryPolicy:
    """Fit a coarse-grid approximate dynamic program and return its VFA policy."""
    knots = _build_knots(config, spacing)
    terminal = tuple(0.0 for _ in knots)
    rows: list[tuple[float, ...]] = [terminal for _ in range(config.horizon + 1)]
    approximate = CoarseValueFunction(knots=knots, values_by_time=tuple(rows))

    for t in range(config.horizon - 1, -1, -1):
        values: list[float] = []
        for inventory in knots:
            def action_value(order: int) -> float:
                total = order_cost(order, config)
                expected = 0.0
                for demand, probability in zip(config.demand_values, config.probabilities_at(t)):
                    ending = inventory + order - demand
                    expected += probability * (
                        inventory_cost(ending, config)
                        + config.discount * approximate.value(t + 1, ending)
                    )
                return total + expected

            values.append(min(action_value(q) for q in range(config.max_order + 1)))

        rows[t] = tuple(values)
        approximate = CoarseValueFunction(knots=knots, values_by_time=tuple(rows))

    return VFAInventoryPolicy(approximate)


@dataclass(frozen=True)
class DeterministicLookaheadPolicy:
    """Direct Lookahead Approximation using a rolling deterministic planning model."""

    lookahead_horizon: int = 4
    forecast_bias: float = 0.0
    terminal_backlog_multiplier: float = 1.0
    name: str = "DLA-deterministic-lookahead"

    def __post_init__(self) -> None:
        if self.lookahead_horizon <= 0:
            raise ValueError("lookahead_horizon must be positive")
        if not math.isfinite(self.forecast_bias):
            raise ValueError("forecast_bias must be finite")
        if self.terminal_backlog_multiplier <= 0:
            raise ValueError("terminal_backlog_multiplier must be positive")

    def decide(self, state: InventoryState, config: InventoryConfig) -> int:
        remaining = config.horizon - state.time
        depth = min(self.lookahead_horizon, remaining)

        @lru_cache(maxsize=None)
        def plan(stage: int, inventory: int) -> tuple[float, int]:
            if stage == depth:
                terminal = self.terminal_backlog_multiplier * config.backlog_cost * max(-inventory, 0)
                return terminal, 0

            best_value = math.inf
            best_order = 0
            forecast = max(0, int(round(config.mean_demand_at(state.time + stage) + self.forecast_bias)))
            for order in range(config.max_order + 1):
                ending = inventory + order - forecast
                future, _ = plan(stage + 1, ending)
                value = (
                    order_cost(order, config)
                    + inventory_cost(ending, config)
                    + config.discount * future
                )
                if value < best_value - 1e-12 or (
                    math.isclose(value, best_value, abs_tol=1e-12) and order < best_order
                ):
                    best_value = value
                    best_order = order
            return best_value, best_order

        return plan(0, state.inventory)[1]
