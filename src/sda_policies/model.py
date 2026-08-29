from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol, Sequence


@dataclass(frozen=True)
class InventoryConfig:
    """Finite-horizon stochastic inventory model used by every policy class."""

    horizon: int = 12
    initial_inventory: int = 6
    max_order: int = 10
    holding_cost: float = 0.8
    backlog_cost: float = 5.0
    unit_order_cost: float = 1.0
    fixed_order_cost: float = 2.5
    discount: float = 1.0
    demand_values: tuple[int, ...] = (0, 2, 4, 6, 8)
    demand_probabilities: tuple[float, ...] = (0.05, 0.20, 0.40, 0.25, 0.10)
    demand_probability_schedule: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.max_order <= 0:
            raise ValueError("max_order must be positive")
        for name, value in (
            ("holding_cost", self.holding_cost),
            ("backlog_cost", self.backlog_cost),
            ("unit_order_cost", self.unit_order_cost),
            ("fixed_order_cost", self.fixed_order_cost),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if len(self.demand_values) == 0:
            raise ValueError("demand support cannot be empty")
        if len(self.demand_values) != len(self.demand_probabilities):
            raise ValueError("demand values and probabilities must have equal length")
        if any((not isinstance(d, int)) or d < 0 for d in self.demand_values):
            raise ValueError("demand values must be nonnegative integers")
        if tuple(sorted(set(self.demand_values))) != self.demand_values:
            raise ValueError("demand values must be strictly increasing")
        if any(p < 0 or not math.isfinite(p) for p in self.demand_probabilities):
            raise ValueError("demand probabilities must be finite and nonnegative")
        if not math.isclose(sum(self.demand_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("demand probabilities must sum to one")
        if self.demand_probability_schedule is not None:
            if len(self.demand_probability_schedule) != self.horizon:
                raise ValueError("demand probability schedule length must equal the horizon")
            for row in self.demand_probability_schedule:
                if len(row) != len(self.demand_values):
                    raise ValueError("every scheduled probability row must match the demand support")
                if any(p < 0 or not math.isfinite(p) for p in row):
                    raise ValueError("scheduled probabilities must be finite and nonnegative")
                if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("every scheduled probability row must sum to one")

    @property
    def mean_demand(self) -> float:
        return sum(self.mean_demand_at(t) for t in range(self.horizon)) / self.horizon

    def probabilities_at(self, time: int) -> tuple[float, ...]:
        if time < 0 or time >= self.horizon:
            raise ValueError("time must be inside the planning horizon")
        if self.demand_probability_schedule is None:
            return self.demand_probabilities
        return self.demand_probability_schedule[time]

    def mean_demand_at(self, time: int) -> float:
        return sum(d * p for d, p in zip(self.demand_values, self.probabilities_at(time)))

    @property
    def max_demand(self) -> int:
        return self.demand_values[-1]

    def reachable_inventory_bounds(self) -> tuple[int, int]:
        minimum = self.initial_inventory - self.horizon * self.max_demand
        maximum = self.initial_inventory + self.horizon * self.max_order
        return minimum, maximum


@dataclass(frozen=True)
class InventoryState:
    """State variable S_t = (time, net inventory). Negative inventory is backlog."""

    time: int
    inventory: int


@dataclass(frozen=True)
class StepRecord:
    time: int
    starting_inventory: int
    order: int
    demand: int
    ending_inventory: int
    order_cost: float
    inventory_cost: float
    total_cost: float


@dataclass(frozen=True)
class SimulationResult:
    total_discounted_cost: float
    final_inventory: int
    records: tuple[StepRecord, ...]


class Policy(Protocol):
    name: str

    def decide(self, state: InventoryState, config: InventoryConfig) -> int:
        ...


def validate_action(action: int, config: InventoryConfig) -> int:
    if not isinstance(action, int):
        raise TypeError("policy actions must be integers")
    if action < 0 or action > config.max_order:
        raise ValueError("policy action violates the order bounds")
    return action


def order_cost(order: int, config: InventoryConfig) -> float:
    validate_action(order, config)
    return config.unit_order_cost * order + (config.fixed_order_cost if order > 0 else 0.0)


def inventory_cost(ending_inventory: int, config: InventoryConfig) -> float:
    if ending_inventory >= 0:
        return config.holding_cost * ending_inventory
    return config.backlog_cost * (-ending_inventory)


def transition(state: InventoryState, order: int, demand: int, config: InventoryConfig) -> InventoryState:
    validate_action(order, config)
    if state.time < 0 or state.time >= config.horizon:
        raise ValueError("state time must be inside the decision horizon")
    if demand not in config.demand_values:
        raise ValueError("demand must belong to the configured exogenous-information support")
    return InventoryState(time=state.time + 1, inventory=state.inventory + order - demand)


def generate_demand_trace(config: InventoryConfig, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(
        rng.choices(config.demand_values, weights=config.probabilities_at(t), k=1)[0]
        for t in range(config.horizon)
    )


def seasonal_demo_config() -> InventoryConfig:
    """Return a reproducible nonstationary demand profile used by the repository demo."""
    low = (0.12, 0.33, 0.38, 0.13, 0.04)
    medium = (0.05, 0.20, 0.40, 0.25, 0.10)
    peak = (0.02, 0.08, 0.25, 0.40, 0.25)
    schedule = (low, low, medium, medium, peak, peak, peak, medium, medium, low, low, low)
    return InventoryConfig(demand_probability_schedule=schedule)


def simulate_policy(
    config: InventoryConfig,
    policy: Policy,
    demand_trace: Sequence[int],
) -> SimulationResult:
    if len(demand_trace) != config.horizon:
        raise ValueError("demand trace length must equal the planning horizon")

    state = InventoryState(time=0, inventory=config.initial_inventory)
    records: list[StepRecord] = []
    discounted_total = 0.0

    for t, demand in enumerate(demand_trace):
        if demand not in config.demand_values:
            raise ValueError("demand trace contains a value outside the configured support")
        action = validate_action(policy.decide(state, config), config)
        next_state = transition(state, action, int(demand), config)
        ordering = order_cost(action, config)
        inventory = inventory_cost(next_state.inventory, config)
        one_period = ordering + inventory
        discounted_total += (config.discount ** t) * one_period
        records.append(
            StepRecord(
                time=t,
                starting_inventory=state.inventory,
                order=action,
                demand=int(demand),
                ending_inventory=next_state.inventory,
                order_cost=ordering,
                inventory_cost=inventory,
                total_cost=one_period,
            )
        )
        state = next_state

    return SimulationResult(
        total_discounted_cost=discounted_total,
        final_inventory=state.inventory,
        records=tuple(records),
    )
