import math

import pytest

from sda_policies.model import (
    InventoryConfig,
    InventoryState,
    generate_demand_trace,
    inventory_cost,
    order_cost,
    transition,
)


def test_config_rejects_invalid_probability_mass():
    with pytest.raises(ValueError):
        InventoryConfig(demand_probabilities=(0.1, 0.2, 0.3, 0.2, 0.1))


def test_cost_and_transition_are_consistent():
    cfg = InventoryConfig()
    state = InventoryState(time=0, inventory=3)
    next_state = transition(state, order=4, demand=6, config=cfg)
    assert next_state == InventoryState(time=1, inventory=1)
    assert math.isclose(order_cost(4, cfg), 6.5)
    assert math.isclose(inventory_cost(next_state.inventory, cfg), 0.8)
    assert math.isclose(inventory_cost(-2, cfg), 10.0)


def test_demand_trace_is_reproducible_and_supported():
    cfg = InventoryConfig()
    first = generate_demand_trace(cfg, 123)
    second = generate_demand_trace(cfg, 123)
    third = generate_demand_trace(cfg, 124)
    assert first == second
    assert first != third
    assert len(first) == cfg.horizon
    assert set(first).issubset(set(cfg.demand_values))
