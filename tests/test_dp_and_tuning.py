import math

from sda_policies.dp import ExactDynamicProgrammingPolicy, solve_exact_dp
from sda_policies.model import InventoryConfig, generate_demand_trace, simulate_policy
from sda_policies.policies import OrderUpToPFA
from sda_policies.tuning import grid_search_parameter


def test_exact_dp_is_self_consistent_on_one_period_problem():
    cfg = InventoryConfig(
        horizon=1,
        initial_inventory=0,
        max_order=4,
        fixed_order_cost=0.0,
        unit_order_cost=0.0,
        holding_cost=1.0,
        backlog_cost=10.0,
        demand_values=(0, 2),
        demand_probabilities=(0.5, 0.5),
    )
    result = solve_exact_dp(cfg)
    assert result.policy_table[(0, 0)] == 2
    assert math.isclose(result.expected_cost, 1.0)


def test_exact_policy_simulates_on_generated_traces():
    cfg = InventoryConfig(horizon=5)
    result = solve_exact_dp(cfg)
    policy = ExactDynamicProgrammingPolicy(result.policy_table)
    trace = generate_demand_trace(cfg, 99)
    simulation = simulate_policy(cfg, policy, trace)
    assert len(simulation.records) == cfg.horizon
    assert simulation.total_discounted_cost >= 0


def test_grid_search_uses_same_traces_and_is_reproducible():
    cfg = InventoryConfig(horizon=5)
    traces = [generate_demand_trace(cfg, 500 + i) for i in range(20)]
    first = grid_search_parameter(
        cfg,
        range(4, 10),
        lambda target: OrderUpToPFA(int(target)),
        traces,
    )
    second = grid_search_parameter(
        cfg,
        range(4, 10),
        lambda target: OrderUpToPFA(int(target)),
        traces,
    )
    assert first == second
    assert 4 <= first.parameter <= 9
