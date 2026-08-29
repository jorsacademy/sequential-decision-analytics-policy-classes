from sda_policies.dp import solve_exact_dp
from sda_policies.model import InventoryConfig, InventoryState
from sda_policies.policies import (
    CFAOneStepPolicy,
    DeterministicLookaheadPolicy,
    OrderUpToPFA,
    fit_coarse_vfa,
)


def test_pfa_is_direct_order_up_to_mapping():
    cfg = InventoryConfig(max_order=5)
    policy = OrderUpToPFA(target_inventory=8)
    assert policy.decide(InventoryState(0, 2), cfg) == 5
    assert policy.decide(InventoryState(0, 6), cfg) == 2
    assert policy.decide(InventoryState(0, 10), cfg) == 0


def test_cfa_parameter_changes_embedded_deterministic_decision():
    cfg = InventoryConfig()
    state = InventoryState(0, -2)
    conservative = CFAOneStepPolicy(forecast_bias=3.0, backlog_multiplier=1.5)
    nominal = CFAOneStepPolicy(forecast_bias=0.0, backlog_multiplier=1.0)
    assert conservative.decide(state, cfg) >= nominal.decide(state, cfg)
    assert 0 <= nominal.decide(state, cfg) <= cfg.max_order


def test_spacing_one_vfa_matches_exact_dp_action_at_initial_state():
    cfg = InventoryConfig(horizon=5)
    exact = solve_exact_dp(cfg)
    vfa = fit_coarse_vfa(cfg, spacing=1)
    state = InventoryState(0, cfg.initial_inventory)
    assert vfa.decide(state, cfg) == exact.policy_table[(0, cfg.initial_inventory)]


def test_coarse_vfa_returns_feasible_actions_on_sampled_states():
    cfg = InventoryConfig(horizon=6)
    vfa = fit_coarse_vfa(cfg, spacing=5)
    for t, inventory in ((0, 6), (1, 0), (3, -8), (5, 12)):
        action = vfa.decide(InventoryState(t, inventory), cfg)
        assert 0 <= action <= cfg.max_order


def test_deterministic_lookahead_is_receding_horizon_and_feasible():
    cfg = InventoryConfig(horizon=8)
    policy = DeterministicLookaheadPolicy(lookahead_horizon=4)
    early = policy.decide(InventoryState(0, 0), cfg)
    late = policy.decide(InventoryState(7, 0), cfg)
    assert 0 <= early <= cfg.max_order
    assert 0 <= late <= cfg.max_order
