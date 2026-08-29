from __future__ import annotations

from dataclasses import asdict

from .benchmark import benchmark_policies
from .model import seasonal_demo_config


def run_experiment(
    training_replications: int = 120,
    validation_replications: int = 500,
    seed: int = 2026,
) -> dict:
    config = seasonal_demo_config()
    result = benchmark_policies(
        config=config,
        training_replications=training_replications,
        validation_replications=validation_replications,
        seed=seed,
    )
    return {
        "model": {
            "horizon": config.horizon,
            "initial_inventory": config.initial_inventory,
            "max_order": config.max_order,
            "mean_demand": config.mean_demand,
            "mean_demand_by_period": [config.mean_demand_at(t) for t in range(config.horizon)],
            "demand_values": list(config.demand_values),
            "demand_probability_schedule": [list(config.probabilities_at(t)) for t in range(config.horizon)],
        },
        "tuning": {
            "pfa_target_inventory": result.tuned_pfa_target,
            "cfa_forecast_bias": result.tuned_cfa_bias,
        },
        "exact_dp_expected_cost": result.exact_expected_cost,
        "out_of_sample": [asdict(summary) for summary in result.summaries],
    }
