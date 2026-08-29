import json
import subprocess
import sys

from sda_policies.benchmark import benchmark_policies
from sda_policies.model import InventoryConfig


def test_benchmark_contains_all_four_classes_and_exact_reference():
    result = benchmark_policies(
        config=InventoryConfig(horizon=6),
        training_replications=20,
        validation_replications=30,
        seed=7,
    )
    names = {summary.name for summary in result.summaries}
    assert names == {
        "PFA-order-up-to",
        "CFA-one-step",
        "VFA-coarse-value",
        "DLA-deterministic-lookahead",
        "exact-DP-benchmark",
    }
    assert result.exact_expected_cost >= 0
    for summary in result.summaries:
        assert summary.ci95_low <= summary.mean_cost <= summary.ci95_high


def test_benchmark_is_reproducible():
    kwargs = dict(
        config=InventoryConfig(horizon=5),
        training_replications=15,
        validation_replications=20,
        seed=44,
    )
    assert benchmark_policies(**kwargs) == benchmark_policies(**kwargs)


def test_cli_outputs_valid_json():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sda_policies",
            "--training-replications",
            "10",
            "--validation-replications",
            "12",
            "--seed",
            "11",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert "out_of_sample" in payload
    assert len(payload["out_of_sample"]) == 5
