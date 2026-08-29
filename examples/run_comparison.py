from pprint import pprint

from sda_policies.experiment import run_experiment


if __name__ == "__main__":
    pprint(run_experiment(training_replications=120, validation_replications=500, seed=2026))
