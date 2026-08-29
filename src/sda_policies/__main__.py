from __future__ import annotations

import argparse
import json

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the four Sequential Decision Analytics policy classes on one inventory model."
    )
    parser.add_argument("--training-replications", type=int, default=120)
    parser.add_argument("--validation-replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_experiment(
        training_replications=args.training_replications,
        validation_replications=args.validation_replications,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
