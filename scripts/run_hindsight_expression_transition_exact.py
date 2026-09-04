"""Emit an immutable exact counterexample receipt."""
import argparse
import json
from pathlib import Path

from interaction_sprint.hindsight_expression_transition import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report(), handle, indent=2, sort_keys=True)
    print(json.dumps(report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

