"""Emit the exact two-mechanism minimax decision lower bound."""
import argparse
import json
from pathlib import Path

from interaction_sprint.hindsight_expression_transition import (
    immediate_joint,
    observational_minimax_regret,
    total_variation,
)


def result() -> dict[str, object]:
    p = .6
    propensity = {0: .35, 1: .7}
    copying = {0: 1., 1: 0.}
    expression = immediate_joint(p, propensity, copying, mechanism="expression")
    transition = immediate_joint(p, propensity, copying, mechanism="transition")
    return {
        "decision": "POSITIVE_OBSERVATIONAL_MINIMAX_REGRET_LOWER_BOUND",
        "parameters": {
            "p_z0_one": p,
            "logging_action_one_probability": propensity,
            "copying_by_action": copying,
        },
        "immediate_log_total_variation": total_variation(expression, transition),
        "bound": observational_minimax_regret(p, copying),
        "scope": (
            "Exact binary two-point decision bound for constant actions when a new user's initial "
            "preference is unavailable. It is independent of passive sample size because the two "
            "immediate-log laws are identical. It is not a prevalence or welfare claim."
        ),
        "paper_green_light": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = result()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
