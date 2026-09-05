"""Exact arithmetic checks for the corrected appendix; no models or simulations.

This independent implementation preserves the old frozen evidence. Its output
is a proof-check receipt, not a new empirical experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from pathlib import Path


def witness_joint(mechanism: str, e0: F, e1: F, delayed: bool) -> dict:
    if mechanism not in {"E", "T"}:
        raise ValueError("unknown mechanism")
    result = defaultdict(F)
    for z, a in product((0, 1), repeat=2):
        mass = (F(3, 5) if z else F(2, 5))
        propensity = e1 if z else e0
        mass *= propensity if a else 1 - propensity
        # Exact witness c0=1,c1=0, encoded independently of source functions.
        o = 0 if a == 0 else z
        persistent = z if mechanism == "E" else o
        key = (z, a, o, persistent) if delayed else (z, a, o)
        result[key] += mass
    return dict(result)


def tv(left: dict, right: dict) -> F:
    return sum((abs(left.get(k, F()) - right.get(k, F()))
                for k in left.keys() | right.keys()), F()) / 2


def risk(q: F) -> tuple[F, F]:
    return (1 - q) / 5, 2 * q / 5


def fixed_state_policy_value(mechanism: str, action_for_z: tuple[int, int]) -> F:
    value = F()
    for z in (0, 1):
        a = action_for_z[z]
        persistent = z if mechanism == "E" or a == 1 else 0
        value += (F(3, 5) if z else F(2, 5)) * int(a == persistent)
    return value


def matmul(left: list, right: list) -> list:
    return [[sum((x * y for x, y in zip(row, col)), F())
             for col in zip(*right)] for row in left]


def inverse(matrix: list) -> list:
    n = len(matrix)
    rows = [list(row) + [F(i == j) for j in range(n)]
            for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j]), None)
        if pivot is None:
            raise ValueError("singular matrix")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [x / scale for x in rows[j]]
        for i in range(n):
            if i != j:
                scale = rows[i][j]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[j])]
    return [row[n:] for row in rows]


def clipping_counterexample() -> dict:
    immediate, delayed = (F(1), F(0)), (F(0), F(0))
    estimates = [sum(immediate) / 2 + delayed[i] - immediate[i]
                 for i in (0, 1)]
    clipped = [min(F(1), max(F(0), x)) for x in estimates]
    return {"unclipped": estimates, "clipped": clipped,
            "unclipped_expectation": sum(estimates) / 2,
            "clipped_expectation": sum(clipped) / 2}


def enumerate_variance(law: dict) -> tuple[F, F]:
    """Check appendix variance identity exactly for N=2, k=1."""
    mean_o = sum((o * p for (o, b), p in law.items()), F())
    mean_b = sum((b * p for (o, b), p in law.items()), F())
    var_o = sum(((o - mean_o)**2 * p for (o, b), p in law.items()), F())
    var_b = sum(((b - mean_b)**2 * p for (o, b), p in law.items()), F())
    cov = sum(((o - mean_o) * (b - mean_b) * p
               for (o, b), p in law.items()), F())
    exact_var, exact_mean = F(), F()
    for (first, p1), (second, p2) in product(law.items(), repeat=2):
        for selected in (first, second):
            estimate = (first[0] + second[0]) / F(2) + selected[1] - selected[0]
            weight = p1 * p2 / 2
            exact_mean += weight * estimate
            exact_var += weight * (estimate - mean_b)**2
    assert exact_mean == mean_b
    formula_var = var_b + F(1, 2) * (var_o - 2 * cov)
    return exact_var, formula_var


def verify() -> dict:
    balanced = (F(1, 2), F(1, 2))
    state_dependent = (F(7, 20), F(7, 10))
    expected_atoms = {(0, 0, 0): F(1, 5), (0, 1, 0): F(1, 5),
                      (1, 0, 0): F(3, 10), (1, 1, 1): F(3, 10)}
    assert witness_joint("E", *balanced, False) == expected_atoms
    for logging in (balanced, state_dependent):
        left = witness_joint("E", *logging, False)
        right = witness_joint("T", *logging, False)
        assert sum(left.values()) == 1 and left == right
    balanced_tv = tv(witness_joint("E", *balanced, True),
                     witness_joint("T", *balanced, True))
    state_dependent_tv = tv(witness_joint("E", *state_dependent, True),
                           witness_joint("T", *state_dependent, True))
    assert balanced_tv == F(3, 10) and state_dependent_tv == F(9, 50)
    values = {m: [fixed_state_policy_value(m, (a, a)) for a in (0, 1)]
              for m in ("E", "T")}
    assert values == {"E": [F(2, 5), F(3, 5)], "T": [F(1), F(3, 5)]}
    assert risk(F(1, 3)) == (F(2, 15), F(2, 15))
    # Affine monotonicity proves the all-q bound in the appendix; these are
    # independent corner/intersection checks, not an empirical minimization.
    assert risk(F(0))[0] == F(1, 5) and risk(F(1))[1] == F(2, 5)
    deterministic_data_risks = risk(F(2, 5))
    assert deterministic_data_risks == (F(3, 25), F(4, 25))
    assert max(deterministic_data_risks) < F(1, 5)
    assert all(fixed_state_policy_value(m, (0, 1)) == 1 for m in ("E", "T"))
    emission = [[F(85, 100), F(10, 100), F(5, 100)],
                [F(5, 100), F(90, 100), F(5, 100)],
                [F(10, 100), F(10, 100), F(80, 100)]]
    transition = [[F(8, 10), F(1, 10), F(1, 10)],
                  [F(2, 10), F(7, 10), F(1, 10)],
                  [F(1, 10), F(2, 10), F(7, 10)]]
    assert matmul(matmul(transition, emission), inverse(emission)) == transition
    deficient = [[F(1), F(0)], [F(0), F(1)], [F(0), F(1)]]
    identity = [[F(i == j) for j in range(3)] for i in range(3)]
    swap = [[F(1), F(0), F(0)], [F(0), F(0), F(1)], [F(0), F(1), F(0)]]
    assert identity != swap and matmul(identity, deficient) == matmul(swap, deficient)
    clipping = clipping_counterexample()
    assert clipping["unclipped_expectation"] == 0
    assert clipping["clipped_expectation"] == F(1, 4)
    for law in [{(0, 0): F(1, 2), (1, 1): F(1, 2)},
                {(0, 1): F(1, 2), (1, 0): F(1, 2)},
                {(0, 0): F(1, 5), (0, 1): F(1, 10),
                 (1, 0): F(2, 5), (1, 1): F(3, 10)}]:
        exact, formula = enumerate_variance(law)
        assert exact == formula
    # Exact HT expectation with nonuniform inclusion, no simulation.
    inclusion = (F(1, 4), F(3, 4))
    o, b = (F(1), F(0)), (F(0), F(1))
    expectation = F()
    for selected in product((0, 1), repeat=2):
        mass = F(1)
        for i, r in enumerate(selected):
            mass *= inclusion[i] if r else 1 - inclusion[i]
        estimate = sum((o[i] + selected[i] / inclusion[i] * (b[i] - o[i])
                        for i in range(2)), F()) / 2
        expectation += mass * estimate
    assert expectation == sum(b) / 2
    return {"status": "CORRECTED_THEORY_EXACT_CHECKS_PASS",
            "classification": "mathematical_replay_not_empirical_experiment",
            "immediate_tv": F(0), "balanced_delayed_tv": balanced_tv,
            "state_dependent_delayed_tv": state_dependent_tv,
            "constant_action_values": values, "randomized_minimax": F(2, 15),
            "minimax_q1": F(1, 3), "fixed_action_minimax": F(1, 5),
            "deterministic_data_rule_expected_risks": deterministic_data_risks,
            "observed_current_state_value_both_worlds": F(1),
            "clipping_counterexample": clipping,
            "known_emission_recovery_exact": True,
            "rank_deficient_alias_exact": True,
            "variance_identity_exact_laws_checked": 3,
            "ht_difference_expectation": expectation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {str(p.relative_to(root)).replace("\\", "/"):
        sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__).resolve(), root / "paper/sections/theory.tex")}
    serialized = json.dumps(result, indent=2, default=str, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
