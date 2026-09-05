"""Joint DEV routing simulation and adversarial fixtures; not neural statistical power.

All rows below are artificial, with user/task/rotation dependence preserved.
The actual production decision function is invoked for every simulated study.
No dataset, model, confirmation file or external service is accessed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from interaction_sprint.hindsight_pahf_reduced import LETTERS, TRAINED_ARMS, binding, method_decision

ROOT = Path(__file__).resolve().parents[1]


def artificial_development():
    rows = []
    for base in range(96):
        for rotation in range(4):
            old, new = (base % 4 - rotation) % 4, ((base + 1) % 4 - rotation) % 4
            options = [f"semantic option {(j+rotation)%4}" for j in range(4)]
            rows.append({"id": f"synthetic-{base:03d}-r{rotation}", "base_id": f"synthetic-{base:03d}",
                         "label_rotation": rotation, "prompt": f"user{base%19:02d}: synthetic task\n" + "\n".join(f"{LETTERS[j]}) {text}" for j,text in enumerate(options)),
                         "old_target": LETTERS[old], "new_target": LETTERS[new],
                         "immediate_followup": "new", "delayed_expression_followup": "old", "delayed_transition_followup": "new"})
    return rows


def predictions(rows, old=.6, new=None, mass=.9, *, one_hot=False):
    old = np.broadcast_to(np.asarray(old, dtype=float), (96,))
    new = .5*(1-old) if new is None else np.broadcast_to(np.asarray(new, dtype=float), (96,))
    mass = np.broadcast_to(np.asarray(mass, dtype=float), (96,))
    output = []
    for index, row in enumerate(rows):
        b = index // 4
        p = [(1-old[b]-new[b])/2] * 4
        p[LETTERS.index(row["old_target"])] = old[b]
        p[LETTERS.index(row["new_target"])] = new[b]
        if one_hot:
            choose = row["old_target"] if b % 10 < 5 else row["new_target"]
            p = [float(l == choose) for l in LETTERS]
            full = p
            output.append({**binding(row), "full_vocab_choice_probabilities": full,
                           "normalized_choice_probabilities": p, "full_vocabulary_choice_mass": 1.})
        else:
            assert min(p) > 0 and abs(sum(p)-1) < 1e-12
            full = [float(mass[b]*v) for v in p]
            output.append({**binding(row), "full_vocab_choice_log_probabilities": [math.log(v) for v in full],
                           "normalized_choice_log_probabilities": [math.log(v) for v in p],
                           "full_vocabulary_choice_mass": float(mass[b])})
    return output


def fixture(rows):
    arms = {"baseline": predictions(rows, .26, .24), "raw_immediate": predictions(rows, .15, .65),
            "oracle_delayed": predictions(rows, .80, .10)}
    arms.update({name: predictions(rows, .60, .20) for name in ("pooled_sft", "pooled_sdpo", "mixture")})
    arms["residual"] = predictions(rows, .72, .14)
    cheap = {name: predictions(rows, one_hot=True) for name in ("anchor_memory", "anchor_profile")}
    return arms, cheap


def adversarial_checks(rows, config):
    outcomes = {}
    def check(name, arms, cheap, expected):
        actual = method_decision(rows, arms, cheap, config)
        assert actual["decision"] == expected, (name, actual["decision"])
        assert actual["paper_green_light"] is False
        outcomes[name] = {"decision": actual["decision"], "expected": expected}
    arms, cheap = fixture(rows)
    check("strong_controlled_alternative", arms, cheap, "REDUCED_DEV_METHOD_PROMISING")
    arms, cheap = fixture(rows)
    arms["residual"] = copy.deepcopy(arms["pooled_sdpo"])
    check("exact_equal_method_null", arms, cheap, "REDUCED_DEV_METHOD_NEGATIVE")
    arms, cheap = fixture(rows)
    for name in ("pooled_sft", "pooled_sdpo", "mixture"):
        arms[name] = predictions(rows, .60, .20, mass=.65)
    arms["residual"] = predictions(rows, .60, .20, mass=.99)
    check("format_only_method_gain", arms, cheap, "REDUCED_DEV_METHOD_NEGATIVE")
    arms, cheap = fixture(rows)
    arms["oracle_delayed"] = predictions(rows, .26, .24, mass=.99)
    check("oracle_format_only_acquisition", arms, cheap, "REDUCED_INVALID_ASSAY")
    arms, cheap = fixture(rows)
    cheap["anchor_memory"] = copy.deepcopy(arms["residual"])
    check("cheap_readout_matches", arms, cheap, "REDUCED_DEV_METHOD_NEGATIVE")
    arms, cheap = fixture(rows)
    # Positive overall gains concentrated in one user; leaving that user out loses.
    gains = np.asarray([.99 if b % 19 == 0 else .599 for b in range(96)])
    arms["residual"] = predictions(rows, gains, mass=.99)
    check("single_user_drives_gain", arms, cheap, "REDUCED_DEV_METHOD_NEGATIVE")
    arms, cheap = fixture(rows)
    # Opposite semantic order effects across half the tasks cancel globally.
    for index, pred in enumerate(arms["residual"]):
        base, rotation = index // 4, index % 4
        pold = .95 if ((base % 2) == (rotation % 2)) else .20
        replacement = predictions(rows, pold)[index]
        pred.update(replacement)
    check("opposing_order_effects_cancel_globally", arms, cheap, "REDUCED_INVALID_ASSAY")
    malformed = {}
    for case in ("missing_id", "duplicate_id", "wrong_user", "wrong_rotation", "inconsistent_mass"):
        arms, cheap = fixture(rows)
        if case == "missing_id": arms["residual"].pop()
        elif case == "duplicate_id": arms["residual"][1] = copy.deepcopy(arms["residual"][0])
        elif case == "wrong_user": arms["residual"][0]["source_user"] = "wrong"
        elif case == "wrong_rotation": arms["residual"][0]["label_rotation"] = 2
        elif case == "inconsistent_mass": arms["residual"][0]["full_vocabulary_choice_mass"] = .2
        try:
            method_decision(rows, arms, cheap, config)
        except ValueError as exc:
            malformed[case] = str(exc)
        else:
            raise AssertionError(f"accepted malformed fixture: {case}")
    return {"decision_cases": outcomes, "malformed_cases_rejected": malformed}


def interval(successes, trials):
    z = 1.959963984540054
    p, d = successes/trials, 1+z*z/trials
    c = (p+z*z/(2*trials))/d
    half = z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/d
    return [max(0., c-half), min(1., c+half)]


def simulate(rows, config, trials):
    # Coefficients are illustrative design stress levels, not estimated neural noise.
    scenarios = [("null_low_user_noise", 0., .10), ("null_high_user_noise", 0., .65),
                 ("modest_gain_low_noise", .16, .10), ("modest_gain_high_noise", .16, .65),
                 ("large_gain_low_noise", .60, .10), ("large_gain_high_noise", .60, .65)]
    results = []
    for si, (name, residual_logit_gain, user_sd) in enumerate(scenarios):
        rng = np.random.default_rng(2026090501 + si)
        counts = Counter()
        for _ in range(trials):
            arms, cheap = fixture(rows)
            # Same task difficulty across methods; additional correlated user effects
            # are constant across all rotations and all tasks of the given user.
            shared = rng.normal(0, .30, 96)
            for arm in ("pooled_sft", "pooled_sdpo", "mixture", "residual"):
                per_user = rng.normal(0, user_sd, 19)
                perturb = shared + per_user[np.arange(96) % 19] + rng.normal(0, .15, 96)
                logit = math.log(.60/.40) + perturb + (residual_logit_gain if arm == "residual" else 0.)
                p = 1/(1+np.exp(-logit))
                arms[arm] = predictions(rows, p)
            result = method_decision(rows, arms, cheap, config)
            counts[result["decision"]] += 1
        passes = counts["REDUCED_DEV_METHOD_PROMISING"]
        row = {"scenario": name, "trials": trials, "residual_logit_gain": residual_logit_gain,
               "user_logit_sd": user_sd, "task_logit_sd": .30, "method_task_logit_sd": .15,
               "routing_counts": counts, "route_rate": passes/trials,
               "monte_carlo_wilson_95_interval": interval(passes, trials)}
        results.append(row)
        print(json.dumps({"scenario": name, "rate": passes/trials, "counts": counts}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hindsight_pahf_reduced_dev_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    if args.trials < 100:
        raise ValueError("at least100 Monte Carlo studies per scenario")
    if args.out.exists():
        raise FileExistsError("preserve existing decision-rule receipt")
    started = time.monotonic()
    source = ROOT / "src/interaction_sprint/hindsight_pahf_reduced.py"
    bound_paths = (source, args.config, Path(__file__))
    hashes_before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound_paths}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = artificial_development()
    checks = adversarial_checks(rows, config)
    results = simulate(rows, config, args.trials)
    hashes_after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("decision source/config changed during audit; discard this execution and rerun after freeze")
    report = {"status": "JOINT_RULE_MODEL_FREE_STRESS_AUDIT", "checks": checks, "scenarios": results,
              "method": "Every simulated study invokes the complete production method_decision; correlated users, shared task difficulty and repeated rotations are preserved.",
              "limits": "Illustrative distributions, not measured model variability. Monte Carlo intervals concern the artificial routing probability only. No neural power, real false-positive guarantee, confirmatory inference or paper green light follows.",
              "synthetic_bases": 96, "synthetic_users": 19, "rotations_per_base": 4,
              "source_sha256": hashes_before,
              "elapsed_seconds": time.monotonic()-started, "paper_green_light": False, "confirmation_opened": False, "new_model_experiments": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
