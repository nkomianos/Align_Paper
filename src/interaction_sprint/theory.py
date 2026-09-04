"""Executable finite-model checks; proofs and assumptions live in the review.

No LLM, preference learner or human data is used by this module. Do not call a
designed stochastic simulator a discovery about deployed interaction learners.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import shutil

import numpy as np


def channel(q, rho, compliance):
    """P(next message=1 | action=a), with transition then expression copying."""
    return np.array([(1-rho[a]) * q + rho[a] * a for a in (0, 1)]) * (1-np.array(compliance)) + np.array(compliance) * np.arange(2)


def equivalence():
    # All numbers rational: equivalence is exact, not Monte-Carlo approximation.
    q_a, q_b = F(7, 10), F(1, 5)
    c = (F(3, 4), F(1, 5))
    r = (F(1, 8), F(7, 10))
    expression = [(1-c[a])*q_a + c[a]*a for a in (0, 1)]
    transition = [(1-r[a])*q_b + r[a]*a for a in (0, 1)]
    assert expression == transition
    return {"identical_immediate_channel": list(map(str, expression)),
            "equivalent_for_any_randomized_one_step_logging_policy": True,
            "initial_preference_utility_A_action0_action1": [float(1-q_a), float(q_a)],
            "initial_preference_utility_B_action0_action1": [float(1-q_b), float(q_b)],
            "limitation": "independent one-step episodes; not equivalence for every repeated-user longitudinal design"}


def bayes_sdpo_gradient(p, message_one):
    """Exact expected stopped-teacher log-ratio policy gradient in logit(p).

Fixed action-to-message channel; no derivative through user dynamics. This is
the mutual-information gradient, not generally a preference-welfare gradient.
"""
    if not 0 < p < 1:
        raise ValueError("interior policy required")
    likelihood = np.stack((1-np.array(message_one), message_one), axis=1)
    marginal = np.array([1-p, p]) @ likelihood
    advantages = np.array([sum(v * np.log(v / marginal[o]) for o, v in enumerate(row) if v > 0)
                           for row in likelihood])
    return float(p * (1-p) * (advantages[1]-advantages[0]))


def dr_effect(action, outcome, measured, x, propensity, sampling, nuisance):
    """Two-arm AIPW with independently sampled anchor measurements.

Requires exchangeability/positivity and MAR anchor acquisition conditional on X.
The nuisance [n,2] array must be cross-fitted or independently trained.
Double robustness includes the product of action and measurement propensities.
"""
    a, y, s = map(np.asarray, (action, outcome, measured))
    e, rate, mu = map(np.asarray, (propensity, sampling, nuisance))
    if not ((e > 0).all() and (e < 1).all() and (rate > 0).all() and (rate <= 1).all()):
        raise ValueError("positivity failure")
    if mu.shape != (len(a), 2) or not (len(a) == len(y) == len(s) == len(x) == len(e) == len(rate)):
        raise ValueError("shape mismatch")
    if not np.isfinite(y[s.astype(bool)]).all():
        raise ValueError("observed anchors cannot be missing")
    safe_y = np.where(s, y, 0)
    residual = a/e*(safe_y-mu[:, 1]) - (1-a)/(1-e)*(safe_y-mu[:, 0])
    return float(np.mean(mu[:, 1]-mu[:, 0] + s/rate*residual))


def anchor_identify(q, immediate, delayed):
    """Perfect neutral delayed anchors; known pre-treatment preference mean.

No claim that randomizing responses alone identifies expression vs transition.
"""
    if not 0 < q < 1:
        raise ValueError("preference support required")
    rho = np.array([(q-delayed[0])/q, (delayed[1]-q)/(1-q)])
    denominator = np.arange(2)-np.asarray(delayed)
    if (np.abs(denominator) < 1e-10).any():
        raise ValueError("expression saturation: not identifiable")
    comp = (np.asarray(immediate)-delayed)/denominator
    return rho, comp


def mean_dynamics(m=0., b=.01, rho=.4, eta=.3, steps=100):
    rows = []
    for _ in range(steps):
        m = (1-rho)*m + rho*b
        b = (1-eta)*b + eta*m
        rows.append([m, b])
    return rows


def anchor_sensitivity_interval(observed_ate, max_mismeasurement):
    """Conservative, not a claimed novel sharp partial-identification bound."""
    if not 0 <= max_mismeasurement <= 1:
        raise ValueError("measurement-error probability must be in [0,1]")
    return [max(-1., observed_ate-2*max_mismeasurement),
            min(1., observed_ate+2*max_mismeasurement)]


def stochastic_anchor_audit(seed=90326, n=12000):
    """Independent simulated users, known logging/measurement propensities.

Cross-fitting is by person index. No learned language policy and no human data.
Outcome is the delayed neutral binary anchor, not uniquely privileged welfare.
"""
    rng = np.random.default_rng(seed)
    results = []
    for name, rho, compliance in (("static", 0., 0.), ("expression", 0., .7),
                                  ("transition", .5, 0.), ("mixed", .5, .7)):
        x = rng.integers(0, 2, n)
        q = .25 + .5*x
        e = .2 + .6*x
        a = (rng.random(n) < e).astype(int)
        pre = (rng.random(n) < q).astype(int)
        post = np.where(rng.random(n) < rho, a, pre)
        message = np.where(rng.random(n) < compliance, a, post)
        measured = rng.random(n) < .15
        y = np.where(measured, post.astype(float), np.nan)
        mu = np.empty((n, 2))
        fold = np.arange(n) % 2
        for held in (0, 1):
            for context in (0, 1):
                for action in (0, 1):
                    train = (fold != held) & (x == context) & (a == action) & measured
                    if not train.any():
                        raise ValueError("missing measured nuisance cell")
                    mu[(fold == held) & (x == context), action] = y[train].mean()
        sampling = np.full(n, .15)
        oracle_mu = np.stack(((1-rho)*q, (1-rho)*q+rho), axis=1)
        results.append({"regime": name, "n": n, "anchors": int(measured.sum()),
            "true_delayed_anchor_ATE": rho,
            "true_immediate_expression_ATE": rho + compliance - rho*compliance,
            "cross_fitted_DR": dr_effect(a, y, measured, x, e, sampling, mu),
            "correct_propensity_bad_nuisance": dr_effect(a, y, measured, x, e, sampling, np.full((n, 2), .5)),
            "wrong_propensity_correct_nuisance": dr_effect(a, y, measured, x, np.full(n, .5), sampling, oracle_mu),
            "both_wrong": dr_effect(a, y, measured, x, np.full(n, .5), sampling, np.full((n, 2), .5)),
            "observed_agreement": float(np.mean(message == a)),
            "realized_transition_rate": float(np.mean(post != pre))})
    return results


def audit():
    q, rho, compliance = .6, np.array([.2, .5]), np.array([.3, .4])
    delayed = np.array([(1-rho[a])*q+rho[a]*a for a in (0, 1)])
    identified = anchor_identify(q, channel(q, rho, compliance), delayed)
    matrix = np.array([[.6, .4], [.3*.6, .7+.3*.4]])
    gradients = {str(p): bayes_sdpo_gradient(p, [.1, .9]) for p in (.1, .49, .5, .51, .9)}
    return {"scope": "analytic_and_numeric_audit_NOT_LM_experiment",
            "equivalence": equivalence(), "symmetric_bayes_sdpo_logit_gradients": gradients,
            "polarization_claim_in_this_model": "FALSE_balanced_policy_is_attracting",
            "linear_feedback_eigenvalues": sorted(np.linalg.eigvals(matrix).tolist()),
            "linear_feedback_final_state": mean_dynamics()[-1],
            "initial_agreement_balanced_users_unpersonalized_policy": .5,
            "anchor_recovered_rho": identified[0].tolist(),
            "anchor_recovered_compliance": identified[1].tolist(),
            "stochastic_anchor_checks": stochastic_anchor_audit(),
            "anchor_contamination_illustration": {str(d): anchor_sensitivity_interval(.2, d) for d in (0, .05, .1, .2)},
            "transport_counterexample": {"target_map": "x -> x", "learned_map": "x -> -x by 180-degree rotation in R^2 for isotropic Gaussian",
                "round_trip_error": 0, "endpoint_distribution_discrepancy": 0,
                "mean_squared_paired_endpoint_error_in_2D": 8,
                "conclusion": "distribution correctness plus cycle consistency cannot certify a designated coupling"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = audit()
    with (args.output / "THEORY_AUDIT.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    shutil.copy2(__file__, args.output / "theory_source.py")
    from .run import seal
    seal(args.output)
    print(json.dumps(report, indent=2))
