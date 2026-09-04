import copy
import json

import numpy as np
import pytest

from interaction_sprint.fixtures import BUILDERS, reduce_ops, settings
from interaction_sprint.theory import (anchor_identify, audit, bayes_sdpo_gradient,
    channel, dr_effect, equivalence, mean_dynamics, stochastic_anchor_audit, anchor_sensitivity_interval)
from interaction_sprint.run import prepare, inputs, seal, select, validate, write
from interaction_sprint.analyze import analyze, verify


def test_exact_equivalence_opposite_initial_policy_ranking():
    result = equivalence()
    assert result["identical_immediate_channel"] == ["7/40", "19/25"]
    assert result["initial_preference_utility_A_action0_action1"][1] > .5
    assert result["initial_preference_utility_B_action0_action1"][1] < .5


@pytest.mark.parametrize("p", [.1, .3, .49, .51, .7, .9])
def test_bayesian_sdpo_counterexample_restores_balance(p):
    assert bayes_sdpo_gradient(p, [.1, .9]) * (p-.5) < 0


def test_zero_influence_channel_no_bayesian_update():
    assert abs(bayes_sdpo_gradient(.7, [.6, .6])) < 1e-14
    assert abs(bayes_sdpo_gradient(.5, [.1, .9])) < 1e-14


def test_gradient_matches_mutual_information_finite_difference():
    def mi(p):
        likelihood = np.array([[.9, .1], [.1, .9]])
        weights = np.array([1-p, p])
        marginal = weights @ likelihood
        return float(np.sum(weights[:, None]*likelihood*np.log(likelihood/marginal)))
    for p in (.2, .4, .8):
        finite = (mi(p+1e-5)-mi(p-1e-5))/2e-5*p*(1-p)
        assert abs(finite-bayes_sdpo_gradient(p, [.1, .9])) < 1e-8


def test_no_polarized_linear_feedback_from_small_imbalance():
    path = np.array(mean_dynamics())
    assert np.max(np.abs(path)) <= .01
    assert abs(path[-1, 0]-path[-1, 1]) < 1e-10


def test_anchor_identification_and_saturation():
    rho, c, q = np.array([.2, .5]), np.array([.3, .4]), .6
    delayed = (1-rho)*q + rho*np.arange(2)
    r_hat, c_hat = anchor_identify(q, channel(q, rho, c), delayed)
    np.testing.assert_allclose(r_hat, rho)
    np.testing.assert_allclose(c_hat, c)
    with pytest.raises(ValueError, match="saturation"):
        anchor_identify(q, [0, 1], [0, 1])


def test_randomization_alone_cannot_separate_expression_transition():
    np.testing.assert_allclose(channel(.6, [0, 0], [.4, .4]), channel(.6, [.4, .4], [0, 0]))


def test_anchor_contamination_can_remove_effect_sign():
    assert anchor_sensitivity_interval(.2, .1) == [0., .4]
    assert anchor_sensitivity_interval(.2, .2)[0] < 0


def test_anchor_aipw_two_robustness_cases_and_failure_control():
    results = stochastic_anchor_audit(n=60000)
    for result in results:
        for field in ("cross_fitted_DR", "correct_propensity_bad_nuisance", "wrong_propensity_correct_nuisance"):
            assert abs(result[field]-result["true_delayed_anchor_ATE"]) < .05
    assert abs(results[0]["both_wrong"]) > .2
    with pytest.raises(ValueError, match="positivity"):
        dr_effect([0], [1], [1], [0], [0], [1], [[.5, .5]])


def test_retraction_is_not_identity_on_preexisting_register():
    operations = [("set", "x", "new"), ("clear", "x", None)]
    assert reduce_ops({}, operations) == {}
    assert reduce_ops({"x": "old"}, operations) != {"x": "old"}


def test_last_write_and_independent_commutation():
    assert reduce_ops({}, [("set", "x", "a"), ("set", "x", "b")]) == {"x": "b"}
    assert reduce_ops({}, [("set", "x", "a"), ("set", "y", "b")]) == reduce_ops({}, [("set", "y", "b"), ("set", "x", "a")])


@pytest.mark.parametrize("study,full,smoke", [("undo", 768, 48), ("endo_signal", 512, 64)])
def test_frozen_cardinality_and_determinism(study, full, smoke):
    cases, key = BUILDERS[study]()
    assert len(cases) == len(key) == full
    assert len(select(cases, "smoke")) == smoke
    assert (cases, key) == BUILDERS[study]()
    assert len({c["case_id"] for c in cases}) == full


def test_undo_same_terminal_state_and_relevant_cf():
    cases, key = BUILDERS["undo"]()
    for case in cases:
        if case["arm"] != "expanded":
            continue
        answer = key[case["case_id"]]
        assert len(answer["expanded_operations"]) == case["depth"]
        assert reduce_ops({}, answer["expanded_operations"]) == answer["state"]
        assert key[case["pair_id"]+"/canonical"]["state"] == answer["state"]
        assert key[case["pair_id"]+"/padded"]["state"] == answer["state"]
        assert key[case["pair_id"]+"/counterfactual"]["answer"] != answer["answer"]


def test_expression_transition_model_inputs_identical():
    cases, key = BUILDERS["endo_signal"]()
    index = {c["case_id"]: c for c in cases}
    for case in cases:
        if case["regime"] != "expression":
            continue
        other = case["case_id"].replace("-expression-", "-transition-")
        assert case["messages"] == index[other]["messages"]
        assert key[case["case_id"]]["initial"] == key[other]["initial"]


def records(cases, key):
    result = []
    for case in cases:
        n = len(case["choices"])
        probability = np.full(n, .02/(n-1))
        answer = case["choices"].index(key[case["case_id"]]["answer"])
        probability[answer] = .98
        result.append({"case_id": case["case_id"], "choice_probability": probability.tolist(),
            "log_prob": np.log(probability*.9).tolist(), "choice_mass": .9,
            "predicted": case["choices"][answer], "input_tokens": 12})
    return result


@pytest.mark.parametrize("study", ["undo", "endo_signal"])
def test_smoke_never_claims_thesis_and_rejects_incomplete(study):
    cases, key = BUILDERS[study]()
    cases = select(cases, "smoke")
    rows = records(cases, key)
    assert analyze(study, cases, key, rows, "smoke")["decision"] == "SMOKE_ONLY"
    with pytest.raises(ValueError, match="incomplete"):
        analyze(study, cases, key, rows[:-1], "smoke")


def test_probabilities_cannot_disagree_with_logged_logps():
    cases, key = BUILDERS["undo"]()
    cases = select(cases, "smoke")
    rows = records(cases, key)
    rows[0]["choice_mass"] = .4
    with pytest.raises(ValueError, match="probabilities disagree"):
        analyze("undo", cases, key, rows, "smoke")


def test_no_residue_does_not_masquerade_as_training_success():
    cases, key = BUILDERS["undo"]()
    report = analyze("undo", cases, key, records(cases, key), "full")
    assert report["decision"] == "NO_REGISTERED_RESIDUE_SIGNAL_PARK_THIS_ASSAY"
    assert report["weight_updates"] == 0


def test_snapshot_verifier_immutable_and_tamper_rejected(tmp_path):
    root = tmp_path / "result"
    root.mkdir()
    prepare("undo", root / "inputs")
    spec, all_cases, key = inputs(root / "inputs")
    cases = select(all_cases, "smoke")
    rows = records(cases, key)
    write(root / "runtime.json", {"model": spec["model"], "revision": spec["revision"], "weight_updates": 0})
    write(root / "plan.json", {"settings": spec, "mode": "smoke", "case_ids": [c["case_id"] for c in cases]})
    write(root / "COMPLETE.json", {"records": len(cases), "weight_updates": 0})
    write(root / "BUDGET_AUDIT.json", [{"case_id": c["case_id"], "input_tokens": 12} for c in cases])
    (root / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    seal(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert verify(root)["decision"] == "SMOKE_ONLY"
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (root / "raw.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify(root)


def test_preparation_cannot_overwrite(tmp_path):
    root = tmp_path / "inputs"
    prepare("endo_signal", root)
    with pytest.raises(FileExistsError):
        prepare("endo_signal", root)
    (root / "private_answer_key.json").write_text("{}")
    with pytest.raises(ValueError):
        inputs(root)
