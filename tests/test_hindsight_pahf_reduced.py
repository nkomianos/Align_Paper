"""Prospective semantic, identity and joint-rule tests; no model loads/forwards."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from interaction_sprint.hindsight_pahf_reduced import (
    aggregate_predictions, binding, build_interface_jobs, build_schedules,
    cheap_readouts, method_decision, qualify_acquisition, qualify_interface,
    validate_learning_dev,
)


@pytest.fixture
def cfg():
    return json.loads((Path(__file__).parents[1] / "configs/hindsight_pahf_reduced_dev_v1.json").read_text())


def records(n, partition="evaluation"):
    result = []
    for base in range(n):
        original = [f"item{base} red quiet", f"item{base} blue loud", f"item{base} green smooth", "Do not buy"]
        for rotation in range(4):
            old, new = "ABCD"[(-rotation) % 4], "ABCD"[(1 - rotation) % 4]
            options = [original[(i + rotation) % 4] for i in range(4)]
            prompt = f"User{base % 2}: choose item{base}\n" + "\n".join(f"{letter}) {text}" for letter, text in zip("ABCD", options)) + "\nReply with exactly A, B, C, or D."
            result.append({"id": f"{partition}-{base}-rotation-{rotation}", "base_id": f"{partition}-{base}",
                "partition": partition, "label_rotation": rotation, "old_target": old, "new_target": new,
                "prompt": prompt, "immediate_followup": f"Option {new}: {original[1]}",
                "delayed_transition_followup": f"Option {new}: {original[1]}",
                "delayed_expression_followup": f"Option {old}: {original[0]}"})
    return result


def prediction(row, old=.4, new=.2, mass=.8):
    probs = [0.] * 4
    probs["ABCD".index(row["old_target"])] = old
    probs["ABCD".index(row["new_target"])] = new
    for i in range(4):
        if not probs[i]:
            probs[i] = (1 - old - new) / 2
    return {**binding(row), "full_vocab_choice_log_probabilities": [math.log(mass * p) for p in probs],
            "normalized_choice_log_probabilities": [math.log(p) for p in probs], "full_vocabulary_choice_mass": mass}


def arms(dev):
    params = {"baseline": (.3, .29), "raw_immediate": (.2, .6), "oracle_delayed": (.7, .1),
              "pooled_sft": (.35, .2), "pooled_sdpo": (.4, .2), "residual": (.55, .2), "mixture": (.38, .2)}
    return {name: [prediction(row, *pair) for row in dev] for name, pair in params.items()}


def wrong_readouts(dev):
    rows = []
    for row in dev:
        p = [float(letter == row["new_target"]) for letter in "ABCD"]
        rows.append({**binding(row), "full_vocab_choice_probabilities": p,
                     "normalized_choice_probabilities": p, "full_vocabulary_choice_mass": 1.0})
    return {"anchor_memory": rows, "anchor_profile": deepcopy(rows)}


def test_schedule_covers_unique_learning_and_pooled_anchors(cfg):
    learning = records(630, "learning")
    result = build_schedules(learning, cfg)
    ids = {r["id"]: r for r in learning}
    assert len(result["population"]) == len(result["anchors"]) == 35
    assert len({ids[id]["base_id"] for b in result["population"] for id in b}) == 630
    assert len({id for b in result["population"] for id in b}) == 630
    assert len({ids[id]["base_id"] for b in result["anchors"][:32] for id in b}) == 64
    assert len({id for b in result["anchors"][:32] for id in b}) == 256
    assert sum(map(len, result["anchors"])) == 280
    assert result == build_schedules(list(reversed(learning)), cfg)


def test_schedule_and_preflight_do_not_use_nonanchor_delayed_labels(cfg):
    learning = records(630, "learning")
    schedule = build_schedules(learning, cfg)
    poisoned = deepcopy(learning)
    for row in poisoned:
        if row["base_id"] not in schedule["pooled_base_ids"]:
            del row["old_target"]
            del row["delayed_expression_followup"]
    assert build_schedules(poisoned, cfg) == schedule
    jobs = build_interface_jobs(poisoned, cfg)
    assert len(jobs) == 128
    assert len({j["base_id"] for j in jobs}) == 16
    assert {j["base_id"] for j in jobs} <= set(schedule["pooled_base_ids"])


def test_full_joint_rule_pass_is_developmental_only(cfg):
    dev = records(96)
    result = method_decision(dev, arms(dev), wrong_readouts(dev), cfg)
    assert result["qualified"]
    assert result["classification"] == "Developmental/apparatus only"
    assert result["paper_green_light"] is False
    assert "no_update" in result["readout_gates"]


def test_uniform_no_update_is_valid_stochastic_baseline(cfg):
    dev = records(96)
    pred = arms(dev)
    pred["baseline"] = [prediction(r, .25, .25, .8) for r in dev]
    acquisition = qualify_acquisition(dev, pred, cfg)
    assert acquisition["qualified"]
    assert not acquisition["diagnostics"]["baseline"]["argmax_disagreement_required"]
    assert acquisition["diagnostics"]["baseline"]["values"]["semantic_disagreement_fraction"] == 1
    result = method_decision(dev, pred, wrong_readouts(dev), cfg)
    assert result["qualified"]
    assert not result["diagnostics"]["baseline"]["argmax_disagreement_required"]


def test_format_only_acquisition_cannot_pass(cfg):
    dev = records(96)
    pred = arms(dev)
    pred["baseline"] = [prediction(r, .3, .29, .5) for r in dev]
    pred["raw_immediate"] = [prediction(r, .3, .29, .9) for r in dev]
    assert not qualify_acquisition(dev, pred, cfg)["qualified"]


def test_format_only_method_improvement_cannot_pass(cfg):
    dev = records(96)
    pred = arms(dev)
    pred["residual"] = [prediction(r, .4, .2, .99) for r in dev]
    result = method_decision(dev, pred, wrong_readouts(dev), cfg)
    assert result["assay_qualified"]
    assert not result["qualified"]
    assert result["classification"] == "Valid negative"


@pytest.mark.parametrize("trap", ["rotation", "user"])
def test_positive_aggregate_cannot_hide_bad_probability_stratum(cfg, trap):
    dev = records(96)
    pred = arms(dev)
    if trap == "rotation":
        pred["residual"] = [prediction(r, .39 if r["label_rotation"] == 0 else .55, .2) for r in dev]
    else:
        pred["residual"] = [prediction(r, .35 if binding(r)["source_user"] == "User0" else .75, .1) for r in dev]
    result = method_decision(dev, pred, wrong_readouts(dev), cfg)
    assert result["assay_qualified"]
    assert result["method_gates"]["pooled_sdpo"]["aggregate_floors"]
    assert not result["method_gates"]["pooled_sdpo"]["rotation_and_user_nonnegative"]
    assert not result["qualified"]


def test_cheap_readout_matching_residual_stops_neural_claim(cfg):
    dev = records(96)
    pred = arms(dev)
    readouts = wrong_readouts(dev)
    readouts["anchor_memory"] = deepcopy(pred["residual"])
    result = method_decision(dev, pred, readouts, cfg)
    assert result["assay_qualified"]
    assert not result["readout_gates"]["anchor_memory"]
    assert not result["qualified"]


def test_exact_half_point_readout_boundary_does_not_pass_from_roundoff(cfg):
    dev = records(96)
    pred = arms(dev)
    readouts = wrong_readouts(dev)
    readouts["anchor_memory"] = [prediction(row, .545, .2, .435 / .545) for row in dev]
    result = method_decision(dev, pred, readouts, cfg)
    gains = result["readout_comparisons"]["anchor_memory"]["aggregate"]
    assert gains["full_probability_gain"] == pytest.approx(.005)
    assert gains["conditional_probability_gain"] == pytest.approx(.005)
    assert not result["readout_gates"]["anchor_memory"]


def test_per_base_position_failure_cannot_cancel_in_global_means(cfg):
    dev = records(96)
    pred = []
    for row in dev:
        p = [.85, .05, .05, .05]
        pred.append({**binding(row), "full_vocab_choice_log_probabilities": [math.log(.8 * v) for v in p],
                     "normalized_choice_log_probabilities": [math.log(v) for v in p], "full_vocabulary_choice_mass": .8})
    summary = aggregate_predictions(dev, pred)
    assert summary["position_diagnostics"]["semantic_disagreement_fraction"] == 1
    assert summary["position_diagnostics"]["mean_paired_position_range"] > .79


def test_identity_and_user_binding_reject_malformed_grids(cfg):
    dev = records(96)
    pred = arms(dev)["baseline"]
    for field, value in [("base_id", "other"), ("label_rotation", 3), ("source_user", "Other")]:
        bad = deepcopy(pred)
        bad[0][field] = value
        with pytest.raises(ValueError):
            aggregate_predictions(dev, bad)
    with pytest.raises(ValueError):
        aggregate_predictions(dev, pred[:-1] + [pred[0]])
    one_user = records(96)
    for row in one_user:
        row["prompt"] = row["prompt"].replace("User1:", "User0:")
    with pytest.raises(ValueError, match="at least two"):
        aggregate_predictions(one_user, [prediction(r) for r in one_user])


def test_transition_identity_and_split_overlap_are_prelaunch_checks(cfg):
    learning, dev = records(630, "learning"), records(96)
    validate_learning_dev(learning, dev, cfg)
    learning[0]["delayed_transition_followup"] = "different"
    with pytest.raises(ValueError, match="identity"):
        validate_learning_dev(learning, dev, cfg)


def test_exact_interface_complete_grid_and_fixed_thresholds(cfg):
    jobs = build_interface_jobs(records(630, "learning"), cfg)
    preds = []
    for job in jobs:
        p = [.8 if letter == job["target"] else .2 / 3 for letter in "ABCD"]
        preds.append({**job, "full_vocab_choice_log_probabilities": [math.log(.8 * x) for x in p],
                      "normalized_choice_log_probabilities": [math.log(x) for x in p], "full_vocabulary_choice_mass": .8})
    assert qualify_interface(jobs, preds, cfg)["qualified"]
    with pytest.raises(ValueError, match="grid"):
        qualify_interface(jobs, preds[:-1] + [preds[0]], cfg)


def test_readouts_are_one_hot_rotation_equivariant_and_anchor_limited(cfg):
    learning, dev = records(630, "learning"), records(96)
    pooled = build_schedules(learning, cfg)["pooled_base_ids"]
    for row in learning:
        if row["base_id"] not in pooled:
            del row["old_target"]
            del row["delayed_expression_followup"]
    result = cheap_readouts(learning, dev, pooled, cfg)
    for rows in result.values():
        assert all(sum(p["full_vocab_choice_probabilities"]) == 1 for p in rows)
        assert all(set(p["full_vocab_choice_probabilities"]) <= {0., 1.} for p in rows)
        assert aggregate_predictions(dev, rows)["position_diagnostics"]["semantic_disagreement_fraction"] == 0
