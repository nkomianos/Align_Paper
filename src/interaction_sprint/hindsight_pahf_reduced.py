"""Prospective reduced EndoPAHF DEV schedules, estimands and complete route rule.

Pure CPU arithmetic; no model loading, filesystem data loading, or confirmation
support. Labels describe constructed delayed targets, not measured human change.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import random
import re
from typing import Mapping, Sequence

LETTERS = "ABCD"
TRAINED_ARMS = ("raw_immediate", "oracle_delayed", "pooled_sft", "pooled_sdpo", "residual", "mixture")
COMPARATORS = ("raw_immediate", "pooled_sft", "pooled_sdpo", "mixture")
HINDSIGHT_BLOCK = "\n\n=== HINDSIGHT CONTEXT ===\n[The following is a future user message. Use this to guide your answer to the user prompt.]\n{follow_up}"


def rank(value: str, salt: str) -> str:
    return hashlib.sha256(f"{value}|{salt}".encode()).hexdigest()


def source_user(row: Mapping) -> str:
    first = str(row["prompt"]).splitlines()[0]
    if ":" not in first or not first.split(":", 1)[0].strip():
        raise ValueError("prompt lacks explicit display-name prefix")
    return first.split(":", 1)[0].strip()


def binding(row: Mapping) -> dict:
    return {"id": str(row["id"]), "base_id": str(row["base_id"]),
            "label_rotation": int(row["label_rotation"]), "source_user": source_user(row)}


def group_bases(rows: Sequence[Mapping], expected: int | None = None, *, validate_targets: bool = True) -> dict:
    grouped = defaultdict(list)
    if len({str(r["id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate row id")
    for row in rows:
        b = binding(row)
        if not b["id"] or not b["base_id"]:
            raise ValueError("empty identity")
        grouped[b["base_id"]].append(row)
    if not grouped or (expected is not None and len(grouped) != expected):
        raise ValueError("wrong base count")
    for base, variants in grouped.items():
        if len(variants) != 4 or {int(r["label_rotation"]) for r in variants} != {0, 1, 2, 3}:
            raise ValueError(f"incomplete rotations: {base}")
        if len({source_user(r) for r in variants}) != 1:
            raise ValueError("source-user binding differs across rotations")
        for target in (("old_target", "new_target") if validate_targets else ()):
            if {r[target] for r in variants} != set(LETTERS):
                raise ValueError("target positions not counterbalanced")
            if len({(LETTERS.index(r[target]) + int(r["label_rotation"])) % 4 for r in variants}) != 1:
                raise ValueError("target semantic identity changes under rotation")
        if validate_targets and any(r["old_target"] == r["new_target"] for r in variants):
            raise ValueError("unchanged expression target")
    return dict(grouped)


def validate_learning_dev(learning: Sequence[Mapping], development: Sequence[Mapping], config: Mapping) -> None:
    a = group_bases(learning, config["learning_bases"])
    b = group_bases(development, config["dev_bases"])
    if set(a) & set(b) or {r["id"] for r in learning} & {r["id"] for r in development}:
        raise ValueError("learning/DEV identity overlap")
    for rows in (learning, development):
        for row in rows:
            if row["immediate_followup"] != row["delayed_transition_followup"]:
                raise ValueError("transition residual identity fails")
            if row["immediate_followup"] == row["delayed_expression_followup"]:
                raise ValueError("expression feedback is not distinct")


def build_schedules(learning: Sequence[Mapping], config: Mapping) -> dict:
    grouped = group_bases(learning, config["learning_bases"], validate_targets=False)
    panel_order = sorted(grouped, key=lambda b: rank(b, config["anchor_panel_salt"]))[:64]
    panels = [panel_order[i:i + 16] for i in range(0, 64, 16)]
    if len(panel_order) != 64 or len(set(panel_order)) != 64:
        raise ValueError("requires union of four disjoint 16-base panels")
    global_bases = sorted(grouped, key=lambda b: rank(b, config["population_salt"]))
    population = [next(str(r["id"]) for r in grouped[b] if int(r["label_rotation"]) == i % 4)
                  for i, b in enumerate(global_bases)]
    random.Random(config["seed"]).shuffle(population)
    order = sorted(panel_order, key=lambda b: rank(b, config["pooled_order_salt"]))
    revisits = sorted(panel_order, key=lambda b: rank(b, config["pooled_revisit_salt"]))[:6]
    presentations = order + revisits
    anchors = [[str(r["id"]) for b in presentations[i:i + 2]
                for r in sorted(grouped[b], key=lambda r: int(r["label_rotation"]))]
               for i in range(0, len(presentations), 2)]
    batches = [population[i:i + config["population_batch"]]
               for i in range(0, len(population), config["population_batch"])]
    if len(batches) != config["steps"] or any(len(b) != 18 for b in batches):
        raise ValueError("population schedule must be 35 by 18")
    if len(anchors) != config["steps"] or any(len(b) != 8 for b in anchors):
        raise ValueError("anchor schedule must be 35 by 8")
    return {"population": batches, "anchors": anchors, "original_panels": panels,
            "pooled_base_ids": panel_order, "anchor_base_order": presentations,
            "population_unique_bases": len(grouped), "anchor_unique_bases": 64,
            "population_presentations": len(population), "anchor_presentations": 280}


def build_interface_jobs(learning: Sequence[Mapping], config: Mapping, pooled_base_ids: Sequence[str] | None = None) -> list[dict]:
    grouped = group_bases(learning, config["learning_bases"], validate_targets=False)
    pooled = list(pooled_base_ids) if pooled_base_ids is not None else build_schedules(learning, config)["pooled_base_ids"]
    if len(pooled) != 64 or len(set(pooled)) != 64 or not set(pooled) <= set(grouped):
        raise ValueError("preflight must use shared learning-anchor population")
    bases = sorted(pooled, key=lambda b: rank(b, config["preflight_salt"]))[:config["preflight_bases"]]
    jobs = []
    for base in bases:
        for row in sorted(grouped[base], key=lambda r: r["label_rotation"]):
            for context, target, field in (("immediate", "new_target", "immediate_followup"),
                                            ("delayed_expression", "old_target", "delayed_expression_followup")):
                jobs.append({**binding(row), "job_id": str(row["id"]) + "::" + context,
                             "context": context, "target": row[target],
                             "text": str(row["prompt"]) + HINDSIGHT_BLOCK.format(follow_up=str(row[field]).strip())})
    return jobs


def _distribution(prediction: Mapping) -> tuple[list[float], list[float], float]:
    if "full_vocab_choice_log_probabilities" in prediction:
        logs = [float(x) for x in prediction["full_vocab_choice_log_probabilities"]]
        if len(logs) != 4 or any(not math.isfinite(x) or x > 1e-6 for x in logs):
            raise ValueError("invalid full choice logs")
        full = [math.exp(x) for x in logs]
    else:
        full = [float(x) for x in prediction["full_vocab_choice_probabilities"]]
    if len(full) != 4 or any(not math.isfinite(x) or not 0 <= x <= 1 + 1e-6 for x in full):
        raise ValueError("invalid full choice probabilities")
    mass = sum(full)
    if not 0 < mass <= 1 + 1e-5:
        raise ValueError("invalid answer mass")
    conditional = [x / mass for x in full]
    if abs(float(prediction["full_vocabulary_choice_mass"]) - mass) > 1e-5:
        raise ValueError("saved answer mass differs from full distribution")
    if "normalized_choice_log_probabilities" in prediction:
        supplied = [math.exp(float(x)) for x in prediction["normalized_choice_log_probabilities"]]
    else:
        supplied = [float(x) for x in prediction["normalized_choice_probabilities"]]
    if len(supplied) != 4 or any(not math.isfinite(x) for x in supplied) or max(abs(a - b) for a, b in zip(supplied, conditional)) > 1e-5:
        raise ValueError("conditional probability mismatch")
    return full, conditional, mass


def score_prediction_rows(source_rows: Sequence[Mapping], predictions: Sequence[Mapping]) -> list[dict]:
    group_bases(source_rows)
    source = {str(r["id"]): r for r in source_rows}
    lookup = {str(r["id"]): r for r in predictions}
    if len(lookup) != len(predictions) or set(lookup) != set(source):
        raise ValueError("prediction identity grid differs")
    scored = []
    for id in sorted(source):
        row, pred = source[id], lookup[id]
        expected = binding(row)
        if any(pred.get(k) != v for k, v in expected.items()):
            raise ValueError("prediction source binding differs")
        full, cond, mass = _distribution(pred)
        choice = max(range(4), key=cond.__getitem__)
        item = {**expected, "choice_mass": mass, "old_target": row["old_target"],
                "semantic_argmax": (choice + expected["label_rotation"]) % 4}
        for target in ("old", "new"):
            i = LETTERS.index(row[target + "_target"])
            item[target] = {"full_nll": -math.log(full[i]) if full[i] else None,
                            "full_probability": full[i], "conditional_probability": cond[i],
                            "conditional_nll": -math.log(cond[i]) if cond[i] else None,
                            "accuracy": float(choice == i)}
        scored.append(item)
    return scored


def _mean_metrics(rows: Sequence[Mapping]) -> dict:
    result = {"n": len(rows)}
    for target in ("old", "new"):
        result[target] = {}
        for metric in ("full_nll", "full_probability", "conditional_nll", "conditional_probability", "accuracy"):
            values = [r[target][metric] for r in rows]
            result[target][metric] = None if any(v is None for v in values) else sum(values) / len(values)
        result[target]["infinite_nll_count"] = sum(r[target]["full_nll"] is None for r in rows)
    return result


def aggregate_predictions(source_rows: Sequence[Mapping], predictions: Sequence[Mapping]) -> dict:
    rows = score_prediction_rows(source_rows, predictions)
    bases = defaultdict(list)
    for row in rows:
        bases[row["base_id"]].append(row)
    by_base = {base: {**_mean_metrics(values), "source_user": values[0]["source_user"]} for base, values in bases.items()}
    users = sorted({r["source_user"] for r in rows})
    if len(users) < 2:
        raise ValueError("leave-one-user-out rule requires at least two source display names")
    position = {target: [max(r[target]["conditional_probability"] for r in group) - min(r[target]["conditional_probability"] for r in group)
                         for group in bases.values()] for target in ("old", "new")}
    return {"aggregate": _mean_metrics(list(by_base.values())), "by_base": by_base,
            "by_rotation": {str(rot): _mean_metrics([r for r in rows if r["label_rotation"] == rot]) for rot in range(4)},
            "leave_one_user_out": {user: _mean_metrics([r for r in by_base.values() if r["source_user"] != user]) for user in users if len(users) > 1},
            "source_users": users,
            "position_diagnostics": {
                "mean_paired_position_range": max(sum(values) / len(values) for values in position.values()),
                "max_paired_position_range": max(max(values) for values in position.values()),
                "semantic_disagreement_fraction": sum(len({r["semantic_argmax"] for r in group}) > 1 for group in bases.values()) / len(bases),
                "min_label_mean_choice_mass": min(sum(r["choice_mass"] for r in rows if r["old_target"] == letter) / sum(r["old_target"] == letter for r in rows) for letter in LETTERS)}}


def diagnostics_qualified(summary: Mapping, config: Mapping, *, arm_name: str | None = None) -> dict:
    values, t = summary["position_diagnostics"], config["diagnostics"]
    disagreement_required = arm_name != "baseline" or config["baseline_argmax_disagreement_required"]
    checks = {"answer_mass": values["min_label_mean_choice_mass"] >= t["min_label_mean_choice_mass"],
              "paired_mean": values["mean_paired_position_range"] <= t["max_mean_paired_position_range"],
              "paired_max": values["max_paired_position_range"] <= t["max_paired_position_range"],
              "paired_disagreement": not disagreement_required or values["semantic_disagreement_fraction"] <= t["max_semantic_disagreement_fraction"]}
    return {"qualified": all(checks.values()), "checks": checks, "values": dict(values),
            "argmax_disagreement_required": disagreement_required}


def qualify_interface(jobs: Sequence[Mapping], predictions: Sequence[Mapping], config: Mapping) -> dict:
    expected = {j["job_id"]: j for j in jobs}
    observed = {p["job_id"]: p for p in predictions}
    if len(expected) != 128 or len(observed) != len(predictions) or set(observed) != set(expected):
        raise ValueError("interface job grid differs")
    cells = defaultdict(list)
    paired = defaultdict(list)
    paired_choices = defaultdict(set)
    for id, job in expected.items():
        pred = observed[id]
        if any(pred.get(k) != job[k] for k in ("id", "base_id", "label_rotation", "source_user", "context", "target")):
            raise ValueError("interface binding differs")
        _, conditional, mass = _distribution(pred)
        i = LETTERS.index(job["target"])
        cells[job["context"] + "/" + job["target"]].append((max(range(4), key=conditional.__getitem__) == i, conditional[i], mass))
        paired[(job["context"], job["base_id"])].append(conditional[i])
        paired_choices[(job["context"], job["base_id"])].add((max(range(4), key=conditional.__getitem__) + job["label_rotation"]) % 4)
    summary = {key: {"n": len(rows), "correct": sum(r[0] for r in rows),
                     "mean_conditional_probability": sum(r[1] for r in rows) / len(rows),
                     "mean_choice_mass": sum(r[2] for r in rows) / len(rows)} for key, rows in cells.items()}
    gates = {key: r["n"] == 16 and r["correct"] >= config["preflight_min_cell_correct"]
             and r["mean_conditional_probability"] >= config["preflight_min_conditional_probability"]
             and r["mean_choice_mass"] >= config["preflight_min_choice_mass"] for key, r in summary.items()}
    ranges = [max(v) - min(v) for v in paired.values()]
    gates["paired_mean_range"] = sum(ranges) / len(ranges) <= config["diagnostics"]["max_mean_paired_position_range"]
    gates["paired_max_range"] = max(ranges) <= config["diagnostics"]["max_paired_position_range"]
    gates["paired_disagreement"] = sum(len(v) > 1 for v in paired_choices.values()) / len(paired_choices) <= config["diagnostics"]["max_semantic_disagreement_fraction"]
    good = len(cells) == 8 and all(gates.values())
    return {"decision": "REDUCED_INTERFACE_QUALIFIED" if good else "REDUCED_INVALID_INTERFACE", "qualified": good,
            "classification": "Developmental/apparatus only" if good else "Invalid assay/capability",
            "cells": summary, "gates": gates, "paired_target_probability_ranges": ranges, "paper_green_light": False}


def comparison(candidate: Mapping, baseline: Mapping, target: str = "old") -> dict:
    if set(candidate["by_base"]) != set(baseline["by_base"]):
        raise ValueError("comparison base grid differs")
    def gains(a, b):
        aa, bb = a[target], b[target]
        return {"full_nll_gain": None if aa["full_nll"] is None or bb["full_nll"] is None else bb["full_nll"] - aa["full_nll"],
                "full_probability_gain": aa["full_probability"] - bb["full_probability"],
                "conditional_probability_gain": aa["conditional_probability"] - bb["conditional_probability"]}
    return {"aggregate": gains(candidate["aggregate"], baseline["aggregate"]),
            "by_rotation": {k: gains(v, baseline["by_rotation"][k]) for k, v in candidate["by_rotation"].items()},
            "leave_one_user_out": {k: gains(v, baseline["leave_one_user_out"][k]) for k, v in candidate["leave_one_user_out"].items()}}


def qualify_acquisition(development: Sequence[Mapping], arm_predictions: Mapping, config: Mapping) -> dict:
    needed = ("baseline", "raw_immediate", "oracle_delayed")
    if not set(needed) <= set(arm_predictions):
        raise ValueError("missing acquisition controls")
    summaries = {name: aggregate_predictions(development, arm_predictions[name]) for name in needed}
    comparisons = {"oracle_old_vs_baseline": comparison(summaries["oracle_delayed"], summaries["baseline"]),
                   "oracle_old_vs_raw": comparison(summaries["oracle_delayed"], summaries["raw_immediate"]),
                   "raw_new_vs_baseline": comparison(summaries["raw_immediate"], summaries["baseline"], "new")}
    t = config["acquisition"]
    checks = {name: r["aggregate"]["full_nll_gain"] is not None
              and r["aggregate"]["full_nll_gain"] >= t["min_full_nll_gain"]
              and r["aggregate"]["full_probability_gain"] > t["min_full_probability_gain_exclusive"] + config["strict_comparison_tolerance"]
              and r["aggregate"]["conditional_probability_gain"] > t["min_conditional_probability_gain_exclusive"] + config["strict_comparison_tolerance"] for name, r in comparisons.items()}
    diagnostics = {name: diagnostics_qualified(summary, config, arm_name=name) for name, summary in summaries.items()}
    good = all(checks.values()) and all(d["qualified"] for d in diagnostics.values())
    return {"decision": "REDUCED_ACQUISITION_QUALIFIED" if good else "REDUCED_INVALID_ACQUISITION", "qualified": good,
            "classification": "Developmental/apparatus only" if good else "Invalid assay/capability",
            "checks": checks, "comparisons": comparisons, "diagnostics": diagnostics, "paper_green_light": False}


def method_decision(development: Sequence[Mapping], arm_predictions: Mapping, cheap_controls: Mapping, config: Mapping) -> dict:
    if set(arm_predictions) != {"baseline", *TRAINED_ARMS}:
        raise ValueError("exact seven-arm evaluation grid required")
    if set(cheap_controls) != set(config["cheap_controls"]["names"]):
        raise ValueError("cheap readout grid differs")
    acquisition = qualify_acquisition(development, arm_predictions, config)
    summaries = {name: aggregate_predictions(development, preds) for name, preds in arm_predictions.items()}
    diagnostics = {name: diagnostics_qualified(s, config, arm_name=name) for name, s in summaries.items()}
    comparisons = {name: comparison(summaries["residual"], summaries[name]) for name in COMPARATORS}
    t = config["method"]
    gates = {}
    for name, comp in comparisons.items():
        a = comp["aggregate"]
        floor = a["full_nll_gain"] is not None and a["full_nll_gain"] >= t["min_full_nll_gain"] and a["full_probability_gain"] >= t["min_full_probability_gain"] and a["conditional_probability_gain"] >= t["min_conditional_probability_gain"]
        strata = list(comp["by_rotation"].values()) + list(comp["leave_one_user_out"].values())
        stable = all(all(s[k] >= t["min_stratum_gain"] for k in ("full_probability_gain", "conditional_probability_gain")) for s in strata)
        gates[name] = {"aggregate_floors": floor, "rotation_and_user_nonnegative": stable}
    readouts = {name: comparison(summaries["residual"], aggregate_predictions(development, preds)) for name, preds in cheap_controls.items()}
    readouts["no_update"] = comparison(summaries["residual"], summaries["baseline"])
    readout_gates = {name: all(c["aggregate"][k] > t["cheap_probability_margin_exclusive"] + config["strict_comparison_tolerance"] for k in ("full_probability_gain", "conditional_probability_gain")) for name, c in readouts.items()}
    valid = acquisition["qualified"] and all(d["qualified"] for d in diagnostics.values())
    good = valid and all(all(g.values()) for g in gates.values()) and all(readout_gates.values())
    return {"decision": "REDUCED_DEV_METHOD_PROMISING" if good else "REDUCED_INVALID_ASSAY" if not valid else "REDUCED_DEV_METHOD_NEGATIVE",
            "classification": "Developmental/apparatus only" if good else "Invalid assay/capability" if not valid else "Valid negative",
            "qualified": good, "assay_qualified": valid, "acquisition": acquisition, "diagnostics": diagnostics,
            "comparisons": comparisons, "method_gates": gates, "readout_comparisons": readouts, "readout_gates": readout_gates,
            "summaries": summaries, "paper_green_light": False, "confirmation_opened": False}


def _options(row: Mapping) -> dict[str, str]:
    options = {}
    for letter in LETTERS:
        matches = [line[3:] for line in str(row["prompt"]).splitlines() if line.startswith(letter + ") ")]
        if len(matches) != 1:
            raise ValueError("ambiguous option block")
        options[letter] = matches[0]
    if len(set(options.values())) != 4:
        raise ValueError("duplicate semantic option text")
    return options


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def cheap_readouts(learning: Sequence[Mapping], development: Sequence[Mapping], pooled_base_ids: Sequence[str], config: Mapping) -> dict:
    """Direct deterministic readouts; reads old labels only for 64 supplied anchors."""
    grouped = group_bases(learning, config["learning_bases"], validate_targets=False)
    if len(pooled_base_ids) != 64 or len(set(pooled_base_ids)) != 64 or not set(pooled_base_ids) <= set(grouped):
        raise ValueError("readouts require exactly the shared 64 anchor bases")
    anchor_rows = [next(r for r in grouped[b] if r["label_rotation"] == 0) for b in pooled_base_ids]
    anchors = []
    for row in anchor_rows:
        options = _options(row)
        preferred = options[row["old_target"]]
        anchors.append({"base_id": row["base_id"], "user": source_user(row), "preferred": _tokens(preferred),
                        "other_options": [_tokens(v) for label, v in options.items() if label != row["old_target"]], "query": _tokens(" ".join(options.values()))})
    output = {"anchor_memory": [], "anchor_profile": []}
    salt = config["cheap_controls"]["tie_salt"]
    for row in development:
        options = _options(row)
        candidates = [a for a in anchors if a["user"] == source_user(row)] or anchors
        query = _tokens(" ".join(options.values()))
        nearest = min(candidates, key=lambda a: (-_jaccard(query, a["query"]), rank(a["base_id"], salt)))
        weights = Counter()
        for anchor in candidates:
            for token in anchor["preferred"]:
                weights[token] += 1 / len(candidates)
            others = anchor["other_options"]
            for other in others:
                for token in other:
                    weights[token] -= 1 / (len(candidates) * len(others))
        for name in output:
            def score(letter):
                tok = _tokens(options[letter])
                return _jaccard(tok, nearest["preferred"]) if name == "anchor_memory" else sum(weights[t] for t in tok) / max(1, len(tok))
            selected = min(LETTERS, key=lambda letter: (-score(letter), rank(str(row["base_id"]) + "|" + options[letter].casefold(), salt)))
            p = [float(letter == selected) for letter in LETTERS]
            output[name].append({**binding(row), "full_vocab_choice_probabilities": p, "normalized_choice_probabilities": p,
                                 "full_vocabulary_choice_mass": 1.0, "source": name, "accessible_delayed_bases": list(pooled_base_ids)})
    return output
