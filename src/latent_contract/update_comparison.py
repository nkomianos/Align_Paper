"""Paired, question-level analysis for natural-update development evidence."""
import math

import numpy as np

from latent_contract.answer_scoring import option_label

ARMS = ["receiver"] + [version + "_" + arm for version in ("old", "new")
                       for arm in ("sender", "c2c", "disabled", "background", "text")]
DECISION_RULE = {"min_parse_rate": .95, "min_disabled_agreement": .98,
                 "min_old_c2c_gain": .05, "max_sender_loss": .03125,
                 "min_extra_latent_loss": .10, "bootstrap_samples": 5000,
                 "bootstrap_seed": 202609043}


def paired_interval(values, datasets, samples=5000, seed=202609043):
    values = np.asarray(values, dtype=float)
    if not len(values) or len(values) != len(datasets) or not np.isfinite(values).all():
        raise ValueError("invalid paired values")
    groups = [np.flatnonzero(np.array(datasets) == label) for label in sorted(set(datasets))]
    rng = np.random.default_rng(seed)
    replicates = np.zeros(samples)
    for indices in groups:
        draws = rng.choice(indices, size=(samples, len(indices)), replace=True)
        replicates += values[draws].sum(axis=1)/len(values)
    return [float(x) for x in np.quantile(replicates, [.025, .975])]


def analyze(cases, key, rows):
    ids = [c["case_id"] for c in cases]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(key):
        raise ValueError("invalid case/key grid")
    indexed = {(r["case_id"], r["arm"]): r for r in rows}
    if len(rows) != len(ids)*len(ARMS) or set(indexed) != {(cid, arm) for cid in ids for arm in ARMS}:
        raise ValueError("missing/duplicate/unexpected paired outputs")
    datasets = {c["case_id"]: c["dataset"] for c in cases}
    for row in rows:
        if row["dataset"] != datasets[row["case_id"]] or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError("invalid row metadata")
        if not 0 < len(row["input_ids"]) <= 2048 or len(row["generated_ids"]) > 64:
            raise ValueError("token limits differ")
        if row["hit_token_limit"] != (len(row["generated_ids"]) == 64):
            raise ValueError("truncation flag differs")
    for cid in ids:
        direct = [indexed[cid, arm] for arm in ARMS if not arm.endswith(("_text", "_background"))]
        if any(row["input_ids"] != direct[0]["input_ids"] or row["messages"] != direct[0]["messages"] for row in direct):
            raise ValueError("unmatched direct prompts")
        for version in ("old", "new"):
            background, text = [indexed[cid, version+"_"+arm] for arm in ("background", "text")]
            expected = [*background["messages"], {"role": "assistant", "content": background["completion"]}, *direct[0]["messages"]]
            if text["messages"] != expected:
                raise ValueError("incorrect background transfer")
        if indexed[cid, "old_background"]["input_ids"] != indexed[cid, "new_background"]["input_ids"]:
            raise ValueError("unmatched background queries")
    answered = [a for a in ARMS if not a.endswith("_background")]
    scores, metrics = {}, {}
    for arm in answered:
        outputs = [indexed[cid, arm] for cid in ids]
        scores[arm] = np.array([int(option_label(r["completion"]) == key[r["case_id"]]) for r in outputs])
        cost_rows = outputs + ([indexed[cid, arm.replace("_text", "_background")] for cid in ids] if arm.endswith("_text") else [])
        metrics[arm] = {"accuracy": float(scores[arm].mean()), "correct": int(scores[arm].sum()), "n": len(ids),
                        "parse_rate": sum(option_label(r["completion"]) is not None for r in outputs)/len(ids),
                        "token_limits": sum(r["hit_token_limit"] for r in outputs),
                        "serial_generation_seconds": sum(r["seconds"] for r in cost_rows)}
    changes = {arm: scores["new_"+arm] - scores["old_"+arm] for arm in ("sender", "c2c", "text")}
    extra = changes["c2c"] - changes["text"]
    ci = paired_interval(extra, [datasets[cid] for cid in ids], DECISION_RULE["bootstrap_samples"], DECISION_RULE["bootstrap_seed"])
    agreements = {v: sum(indexed[cid, v+"_disabled"]["completion"].strip() == indexed[cid, "receiver"]["completion"].strip()
                         for cid in ids)/len(ids) for v in ("old", "new")}
    controls = min(agreements.values()) >= DECISION_RULE["min_disabled_agreement"] and min(m["parse_rate"] for m in metrics.values()) >= DECISION_RULE["min_parse_rate"]
    usable = metrics["old_c2c"]["accuracy"] - metrics["receiver"]["accuracy"] >= DECISION_RULE["min_old_c2c_gain"]
    retained = float(changes["sender"].mean()) >= -DECISION_RULE["max_sender_loss"]
    signal = float(extra.mean()) <= -DECISION_RULE["min_extra_latent_loss"] and ci[1] < 0
    decision = ("INVALID_PAIRED_ASSAY" if not controls else "NO_USABLE_OLD_BRIDGE_ON_FINAL_DEV" if not usable
                else "SENDER_CAPABILITY_CHANGE_CONFOUNDS_INTERPRETATION" if not retained
                else "EXTRA_LATENT_LOSS_SIGNAL_REPAIR_STUDY_NEEDED" if signal
                else "NO_REQUIRED_EXTRA_LATENT_LOSS_ON_THIS_UPDATE")
    return {"decision": decision, "metrics": metrics, "disabled_agreement": agreements,
            "changes_pp": {k: float(v.mean()*100) for k, v in changes.items()},
            "extra_latent_change_pp": float(extra.mean()*100), "paired_question_bootstrap_95_pp": [v*100 for v in ci],
            "background_token_limits": {v: sum(indexed[cid, v+"_background"]["hit_token_limit"] for cid in ids) for v in ("old", "new")},
            "controls_pass": controls, "old_bridge_usable": usable, "sender_retained": retained,
            "interpretation": "One natural update and public DEV questions. Difference-in-differences is descriptive, not geometric causal identification. Question bootstrap does not quantify uncertainty across updates. No paper go or repair success follows."}
