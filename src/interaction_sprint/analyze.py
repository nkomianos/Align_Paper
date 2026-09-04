"""Read-only descriptive gates, never claims of training or causal human effects."""
import argparse
import json
from pathlib import Path

import numpy as np

from .run import inputs, select, validate, write


def analyze(study, cases, key, records, mode):
    expected = {c["case_id"] for c in cases}
    index = {r["case_id"]: r for r in records}
    if len(expected) != len(cases) or len(index) != len(records) or set(index) != expected:
        raise ValueError("incomplete, duplicate or unexpected record")
    for case in cases:
        row = index[case["case_id"]]
        probs = np.array(row["choice_probability"])
        lp = np.array(row["log_prob"])
        if (len(probs) != len(case["choices"]) or lp.shape != probs.shape
                or not np.isfinite(probs).all() or not np.isfinite(lp).all()
                or (probs < 0).any() or abs(probs.sum()-1) > 1e-5
                or not 0 <= row["choice_mass"] <= 1.00001):
            raise ValueError("invalid probability record")
        if row["predicted"] != case["choices"][int(probs.argmax())]:
            raise ValueError("prediction/probability disagreement")
        reconstructed = np.exp(lp-lp.max()); reconstructed /= reconstructed.sum()
        if not np.allclose(probs, reconstructed, atol=1e-5) or not np.isclose(np.exp(lp).sum(), row["choice_mass"], atol=1e-5):
            raise ValueError("full-vocabulary and conditional probabilities disagree")
    common = {"scope": "frozen_model_assay_only", "weight_updates": 0,
              "forwards": len(records), "mode": mode,
              "mean_choice_mass": float(np.mean([r["choice_mass"] for r in records]))}
    if study == "undo_audit":
        from .undo_audit import analyze as audit_analyze
        return audit_analyze(cases, key, records, mode, common)
    if study == "undo":
        def accuracy(group):
            return float(np.mean([index[c["case_id"]]["predicted"] == key[c["case_id"]]["answer"] for c in group]))
        cells = []
        for depth in sorted({c["depth"] for c in cases}):
            group = [c for c in cases if c["depth"] == depth]
            acc = {arm: accuracy([c for c in group if c["arm"] == arm])
                   for arm in ("canonical", "expanded", "padded", "counterfactual")}
            paired = [c for c in group if c["arm"] == "expanded" and c["relation"] != "commute"]
            residue, contrast = [], []
            for case in paired:
                cid = case["case_id"]
                stale = case["choices"].index(key[cid]["stale_answer"])
                padded = index[case["pair_id"]+"/padded"]
                expanded = index[cid]
                contrast.append(expanded["choice_probability"][stale]-padded["choice_probability"][stale])
                residue.append(expanded["predicted"] == key[cid]["stale_answer"])
            cells.append({"depth": depth, "accuracy": acc, "stale_choice_rate": float(np.mean(residue)),
                          "stale_probability_excess_over_padded": float(np.mean(contrast))})
        valid = (common["mean_choice_mass"] >= .5 and all(c["accuracy"]["canonical"] >= .9
                 and c["accuracy"]["counterfactual"] >= .8 and c["accuracy"]["padded"] >= .8 for c in cells))
        signal = any(c["depth"] >= 20 and c["accuracy"]["padded"]-c["accuracy"]["expanded"] >= .1
                     and c["stale_probability_excess_over_padded"] >= .05 for c in cells)
        decision = ("SMOKE_ONLY" if mode == "smoke" else "INVALID_UNDO_ASSAY" if not valid else
                    "RESIDUE_SIGNAL_DESIGN_TRAINING_NEXT" if signal else "NO_REGISTERED_RESIDUE_SIGNAL_PARK_THIS_ASSAY")
        return {**common, "decision": decision, "by_depth": cells,
                "stale_metric_scope": "fresh_cancel_and_overwrite_only; commute alternative was never asserted",
                "warning": "No local-relation training or theorem generalization was tested"}
    pairs = {}
    for c in cases:
        pairs.setdefault(c["pair_id"], {})[c["arm"]] = c
    rows = []
    for pair, group in pairs.items():
        if set(group) != {"base", "followup", "anchor", "followup_anchor"}:
            raise ValueError("incomplete matched signal block")
        base_case = group["base"]
        truth = key[base_case["case_id"]]
        get = lambda arm: index[group[arm]["case_id"]]
        base = get("base")
        action, initial = truth["action"], truth["initial"]
        raw_adv = get("followup")["log_prob"][action] - base["log_prob"][action]
        anchored_adv = get("followup_anchor")["log_prob"][action] - base["log_prob"][action]
        rows.append({"pair_id": pair, "surface": base_case["surface"], "regime": base_case["regime"],
                     "conflict": action != initial, "raw_log_ratio_on_logged_action": raw_adv,
                     "anchor_log_ratio_on_logged_action": anchored_adv,
                     "anchor_initial_probability": get("anchor")["choice_probability"][initial],
                     "raw_initial_probability": get("followup")["choice_probability"][initial],
                     "corrected_initial_probability": get("followup_anchor")["choice_probability"][initial]})
    conflict = [r for r in rows if r["conflict"] and r["regime"] != "static"]
    mean = lambda group, name: float(np.mean([r[name] for r in group]))
    metrics = {"wrong_logged_action_raw_advantage": mean(conflict, "raw_log_ratio_on_logged_action"),
               "wrong_logged_action_anchored_advantage": mean(conflict, "anchor_log_ratio_on_logged_action"),
               "independent_anchor_initial_probability": mean(rows, "anchor_initial_probability"),
               "truthful_correction_initial_probability": mean([r for r in rows if r["regime"] == "static"], "raw_initial_probability"),
               "anchor_restoration_on_conflicting_feedback": mean(conflict, "corrected_initial_probability")-mean(conflict, "raw_initial_probability")}
    valid = (common["mean_choice_mass"] >= .5 and metrics["independent_anchor_initial_probability"] >= .8
             and metrics["truthful_correction_initial_probability"] >= .8)
    signal = metrics["wrong_logged_action_raw_advantage"] > .1 and metrics["anchor_restoration_on_conflicting_feedback"] >= .1
    decision = ("SMOKE_ONLY" if mode == "smoke" else "INVALID_ENDO_LANGUAGE_ASSAY" if not valid else
                "SIGNAL_PRESENT_TRAINING_AND_IDENTIFICATION_STILL_REQUIRED" if signal else
                "NO_SIGNAL_IN_THIS_PROMPT_ASSAY_DO_NOT_INFER_THESIS_FALSE")
    return {**common, "decision": decision, "metrics": metrics, "pair_diagnostics": rows,
            "warning": "Conditional forced interventions; not on-policy sampling, welfare decline, polarization, or an SDPO reproduction"}


def verify(root):
    validate(root)
    spec, all_cases, key = inputs(root / "inputs")
    plan = json.loads((root / "plan.json").read_text())
    cases = select(all_cases, plan["mode"])
    if plan["settings"] != spec or plan["case_ids"] != [c["case_id"] for c in cases]:
        raise ValueError("plan mismatch")
    if not (root / "COMPLETE.json").exists() or (root / "FAILED.json").exists():
        raise ValueError("incomplete evidence")
    runtime = json.loads((root / "runtime.json").read_text())
    if runtime["model"] != spec["model"] or runtime["revision"] != spec["revision"] or runtime["weight_updates"] != 0:
        raise ValueError("runtime/model mismatch")
    records = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    if json.loads((root / "COMPLETE.json").read_text()) != {"records": len(cases), "weight_updates": 0}:
        raise ValueError("completion count mismatch")
    audit = json.loads((root / "BUDGET_AUDIT.json").read_text())
    budget = {a["case_id"]: a["input_tokens"] for a in audit}
    if len(budget) != len(audit) or set(budget) != set(plan["case_ids"]):
        raise ValueError("budget audit incomplete")
    for row in records:
        if row["input_tokens"] != budget[row["case_id"]] or not 0 < row["input_tokens"] <= spec["max_context_tokens"]:
            raise ValueError("budget mismatch")
    return analyze(spec["study"], cases, key, records, plan["mode"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("report must be outside immutable evidence")
    report = verify(args.root)
    write(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "pair_diagnostics"}, indent=2))
