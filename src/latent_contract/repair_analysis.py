"""Prespecified primary ridge-repair analysis; alternatives are not winner-selected."""
import math

import numpy as np

from latent_contract.answer_scoring import option_label
from latent_contract.update_comparison import paired_interval

REPAIR_ARMS = ["identity", "diagonal", "ridge", "orthogonal", "retuned"]


def analyze(cases, key, rows, paired_rows):
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids) or set(ids) != set(key):
        raise ValueError("repair case/key mismatch")
    index = {(r["case_id"], r["arm"]): r for r in rows}
    if len(rows) != len(ids)*len(REPAIR_ARMS) or set(index) != {(cid, arm) for cid in ids for arm in REPAIR_ARMS}:
        raise ValueError("repair grid mismatch")
    parent = {(r["case_id"], r["arm"]): r for r in paired_rows}
    for row in rows:
        reference = parent[row["case_id"], "new_c2c"]
        if row["messages"] != reference["messages"] or row["input_ids"] != reference["input_ids"] or row["dataset"] != reference["dataset"]:
            raise ValueError("repair prompt/data mismatch")
        if len(row["generated_ids"]) > 64 or row["hit_token_limit"] != (len(row["generated_ids"]) == 64):
            raise ValueError("repair decoding limit differs")
        if not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError("invalid repair timing")
    metrics = {}
    scores = {}
    for arm in REPAIR_ARMS:
        selected = [index[cid, arm] for cid in ids]
        scores[arm] = np.array([int(option_label(r["completion"]) == key[r["case_id"]]) for r in selected])
        metrics[arm] = {"n": len(ids), "accuracy": float(scores[arm].mean()),
                        "parse_rate": sum(option_label(r["completion"]) is not None for r in selected)/len(ids),
                        "token_limits": sum(r["hit_token_limit"] for r in selected),
                        "generation_seconds": sum(r["seconds"] for r in selected)}
    old, new = [np.array([int(option_label(parent[cid, v+"_c2c"]["completion"]) == key[cid]) for cid in ids]) for v in ("old", "new")]
    loss = float((old-new).mean())
    delta = scores["ridge"]-new
    recovery = float(delta.mean())/loss if loss > 0 else None
    ci = paired_interval(delta, [c["dataset"] for c in cases], seed=202609044)
    agreement = sum(index[cid, "identity"]["generated_ids"] == parent[cid, "new_c2c"]["generated_ids"] for cid in ids)/len(ids)
    valid = agreement >= .98 and min(m["parse_rate"] for m in metrics.values()) >= .95
    recovered = valid and recovery is not None and recovery >= .75 and ci[0] > 0
    decision = ("INVALID_REPAIR_ASSAY" if not valid else "NO_ORIGINAL_BRIDGE_LOSS_TO_RECOVER" if loss <= 0 else
                "EXISTING_RIDGE_BASELINE_RECOVERS_DAMAGE_NOT_NEW_METHOD" if recovered else
                "PRIMARY_RIDGE_RECOVERY_CRITERION_NOT_MET")
    return {"decision": decision, "metrics": metrics, "identity_token_agreement": agreement,
            "original_bridge_loss_pp": loss*100, "ridge_improvement_pp": float(delta.mean()*100),
            "ridge_paired_question_bootstrap_95_pp": [x*100 for x in ci], "ridge_recovery_fraction": recovery,
            "interpretation": "Repair algorithms are existing baselines. A recovery result is not a novel method, geometric identification, efficiency result or paper green light. Secondary repairs are reported without selecting a new primary winner."}
