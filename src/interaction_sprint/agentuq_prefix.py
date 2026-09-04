"""Fixed observable-prefix telemetry baseline on public agent trajectories.

No user simulator internals, final lengths, future-normalized times or terminal
uncertainty summaries enter features. This is not a novel monitor algorithm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import agentuq_calibrated, agentuq_inventory
from .cluster_certificate_audit import sha

FEATURES = ["mean_completion_tokens", "tool_calls_per_action", "unique_tool_fraction",
            "repeated_call_fraction", "error_response_fraction", "same_tool_transition_fraction",
            "mean_tool_response_chars", "mean_assistant_text_chars"]


def generated(m):
    return m.get("role") == "assistant" and (m.get("raw_data") is not None or m.get("usage") is not None)


def prefix_features(messages, k):
    if k < 1:
        raise ValueError("positive action clock required")
    actions, responses, n = [], [], 0
    pending = set()
    for m in messages:
        # After the kth action, only its immediately following tool responses
        # are observed. Stop before any subsequent user or assistant content.
        if n == k and m.get("role") != "tool":
            break
        if generated(m):
            n += 1
            actions.append(m)
            for call in m.get("tool_calls") or []:
                if call.get("id"):
                    pending.add(call["id"])
        elif m.get("role") == "tool" and m.get("requestor") == "assistant":
            if m.get("id") not in pending:
                raise ValueError("assistant tool response has no observed request")
            responses.append(m)
            pending.remove(m["id"])
    if n < k:
        return None
    counts = [(m.get("usage") or {}).get("completion_tokens") for m in actions]
    if any(x is None for x in counts):
        raise ValueError("missing completion token metadata")
    calls = [c for m in actions for c in (m.get("tool_calls") or [])]
    names = [c["name"] for c in calls]
    identities = [json.dumps([c["name"], c.get("arguments")], sort_keys=True) for c in calls]
    errors = [bool(m.get("error")) or str(m.get("content", "")).lstrip().lower().startswith("error:") for m in responses]
    return {
        "mean_completion_tokens": float(np.log1p(np.mean(counts))),
        "tool_calls_per_action": len(calls)/k,
        "unique_tool_fraction": len(set(names))/max(1, len(names)),
        "repeated_call_fraction": (len(identities)-len(set(identities)))/max(1, len(identities)),
        "error_response_fraction": sum(errors)/max(1, len(responses)),
        "same_tool_transition_fraction": sum(a == b for a,b in zip(names, names[1:]))/max(1, len(names)-1),
        "mean_tool_response_chars": float(np.log1p(np.mean([len(str(m.get("content", ""))) for m in responses]))) if responses else 0.,
        "mean_assistant_text_chars": float(np.log1p(np.mean([len(str(m.get("content") or "")) for m in actions]))),
    }


def crossfit(rows, fields):
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task")
    if len(ids) < 10:
        return {"insufficient_tasks": True, "tasks": len(ids)}
    x = np.array([[r["features"][f] for f in fields] for r in rows], dtype=float)
    y = np.array([r["failure"] for r in rows])
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid features/labels")
    folds = agentuq_calibrated.task_folds(ids)
    preds, base = np.zeros(len(ids)), np.zeros(len(ids))
    fits = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        base[test] = (y[train].sum()+1)/(train.sum()+2)
        means, scales = x[train].mean(axis=0), x[train].std(axis=0)
        scales[scales < 1e-12] = 1.
        params = None
        if len(set(y[train])) < 2:
            preds[test] = base[test]
        else:
            clf = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-9)
            clf.fit((x[train]-means)/scales, y[train])
            if clf.n_iter_[0] >= 1000:
                raise ValueError("unconverged logistic fit")
            preds[test] = clf.predict_proba((x[test]-means)/scales)[:, 1]
            params = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
        fits.append({"fold": fold, "train_ids": [ids[i] for i in np.flatnonzero(train)],
                     "test_ids": [ids[i] for i in np.flatnonzero(test)], "means": means.tolist(),
                     "scales": scales.tolist(), "params": params})
    scored = [{"task_id": tid, "failure": int(y[i]), "prediction": float(preds[i]),
               "base": float(base[i]), "fold": int(folds[i])} for i, tid in enumerate(ids)]
    return {"tasks": len(ids), "failures": int(y.sum()),
            "auroc": agentuq_inventory.auc(scored, "prediction")["auroc"],
            "brier": float(np.mean((preds-y)**2)), "constant_brier": float(np.mean((base-y)**2)),
            "per_task": scored, "fits": fits}


def analyze(data):
    cells = []
    for path in sorted(data.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf8"))
        for k in [2, 4, 8]:
            rows = []
            for sim in raw["simulations"]:
                features = prefix_features(sim["messages"], k)
                if features is not None:
                    reward = sim["reward_info"]["reward"]
                    if reward not in [0, 1]:
                        raise ValueError("nonbinary outcome")
                    rows.append({"task_id": sim["task_id"], "failure": int(1-reward), "features": features})
            cells.append({"file": path.name, "prefix_actions": k, "total_traces": len(raw["simulations"]),
                          "cohort_rows": rows, "length_tool_baseline": crossfit(rows, FEATURES[:2]),
                          "execution_telemetry": crossfit(rows, FEATURES)})
    return {"kind": "EXPLORATORY_OBSERVABLE_PREFIX_TELEMETRY_BASELINE",
            "source_sha256": {Path(__file__).name: sha(Path(__file__)),
                              "agentuq_calibrated.py": sha(Path(agentuq_calibrated.__file__)),
                              "agentuq_inventory.py": sha(Path(agentuq_inventory.__file__))},
            "input_sha256": {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()},
            "features": FEATURES, "cells": cells,
            "scope": ["Fixed action clocks; only traces reaching clock included. Cross-clock populations differ.",
                      "Test-task-exclusive fitting/scaling; separate per model/domain/prefix; no tuning.",
                      "Tool error labels come from observed responses, not final reward or private simulator reasoning.",
                      "No monitor interventions or Progress Advantage/CEB/automaton baseline reproduction.",
                      "Exploratory public data after earlier analyses; not pristine confirmatory evidence."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-receipt", type=Path)
    args = parser.parse_args()
    result = analyze(args.data_root)
    if args.verify_receipt:
        if json.loads((args.output/"MANIFEST.json").read_text(encoding="utf8")) != {"RESULT.json": sha(args.output/"RESULT.json")}:
            raise ValueError("manifest mismatch")
        if result != json.loads((args.output/"RESULT.json").read_text(encoding="utf8")):
            raise ValueError("source/input/replay mismatch")
        receipt = {"verified": True, "manifest_sha256": sha(args.output/"MANIFEST.json"), "scope": "Same-implementation replay, not independent replication."}
        with args.verify_receipt.open("x", encoding="utf8") as out:
            json.dump(receipt, out, indent=2)
        print(json.dumps(receipt))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        (args.output/"MANIFEST.json").write_text(json.dumps({"RESULT.json": sha(args.output/"RESULT.json")}), encoding="utf8")
        for c in result["cells"]:
            print(json.dumps({"file": c["file"], "k": c["prefix_actions"], **{arm: {k:v for k,v in c[arm].items() if k not in ["per_task", "fits"]} for arm in ["length_tool_baseline", "execution_telemetry"]}}))
