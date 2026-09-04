"""Post-hoc, task-held-out calibration follow-up to the matched rank audit.

Tests an existing logistic baseline, not a novel learner. No model inference.
Each test task is excluded from feature scaling and outcome fitting together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import agentuq_matched as matched
from .cluster_certificate_audit import sha


def task_folds(ids):
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        ("agentuq-cal-v1:"+str(ids[i])).encode()).hexdigest())
    fold = np.empty(len(ids), dtype=int)
    for j, i in enumerate(order):
        fold[i] = j % 5
    return fold


def design(scores, means, scales):
    n = len(scores)
    z = (scores-means)/scales
    # Separate model intercepts/slopes; intercept regularized via model indicator.
    model = np.tile([0., 1.], n)
    return np.column_stack([model, (z*np.array([1, 0])).ravel(),
                            (z*np.array([0, 1])).ravel()])


def crossfit(ids, scores, y):
    scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
    if scores.shape != y.shape or scores.shape != (len(ids), 2):
        raise ValueError("expected n by 2 matched arrays")
    if len(set(ids)) != len(ids) or len(ids) < 10:
        raise ValueError("duplicate/insufficient tasks")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid data")
    y = y.astype(int)
    folds = task_folds(ids)
    predicted = np.zeros_like(scores)
    constant = np.zeros_like(scores)
    fits = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        means = scores[train].mean(axis=0)
        scales = scores[train].std(axis=0)
        scales[scales < 1e-12] = 1.
        rates = (y[train].sum(axis=0)+1)/(train.sum()+2)
        constant[test] = rates
        labels = y[train].ravel()
        if len(np.unique(labels)) < 2:
            predicted[test] = rates
            parameters = None
        else:
            clf = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-9)
            clf.fit(design(scores[train], means, scales), labels)
            if clf.n_iter_[0] >= 1000:
                raise ValueError("unconverged fit")
            predicted[test] = clf.predict_proba(design(scores[test], means, scales))[:, 1].reshape(-1, 2)
            parameters = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
        fits.append({"fold": fold, "train_ids": [ids[i] for i in np.flatnonzero(train)],
                     "test_ids": [ids[i] for i in np.flatnonzero(test)],
                     "means": means.tolist(), "scales": scales.tolist(),
                     "base_success_rates": rates.tolist(), "parameters": parameters})
    selected = matched.selector(y, -predicted)
    baseline = matched.selector(y, -constant)
    return {
        "tasks": len(ids), "selector_success": float(selected.mean()),
        "constant_success": float(baseline.mean()),
        "delta": float((selected-baseline).mean()),
        "conditional_bootstrap95_delta": matched.paired_bootstrap(selected-baseline),
        "brier": float(((predicted-y)**2).mean()),
        "constant_brier": float(((constant-y)**2).mean()),
        "selected_gpt": int((predicted[:, 0] > predicted[:, 1]).sum()),
        "per_task": [{"id": tid, "fold": int(folds[i]), "predicted": predicted[i].tolist(),
                      "baseline_predicted": constant[i].tolist(), "success": y[i].tolist(),
                      "selected_success": float(selected[i]), "constant_success": float(baseline[i])}
                     for i, tid in enumerate(ids)],
        "fits": fits,
    }


def analyze(root):
    source = json.loads((root/"RESULT.json").read_text(encoding="utf8"))
    if json.loads((root/"MANIFEST.json").read_text(encoding="utf8")) != {"RESULT.json": sha(root/"RESULT.json")}:
        raise ValueError("input manifest mismatch")
    cells = []
    for cell in source["cells"]:
        features = []
        for f in cell["features"]:
            rows = f["per_task"]
            result = crossfit([r["task_id"] for r in rows], [r["raw_scores"] for r in rows],
                              [r["success"] for r in rows])
            features.append({"field": f["field"], **result})
        cells.append({"domain": cell["domain"], "features": features})
    return {
        "kind": "POSTHOC_GROUPED_CROSSFIT_CALIBRATION_NOT_NEW_METHOD",
        "source_sha256": {"agentuq_calibrated.py": sha(Path(__file__)),
                          "agentuq_matched.py": sha(Path(matched.__file__))},
        "input_manifest_sha256": sha(root/"MANIFEST.json"),
        "protocol": "Five deterministic task-held-out folds per domain; train-only standardization; logistic C=1, model identity and model-specific feature slopes; no tuning.",
        "scope": "Exploratory after rank audit. Completed-trace selection is not live intervention. Bootstrap conditions on crossfit predictions; no refitting, independence or significance guarantee. All five features reported.",
        "cells": cells,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-receipt", type=Path)
    args = p.parse_args()
    result = analyze(args.input_root)
    if args.verify_receipt:
        if json.loads((args.output/"MANIFEST.json").read_text(encoding="utf8")) != {"RESULT.json": sha(args.output/"RESULT.json")}:
            raise ValueError("result manifest mismatch")
        if result != json.loads((args.output/"RESULT.json").read_text(encoding="utf8")):
            raise ValueError("deterministic replay mismatch")
        receipt = {"verified": True, "manifest_sha256": sha(args.output/"MANIFEST.json"),
                   "scope": "Same-implementation deterministic replay, not independent replication."}
        with args.verify_receipt.open("x", encoding="utf8") as out:
            json.dump(receipt, out, indent=2)
        print(json.dumps(receipt))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        (args.output/"MANIFEST.json").write_text(json.dumps({"RESULT.json": sha(args.output/"RESULT.json")}), encoding="utf8")
        for c in result["cells"]:
            print(json.dumps({"domain": c["domain"], "features": [
                {k:v for k,v in f.items() if k not in ["per_task", "fits"]} for f in c["features"]]}))
