"""Exploratory matched-task selection audit; no model calls or trained monitor.

Ranks are transductive, label-free within-model/domain calibrations. Selection
uses two *already completed* traces and does not simulate switching live agents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import agentuq_inventory
from .cluster_certificate_audit import sha

FIELDS = ["assistant_avg_token_nll", "assistant_mean_topk_entropy",
          "final_action_count", "prefix2_mean_completion_tokens",
          "prefix2_tool_fraction"]
DOMAINS = ["airline", "retail", "telecom"]


def keyed(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate identity")
    return result


def percentiles(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("empty/nonfinite scores")
    return (rankdata(values, method="average") - .5) / len(values)


def selector(y, scores):
    """Lower score is better. Exact ties randomize with probability one half."""
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2 or scores.shape != y.shape:
        raise ValueError("expected matched n by 2 arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("invalid labels/scores")
    first = np.where(scores[:, 0] < scores[:, 1], 1.,
                     np.where(scores[:, 0] > scores[:, 1], 0., .5))
    return first*y[:, 0] + (1-first)*y[:, 1]


def loo_constant(y):
    """Choose the best constant model using the other tasks' outcomes only."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        raise ValueError("need at least two tasks")
    return selector(y, -(y.sum(axis=0, keepdims=True)-y))


def paired_bootstrap(delta, seed=9040731, repeats=5000):
    """Conditional descriptive interval; not a refit or selective-inference CI."""
    delta = np.asarray(delta, dtype=float)
    if not len(delta):
        return None
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(len(delta), size=(repeats, len(delta)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def compare(left, right, field):
    ids = sorted(set(left) & set(right))
    ids = [i for i in ids if left[i].get(field) is not None and right[i].get(field) is not None]
    if len(ids) < 2:
        return {"tasks": len(ids), "field": field, "insufficient": True}
    y = np.array([[1-left[i]["failure"], 1-right[i]["failure"]] for i in ids])
    raw = np.array([[left[i][field], right[i][field]] for i in ids])
    ranked = np.column_stack([percentiles(raw[:, j]) for j in range(2)])
    selected, constant = selector(y, ranked), loo_constant(y)
    discordant = y[:, 0] != y[:, 1]
    # Same-task comparison is a cross-model *selection* target, not within-policy UQ.
    pooled = [{"failure": 1-y[i, j], "score": ranked[i, j]}
              for i in range(len(ids)) for j in range(2)]
    return {
        "field": field, "tasks": len(ids), "discordant_tasks": int(discordant.sum()),
        "both_success": int(np.all(y == 1, axis=1).sum()),
        "both_fail": int(np.all(y == 0, axis=1).sum()),
        "gpt_only_success": int(np.all(y == [1, 0], axis=1).sum()),
        "kimi_only_success": int(np.all(y == [0, 1], axis=1).sum()),
        "gpt_success": float(y[:, 0].mean()), "kimi_success": float(y[:, 1].mean()),
        "oracle_success": float(y.max(axis=1).mean()),
        "loo_constant_success": float(constant.mean()),
        "in_sample_best_constant_success": float(y.mean(axis=0).max()),
        "rank_selector_success": float(selected.mean()),
        "delta_vs_loo_constant": float((selected-constant).mean()),
        "conditional_bootstrap95_delta": paired_bootstrap(selected-constant),
        "discordant_concordance": float(selected[discordant].mean()) if discordant.any() else None,
        "pooled_rank_auroc": agentuq_inventory.auc(pooled, "score")["auroc"],
        "per_task": [{"task_id": i, "success": y[k].tolist(), "raw_scores": raw[k].tolist(),
                      "ranks": ranked[k].tolist(), "selected_success": float(selected[k]),
                      "loo_constant_success": float(constant[k])} for k, i in enumerate(ids)],
    }


def analyze(data):
    cells = []
    for domain in DOMAINS:
        paths = [data/f"gpt41_kimik25_{domain}.json", data/f"Kimi-K2.5_{domain}.json"]
        raw = [json.loads(p.read_text(encoding="utf8")) for p in paths]
        # IDs alone are not evidence of matched tasks. Compare all task specs.
        if keyed(raw[0]["tasks"], "id") != keyed(raw[1]["tasks"], "id"):
            raise ValueError(f"task definitions differ: {domain}")
        for item in ["user_info", "environment_info", "max_steps", "max_errors"]:
            if raw[0]["info"][item] != raw[1]["info"][item]:
                raise ValueError(f"unmatched protocol {domain}: {item}")
        rows = [keyed(agentuq_inventory.summarize(p)["per_trajectory"], "task_id") for p in paths]
        if set(rows[0]) != set(rows[1]):
            raise ValueError("different task coverage")
        if set(rows[0]) != set(keyed(raw[0]["tasks"], "id")):
            raise ValueError("simulation/task mismatch")
        cells.append({"domain": domain, "task_and_protocol_match": True,
                      "features": [compare(*rows, f) for f in FIELDS]})
    return {
        "kind": "EXPLORATORY_MATCHED_TASK_CROSS_MODEL_SELECTION",
        "dataset_revision": "824d9ec3b53067cb65153fed7c6bbc3815f7e2bb",
        "source_sha256": {"agentuq_matched.py": sha(Path(__file__)),
                          "agentuq_inventory.py": sha(Path(agentuq_inventory.__file__))},
        "input_sha256": {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()},
        "cells": cells,
        "scope": [
            "One trial per task/model. No causal or within-policy rollout claim.",
            "Higher raw uncertainty/length/tool fraction prespecified as worse; no sign tuning.",
            "Percentile normalization is label-free but transductive, not a deployed calibration.",
            "First-two-action cohorts require both traces to reach two generated actions.",
            "Outcome of a switched/aborted tool execution is NOT observed or estimated.",
            "Terminal selection is a retrospective oracle for early routing, not a valid policy execution.",
            "Intervals bootstrap fixed per-task payoffs without refitting ranks or baseline; descriptive only.",
            "Multiple exploratory features, no confirmatory significance or paper go/no-go test.",
            "Progress Advantage scores are not provided in these files and were not computed.",
        ],
    }


def verify(root, data):
    manifest = json.loads((root/"MANIFEST.json").read_text(encoding="utf8"))
    if manifest != {"RESULT.json": sha(root/"RESULT.json")}:
        raise ValueError("manifest mismatch")
    saved = json.loads((root/"RESULT.json").read_text(encoding="utf8"))
    if saved != analyze(data):
        raise ValueError("source/input or deterministic replay mismatch")
    return {"verified": True, "manifest_sha256": sha(root/"MANIFEST.json"),
            "scope": "Hashes and same-implementation replay; not independent replication."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-receipt", type=Path)
    args = p.parse_args()
    if args.verify_receipt:
        receipt = verify(args.output, args.data_root)
        with args.verify_receipt.open("x", encoding="utf8") as out:
            json.dump(receipt, out, indent=2)
        print(json.dumps(receipt))
    else:
        result = analyze(args.data_root)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        (args.output/"MANIFEST.json").write_text(json.dumps({"RESULT.json": sha(args.output/"RESULT.json")}), encoding="utf8")
        for cell in result["cells"]:
            print(json.dumps({"domain": cell["domain"], "features": [
                {k: v for k, v in row.items() if k != "per_task"} for row in cell["features"]]}))
