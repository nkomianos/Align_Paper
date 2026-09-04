"""Read-only descriptive audit of pinned public AgentUQ v1.1 trajectories.

No inference, training, prefix logprob reconstruction, or gate selection.
Positive AUROC label is failure; score orientation is never selected by result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .cluster_certificate_audit import sha


def auc(rows, field):
    values = [(r["failure"], r[field]) for r in rows if r.get(field) is not None]
    if not values:
        return {"n": 0, "failures": 0, "auroc": None}
    y, scores = np.array(values).T
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid scores/labels")
    npos, nneg = int(y.sum()), int(len(y)-y.sum())
    score = None if not npos or not nneg else float(
        (rankdata(scores)[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg))
    return {"n": len(y), "failures": npos, "auroc": score}


def summarize(path: Path):
    data = json.loads(path.read_text(encoding="utf8"))
    rows, logical_ids, coverage = [], [], Counter()
    for sim in data["simulations"]:
        reward = sim["reward_info"]["reward"]
        if reward not in [0, 1]:
            raise ValueError("fractional reward requires an explicit new analysis")
        identity = (sim["task_id"], sim.get("trial"), sim.get("seed"))
        if identity in logical_ids:
            raise ValueError("duplicate task/trial/seed")
        logical_ids.append(identity)
        # Fixed greeting has no raw generation record or usage; exclude it.
        actions = [m for m in sim["messages"] if m.get("role") == "assistant"
                   and (m.get("raw_data") is not None or m.get("usage") is not None)]
        row = {"task_id": sim["task_id"], "trial": sim.get("trial"),
               "seed": sim.get("seed"), "failure": int(1-reward),
               "final_action_count": len(actions)}
        for m in actions:
            kind = "tool" if m.get("tool_calls") else "text"
            coverage[kind] += 1
            uq = (m.get("raw_data") or {}).get("uq_logprobs")
            if uq:
                coverage[kind+"_with_logprobs"] += 1
        for role in ["assistant", "user", "combined"]:
            summary = sim.get("uq_summary", {}).get(role, {})
            for key in ["avg_token_nll", "mean_topk_entropy", "total_tokens"]:
                row[role+"_"+key] = summary.get(key)
        for k in [1, 2, 4, 8]:
            prefix = actions[:k]
            counts = [(m.get("usage") or {}).get("completion_tokens") for m in prefix]
            row[f"prefix{k}_mean_completion_tokens"] = (
                float(np.mean(counts)) if len(prefix) == k and all(v is not None for v in counts) else None)
            row[f"prefix{k}_tool_fraction"] = (
                sum(bool(m.get("tool_calls")) for m in prefix)/k if len(prefix)==k else None)
        rows.append(row)
    fields = [k for k in rows[0] if k not in ["task_id", "trial", "seed", "failure"]]
    return {
        "file": path.name, "rows": len(rows),
        "task_count": len(set(r["task_id"] for r in rows)),
        "trials_per_task": dict(Counter(Counter(r["task_id"] for r in rows).values())),
        "coverage": dict(coverage),
        "scores": {field: auc(rows, field) for field in fields},
        "per_trajectory": rows,
    }


def analyze(root: Path):
    files = sorted(root.glob("*.json"))
    if len(files) != 6:
        raise ValueError("expected exactly six pinned trajectory files")
    return {
        "kind": "EXPLORATORY_PUBLIC_DATA_REANALYSIS_NOT_NEW_MODEL_EXPERIMENT",
        "dataset": "changdae/tau2-uq-artifacts",
        "revision": "824d9ec3b53067cb65153fed7c6bbc3815f7e2bb",
        "source_sha256": sha(Path(__file__)),
        "data_sha256": {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file()},
        "cells": [summarize(p) for p in files],
        "scope": [
            "Terminal UQ and final action count are retrospective, not online features.",
            "Prefix scores use fixed assistant-action counts, not future-normalized progress.",
            "A prefix-k cohort includes only trajectories reaching k; cross-k AUCs change population.",
            "No sign flipping, fitted predictors, threshold selection, or new model calls.",
            "GPT tool-call argument logprobs are absent; Kimi tokens include reasoning/markup.",
            "Single trial per task/policy cannot identify within-policy rollout heterogeneity.",
            "Neither pooled score differences nor descriptive AUCs prove intervention utility.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = analyze(args.data_root)
    (args.output/"RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    (args.output/"MANIFEST.json").write_text(json.dumps({"RESULT.json": sha(args.output/"RESULT.json")}, indent=2), encoding="utf8")
    for cell in result["cells"]:
        print(json.dumps({k:v for k,v in cell.items() if k != "per_trajectory"}))
