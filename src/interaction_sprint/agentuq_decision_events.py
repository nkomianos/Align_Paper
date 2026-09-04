"""Separate protocol-event prediction from agent decision-type prediction.

Classical smoothed categorical baselines; no model calls or paper reproduction.
All held-out labels absent from the training alphabet map to a reserved UNK.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from . import agentuq_calibrated, agentuq_prefix
from .cluster_certificate_audit import sha

SCOPES = ["all_events", "agent_given_last_event", "agent_given_last_agent", "tool_given_last_agent"]
UNK = "<UNK>"
START = "<START>"


def events(messages):
    labels = []
    for m in messages:
        role = m.get("role")
        if agentuq_prefix.generated(m):
            calls = m.get("tool_calls") or []
            label = "agent:tools:"+json.dumps([c["name"] for c in calls]) if calls else "agent:text"
        elif role == "assistant":
            continue  # fixed greeting, not an agent-generated decision
        elif role == "user":
            label = "user:message"
        elif role == "tool":
            label = "tool:"+str(m.get("requestor") or "unknown")
        else:
            continue
        labels.append(label)
    return labels


def examples(labels, scope):
    last_event, last_agent = START, START
    result = []
    for label in labels:
        is_agent = label.startswith("agent:")
        include = (scope == "all_events" or
                   (scope.startswith("agent_") and is_agent) or
                   (scope == "tool_given_last_agent" and label.startswith("agent:tools:")))
        if include:
            context = last_agent if scope.endswith("last_agent") else last_event
            result.append((context, label))
        last_event = label
        if is_agent:
            last_agent = label
    return result


def fit(sequences):
    global_counts, counts = Counter(), defaultdict(Counter)
    for seq in sequences:
        for context, target in seq:
            counts[context][target] += 1
            global_counts[target] += 1
    if not global_counts:
        raise ValueError("no training examples")
    return global_counts, counts, sorted([*global_counts, UNK])


def score(model, seq):
    global_counts, conditional, vocab = model
    if not seq:
        return None
    totals = {"unigram_bits": 0., "markov_bits": 0., "unigram_accuracy": 0., "markov_accuracy": 0.}
    unknown = 0
    for context, raw_target in seq:
        target = raw_target if raw_target in global_counts else UNK
        unknown += target == UNK
        for name, counts in [("unigram", global_counts), ("markov", conditional.get(context, global_counts))]:
            # Unseen contexts back off to unigram, not artificial uniform noise.
            probability = (counts[target]+.5)/(sum(counts.values())+.5*len(vocab))
            totals[name+"_bits"] -= math.log2(probability)
            highest = max(counts.get(v, 0) for v in vocab)
            tied = [v for v in vocab if counts.get(v, 0) == highest]
            totals[name+"_accuracy"] += (1/len(tied)) if target in tied else 0.
    return {"n": len(seq), "unknown": unknown, **{k:v/len(seq) for k,v in totals.items()}}


def evaluate(rows, scope):
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate tasks")
    folds = agentuq_calibrated.task_folds(ids)
    seqs = [examples(r["labels"], scope) for r in rows]
    scored, fits = [], []
    for f in range(5):
        train = [i for i in range(len(rows)) if folds[i] != f]
        test = [i for i in range(len(rows)) if folds[i] == f]
        model = fit([seqs[i] for i in train])
        fits.append({"fold": f, "train_ids": [ids[i] for i in train], "test_ids": [ids[i] for i in test],
                     "alphabet": model[2], "unigram": dict(model[0]),
                     "conditional": {k: dict(v) for k,v in model[1].items()}})
        for i in test:
            result = score(model, seqs[i])
            if result:
                scored.append({"task_id": ids[i], "fold": f, **result})
    metrics = ["unigram_bits", "markov_bits", "unigram_accuracy", "markov_accuracy"]
    return {"scope": scope, "tasks": len(scored), "events": sum(r["n"] for r in scored),
            "unknown_events": sum(r["unknown"] for r in scored),
            "macro": {m: float(np.mean([r[m] for r in scored])) for m in metrics},
            "micro": {m: float(sum(r[m]*r["n"] for r in scored)/sum(r["n"] for r in scored)) for m in metrics},
            "per_task": scored, "fits": fits}


def analyze(data):
    cells = []
    for path in sorted(data.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf8"))
        rows = [{"task_id": s["task_id"], "labels": events(s["messages"])} for s in raw["simulations"]]
        cells.append({"file": path.name, "tasks": len(rows), "sequences": rows,
                      "results": [evaluate(rows, scope) for scope in SCOPES]})
    return {"kind": "EXPLORATORY_AGENT_DECISION_VS_PROTOCOL_PREDICTION",
            "source_sha256": {Path(__file__).name: sha(Path(__file__)),
                              "agentuq_prefix.py": sha(Path(agentuq_prefix.__file__)),
                              "agentuq_calibrated.py": sha(Path(agentuq_calibrated.__file__))},
            "input_sha256": {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()},
            "cells": cells, "scope": [
                "Classical unigram and first-order conditional counts; fixed Jeffreys smoothing .5, no tuning.",
                "Task-held-out counts/alphabets; unseen labels pooled to UNK. Alphabet changes across folds.",
                "Different prediction targets have different entropy; absolute scores across scopes are not fair head-to-head comparisons.",
                "Agent labels encode text versus ordered tool-name bundles, not argument correctness or semantic success.",
                "Tool-only scoring conditions on a tool call occurring; it does not predict whether to call a tool.",
                "No execution, external replication, causal improvement or new algorithm claim."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-receipt", type=Path)
    a = p.parse_args()
    result = analyze(a.data_root)
    if a.verify_receipt:
        if json.loads((a.output/"MANIFEST.json").read_text(encoding="utf8")) != {"RESULT.json": sha(a.output/"RESULT.json")}:
            raise ValueError("manifest mismatch")
        if result != json.loads((a.output/"RESULT.json").read_text(encoding="utf8")):
            raise ValueError("source/input/replay mismatch")
        receipt = {"verified": True, "manifest_sha256": sha(a.output/"MANIFEST.json"), "scope": "Same-implementation replay only."}
        with a.verify_receipt.open("x", encoding="utf8") as out:
            json.dump(receipt, out, indent=2)
        print(json.dumps(receipt))
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output/"RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        (a.output/"MANIFEST.json").write_text(json.dumps({"RESULT.json": sha(a.output/"RESULT.json")}), encoding="utf8")
        for c in result["cells"]:
            print(json.dumps({"file": c["file"], "results": [{k:v for k,v in r.items() if k not in ["per_task", "fits"]} for r in c["results"]]}))
