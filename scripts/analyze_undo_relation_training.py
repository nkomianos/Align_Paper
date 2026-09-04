"""Read-only evidence and descriptive paired comparison; no automatic go."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ARMS = ("baseline", "terminal_sft", "canonical_distillation", "local_relation")


def qualify(rows):
    controls = {}
    for arm in ("canonical", "padded", "counterfactual"):
        group = [r for r in rows if r["condition"] == arm]
        controls[arm] = {"n": len(group), "accuracy": sum(r["prediction"] == r["target"] for r in group) / len(group) if group else 0.0}
    mass = float(np.mean([r["choice_mass"] for r in rows])) if rows else 0.0
    return {"qualified": bool(rows) and all(x["n"] and x["accuracy"] >= .9 for x in controls.values()) and mass >= .5,
            "controls": controls, "mean_choice_mass": mass}


def validate_records(data, rows, tokenized):
    expected = {r["id"]: r for r in data}
    records = {r["id"]: r for r in rows}
    if len(data) != len(expected) or len(rows) != len(records) or set(records) != set(expected):
        raise ValueError("missing or duplicate outputs")
    lengths = {r["id"]: len(ids) for r, ids in zip(data, tokenized)}
    if len(tokenized) != len(data):
        raise ValueError("tokenized length mismatch")
    for rid, row in records.items():
        for field in ("target", "condition", "depth", "pair_id", "prompt"):
            if row[field] != expected[rid][field]:
                raise ValueError(f"changed input {rid}/{field}")
        p = np.asarray(row["choice_probabilities"], dtype=float)
        if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() > 1.001:
            raise ValueError("bad choice probabilities")
        if abs(p.sum() - row["choice_mass"]) > .002 or row["prediction"] != "ABCD"[int(p.argmax())]:
            raise ValueError("choice arithmetic mismatch")
        if row["prompt_tokens"] != lengths[rid]:
            raise ValueError("prompt token mismatch")
    return records


def verify_budget(root, manifest, freeze, tokenized, summary):
    config = freeze["arguments"]
    n = len(tokenized["student"]["train"])
    rng = random.Random(config["seed"])
    plan = []
    for _ in range(config["epochs"]):
        ids = list(range(n)); rng.shuffle(ids)
        plan.extend(ids[i:i + config["batch_size"]] for i in range(0, n, config["batch_size"]))
    if tokenized["schedule"] != plan:
        raise ValueError("schedule does not match frozen budget")
    tokens = [sum(len(tokenized["student"]["train"][i]) for i in ids) for ids in plan]
    if summary["training_steps_per_arm"] != len(plan) or summary["student_tokens_per_arm"] != sum(tokens):
        raise ValueError("summary budget mismatch")
    for arm in ARMS[1:]:
        required = {f"{arm}/training.jsonl", f"{arm}/final_adapter.pt", f"{arm}/final_optimizer.pt", "initial_adapter.pt"}
        if not required.issubset(manifest) or any((root / p).stat().st_size == 0 for p in required):
            raise ValueError("checkpoint or optimizer evidence missing")
        logs = [json.loads(line) for line in (root / arm / "training.jsonl").read_text().splitlines()]
        if len(logs) != len(plan):
            raise ValueError("incomplete training steps")
        for step, (record, ids, count) in enumerate(zip(logs, plan, tokens)):
            if record["step"] != step or record["indices"] != ids or record["student_tokens"] != count:
                raise ValueError("unequal training exposure")
            if not all(np.isfinite(record[k]) for k in ("loss", "grad_norm", "elapsed_seconds")):
                raise ValueError("nonfinite training record")
    for arm in ARMS[2:]:
        if f"{arm}_teacher.pt" not in manifest or summary["teacher_tokens"][arm] != sum(map(len, tokenized["teacher"][arm])):
            raise ValueError("teacher accounting mismatch")
    return {"steps_per_arm": len(plan), "student_tokens_per_arm": sum(tokens), "schedule_reproduced": True,
            "checkpoints_and_optimizer_present": True, "neural_or_optimizer_replay": False}


def paired_difference(left, right):
    keys = sorted(left)
    if set(keys) != set(right) or not keys:
        raise ValueError("paired IDs mismatch or empty")
    delta = np.array([int(left[k]) - int(right[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(9041815)
    draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    return {"n": len(keys), "gain_pp": 100 * float(delta.mean()),
            "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
            "descriptive_paired_bootstrap_95_pp": [float(x) for x in 100 * np.quantile(draws, [.025, .975])]}


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    required = {"train.json", "dev.json", "eval.json", "freeze.json", "status.json", "tokenized.json", "runner_source.py", "lora_source.py"}
    if not required.issubset(manifest):
        raise ValueError("analysis inputs missing from manifest")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("manifest path escape")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"hash mismatch {name}")
    freeze = json.loads((root / "freeze.json").read_text())
    for split, digest in freeze["input_sha256"].items():
        if hashlib.sha256((root / f"{split}.json").read_bytes()).hexdigest() != digest:
            raise ValueError("input freeze mismatch")
    for name, field in (("runner_source.py", "source_sha256"), ("lora_source.py", "lora_source_sha256")):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != freeze[field]:
            raise ValueError("source freeze mismatch")
    status = json.loads((root / "status.json").read_text())["status"]
    tokenized = json.loads((root / "tokenized.json").read_text())
    if status == "INVALID_MODEL_LOADING":
        if "loading_info.json" not in manifest:
            raise ValueError("loading failure evidence missing")
        loading = json.loads((root / "loading_info.json").read_text())
        if not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("loading failure not supported")
        return {"status": status, "verified_files": len(manifest), "scientific_decision": None, "interpretation": "Infrastructure failure, not a rejected paper hypothesis."}
    if not {"qualification.json", "baseline_dev.jsonl"}.issubset(manifest):
        raise ValueError("qualification evidence missing")
    devdata = json.loads((root / "dev.json").read_text())
    devrows = [json.loads(line) for line in (root / "baseline_dev.jsonl").read_text().splitlines()]
    validate_records(devdata, devrows, tokenized["student"]["dev"])
    qualification = qualify(devrows)
    stored = json.loads((root / "qualification.json").read_text())
    if qualification["qualified"] != stored["qualified"] or qualification["controls"] != stored["controls"] or abs(qualification["mean_choice_mass"] - stored["mean_choice_mass"]) > 1e-8:
        raise ValueError("qualification mismatch")
    if status == "UNQUALIFIED_ASSAY":
        if qualification["qualified"] or any(p.startswith(tuple(f"{arm}/" for arm in ARMS[1:])) for p in manifest) or "baseline_eval.jsonl" in manifest:
            raise ValueError("invalid early-stop evidence")
        return {"status": status, "qualification": qualification, "verified_files": len(manifest), "scientific_decision": None, "interpretation": "Development controls failed; no trained-method or paper kill conclusion."}
    if status != "COMPLETE" or not qualification["qualified"]:
        raise ValueError("unsupported terminal state")
    required_complete = {"summary.json", "baseline_eval.jsonl"} | {f"{arm}/{split}.jsonl" for arm in ARMS[1:] for split in ("dev", "eval")}
    if not required_complete.issubset(manifest):
        raise ValueError("complete outputs missing")
    run_summary = json.loads((root / "summary.json").read_text())
    budget = verify_budget(root, manifest, freeze, tokenized, run_summary)
    data = json.loads((root / "eval.json").read_text())
    expected = {r["id"]: r for r in data}
    if len(expected) != len(data):
        raise ValueError("duplicate dataset ID")
    all_rows, summary = {}, {}
    for arm in ARMS:
        path = root / "baseline_eval.jsonl" if arm == "baseline" else root / arm / "eval.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        records = {r["id"]: r for r in rows}
        validate_records(data, rows, tokenized["student"]["eval"])
        if len(rows) != len(records) or set(records) != set(expected):
            raise ValueError("missing or duplicate outputs")
        for rid, row in records.items():
            for field in ("target", "condition", "depth", "pair_id", "prompt"):
                if row[field] != expected[rid][field]:
                    raise ValueError(f"changed input {rid}/{field}")
            p = np.asarray(row["choice_probabilities"], dtype=float)
            if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() > 1.001:
                raise ValueError("bad full-vocabulary choice probabilities")
            if abs(p.sum() - row["choice_mass"]) > .002:
                raise ValueError("choice mass mismatch")
            if row["prediction"] != "ABCD"[int(p.argmax())]:
                raise ValueError("prediction mismatch")
        all_rows[arm] = records
        for split, splitdata in (("dev", devdata), ("eval", data)):
            output = root / f"baseline_{split}.jsonl" if arm == "baseline" else root / arm / f"{split}.jsonl"
            outrows = [json.loads(line) for line in output.read_text().splitlines()]
            validate_records(splitdata, outrows, tokenized["student"][split])
            correct = sum(r["prediction"] == r["target"] for r in outrows)
            reported = run_summary["results"][arm][split]
            if reported["correct"] != correct or reported["total"] != len(outrows) or abs(reported["accuracy"] - correct / len(outrows)) > 1e-10:
                raise ValueError("summary results mismatch")
        cells = []
        for depth in sorted({r["depth"] for r in rows}):
            for condition in sorted({r["condition"] for r in rows}):
                group = [r for r in rows if r["depth"] == depth and r["condition"] == condition]
                cells.append({"depth": depth, "condition": condition, "n": len(group),
                    "correct": sum(r["prediction"] == r["target"] for r in group),
                    "mean_choice_mass": float(np.mean([r["choice_mass"] for r in group])),
                    "stale_choices": sum(r["prediction"] == r["stale_target"] for r in group)})
        summary[arm] = cells
    long_scores = {arm: {rid: r["prediction"] == r["target"] for rid, r in rows.items()
                        if r["depth"] >= 60 and r["condition"] == "history"} for arm, rows in all_rows.items()}
    return {"manifest_sha256": hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest(),
            "status": status, "qualification": qualification, "training_accounting": budget, "verified_files": len(manifest), "cells": summary,
            "local_long_history_vs": {arm: paired_difference(long_scores["local_relation"], long_scores[arm]) for arm in ARMS[:-1]},
            "scope": "Single-seed synthetic development; hashes and output arithmetic verified, no neural replay. No paper go decision. Inspect capability controls and baseline strength before interpretation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("report")
    args = parser.parse_args()
    report = analyze(args.root)
    with open(args.report, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
