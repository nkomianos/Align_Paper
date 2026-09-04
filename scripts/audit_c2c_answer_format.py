"""Post-hoc scoring diagnosis against the pre-existing upstream parser.

Preserves the frozen strict report. No generation, model changes or new gate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import random
import subprocess
from typing import Optional

from scripts.verify_c2c_baseline_dev import answer, verify


def explicit_answer(text):
    match = re.match(r'^The correct answer is\s+([ABCD])\b', text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else answer(text)


def audit(root, prepared, upstream):
    frozen = verify(root, prepared)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != "113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a":
        raise ValueError("unreviewed upstream revision")
    source = subprocess.check_output(["git", "-C", str(upstream), "show", commit + ":rosetta/utils/evaluate.py"])
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "extract_answer_from_content")
    # Only the inspected, pure parser is executed; avoid importing model code.
    namespace = {"re": re, "Optional": Optional}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_parser", "exec"), namespace)
    upstream_answer = namespace["extract_answer_from_content"]
    key = json.loads((prepared / "private_answer_key.json").read_text())
    rows = [json.loads(s) for s in (root / "raw.jsonl").read_text().splitlines()]
    result = {"scope": "POSTHOC_SCORING_DIAGNOSIS_NO_NEW_GATE", "frozen_decision": frozen["decision"],
              "upstream_commit": commit, "upstream_evaluate_sha256": hashlib.sha256(source).hexdigest(), "groups": {}}
    for dataset in ["all"] + sorted({r["dataset"] for r in rows}):
        result["groups"][dataset] = {}
        for arm in ("receiver", "sender", "c2c", "disabled_fuser"):
            selected = [r for r in rows if r["arm"] == arm and (dataset == "all" or r["dataset"] == dataset)]
            stats = {}
            for name, parser in (("strict", answer), ("explicit_prefix", explicit_answer), ("upstream", upstream_answer)):
                predictions = [parser(r["completion"]) for r in selected]
                stats[name] = {"n": len(selected), "correct": sum(p == key[r["case_id"]] for p, r in zip(predictions, selected)),
                               "parsed": sum(p is not None for p in predictions)}
            stats["explicit_upstream_disagreements"] = sum(explicit_answer(r["completion"]) != upstream_answer(r["completion"]) for r in selected)
            result["groups"][dataset][arm] = stats
    paired = {(r["case_id"], r["arm"]): r for r in rows}
    groups = {name: sorted({r["case_id"] for r in rows if r["dataset"] == name})
              for name in sorted({r["dataset"] for r in rows})}
    result["exploratory_paired_bootstrap"] = {}
    for name, parser in (("explicit_prefix", explicit_answer), ("upstream", upstream_answer)):
        differences = {cid: int(parser(paired[cid, "c2c"]["completion"]) == key[cid])
                           - int(parser(paired[cid, "receiver"]["completion"]) == key[cid])
                       for ids in groups.values() for cid in ids}
        rng = random.Random(20260904)
        samples = sorted(sum(differences[cid] for ids in groups.values() for cid in rng.choices(ids, k=len(ids))) / 128
                         for _ in range(5000))
        result["exploratory_paired_bootstrap"][name] = {
            "mean_difference": sum(differences.values()) / 128,
            "percentile_95_interval": [samples[125], samples[4874]],
            "resampling_unit": "question paired across arms, stratified by dataset",
            "scope": "exploratory after scoring audit; no seed or population-wide coverage claim"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "prepared", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.prepared, args.upstream)):
        parser.error("output must be outside immutable inputs")
    result = audit(args.root, args.prepared, args.upstream)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
