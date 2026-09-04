"""Post-hoc paired utility audit; no inference, selection policy, or new gate.

Reverify immutable baseline evidence before comparing C2C with sender alone.
The answer-key oracle is an accuracy upper bound, not a deployable router.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Optional

from scripts.audit_c2c_answer_format import explicit_answer
from scripts.verify_c2c_baseline_dev import verify

UPSTREAM_COMMIT = "113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a"
EVALUATE_SHA = "b341338aecf1cf07a5cb61664b8dac5f2ff96b32622e54eb37b32604deeddf16"


def compare(rows, key, parser):
    selected = [r for r in rows if r["arm"] in ("sender", "c2c")]
    pairs = {(r["case_id"], r["arm"]): r for r in selected}
    ids = sorted({r["case_id"] for r in selected})
    expected = {(cid, arm) for cid in ids for arm in ("sender", "c2c")}
    if not ids or len(selected) != len(expected) or set(pairs) != expected:
        raise ValueError("incomplete or duplicate pairs")
    counts = dict(both_correct=0, sender_only_correct=0, c2c_only_correct=0, neither_correct=0)
    times = {arm: 0.0 for arm in ("sender", "c2c")}
    for cid in ids:
        correct = []
        for arm in ("sender", "c2c"):
            row = pairs[cid, arm]
            seconds = row["seconds"]
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid recorded generation time")
            times[arm] += seconds
            correct.append(parser(row["completion"]) == key[cid])
        category = {(True, True): "both_correct", (True, False): "sender_only_correct",
                    (False, True): "c2c_only_correct", (False, False): "neither_correct"}[tuple(correct)]
        counts[category] += 1
    n = len(ids)
    sender = counts["both_correct"] + counts["sender_only_correct"]
    c2c = counts["both_correct"] + counts["c2c_only_correct"]
    return {"n": n, **counts, "sender_correct": sender, "c2c_correct": c2c,
            "c2c_minus_sender_accuracy": (c2c - sender) / n,
            "answer_key_oracle_correct": n - counts["neither_correct"],
            "oracle_gain_over_sender": counts["c2c_only_correct"] / n,
            "recorded_generation_seconds": times,
            "sender_dominates_aggregate_accuracy_and_recorded_time": sender >= c2c and times["sender"] <= times["c2c"]}


def audit(root, prepared, upstream):
    original = verify(root, prepared)
    source = subprocess.check_output(["git", "-C", str(upstream), "show",
                                      UPSTREAM_COMMIT + ":rosetta/utils/evaluate.py"])
    if hashlib.sha256(source).hexdigest() != EVALUATE_SHA:
        raise ValueError("upstream parser source mismatch")
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "extract_answer_from_content")
    namespace = {"re": re, "Optional": Optional}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned_upstream_parser", "exec"), namespace)
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    key = json.loads((prepared / "private_answer_key.json").read_text())
    parsers = {"explicit_prefix": explicit_answer, "published_parser": namespace["extract_answer_from_content"]}
    groups = {"all": rows, **{d: [r for r in rows if r["dataset"] == d]
                             for d in sorted({r["dataset"] for r in rows})}}
    return {"scope": "POSTHOC_SAME_INPUT_PAIRED_UTILITY_NOT_A_NEW_GATE",
            "frozen_decision_preserved": original["decision"], "paper_green_light": False,
            "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in ("MANIFEST.json", "raw.jsonl")},
            "upstream_parser_sha256": EVALUATE_SHA,
            "groups": {g: {p: compare(rs, key, parser) for p, parser in parsers.items()}
                       for g, rs in groups.items()},
            "limitations": ["Same input and short-answer public DEV; no private-context collaboration.",
                            "Oracle uses answer labels and is not a trained or deployable selector.",
                            "Timing is recorded generation only, not model loading, memory, serving throughput or training.",
                            "Sequential single run; latency order/warmup effects not randomized away.",
                            "Published-parser fallback differs from conservative explicit-prefix scoring.",
                            "No update experiment and no independent confirmatory evaluation."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prepared", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.prepared, args.upstream)):
        parser.error("output must be outside immutable inputs")
    result = audit(args.root, args.prepared, args.upstream)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
