"""Verify a prospective T2T run and compare preserved arms with one scorer."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

from latent_contract.answer_scoring import option_label
from scripts.run_c2c_text_baseline import SPEC, messages
from scripts.verify_c2c_baseline_dev import check_manifest, verify as verify_baseline

RUNNER_SHA = "17cd70b2ff39b8b3f15f0e220d0e23a5fc6988d03c313060a342dcef14644cdd"
SCORER_SHA = "f9bb7fb3a112004cf52c4d086ac74c42d76e5006511f9c8dc2cb74dfe4712c63"


def verify(root, base_root, prepared, upstream):
    verify_baseline(base_root, prepared)
    check_manifest(root)
    scorer = Path(__file__).resolve().parents[1] / "src/latent_contract/answer_scoring.py"
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != SCORER_SHA:
        raise ValueError("scoring implementation drift")
    if hashlib.sha256((root / "runner.py").read_bytes()).hexdigest() != RUNNER_SHA:
        raise ValueError("runner digest mismatch")
    if (root / "FAILED.json").exists() or json.loads((root / "COMPLETE.json").read_text()) != {"calls": 256, "updates": 0}:
        raise ValueError("incomplete text comparator")
    if json.loads((root / "spec.json").read_text()) != SPEC:
        raise ValueError("spec mismatch")
    if (root / "cases.json").read_bytes() != (prepared / "cases.json").read_bytes():
        raise ValueError("wrong cases")
    runtime = json.loads((root / "runtime.json").read_text())
    if any(runtime[k] != SPEC[k] for k in ("torch", "transformers")) or any(
            runtime["models"][role] != SPEC[role + "_revision"] for role in ("sender", "receiver")):
        raise ValueError("runtime mismatch")
    source = subprocess.check_output(["git", "-C", str(upstream), "show", SPEC["upstream_commit"] + ":rosetta/utils/evaluate.py"])
    if hashlib.sha256(source).hexdigest() != "b341338aecf1cf07a5cb61664b8dac5f2ff96b32622e54eb37b32604deeddf16":
        raise ValueError("upstream prompt source mismatch")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_prompt", "exec"), namespace)
    cases = json.loads((prepared / "cases.json").read_text())
    key = json.loads((prepared / "private_answer_key.json").read_text())
    rows = [json.loads(s) for s in (root / "raw.jsonl").read_text().splitlines()]
    by_id = {(r["case_id"], r["arm"]): r for r in rows}
    expected = {(c["case_id"], arm) for c in cases for arm in SPEC["arms"]}
    datasets = {c["case_id"]: c["dataset"] for c in cases}
    if len(rows) != 256 or set(by_id) != expected:
        raise ValueError("missing/duplicate/unexpected output")
    for case in cases:
        choices = "".join(f"{chr(65+i)}. {text}\n" for i, text in enumerate(case["choices"]))
        prompt = namespace["build_prompt"]("mmlu-redux", "", case["question"], choices, False, True)
        background = by_id[case["case_id"], "background"]
        response = by_id[case["case_id"], "answer_with_background"]
        if background["messages"] != messages(case["question"], prompt):
            raise ValueError("background prompt mismatch")
        if response["messages"] != messages(case["question"], prompt, background["completion"]):
            raise ValueError("message transfer mismatch")
    for row in rows:
        if row["dataset"] != datasets[row["case_id"]] or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError("invalid dataset or generation timing")
        if not 0 < len(row["input_ids"]) <= SPEC["max_input_tokens"] or len(row["generated_ids"]) > 64:
            raise ValueError("token budget mismatch")
        if row["hit_token_limit"] != (len(row["generated_ids"]) == 64):
            raise ValueError("cap flag mismatch")
    original = [json.loads(s) for s in (base_root / "raw.jsonl").read_text().splitlines()]
    groups = {arm: [r for r in original if r["arm"] == arm] for arm in ("receiver", "sender", "c2c", "disabled_fuser")}
    groups["text_transfer"] = [r for r in rows if r["arm"] == "answer_with_background"]
    metrics = {}
    for arm, outputs in groups.items():
        latency = sum(r["seconds"] for r in (rows if arm == "text_transfer" else outputs))
        metrics[arm] = {"n": len(outputs), "correct": sum(option_label(r["completion"]) == key[r["case_id"]] for r in outputs),
                        "parsed": sum(option_label(r["completion"]) is not None for r in outputs),
                        "serial_generation_seconds": latency}
    return {"scope": SPEC["scope"], "metrics": metrics,
            "background_token_limits": sum(r["hit_token_limit"] for r in rows if r["arm"] == "background"),
            "answer_token_limits": sum(r["hit_token_limit"] for r in rows if r["arm"] == "answer_with_background"),
            "decision": "COMPARATOR_RECORDED_NO_AUTOMATIC_TRAINING",
            "interpretation": "Historical arms rescored under a prospectively fixed comparator rule; original strict reports unchanged. No natural-update effect or paper acceptance evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "base-root", "prepared", "upstream", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.base_root, args.prepared, args.upstream)):
        parser.error("output must be outside input roots")
    result = verify(args.root, args.base_root, args.prepared, args.upstream)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
