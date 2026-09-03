"""Read-only LC0 verifier with a fresh report outside the evidence root."""
import argparse
import json
from pathlib import Path

from .channel import ARMS, SETTINGS, analyze, plan, validate, validate_inputs, write


def verify(root):
    validate(root)
    validate_inputs(root / "inputs")
    if not (root / "COMPLETE.json").exists() or (root / "FAILED.json").exists():
        raise ValueError("incomplete/failed run: preserve but do not score")
    run_plan = json.loads((root / "plan.json").read_text())
    if run_plan["settings"] != SETTINGS or json.loads((root / "inputs/settings.json").read_text()) != SETTINGS:
        raise ValueError("frozen settings mismatch")
    runtime = json.loads((root / "runtime.json").read_text())
    if (runtime["model"], runtime["revision"]) != (SETTINGS["model"], SETTINGS["revision"]):
        raise ValueError("model mismatch")
    cases = plan(json.loads((root / "inputs/cases.json").read_text()), run_plan["mode"])
    if run_plan["case_ids"] != [c["case_id"] for c in cases] or run_plan["arms"] != list(ARMS):
        raise ValueError("run plan mismatch")
    records = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    if (run_plan["receiver_forwards"] != len(cases) * len(ARMS)
            or run_plan["sender_prefills"] != len(cases)
            or json.loads((root / "COMPLETE.json").read_text()) != {"records": len(records), "updates_trained": 0}):
        raise ValueError("declared workload/completion mismatch")
    audit = json.loads((root / "BUDGET_AUDIT.json").read_text())
    budget = {a["case_id"]: a for a in audit}
    if len(budget) != len(cases) or len(audit) != len(cases) or set(budget) != set(run_plan["case_ids"]):
        raise ValueError("budget audit incomplete")
    for row in records:
        name = "input_tokens_without_message" if row["arm"] == "no_message" else "input_tokens_with_message"
        if row["input_tokens"] != budget[row["case_id"]][name] or row["input_tokens"] > SETTINGS["max_context"]:
            raise ValueError("input budget mismatch")
    key = json.loads((root / "inputs/private_answer_key.json").read_text())
    return analyze(cases, key, records, run_plan["mode"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("report must be outside immutable evidence")
    result = verify(args.root)
    write(args.output, result)
    print(json.dumps(result, indent=2))
