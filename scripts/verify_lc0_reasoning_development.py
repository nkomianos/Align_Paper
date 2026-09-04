"""Read-only verifier for the separately frozen reasoning-enabled DEV study."""
import argparse
import hashlib
import json
from pathlib import Path

from latent_contract.channel import analyze, validate, validate_inputs
from scripts.run_lc0_reasoning_development import ARMS, SPEC, final_answer

RUNNER_SHA256 = "956b1165b5ebdedfa6dd228f49eb4e39e0e315d840b4dd4b3add72d94f5f7a64"


def verify(root, prepared):
    validate(root)
    validate_inputs(prepared)
    if (root / "FAILED.json").exists() or json.loads((root / "COMPLETE.json").read_text()) != {"records": 96, "updates_trained": 0}:
        raise ValueError("incomplete development study")
    if hashlib.sha256((root / "runner.py").read_bytes()).hexdigest() != RUNNER_SHA256:
        raise ValueError("runner digest mismatch")
    if json.loads((root / "spec.json").read_text()) != SPEC:
        raise ValueError("spec mismatch")
    if hashlib.sha256((prepared / "MANIFEST.json").read_bytes()).hexdigest() != SPEC["prepared_manifest_sha256"]:
        raise ValueError("input root mismatch")
    cases = [c for c in json.loads((prepared / "cases.json").read_text()) if c["pair_id"] in SPEC["pair_ids"]]
    if len(cases) != 16 or json.loads((root / "cases.json").read_text()) != cases:
        raise ValueError("wrong DEV slice")
    if json.loads((root / "runtime.json").read_text())["resolved_commit"] != SPEC["revision"]:
        raise ValueError("model mismatch")
    key = json.loads((prepared / "private_answer_key.json").read_text())
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    expected = {(c["case_id"], arm) for c in cases for arm in ARMS}
    if len(rows) != 96 or {(r["case_id"], r["arm"]) for r in rows} != expected:
        raise ValueError("wrong output grid")
    budget_rows = json.loads((root / "BUDGET_AUDIT.json").read_text())
    budget = {r["case_id"]: r for r in budget_rows}
    if len(budget_rows) != 16 or set(budget) != {c["case_id"] for c in cases}:
        raise ValueError("budget grid mismatch")
    compact, limits = [], {arm: 0 for arm in ARMS}
    for row in rows:
        final = final_answer(row["completion"])
        if row["final_answer"] != final:
            raise ValueError("stored parse differs from frozen parser")
        cap = len(row["generated_ids"]) == SPEC["max_new_tokens"]
        if cap != row["hit_token_limit"] or len(row["generated_ids"]) > SPEC["max_new_tokens"]:
            raise ValueError("token-limit record mismatch")
        name = "without_message" if row["arm"] == "no_message" else "with_message"
        if row["input_tokens"] != budget[row["case_id"]][name] or row["input_tokens"] > 2048:
            raise ValueError("token budget mismatch")
        limits[row["arm"]] += cap
        compact.append({"case_id": row["case_id"], "arm": row["arm"], "completion": final or ""})
    criteria = analyze(cases, key, compact, "full")
    criteria["scope"] = SPEC["scope"]
    return {"scope": SPEC["scope"], "engineering_criteria": criteria,
            "token_limit_counts": limits, "calls": 96, "independent_pairs": 8,
            "total_generation_seconds": sum(r["seconds"] for r in rows),
            "interpretation": "DEV apparatus study, not a replacement frozen gate or update experiment"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.prepared)):
        parser.error("report must be outside evidence and prepared inputs")
    result = verify(args.root, args.prepared)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
