"""Read-only integrity and matched-interface analysis for the DEV diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path

from latent_contract.channel import MODEL, REVISION, validate
from latent_contract.verify import verify as verify_original
from scripts.diagnose_lc0_text_interface import ARMS, final_answer

RUNNER_SHA256 = "48824cbf34cf3f7c89b1fcf49ebe463df2c8b4c78c512a32a2f3351fc2275270"


def analyze(root, original):
    validate(root)
    verify_original(original)
    if (root / "FAILED.json").exists() or json.loads((root / "COMPLETE.json").read_text()) != {"records": 24}:
        raise ValueError("incomplete diagnostic")
    if hashlib.sha256((root / "runner.py").read_bytes()).hexdigest() != RUNNER_SHA256:
        raise ValueError("runner changed")
    plan = json.loads((root / "plan.json").read_text())
    baseline_plan = json.loads((original / "plan.json").read_text())
    assert plan["cases"] == baseline_plan["case_ids"]
    assert (plan["model"], plan["revision"]) == (MODEL, REVISION)
    assert plan["arms"] == list(ARMS) and plan["forwards"] == 24
    assert plan["max_new_tokens"] == {"ids_no_thinking": 8, "embeds_no_thinking": 8, "ids_thinking": 256}
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime["resolved_commit"] == REVISION
    assert plan["prepared_manifest_sha256"] == hashlib.sha256((original / "inputs/MANIFEST.json").read_bytes()).hexdigest()
    cases = json.loads((root / "cases.json").read_text())
    expected_cases = {c["case_id"]: c for c in json.loads((original / "inputs/cases.json").read_text())}
    assert cases == [expected_cases[cid] for cid in plan["cases"]]
    key = json.loads((original / "inputs/private_answer_key.json").read_text())
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    grid = {(r["case_id"], r["arm"]): r for r in rows}
    assert len(rows) == len(grid) == 24
    assert set(grid) == {(cid, arm) for cid in plan["cases"] for arm in ARMS}
    old = {r["case_id"]: r for r in map(json.loads, (original / "raw.jsonl").read_text().splitlines()) if r["arm"] == "text"}
    counts = {arm: {"n": 8, "correct": 0, "format_valid": 0, "token_limit": 0,
                    "generated_tokens": 0, "seconds": 0.0} for arm in ARMS}
    equivalence = {"identical_inputs": 0, "identical_generations": 0, "original_text_reproduced": 0}
    for case in cases:
        cid = case["case_id"]
        expected = chr(65 + case["choices"].index(key[cid]))
        a, b = grid[cid, "ids_no_thinking"], grid[cid, "embeds_no_thinking"]
        equivalence["identical_inputs"] += a["input_ids"] == b["input_ids"]
        equivalence["identical_generations"] += a["generated_ids"] == b["generated_ids"]
        equivalence["original_text_reproduced"] += b["generated_ids"] == old[cid]["generated_ids"]
        for arm in ARMS:
            row = grid[cid, arm]
            parsed = final_answer(row["completion"], arm == "ids_thinking")
            assert parsed == row["final_answer"]
            assert row["hit_token_limit"] == (len(row["generated_ids"]) == plan["max_new_tokens"][arm])
            assert len(row["generated_ids"]) <= plan["max_new_tokens"][arm]
            count = counts[arm]
            count["correct"] += parsed == expected
            count["format_valid"] += parsed is not None
            count["token_limit"] += row["hit_token_limit"]
            count["generated_tokens"] += len(row["generated_ids"])
            count["seconds"] += row["seconds"]
    return {"scope": "post_hoc_interface_diagnosis_no_latent_or_paper_decision",
            "counts": counts, "equivalence": equivalence,
            "note": "Reasoning mode and token budget change jointly; original gate scores are unchanged."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.original)):
        parser.error("report must be outside evidence")
    result = analyze(args.root, args.original)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
