"""Post-hoc diagnostics only; never rescoring a frozen gate as a pass.

Reads original verified evidence and writes a fresh report outside both roots.
Leading-letter scores measure how much strict output formatting can explain.
They are not substituted for registered metrics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def leading_letter(text, alphabet):
    match = re.match(r"^([" + alphabet + r"])(?=$|[.\s:)])", text.strip())
    return match.group(1) if match else None


def lc0_audit(root):
    from latent_contract.verify import verify
    registered = verify(root)
    cases = {c["case_id"]: c for c in json.loads((root / "inputs/cases.json").read_text())}
    key = json.loads((root / "inputs/private_answer_key.json").read_text())
    records = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    counts = {}
    for row in records:
        cid, arm = row["case_id"], row["arm"]
        target = cid
        if arm.startswith("counterfactual_"):
            target = cid.rsplit("/", 1)[0] + ("/counterfactual" if cid.endswith("/original") else "/original")
        expected = chr(65 + cases[cid]["choices"].index(key[target]))
        letter = leading_letter(row["completion"], "A-D")
        item = counts.setdefault(arm, {"n": 0, "strict_target_correct": 0,
                                      "leading_letter_target_correct": 0, "format_rescued": 0})
        strict = row["completion"].strip() == expected
        relaxed = letter == expected
        item["n"] += 1
        item["strict_target_correct"] += strict
        item["leading_letter_target_correct"] += relaxed
        item["format_rescued"] += relaxed and not strict
    return {"registered": registered, "diagnostic_counts": counts}


def ep0_audit(root):
    from efference_pair.verify import verify
    registered = verify(root)
    cases = {c["case_id"]: c for c in map(json.loads, (root / "inputs/cases.jsonl").read_text().splitlines())}
    key = json.loads((root / "inputs/answer_key.json").read_text())
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    grouped = {}
    detail = []
    for row in rows:
        cid = row["case_id"]
        case = cases[cid]
        options = dict(re.findall(r"([A-E])\. (left|right|up|down|stationary)", case["prompt"]))
        prediction = options.get(row["completion"].strip())
        expected = options[key[cid]["answer"]]
        grouped.setdefault((case["scene"], case["stratum"]), {})[case["condition"]] = prediction
        detail.append({"case_id": cid, "predicted_direction": prediction, "expected_direction": expected})
    return {"registered": registered, "predicted_directions": dict(Counter(r["predicted_direction"] for r in detail)),
            "same_prediction_across_conditions": sum(len(set(v.values())) == 1 for v in grouped.values()),
            "matched_scene_questions": len(grouped), "details": detail}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", type=Path, required=True)
    parser.add_argument("--ep0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(root.resolve()) for root in (args.lc0, args.ep0)):
        parser.error("output must be outside evidence roots")
    report = {"scope": "post_hoc_interface_diagnosis_not_new_gate", "lc0": lc0_audit(args.lc0),
              "ep0": ep0_audit(args.ep0)}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
