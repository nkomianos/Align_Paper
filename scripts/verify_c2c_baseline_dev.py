"""Immutable-evidence verifier for the frozen released-C2C DEV baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.run_c2c_baseline_dev import SPEC

RUNNER_SHA = "55a5bc561eaf3ae37e16b9a70db6fd6e893258a1d0e67b09012e93314d3637ef"
PREPARED_MANIFEST_SHA = "7104a4165454138c6ca24099bcd6dc8f8f470b228b2b43590c03eefe1aa2be87"


def answer(text):
    match = re.fullmatch(r'(?:The correct answer is\s*)?([ABCD])\.?', text.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def check_manifest(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())["files"]
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
    if actual != set(manifest):
        raise ValueError("manifest file coverage mismatch")
    for name, digest in manifest.items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("artifact checksum mismatch")


def analyze(cases, key, rows):
    ids = {c["case_id"] for c in cases}
    grid = {(cid, arm) for cid in ids for arm in SPEC["arms"]}
    if len(rows) != len(grid) or {(r["case_id"], r["arm"]) for r in rows} != grid:
        raise ValueError("missing, duplicate or unexpected outputs")
    metrics = {}
    for arm in SPEC["arms"]:
        selected = [r for r in rows if r["arm"] == arm]
        metrics[arm] = {"n": len(selected), "accuracy": sum(answer(r["completion"]) == key[r["case_id"]] for r in selected)/len(selected),
                        "parse_rate": sum(answer(r["completion"]) is not None for r in selected)/len(selected),
                        "hit_token_limit": sum(r["hit_token_limit"] for r in selected),
                        "generation_seconds": sum(r["seconds"] for r in selected)}
    pairs = {(r["case_id"], r["arm"]): r for r in rows}
    agreement = sum(pairs[cid, "receiver"]["completion"].strip() == pairs[cid, "disabled_fuser"]["completion"].strip() for cid in ids)/len(ids)
    for cid in ids:
        inputs = [pairs[cid, arm]["input_ids"] for arm in SPEC["arms"]]
        chats = [pairs[cid, arm]["chat"] for arm in SPEC["arms"]]
        if any(x != inputs[0] for x in inputs) or any(x != chats[0] for x in chats):
            raise ValueError("unmatched paired inputs")
    for row in rows:
        if not 0 < len(row["input_ids"]) <= SPEC["max_input_tokens"]:
            raise ValueError("input limit mismatch")
        if len(row["generated_ids"]) > SPEC["max_new_tokens"] or row["hit_token_limit"] != (len(row["generated_ids"]) == SPEC["max_new_tokens"]):
            raise ValueError("output limit mismatch")
    valid = agreement >= 0.98 and all(m["parse_rate"] >= 0.95 for m in metrics.values())
    gain = metrics["c2c"]["accuracy"] - metrics["receiver"]["accuracy"]
    return {"scope": SPEC["scope"], "metrics": metrics, "disabled_fuser_text_agreement": agreement,
            "functional_controls_pass": valid, "c2c_minus_receiver_accuracy": gain,
            "decision": "BASELINE_SIGNAL_CONTINUE_PROTOCOL_DESIGN" if valid and gain >= .05 else "BASELINE_INCONCLUSIVE_NO_UPDATE_TRAINING",
            "interpretation": "Public validation DEV; no sender update, text-transfer comparison or original-paper numerical reproduction. Not a paper go/no-go."}


def verify(root, prepared):
    check_manifest(root)
    check_manifest(prepared)
    if hashlib.sha256((root / "runner.py").read_bytes()).hexdigest() != RUNNER_SHA:
        raise ValueError("runner digest mismatch")
    if hashlib.sha256((prepared / "MANIFEST.json").read_bytes()).hexdigest() != PREPARED_MANIFEST_SHA:
        raise ValueError("input preparation mismatch")
    if (root / "FAILED.json").exists() or json.loads((root / "COMPLETE.json").read_text()) != {"calls": 512, "updates": 0}:
        raise ValueError("incomplete baseline")
    if json.loads((root / "spec.json").read_text()) != SPEC:
        raise ValueError("specification mismatch")
    if (root / "cases.json").read_bytes() != (prepared / "cases.json").read_bytes():
        raise ValueError("different input cases")
    runtime = json.loads((root / "runtime.json").read_text())
    if runtime["strict_projectors_loaded"] != 28 or any(runtime[k] != SPEC[k] for k in ("torch", "transformers")):
        raise ValueError("runtime mismatch")
    if any(runtime["models"][role] != SPEC[role + "_revision"] for role in ("sender", "receiver")):
        raise ValueError("model mismatch")
    cases = json.loads((prepared / "cases.json").read_text())
    key = json.loads((prepared / "private_answer_key.json").read_text())
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    result = analyze(cases, key, rows)
    result["by_dataset"] = {name: analyze([c for c in cases if c["dataset"] == name], key,
                             [r for r in rows if r["dataset"] == name]) for name in sorted({c["dataset"] for c in cases})}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.root, args.prepared)):
        parser.error("report must be outside immutable evidence")
    result = verify(args.root, args.prepared)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
