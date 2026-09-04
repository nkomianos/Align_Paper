"""Read-only qualification, not a paper gate and not a bridge-effect test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from latent_contract.answer_scoring import option_label
from scripts.run_c2c_sender_update import SPEC, SOURCE_FILES, sha256


def summarize(rows, key):
    if len({r["case_id"] for r in rows}) != len(rows) or not rows:
        raise ValueError("duplicate/empty rows")
    losses = []
    for row in rows:
        scores = row["choice_sequence_logps"]
        if set(scores) != set("ABCD") or not all(math.isfinite(x) for x in scores.values()):
            raise ValueError("invalid option likelihoods")
        maximum = max(scores.values())
        denominator = maximum + math.log(sum(math.exp(x-maximum) for x in scores.values()))
        losses.append(denominator - scores[key[row["case_id"]]])
    return {"n": len(rows), "accuracy": sum(option_label(r["completion"]) == key[r["case_id"]] for r in rows)/len(rows),
            "parse_rate": sum(option_label(r["completion"]) is not None for r in rows)/len(rows),
            "choice_ce": sum(losses)/len(losses), "token_limits": sum(r["hit_token_limit"] for r in rows)}


def qualify(base, updated):
    retained = updated["accuracy"] >= base["accuracy"] - SPEC["qualification_accuracy_retention_pp"]/100
    useful = base["choice_ce"] > 0 and updated["choice_ce"] <= base["choice_ce"] * (1-SPEC["qualification_choice_ce_relative_reduction"])
    valid = min(base["parse_rate"], updated["parse_rate"]) >= SPEC["qualification_parse_rate"]
    return {"retained_accuracy": retained, "choice_ce_improved": useful, "format_valid": valid,
            "decision": "QUALIFIED_FOR_PAIRED_INTERFACE_MEASUREMENT" if retained and useful and valid
                        else "UPDATE_NOT_QUALIFIED_NO_INTERFACE_CONCLUSION"}


def verify(root, key_path, expected_commit):
    files = json.loads((root / "MANIFEST.json").read_text())["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
    if set(files) != actual:
        raise ValueError("manifest coverage mismatch")
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError("checksum/path mismatch")
    if (root / "FAILED.json").exists():
        raise ValueError("failed runner")
    recorded_spec = json.loads((root / "spec.json").read_text())
    seed = recorded_spec.pop("seed")
    if recorded_spec != SPEC or seed not in SPEC["seeds"]:
        raise ValueError("spec drift")
    complete = json.loads((root / "COMPLETE.json").read_text())
    if complete != {"seed": seed, "steps": 128, "qualification_cases": 128, "bridge_calls": 0, "final_eval_accessed": False}:
        raise ValueError("completion mismatch")
    own_repo = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "source.json").read_text())
    if sources["commit"] != expected_commit or set(sources["files"]) != set(SOURCE_FILES):
        raise ValueError("source revision mismatch")
    for relative in SOURCE_FILES:
        expected = subprocess.check_output(["git", "-C", str(own_repo), "show", expected_commit + ":" + relative])
        if (root / "source" / relative).read_bytes() != expected or hashlib.sha256(expected).hexdigest() != sources["files"][relative]:
            raise ValueError("source bytes mismatch")
    runtime = json.loads((root / "runtime.json").read_text())
    if any(runtime[k] != SPEC[k] for k in ("torch", "transformers", "revision")):
        raise ValueError("runtime drift")
    if sha256(root / "update_train.json") != SPEC["train_sha256"] or sha256(root / "qualification.json") != SPEC["qualification_sha256"]:
        raise ValueError("input drift")
    if sha256(key_path) != "1aec66fe9cdaff557a6d834be98ad3fadde195ba10248a8ce3637b62b55c7c41":
        raise ValueError("private key mismatch")
    key = json.loads(key_path.read_text())["qualification"]
    cases = json.loads((root / "qualification.json").read_text())
    prompts = json.loads((root / "qualification_prompt_ids.json").read_text())
    stages = {}
    for stage in ("base", "noop", "wrapped_final", "updated"):
        rows = [json.loads(line) for line in (root / (stage + ".jsonl")).read_text().splitlines()]
        subset = cases[:8] if stage in {"noop", "wrapped_final"} else cases
        if [r["case_id"] for r in rows] != [c["case_id"] for c in subset]:
            raise ValueError("qualification grid mismatch")
        for row in rows:
            if row["stage"] != stage or row["input_ids"] != prompts[row["case_id"]]:
                raise ValueError("unmatched qualification prompt")
            if not 0 < len(row["input_ids"]) <= SPEC["max_input_tokens"] or len(row["generated_ids"]) > SPEC["max_new_tokens"]:
                raise ValueError("token limit mismatch")
            if row["hit_token_limit"] != (len(row["generated_ids"]) == SPEC["max_new_tokens"]) or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
                raise ValueError("invalid timing/truncation")
        stages[stage] = rows
    for a, b in zip(stages["base"], stages["noop"]):
        if a["generated_ids"] != b["generated_ids"] or a["choice_sequence_logps"] != b["choice_sequence_logps"]:
            raise ValueError("no-op mismatch")
    history = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
    if [r["step"] for r in history] != list(range(1, 129)):
        raise ValueError("step mismatch")
    if sorted(i for r in history for i in r["example_indices"]) != list(range(1024)):
        raise ValueError("training example coverage mismatch")
    if any(not math.isfinite(r[k]) for r in history for k in ("mean_completion_token_nll", "gradient_norm_before_clip", "elapsed_seconds")):
        raise ValueError("nonfinite training telemetry")
    for name in ("adapter_initial.pt", "adapter_step_0064.pt", "adapter_final.pt", "optimizer_final.pt", "merged_sender/config.json"):
        if name not in files:
            raise ValueError("missing checkpoint")
    base, updated = [summarize(stages[s], key) for s in ("base", "updated")]
    return {"scope": SPEC["scope"], "seed": seed, "base": base, "updated": updated,
            **qualify(base, updated), "merge_audit": json.loads((root / "merge_audit.json").read_text()),
            "interpretation": "Sender-only screening on public DEV; no bridge effect, repair success or paper viability established. Both prespecified seeds must qualify before the paired study; do not select one favorable seed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "key", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()) or args.output.resolve() == args.key.resolve():
        parser.error("report must be outside evidence")
    result = verify(args.root, args.key, args.expected_commit)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
