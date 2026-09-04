"""Verify paired evidence against the locally issued release ticket."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from latent_contract.update_comparison import analyze
from scripts.run_c2c_paired_update import SPEC, SOURCE_FILES
from scripts.run_c2c_sender_update import SPEC as UPDATE_SPEC, sha256, pinned_prompt
from scripts.run_c2c_text_baseline import messages


def verify(root, key_path, ticket_path, expected_commit, upstream):
    files = json.loads((root / "MANIFEST.json").read_text())["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
    if set(files) != actual:
        raise ValueError("manifest coverage differs")
    for name, digest in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target) != digest:
            raise ValueError("manifest digest/path differs")
    if (root / "FAILED.json").exists():
        raise ValueError("failed paired run")
    spec = json.loads((root / "spec.json").read_text())
    seed = spec.pop("seed")
    if spec != SPEC or seed not in UPDATE_SPEC["seeds"]:
        raise ValueError("specification drift")
    if json.loads((root / "COMPLETE.json").read_text()) != {"calls": 2816, "seed": seed, "updates": 0, "repairs": 0}:
        raise ValueError("incomplete paired run")
    if (root / "release_ticket.json").read_bytes() != ticket_path.read_bytes():
        raise ValueError("not the locally issued release ticket")
    ticket = json.loads(ticket_path.read_text())
    if ticket["scope"] != "RELEASE_PAIRED_DEV_ONLY_NOT_PAPER_EXPANSION" or set(ticket["updates"]) != {str(s) for s in UPDATE_SPEC["seeds"]}:
        raise ValueError("invalid release scope")
    if any(x["qualification"]["decision"] != "QUALIFIED_FOR_PAIRED_INTERFACE_MEASUREMENT" for x in ticket["updates"].values()):
        raise ValueError("release contains an unqualified update")
    runtime = json.loads((root / "runtime.json").read_text())
    if any(runtime[k] != SPEC[k] for k in ("torch", "transformers")):
        raise ValueError("runtime differs")
    if runtime["models"]["old"] != SPEC["sender_revision"] or runtime["models"]["receiver"] != SPEC["receiver_revision"]:
        raise ValueError("base model differs")
    if runtime["ticket_sha256"] != sha256(ticket_path) or runtime["updated_model_files"] != ticket["updates"][str(seed)]["merged_files"]:
        raise ValueError("updated model attestation differs")
    expected_mapping = {"0": {"1": {str(i): [[i+8, i]] for i in range(28)}}}
    if runtime["strict_projectors_loaded"] != 28 or runtime["mapping"] != expected_mapping:
        raise ValueError("fuser mapping differs")
    sources = json.loads((root / "source.json").read_text())
    if sources["commit"] != expected_commit or set(sources["files"]) != set(SOURCE_FILES):
        raise ValueError("source commit differs")
    repo = Path(__file__).resolve().parents[1]
    for relative in SOURCE_FILES:
        source = subprocess.check_output(["git", "-C", str(repo), "show", expected_commit+":"+relative])
        if hashlib.sha256(source).hexdigest() != sources["files"][relative] or (root / "source" / relative).read_bytes() != source:
            raise ValueError("source bytes differ")
    if sha256(root / "cases.json") != SPEC["cases_sha256"] or sha256(key_path) != "1aec66fe9cdaff557a6d834be98ad3fadde195ba10248a8ce3637b62b55c7c41":
        raise ValueError("case/key digest differs")
    cases = json.loads((root / "cases.json").read_text())
    key = json.loads(key_path.read_text())["final_eval"]
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    build_prompt = pinned_prompt(upstream)
    by_id = {c["case_id"]: c for c in cases}
    for row in rows:
        case = by_id[row["case_id"]]
        choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(case["choices"]))
        prompt = build_prompt("mmlu-redux", "", case["question"], choices, False, True)
        if row["arm"].endswith("_background"):
            expected = messages(case["question"], prompt)
        elif row["arm"].endswith("_text"):
            continue  # paired analysis reconstructs the transferred background.
        else:
            expected = [{"role": "user", "content": prompt}]
        if row["messages"] != expected:
            raise ValueError("prompt differs from pinned builder")
    report = analyze(cases, key, rows)
    report.update({"scope": SPEC["scope"], "seed": seed,
                   "source_commit": expected_commit, "ticket_sha256": sha256(ticket_path)})
    report["by_dataset"] = {label: analyze([c for c in cases if c["dataset"] == label],
                                         {c["case_id"]: key[c["case_id"]] for c in cases if c["dataset"] == label},
                                         [r for r in rows if r["dataset"] == label])
                            for label in sorted({c["dataset"] for c in cases})}
    return report, rows


def combine(reports, records):
    if len(reports) != 2 or {r["seed"] for r in reports} != set(UPDATE_SPEC["seeds"]):
        raise ValueError("two independent fixed seeds required")
    if reports[0]["ticket_sha256"] != reports[1]["ticket_sha256"]:
        raise ValueError("different release tickets")
    old = [{(r["case_id"], r["arm"]): r["generated_ids"] for r in rows
            if r["arm"] == "receiver" or r["arm"].startswith("old_")} for rows in records]
    if not old[0] or old[0].keys() != old[1].keys():
        raise ValueError("unmatched reference outputs across seeds")
    agreement = sum(old[0][key] == old[1][key] for key in old[0])/len(old[0])
    repeated = all(r["decision"] == "EXTRA_LATENT_LOSS_SIGNAL_REPAIR_STUDY_NEEDED" for r in reports)
    decision = ("REFERENCE_REPRODUCIBILITY_FAILURE" if agreement < .98 else
                "REPEATED_EXTRA_LATENT_LOSS_REPAIR_STUDY_NEEDED" if repeated else
                "NO_REPEATED_QUALIFIED_SIGNAL_DO_NOT_EXPAND")
    return {"decision": decision, "old_reference_token_agreement": agreement, "per_seed": reports,
            "interpretation": "Two updates are not a population-level replication. Even a repeated effect is not a novel repair or a viable paper; no expansion is launched automatically."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, nargs=2, required=True)
    for name in ("key", "ticket", "upstream", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in [*args.root, args.upstream]) or args.output.resolve() in {args.key.resolve(), args.ticket.resolve()}:
        parser.error("output must be outside input evidence")
    values = [verify(root, args.key, args.ticket, args.expected_commit, args.upstream) for root in args.root]
    result = combine([v[0] for v in values], [v[1] for v in values])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result))
